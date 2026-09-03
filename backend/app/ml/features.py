"""
Feature engineering for the Nigerian credit scoring model.
Combines Home Credit Default Risk features with synthetic Nigerian
alternative data (airtime, mobile money, bill payment patterns).
"""
import numpy as np
import pandas as pd
from typing import Optional

EMPLOYMENT_ENCODING = {
    "civil_servant": 0,
    "bank_staff": 1,
    "formal_private": 2,
    "market_trader": 3,
    "artisan": 4,
    "gig_worker": 5,
    "informal_trader": 6,
}

STATE_TO_ZONE = {
    "Lagos": "SW", "Ogun": "SW", "Oyo": "SW", "Osun": "SW", "Ondo": "SW", "Ekiti": "SW",
    "Abuja": "NC", "Niger": "NC", "Kogi": "NC", "Benue": "NC", "Plateau": "NC",
    "Kano": "NW", "Kaduna": "NW", "Katsina": "NW", "Sokoto": "NW", "Zamfara": "NW",
    "Rivers": "SS", "Delta": "SS", "Edo": "SS", "Bayelsa": "SS", "Cross River": "SS",
    "Anambra": "SE", "Enugu": "SE", "Imo": "SE", "Abia": "SE", "Ebonyi": "SE",
    "Borno": "NE", "Adamawa": "NE", "Taraba": "NE", "Yobe": "NE", "Gombe": "NE",
}

ZONE_ENCODING = {"SW": 0, "NC": 1, "NW": 2, "SS": 3, "SE": 4, "NE": 5}

FEATURE_NAMES = [
    # Financial ratios
    "loan_to_income_ratio",
    "monthly_repayment_burden",
    "post_loan_dti",
    "income_log",
    # Employment
    "employment_type_encoded",
    "geo_zone_encoded",
    "age",
    # Alternative data (Nigerian-specific)
    "airtime_topup_frequency",
    "mobile_money_velocity",
    "bill_payment_regularity",
    "balance_stability_score",
    "alt_data_composite",
    # Interaction features
    "income_x_alt_data",
    "gig_worker_flag",
    "informal_sector_flag",
    "northern_zone_flag",
]

FEATURE_LABELS = {
    "loan_to_income_ratio": "Loan-to-Income Ratio",
    "monthly_repayment_burden": "Monthly Repayment Burden",
    "post_loan_dti": "Post-Loan DTI",
    "income_log": "Log Monthly Income",
    "employment_type_encoded": "Employment Type",
    "geo_zone_encoded": "Geographic Zone",
    "age": "Applicant Age",
    "airtime_topup_frequency": "Airtime Top-up Frequency",
    "mobile_money_velocity": "Mobile Money Velocity",
    "bill_payment_regularity": "Bill Payment Regularity",
    "balance_stability_score": "Balance Stability Score",
    "alt_data_composite": "Alternative Data Composite",
    "income_x_alt_data": "Income × Alt-Data Interaction",
    "gig_worker_flag": "Gig Worker Indicator",
    "informal_sector_flag": "Informal Sector Indicator",
    "northern_zone_flag": "Northern Zone Indicator",
}

def engineer_features(row: dict) -> np.ndarray:
    """Transform a single applicant dict into the feature vector for inference."""
    monthly_income = float(row["monthly_income"])
    loan_amount = float(row["loan_amount"])
    tenor = int(row["loan_tenor_months"])
    dti = float(row["existing_dti"])
    
    # Clip extreme synthetic alternative data values for inference
    airtime = min(float(row["airtime_topup_frequency"]), 20.0)
    mm_velocity = min(float(row["mobile_money_velocity"]), 15.0)
    bill_reg = float(row["bill_payment_regularity"])
    bal_stability = float(row["balance_stability_score"])
    
    monthly_repayment = loan_amount / max(tenor, 1)
    lti = loan_amount / max(monthly_income, 1)
    repayment_burden = monthly_repayment / max(monthly_income, 1)
    post_loan_dti = dti + repayment_burden
    income_log = np.log1p(monthly_income)
    emp_enc = EMPLOYMENT_ENCODING.get(row["employment_type"], 6)
    zone = STATE_TO_ZONE.get(row["state"], "NC")
    zone_enc = ZONE_ENCODING.get(zone, 1)
    age = int(row.get("age", 35))
    
    alt_composite = (
        (airtime / 30) * 0.25
        + (mm_velocity / 20) * 0.30
        + (bill_reg / 100) * 0.30
        + (bal_stability / 100) * 0.15
    )
    income_x_alt = income_log * alt_composite
    
    gig_flag = 1.0 if row["employment_type"] == "gig_worker" else 0.0
    informal_flag = 1.0 if row["employment_type"] in ("informal_trader", "market_trader", "artisan") else 0.0
    north_flag = 1.0 if zone in ("NW", "NE") else 0.0
    
    features = np.array([
        lti,
        repayment_burden,
        post_loan_dti,
        income_log,
        emp_enc,
        zone_enc,
        age,
        airtime,
        mm_velocity,
        bill_reg,
        bal_stability,
        alt_composite,
        income_x_alt,
        gig_flag,
        informal_flag,
        north_flag,
    ], dtype=np.float32)
    
    return features

def engineer_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised feature engineering for a whole DataFrame (training)."""
    out = pd.DataFrame()
    
    monthly_repayment = df["loan_amount"] / df["loan_tenor_months"].clip(lower=1)
    out["loan_to_income_ratio"] = df["loan_amount"] / df["monthly_income"].clip(lower=1)
    out["monthly_repayment_burden"] = monthly_repayment / df["monthly_income"].clip(lower=1)
    out["post_loan_dti"] = df["existing_dti"] + out["monthly_repayment_burden"]
    out["income_log"] = np.log1p(df["monthly_income"])
    out["employment_type_encoded"] = df["employment_type"].map(EMPLOYMENT_ENCODING).fillna(6).astype(int)
    out["geo_zone_encoded"] = df["state"].map(STATE_TO_ZONE).map(ZONE_ENCODING).fillna(1).astype(int)
    out["age"] = df["age"].astype(float)
    
    # Clip extreme synthetic alternative data values for training consistency
    out["airtime_topup_frequency"] = df["airtime_topup_frequency"].clip(upper=20.0)
    out["mobile_money_velocity"] = df["mobile_money_velocity"].clip(upper=15.0)
    out["bill_payment_regularity"] = df["bill_payment_regularity"]
    out["balance_stability_score"] = df["balance_stability_score"]
    
    out["alt_data_composite"] = (
        (out["airtime_topup_frequency"] / 30) * 0.25
        + (out["mobile_money_velocity"] / 20) * 0.30
        + (out["bill_payment_regularity"] / 100) * 0.30
        + (out["balance_stability_score"] / 100) * 0.15
    )
    out["income_x_alt_data"] = out["income_log"] * out["alt_data_composite"]
    
    out["gig_worker_flag"] = (df["employment_type"] == "gig_worker").astype(float)
    out["informal_sector_flag"] = df["employment_type"].isin(["informal_trader", "market_trader", "artisan"]).astype(float)
    out["northern_zone_flag"] = df["state"].map(STATE_TO_ZONE).isin(["NW", "NE"]).astype(float)
    
    return out[FEATURE_NAMES]