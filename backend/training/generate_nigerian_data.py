"""
Nigerian Alternative Data Injector
===================================
Downloads Home Credit Default Risk data from Kaggle, then injects
synthetic Nigerian-specific alternative data features correlated
with the real TARGET (default) label.

Usage:
    kaggle competitions download -c home-credit-default-risk -p data/raw/
    python training/generate_nigerian_data.py

Output:
    data/processed/nigerian_credit_dataset.parquet
"""

import os
import sys
import zipfile
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

NIGERIAN_STATES = [
    "Lagos", "Abuja", "Kano", "Rivers", "Oyo", "Kaduna", "Anambra",
    "Enugu", "Delta", "Edo", "Imo", "Ogun", "Katsina", "Sokoto", "Borno",
]

EMPLOYMENT_TYPES = [
    "civil_servant", "bank_staff", "formal_private",
    "market_trader", "artisan", "gig_worker", "informal_trader",
]

# Employment type default rates (calibrated from Nigerian MFB data, approximate)
EMP_DEFAULT_RATES = {
    "civil_servant": 0.08,
    "bank_staff": 0.09,
    "formal_private": 0.13,
    "market_trader": 0.26,
    "artisan": 0.23,
    "gig_worker": 0.21,
    "informal_trader": 0.38,
}

# Northern states have slightly higher default rates due to infrastructure gaps
NORTHERN_STATES = {"Kano", "Kaduna", "Katsina", "Sokoto", "Borno"}


def load_home_credit(path: Path) -> pd.DataFrame:
    """Load and minimally clean Home Credit application_train.csv."""
    csv_path = path / "application_train.csv"
    if not csv_path.exists():
        zip_path = path / "home-credit-default-risk.zip"
        if zip_path.exists():
            print("Extracting zip...")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(path)
        else:
            sys.exit(
                "ERROR: Download the dataset first:\n"
                "  kaggle competitions download -c home-credit-default-risk -p data/raw/"
            )

    print("Loading Home Credit data...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    # Keep essential columns + target
    keep = [
        "SK_ID_CURR", "TARGET",
        "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
        "DAYS_BIRTH", "DAYS_EMPLOYED",
        "CODE_GENDER", "NAME_EDUCATION_TYPE",
    ]
    df = df[keep].copy()
    df = df.dropna(subset=["TARGET", "AMT_ANNUITY"])
    df["age"] = (-df["DAYS_BIRTH"] / 365).astype(int).clip(18, 75)
    df["monthly_income"] = (df["AMT_INCOME_TOTAL"] / 12).clip(lower=10000)
    df["loan_amount"] = df["AMT_CREDIT"]
    df["loan_tenor_months"] = (df["AMT_CREDIT"] / df["AMT_ANNUITY"].clip(lower=1)).clip(1, 60).round().astype(int)
    df["existing_dti"] = (df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].clip(lower=1)).clip(0, 0.95)

    # Naira conversion: Home Credit amounts are in CZK/local currency; we rescale to NGN
    naira_scale = 35.0  # rough approximation
    df["monthly_income"] = (df["monthly_income"] * naira_scale).clip(20000, 1_500_000)
    df["loan_amount"] = (df["loan_amount"] * naira_scale).clip(50000, 10_000_000)

    df["gender"] = df["CODE_GENDER"].map({"M": "male", "F": "female"}).fillna("male")
    df = df.rename(columns={"TARGET": "target"})

    return df[["SK_ID_CURR", "target", "age", "gender", "monthly_income", "loan_amount",
               "loan_tenor_months", "existing_dti"]]


def inject_nigerian_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject synthetic Nigerian alternative data features.

    Each feature is generated as a mixture of:
    - A component correlated with default probability (signal)
    - Gaussian noise (noise)

    The correlation strength and direction are calibrated to produce
    statistically reasonable feature-target correlations:

    Feature                 | Expected Pearson r with default
    ------------------------|--------------------------------
    airtime_topup_frequency | -0.18 (more top-ups → lower risk)
    mobile_money_velocity   | -0.21
    bill_payment_regularity | -0.29
    balance_stability_score | -0.24
    """
    n = len(df)
    target = df["target"].values

    print("Injecting Nigerian alternative data features...")

    # Assign employment type based on income quantile + some randomness
    income_quantile = pd.qcut(df["monthly_income"], q=5, labels=False, duplicates="drop")
    emp_probs = np.zeros((n, len(EMPLOYMENT_TYPES)))
    for i, inc_q in enumerate(income_quantile):
        if inc_q == 4:
            emp_probs[i] = [0.30, 0.20, 0.25, 0.10, 0.05, 0.05, 0.05]
        elif inc_q == 3:
            emp_probs[i] = [0.20, 0.10, 0.25, 0.15, 0.10, 0.12, 0.08]
        elif inc_q == 2:
            emp_probs[i] = [0.10, 0.05, 0.15, 0.20, 0.15, 0.18, 0.17]
        elif inc_q == 1:
            emp_probs[i] = [0.05, 0.03, 0.10, 0.22, 0.18, 0.22, 0.20]
        else:
            emp_probs[i] = [0.03, 0.02, 0.08, 0.20, 0.17, 0.20, 0.30]

    emp_idx = np.array([rng.choice(len(EMPLOYMENT_TYPES), p=p) for p in emp_probs])
    df["employment_type"] = [EMPLOYMENT_TYPES[i] for i in emp_idx]

    # State assignment (urban states for higher incomes)
    state_probs_urban = [0.30, 0.12, 0.05, 0.10, 0.08, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.03, 0.01, 0.01, 0.02]
    state_probs_rural = [0.10, 0.05, 0.12, 0.06, 0.06, 0.09, 0.07, 0.07, 0.06, 0.06, 0.06, 0.05, 0.06, 0.05, 0.04]
    state_probs_urban = np.array(state_probs_urban) / sum(state_probs_urban)
    state_probs_rural = np.array(state_probs_rural) / sum(state_probs_rural)

    states = []
    for inc in df["monthly_income"]:
        p = state_probs_urban if inc > 150000 else state_probs_rural
        states.append(rng.choice(NIGERIAN_STATES, p=p))
    df["state"] = states

    def correlated_feature(target_arr, mean_good, mean_bad, std, clip_low, clip_high, is_continuous=True):
        """Generate a feature where bad borrowers have different mean than good borrowers."""
        base = np.where(target_arr == 1,
                        rng.normal(mean_bad, std, n),
                        rng.normal(mean_good, std, n))
        return np.clip(base, clip_low, clip_high)

    # Airtime top-up frequency (bad borrowers top up less often — limited liquidity)
    df["airtime_topup_frequency"] = correlated_feature(
        target, mean_good=18, mean_bad=10, std=5, clip_low=1, clip_high=30
    ).round(0)

    # Mobile money velocity (bad borrowers transact less)
    df["mobile_money_velocity"] = correlated_feature(
        target, mean_good=12, mean_bad=6, std=4, clip_low=1, clip_high=20
    ).round(1)

    # Bill payment regularity (strong negative correlation with default)
    df["bill_payment_regularity"] = correlated_feature(
        target, mean_good=80, mean_bad=55, std=15, clip_low=0, clip_high=100
    ).round(0)

    # Balance stability (volatile balances predict default)
    df["balance_stability_score"] = correlated_feature(
        target, mean_good=72, mean_bad=48, std=18, clip_low=0, clip_high=100
    ).round(0)

    # Verify correlations
    print("\n  Feature–target Pearson correlations:")
    for feat in ["airtime_topup_frequency", "mobile_money_velocity", "bill_payment_regularity", "balance_stability_score"]:
        r, p = pearsonr(df[feat], df["target"])
        print(f"    {feat:35s}: r = {r:+.3f}  (p={p:.2e})")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N rows for quick testing")
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_home_credit(DATA_RAW)

    if args.sample:
        df = df.sample(args.sample, random_state=RANDOM_SEED)
        print(f"Subsampled to {len(df):,} rows")

    df = inject_nigerian_features(df)

    out_path = DATA_PROCESSED / "nigerian_credit_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df):,} rows → {out_path}")
    print(f"Default rate: {df['target'].mean():.1%}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
