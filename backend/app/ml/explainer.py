"""
SHAP explanation generation for the credit scoring model.
Produces regulator-grade reason codes per CBN / NDPC requirements.
"""

import shap
import numpy as np
from typing import Optional
import lightgbm as lgb

from app.ml.features import FEATURE_LABELS, FEATURE_NAMES, engineer_features


REASON_CODE_TEMPLATES = {
    "loan_to_income_ratio": "Loan amount is {val} times monthly income, exceeding the recommended ceiling of 3×",
    "monthly_repayment_burden": "Estimated monthly repayment of ₦{val} represents {pct}% of declared income",
    "post_loan_dti": "Post-disbursement debt-to-income ratio of {val}% exceeds 45% policy threshold",
    "airtime_topup_frequency": "Mobile activity (airtime top-ups: {val}×/month) indicates limited economic engagement",
    "mobile_money_velocity": "Low mobile money transaction velocity ({val} txns/week) reduces income verifiability",
    "bill_payment_regularity": "Bill payment history shows only {val}% on-time payments over the last 12 months",
    "balance_stability_score": "Account balance stability score of {val}/100 signals volatile cash flow",
    "employment_type_encoded": "Employment category ({val}) carries elevated default probability in historical data",
    "existing_dti": "Existing debt obligations consume {val}% of income before loan disbursement",
}


class CreditExplainer:
    def __init__(self, model: lgb.Booster, X_background: Optional[np.ndarray] = None):
        self.model = model
        if X_background is not None:
            self.explainer = shap.TreeExplainer(model, data=X_background, feature_perturbation="interventional")
        else:
            self.explainer = shap.TreeExplainer(model)

    def explain(self, feature_vector: np.ndarray, applicant_row: dict) -> dict:
        """
        Returns SHAP values and formatted reason codes for a single applicant.

        Args:
            feature_vector: engineered feature array (shape: [n_features])
            applicant_row: raw applicant dict for formatting reason codes

        Returns:
            dict with shap_values, base_value, feature_attributions, reason_codes
        """
        x = feature_vector.reshape(1, -1)
        shap_values = self.explainer.shap_values(x)

        # For binary classification LightGBM, shap_values is shape [n_samples, n_features]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # class=1 (default) SHAP values
        else:
            sv = shap_values[0]

        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1])
        else:
            base_value = float(base_value)

        attributions = [
            {
                "name": FEATURE_NAMES[i],
                "label": FEATURE_LABELS.get(FEATURE_NAMES[i], FEATURE_NAMES[i]),
                "shap_value": float(sv[i]),
                "raw_value": float(feature_vector[i]),
                "direction": "risk" if sv[i] > 0.005 else ("protective" if sv[i] < -0.005 else "neutral"),
            }
            for i in range(len(FEATURE_NAMES))
        ]

        attributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        reason_codes = self._generate_reason_codes(attributions, applicant_row)

        return {
            "shap_values": attributions,
            "base_value": base_value,
            "final_value": base_value + float(sv.sum()),
            "reason_codes": reason_codes,
        }

    def _generate_reason_codes(self, attributions: list, row: dict) -> list[str]:
        """Generate plain-English adverse action reason codes for top risk factors."""
        codes = []
        for attr in attributions:
            if attr["direction"] != "risk":
                continue
            name = attr["name"]
            raw = attr["raw_value"]

            if name == "loan_to_income_ratio":
                codes.append(f"Loan-to-income ratio of {raw:.1f}× exceeds the recommended ceiling of 3×")
            elif name == "monthly_repayment_burden":
                codes.append(f"Monthly repayment burden of {raw * 100:.0f}% of income leaves insufficient buffer")
            elif name == "post_loan_dti":
                codes.append(f"Post-disbursement DTI of {raw * 100:.0f}% exceeds the 45% policy threshold")
            elif name == "bill_payment_regularity":
                codes.append(f"Bill payment regularity of {raw:.0f}% indicates {100 - raw:.0f}% missed/late payments")
            elif name == "mobile_money_velocity":
                codes.append(f"Mobile money velocity of {raw:.0f} transactions/week limits income verifiability")
            elif name == "balance_stability_score":
                codes.append(f"Balance stability score of {raw:.0f}/100 signals volatile cash flow patterns")

            if len(codes) >= 3:
                break

        return codes
