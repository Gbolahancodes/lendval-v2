"""
SHAP explanation generation for the credit scoring model.
Produces regulator-grade reason codes per CBN / NDPC requirements.
"""
import shap
import numpy as np
from typing import Optional
import lightgbm as lgb
from app.ml.features import FEATURE_LABELS, FEATURE_NAMES, engineer_features

class CreditExplainer:
    def __init__(self, model: lgb.Booster, X_background: Optional[np.ndarray] = None):
        self.model = model
        if X_background is not None:
            self.explainer = shap.TreeExplainer(model, data=X_background, feature_perturbation="interventional")
        else:
            self.explainer = shap.TreeExplainer(model)

    def explain(self, feature_vector: np.ndarray, applicant_row: dict) -> dict:
        x = feature_vector.reshape(1, -1)
        shap_values = self.explainer.shap_values(x)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
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
        codes = []
        for attr in attributions:
            if attr["direction"] != "risk":
                continue
            name = attr["name"]
            raw = attr["raw_value"]

            if name == "loan_to_income_ratio" and raw > 3.0:
                codes.append(f"Loan-to-income ratio of {raw:.1f}× exceeds the recommended ceiling of 3.0×")
            elif name == "monthly_repayment_burden" and raw > 0.30:
                codes.append(f"Monthly repayment burden of {raw * 100:.0f}% of income leaves insufficient buffer")
            elif name == "post_loan_dti" and raw > 0.45:
                codes.append(f"Post-disbursement DTI of {raw * 100:.0f}% exceeds the 45% policy threshold")
            elif name == "bill_payment_regularity" and raw < 70:
                codes.append(f"Bill payment regularity of {raw:.0f}% indicates {100 - raw:.0f}% missed/late payments")
            elif name == "mobile_money_velocity" and raw < 8:
                codes.append(f"Mobile money velocity of {raw:.0f} transactions/week limits income verifiability")
            elif name == "balance_stability_score" and raw < 60:
                codes.append(f"Balance stability score of {raw:.0f}/100 signals volatile cash flow patterns")

            if len(codes) >= 3:
                break
        return codes