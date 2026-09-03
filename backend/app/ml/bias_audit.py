"""
Fairlearn bias audit for the Nigerian credit scoring model.

Audits demographic parity and equalized odds across:
- Employment type (main protected attribute in Nigerian lending context)
- Gender
- Geographic zone
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity


def run_fairness_audit(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                       sensitive_features: pd.DataFrame) -> dict:
    """
    Compute fairness metrics across multiple sensitive attributes.

    Args:
        y_true: ground truth labels (0/1)
        y_pred: binary predictions from the model
        y_prob: predicted default probabilities
        sensitive_features: DataFrame with columns like 'employment_type', 'gender', 'geo_zone'

    Returns:
        dict of metric results per sensitive attribute
    """
    results = {}

    for col in sensitive_features.columns:
        sf = sensitive_features[col]

        dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
        dpr = demographic_parity_ratio(y_true, y_pred, sensitive_features=sf)
        eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sf)

        group_stats = []
        for group in sf.unique():
            mask = sf == group
            n = int(mask.sum())
            if n == 0:
                continue
            approval_rate = float((1 - y_pred[mask]).mean())
            default_rate = float(y_true[mask].mean())
            avg_prob = float(y_prob[mask].mean())

            # False negative rate (creditworthy borrowers wrongly declined)
            creditworthy = y_true[mask] == 0
            if creditworthy.sum() > 0:
                fnr = float(y_pred[mask][creditworthy].mean())
            else:
                fnr = 0.0

            group_stats.append({
                "group": group,
                "n": n,
                "approval_rate": round(approval_rate, 4),
                "default_rate": round(default_rate, 4),
                "avg_risk_score": round(avg_prob * 100, 2),
                "false_negative_rate": round(fnr, 4),
            })

        group_stats.sort(key=lambda x: x["approval_rate"], reverse=True)

        results[col] = {
            "demographic_parity_difference": round(float(dpd), 4),
            "demographic_parity_ratio": round(float(dpr), 4),
            "equalized_odds_difference": round(float(eod), 4),
            "group_stats": group_stats,
            "pass_dpd": abs(dpd) < 0.05,
            "pass_eod": abs(eod) < 0.10,
        }

    return results


def mitigate_bias(estimator, X_train: np.ndarray, y_train: np.ndarray,
                  sensitive_features: pd.Series, constraints: str = "DemographicParity"):
    """
    Apply Fairlearn ExponentiatedGradient post-processing to reduce bias.
    Returns a mitigated predictor.
    """
    constraint = DemographicParity()
    mitigator = ExponentiatedGradient(estimator, constraints=constraint)
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
    return mitigator


def compute_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — detects feature distribution drift."""
    baseline_counts, edges = np.histogram(baseline, bins=bins)
    current_counts, _ = np.histogram(current, bins=edges)

    # Avoid division by zero
    baseline_pct = (baseline_counts + 1e-4) / (len(baseline) + 1e-4 * bins)
    current_pct = (current_counts + 1e-4) / (len(current) + 1e-4 * bins)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)
