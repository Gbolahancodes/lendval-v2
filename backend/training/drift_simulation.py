"""
Evidently AI Drift Simulation
===============================
Simulates a macro-economic shock (post-election naira depreciation, Sep 2025)
and generates a drift report comparing baseline (pre-shock) vs current (post-shock) data.

Usage:
    python training/drift_simulation.py

Outputs:
    reports/drift_report.html  — interactive Evidently HTML report
    reports/drift_metrics.json — machine-readable PSI scores
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently import ColumnMapping
except ImportError:
    sys.exit("Install evidently: pip install evidently")

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.ml.bias_audit import compute_psi

DATA_PATH = Path("data/processed/nigerian_credit_dataset.parquet")
REPORTS_DIR = Path("reports")
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def simulate_shock(df: pd.DataFrame, shock_intensity: float = 1.0) -> pd.DataFrame:
    """
    Apply post-election macro shock to feature distributions.

    Simulates:
    - Naira depreciation → reduced purchasing power → higher DTI
    - CBN rate hike → higher borrowing costs → lower mobile money activity
    - Market contraction → reduced bill payment regularity
    """
    shocked = df.copy()

    # DTI rises as inflation erodes real income
    shocked["existing_dti"] = (shocked["existing_dti"] + rng.normal(0.14 * shock_intensity, 0.03, len(df))).clip(0, 0.95)

    # Mobile money velocity drops (less disposable income)
    shocked["mobile_money_velocity"] = (
        shocked["mobile_money_velocity"] - rng.normal(4.5 * shock_intensity, 1.0, len(df))
    ).clip(1, 20)

    # Bill payment regularity drops sharply
    shocked["bill_payment_regularity"] = (
        shocked["bill_payment_regularity"] - rng.normal(18 * shock_intensity, 4, len(df))
    ).clip(0, 100)

    # Balance stability drops
    shocked["balance_stability_score"] = (
        shocked["balance_stability_score"] - rng.normal(12 * shock_intensity, 3, len(df))
    ).clip(0, 100)

    # Default rate increases
    base_default_prob = shocked["target"].mean()
    additional_defaults = rng.binomial(1, 0.10 * shock_intensity, len(df))
    shocked["target"] = np.clip(shocked["target"] + additional_defaults, 0, 1)

    return shocked


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        sys.exit(f"ERROR: {DATA_PATH} not found. Run generate_nigerian_data.py first.")

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    n = len(df)

    # Pre-shock (baseline): first 6 months
    baseline = df.sample(frac=0.5, random_state=42).reset_index(drop=True)
    # Post-shock (current): second half with shock applied
    remaining_idx = df.index.difference(baseline.index)
    current_raw = df.loc[remaining_idx].reset_index(drop=True)
    current = simulate_shock(current_raw, shock_intensity=1.0)

    print(f"Baseline: {len(baseline):,} rows · default rate: {baseline['target'].mean():.1%}")
    print(f"Post-shock: {len(current):,} rows · default rate: {current['target'].mean():.1%}")

    # PSI computation
    alt_features = [
        "existing_dti",
        "airtime_topup_frequency",
        "mobile_money_velocity",
        "bill_payment_regularity",
        "balance_stability_score",
    ]

    print("\nPopulation Stability Index (PSI):")
    psi_results = {}
    for feat in alt_features:
        psi = compute_psi(baseline[feat].values, current[feat].values)
        status = "stable" if psi < 0.10 else ("warning" if psi < 0.25 else "drift")
        psi_results[feat] = {"psi": round(psi, 4), "status": status}
        flag = "🔴" if status == "drift" else ("🟡" if status == "warning" else "🟢")
        print(f"  {flag} {feat:35s}: PSI = {psi:.4f} ({status})")

    # Save PSI metrics
    with open(REPORTS_DIR / "drift_metrics.json", "w") as f:
        json.dump({
            "shock_description": "Post-election naira depreciation + CBN rate hike (Sep 2025)",
            "baseline_n": len(baseline),
            "current_n": len(current),
            "baseline_default_rate": round(float(baseline["target"].mean()), 4),
            "current_default_rate": round(float(current["target"].mean()), 4),
            "psi_scores": psi_results,
        }, f, indent=2)

    # Evidently HTML report
    print("\nGenerating Evidently drift report...")
    feature_cols = alt_features + ["monthly_income", "loan_amount", "age"]
    column_mapping = ColumnMapping(target="target", numerical_features=feature_cols)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline[feature_cols + ["target"]],
               current_data=current[feature_cols + ["target"]],
               column_mapping=column_mapping)

    html_path = REPORTS_DIR / "drift_report.html"
    report.save_html(str(html_path))
    print(f"  Saved: {html_path}")

    json_path = REPORTS_DIR / "drift_report.json"
    report.save_json(str(json_path))
    print(f"  Saved: {json_path}")

    print("\nDrift simulation complete.")
    print(f"Open {html_path} in a browser to view the interactive report.")


if __name__ == "__main__":
    main()
