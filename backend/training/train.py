"""
Model Training Pipeline — CreditIQ Nigeria
==========================================
Trains LightGBM + logistic regression baseline, runs SHAP + Fairlearn audit,
logs everything to MLflow, and saves artifacts for ONNX export.

Usage:
    python training/train.py [--no-mlflow] [--sample 50000]
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import mlflow
import mlflow.lightgbm
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Project path setup
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.features import engineer_dataframe, FEATURE_NAMES
from app.ml.bias_audit import run_fairness_audit

DATA_PATH = Path("data/processed/nigerian_credit_dataset.parquet")
MODEL_DIR = Path("models")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")

LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "scale_pos_weight": 4.0,  # Handles class imbalance (~14% default rate)
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
    # 1: Higher value strictly increases risk | -1: strictly decreases risk | 0: no constraint
    # Mapped exactly to the order of FEATURE_NAMES in app.ml.features
    "monotone_constraints": [
        1,   # loan_to_income_ratio 
        1,   # monthly_repayment_burden
        1,   # post_loan_dti
        -1,  # income_log 
        0,   # employment_type_encoded
        0,   # geo_zone_encoded
        0,   # age
        -1,  # airtime_topup_frequency
        -1,  # mobile_money_velocity
        -1,  # bill_payment_regularity
        -1,  # balance_stability_score
        -1,  # alt_data_composite
        -1,  # income_x_alt_data
        0,   # gig_worker_flag
        0,   # informal_sector_flag
        0    # northern_zone_flag
    ],
    "monotone_constraints_method": "advanced"
}

EVAL_THRESHOLD = 0.35  # Probability threshold for binary decision

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        sys.exit(
            f"ERROR: {DATA_PATH} not found.\n"
            "Run: python training/generate_nigerian_data.py"
        )
    print(f"Loading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  {len(df):,} rows | default rate: {df['target'].mean():.1%}")
    return df

def evaluate(y_true, y_prob, name: str) -> dict:
    y_pred = (y_prob >= EVAL_THRESHOLD).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "threshold": EVAL_THRESHOLD,
    }
    print(f"\n{name}")
    print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision:{metrics['avg_precision']:.4f}")
    print(f"  Brier Score:  {metrics['brier_score']:.4f}")
    print(classification_report(y_true, y_pred, target_names=["No Default", "Default"], zero_division=0))
    return metrics

def main(use_mlflow: bool = True, sample: int = None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    if sample:
        df = df.sample(sample, random_state=42)
        print(f"Subsampled to {len(df):,} rows")

    # Feature engineering
    print("\nEngineering features...")
    X = engineer_dataframe(df)
    y = df["target"].values
    print(f"  Feature matrix: {X.shape}")

    # Train/val/test split (70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Save SHAP background sample
    bg_idx = np.random.choice(len(X_train), size=min(500, len(X_train)), replace=False)
    np.save(MODEL_DIR / "shap_background.npy", X_train.values[bg_idx])

    # Sensitive features for fairness audit
    test_mask = df.index.isin(X_test.index) if hasattr(X_test.index, 'isin') else slice(None)
    sensitive_test = df.loc[X_test.index, ["employment_type", "gender", "state"]].copy()
    sensitive_test["geo_zone"] = sensitive_test["state"].map({
        "Lagos": "SW", "Ogun": "SW", "Oyo": "SW",
        "Abuja": "NC", "Kano": "NW", "Kaduna": "NW",
        "Rivers": "SS", "Anambra": "SE", "Borno": "NE",
    }).fillna("Other")

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("nigerian_credit_scoring")
        run = mlflow.start_run(run_name="ng_credit_v2_lgbm")

    # --- Baseline: Logistic Regression ---
    print("\nTraining logistic regression baseline...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.5, class_weight={0: 1, 1: 4}, random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_val_prob = lr_pipe.predict_proba(X_val)[:, 1]
    lr_test_prob = lr_pipe.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate(y_test, lr_test_prob, "Logistic Regression (test)")

    # --- LightGBM ---
    print("\nTraining LightGBM...")
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=FEATURE_NAMES)

    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                 lgb.log_evaluation(period=50)]

    booster = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    lgbm_val_prob = booster.predict(X_val)
    lgbm_test_prob = booster.predict(X_test)
    lgbm_metrics = evaluate(y_test, lgbm_test_prob, "LightGBM (test)")
    print(f"\nAUC lift vs. baseline: +{lgbm_metrics['roc_auc'] - lr_metrics['roc_auc']:.4f}")

    # --- SHAP ---
    print("\nComputing SHAP values (may take a few minutes)...")
    explainer = shap.TreeExplainer(booster, data=X_train.values[bg_idx], feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test.values[:500])  # subset for speed

    shap_importance = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    print("\nTop 10 features by mean |SHAP|:")
    print(shap_importance.head(10).to_string(index=False))

    # --- Fairness Audit ---
    print("\nRunning Fairlearn bias audit...")
    lgbm_test_pred = (lgbm_test_prob >= EVAL_THRESHOLD).astype(int)
    fairness_results = run_fairness_audit(y_test, lgbm_test_pred, lgbm_test_prob, sensitive_test[["employment_type", "gender", "geo_zone"]])

    for group, result in fairness_results.items():
        dpd = result["demographic_parity_difference"]
        eod = result["equalized_odds_difference"]
        print(f"\n  {group}:")
        print(f"    Demographic Parity Difference: {dpd:.3f} ({'FAIL' if abs(dpd) > 0.1 else 'PASS'})")
        print(f"    Equalized Odds Difference:     {eod:.3f} ({'FAIL' if abs(eod) > 0.1 else 'PASS'})")

    # --- Save model artifacts ---
    print("\nSaving model artifacts...")
    booster.save_model(str(MODEL_DIR / "credit_model.txt"))
    shap_importance.to_csv(MODEL_DIR / "shap_importance.csv", index=False)
    
    with open(MODEL_DIR / "fairness_audit.json", "w") as f:
        json.dump(fairness_results, f, indent=2, default=str)

    metrics_summary = {
        "lgbm_roc_auc": lgbm_metrics["roc_auc"],
        "lgbm_avg_precision": lgbm_metrics["avg_precision"],
        "lgbm_brier_score": lgbm_metrics["brier_score"],
        "lr_roc_auc": lr_metrics["roc_auc"],
        "auc_lift": lgbm_metrics["roc_auc"] - lr_metrics["roc_auc"],
        "best_iteration": booster.best_iteration,
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    if use_mlflow:
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_metrics(metrics_summary)
        mlflow.lightgbm.log_model(booster, "lightgbm_model")
        mlflow.log_artifact(str(MODEL_DIR / "shap_importance.csv"))
        mlflow.log_artifact(str(MODEL_DIR / "fairness_audit.json"))
        mlflow.end_run()
        print(f"\nMLflow run saved to: {MLFLOW_URI}")

    print(f"\nArtifacts saved to: {MODEL_DIR}/")
    print("Next step: run training/export_onnx.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    
    main(use_mlflow=not args.no_mlflow, sample=args.sample)