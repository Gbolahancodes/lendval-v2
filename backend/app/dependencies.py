"""Shared FastAPI dependencies for model loading."""

import os
from functools import lru_cache
from pathlib import Path

import onnxruntime as rt
import lightgbm as lgb
import numpy as np

from app.ml.explainer import CreditExplainer

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

@lru_cache(maxsize=1)
def _load_onnx_session() -> rt.InferenceSession:
    path = MODEL_DIR / "credit_model.onnx"
    if not path.exists():
        raise RuntimeError(f"ONNX model not found at {path}. Run training/train.py first.")
    return rt.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _load_explainer() -> CreditExplainer:
    lgb_path = MODEL_DIR / "credit_model.txt"
    bg_path = MODEL_DIR / "shap_background.npy"
    if not lgb_path.exists():
        raise RuntimeError(f"LightGBM model not found at {lgb_path}.")
    booster = lgb.Booster(model_file=str(lgb_path))
    background = np.load(str(bg_path)) if bg_path.exists() else None
    return CreditExplainer(booster, background)


def get_onnx_session() -> rt.InferenceSession:
    return _load_onnx_session()


def get_explainer() -> CreditExplainer:
    return _load_explainer()
