"""
CreditIQ Nigeria — FastAPI backend

Endpoints:
  POST /predict          — score a single applicant
  POST /assistant/chat   — LLM loan officer assistant (tool-calling)
  GET  /fairness         — fairness audit metrics
  GET  /drift            — Evidently drift report
  GET  /health           — liveness check
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predict import router as predict_router
from app.routes.assistant import router as assistant_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger("creditiq")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading CreditIQ model artifacts...")
    # Model loading happens in dependencies.py on first request
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="CreditIQ Nigeria",
    description="Thin-file credit scoring API for Nigerian digital lenders",
    version="2.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api/v1", tags=["Scoring"])
app.include_router(assistant_router, prefix="/api/v1", tags=["Assistant"])


@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok", "model_version": "ng_credit_v2.4_lgbm", "framework": "fastapi"}


@app.get("/api/v1/fairness", tags=["Fairness"])
async def fairness_summary():
    """Return pre-computed fairness metrics from the last audit run."""
    return {
        "audit_date": "2025-09-02",
        "n_applications": 13798,
        "metrics": [
            {"name": "Demographic Parity Difference (Employment)", "value": 0.49, "threshold": 0.10, "status": "fail"},
            {"name": "Equalized Odds (FNR gap, Employment)", "value": 0.21, "threshold": 0.10, "status": "fail"},
            {"name": "Gender Parity Difference", "value": 0.04, "threshold": 0.05, "status": "pass"},
            {"name": "Geographic Parity (Urban vs Other)", "value": 0.22, "threshold": 0.10, "status": "fail"},
            {"name": "Calibration Error", "value": 0.03, "threshold": 0.05, "status": "pass"},
        ],
    }


@app.get("/api/v1/drift", tags=["Monitoring"])
async def drift_report():
    """Return Evidently PSI scores for key features."""
    return {
        "report_date": "2025-09-02",
        "shock_detected": True,
        "shock_description": "Post-election naira depreciation + CBN rate hike (Sep 2025)",
        "features": [
            {"feature": "post_loan_dti", "psi": 0.31, "status": "drift"},
            {"feature": "mobile_money_velocity", "psi": 0.19, "status": "warning"},
            {"feature": "bill_payment_regularity", "psi": 0.28, "status": "drift"},
            {"feature": "observed_default_rate_30d", "psi": 0.34, "status": "drift"},
            {"feature": "airtime_topup_frequency", "psi": 0.07, "status": "stable"},
            {"feature": "monthly_income_declared", "psi": 0.04, "status": "stable"},
        ],
        "recommendation": "Emergency retraining on rolling 6-month window required. Pause auto-approval for LTI > 3×.",
    }
