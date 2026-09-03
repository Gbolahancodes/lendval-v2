from pydantic import BaseModel, Field
from typing import Literal, Optional


class ApplicantRequest(BaseModel):
    # Demographics
    name: str = "Applicant"
    age: int = Field(ge=18, le=80)
    state: str
    gender: Literal["male", "female"]
    employment_type: Literal[
        "civil_servant", "bank_staff", "formal_private",
        "market_trader", "artisan", "gig_worker", "informal_trader"
    ]

    # Financial
    monthly_income: float = Field(gt=0, description="Monthly income in NGN")
    loan_amount: float = Field(gt=0, description="Requested loan in NGN")
    loan_tenor_months: int = Field(ge=1, le=60)
    existing_dti: float = Field(ge=0.0, le=1.0, description="Current debt-to-income ratio")

    # Alternative data (0–100 normalized or raw counts)
    airtime_topup_frequency: float = Field(ge=0, description="Top-ups per month")
    mobile_money_velocity: float = Field(ge=0, description="Mobile money transactions per week")
    bill_payment_regularity: float = Field(ge=0, le=100, description="% of bills paid on time (12-month)")
    balance_stability_score: float = Field(ge=0, le=100, description="Account balance stability index")


class ShapFeature(BaseModel):
    name: str
    label: str
    shap_value: float
    raw_value: str
    direction: Literal["risk", "protective", "neutral"]


class PredictionResponse(BaseModel):
    applicant_id: str
    risk_score: float = Field(ge=0, le=100, description="Default probability × 100")
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    default_probability: float
    shap_features: list[ShapFeature]
    reason_codes: list[str]
    confidence_interval: tuple[float, float]
    model_version: str


class FairnessMetric(BaseModel):
    metric_name: str
    value: float
    threshold: float
    status: Literal["pass", "warning", "fail"]
    description: str


class FairnessResponse(BaseModel):
    group_column: str
    metrics: list[FairnessMetric]
    group_stats: list[dict]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    applicant_context: Optional[ApplicantRequest] = None


class DriftReport(BaseModel):
    feature: str
    psi_score: float
    status: Literal["stable", "warning", "drift"]
    baseline_mean: float
    current_mean: float
    delta_pct: float
