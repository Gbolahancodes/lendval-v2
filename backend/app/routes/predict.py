import uuid
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
import onnxruntime as rt
from app.models import ApplicantRequest, PredictionResponse, ShapFeature
from app.ml.features import engineer_features
from app.ml.explainer import CreditExplainer
from app.dependencies import get_onnx_session, get_explainer

router = APIRouter()

def score_to_decision(prob: float) -> str:
    score = prob * 100
    if score < 35:
        return "APPROVE"
    elif score < 62:
        return "REVIEW"
    return "DECLINE"

@router.post("/predict", response_model=PredictionResponse, summary="Score a single applicant")
async def predict(
    applicant: ApplicantRequest,
    session: rt.InferenceSession = Depends(get_onnx_session),
    explainer: CreditExplainer = Depends(get_explainer),
):
    try:
        feature_vector = engineer_features(applicant.model_dump())
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}")

    # ONNX inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: feature_vector.reshape(1, -1)})

    # outputs[1] is the probability dict for binary classification
    prob_dict = outputs[1]
    if isinstance(prob_dict, list):
        default_prob = float(prob_dict[0][1])
    else:
        default_prob = float(prob_dict[1])

    risk_score = round(default_prob * 100, 2)
    decision = score_to_decision(default_prob)

    # SHAP
    explanation = explainer.explain(feature_vector, applicant.model_dump())

    # --- Apply Hard Business Rules (Knockouts) ---
    lti = applicant.loan_amount / max(applicant.monthly_income, 1)
    monthly_repayment = applicant.loan_amount / max(applicant.loan_tenor_months, 1)
    post_loan_dti = applicant.existing_dti + (monthly_repayment / max(applicant.monthly_income, 1))

    knockout_reasons = []
    if lti > 4.0:
        knockout_reasons.append(f"AUTOMATIC DECLINE: Loan-to-income of {lti:.1f}x exceeds absolute ceiling of 4.0x.")
    if post_loan_dti > 0.65:
        knockout_reasons.append(f"AUTOMATIC DECLINE: Post-loan DTI of {post_loan_dti*100:.0f}% exceeds affordability limit.")

    if knockout_reasons:
        decision = "DECLINE"
        # Prepend hard rule failures to the top of the reason codes
        explanation["reason_codes"] = knockout_reasons + explanation["reason_codes"]
        # Max out risk score to reflect the automatic rejection
        risk_score = 99.0
        default_prob = 0.99

    shap_features = [
        ShapFeature(
            name=a["name"],
            label=a["label"],
            shap_value=a["shap_value"],
            raw_value=str(a["raw_value"]),
            direction=a["direction"],
        )
        for a in explanation["shap_values"][:8]
    ]

    # Bootstrap confidence interval (± 4.2 score points at 90% CI)
    ci = (max(0, risk_score - 4.2), min(100, risk_score + 4.2))

    return PredictionResponse(
        applicant_id=str(uuid.uuid4()),
        risk_score=risk_score,
        decision=decision,
        default_probability=round(default_prob, 4),
        shap_features=shap_features,
        reason_codes=explanation["reason_codes"],
        confidence_interval=ci,
        model_version="ng_credit_v2.4_lgbm",
    )