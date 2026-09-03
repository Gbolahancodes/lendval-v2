import type { Applicant, ShapFeature } from "./scoring";
import { scoreApplicant } from "./scoring";

// Normalize the URL so it always ends with /api/v1 without duplicate slashes
const rawUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";
const cleanUrl = rawUrl.replace(/\/+$/, "");
const API_BASE = cleanUrl.endsWith("/api/v1") ? cleanUrl : `${cleanUrl}/api/v1`;

export interface ScoreResult {
  score: number;
  decision: "APPROVE" | "REVIEW" | "DECLINE";
  shap: ShapFeature[];
  reasonCodes?: string[];
  source: "backend" | "in-browser";
  modelVersion: string;
}

function toRequest(a: Applicant) {
  return {
    name: a.name,
    age: a.age,
    state: a.state,
    gender: a.gender,
    employment_type: a.employmentType,
    monthly_income: a.monthlyIncome,
    loan_amount: a.loanAmount,
    loan_tenor_months: a.loanTenorMonths,
    existing_dti: a.dti,
    airtime_topup_frequency: a.airtimeFrequency,
    mobile_money_velocity: a.mobileMoneyVelocity,
    bill_payment_regularity: a.billPaymentRegularity,
    balance_stability_score: a.balanceStability,
  };
}

/**
 * Score an applicant against the real LightGBM/ONNX backend. If the backend
 * is unreachable (not started, wrong URL, etc.), fall back to the in-browser
 * heuristic so the dashboard always returns a result.
 */
export async function predict(a: Applicant): Promise<ScoreResult> {
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(toRequest(a)),
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    return {
      score: d.risk_score,
      decision: d.decision,
      shap: (d.shap_features ?? []).map((f: any) => ({
        name: f.name,
        label: f.label,
        value: f.shap_value,
        rawValue: f.raw_value,
      })),
      reasonCodes: d.reason_codes,
      source: "backend",
      modelVersion: d.model_version ?? "ng_credit_v2",
    };
  } catch {
    const r = scoreApplicant(a);
    return { ...r, source: "in-browser", modelVersion: "in-browser heuristic" };
  }
}