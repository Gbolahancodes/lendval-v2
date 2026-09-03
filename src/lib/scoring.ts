export interface Applicant {
  id: string;
  name: string;
  age: number;
  state: string;
  gender: "male" | "female";
  employmentType: string;
  monthlyIncome: number;
  loanAmount: number;
  loanTenorMonths: number;
  dti: number;
  // Alternative data
  airtimeFrequency: number; // top-ups per month
  mobileMoneyVelocity: number; // transactions per week
  billPaymentRegularity: number; // 0–100
  balanceStability: number; // 0–100
  // Derived
  creditScore?: number;
  decision?: "APPROVE" | "REVIEW" | "DECLINE";
  shapValues?: ShapFeature[];
}

export interface ShapFeature {
  name: string;
  value: number; // SHAP contribution (positive = increases risk)
  rawValue: number | string;
  label: string;
}

// Model baseline: average portfolio default-risk score before any applicant
// signal is applied. Every contribution below is centered so that a typical
// applicant sits near this baseline, and each is clamped so no single feature
// can dominate the decision.
export const BASELINE_SCORE = 30;

export const DECISION_BANDS = {
  approve: { min: 0, max: 34 },
  review: { min: 35, max: 60 },
  decline: { min: 61, max: 100 },
};

const EMPLOYMENT_RISK: Record<string, number> = {
  civil_servant: -10,
  bank_staff: -12,
  formal_private: -6,
  market_trader: 5,
  artisan: 7,
  gig_worker: 8,
  informal_trader: 12,
};

const EMPLOYMENT_LABELS: Record<string, string> = {
  civil_servant: "Civil Servant",
  bank_staff: "Bank Staff",
  formal_private: "Formal Private Sector",
  market_trader: "Market Trader",
  artisan: "Artisan / Craftsperson",
  gig_worker: "Gig Worker (Bolt/Uber)",
  informal_trader: "Informal Trader",
};

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const round1 = (v: number) => +v.toFixed(1);

export function scoreApplicant(a: Applicant): {
  score: number;
  decision: "APPROVE" | "REVIEW" | "DECLINE";
  shap: ShapFeature[];
} {
  const loanToIncome = a.loanAmount / Math.max(a.monthlyIncome, 1);
  const monthlyRepayment = a.loanAmount / Math.max(a.loanTenorMonths, 1);
  const repaymentRatio = monthlyRepayment / Math.max(a.monthlyIncome, 1);

  const contributions: ShapFeature[] = [
    {
      // Affordability is the dominant driver. Centered at 15% of income.
      name: "repayment_ratio",
      label: "Monthly Repayment Burden",
      value: round1(clamp((repaymentRatio - 0.15) * 65, -14, 26)),
      rawValue: `${(repaymentRatio * 100).toFixed(0)}% of income`,
    },
    {
      // Existing debt load. Centered at a 35% DTI.
      name: "dti",
      label: "Debt-to-Income Ratio",
      value: round1(clamp((a.dti - 0.35) * 42, -13, 22)),
      rawValue: `${(a.dti * 100).toFixed(0)}%`,
    },
    {
      // Loan size relative to income. Neutral around 1.5× monthly income.
      name: "loan_to_income",
      label: "Loan-to-Income Ratio",
      value: round1(clamp((loanToIncome - 1.5) * 4.5, -10, 15)),
      rawValue: `${loanToIncome.toFixed(1)}×`,
    },
    {
      name: "employment_type",
      label: "Employment Type",
      value: EMPLOYMENT_RISK[a.employmentType] ?? 0,
      rawValue: EMPLOYMENT_LABELS[a.employmentType] ?? a.employmentType,
    },
    {
      // Higher income lowers risk. Neutral at ₦150k/month, bounded both ways.
      name: "monthly_income",
      label: "Monthly Income",
      value: round1(clamp(((150000 - a.monthlyIncome) / 150000) * 12, -12, 14)),
      rawValue: `₦${a.monthlyIncome.toLocaleString()}`,
    },
    {
      name: "airtime_frequency",
      label: "Airtime Top-up Frequency",
      value: round1(clamp(((15 - a.airtimeFrequency) / 15) * 6, -5, 6)),
      rawValue: `${a.airtimeFrequency}×/month`,
    },
    {
      name: "mobile_money_velocity",
      label: "Mobile Money Velocity",
      value: round1(clamp(((8 - a.mobileMoneyVelocity) / 8) * 8, -6, 8)),
      rawValue: `${a.mobileMoneyVelocity} txns/week`,
    },
    {
      name: "bill_payment_regularity",
      label: "Bill Payment Regularity",
      value: round1(clamp((70 - a.billPaymentRegularity) / 8, -6, 6)),
      rawValue: `${a.billPaymentRegularity}%`,
    },
    {
      name: "balance_stability",
      label: "Balance Stability Score",
      value: round1(clamp((65 - a.balanceStability) / 10, -5, 5)),
      rawValue: `${a.balanceStability}/100`,
    },
  ];

  const total = BASELINE_SCORE + contributions.reduce((s, f) => s + f.value, 0);
  const score = clamp(total, 3, 97);

  let decision: "APPROVE" | "REVIEW" | "DECLINE";
  if (score <= DECISION_BANDS.approve.max) decision = "APPROVE";
  else if (score <= DECISION_BANDS.review.max) decision = "REVIEW";
  else decision = "DECLINE";

  return { score: round1(score), decision, shap: contributions };
}

export const PRESETS: Partial<Applicant>[] = [
  {
    name: "Adaeze Okonkwo",
    age: 34,
    state: "Abuja",
    gender: "female",
    employmentType: "civil_servant",
    monthlyIncome: 145000,
    loanAmount: 500000,
    loanTenorMonths: 12,
    dti: 0.25,
    airtimeFrequency: 18,
    mobileMoneyVelocity: 12,
    billPaymentRegularity: 88,
    balanceStability: 82,
  },
  {
    name: "Musa Aliyu",
    age: 41,
    state: "Kano",
    gender: "male",
    employmentType: "market_trader",
    monthlyIncome: 78000,
    loanAmount: 300000,
    loanTenorMonths: 6,
    dti: 0.48,
    airtimeFrequency: 26,
    mobileMoneyVelocity: 18,
    billPaymentRegularity: 72,
    balanceStability: 61,
  },
  {
    name: "Emeka Nwosu",
    age: 28,
    state: "Lagos",
    gender: "male",
    employmentType: "gig_worker",
    monthlyIncome: 92000,
    loanAmount: 400000,
    loanTenorMonths: 9,
    dti: 0.55,
    airtimeFrequency: 22,
    mobileMoneyVelocity: 15,
    billPaymentRegularity: 65,
    balanceStability: 55,
  },
];

export const EMPLOYMENT_OPTIONS = Object.entries(EMPLOYMENT_LABELS).map(([value, label]) => ({ value, label }));

export const NIGERIAN_STATES = [
  "Lagos", "Abuja", "Kano", "Rivers", "Oyo", "Kaduna", "Anambra",
  "Enugu", "Delta", "Edo", "Imo", "Ogun", "Katsina", "Sokoto", "Borno",
];
