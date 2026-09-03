import { useState, useCallback } from "react";
import type { Applicant } from "../lib/scoring";
import { PRESETS, EMPLOYMENT_OPTIONS, NIGERIAN_STATES } from "../lib/scoring";
import { predict, type ScoreResult } from "../lib/api";
import RiskGauge from "./RiskGauge";

const DEFAULT: Omit<Applicant, "id"> = {
  name: "Chidi Obi",
  age: 32,
  state: "Lagos",
  gender: "male",
  employmentType: "gig_worker",
  monthlyIncome: 95000,
  loanAmount: 400000,
  loanTenorMonths: 9,
  dti: 0.42,
  airtimeFrequency: 14,
  mobileMoneyVelocity: 9,
  billPaymentRegularity: 67,
  balanceStability: 58,
};

const GREEN = "#2E7D57";
const RED = "#B23A2E";

function Slider({
  label, value, min, max, step = 1, format, onChange,
}: {
  label: string; value: number; min: number; max: number; step?: number;
  format: (v: number) => string; onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex justify-between items-baseline mb-1.5">
        <label className="text-[12px] text-[#9A9284]">{label}</label>
        <span className="text-[12px] font-mono text-[#EFE9DC]">{format(value)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(+e.target.value)} className="w-full" />
    </div>
  );
}

const inkInput =
  "w-full bg-[rgba(239,233,220,0.06)] border border-[rgba(239,233,220,0.16)] rounded-sm px-3 py-2 text-[13px] text-[#EFE9DC] outline-none focus:border-[#2E7D57] transition-colors";

const PRESET_LABELS = ["Salaried", "Trader", "Gig Worker"];

export default function ApplicantBuilder() {
  const [form, setForm] = useState<Omit<Applicant, "id">>(DEFAULT);
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [loading, setLoading] = useState(false);

  const set = useCallback(<K extends keyof typeof form>(k: K, v: typeof form[K]) => {
    setForm((prev) => ({ ...prev, [k]: v }));
    setResult(null);
  }, []);

  function applyPreset(idx: number) {
    setForm((prev) => ({ ...prev, ...PRESETS[idx] }));
    setResult(null);
  }

  async function run() {
    setLoading(true);
    const r = await predict({ ...form, id: "preview" });
    setResult(r);
    setLoading(false);
  }

  const adverse = result ? result.shap.filter((f) => f.value > 1).sort((a, b) => b.value - a.value).slice(0, 3) : [];
  const favourable = result ? result.shap.filter((f) => f.value < -1).sort((a, b) => a.value - b.value).slice(0, 3) : [];

  return (
    <div className="flex-1 flex flex-col lg:grid lg:grid-cols-[400px_1fr] lg:overflow-hidden">
      {/* LEFT — application form */}
      <div className="on-panel bg-[#211E1A] text-[#EFE9DC] lg:overflow-y-auto border-b-2 lg:border-b-0 lg:border-r-2 border-[#211E1A]">
        <div className="px-4 sm:px-6 py-5 space-y-6">
          <div className="flex flex-wrap gap-1.5">
            {PRESET_LABELS.map((label, i) => (
              <button key={label} onClick={() => applyPreset(i)}
                className="px-2.5 py-1 rounded-sm text-[11px] border border-[rgba(239,233,220,0.18)] text-[#9A9284] hover:text-[#EFE9DC] hover:border-[rgba(239,233,220,0.4)] transition-colors">
                {label}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="text-[12px] text-[#9A9284] block mb-1.5">Name</label>
              <input type="text" value={form.name} onChange={(e) => set("name", e.target.value)} className={inkInput} />
            </div>
            <div>
              <label className="text-[12px] text-[#9A9284] block mb-1.5">Age</label>
              <input type="number" value={form.age} min={18} max={70} onChange={(e) => set("age", +e.target.value)} className={inkInput} />
            </div>
            <div>
              <label className="text-[12px] text-[#9A9284] block mb-1.5">State</label>
              <select value={form.state} onChange={(e) => set("state", e.target.value)} className={inkInput}>
                {NIGERIAN_STATES.map((s) => <option key={s} value={s} style={{ background: "#211E1A" }}>{s}</option>)}
              </select>
            </div>
            <div>
              <label className="text-[12px] text-[#9A9284] block mb-1.5">Employment</label>
              <select value={form.employmentType} onChange={(e) => set("employmentType", e.target.value)} className={inkInput}>
                {EMPLOYMENT_OPTIONS.map((o) => <option key={o.value} value={o.value} style={{ background: "#211E1A" }}>{o.label}</option>)}
              </select>
            </div>
          </div>

          <div className="space-y-4">
            <Slider label="Monthly income" value={form.monthlyIncome} min={20000} max={500000} step={5000}
              format={(v) => `₦${(v / 1000).toFixed(0)}k`} onChange={(v) => set("monthlyIncome", v)} />
            <Slider label="Loan amount" value={form.loanAmount} min={50000} max={5000000} step={25000}
              format={(v) => v >= 1000000 ? `₦${(v / 1000000).toFixed(1)}M` : `₦${(v / 1000).toFixed(0)}k`}
              onChange={(v) => set("loanAmount", v)} />
            <div>
              <label className="text-[12px] text-[#9A9284] block mb-1.5">Loan tenor</label>
              <div className="flex flex-wrap gap-1.5">
                {[3, 6, 9, 12, 18, 24].map((m) => (
                  <button key={m} onClick={() => set("loanTenorMonths", m)}
                    className="px-2.5 py-1 rounded-sm text-[11px] font-mono border transition-colors"
                    style={{
                      background: form.loanTenorMonths === m ? "rgba(46,125,87,0.18)" : "transparent",
                      borderColor: form.loanTenorMonths === m ? "#2E7D57" : "rgba(239,233,220,0.14)",
                      color: form.loanTenorMonths === m ? "#EFE9DC" : "#9A9284",
                    }}>
                    {m}
                  </button>
                ))}
              </div>
            </div>
            <Slider label="Existing debt-to-income" value={form.dti} min={0} max={0.9} step={0.01}
              format={(v) => `${(v * 100).toFixed(0)}%`} onChange={(v) => set("dti", v)} />
          </div>

          <div className="space-y-4 pt-2 border-t border-[rgba(239,233,220,0.14)]">
            <div className="ledger-label" style={{ color: "#9A9284" }}>Alternative data</div>
            <Slider label="Airtime top-ups" value={form.airtimeFrequency} min={1} max={30}
              format={(v) => `${v}×/mo`} onChange={(v) => set("airtimeFrequency", v)} />
            <Slider label="Mobile money velocity" value={form.mobileMoneyVelocity} min={1} max={20}
              format={(v) => `${v}/wk`} onChange={(v) => set("mobileMoneyVelocity", v)} />
            <Slider label="Bill regularity" value={form.billPaymentRegularity} min={0} max={100}
              format={(v) => `${v}%`} onChange={(v) => set("billPaymentRegularity", v)} />
            <Slider label="Balance stability" value={form.balanceStability} min={0} max={100}
              format={(v) => `${v}`} onChange={(v) => set("balanceStability", v)} />
          </div>

          <button onClick={run} disabled={loading}
            className="w-full py-2.5 rounded-sm text-[13px] font-medium transition-colors cursor-pointer"
            style={{ background: GREEN, color: "#F4EFE4", opacity: loading ? 0.6 : 1 }}>
            {loading ? "Scoring…" : "Run assessment"}
          </button>
        </div>
      </div>

      {/* RIGHT — decision */}
      <div className="flex-1 overflow-y-auto bg-[#F4EFE4]">
        <div className="max-w-[640px] px-4 sm:px-8 py-6 sm:py-8 mx-auto lg:mx-0">
          {!result ? (
            <div className="flex flex-col items-center justify-center text-center py-12 lg:pt-24">
              <div className="font-display italic text-[20px] sm:text-[22px] text-[#A79E8D]">Awaiting assessment</div>
              <p className="text-[13px] text-[#6B6459] mt-2 max-w-xs">
                Complete the applicant details and run the assessment to see the decision and the reasons behind it.
              </p>
            </div>
          ) : (
            <div className="space-y-6 animate-fade-in">
              <div className="flex items-baseline justify-between">
                <div className="font-display text-[20px] sm:text-[22px]" style={{ fontWeight: 500 }}>{form.name}</div>
                <div className="ledger-label">
                  {result.source === "backend" ? "LightGBM model" : "in-browser model"}
                </div>
              </div>

              <RiskGauge score={result.score} decision={result.decision} />

              {/* Why this decision */}
              <div className="rounded-sm border border-[#DDD4C3] bg-[#FBF8F1] p-4 sm:p-5">
                <div className="ledger-label mb-3">Why this decision</div>

                {adverse.length > 0 && (
                  <div className="mb-4">
                    <div className="text-[11px] font-medium mb-2" style={{ color: RED }}>Raised risk</div>
                    <div className="space-y-1.5">
                      {adverse.map((f) => (
                        <div key={f.name} className="flex items-baseline justify-between gap-3">
                          <span className="text-[13px] text-[#211E1A]">{f.label}</span>
                          <span className="font-mono text-[11px] text-[#6B6459]">
                            {typeof f.rawValue === "number" ? f.rawValue.toFixed(2) : String(f.rawValue)} · +{f.value.toFixed(1)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {favourable.length > 0 && (
                  <div>
                    <div className="text-[11px] font-medium mb-2" style={{ color: GREEN }}>Lowered risk</div>
                    <div className="space-y-1.5">
                      {favourable.map((f) => (
                        <div key={f.name} className="flex items-baseline justify-between gap-3">
                          <span className="text-[13px] text-[#211E1A]">{f.label}</span>
                          <span className="font-mono text-[11px] text-[#6B6459]">
                            {typeof f.rawValue === "number" ? f.rawValue.toFixed(2) : String(f.rawValue)} · {f.value > 0 ? "+" : ""}{f.value.toFixed(1)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {adverse.length === 0 && favourable.length === 0 && (
                  <div className="text-[13px] text-[#6B6459]">No single factor moved the score materially.</div>
                )}
              </div>

              {/* Key numbers */}
              <div className="grid grid-cols-2 sm:flex sm:gap-8 gap-4 text-[13px]">
                <div>
                  <div className="ledger-label">Monthly repayment</div>
                  <div className="font-mono text-[15px] sm:text-[16px] mt-1">₦{(form.loanAmount / form.loanTenorMonths).toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                </div>
                <div>
                  <div className="ledger-label">Loan-to-income</div>
                  <div className="font-mono text-[15px] sm:text-[16px] mt-1">{(form.loanAmount / form.monthlyIncome).toFixed(1)}×</div>
                </div>
              </div>

              {/* Plain English Summary */}
              <div className="rounded-sm border border-[#DDD4C3] bg-[#FBF8F1] p-4 sm:p-5">
                <div className="ledger-label mb-3">Decision Summary</div>
                {result.reasonCodes && result.reasonCodes.length > 0 ? (
                  <ul className="list-disc pl-4 space-y-1.5 text-[13px] text-[#211E1A]">
                    {result.reasonCodes.map((code, idx) => (
                      <li key={idx} className="leading-relaxed">{code}</li>
                    ))}
                  </ul>
                ) : result.decision === "APPROVE" ? (
                  <p className="text-[13px] text-[#211E1A] leading-relaxed">
                    Applicant meets lending criteria. Financial ratios are within healthy limits, and alternative data signals indicate stable economic activity.
                  </p>
                ) : (
                  <p className="text-[13px] text-[#211E1A] leading-relaxed">
                    Application requires manual review based on a cumulative combination of marginal risk factors.
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}