import { DECISION_BANDS } from "../lib/scoring";

interface Props {
  score: number;
  decision: "APPROVE" | "REVIEW" | "DECLINE";
}

const DECISION_META = {
  APPROVE: {
    label: "Approve",
    color: "#2E7D57",
    tint: "rgba(46,125,87,0.1)",
    description: "Low default risk. Proceed on standard terms.",
  },
  REVIEW: {
    label: "Manual Review",
    color: "#B9821B",
    tint: "rgba(185,130,27,0.12)",
    description: "Borderline profile. Senior officer review required.",
  },
  DECLINE: {
    label: "Decline",
    color: "#B23A2E",
    tint: "rgba(178,58,46,0.1)",
    description: "Elevated default risk. Below current lending criteria.",
  },
};

export default function RiskGauge({ score, decision }: Props) {
  const meta = DECISION_META[decision];
  const bands = [
    { key: "APPROVE", label: "Approve", range: `0–${DECISION_BANDS.approve.max}`, color: "#2E7D57", width: DECISION_BANDS.approve.max + 1 },
    { key: "REVIEW", label: "Review", range: `${DECISION_BANDS.review.min}–${DECISION_BANDS.review.max}`, color: "#B9821B", width: DECISION_BANDS.review.max - DECISION_BANDS.review.min + 1 },
    { key: "DECLINE", label: "Decline", range: `${DECISION_BANDS.decline.min}–100`, color: "#B23A2E", width: 100 - DECISION_BANDS.decline.min + 1 },
  ];

  return (
    <div className="rounded-sm border border-[#DDD4C3] bg-[#FBF8F1] p-5">
      <div className="flex items-start justify-between gap-6">
        {/* Score */}
        <div>
          <div className="ledger-label">Default-risk score</div>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="font-display leading-none text-[#211E1A]" style={{ fontSize: 64, fontWeight: 600 }}>
              {score.toFixed(0)}
            </span>
            <span className="font-mono text-[13px] text-[#A79E8D]">/ 100</span>
          </div>
        </div>

        {/* Decision stamp */}
        <div
          className="shrink-0 text-right rounded-sm border px-4 py-3"
          style={{ borderColor: meta.color, background: meta.tint }}
        >
          <div className="ledger-label" style={{ color: meta.color }}>Decision</div>
          <div className="font-display text-[22px] leading-tight mt-0.5" style={{ color: meta.color, fontWeight: 600 }}>
            {meta.label}
          </div>
        </div>
      </div>

      <p className="text-[12px] text-[#6B6459] mt-3 leading-relaxed">{meta.description}</p>

      {/* Linear risk rule */}
      <div className="mt-5">
        <div className="relative h-2.5 flex rounded-sm overflow-hidden border border-[#DDD4C3]">
          {bands.map((b) => (
            <div key={b.key} style={{ width: `${b.width}%`, background: b.color, opacity: decision === b.key ? 0.9 : 0.28 }} />
          ))}
          {/* Marker */}
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2"
            style={{ left: `${score}%` }}
          >
            <div className="w-0.5 h-5 bg-[#211E1A]" />
          </div>
        </div>
        <div className="flex justify-between mt-2">
          {bands.map((b) => (
            <div key={b.key} className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-sm" style={{ background: b.color, opacity: decision === b.key ? 1 : 0.35 }} />
              <span className="font-mono text-[10px]" style={{ color: decision === b.key ? "#211E1A" : "#A79E8D" }}>
                {b.label} <span className="text-[#A79E8D]">{b.range}</span>
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
