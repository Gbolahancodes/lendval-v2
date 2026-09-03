import ApplicantBuilder from "./components/ApplicantBuilder";

export default function App() {
  return (
    <div className="h-full flex flex-col bg-[#F4EFE4] text-[#211E1A] overflow-hidden">
      {/* Masthead */}
      <header className="shrink-0 border-b-2 border-[#211E1A] bg-[#F4EFE4] px-8 py-4 flex items-end justify-between">
        <div className="flex items-end gap-4">
          <div className="font-display text-[26px] leading-none" style={{ fontWeight: 600 }}>
            Lend<span className="italic">val</span>
          </div>
          <div className="ledger-label pb-0.5">Thin-file credit engine · Nigeria</div>
        </div>
        <div className="ledger-label"></div>
      </header>

      <main className="flex-1 overflow-hidden">
        <ApplicantBuilder />
      </main>
    </div>
  );
}
