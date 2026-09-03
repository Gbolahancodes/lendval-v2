import ApplicantBuilder from "./components/ApplicantBuilder";

export default function App() {
  return (
    <div className="min-h-screen lg:h-full flex flex-col bg-[#F4EFE4] text-[#211E1A] overflow-y-auto lg:overflow-hidden">
      {/* Masthead */}
      <header className="shrink-0 border-b-2 border-[#211E1A] bg-[#F4EFE4] px-4 sm:px-8 py-4 flex items-center justify-between">
        <div className="flex flex-wrap items-baseline gap-2 sm:gap-4">
          <div className="font-display text-[22px] sm:text-[26px] leading-none" style={{ fontWeight: 600 }}>
            Lend<span className="italic">val</span>
          </div>
          <div className="ledger-label pb-0.5 text-[9px] sm:text-[10px]">Thin-file credit engine · Nigeria</div>
        </div>
      </header>
      <main className="flex-1 flex flex-col min-h-0">
        <ApplicantBuilder />
      </main>
    </div>
  );
}