import { useEffect } from "react";
import { useTelemetryStore } from "./store/telemetry";
import { TopBar } from "./components/TopBar";
import { StressPanel } from "./components/StressPanel";
import { SafetySessionDrawer } from "./components/SafetySessionDrawer";
import { AlertBanner } from "./components/AlertBanner";
import { KpiStrip } from "./components/KpiStrip";
import { ChartsGrid } from "./components/ChartsGrid";
import { CoreGrid } from "./components/CoreGrid";
import { HardwarePanels } from "./components/HardwarePanels";
import { NoThermalModal } from "./components/NoThermalModal";

export default function App() {
  const init = useTelemetryStore((s) => s.init);
  const emergencyAbort = useTelemetryStore((s) => s.emergencyAbort);
  const noThermalOpen = useTelemetryStore((s) => s.noThermalConfig != null);
  const cancelNoThermal = useTelemetryStore((s) => s.cancelNoThermal);

  useEffect(() => {
    let cleanup: (() => void) | undefined;
    let alive = true;
    void init().then((fn) => {
      if (alive) cleanup = fn;
      else fn();
    });
    return () => {
      alive = false;
      cleanup?.();
    };
  }, [init]);

  // Global Esc safety abort hotkey (closes modal first if open, otherwise aborts immediately).
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (noThermalOpen) {
          cancelNoThermal();
        } else {
          void emergencyAbort();
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [emergencyAbort, noThermalOpen, cancelNoThermal]);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-ink-950 text-slate-200">
      <TopBar />

      {/* Split dashboard: fixed left test/safety sidebar + scrollable telemetry charts */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <aside className="w-full shrink-0 space-y-3 overflow-y-auto border-b border-ink-700/80 bg-ink-950 p-3 lg:w-80 lg:border-r lg:border-b-0 xl:w-88">
          <StressPanel />
          <SafetySessionDrawer />
        </aside>

        <main className="min-w-0 flex-1 space-y-3 overflow-y-auto p-3.5">
          <AlertBanner />
          <KpiStrip />
          <ChartsGrid />
          <CoreGrid />
          <HardwarePanels />
        </main>
      </div>

      <NoThermalModal />
    </div>
  );
}
