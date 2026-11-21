import { useTelemetryStore } from "../store/telemetry";

export function NoThermalModal() {
  const cfg = useTelemetryStore((s) => s.noThermalConfig);
  const confirmNoThermal = useTelemetryStore((s) => s.confirmNoThermal);
  const cancelNoThermal = useTelemetryStore((s) => s.cancelNoThermal);

  if (!cfg) return null;

  return (
    <div
      data-testid="no-thermal-modal"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-xs"
    >
      <div className="w-full max-w-md rounded-xl border border-amber-500/60 bg-ink-900 p-5 shadow-2xl">
        <div className="mb-2 flex items-center gap-2 text-amber-400">
          <span className="rounded bg-amber-500/20 px-2 py-0.5 text-xs font-bold uppercase">Warning</span>
          <h2 className="text-sm font-bold text-white">CPU Thermal Protection Unavailable</h2>
        </div>
        <p className="mb-4 text-xs leading-relaxed text-slate-300">
          The elevated LibreHardwareMonitor sensor sidecar is not reporting CPU temperature (<code className="text-amber-300">Tctl/Tdie</code>).
          Over-temperature auto-abort cannot protect the CPU during this test. If you continue, TestOC will enforce a hard{" "}
          <strong className="text-white">10-minute safety cap</strong>.
        </p>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={cancelNoThermal}
            className="rounded-md border border-ink-700 bg-ink-800 px-3 py-1.5 text-xs text-slate-200 hover:bg-ink-700"
          >
            Cancel
          </button>
          <button
            type="button"
            data-testid="confirm-no-thermal-btn"
            onClick={() => void confirmNoThermal()}
            className="rounded-md border border-amber-400/70 bg-amber-500 px-3.5 py-1.5 text-xs font-bold text-ink-950 hover:bg-amber-400"
          >
            Continue (10 min cap)
          </button>
        </div>
      </div>
    </div>
  );
}
