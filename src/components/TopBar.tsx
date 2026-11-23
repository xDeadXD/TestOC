import { useTelemetryStore } from "../store/telemetry";
import type { SidecarState } from "../lib/types";

const SIDECAR_STYLE: Record<SidecarState, { label: string; cls: string; needsRetry: boolean }> = {
  ok: { label: "LHM Sensors: OK", cls: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300", needsRetry: false },
  initializing: { label: "Sensors: Starting…", cls: "border-sky-500/40 bg-sky-500/10 text-sky-300", needsRetry: false },
  launching: { label: "Sensors: UAC Prompt…", cls: "border-sky-500/40 bg-sky-500/10 text-sky-300", needsRetry: false },
  notStarted: { label: "Sensors: Off", cls: "border-slate-600 bg-slate-800 text-slate-300", needsRetry: true },
  declined: { label: "Sensors: UAC Declined", cls: "border-amber-500/50 bg-amber-500/15 text-amber-300", needsRetry: true },
  stale: { label: "Sensors: Stale", cls: "border-red-500/60 bg-red-500/15 text-red-300", needsRetry: true },
  pawnIoMissing: { label: "PawnIO Missing", cls: "border-amber-500/60 bg-amber-500/15 text-amber-300", needsRetry: true },
  incompatible: { label: "Sensors: Version Mismatch", cls: "border-red-500/60 bg-red-500/15 text-red-300", needsRetry: true },
  error: { label: "Sensors: Error", cls: "border-red-500/60 bg-red-500/15 text-red-300", needsRetry: true },
};

export function TopBar() {
  const info = useTelemetryStore((s) => s.info);
  const sidecar = useTelemetryStore((s) => s.sidecar);
  const stress = useTelemetryStore((s) => s.stress);
  const pollMs = useTelemetryStore((s) => s.pollMs);
  const sampleCount = useTelemetryStore((s) => s.sampleCount);
  const setPollMs = useTelemetryStore((s) => s.setPollMs);
  const retrySidecar = useTelemetryStore((s) => s.retrySidecar);
  const emergencyAbort = useTelemetryStore((s) => s.emergencyAbort);

  const sc = SIDECAR_STYLE[sidecar.state];
  const active = stress?.phase === "running" || stress?.phase === "starting" || stress?.phase === "stopping";

  return (
    <header className="sticky top-0 z-30 flex flex-wrap items-center justify-between gap-3 border-b border-ink-700 bg-ink-900/95 px-4 py-2.5 backdrop-blur">
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md border border-accent/60 bg-accent/15 text-xs font-bold text-accent">
            TO
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold tracking-wide text-white">TESTOC</span>
              <span className="rounded bg-ink-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-400">
                v{info?.appVersion ?? "0.1.1"}
              </span>
              <span className="text-[10px] text-slate-500">by xDead</span>
            </div>
            <div className="truncate text-[11px] text-slate-400">
              {info ? (
                <>
                  <span className="text-slate-200">{info.cpuBrand}</span>
                  <span className="mx-1.5 text-slate-600">·</span>
                  <span>
                    {info.physicalCores}C/{info.logicalCores}T
                  </span>
                  {info.gpus[0] && (
                    <>
                      <span className="mx-1.5 text-slate-600">·</span>
                      <span className="text-slate-200">{info.gpus[0].name}</span>
                    </>
                  )}
                </>
              ) : (
                "Initializing hardware telemetry…"
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2.5">
        <span
          data-testid="sample-counter"
          className="tabular rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-[11px] text-slate-400"
          title="Live samples received"
        >
          #{sampleCount}
        </span>

        <label className="flex items-center gap-1.5 text-[11px] text-slate-400">
          <span>Poll</span>
          <select
            aria-label="Poll interval"
            value={pollMs}
            onChange={(e) => void setPollMs(Number(e.target.value))}
            className="rounded border border-ink-700 bg-ink-950 px-1.5 py-1 font-mono text-[11px] text-slate-200 focus:border-accent focus:outline-none"
          >
            <option value={250}>250 ms</option>
            <option value={500}>500 ms</option>
            <option value={1000}>1000 ms</option>
            <option value={2000}>2000 ms</option>
          </select>
        </label>

        <div className="flex items-center gap-1.5">
          <span
            data-testid="sidecar-badge"
            title={sidecar.message ?? undefined}
            className={`rounded border px-2 py-1 text-[11px] font-medium ${sc.cls}`}
          >
            {sc.label}
          </span>
          {sc.needsRetry && (
            <button
              type="button"
              data-testid="retry-sidecar-btn"
              onClick={() => void retrySidecar()}
              className="rounded border border-sky-500/50 bg-sky-500/15 px-2 py-1 text-[11px] font-medium text-sky-200 hover:bg-sky-500/25"
            >
              Enable Sensors
            </button>
          )}
        </div>

        <button
          type="button"
          data-testid="abort-btn"
          onClick={() => void emergencyAbort()}
          className={`flex items-center gap-1.5 rounded-md border px-3.5 py-1.5 text-xs font-bold tracking-wider uppercase transition ${
            active
              ? "animate-pulse border-red-400 bg-red-600 text-white shadow-lg shadow-red-900/50 hover:bg-red-500"
              : "border-red-500/50 bg-red-950/60 text-red-200 hover:bg-red-800/70 hover:text-white"
          }`}
          title="Instant safety abort (also bound to Esc)"
        >
          <span>ABORT</span>
          <kbd className="rounded bg-black/30 px-1 py-0.5 font-mono text-[10px] font-normal">Esc</kbd>
        </button>
      </div>
    </header>
  );
}
