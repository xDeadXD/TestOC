import { useState } from "react";
import { useTelemetryStore } from "../store/telemetry";

const ABORT_LABELS: Record<string, string> = {
  user: "User Abort",
  cpuTemp: "CPU Temp Limit",
  gpuTemp: "GPU Temp Limit",
  memoryPressure: "Memory Pressure",
  sensorStall: "Sensor Stall",
  workerError: "Hardware Error Limit",
  noThermalTimeCap: "10-Min No-Thermal Cap",
  appExit: "App Exit",
};

export function AlertBanner() {
  const banner = useTelemetryStore((s) => s.bannerAlert);
  const alerts = useTelemetryStore((s) => s.alerts);
  const workerErrors = useTelemetryStore((s) => s.workerErrors);
  const actionError = useTelemetryStore((s) => s.actionError);
  const stress = useTelemetryStore((s) => s.stress);
  const dismissBanner = useTelemetryStore((s) => s.dismissBanner);
  const clearActionError = useTelemetryStore((s) => s.clearActionError);
  const [logOpen, setLogOpen] = useState(false);

  const abortedLabel =
    stress?.phase === "aborted" && stress.abortReason ? `Aborted (${ABORT_LABELS[stress.abortReason] ?? stress.abortReason})` : null;

  return (
    <div className="space-y-2">
      {actionError && (
        <div
          data-testid="action-error"
          className="flex items-center justify-between gap-3 rounded-lg border border-red-500/60 bg-red-950/60 px-3 py-2 text-xs text-red-200"
        >
          <span>{actionError}</span>
          <button type="button" onClick={clearActionError} className="text-red-300 hover:text-white">
            Dismiss
          </button>
        </div>
      )}

      {banner && (
        <div
          data-testid="alert-banner"
          data-level={banner.level}
          className={`flex flex-wrap items-center justify-between gap-3 rounded-lg border px-3.5 py-2.5 text-xs ${
            banner.level === "critical"
              ? "border-red-500 bg-red-950/80 text-red-100"
              : "border-amber-500/70 bg-amber-950/60 text-amber-100"
          }`}
        >
          <div className="flex items-center gap-2.5">
            <span
              className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
                banner.level === "critical" ? "bg-red-600 text-white" : "bg-amber-500 text-ink-950"
              }`}
            >
              {banner.level}
            </span>
            <span className="font-medium">{banner.message}</span>
            {abortedLabel && (
              <span data-testid="aborted-badge" className="rounded bg-black/40 px-2 py-0.5 font-mono text-[11px] text-red-200">
                {abortedLabel}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {(alerts.length > 0 || workerErrors.length > 0) && (
              <button
                type="button"
                onClick={() => setLogOpen((v) => !v)}
                className="rounded border border-white/20 px-2 py-0.5 text-[11px] hover:bg-white/10"
              >
                {logOpen ? "Hide" : "History"} ({alerts.length + workerErrors.length})
              </button>
            )}
            <button type="button" onClick={dismissBanner} className="text-white/70 hover:text-white">
              ✕
            </button>
          </div>
        </div>
      )}

      {!banner && (alerts.length > 0 || workerErrors.length > 0) && (
        <div className="flex items-center justify-end">
          <button
            type="button"
            onClick={() => setLogOpen((v) => !v)}
            className="text-[11px] text-slate-400 hover:text-slate-200"
          >
            {logOpen ? "Hide Event Log" : `Event Log (${alerts.length} alerts, ${workerErrors.length} errors)`}
          </button>
        </div>
      )}

      {logOpen && (
        <div className="max-h-44 overflow-y-auto rounded-lg border border-ink-700 bg-ink-900 p-2.5 font-mono text-[11px]">
          {alerts.map((a, idx) => (
            <div key={`a-${idx}`} className="flex gap-2 border-b border-ink-800 py-1 last:border-0">
              <span className="text-slate-500">{new Date(a.timestampMs).toTimeString().slice(0, 8)}</span>
              <span className={a.level === "critical" ? "font-bold text-red-400" : "text-amber-400"}>[{a.level}]</span>
              <span className="text-slate-200">{a.message}</span>
            </div>
          ))}
          {workerErrors.map((e, idx) => (
            <div key={`e-${idx}`} className="flex gap-2 border-b border-ink-800 py-1 last:border-0">
              <span className="text-slate-500">{new Date(e.timestampMs).toTimeString().slice(0, 8)}</span>
              <span className="font-bold text-red-400">[{e.kind}]</span>
              <span className="text-slate-200">{e.detail}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
