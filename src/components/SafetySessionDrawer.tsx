import { useEffect, useState } from "react";
import { save } from "@tauri-apps/plugin-dialog";
import { useTelemetryStore } from "../store/telemetry";
import { fmt } from "../lib/format";
import type { SafetyLimits } from "../lib/types";

export function SafetySessionDrawer() {
  const limits = useTelemetryStore((s) => s.limits);
  const session = useTelemetryStore((s) => s.session);
  const lastExport = useTelemetryStore((s) => s.lastExport);
  const updateLimits = useTelemetryStore((s) => s.updateLimits);
  const startManualSession = useTelemetryStore((s) => s.startManualSession);
  const stopManualSession = useTelemetryStore((s) => s.stopManualSession);
  const exportCsvTo = useTelemetryStore((s) => s.exportCsvTo);

  const [draft, setDraft] = useState<SafetyLimits>(limits);
  const [limitMsg, setLimitMsg] = useState<string | null>(null);
  const [safetyOpen, setSafetyOpen] = useState(true);

  useEffect(() => {
    setDraft(limits);
  }, [limits]);

  const applyLimits = async () => {
    const err = await updateLimits(draft);
    setLimitMsg(err ?? "Saved");
    if (!err) setTimeout(() => setLimitMsg(null), 2000);
  };

  const onExport = async () => {
    const defaultName = `${session?.label ?? "testoc-session"}.csv`;
    const chosen = await save({
      title: "Export TestOC Session CSV",
      defaultPath: defaultName,
      filters: [{ name: "CSV Report", extensions: ["csv"] }],
    });
    if (chosen) {
      await exportCsvTo(chosen);
    }
  };

  return (
    <div className="space-y-3">
      {/* Session & CSV Export */}
      <section data-testid="session-panel" className="rounded-lg border border-ink-700/80 bg-ink-900 p-3.5">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-xs font-bold tracking-wider text-slate-200 uppercase">Session & Export</h2>
          {session?.recording && (
            <span className="flex items-center gap-1 rounded bg-red-500/15 px-1.5 py-0.5 font-mono text-[10px] text-red-300">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-red-400" />
              REC
            </span>
          )}
        </div>

        <div className="mb-2.5 flex items-center justify-between rounded border border-ink-800 bg-ink-950/70 px-2.5 py-1.5 font-mono text-[11px]">
          <span className="truncate text-slate-300">{session ? session.label : "No session yet"}</span>
          <span className="tabular text-slate-400">{session ? `${session.samples} rows` : "0 rows"}</span>
        </div>

        <div className="flex gap-2">
          {!session?.recording ? (
            <button
              type="button"
              data-testid="session-start-btn"
              onClick={() => void startManualSession()}
              className="flex-1 rounded border border-ink-700 bg-ink-800 px-2.5 py-1.5 text-xs font-medium text-slate-200 hover:bg-ink-700"
            >
              Record
            </button>
          ) : (
            <button
              type="button"
              data-testid="session-stop-btn"
              onClick={() => void stopManualSession()}
              className="flex-1 rounded border border-amber-500/50 bg-amber-500/10 px-2.5 py-1.5 text-xs font-medium text-amber-200 hover:bg-amber-500/20"
            >
              Stop Rec
            </button>
          )}
          <button
            type="button"
            data-testid="export-csv-btn"
            disabled={!session || session.samples === 0}
            onClick={() => void onExport()}
            className="flex-1 rounded border border-sky-500/50 bg-sky-500/15 px-2.5 py-1.5 text-xs font-semibold text-sky-200 hover:bg-sky-500/25 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Export CSV
          </button>
        </div>

        {lastExport && (
          <div data-testid="export-result" className="mt-2 rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-1.5 text-[10px] text-emerald-200">
            Exported {lastExport.rows} rows ({fmt.bytes(lastExport.bytes)}) + summary JSON
          </div>
        )}
      </section>

      {/* Safety Limits */}
      <section data-testid="safety-panel" className="rounded-lg border border-ink-700/80 bg-ink-900 p-3.5">
        <button
          type="button"
          onClick={() => setSafetyOpen((v) => !v)}
          className="flex w-full items-center justify-between text-xs font-bold tracking-wider text-slate-200 uppercase"
        >
          <span>Safety Limits</span>
          <span className="text-slate-500">{safetyOpen ? "−" : "+"}</span>
        </button>

        {safetyOpen && (
          <div className="mt-3 space-y-2.5 text-[11px]">
            <label className="flex cursor-pointer items-center justify-between gap-2 rounded border border-ink-800 bg-ink-950/70 px-2 py-1.5 text-[11px] text-slate-200">
              <span>Auto-abort on temp limit</span>
              <input
                type="checkbox"
                data-testid="thermal-abort-toggle"
                checked={draft.thermalAbortEnabled}
                onChange={(e) => setDraft({ ...draft, thermalAbortEnabled: e.target.checked })}
                className="h-3.5 w-3.5 accent-accent"
              />
            </label>
            <div className="grid grid-cols-2 gap-2">
              <label className="block">
                <span className="text-[10px] text-slate-400">CPU Warn (°C)</span>
                <input
                  data-testid="cpu-warn-input"
                  type="number"
                  min={40}
                  max={115}
                  value={draft.cpuWarnC}
                  onChange={(e) => setDraft({ ...draft, cpuWarnC: Number(e.target.value) })}
                  className="tabular mt-0.5 w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-slate-200"
                />
              </label>
              <label className="block">
                <span className="text-[10px] text-red-300">CPU Abort (°C)</span>
                <input
                  data-testid="cpu-abort-input"
                  type="number"
                  min={60}
                  max={115}
                  value={draft.cpuAbortC}
                  onChange={(e) => setDraft({ ...draft, cpuAbortC: Number(e.target.value) })}
                  className="tabular mt-0.5 w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-slate-200"
                />
              </label>
              <label className="block">
                <span className="text-[10px] text-slate-400">GPU Warn (°C)</span>
                <input
                  data-testid="gpu-warn-input"
                  type="number"
                  min={40}
                  max={110}
                  value={draft.gpuWarnC}
                  onChange={(e) => setDraft({ ...draft, gpuWarnC: Number(e.target.value) })}
                  className="tabular mt-0.5 w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-slate-200"
                />
              </label>
              <label className="block">
                <span className="text-[10px] text-red-300">GPU Abort (°C)</span>
                <input
                  data-testid="gpu-abort-input"
                  type="number"
                  min={60}
                  max={110}
                  value={draft.gpuAbortC}
                  onChange={(e) => setDraft({ ...draft, gpuAbortC: Number(e.target.value) })}
                  className="tabular mt-0.5 w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-slate-200"
                />
              </label>
            </div>
            <label className="block">
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>Max RAM Cap (100% = Uncapped)</span>
                <span className="tabular font-mono text-slate-200">
                  {draft.memMaxPct >= 100 ? "100% (Max)" : `${draft.memMaxPct}%`}
                </span>
              </div>
              <input
                data-testid="mem-cap-slider"
                type="range"
                min={10}
                max={100}
                value={draft.memMaxPct}
                onChange={(e) => setDraft({ ...draft, memMaxPct: Number(e.target.value) })}
                className="w-full accent-accent"
              />
            </label>
            <div className="flex items-center justify-between gap-2 pt-1">
              <span className={`text-[10px] ${limitMsg === "Saved" ? "text-emerald-400" : "text-amber-300"}`}>
                {limitMsg ??
                  (draft.thermalAbortEnabled
                    ? "Enforced by safety watchdog"
                    : "Temp alerts only (PROCHOT active)")}
              </span>
              <button
                type="button"
                data-testid="save-limits-btn"
                onClick={() => void applyLimits()}
                className="rounded border border-ink-600 bg-ink-800 px-2.5 py-1 text-[11px] font-medium text-slate-200 hover:bg-ink-700"
              >
                Apply
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
