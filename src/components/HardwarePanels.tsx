import { useTelemetryStore } from "../store/telemetry";
import { fmt } from "../lib/format";

export function HardwarePanels() {
  const latest = useTelemetryStore((s) => s.latest);
  const info = useTelemetryStore((s) => s.info);
  const gpu = latest?.gpus[0];
  const board = latest?.board;

  const vramPct =
    gpu?.vramUsedBytes != null && gpu?.vramTotalBytes
      ? Math.min(100, (gpu.vramUsedBytes / gpu.vramTotalBytes) * 100)
      : null;

  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
      {/* GPU Detail Panel */}
      <section data-testid="gpu-panel" className="rounded-lg border border-ink-700/80 bg-ink-900 p-3">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 className="truncate text-xs font-semibold tracking-wider text-slate-300 uppercase">
            GPU · {gpu?.name ?? info?.gpuStressAdapter ?? "Not detected"}
          </h3>
          {gpu?.tempSlowdownC != null && (
            <span className="font-mono text-[10px] text-slate-500">Slowdown {gpu.tempSlowdownC}°C</span>
          )}
        </div>
        {gpu ? (
          <div className="space-y-2.5 text-xs">
            <div className="grid grid-cols-3 gap-2 font-mono text-[11px]">
              <div className="rounded border border-ink-800 bg-ink-950/70 p-2">
                <div className="text-[9px] text-slate-500 uppercase">Core Clock</div>
                <div className="tabular text-slate-100">{fmt.num(gpu.coreMhz, 0, " MHz")}</div>
              </div>
              <div className="rounded border border-ink-800 bg-ink-950/70 p-2">
                <div className="text-[9px] text-slate-500 uppercase">Mem Clock</div>
                <div className="tabular text-slate-100">{fmt.num(gpu.memMhz, 0, " MHz")}</div>
              </div>
              <div className="rounded border border-ink-800 bg-ink-950/70 p-2">
                <div className="text-[9px] text-slate-500 uppercase">Fan / Util</div>
                <div className="tabular text-slate-100">
                  {fmt.num(gpu.fanPct, 0, "%")} / {fmt.num(gpu.utilPct, 0, "%")}
                </div>
              </div>
            </div>

            <div className="rounded border border-ink-800 bg-ink-950/70 p-2 font-mono text-[11px]">
              <div className="mb-1 flex justify-between">
                <span className="text-[10px] text-slate-400">
                  VRAM: {fmt.bytes(gpu.vramUsedBytes)} / {fmt.bytes(gpu.vramTotalBytes)}
                </span>
                <span className="tabular text-[10px] text-slate-300">
                  Power: {fmt.num(gpu.powerW, 1, " W")} / {fmt.num(gpu.powerLimitW, 0, " W")}
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-800">
                <div className="h-full bg-violet-500 transition-[width] duration-300" style={{ width: `${vramPct ?? 0}%` }} />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
              <span className="text-[10px] text-slate-500 uppercase">Throttle:</span>
              {gpu.throttleReasons.length > 0 ? (
                gpu.throttleReasons.map((r) => (
                  <span
                    key={r}
                    className={`rounded border px-1.5 py-0.5 font-mono text-[10px] ${
                      r === "Power Cap" || r === "Sync Boost" || r === "App Clocks"
                        ? "border-sky-500/40 bg-sky-500/10 text-sky-300"
                        : "border-amber-500/60 bg-amber-500/20 text-amber-200"
                    }`}
                  >
                    {r}
                  </span>
                ))
              ) : (
                <span className="font-mono text-[10px] text-emerald-400">None</span>
              )}
            </div>
          </div>
        ) : (
          <div className="py-6 text-center text-xs text-slate-500">
            GPU telemetry unavailable (install GPU driver or enable LHM sensors).
          </div>
        )}
      </section>

      {/* Motherboard / SuperIO Panel */}
      <section data-testid="board-panel" className="rounded-lg border border-ink-700/80 bg-ink-900 p-3">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-xs font-semibold tracking-wider text-slate-300 uppercase">Motherboard & Fans (SuperIO)</h3>
          <span className="font-mono text-[10px] text-slate-500">
            {(board?.temps.length ?? 0) + (board?.fans.length ?? 0)} sensors
          </span>
        </div>
        {board && (board.temps.length > 0 || board.fans.length > 0) ? (
          <div className="grid grid-cols-2 gap-3 font-mono text-[11px]">
            <div>
              <div className="mb-1 text-[9px] text-slate-500 uppercase">Temperatures</div>
              <div className="space-y-1">
                {board.temps.map((t, i) => (
                  <div key={`${t.name}-${i}`} className="flex justify-between rounded bg-ink-950/70 px-2 py-1">
                    <span className="truncate text-slate-300">{t.name}</span>
                    <span className="tabular text-amber-300">{t.value.toFixed(1)} °C</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <div className="mb-1 text-[9px] text-slate-500 uppercase">Fans</div>
              <div className="space-y-1">
                {board.fans.map((f, i) => (
                  <div key={`${f.name}-${i}`} className="flex justify-between rounded bg-ink-950/70 px-2 py-1">
                    <span className="truncate text-slate-300">{f.name}</span>
                    <span className="tabular text-sky-300">{Math.round(f.value)} RPM</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="py-6 text-center text-xs text-slate-500">
            SuperIO temperatures and fan speeds appear when the elevated LHM sensor sidecar is active.
          </div>
        )}
      </section>
    </div>
  );
}
