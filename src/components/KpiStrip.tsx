import { useTelemetryStore } from "../store/telemetry";
import { fmt } from "../lib/format";
import { classifyPct, classifyTemp, levelBorder, levelText } from "../lib/thresholds";

const SEVERE_THROTTLES = new Set(["Thermal Slowdown", "HW Slowdown", "Power Brake", "Thermal Limit", "PROCHOT"]);

function throttleBadgeClass(reason: string): string {
  if (SEVERE_THROTTLES.has(reason) || reason.toLowerCase().includes("thermal") || reason.toLowerCase().includes("prochot")) {
    return "border-amber-500/60 bg-amber-500/20 text-amber-200";
  }
  // Informational (e.g. GPU Power Cap, Sync Boost, App Clocks)
  return "border-sky-500/40 bg-sky-500/10 text-sky-300";
}

export function KpiStrip() {
  const latest = useTelemetryStore((s) => s.latest);
  const limits = useTelemetryStore((s) => s.limits);
  const cpu = latest?.cpu;
  const mem = latest?.memory;
  const gpu = latest?.gpus[0];

  const cpuTempLvl = classifyTemp(cpu?.packageTempC, limits.cpuWarnC, limits.cpuAbortC);
  const gpuTempLvl = classifyTemp(gpu?.tempC, limits.gpuWarnC, limits.gpuAbortC);
  const memLvl = classifyPct(mem?.usedPct, limits.memMaxPct - 10, limits.memMaxPct);

  return (
    <div data-testid="kpi-strip" className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      {/* CPU Load & Clocks */}
      <div className="rounded-lg border border-ink-700/80 bg-ink-900 p-3">
        <div className="flex items-center justify-between text-[10px] font-semibold tracking-wider text-slate-400 uppercase">
          <span>CPU Load</span>
          <span className="tabular font-mono text-slate-500">{fmt.num(latest?.pollDurationUs ? latest.pollDurationUs / 1000 : null, 1, " ms")}</span>
        </div>
        <div className="mt-1 flex items-baseline justify-between">
          <span className="tabular font-mono text-2xl font-bold text-white">{fmt.num(cpu?.totalLoadPct, 1, "%")}</span>
          <span className="tabular font-mono text-xs text-sky-300">{fmt.mhz(cpu?.avgEffectiveMhz)}</span>
        </div>
        <div className="mt-1.5 flex items-center justify-between text-[11px] text-slate-400">
          <span>Peak: {fmt.mhz(cpu?.maxEffectiveMhz)}</span>
          <span>Core V: {fmt.num(cpu?.coreVoltageV, 3, " V")}</span>
        </div>
      </div>

      {/* CPU Temp & Power */}
      <div className={`rounded-lg border bg-ink-900 p-3 ${levelBorder[cpuTempLvl]}`}>
        <div className="flex items-center justify-between text-[10px] font-semibold tracking-wider text-slate-400 uppercase">
          <span>CPU Thermal & Power</span>
          <span className="font-mono text-[10px] text-slate-500">
            {limits.thermalAbortEnabled ? `Limit ${limits.cpuAbortC}°C` : "No Abort"}
          </span>
        </div>
        <div className="mt-1 flex items-baseline justify-between">
          <span data-testid="cpu-temp-kpi" className={`tabular font-mono text-2xl font-bold ${levelText[cpuTempLvl]}`}>
            {cpu?.packageTempC != null ? fmt.num(cpu.packageTempC, 1, " °C") : "N/A"}
          </span>
          <span className="tabular font-mono text-xs text-amber-300">{fmt.num(cpu?.packagePowerW, 1, " W")}</span>
        </div>
        <div className="mt-1.5 flex flex-wrap items-center justify-between gap-1 text-[11px] text-slate-400">
          <span>CCD: {fmt.num(cpu?.ccdTempC, 1, " °C")}</span>
          {cpu?.throttleFlags.length ? (
            <div className="flex flex-wrap gap-1">
              {cpu.throttleFlags.map((f) => (
                <span key={f} className={`rounded border px-1.5 py-0.2 text-[9px] font-semibold ${throttleBadgeClass(f)}`}>
                  {f}
                </span>
              ))}
            </div>
          ) : (
            <span className="text-[10px] text-slate-500">{cpu?.packageTempC == null ? "Enable LHM sensors" : "No throttle"}</span>
          )}
        </div>
      </div>

      {/* Memory */}
      <div className={`rounded-lg border bg-ink-900 p-3 ${levelBorder[memLvl]}`}>
        <div className="flex items-center justify-between text-[10px] font-semibold tracking-wider text-slate-400 uppercase">
          <span>System Memory</span>
          <span className="font-mono text-[10px] text-slate-500">
            {limits.memMaxPct >= 100 ? "Uncapped" : `Cap ${limits.memMaxPct}%`}
          </span>
        </div>
        <div className="mt-1 flex items-baseline justify-between">
          <span className={`tabular font-mono text-2xl font-bold ${levelText[memLvl]}`}>{fmt.num(mem?.usedPct, 1, "%")}</span>
          <span className="tabular font-mono text-xs text-slate-300">{fmt.bytes(mem?.usedBytes)}</span>
        </div>
        <div className="mt-1.5 flex items-center justify-between text-[11px] text-slate-400">
          <span>Avail: {fmt.bytes(mem?.availableBytes)}</span>
          <span>Total: {fmt.bytes(mem?.totalBytes)}</span>
        </div>
      </div>

      {/* GPU */}
      <div className={`rounded-lg border bg-ink-900 p-3 ${levelBorder[gpuTempLvl]}`}>
        <div className="flex items-center justify-between text-[10px] font-semibold tracking-wider text-slate-400 uppercase">
          <span>GPU Thermal & Load</span>
          <span className="font-mono text-[10px] text-slate-500">
            {limits.thermalAbortEnabled ? `Limit ${limits.gpuAbortC}°C` : "No Abort"}
          </span>
        </div>
        <div className="mt-1 flex items-baseline justify-between">
          <span data-testid="gpu-temp-kpi" className={`tabular font-mono text-2xl font-bold ${levelText[gpuTempLvl]}`}>
            {fmt.num(gpu?.tempC, 0, " °C")}
          </span>
          <span className="tabular font-mono text-xs text-violet-300">
            {fmt.num(gpu?.utilPct, 0, "%")} · {fmt.num(gpu?.powerW, 0, " W")}
          </span>
        </div>
        <div className="mt-1.5 flex flex-wrap items-center justify-between gap-1 text-[11px] text-slate-400">
          <span>VRAM: {fmt.bytes(gpu?.vramUsedBytes)}</span>
          {gpu?.throttleReasons.length ? (
            <div data-testid="gpu-throttle-pills" className="flex flex-wrap gap-1">
              {gpu.throttleReasons.map((r) => (
                <span key={r} className={`rounded border px-1.5 py-0.2 text-[9px] font-semibold ${throttleBadgeClass(r)}`}>
                  {r}
                </span>
              ))}
            </div>
          ) : (
            <span className="text-[10px] text-slate-500">No throttle</span>
          )}
        </div>
      </div>
    </div>
  );
}
