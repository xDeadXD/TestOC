import { useMemo } from "react";
import { UPlotChart, type SeriesDef, type ThresholdLine } from "../charts/UPlotChart";
import { useTelemetryStore, type WindowMinutes } from "../store/telemetry";

const CORE_PALETTE = [
  "#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb7185", "#fb923c", "#facc15", "#a3e635",
  "#4ade80", "#2dd4bf", "#22d3ee", "#60a5fa", "#a78bfa", "#e879f9", "#f87171", "#fdba74",
];

const WINDOWS: WindowMinutes[] = [1, 5, 15];

export function ChartsGrid() {
  const info = useTelemetryStore((s) => s.info);
  const limits = useTelemetryStore((s) => s.limits);
  const windowMin = useTelemetryStore((s) => s.windowMin);
  const selectedCores = useTelemetryStore((s) => s.selectedCores);
  const setWindowMin = useTelemetryStore((s) => s.setWindowMin);
  const toggleCore = useTelemetryStore((s) => s.toggleCore);
  const setAllCores = useTelemetryStore((s) => s.setAllCores);

  const logical = info?.logicalCores ?? 16;

  const cpuLoadSeries = useMemo<SeriesDef[]>(() => {
    const list: SeriesDef[] = [{ key: "cpuLoad", label: "Total", stroke: "#ffffff", width: 2.2, unit: "%" }];
    for (let i = 0; i < logical; i++) {
      if (selectedCores[i]) {
        list.push({ key: `cLoad${i}`, label: `T${i}`, stroke: CORE_PALETTE[i % CORE_PALETTE.length], width: 1.15, unit: "%" });
      }
    }
    return list;
  }, [logical, selectedCores]);

  const clockSeries = useMemo<SeriesDef[]>(() => {
    const list: SeriesDef[] = [{ key: "cpuAvgMhz", label: "Avg", stroke: "#38bdf8", width: 2.2, unit: " MHz", decimals: 0 }];
    for (let i = 0; i < logical; i++) {
      if (selectedCores[i]) {
        list.push({
          key: `cMhz${i}`,
          label: `T${i}`,
          stroke: CORE_PALETTE[i % CORE_PALETTE.length],
          width: 1.1,
          unit: " MHz",
          decimals: 0,
        });
      }
    }
    return list;
  }, [logical, selectedCores]);

  const tempSeries = useMemo<SeriesDef[]>(
    () => [
      { key: "cpuTemp", label: "CPU Tctl", stroke: "#ff5a36", width: 2.2, unit: "°C" },
      { key: "ccdTemp", label: "CPU CCD", stroke: "#fb923c", width: 1.4, dash: [4, 3], unit: "°C" },
      { key: "gpuTemp", label: "GPU", stroke: "#a855f7", width: 2.0, unit: "°C" },
    ],
    []
  );

  const tempThresholds = useMemo<ThresholdLine[]>(
    () => [
      { value: limits.cpuWarnC, stroke: "rgba(251, 191, 36, 0.7)", label: `CPU Warn ${limits.cpuWarnC}°` },
      {
        value: limits.cpuAbortC,
        stroke: limits.thermalAbortEnabled ? "rgba(239, 68, 68, 0.85)" : "rgba(239, 68, 68, 0.35)",
        label: limits.thermalAbortEnabled ? `CPU Abort ${limits.cpuAbortC}°` : `CPU ${limits.cpuAbortC}° (Alert)`,
      },
    ],
    [limits.cpuWarnC, limits.cpuAbortC, limits.thermalAbortEnabled]
  );

  const tempYMax = Math.max(105, Math.ceil((Math.max(limits.cpuAbortC, limits.gpuAbortC) + 5) / 5) * 5);

  const powerMemSeries = useMemo<SeriesDef[]>(
    () => [
      { key: "cpuPower", label: "CPU W", stroke: "#f59e0b", width: 2.0, unit: " W", scale: "y" },
      { key: "gpuPower", label: "GPU W", stroke: "#c084fc", width: 2.0, unit: " W", scale: "y" },
      { key: "memPct", label: "RAM %", stroke: "#38bdf8", width: 1.8, unit: "%", scale: "y2" },
    ],
    []
  );

  const activeCount = selectedCores.filter(Boolean).length;

  return (
    <section className="space-y-2.5">
      {/* Shared chart toolbar: 1 / 5 / 15 min window + per-core series toggles */}
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-ink-700/80 bg-ink-900 px-3 py-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-[10px] font-semibold tracking-wider text-slate-400 uppercase">Per-Core Series ({activeCount}/{logical}):</span>
          <button
            type="button"
            data-testid="cores-all-btn"
            onClick={() => setAllCores(true)}
            className="rounded border border-ink-700 bg-ink-950 px-1.5 py-0.5 text-[10px] text-slate-300 hover:border-ink-600"
          >
            All
          </button>
          <button
            type="button"
            data-testid="cores-none-btn"
            onClick={() => setAllCores(false)}
            className="rounded border border-ink-700 bg-ink-950 px-1.5 py-0.5 text-[10px] text-slate-300 hover:border-ink-600"
          >
            None
          </button>
          <div className="flex flex-wrap gap-1">
            {Array.from({ length: logical }, (_, i) => {
              const on = Boolean(selectedCores[i]);
              const col = CORE_PALETTE[i % CORE_PALETTE.length];
              return (
                <button
                  key={i}
                  type="button"
                  data-testid={`core-toggle-${i}`}
                  aria-pressed={on}
                  onClick={() => toggleCore(i)}
                  style={on ? { borderColor: col, color: col, backgroundColor: `${col}1c` } : undefined}
                  className={`tabular rounded border px-1.5 py-0.5 font-mono text-[10px] transition ${
                    on ? "font-semibold" : "border-ink-800 bg-ink-950 text-slate-500 hover:text-slate-300"
                  }`}
                >
                  {i}
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-1">
          <span className="mr-1 text-[10px] font-semibold text-slate-400 uppercase">Window:</span>
          {WINDOWS.map((m) => (
            <button
              key={m}
              type="button"
              data-testid={`window-${m}m`}
              aria-pressed={windowMin === m}
              onClick={() => setWindowMin(m)}
              className={`rounded border px-2 py-0.5 font-mono text-[11px] ${
                windowMin === m
                  ? "border-accent bg-accent/20 font-semibold text-white"
                  : "border-ink-700 bg-ink-950 text-slate-400 hover:text-slate-200"
              }`}
            >
              {m}m
            </button>
          ))}
        </div>
      </div>

      {/* 2x2 uPlot Canvas Grid */}
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <UPlotChart title="CPU Load" testId="chart-cpu-load" series={cpuLoadSeries} yRange={[0, 100]} yUnit="%" />
        <UPlotChart
          title="Temperatures"
          testId="chart-temps"
          series={tempSeries}
          yRange={[20, tempYMax]}
          yUnit="°"
          thresholds={tempThresholds}
        />
        <UPlotChart title="Effective Clocks" testId="chart-clocks" series={clockSeries} yUnit="" />
        <UPlotChart
          title="Power & Memory"
          testId="chart-power-mem"
          series={powerMemSeries}
          yUnit="W"
          y2Range={[0, 100]}
          y2Unit="%"
        />
      </div>
    </section>
  );
}
