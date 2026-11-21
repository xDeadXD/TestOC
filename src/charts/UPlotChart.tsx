import { useEffect, useRef } from "react";
import uPlot from "uplot";
import { historyColumns, subscribeChartTicks, useTelemetryStore } from "../store/telemetry";

export interface SeriesDef {
  key: string;
  label: string;
  stroke: string;
  width?: number;
  dash?: number[];
  unit?: string;
  decimals?: number;
  scale?: string;
}

export interface ThresholdLine {
  value: number;
  stroke: string;
  label?: string;
  scale?: string;
}

interface UPlotChartProps {
  title: string;
  series: SeriesDef[];
  yRange?: [number, number];
  yUnit?: string;
  y2Range?: [number, number];
  y2Unit?: string;
  thresholds?: ThresholdLine[];
  height?: number;
  testId?: string;
  headerRight?: React.ReactNode;
}

function buildAlignedData(series: SeriesDef[], windowSamples: number): uPlot.AlignedData {
  const rawT = historyColumns.window("t", windowSamples);
  const t: number[] = rawT.map((v, i) => (typeof v === "number" ? v : i));
  const ys: (number | null)[][] = series.map((s) => historyColumns.window(s.key, windowSamples));
  return [t, ...ys] as uPlot.AlignedData;
}

export function UPlotChart({
  title,
  series,
  yRange,
  yUnit = "",
  y2Range,
  y2Unit,
  thresholds = [],
  height = 185,
  testId,
  headerRight,
}: UPlotChartProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const thresholdsRef = useRef<ThresholdLine[]>(thresholds);
  thresholdsRef.current = thresholds;

  // Serialize series keys/visibility so uPlot only rebuilds when series config changes.
  const seriesSig = series.map((s) => `${s.key}:${s.stroke}:${s.scale ?? "y"}`).join("|");
  const rangeSig = `${yRange?.[0] ?? ""}:${yRange?.[1] ?? ""}:${y2Range?.[0] ?? ""}:${y2Range?.[1] ?? ""}`;

  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;

    const getWindowSamples = () => {
      const { windowMin, pollMs } = useTelemetryStore.getState();
      return Math.max(10, Math.round((windowMin * 60 * 1000) / Math.max(250, pollMs)));
    };

    const hasY2 = series.some((s) => s.scale === "y2");
    const axes: uPlot.Axis[] = [
      {
        stroke: "#64748b",
        font: "10px ui-monospace, monospace",
        grid: { stroke: "rgba(51,65,85,0.35)", width: 1 },
        ticks: { stroke: "rgba(51,65,85,0.5)", width: 1 },
        values: (_, ticks) =>
          ticks.map((v) => {
            if (!Number.isFinite(v) || v < 1_000_000) return "";
            const d = new Date(v * 1000);
            return d.toTimeString().slice(0, 8);
          }),
      },
      {
        scale: "y",
        stroke: "#94a3b8",
        font: "10px ui-monospace, monospace",
        size: 46,
        grid: { stroke: "rgba(51,65,85,0.35)", width: 1 },
        ticks: { stroke: "rgba(51,65,85,0.5)", width: 1 },
        values: (_, ticks) => ticks.map((v) => (v == null ? "" : `${Math.round(v)}${yUnit}`)),
      },
    ];
    if (hasY2) {
      axes.push({
        scale: "y2",
        side: 1,
        stroke: "#60a5fa",
        font: "10px ui-monospace, monospace",
        size: 44,
        grid: { show: false },
        ticks: { stroke: "rgba(96,165,250,0.4)", width: 1 },
        values: (_, ticks) => ticks.map((v) => (v == null ? "" : `${Math.round(v)}${y2Unit ?? ""}`)),
      });
    }

    const scales: uPlot.Scales = {
      x: {
        time: false,
        range: (_u, dataMin, dataMax) => {
          const spanSec = useTelemetryStore.getState().windowMin * 60;
          if (!Number.isFinite(dataMax) || dataMax <= 0) return [0, spanSec];
          const min = Math.min(dataMin, dataMax - spanSec);
          return [dataMax - spanSec < min ? min : dataMax - spanSec, dataMax];
        },
      },
      y: yRange ? { auto: false, range: yRange } : { auto: true },
    };
    if (hasY2) {
      scales.y2 = y2Range ? { auto: false, range: y2Range } : { auto: true };
    }

    const opts: uPlot.Options = {
      width: Math.max(240, el.clientWidth || 480),
      height,
      pxAlign: 0,
      cursor: { drag: { x: false, y: false }, points: { size: 5 } },
      legend: { show: true, live: true },
      scales,
      axes,
      series: [
        {
          label: "Time",
          value: (_, v) => {
            const ts = v ?? historyColumns.latest("t");
            return typeof ts === "number" && ts > 1_000_000 ? new Date(ts * 1000).toTimeString().slice(0, 8) : "—";
          },
        },
        ...series.map<uPlot.Series>((s) => ({
          label: s.label,
          stroke: s.stroke,
          width: s.width ?? 1.75,
          dash: s.dash,
          scale: s.scale ?? "y",
          points: { show: false },
          spanGaps: false,
          value: (_, v) => {
            const val = v ?? historyColumns.latest(s.key);
            return typeof val === "number" ? `${val.toFixed(s.decimals ?? 1)}${s.unit ?? yUnit}` : "—";
          },
        })),
      ],
      hooks: {
        draw: [
          (u) => {
            const ctx = u.ctx;
            const { left, width: plotW } = u.bbox;
            if (plotW <= 0) return;
            ctx.save();
            ctx.lineWidth = 1 * devicePixelRatio;
            ctx.setLineDash([5 * devicePixelRatio, 4 * devicePixelRatio]);
            ctx.font = `${10 * devicePixelRatio}px ui-monospace, monospace`;
            ctx.textAlign = "right";
            ctx.textBaseline = "bottom";
            for (const th of thresholdsRef.current) {
              const py = u.valToPos(th.value, th.scale ?? "y", true);
              if (!Number.isFinite(py) || py < u.bbox.top || py > u.bbox.top + u.bbox.height) continue;
              ctx.strokeStyle = th.stroke;
              ctx.beginPath();
              ctx.moveTo(left, py);
              ctx.lineTo(left + plotW, py);
              ctx.stroke();
              if (th.label) {
                ctx.fillStyle = th.stroke;
                ctx.fillText(th.label, left + plotW - 6 * devicePixelRatio, py - 3 * devicePixelRatio);
              }
            }
            ctx.restore();
          },
        ],
      },
    };

    const u = new uPlot(opts, buildAlignedData(series, getWindowSamples()), el);
    // Push incoming ring-buffer columns directly to uPlot so React never re-renders on telemetry ticks.
    const unsub = subscribeChartTicks(() => {
      u.setData(buildAlignedData(series, getWindowSamples()));
    });

    const ro = new ResizeObserver((entries) => {
      const w = Math.floor(entries[0]?.contentRect.width ?? 0);
      if (w >= 180) u.setSize({ width: w, height });
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      unsub();
      u.destroy();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seriesSig, rangeSig, height, yUnit, y2Unit]);

  return (
    <div
      data-testid={testId}
      className="flex flex-col rounded-lg border border-ink-700/80 bg-ink-900/90 p-3 shadow-sm"
    >
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <h3 className="text-xs font-semibold tracking-wider text-slate-300 uppercase">{title}</h3>
        {headerRight}
      </div>
      <div ref={hostRef} className="w-full min-w-0 overflow-hidden" />
    </div>
  );
}
