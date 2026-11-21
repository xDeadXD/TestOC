import { useTelemetryStore } from "../store/telemetry";
import { loadColor } from "../lib/thresholds";
import type { CoreMetrics } from "../lib/types";

const EMPTY_CORES: CoreMetrics[] = [];
const EMPTY_ERRORS: number[] = [];

export function CoreGrid() {
  const cores = useTelemetryStore((s) => s.latest?.cpu.cores ?? EMPTY_CORES);
  const coreErrors = useTelemetryStore((s) => s.stress?.coreErrors ?? EMPTY_ERRORS);
  const source = cores[0]?.clockSource;

  return (
    <section data-testid="core-grid" className="rounded-lg border border-ink-700/80 bg-ink-900 p-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold tracking-wider text-slate-300 uppercase">
          Logical Cores ({cores.length})
        </h3>
        {source && (
          <span className="rounded border border-ink-700 bg-ink-950 px-1.5 py-0.5 font-mono text-[10px] text-slate-400 uppercase">
            Clock src: {source}
          </span>
        )}
      </div>
      <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
        {cores.map((c) => {
          const errs = coreErrors[c.id] ?? 0;
          const pct = Math.max(0, Math.min(100, c.loadPct));
          return (
            <div
              key={c.id}
              data-testid={`core-tile-${c.id}`}
              className={`relative overflow-hidden rounded-md border p-2 ${
                errs > 0 ? "border-red-500/80 bg-red-950/30" : "border-ink-800 bg-ink-950/85"
              }`}
            >
              <div
                className="pointer-events-none absolute inset-y-0 left-0 opacity-25 transition-[width] duration-300"
                style={{ width: `${pct}%`, backgroundColor: loadColor(pct) }}
              />
              <div className="relative flex items-center justify-between font-mono text-[10px]">
                <span className="text-slate-400">C{c.id}</span>
                <span className="tabular font-semibold text-white">{pct.toFixed(0)}%</span>
              </div>
              <div className="relative mt-1 flex items-center justify-between font-mono text-[10px]">
                <span className="tabular text-sky-300">{c.effectiveMhz != null ? `${Math.round(c.effectiveMhz)}` : "—"}</span>
                <span className="tabular text-slate-500">{c.vid != null ? `${c.vid.toFixed(2)}V` : ""}</span>
              </div>
              {errs > 0 && <div className="relative mt-0.5 font-mono text-[9px] font-bold text-red-400">ERR {errs}</div>}
            </div>
          );
        })}
      </div>
    </section>
  );
}
