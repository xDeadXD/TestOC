export type Level = "na" | "ok" | "warn" | "crit";

/** Display classification only: aborts are decided by the Rust watchdog. */
export function classifyTemp(v: number | null | undefined, warn: number, abort: number): Level {
  if (v == null || !Number.isFinite(v)) return "na";
  if (v >= abort) return "crit";
  if (v >= warn) return "warn";
  return "ok";
}

export function classifyPct(v: number | null | undefined, warn: number, crit: number): Level {
  return classifyTemp(v, warn, crit);
}

export const levelText: Record<Level, string> = {
  na: "text-slate-500",
  ok: "text-emerald-400",
  warn: "text-amber-400",
  crit: "text-red-400",
};

export const levelBorder: Record<Level, string> = {
  na: "border-ink-700",
  ok: "border-ink-700",
  warn: "border-amber-500/60",
  crit: "border-red-500/80",
};

/** Heat color for a 0-100 load value (core tiles). */
export function loadColor(pct: number): string {
  const p = Math.max(0, Math.min(100, pct)) / 100;
  // slate -> amber -> red
  const hue = 210 - 210 * p;
  const sat = 20 + 70 * p;
  const light = 22 + 20 * p;
  return `hsl(${hue.toFixed(0)} ${sat.toFixed(0)}% ${light.toFixed(0)}%)`;
}
