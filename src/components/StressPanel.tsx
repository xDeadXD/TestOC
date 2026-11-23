import { useEffect, useState } from "react";
import { useTelemetryStore } from "../store/telemetry";
import { buildConfig, defaultForm, loadSavedForm, saveForm, type StressForm } from "../lib/stressForm";
import { fmt } from "../lib/format";
import type {
  CpuWorkload,
  Dataset,
  FftPreset,
  GpuProfile,
  IsaChoice,
  LoadMode,
  MemoryPattern,
  MemoryProfile,
  StressKind,
} from "../lib/types";

const KINDS: Array<{ id: StressKind; label: string; desc: string }> = [
  { id: "cpu", label: "CPU", desc: "FFT + FMA + NTT64" },
  { id: "memory", label: "Memory", desc: "Absolut 32-Step / Custom" },
  { id: "combined", label: "CPU+RAM", desc: "Blend FFTs + Absolut" },
  { id: "gpu", label: "GPU", desc: "3D Pixel-Check / FMA" },
  { id: "vram", label: "VRAM", desc: "8x Absolut GDDR" },
  { id: "power", label: "Power", desc: "CPU + GPU max draw" },
];

const MEM_PROFILES: Array<{ id: MemoryProfile; label: string; sub: string; badge: string }> = [
  { id: "absolut", label: "Absolut (32-Step)", sub: "LCG R/W + Mirror128", badge: "Harsh" },
  { id: "standard", label: "Standard", sub: "Selectable bit-exact", badge: "Custom" },
];

const CYCLE_PRESETS: Array<{ label: string; cycles: number | null }> = [
  { label: "∞", cycles: null },
  { label: "1C", cycles: 1 },
  { label: "3C", cycles: 3 },
  { label: "6C", cycles: 6 },
  { label: "12C", cycles: 12 },
];

const FFT_PRESETS: Array<{ id: FftPreset; label: string; sub: string }> = [
  { id: "smallest", label: "Smallest", sub: "L1 Max Heat" },
  { id: "small", label: "Small", sub: "L2 Heavy" },
  { id: "large", label: "Large", sub: ">L3 RAM/IMC" },
  { id: "blend", label: "Blend", sub: "L1→RAM All" },
];

const GPU_PROFILES: Array<{ id: GpuProfile; label: string; sub: string; heat: string }> = [
  { id: "standard", label: "Standard", sub: "FP32 FMA", heat: "Mod" },
  { id: "heavy", label: "Heavy", sub: "Multi-Unit", heat: "High" },
  { id: "extreme", label: "Extreme", sub: "3D Full-TDP", heat: "Max" },
];

const DURATION_PRESETS: Array<{ label: string; min: number | null }> = [
  { label: "∞", min: null },
  { label: "5m", min: 5 },
  { label: "15m", min: 15 },
  { label: "30m", min: 30 },
  { label: "1h", min: 60 },
];

const PHASE_BADGE: Record<string, string> = {
  idle: "bg-ink-800 text-slate-400 border-ink-700",
  starting: "bg-sky-500/15 text-sky-300 border-sky-500/40",
  running: "bg-emerald-500/15 text-emerald-300 border-emerald-500/50",
  stopping: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  completed: "bg-emerald-500/10 text-emerald-300 border-emerald-500/40",
  stopped: "bg-slate-700/60 text-slate-200 border-slate-600",
  aborted: "bg-red-500/20 text-red-300 border-red-500/60",
  failed: "bg-red-500/20 text-red-300 border-red-500/60",
};

export function StressPanel() {
  const info = useTelemetryStore((s) => s.info);
  const stress = useTelemetryStore((s) => s.stress);
  const workerErrors = useTelemetryStore((s) => s.workerErrors);
  const limits = useTelemetryStore((s) => s.limits);
  const updateLimits = useTelemetryStore((s) => s.updateLimits);
  const startStress = useTelemetryStore((s) => s.startStress);
  const stopStress = useTelemetryStore((s) => s.stopStress);

  const [form, setForm] = useState<StressForm>(loadSavedForm);
  const [validationErr, setValidationErr] = useState<string | null>(null);

  useEffect(() => {
    saveForm(form);
  }, [form]);

  const logical = info?.logicalCores ?? 16;
  const physical = info?.physicalCores ?? Math.max(1, Math.floor(logical / 2));
  const gpuReady = Boolean(info?.gpuStressAdapter);
  const active = stress?.phase === "running" || stress?.phase === "starting" || stress?.phase === "stopping";
  const phase = stress?.phase ?? "idle";

  const onStart = () => {
    const built = buildConfig(
      form,
      info ?? { physicalCores: 8, logicalCores: 16, isa: { avx2: true, fma: true, avx512f: false } }
    );
    if (!built.ok) {
      setValidationErr(built.error);
      return;
    }
    setValidationErr(null);
    void startStress(built.config);
  };

  const patch = <K extends keyof StressForm>(k: K, v: StressForm[K]) => {
    setValidationErr(null);
    setForm((f) => ({ ...f, [k]: v }));
  };

  const onResetDefaults = () => {
    setValidationErr(null);
    setForm((f) => ({ ...defaultForm(), kind: f.kind }));
  };

  const maxThreadsForKind =
    form.kind === "cpu" && (form.physicalCoresOnly || (form.mode === "coreCycler" && form.threads == null))
      ? physical
      : logical;
  const defaultThreadsForKind =
    form.kind === "memory"
      ? form.memProfile === "absolut"
        ? logical
        : Math.min(8, Math.max(1, Math.floor(logical / 2)))
      : maxThreadsForKind;

  return (
    <section data-testid="stress-panel" className="rounded-lg border border-ink-700/80 bg-ink-900 p-3.5">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-xs font-bold tracking-wider text-slate-200 uppercase">Stability Test</h2>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            data-testid="reset-prefs-btn"
            disabled={active}
            onClick={onResetDefaults}
            title="Reset test preferences to defaults"
            className="rounded border border-ink-700 bg-ink-950/70 px-1.5 py-0.5 text-[9px] text-slate-400 uppercase hover:border-ink-600 hover:text-slate-200 disabled:opacity-40"
          >
            Reset
          </button>
          <span
            data-testid="stress-phase-badge"
            className={`rounded border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase ${PHASE_BADGE[phase] ?? PHASE_BADGE.idle}`}
          >
            {phase}
            {phase === "aborted" && stress?.abortReason ? ` · ${stress.abortReason}` : ""}
          </span>
        </div>
      </div>

      {/* Workload selector */}
      <div className="mb-3 grid grid-cols-3 gap-1.5">
        {KINDS.map((k) => {
          const disabled = (k.id === "gpu" || k.id === "vram" || k.id === "power") && !gpuReady;
          const selected = form.kind === k.id;
          return (
            <button
              key={k.id}
              type="button"
              data-testid={`kind-${k.id}`}
              disabled={active || disabled}
              onClick={() => patch("kind", k.id)}
              title={disabled ? "No Vulkan compute adapter detected" : k.desc}
              className={`rounded-md border px-2 py-1.5 text-left transition ${
                selected
                  ? "border-accent bg-accent/15 text-white"
                  : "border-ink-700 bg-ink-950/70 text-slate-300 hover:border-ink-600"
              } disabled:cursor-not-allowed disabled:opacity-45`}
            >
              <div className="text-xs font-semibold">{k.label}</div>
              <div className="truncate text-[9px] text-slate-400">{k.desc}</div>
            </button>
          );
        })}
      </div>

      {/* Mode-specific workload preferences */}
      <div className="space-y-2.5 text-xs">
        {/* CPU mode-specific: Load Mode, Workload, Dataset, Variable / CoreCycler params */}
        {form.kind === "cpu" && (
          <>
            <div className="grid grid-cols-2 gap-2">
              <label className="block">
                <span className="mb-1 block text-[10px] text-slate-400 uppercase">Load Mode</span>
                <select
                  data-testid="cpu-mode"
                  disabled={active}
                  value={form.mode}
                  onChange={(e) => patch("mode", e.target.value as LoadMode)}
                  className="w-full rounded border border-ink-700 bg-ink-950 px-2 py-1.5 text-xs text-slate-200"
                >
                  <option value="steady">Steady (100% All-Core)</option>
                  <option value="variable">Variable (Vdroop Ramp)</option>
                  <option value="coreCycler">Core Cycler (Peak Fmax)</option>
                </select>
              </label>
              <label className="block">
                <span className="mb-1 block text-[10px] text-slate-400 uppercase">Dataset</span>
                <select
                  data-testid="cpu-dataset"
                  disabled={active || form.cpuWorkload === "fftTorture"}
                  value={form.dataset}
                  onChange={(e) => patch("dataset", e.target.value as Dataset)}
                  className="w-full rounded border border-ink-700 bg-ink-950 px-2 py-1.5 text-xs text-slate-200 disabled:opacity-45"
                >
                  <option value="small">Small (L2, Max Heat)</option>
                  <option value="medium">Medium (L3 Balanced)</option>
                  <option value="large">Large (RAM / IMC Stream)</option>
                </select>
              </label>
            </div>
            <label className="block">
              <span className="mb-1 block text-[10px] text-slate-400 uppercase">CPU Workload</span>
              <select
                data-testid="cpu-workload"
                disabled={active}
                value={form.cpuWorkload}
                onChange={(e) => patch("cpuWorkload", e.target.value as CpuWorkload)}
                className="w-full rounded border border-ink-700 bg-ink-950 px-2 py-1.5 text-xs text-slate-200"
              >
                <option value="extreme">Extreme (FFT + FMA + NTT64)</option>
                <option value="fftTorture">FFT Torture (Smallest–Large, Rounding &lt; 0.40)</option>
                <option value="matrixFma">Dense FMA GEMM (FP64 Matrix Peak Heat)</option>
                <option value="nttPrime">NTT64 Prime (64-Bit Modular Avalanche)</option>
              </select>
            </label>
            {(form.cpuWorkload === "extreme" || form.cpuWorkload === "fftTorture") && (
              <div>
                <span className="mb-1 block text-[10px] text-slate-400 uppercase">FFT Torture Preset</span>
                <div className="grid grid-cols-4 gap-1">
                  {FFT_PRESETS.map((fp) => {
                    const selected = form.fftPreset === fp.id;
                    return (
                      <button
                        key={fp.id}
                        type="button"
                        data-testid={`fft-preset-${fp.id}`}
                        disabled={active}
                        onClick={() => patch("fftPreset", fp.id)}
                        className={`rounded border px-1.5 py-1 text-left transition ${
                          selected
                            ? "border-accent bg-accent/15 text-white"
                            : "border-ink-700 bg-ink-950/70 text-slate-300 hover:border-ink-600"
                        } disabled:opacity-45`}
                      >
                        <div className="text-[10px] font-semibold">{fp.label}</div>
                        <div className="truncate text-[8px] text-slate-400">{fp.sub}</div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
            {form.mode === "variable" && (
              <div className="grid grid-cols-2 gap-2">
                <label className="block">
                  <span className="mb-1 block text-[10px] text-slate-400 uppercase">Min Load %</span>
                  <input
                    data-testid="cpu-var-min"
                    type="number"
                    min={10}
                    max={100}
                    disabled={active}
                    value={form.cpuVarMin}
                    onChange={(e) => patch("cpuVarMin", Number(e.target.value))}
                    className="tabular w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-xs text-slate-200"
                  />
                </label>
                <label className="block">
                  <span className="mb-1 block text-[10px] text-slate-400 uppercase">Max Load %</span>
                  <input
                    data-testid="cpu-var-max"
                    type="number"
                    min={10}
                    max={100}
                    disabled={active}
                    value={form.cpuVarMax}
                    onChange={(e) => patch("cpuVarMax", Number(e.target.value))}
                    className="tabular w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-xs text-slate-200"
                  />
                </label>
              </div>
            )}
            {form.mode === "coreCycler" && (
              <label className="block">
                <span className="mb-1 block text-[10px] text-slate-400 uppercase">Core Dwell Time (s / core)</span>
                <input
                  data-testid="cpu-core-cycle-s"
                  type="number"
                  min={1}
                  max={60}
                  disabled={active}
                  value={form.cpuCoreCycleS}
                  onChange={(e) => patch("cpuCoreCycleS", Number(e.target.value))}
                  className="tabular w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-xs text-slate-200"
                />
              </label>
            )}
          </>
        )}

        {/* CPU Dataset + ISA + Workload shared for Combined and Power */}
        {(form.kind === "combined" || form.kind === "power") && (
          <>
            <div className="grid grid-cols-2 gap-2">
              <label className="block">
                <span className="mb-1 block text-[10px] text-slate-400 uppercase">CPU Dataset</span>
                <select
                  data-testid="cpu-dataset"
                  disabled={active || form.cpuWorkload === "fftTorture"}
                  value={form.dataset}
                  onChange={(e) => patch("dataset", e.target.value as Dataset)}
                  className="w-full rounded border border-ink-700 bg-ink-950 px-2 py-1.5 text-xs text-slate-200 disabled:opacity-45"
                >
                  <option value="small">Small (L2, Max Heat)</option>
                  <option value="medium">Medium (L3 Balanced)</option>
                  <option value="large">Large (RAM / IMC Stream)</option>
                </select>
              </label>
              <label className="block">
                <span className="mb-1 block text-[10px] text-slate-400 uppercase">CPU ISA</span>
                <select
                  data-testid="cpu-isa"
                  disabled={active}
                  value={form.isa}
                  onChange={(e) => patch("isa", e.target.value as IsaChoice)}
                  className="w-full rounded border border-ink-700 bg-ink-950 px-2 py-1.5 text-xs text-slate-200"
                >
                  <option value="auto">Auto (Best)</option>
                  <option value="avx2Fma" disabled={info ? !(info.isa.avx2 && info.isa.fma) : false}>
                    AVX2 + FMA
                  </option>
                  <option value="scalar">Scalar (SSE2)</option>
                  <option value="avx512" disabled={info ? !info.isa.avx512f : true}>
                    AVX-512F {info && !info.isa.avx512f ? "(N/A)" : ""}
                  </option>
                </select>
              </label>
            </div>
            <label className="block">
              <span className="mb-1 block text-[10px] text-slate-400 uppercase">CPU Workload</span>
              <select
                data-testid="cpu-workload"
                disabled={active}
                value={form.cpuWorkload}
                onChange={(e) => patch("cpuWorkload", e.target.value as CpuWorkload)}
                className="w-full rounded border border-ink-700 bg-ink-950 px-2 py-1.5 text-xs text-slate-200"
              >
                <option value="extreme">Extreme (FFT + FMA + NTT64)</option>
                <option value="fftTorture">FFT Torture (Smallest–Large, Rounding &lt; 0.40)</option>
                <option value="matrixFma">Dense FMA GEMM (FP64 Matrix)</option>
                <option value="nttPrime">NTT64 Prime (64-Bit Modular Avalanche)</option>
              </select>
            </label>
            {(form.cpuWorkload === "extreme" || form.cpuWorkload === "fftTorture") && (
              <div>
                <span className="mb-1 block text-[10px] text-slate-400 uppercase">FFT Torture Preset</span>
                <div className="grid grid-cols-4 gap-1">
                  {FFT_PRESETS.map((fp) => {
                    const activePreset = form.kind === "combined" ? form.combinedFftPreset : form.fftPreset;
                    const selected = activePreset === fp.id;
                    return (
                      <button
                        key={fp.id}
                        type="button"
                        data-testid={`fft-preset-${fp.id}`}
                        disabled={active}
                        onClick={() =>
                          patch(form.kind === "combined" ? "combinedFftPreset" : "fftPreset", fp.id)
                        }
                        className={`rounded border px-1.5 py-1 text-left transition ${
                          selected
                            ? "border-accent bg-accent/15 text-white"
                            : "border-ink-700 bg-ink-950/70 text-slate-300 hover:border-ink-600"
                        } disabled:opacity-45`}
                      >
                        <div className="text-[10px] font-semibold">{fp.label}</div>
                        <div className="truncate text-[8px] text-slate-400">{fp.sub}</div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}

        {form.kind === "cpu" && (
          <label className="block">
            <span className="mb-1 block text-[10px] text-slate-400 uppercase">Instruction Set</span>
            <select
              data-testid="cpu-isa"
              disabled={active}
              value={form.isa}
              onChange={(e) => patch("isa", e.target.value as IsaChoice)}
              className="w-full rounded border border-ink-700 bg-ink-950 px-2 py-1.5 text-xs text-slate-200"
            >
              <option value="auto">Auto (Best Supported)</option>
              <option value="avx2Fma" disabled={info ? !(info.isa.avx2 && info.isa.fma) : false}>
                AVX2 + FMA
              </option>
              <option value="scalar">Scalar (SSE2)</option>
              <option value="avx512" disabled={info ? !info.isa.avx512f : true}>
                AVX-512F {info && !info.isa.avx512f ? "(Unsupported)" : ""}
              </option>
            </select>
          </label>
        )}

        {(form.kind === "cpu" || form.kind === "memory" || form.kind === "combined" || form.kind === "power") && (
          <div className="space-y-1.5">
            <label className="block">
              <div className="mb-1 flex justify-between text-[10px] text-slate-400 uppercase">
                <span>{form.kind === "memory" ? "Memory Threads" : "CPU Threads"}</span>
                <span className="tabular font-mono text-slate-300">
                  {form.threads ?? defaultThreadsForKind} / {maxThreadsForKind}
                </span>
              </div>
              <input
                data-testid="threads-input"
                type="number"
                min={1}
                max={maxThreadsForKind}
                disabled={active}
                placeholder={`Auto (${defaultThreadsForKind})`}
                value={form.threads ?? ""}
                onChange={(e) => patch("threads", e.target.value === "" ? null : Number(e.target.value))}
                className="tabular w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-xs text-slate-200"
              />
            </label>
            {form.kind === "cpu" && logical > physical && (
              <label className="flex cursor-pointer items-center gap-2 text-[11px] text-slate-300">
                <input
                  type="checkbox"
                  data-testid="physical-cores-checkbox"
                  disabled={active}
                  checked={form.physicalCoresOnly}
                  onChange={(e) => {
                    const checked = e.target.checked;
                    setValidationErr(null);
                    setForm((f) => ({
                      ...f,
                      physicalCoresOnly: checked,
                      threads: checked && f.threads != null && f.threads > physical ? physical : f.threads,
                    }));
                  }}
                  className="h-3.5 w-3.5 accent-accent"
                />
                <span>Physical cores only (no SMT · {physical}C)</span>
              </label>
            )}
          </div>
        )}

        {/* Memory / Combined RAM controls */}
        {(form.kind === "memory" || form.kind === "combined") && (
          <div className="space-y-1.5">
            <label className="block">
              <div className="mb-1 flex items-center justify-between text-[10px] text-slate-400 uppercase">
                <span>RAM Allocation Target</span>
                <div className="flex items-center gap-1">
                  {![80, 90, 95, 100].includes(form.memPct) && (
                    <span className="tabular mr-0.5 font-mono text-slate-200">{form.memPct}%</span>
                  )}
                  {[80, 90, 95, 100].map((pct) => (
                    <button
                      key={pct}
                      type="button"
                      data-testid={`mem-preset-${pct}`}
                      disabled={active}
                      onClick={() => patch("memPct", pct)}
                      className={`rounded border px-1.5 py-0.2 font-mono text-[9px] ${
                        form.memPct === pct
                          ? "border-accent bg-accent/20 font-semibold text-white"
                          : "border-ink-700 bg-ink-950 text-slate-400 hover:text-slate-200"
                      } disabled:opacity-45`}
                    >
                      {pct}%
                    </button>
                  ))}
                </div>
              </div>
              <input
                data-testid="mem-pct-slider"
                type="range"
                min={10}
                max={100}
                disabled={active}
                value={form.memPct}
                onChange={(e) => patch("memPct", Number(e.target.value))}
                className="w-full accent-accent"
              />
            </label>
            {form.memPct > limits.memMaxPct && (
              <div className="flex items-center justify-between gap-2 rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-[10px] text-amber-200">
                <span>Safety RAM cap is {limits.memMaxPct}%</span>
                <button
                  type="button"
                  data-testid="match-ram-cap-btn"
                  disabled={active}
                  onClick={() => void updateLimits({ ...limits, memMaxPct: form.memPct })}
                  className="rounded border border-amber-400/60 bg-amber-500/20 px-1.5 py-0.5 font-semibold text-amber-100 hover:bg-amber-500/30"
                >
                  Raise cap to {form.memPct}%
                </button>
              </div>
            )}
          </div>
        )}

        {(form.kind === "memory" || form.kind === "combined") && (
          <div>
            <span className="mb-1 block text-[10px] text-slate-400 uppercase">RAM Test Profile</span>
            <div className="grid grid-cols-2 gap-1.5">
              {MEM_PROFILES.map((p) => {
                const selected = form.memProfile === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    data-testid={`mem-profile-${p.id}`}
                    disabled={active}
                    onClick={() => patch("memProfile", p.id)}
                    className={`rounded border px-2 py-1.5 text-left transition ${
                      selected
                        ? p.id === "absolut"
                          ? "border-orange-500 bg-orange-500/15 text-white"
                          : "border-accent bg-accent/15 text-white"
                        : "border-ink-700 bg-ink-950/70 text-slate-300 hover:border-ink-600"
                    } disabled:opacity-45`}
                  >
                    <div className="flex items-center justify-between text-[11px] font-semibold">
                      <span>{p.label}</span>
                      <span
                        className={`rounded px-1 text-[8px] font-bold uppercase ${
                          p.id === "absolut"
                            ? "bg-orange-500/25 text-orange-300"
                            : "bg-sky-500/20 text-sky-300"
                        }`}
                      >
                        {p.badge}
                      </span>
                    </div>
                    <div className="truncate text-[9px] text-slate-400">{p.sub}</div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {form.kind === "memory" && form.memProfile === "absolut" && (
          <div className="rounded border border-orange-500/30 bg-orange-950/20 px-2.5 py-2">
            <div className="flex items-center justify-between text-[10px] text-slate-300 uppercase">
              <span>Absolut Cycles</span>
              <div className="flex gap-1">
                {CYCLE_PRESETS.map((pr) => {
                  const isSel = form.memCycles === pr.cycles;
                  return (
                    <button
                      key={pr.label}
                      type="button"
                      data-testid={`mem-cycles-${pr.cycles ?? "inf"}`}
                      disabled={active}
                      onClick={() => patch("memCycles", pr.cycles)}
                      className={`rounded border px-1.5 py-0.5 font-mono text-[9px] transition ${
                        isSel
                          ? "border-orange-400 bg-orange-500/25 font-semibold text-white"
                          : "border-ink-700 bg-ink-950 text-slate-400 hover:text-slate-200"
                      } disabled:opacity-45`}
                    >
                      {pr.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {form.kind === "memory" && form.memProfile === "standard" && (
          <label className="block">
            <span className="mb-1 block text-[10px] text-slate-400 uppercase">Memory Pattern (Bit-Exact)</span>
            <select
              data-testid="mem-pattern"
              disabled={active}
              value={form.memPattern}
              onChange={(e) => patch("memPattern", e.target.value as MemoryPattern)}
              className="w-full rounded border border-ink-700 bg-ink-950 px-2 py-1.5 text-xs text-slate-200"
            >
              <option value="all">All 5 Patterns (OC Suite, Rotating)</option>
              <option value="movingInversions">Moving Inversions (64-bit Bus Toggle)</option>
              <option value="blockStrided">Strided Bank-Conflict (tRCD/tRFC/IMC)</option>
              <option value="walkingOnes">Walking 1s &amp; 0s (Bit Stuck)</option>
              <option value="addressInAddress">Address-in-Address (Row/Bank)</option>
              <option value="random">PRNG Random (Crosstalk)</option>
            </select>
          </label>
        )}

        {/* GPU thermal profile, steady / adaptive mode, and deterministic verification */}
        {form.kind === "gpu" && (
          <>
            <div>
              <span className="mb-1 block text-[10px] text-slate-400 uppercase">GPU Thermal / Load Profile</span>
              <div className="grid grid-cols-3 gap-1.5">
                {GPU_PROFILES.map((p) => {
                  const selected = form.gpuProfile === p.id;
                  return (
                    <button
                      key={p.id}
                      type="button"
                      data-testid={`gpu-profile-${p.id}`}
                      disabled={active}
                      onClick={() => patch("gpuProfile", p.id)}
                      className={`rounded border px-2 py-1.5 text-left transition ${
                        selected
                          ? p.id === "extreme"
                            ? "border-orange-500 bg-orange-500/15 text-white"
                            : "border-accent bg-accent/15 text-white"
                          : "border-ink-700 bg-ink-950/70 text-slate-300 hover:border-ink-600"
                      } disabled:opacity-45`}
                    >
                      <div className="flex items-center justify-between text-[11px] font-semibold">
                        <span>{p.label}</span>
                        <span
                          className={`rounded px-1 text-[8px] font-bold uppercase ${
                            p.id === "extreme"
                              ? "bg-orange-500/25 text-orange-300"
                              : p.id === "heavy"
                                ? "bg-amber-500/20 text-amber-300"
                                : "bg-sky-500/20 text-sky-300"
                          }`}
                        >
                          {p.heat}
                        </span>
                      </div>
                      <div className="truncate text-[9px] text-slate-400">{p.sub}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            <label className="block">
              <span className="mb-1 block text-[10px] text-slate-400 uppercase">GPU Load Mode</span>
              <select
                data-testid="gpu-mode"
                disabled={active}
                value={form.gpuMode}
                onChange={(e) => patch("gpuMode", e.target.value as LoadMode)}
                className="w-full rounded border border-ink-700 bg-ink-950 px-2 py-1.5 text-xs text-slate-200"
              >
                <option value="steady">Steady (Sustained Heat / TDP)</option>
                <option value="variable">3D Adaptive (Variable V/F Ramp)</option>
              </select>
            </label>

            {form.gpuMode === "steady" ? (
              <label className="block">
                <div className="mb-1 flex justify-between text-[10px] text-slate-400 uppercase">
                  <span>GPU Load Intensity</span>
                  <span className="tabular font-mono text-slate-200">{form.gpuIntensity}%</span>
                </div>
                <input
                  data-testid="gpu-intensity-slider"
                  type="range"
                  min={10}
                  max={100}
                  step={5}
                  disabled={active}
                  value={form.gpuIntensity}
                  onChange={(e) => patch("gpuIntensity", Number(e.target.value))}
                  className="w-full accent-accent"
                />
              </label>
            ) : (
              <div className="grid grid-cols-3 gap-1.5">
                <label className="block">
                  <span className="mb-1 block text-[10px] text-slate-400 uppercase">Start %</span>
                  <input
                    data-testid="gpu-var-min"
                    type="number"
                    min={10}
                    max={100}
                    disabled={active}
                    value={form.gpuVarMin}
                    onChange={(e) => patch("gpuVarMin", Number(e.target.value))}
                    className="tabular w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-xs text-slate-200"
                  />
                </label>
                <label className="block">
                  <span className="mb-1 block text-[10px] text-slate-400 uppercase">Max %</span>
                  <input
                    data-testid="gpu-var-max"
                    type="number"
                    min={10}
                    max={100}
                    disabled={active}
                    value={form.gpuVarMax}
                    onChange={(e) => patch("gpuVarMax", Number(e.target.value))}
                    className="tabular w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-xs text-slate-200"
                  />
                </label>
                <label className="block">
                  <span className="mb-1 block text-[10px] text-slate-400 uppercase">Step (s)</span>
                  <input
                    data-testid="gpu-var-step"
                    type="number"
                    min={1}
                    max={60}
                    disabled={active}
                    value={form.gpuVarStepS}
                    onChange={(e) => patch("gpuVarStepS", Number(e.target.value))}
                    className="tabular w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-xs text-slate-200"
                  />
                </label>
              </div>
            )}

            <label className="flex cursor-pointer items-center gap-2 text-[11px] text-slate-300">
              <input
                type="checkbox"
                data-testid="gpu-error-check"
                disabled={active}
                checked={form.gpuErrorCheck}
                onChange={(e) => patch("gpuErrorCheck", e.target.checked)}
                className="h-3.5 w-3.5 accent-accent"
              />
              <span>Error detection (deterministic GPU verification)</span>
            </label>
          </>
        )}

        {/* VRAM controls */}
        {form.kind === "vram" && (
          <>
            <label className="block">
              <div className="mb-1 flex items-center justify-between text-[10px] text-slate-400 uppercase">
                <span>VRAM Target (OOM-safe)</span>
                <div className="flex items-center gap-1">
                  {![80, 90, 95, 100].includes(form.vramPct) && (
                    <span className="tabular mr-0.5 font-mono text-slate-200">{form.vramPct}%</span>
                  )}
                  {[80, 90, 95, 100].map((pct) => (
                    <button
                      key={pct}
                      type="button"
                      data-testid={`vram-preset-${pct}`}
                      disabled={active}
                      onClick={() => patch("vramPct", pct)}
                      className={`rounded border px-1.5 py-0.2 font-mono text-[9px] ${
                        form.vramPct === pct
                          ? "border-accent bg-accent/20 font-semibold text-white"
                          : "border-ink-700 bg-ink-950 text-slate-400 hover:text-slate-200"
                      } disabled:opacity-45`}
                    >
                      {pct}%
                    </button>
                  ))}
                </div>
              </div>
              <input
                data-testid="vram-pct-slider"
                type="range"
                min={10}
                max={100}
                disabled={active}
                value={form.vramPct}
                onChange={(e) => patch("vramPct", Number(e.target.value))}
                className="w-full accent-accent"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-[10px] text-slate-400 uppercase">VRAM Pattern (Bit-Exact)</span>
              <select
                data-testid="vram-pattern"
                disabled={active}
                value={form.vramPattern}
                onChange={(e) => patch("vramPattern", e.target.value as MemoryPattern)}
                className="w-full rounded border border-ink-700 bg-ink-950 px-2 py-1.5 text-xs text-slate-200"
              >
                <option value="all">All 8 Patterns (Absolut GDDR + MirrorMove128)</option>
                <option value="movingInversions">Moving Inversions (Bus Toggle)</option>
                <option value="blockStrided">Strided Bank-Conflict</option>
                <option value="walkingOnes">Walking 1s &amp; 0s (Bit Stuck)</option>
                <option value="addressInAddress">Address-in-Address (Bank/Bus)</option>
                <option value="random">PRNG Random (GDDR Stress)</option>
              </select>
            </label>
          </>
        )}

        {/* Power (PSU) GPU sub-controls */}
        {form.kind === "power" && (
          <>
            <div>
              <span className="mb-1 block text-[10px] text-slate-400 uppercase">PSU GPU Workload Profile</span>
              <div className="grid grid-cols-3 gap-1.5">
                {GPU_PROFILES.map((p) => {
                  const selected = form.powerGpuProfile === p.id;
                  return (
                    <button
                      key={p.id}
                      type="button"
                      data-testid={`power-gpu-profile-${p.id}`}
                      disabled={active}
                      onClick={() => patch("powerGpuProfile", p.id)}
                      className={`rounded border px-2 py-1 text-center text-[11px] font-semibold transition ${
                        selected
                          ? p.id === "extreme"
                            ? "border-orange-500 bg-orange-500/15 text-white"
                            : "border-accent bg-accent/15 text-white"
                          : "border-ink-700 bg-ink-950/70 text-slate-300 hover:border-ink-600"
                      } disabled:opacity-45`}
                    >
                      {p.label}
                    </button>
                  );
                })}
              </div>
            </div>
            <label className="block">
              <div className="mb-1 flex justify-between text-[10px] text-slate-400 uppercase">
                <span>GPU Load Intensity</span>
                <span className="tabular font-mono text-slate-200">{form.powerGpuIntensity}%</span>
              </div>
              <input
                data-testid="gpu-intensity-slider"
                type="range"
                min={10}
                max={100}
                step={5}
                disabled={active}
                value={form.powerGpuIntensity}
                onChange={(e) => patch("powerGpuIntensity", Number(e.target.value))}
                className="w-full accent-accent"
              />
            </label>
          </>
        )}

        {/* Duration presets, custom duration, and error cutoff */}
        <div>
          <div className="mb-1 flex items-center justify-between text-[10px] text-slate-400 uppercase">
            <span>Duration Preset</span>
            <div className="flex gap-1">
              {DURATION_PRESETS.map((pr) => {
                const isSel = form.durationMin === pr.min;
                return (
                  <button
                    key={pr.label}
                    type="button"
                    data-testid={`dur-preset-${pr.min ?? "inf"}`}
                    disabled={active}
                    onClick={() => patch("durationMin", pr.min)}
                    className={`rounded border px-1.5 py-0.5 font-mono text-[9px] transition ${
                      isSel
                        ? "border-accent bg-accent/20 text-white"
                        : "border-ink-700 bg-ink-950 text-slate-400 hover:text-slate-200"
                    } disabled:opacity-45`}
                  >
                    {pr.label}
                  </button>
                );
              })}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <label className="block">
              <span className="mb-1 block text-[10px] text-slate-400 uppercase">Custom (min)</span>
              <input
                data-testid="duration-input"
                type="number"
                min={0.5}
                step={1}
                disabled={active}
                placeholder="Unlimited"
                value={form.durationMin ?? ""}
                onChange={(e) => patch("durationMin", e.target.value === "" ? null : Number(e.target.value))}
                className="tabular w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-xs text-slate-200"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-[10px] text-slate-400 uppercase">Max Errors Cutoff</span>
              <input
                data-testid="max-errors-input"
                type="number"
                min={0}
                step={1}
                disabled={active || form.stopOnError}
                placeholder={form.stopOnError ? "1 (Stop on 1st)" : "Unlimited"}
                value={form.stopOnError ? "" : (form.maxErrors ?? "")}
                onChange={(e) => patch("maxErrors", e.target.value === "" ? null : Number(e.target.value))}
                className="tabular w-full rounded border border-ink-700 bg-ink-950 px-2 py-1 font-mono text-xs text-slate-200 disabled:opacity-50"
              />
            </label>
          </div>
        </div>

        <label className="flex cursor-pointer items-center gap-2 pt-0.5 text-xs text-slate-300">
          <input
            type="checkbox"
            data-testid="stop-on-error-checkbox"
            disabled={active}
            checked={form.stopOnError}
            onChange={(e) => patch("stopOnError", e.target.checked)}
            className="h-3.5 w-3.5 accent-accent"
          />
          <span>Stop immediately on first error</span>
        </label>

        {validationErr && (
          <div data-testid="form-validation-error" className="rounded border border-red-500/60 bg-red-950/60 px-2.5 py-1.5 text-[11px] text-red-200">
            {validationErr}
          </div>
        )}

        <div className="flex gap-2 pt-1">
          {!active ? (
            <button
              type="button"
              data-testid="start-stress-btn"
              onClick={onStart}
              className="flex-1 rounded-md bg-accent px-3 py-2 text-xs font-bold tracking-wider text-white uppercase shadow hover:bg-accent-dim"
            >
              Start {KINDS.find((k) => k.id === form.kind)?.label} Test
            </button>
          ) : (
            <button
              type="button"
              data-testid="stop-stress-btn"
              onClick={() => void stopStress()}
              className="flex-1 rounded-md border border-amber-500/70 bg-amber-500/15 px-3 py-2 text-xs font-bold tracking-wider text-amber-200 uppercase hover:bg-amber-500/25"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Live Run Telemetry + Error Summary */}
      {stress && stress.phase !== "idle" && (
        <div className="mt-3.5 space-y-2 border-t border-ink-800 pt-3">
          <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
            <div className="rounded border border-ink-800 bg-ink-950/80 p-2">
              <div className="text-[9px] text-slate-500 uppercase">Elapsed</div>
              <div className="tabular text-sm font-bold text-white">
                {fmt.duration(stress.elapsedS)}
                {stress.durationS ? <span className="text-[10px] font-normal text-slate-400"> / {fmt.duration(stress.durationS)}</span> : null}
              </div>
            </div>
            <div className="rounded border border-ink-800 bg-ink-950/80 p-2">
              <div className="text-[9px] text-slate-500 uppercase">Errors</div>
              <div
                data-testid="error-counter"
                className={`tabular text-sm font-bold ${stress.totalErrors > 0 ? "text-red-400" : "text-emerald-400"}`}
              >
                {stress.totalErrors}
                {stress.gpuErrors > 0 ? <span className="ml-1 text-[10px] font-normal text-red-300">(GPU {stress.gpuErrors})</span> : null}
              </div>
            </div>
            {(stress.gflops > 0 || stress.kind === "cpu" || stress.kind === "combined" || stress.kind === "power") && (
              <div className="rounded border border-ink-800 bg-ink-950/80 p-2">
                <div className="text-[9px] text-slate-500 uppercase">CPU GFLOPS</div>
                <div className="tabular text-xs font-semibold text-sky-300">{fmt.num(stress.gflops, 1)}</div>
              </div>
            )}
            {(stress.gpuGflops > 0 || stress.kind === "gpu" || stress.kind === "power") && (
              <div className="rounded border border-ink-800 bg-ink-950/80 p-2">
                <div className="text-[9px] text-slate-500 uppercase">GPU GFLOPS</div>
                <div className="tabular text-xs font-semibold text-violet-300">{fmt.num(stress.gpuGflops, 0)}</div>
              </div>
            )}
            {(stress.memMbps > 0 || stress.kind === "memory" || stress.kind === "vram" || stress.kind === "combined") && (
              <div className="rounded border border-ink-800 bg-ink-950/80 p-2">
                <div className="text-[9px] text-slate-500 uppercase">{stress.kind === "vram" ? "VRAM BW" : "RAM BW"}</div>
                <div className="tabular text-xs font-semibold text-amber-300">{fmt.throughput(stress.memMbps)}</div>
              </div>
            )}
            {(stress.memAllocatedBytes > 0 || stress.vramAllocatedBytes > 0) && (
              <div className="rounded border border-ink-800 bg-ink-950/80 p-2">
                <div className="text-[9px] text-slate-500 uppercase">
                  {stress.kind === "vram" ? "VRAM Alloc" : "RAM Alloc"}
                </div>
                <div data-testid="allocated-bytes-tile" className="tabular text-xs font-semibold text-emerald-300">
                  {fmt.bytes(stress.kind === "vram" ? stress.vramAllocatedBytes : stress.memAllocatedBytes)}
                </div>
              </div>
            )}
            {stress.isaUsed && (
              <div className="rounded border border-ink-800 bg-ink-950/80 p-2">
                <div className="flex items-center justify-between text-[9px] text-slate-500 uppercase">
                  <span>Kernel</span>
                  {active && stress.loadLevel > 0 && stress.loadLevel < 100 && (
                    <span className="text-sky-300">{stress.loadLevel}%</span>
                  )}
                </div>
                <div className="truncate text-[11px] text-slate-200">{stress.isaUsed}</div>
              </div>
            )}
          </div>

          {stress.message && (
            <div
              data-testid="stress-live-stage"
              className="truncate rounded border border-ink-700/80 bg-ink-950/90 px-2 py-1 font-mono text-[10px] text-amber-200"
              title={stress.message}
            >
              {stress.message}
            </div>
          )}

          {stress.timeCapS && (
            <div className="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-[10px] text-amber-200">
              No CPU thermal sensor — capped at {Math.round(stress.timeCapS / 60)} min
            </div>
          )}

          {workerErrors.length > 0 && (
            <div
              data-testid="worker-error-log"
              className="max-h-32 space-y-1 overflow-y-auto rounded border border-red-500/50 bg-red-950/40 p-2 font-mono text-[10px]"
            >
              <div className="text-[9px] font-bold tracking-wider text-red-300 uppercase">
                Detected Errors ({workerErrors.length})
              </div>
              {workerErrors.slice(0, 10).map((e, idx) => (
                <div key={`${e.timestampMs}-${idx}`} className="break-all border-t border-red-500/20 pt-1 text-red-200">
                  <span className="mr-1.5 text-red-400/80">{new Date(e.timestampMs).toTimeString().slice(0, 8)}</span>
                  {e.detail}
                </div>
              ))}
            </div>
          )}

          {/* Per-core error mini-table */}
          {stress.coreErrors.length > 0 && (stress.kind === "cpu" || stress.kind === "combined" || stress.kind === "power") && (
            <div className="rounded border border-ink-800 bg-ink-950/70 p-2">
              <div className="mb-1 text-[9px] text-slate-500 uppercase">Per-Thread Errors</div>
              <div className="grid grid-cols-8 gap-1 font-mono text-[10px]">
                {stress.coreErrors.map((errs, idx) => (
                  <div
                    key={idx}
                    className={`tabular rounded px-1 py-0.5 text-center ${
                      errs > 0 ? "bg-red-500/25 font-bold text-red-300" : "bg-ink-900 text-slate-500"
                    }`}
                    title={`Thread ${idx}: ${errs} error(s)`}
                  >
                    T{idx}:{errs}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
