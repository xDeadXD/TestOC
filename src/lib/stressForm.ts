import type {
  CpuWorkload,
  Dataset,
  FftPreset,
  GpuProfile,
  IsaChoice,
  LoadMode,
  MemoryPattern,
  MemoryProfile,
  StressConfig,
  StressKind,
} from "./types";

export const PREFS_STORAGE_KEY = "testoc.preferences.v5";

export interface StressForm {
  kind: StressKind;
  // CPU / Combined / Power
  threads: number | null; // null = auto
  physicalCoresOnly: boolean;
  isa: IsaChoice;
  dataset: Dataset;
  cpuWorkload: CpuWorkload;
  fftPreset: FftPreset;
  combinedFftPreset: FftPreset;
  mode: LoadMode;
  cpuVarMin: number;
  cpuVarMax: number;
  cpuCoreCycleS: number;
  // Memory / Combined
  memProfile: MemoryProfile;
  memCycles: number | null; // null = unlimited
  memPct: number;
  memPattern: MemoryPattern;
  // GPU
  gpuMode: LoadMode;
  gpuProfile: GpuProfile;
  gpuIntensity: number;
  gpuVarMin: number;
  gpuVarMax: number;
  gpuVarStepS: number;
  gpuErrorCheck: boolean;
  // VRAM
  vramPct: number;
  vramPattern: MemoryPattern;
  // Power
  powerGpuProfile: GpuProfile;
  powerGpuIntensity: number;
  // Shared
  durationMin: number | null; // null = unlimited
  stopOnError: boolean;
  maxErrors: number | null; // null / 0 = unlimited when stopOnError is false
}

export function defaultForm(): StressForm {
  return {
    kind: "cpu",
    threads: null,
    physicalCoresOnly: false,
    isa: "auto",
    dataset: "small",
    cpuWorkload: "extreme",
    fftPreset: "small",
    combinedFftPreset: "blend",
    mode: "steady",
    cpuVarMin: 20,
    cpuVarMax: 100,
    cpuCoreCycleS: 3,
    memProfile: "absolut",
    memCycles: null,
    memPct: 100,
    memPattern: "all",
    gpuMode: "steady",
    gpuProfile: "extreme",
    gpuIntensity: 100,
    gpuVarMin: 20,
    gpuVarMax: 100,
    gpuVarStepS: 5,
    gpuErrorCheck: true,
    vramPct: 100,
    vramPattern: "all",
    powerGpuProfile: "extreme",
    powerGpuIntensity: 100,
    durationMin: null,
    stopOnError: false,
    maxErrors: null,
  };
}

export function loadSavedForm(): StressForm {
  const base = defaultForm();
  try {
    if (typeof window === "undefined" || !window.localStorage) return base;
    const raw = window.localStorage.getItem(PREFS_STORAGE_KEY);
    if (!raw) return base;
    const parsed = JSON.parse(raw) as Partial<StressForm>;
    if (!parsed || typeof parsed !== "object") return base;
    return { ...base, ...parsed };
  } catch {
    return base;
  }
}

export function saveForm(form: StressForm): void {
  try {
    if (typeof window === "undefined" || !window.localStorage) return;
    window.localStorage.setItem(PREFS_STORAGE_KEY, JSON.stringify(form));
  } catch {
    // Ignore storage quota / privacy mode errors.
  }
}

interface InfoLike {
  physicalCores?: number;
  logicalCores: number;
  isa: { avx2: boolean; fma: boolean; avx512f: boolean };
}

export type BuildResult = { ok: true; config: StressConfig } | { ok: false; error: string };

/** Validate the form client-side (the backend re-validates everything). */
export function buildConfig(f: StressForm, info: InfoLike): BuildResult {
  const logical = info.logicalCores;
  const physical = Math.max(1, Math.min(logical, info.physicalCores ?? Math.max(1, Math.floor(logical / 2))));
  const threadsFor = (def: number, max = logical): number | string => {
    const t = f.threads ?? def;
    if (!Number.isInteger(t) || t < 1 || t > max) return `Threads must be 1–${max}`;
    return t;
  };
  const checkIsa = (isa: IsaChoice): string | null => {
    if (isa === "avx512" && !info.isa.avx512f) return "AVX-512 is not supported by this CPU";
    if (isa === "avx2Fma" && !(info.isa.avx2 && info.isa.fma)) return "AVX2/FMA is not supported";
    return null;
  };
  const checkVarRange = (min: number, max: number, label: string): string | null => {
    if (!Number.isFinite(min) || !Number.isFinite(max) || min < 10 || max > 100 || min > max) {
      return `${label} range must satisfy 10 ≤ min ≤ max ≤ 100 %`;
    }
    return null;
  };

  let durationS: number | null = null;
  if (f.durationMin != null) {
    if (!Number.isFinite(f.durationMin) || f.durationMin <= 0 || f.durationMin > 7 * 24 * 60)
      return { ok: false, error: "Duration must be > 0 and ≤ 7 days" };
    durationS = Math.round(f.durationMin * 60);
  }
  let maxErrors: number | null = null;
  if (!f.stopOnError && f.maxErrors != null && f.maxErrors > 0) {
    if (!Number.isInteger(f.maxErrors) || f.maxErrors > 1_000_000)
      return { ok: false, error: "Max errors must be an integer 1–1,000,000" };
    maxErrors = f.maxErrors;
  }

  switch (f.kind) {
    case "cpu": {
      const physMode = f.physicalCoresOnly || (f.mode === "coreCycler" && f.threads == null);
      const maxT = physMode ? physical : logical;
      const t = threadsFor(maxT, maxT);
      if (typeof t === "string") return { ok: false, error: t };
      const isaErr = checkIsa(f.isa);
      if (isaErr) return { ok: false, error: isaErr };
      if (f.mode === "variable") {
        const vrErr = checkVarRange(f.cpuVarMin, f.cpuVarMax, "Variable load");
        if (vrErr) return { ok: false, error: vrErr };
      }
      if (f.mode === "coreCycler") {
        if (!Number.isInteger(f.cpuCoreCycleS) || f.cpuCoreCycleS < 1 || f.cpuCoreCycleS > 60)
          return { ok: false, error: "Core cycler step interval must be 1–60 s" };
      }
      return {
        ok: true,
        config: {
          kind: "cpu",
          threads: t,
          physicalCoresOnly: physMode,
          isa: f.isa,
          dataset: f.dataset,
          workload: f.cpuWorkload,
          fftPreset: f.fftPreset,
          mode: f.mode,
          variableMinPct: Math.round(f.cpuVarMin),
          variableMaxPct: Math.round(f.cpuVarMax),
          coreCycleS: Math.round(f.cpuCoreCycleS),
          stopOnError: f.stopOnError,
          maxErrors,
          durationS,
        },
      };
    }
    case "memory": {
      const defThreads = f.memProfile === "absolut" ? logical : Math.min(8, Math.max(1, Math.floor(logical / 2)));
      const t = threadsFor(defThreads);
      if (typeof t === "string") return { ok: false, error: t };
      if (!(f.memPct >= 10 && f.memPct <= 100)) return { ok: false, error: "Memory target must be 10–100 %" };
      let cycles: number | null = null;
      if (f.memProfile === "absolut" && f.memCycles != null) {
        if (!Number.isInteger(f.memCycles) || f.memCycles < 1 || f.memCycles > 100)
          return { ok: false, error: "Absolut cycle target must be 1–100" };
        cycles = f.memCycles;
      }
      return {
        ok: true,
        config: {
          kind: "memory",
          threads: t,
          targetPct: Math.round(f.memPct),
          profile: f.memProfile,
          pattern: f.memPattern,
          cycles,
          stopOnError: f.stopOnError,
          maxErrors,
          durationS,
        },
      };
    }
    case "combined": {
      const t = threadsFor(logical);
      if (typeof t === "string") return { ok: false, error: t };
      const isaErr = checkIsa(f.isa);
      if (isaErr) return { ok: false, error: isaErr };
      if (!(f.memPct >= 10 && f.memPct <= 100)) return { ok: false, error: "Memory target must be 10–100 %" };
      return {
        ok: true,
        config: {
          kind: "combined",
          cpuThreads: t,
          cpuIsa: f.isa,
          cpuDataset: f.dataset,
          cpuWorkload: f.cpuWorkload,
          cpuFftPreset: f.combinedFftPreset,
          memTargetPct: Math.round(f.memPct),
          memProfile: f.memProfile,
          stopOnError: f.stopOnError,
          maxErrors,
          durationS,
        },
      };
    }
    case "gpu": {
      if (!(f.gpuIntensity >= 10 && f.gpuIntensity <= 100))
        return { ok: false, error: "GPU intensity must be 10–100 %" };
      if (f.gpuMode === "variable") {
        const vrErr = checkVarRange(f.gpuVarMin, f.gpuVarMax, "Adaptive GPU load");
        if (vrErr) return { ok: false, error: vrErr };
        if (!Number.isInteger(f.gpuVarStepS) || f.gpuVarStepS < 1 || f.gpuVarStepS > 60)
          return { ok: false, error: "Adaptive step interval must be 1–60 s" };
      }
      return {
        ok: true,
        config: {
          kind: "gpu",
          mode: f.gpuMode,
          profile: f.gpuProfile,
          intensity: Math.round(f.gpuIntensity),
          variableMinPct: Math.round(f.gpuVarMin),
          variableMaxPct: Math.round(f.gpuVarMax),
          variableStepS: Math.round(f.gpuVarStepS),
          errorCheck: f.gpuErrorCheck,
          stopOnError: f.stopOnError,
          maxErrors,
          durationS,
        },
      };
    }
    case "vram": {
      if (!(f.vramPct >= 10 && f.vramPct <= 100)) return { ok: false, error: "VRAM target must be 10–100 %" };
      return {
        ok: true,
        config: {
          kind: "vram",
          targetPct: Math.round(f.vramPct),
          pattern: f.vramPattern,
          stopOnError: f.stopOnError,
          maxErrors,
          durationS,
        },
      };
    }
    case "power": {
      const t = threadsFor(logical);
      if (typeof t === "string") return { ok: false, error: t };
      const isaErr = checkIsa(f.isa);
      if (isaErr) return { ok: false, error: isaErr };
      if (!(f.powerGpuIntensity >= 10 && f.powerGpuIntensity <= 100))
        return { ok: false, error: "GPU intensity must be 10–100 %" };
      return {
        ok: true,
        config: {
          kind: "power",
          cpuThreads: t,
          cpuIsa: f.isa,
          cpuDataset: f.dataset,
          cpuWorkload: f.cpuWorkload,
          cpuFftPreset: f.fftPreset,
          gpuProfile: f.powerGpuProfile,
          gpuIntensity: Math.round(f.powerGpuIntensity),
          stopOnError: f.stopOnError,
          maxErrors,
          durationS,
        },
      };
    }
  }
}
