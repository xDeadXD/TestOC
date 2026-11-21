import { describe, expect, it } from "vitest";
import { classifyTemp, loadColor } from "./thresholds";
import { fmt } from "./format";
import { buildConfig, defaultForm } from "./stressForm";
import { mergeStressStatus } from "../store/telemetry";
import type { StressStatus } from "./types";

describe("classifyTemp", () => {
  it("classifies against warn/abort", () => {
    expect(classifyTemp(null, 90, 95)).toBe("na");
    expect(classifyTemp(Number.NaN, 90, 95)).toBe("na");
    expect(classifyTemp(70, 90, 95)).toBe("ok");
    expect(classifyTemp(90, 90, 95)).toBe("warn");
    expect(classifyTemp(95, 90, 95)).toBe("crit");
  });
  it("load colors are valid css", () => {
    expect(loadColor(0)).toMatch(/^hsl\(/);
    expect(loadColor(150)).toBe(loadColor(100));
  });
});

describe("fmt", () => {
  it("formats values", () => {
    expect(fmt.num(null)).toBe("—");
    expect(fmt.num(12.345, 1, " °C")).toBe("12.3 °C");
    expect(fmt.mhz(4650)).toBe("4.65 GHz");
    expect(fmt.mhz(800)).toBe("800 MHz");
    expect(fmt.bytes(16 * 1024 ** 3)).toBe("16.0 GB");
    expect(fmt.duration(3725)).toBe("1:02:05");
    expect(fmt.duration(65)).toBe("01:05");
    expect(fmt.throughput(25_400)).toBe("25.4 GB/s");
  });
});

describe("buildConfig", () => {
  const info = { physicalCores: 8, logicalCores: 16, isa: { avx2: true, fma: true, avx512f: false } };

  it("builds a cpu config matching the StressConfig schema", () => {
    const r = buildConfig(
      { ...defaultForm(), kind: "cpu", dataset: "medium", physicalCoresOnly: true, durationMin: 10, maxErrors: 25 },
      info
    );
    expect(r).toEqual({
      ok: true,
      config: {
        kind: "cpu",
        threads: 8,
        physicalCoresOnly: true,
        isa: "auto",
        dataset: "medium",
        workload: "extreme",
        fftPreset: "small",
        mode: "steady",
        variableMinPct: 20,
        variableMaxPct: 100,
        coreCycleS: 3,
        stopOnError: false,
        maxErrors: 25,
        durationS: 600,
      },
    });

    const cc = buildConfig(
      { ...defaultForm(), kind: "cpu", mode: "coreCycler", cpuWorkload: "nttPrime", cpuCoreCycleS: 5 },
      info
    );
    expect(cc).toEqual({
      ok: true,
      config: {
        kind: "cpu",
        threads: 8,
        physicalCoresOnly: true,
        isa: "auto",
        dataset: "small",
        workload: "nttPrime",
        fftPreset: "small",
        mode: "coreCycler",
        variableMinPct: 20,
        variableMaxPct: 100,
        coreCycleS: 5,
        stopOnError: false,
        maxErrors: null,
        durationS: null,
      },
    });
  });

  it("builds gpu extreme & 3D adaptive configs", () => {
    const g = buildConfig(
      {
        ...defaultForm(),
        kind: "gpu",
        gpuMode: "variable",
        gpuProfile: "extreme",
        gpuVarMin: 30,
        gpuVarMax: 95,
        gpuVarStepS: 4,
        gpuErrorCheck: true,
      },
      info
    );
    expect(g).toEqual({
      ok: true,
      config: {
        kind: "gpu",
        mode: "variable",
        profile: "extreme",
        intensity: 100,
        variableMinPct: 30,
        variableMaxPct: 95,
        variableStepS: 4,
        errorCheck: true,
        stopOnError: false,
        maxErrors: null,
        durationS: null,
      },
    });
  });

  it("builds vram and power configs", () => {
    const v = buildConfig(
      { ...defaultForm(), kind: "vram", vramPct: 80, vramPattern: "random", stopOnError: true },
      info
    );
    expect(v).toEqual({
      ok: true,
      config: {
        kind: "vram",
        targetPct: 80,
        pattern: "random",
        stopOnError: true,
        maxErrors: null,
        durationS: null,
      },
    });
    const p = buildConfig(
      {
        ...defaultForm(),
        kind: "power",
        threads: 12,
        dataset: "large",
        powerGpuProfile: "heavy",
        powerGpuIntensity: 90,
      },
      info
    );
    expect(p).toEqual({
      ok: true,
      config: {
        kind: "power",
        cpuThreads: 12,
        cpuIsa: "auto",
        cpuDataset: "large",
        cpuWorkload: "extreme",
        cpuFftPreset: "small",
        gpuProfile: "heavy",
        gpuIntensity: 90,
        stopOnError: false,
        maxErrors: null,
        durationS: null,
      },
    });
  });

  it("validates ranges and allows up to 100% RAM/VRAM", () => {
    expect(buildConfig({ ...defaultForm(), kind: "cpu", threads: 0 }, info).ok).toBe(false);
    expect(buildConfig({ ...defaultForm(), kind: "cpu", threads: 17 }, info).ok).toBe(false);
    expect(buildConfig({ ...defaultForm(), kind: "cpu", physicalCoresOnly: true, threads: 10 }, info).ok).toBe(false);
    expect(buildConfig({ ...defaultForm(), kind: "cpu", isa: "avx512" }, info).ok).toBe(false);
    expect(buildConfig({ ...defaultForm(), kind: "cpu", mode: "variable", cpuVarMin: 80, cpuVarMax: 40 }, info).ok).toBe(
      false
    );
    expect(buildConfig({ ...defaultForm(), kind: "memory", memPct: 95 }, info).ok).toBe(true);
    expect(buildConfig({ ...defaultForm(), kind: "memory", memPct: 100 }, info).ok).toBe(true);
    expect(buildConfig({ ...defaultForm(), kind: "memory", memPct: 101 }, info).ok).toBe(false);
    expect(buildConfig({ ...defaultForm(), kind: "memory", memPct: 5 }, info).ok).toBe(false);
    expect(buildConfig({ ...defaultForm(), kind: "vram", vramPct: 95 }, info).ok).toBe(true);
    expect(buildConfig({ ...defaultForm(), kind: "vram", vramPct: 100 }, info).ok).toBe(true);
    expect(buildConfig({ ...defaultForm(), kind: "vram", vramPct: 101 }, info).ok).toBe(false);
    expect(buildConfig({ ...defaultForm(), kind: "gpu", gpuIntensity: 5 }, info).ok).toBe(false);
    expect(
      buildConfig({ ...defaultForm(), kind: "gpu", gpuMode: "variable", gpuVarStepS: 0 }, info).ok
    ).toBe(false);
    expect(buildConfig({ ...defaultForm(), kind: "power", powerGpuIntensity: 5 }, info).ok).toBe(false);
    expect(buildConfig({ ...defaultForm(), kind: "cpu", durationMin: -1 }, info).ok).toBe(false);
  });

  it("defaults to Absolut 32-step 100% OC targets (all logical threads) and supports Standard profile switching", () => {
    const r = buildConfig({ ...defaultForm(), kind: "memory", durationMin: null }, info);
    expect(r.ok && r.config).toMatchObject({
      kind: "memory",
      targetPct: 100,
      profile: "absolut",
      pattern: "all",
      cycles: null,
      durationS: null,
      threads: 16,
      maxErrors: null,
    });
    const abs3 = buildConfig({ ...defaultForm(), kind: "memory", memCycles: 3 }, info);
    expect(abs3.ok && abs3.config).toMatchObject({ kind: "memory", profile: "absolut", cycles: 3 });
    const comb = buildConfig(
      { ...defaultForm(), kind: "combined", cpuWorkload: "fftTorture", combinedFftPreset: "large" },
      info
    );
    expect(comb.ok && comb.config).toMatchObject({
      kind: "combined",
      cpuWorkload: "fftTorture",
      cpuFftPreset: "large",
      memProfile: "absolut",
      memTargetPct: 100,
    });
    const std = buildConfig({ ...defaultForm(), kind: "memory", memProfile: "standard", memPattern: "movingInversions" }, info);
    expect(std.ok && std.config).toMatchObject({
      kind: "memory",
      profile: "standard",
      pattern: "movingInversions",
      threads: 8,
      cycles: null,
    });
    const v = buildConfig({ ...defaultForm(), kind: "vram" }, info);
    expect(v.ok && v.config).toMatchObject({ kind: "vram", targetPct: 100, pattern: "all" });
  });
});

describe("mergeStressStatus", () => {
  const base: StressStatus = {
    phase: "stopped",
    kind: "cpu",
    config: null,
    abortReason: null,
    message: null,
    elapsedS: 1,
    durationS: null,
    timeCapS: null,
    thermalProtection: true,
    cpuThreads: 16,
    memThreads: 0,
    isaUsed: "AVX2 + FMA",
    memTargetBytes: 0,
    memAllocatedBytes: 0,
    vramTargetBytes: 0,
    vramAllocatedBytes: 0,
    gflops: 0,
    gpuGflops: 0,
    memMbps: 0,
    totalErrors: 0,
    gpuErrors: 0,
    coreErrors: [],
    loadLevel: 0,
    workersAlive: 0,
  };

  it("ignores stale stopping response after terminal event but permits starting a new test", () => {
    const staleStopping: StressStatus = { ...base, phase: "stopping", workersAlive: 4 };
    expect(mergeStressStatus(base, staleStopping).phase).toBe("stopped");

    const aborted: StressStatus = { ...base, phase: "aborted", abortReason: "user" };
    expect(mergeStressStatus(aborted, staleStopping).phase).toBe("aborted");

    const nextRun: StressStatus = { ...base, phase: "starting", elapsedS: 0 };
    expect(mergeStressStatus(base, nextRun).phase).toBe("starting");
  });
});
