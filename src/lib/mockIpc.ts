import { mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import { emit } from "@tauri-apps/api/event";
import { EV } from "./ipc";
import type {
  AbortReason,
  ExportResult,
  SafetyAlert,
  SafetyLimits,
  SessionInfo,
  SidecarState,
  SidecarStatus,
  StressConfig,
  StressStatus,
  SystemInfo,
  TelemetrySample,
  WorkerError,
} from "./types";

export interface MockController {
  calls: Array<{ cmd: string; args: unknown }>;
  setCpuTemp: (c: number | null) => void;
  setSidecarState: (state: SidecarState, message?: string | null) => void;
  setGpuThrottles: (reasons: string[]) => void;
  emitAlert: (alert: SafetyAlert, abortReason?: AbortReason) => void;
  emitError: (err: WorkerError) => void;
  setSavePath: (path: string | null) => void;
}

declare global {
  interface Window {
    __TESTOC_MOCK__?: MockController;
  }
}

const GB = 1024 ** 3;

const MOCK_INFO: SystemInfo = {
  appVersion: "0.1.0-mock",
  hostname: "TESTOC-MOCK",
  osVersion: "Windows 11 Pro (Mock)",
  cpuBrand: "Demo 8-Core x86_64 Processor",
  physicalCores: 8,
  logicalCores: 16,
  baseMhz: 3400,
  isa: { avx2: true, fma: true, avx512f: false },
  totalMemoryBytes: 16 * GB,
  gpus: [{ index: 0, name: "Demo Discrete GPU", vramTotalBytes: 12 * GB }],
  gpuStressAdapter: "Demo Discrete GPU (Vulkan)",
  gpuVramTotalBytes: 12 * GB,
  pdhClocksAvailable: true,
  nvmlAvailable: true,
};

function idleStatus(): StressStatus {
  return {
    phase: "idle",
    kind: null,
    config: null,
    abortReason: null,
    message: null,
    elapsedS: 0,
    durationS: null,
    timeCapS: null,
    thermalProtection: false,
    cpuThreads: 0,
    memThreads: 0,
    isaUsed: null,
    memTargetBytes: 0,
    memAllocatedBytes: 0,
    vramTargetBytes: 0,
    vramAllocatedBytes: 0,
    gflops: 0,
    gpuGflops: 0,
    memMbps: 0,
    totalErrors: 0,
    gpuErrors: 0,
    coreErrors: new Array(16).fill(0),
    loadLevel: 0,
    workersAlive: 0,
  };
}

let installed = false;

export function installMockIpc(): void {
  if (installed) return;
  installed = true;
  mockWindows("main");

  const calls: Array<{ cmd: string; args: unknown }> = [];
  let pollMs = 500;
  let seq = 0;
  let forcedCpuTemp: number | null = null;
  let gpuThrottles: string[] = [];
  let savePath: string | null = "C:\\Temp\\testoc-mock.csv";
  let sidecar: SidecarStatus = { state: "ok", message: null, writerPid: 4242 };
  let limits: SafetyLimits = {
    cpuWarnC: 90,
    cpuAbortC: 100,
    gpuWarnC: 88,
    gpuAbortC: 95,
    memMaxPct: 100,
    thermalAbortEnabled: true,
  };
  let stress: StressStatus = idleStatus();
  let runStartMs = 0;
  let session: SessionInfo | null = null;
  let sessionSeq = 0;
  let latest: TelemetrySample | null = null;
  let warnedTemp = false;

  const makeSample = (): TelemetrySample => {
    seq++;
    const now = Date.now();
    const t = seq * 0.5;
    const active = stress.phase === "running" || stress.phase === "starting";
    if (active) {
      stress = {
        ...stress,
        phase: "running",
        elapsedS: (now - runStartMs) / 1000,
        gflops: stress.kind === "cpu" || stress.kind === "combined" || stress.kind === "power" ? 412 + Math.sin(t) * 14 : 0,
        gpuGflops: stress.kind === "gpu" || stress.kind === "power" ? 18200 + Math.cos(t) * 450 : 0,
        memMbps: stress.kind === "memory" || stress.kind === "combined" ? 31400 : stress.kind === "vram" ? 582000 : 0,
      };
    }

    const cpuBusy = active && (stress.kind === "cpu" || stress.kind === "combined" || stress.kind === "power");
    const gpuBusy = active && (stress.kind === "gpu" || stress.kind === "vram" || stress.kind === "power");
    const memBusy = active && (stress.kind === "memory" || stress.kind === "combined");
    const vramBusy = active && stress.kind === "vram";

    const gpuProfile =
      stress.config?.kind === "gpu"
        ? stress.config.profile
        : stress.config?.kind === "power"
          ? stress.config.gpuProfile
          : "extreme";
    const gpuIntensity =
      stress.config?.kind === "gpu"
        ? stress.config.mode === "variable"
          ? stress.config.variableMinPct +
            Math.round(((Math.sin(t * 0.4) + 1) / 2) * (stress.config.variableMaxPct - stress.config.variableMinPct))
          : stress.config.intensity
        : stress.config?.kind === "power"
          ? stress.config.gpuIntensity
          : 100;
    const profileWatts = gpuProfile === "extreme" ? 248 : gpuProfile === "heavy" ? 215 : 155;
    const gpuPower = gpuBusy ? (vramBusy ? 165 : (profileWatts * gpuIntensity) / 100) : 14.5;
    const profileBaseTemp = gpuProfile === "extreme" ? 73 : gpuProfile === "heavy" ? 68 : 61;

    const cpuLoad = cpuBusy ? 98.5 + Math.sin(t) * 1.2 : 8 + Math.sin(t * 0.7) * 4;
    const baseTemp = cpuBusy ? 74 + Math.min(12, stress.elapsedS * 0.8) : 44 + Math.sin(t * 0.3) * 2;
    const cpuTemp = sidecar.state === "ok" ? (forcedCpuTemp ?? baseTemp) : null;
    const cpuPower = sidecar.state === "ok" ? (cpuBusy ? 102 + Math.sin(t) * 4 : 26 + Math.sin(t) * 3) : null;
    const cores = Array.from({ length: 16 }, (_, i) => {
      const load = Math.max(0, Math.min(100, cpuLoad + Math.sin(t + i * 0.9) * (cpuBusy ? 1.5 : 8)));
      const mhz = cpuBusy ? 4425 + Math.sin(t + i) * 40 : 3550 + Math.cos(t + i * 0.6) * 450;
      return {
        id: i,
        loadPct: load,
        effectiveMhz: mhz,
        clockSource: sidecar.state === "ok" ? ("lhm" as const) : ("pdh" as const),
        vid: sidecar.state === "ok" ? 1.24 + Math.sin(t + i) * 0.03 : null,
      };
    });
    const mhzList = cores.map((c) => c.effectiveMhz);
    const usedMem = (memBusy ? 12.8 : 6.1) * GB + Math.sin(t) * 80_000_000;
    const gpuTemp = gpuBusy ? profileBaseTemp + Math.min(12, stress.elapsedS * 0.5) : 39 + Math.sin(t * 0.2) * 1.5;
    const activeThrottles =
      gpuThrottles.length > 0 ? [...gpuThrottles] : gpuBusy && gpuPower >= 220 ? ["Power Cap"] : [];

    const sample: TelemetrySample = {
      seq,
      timestampMs: now,
      cpu: {
        totalLoadPct: Math.max(0, Math.min(100, cpuLoad)),
        avgEffectiveMhz: mhzList.reduce((a, b) => a + b, 0) / mhzList.length,
        maxEffectiveMhz: Math.max(...mhzList),
        packageTempC: cpuTemp,
        ccdTempC: cpuTemp != null ? cpuTemp - 2.4 : null,
        packagePowerW: cpuPower,
        coreVoltageV: sidecar.state === "ok" ? 1.26 : null,
        throttleFlags: cpuTemp != null && cpuTemp >= 90 ? ["Thermal Limit"] : [],
        cores,
      },
      board:
        sidecar.state === "ok"
          ? {
              temps: [
                { name: "System", value: 36.5 },
                { name: "VRM MOS", value: cpuBusy ? 58.0 : 41.0 },
                { name: "Chipset", value: 48.5 },
              ],
              fans: [
                { name: "CPU Fan", value: cpuBusy ? 1680 : 940 },
                { name: "Chassis #1", value: 890 },
              ],
            }
          : { temps: [], fans: [] },
      memory: {
        totalBytes: 16 * GB,
        usedBytes: usedMem,
        availableBytes: 16 * GB - usedMem,
        usedPct: (usedMem / (16 * GB)) * 100,
        swapTotalBytes: 4 * GB,
        swapUsedBytes: 0.4 * GB,
      },
      gpus: [
        {
          index: 0,
          name: "Demo Discrete GPU",
          tempC: gpuTemp,
          utilPct: gpuBusy ? 99 : 4 + Math.abs(Math.sin(t)) * 5,
          memUtilPct: vramBusy ? 94 : gpuBusy ? 35 : 8,
          coreMhz: gpuBusy ? 2745 : 210,
          memMhz: gpuBusy ? 14000 : 405,
          powerW: gpuPower,
          powerLimitW: 300,
          vramUsedBytes: (vramBusy ? 10.1 : gpuBusy ? 1.4 : 0.9) * GB,
          vramTotalBytes: 12 * GB,
          fanPct: gpuBusy ? 62 : 0,
          tempSlowdownC: 90,
          throttleReasons: activeThrottles,
        },
      ],
      stress: {
        phase: stress.phase,
        kind: stress.kind,
        elapsedS: stress.elapsedS,
        gflops: stress.gflops,
        gpuGflops: stress.gpuGflops,
        memMbps: stress.memMbps,
        errors: stress.totalErrors,
        loadLevel: active ? gpuIntensity : 0,
      },
      sidecar: sidecar.state,
      pollDurationUs: 4200,
    };

    if (session?.recording) {
      session = { ...session, samples: session.samples + 1 };
    }
    latest = sample;

    if (cpuTemp != null && cpuTemp >= limits.cpuWarnC && !warnedTemp) {
      warnedTemp = true;
      void emit(EV.alert, {
        level: "warning",
        metric: "cpuTemp",
        value: cpuTemp,
        threshold: limits.cpuWarnC,
        action: "none",
        timestampMs: now,
        message: `CPU temperature ${cpuTemp.toFixed(1)} °C above warning ${limits.cpuWarnC.toFixed(0)} °C`,
      } satisfies SafetyAlert);
    } else if (cpuTemp != null && cpuTemp < limits.cpuWarnC - 2) {
      warnedTemp = false;
    }

    return sample;
  };

  let timer: ReturnType<typeof setInterval> | null = null;
  const restartTimer = () => {
    if (timer) clearInterval(timer);
    timer = setInterval(() => {
      const s = makeSample();
      void emit(EV.sample, s);
      if (stress.phase === "running") void emit(EV.stressStatus, stress);
    }, pollMs);
  };

  mockIPC(
    (cmd, args) => {
      calls.push({ cmd, args });
      const a = (args ?? {}) as Record<string, unknown>;
      switch (cmd) {
        case "get_system_info":
          return MOCK_INFO;
        case "get_latest_sample":
          return latest ?? makeSample();
        case "set_poll_interval": {
          const ms = Math.max(250, Math.min(5000, Number(a.ms ?? 500)));
          pollMs = ms;
          restartTimer();
          return ms;
        }
        case "get_sidecar_status":
          return sidecar;
        case "launch_sidecar":
          sidecar = { state: "ok", message: null, writerPid: 4243 };
          void emit(EV.sidecar, sidecar);
          return sidecar;
        case "get_stress_status":
          return stress;
        case "start_stress": {
          const cfg = a.config as StressConfig;
          const ack = Boolean(a.acknowledgeNoThermal);
          const loadsCpu = cfg.kind === "cpu" || cfg.kind === "combined" || cfg.kind === "power";
          if (loadsCpu && sidecar.state !== "ok" && !ack) {
            return Promise.reject({
              code: "thermalUnavailable",
              message: "CPU thermal protection is unavailable (sensor sidecar not running); acknowledgement required",
            });
          }
          runStartMs = Date.now();
          sessionSeq++;
          if (!session?.recording) {
            session = {
              id: sessionSeq,
              label: `stress-mock-${sessionSeq}`,
              startedMs: runStartMs,
              stoppedMs: null,
              samples: 0,
              recording: true,
              auto: true,
            };
          }
          stress = {
            ...idleStatus(),
            phase: "running",
            kind: cfg.kind,
            config: cfg,
            durationS: cfg.durationS,
            timeCapS: loadsCpu && sidecar.state !== "ok" ? 600 : null,
            thermalProtection: sidecar.state === "ok",
            cpuThreads: cfg.kind === "cpu" ? (cfg.threads ?? 16) : cfg.kind === "combined" || cfg.kind === "power" ? (cfg.cpuThreads ?? 16) : 0,
            memThreads: cfg.kind === "memory" ? (cfg.threads ?? (cfg.profile === "absolut" ? 16 : 8)) : cfg.kind === "combined" ? 2 : 0,
            memTargetBytes: cfg.kind === "memory" ? Math.round((10 * GB * cfg.targetPct) / 100) : cfg.kind === "combined" ? Math.round((10 * GB * cfg.memTargetPct) / 100) : 0,
            memAllocatedBytes: cfg.kind === "memory" ? Math.round((10 * GB * cfg.targetPct) / 100) : cfg.kind === "combined" ? Math.round((10 * GB * cfg.memTargetPct) / 100) : 0,
            vramTargetBytes: cfg.kind === "vram" ? Math.round((11 * GB * cfg.targetPct) / 100) : 0,
            vramAllocatedBytes: cfg.kind === "vram" ? Math.round((11 * GB * cfg.targetPct) / 100) : 0,
            message:
              cfg.kind === "memory" && cfg.profile === "absolut"
                ? `Absolut ${cfg.cycles ? `C1/${cfg.cycles}` : "C1"} · Step 1/32 · T1 SimpleTest M2 (LCG 0x7777/0x3333) (1/36)`
                : cfg.kind === "cpu" && cfg.mode === "coreCycler"
                  ? `Core Cycler · Core 1/${cfg.threads ?? 8} (Peak Boost)`
                  : cfg.kind === "cpu" && cfg.workload === "fftTorture"
                    ? `${cfg.fftPreset} FFTs · 64K L2 (AVX2 + FMA) · roundoff 0.00001`
                    : null,
            isaUsed:
              cfg.kind === "gpu"
                ? cfg.profile === "extreme"
                  ? cfg.mode === "variable"
                    ? "Vulkan 3D Extreme (Adaptive)"
                    : "Vulkan 3D Extreme"
                  : cfg.profile === "heavy"
                    ? cfg.mode === "variable"
                      ? "Vulkan Heavy Compute (Adaptive)"
                      : "Vulkan Heavy Compute"
                    : cfg.mode === "variable"
                      ? "Vulkan FP32 FMA (Adaptive)"
                      : "Vulkan FP32 FMA"
                : cfg.kind === "vram"
                  ? "Vulkan VRAM Absolut"
                  : cfg.kind === "memory"
                    ? cfg.profile === "absolut"
                      ? "Absolut 32-Step (AVX2 NT)"
                      : "RAM Standard"
                    : cfg.kind === "power"
                      ? cfg.gpuProfile === "extreme"
                        ? "AVX2+FMA + 3D Extreme"
                        : cfg.gpuProfile === "heavy"
                          ? "AVX2+FMA + GPU Heavy"
                          : "AVX2+FMA + GPU Standard"
                      : cfg.kind === "cpu" && cfg.workload === "fftTorture"
                        ? "FFT Torture (AVX2+FMA)"
                        : cfg.kind === "cpu" && cfg.workload === "nttPrime"
                          ? "NTT64 (Goldilocks)"
                          : "AVX2+FMA + FFT + NTT64",
            loadLevel: 100,
            workersAlive: cfg.kind === "gpu" || cfg.kind === "vram" ? 1 : 16,
          };
          void emit(EV.stressStatus, stress);
          return stress;
        }
        case "stop_stress":
          stress = {
            ...stress,
            phase: "stopped",
            workersAlive: 0,
            loadLevel: 0,
            gflops: 0,
            gpuGflops: 0,
            memMbps: 0,
            memAllocatedBytes: 0,
            vramAllocatedBytes: 0,
          };
          if (session?.auto && session.recording) session = { ...session, recording: false, stoppedMs: Date.now() };
          void emit(EV.stressStatus, stress);
          return stress;
        case "emergency_abort":
          stress = {
            ...stress,
            phase: "aborted",
            abortReason: "user",
            workersAlive: 0,
            loadLevel: 0,
            gflops: 0,
            gpuGflops: 0,
            memMbps: 0,
            memAllocatedBytes: 0,
            vramAllocatedBytes: 0,
          };
          if (session?.auto && session.recording) session = { ...session, recording: false, stoppedMs: Date.now() };
          void emit(EV.stressStatus, stress);
          return stress;
        case "get_safety_limits":
          return limits;
        case "set_safety_limits":
          limits = a.limits as SafetyLimits;
          return limits;
        case "session_start":
          sessionSeq++;
          session = {
            id: sessionSeq,
            label: (a.label as string | null) ?? `session-mock-${sessionSeq}`,
            startedMs: Date.now(),
            stoppedMs: null,
            samples: 0,
            recording: true,
            auto: false,
          };
          return session;
        case "session_stop":
          if (session) session = { ...session, recording: false, stoppedMs: Date.now() };
          return session;
        case "get_session_info":
          return session;
        case "export_session": {
          const p = String(a.csvPath ?? "session.csv");
          const sum = p.replace(/\.csv$/i, "") + ".summary.json";
          return { csvPath: p, summaryPath: sum, rows: Math.max(1, session?.samples ?? seq), bytes: 48_192 } satisfies ExportResult;
        }
        case "plugin:dialog|save":
          return savePath;
        default:
          return null;
      }
    },
    { shouldMockEvents: true }
  );

  window.__TESTOC_MOCK__ = {
    calls,
    setCpuTemp(c) {
      forcedCpuTemp = c;
      const s = makeSample();
      void emit(EV.sample, s);
    },
    setSidecarState(state, message = null) {
      sidecar = { state, message, writerPid: state === "ok" ? 4242 : null };
      void emit(EV.sidecar, sidecar);
      const s = makeSample();
      void emit(EV.sample, s);
    },
    setGpuThrottles(reasons) {
      gpuThrottles = [...reasons];
      const s = makeSample();
      void emit(EV.sample, s);
    },
    emitAlert(alert, abortReason) {
      if (abortReason) {
        stress = {
          ...stress,
          phase: "aborted",
          abortReason,
          workersAlive: 0,
          loadLevel: 0,
          gflops: 0,
          gpuGflops: 0,
          memMbps: 0,
        };
        void emit(EV.stressStatus, stress);
      }
      void emit(EV.alert, alert);
    },
    emitError(err) {
      const ce = [...stress.coreErrors];
      if (err.coreId != null && err.coreId < ce.length) ce[err.coreId]++;
      stress = { ...stress, totalErrors: stress.totalErrors + 1, coreErrors: ce };
      void emit(EV.stressError, err);
      void emit(EV.stressStatus, stress);
    },
    setSavePath(p) {
      savePath = p;
    },
  };

  // Seed initial sample & start 2 Hz stream.
  makeSample();
  restartTimer();
}
