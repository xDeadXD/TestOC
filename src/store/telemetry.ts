import { create } from "zustand";
import { SlidingColumns } from "../lib/ringBuffer";
import { ipc, subscribeAll } from "../lib/ipc";
import { isAppError } from "../lib/types";
import type {
  ExportResult,
  SafetyAlert,
  SafetyLimits,
  SessionInfo,
  SidecarStatus,
  StressConfig,
  StressStatus,
  SystemInfo,
  TelemetrySample,
  WorkerError,
} from "../lib/types";

export type WindowMinutes = 1 | 5 | 15;

/** 15 min at 2 Hz = 1800 samples, stored outside React state for zero-rerender uPlot updates. */
export const historyColumns = new SlidingColumns(1800);

type TickListener = () => void;
const tickListeners = new Set<TickListener>();

export function subscribeChartTicks(fn: TickListener): () => void {
  tickListeners.add(fn);
  return () => tickListeners.delete(fn);
}

function pushSampleToColumns(s: TelemetrySample): void {
  const row: Record<string, number | null> = {
    t: s.timestampMs / 1000,
    cpuLoad: s.cpu.totalLoadPct,
    cpuTemp: s.cpu.packageTempC,
    ccdTemp: s.cpu.ccdTempC,
    cpuPower: s.cpu.packagePowerW,
    cpuAvgMhz: s.cpu.avgEffectiveMhz,
    memPct: s.memory.usedPct,
    gpuTemp: s.gpus[0]?.tempC ?? null,
    gpuUtil: s.gpus[0]?.utilPct ?? null,
    gpuPower: s.gpus[0]?.powerW ?? null,
  };
  for (const c of s.cpu.cores) {
    row[`cLoad${c.id}`] = c.loadPct;
    row[`cMhz${c.id}`] = c.effectiveMhz;
  }
  historyColumns.push(row);
  for (const fn of tickListeners) fn();
}

export interface TelemetryStore {
  ready: boolean;
  info: SystemInfo | null;
  latest: TelemetrySample | null;
  stress: StressStatus | null;
  sidecar: SidecarStatus;
  limits: SafetyLimits;
  session: SessionInfo | null;
  pollMs: number;
  windowMin: WindowMinutes;
  /** Which per-core series are visible on CPU Load / Clock charts. */
  selectedCores: boolean[];
  sampleCount: number;
  alerts: SafetyAlert[];
  workerErrors: WorkerError[];
  bannerAlert: SafetyAlert | null;
  actionError: string | null;
  lastExport: ExportResult | null;
  noThermalConfig: StressConfig | null;

  init: () => Promise<() => void>;
  setWindowMin: (m: WindowMinutes) => void;
  setPollMs: (ms: number) => Promise<void>;
  toggleCore: (id: number) => void;
  setAllCores: (on: boolean) => void;
  startStress: (config: StressConfig, ackNoThermal?: boolean) => Promise<void>;
  confirmNoThermal: () => Promise<void>;
  cancelNoThermal: () => void;
  stopStress: () => Promise<void>;
  emergencyAbort: () => Promise<void>;
  retrySidecar: () => Promise<void>;
  updateLimits: (limits: SafetyLimits) => Promise<string | null>;
  startManualSession: (label?: string) => Promise<void>;
  stopManualSession: () => Promise<void>;
  exportCsvTo: (csvPath: string) => Promise<ExportResult | null>;
  dismissBanner: () => void;
  clearActionError: () => void;
}

const TERMINAL_PHASES = new Set(["completed", "stopped", "aborted", "failed"]);

/** Guard against an out-of-order `stopping` IPC reply overwriting an already-received terminal event. */
export function mergeStressStatus(prev: StressStatus | null, next: StressStatus): StressStatus {
  if (prev && TERMINAL_PHASES.has(prev.phase) && next.phase === "stopping" && prev.kind === next.kind) {
    return prev;
  }
  return next;
}

const DEFAULT_LIMITS: SafetyLimits = {
  cpuWarnC: 90,
  cpuAbortC: 100,
  gpuWarnC: 88,
  gpuAbortC: 95,
  memMaxPct: 100,
  thermalAbortEnabled: true,
};

export const useTelemetryStore = create<TelemetryStore>((set, get) => ({
  ready: false,
  info: null,
  latest: null,
  stress: null,
  sidecar: { state: "notStarted", message: null, writerPid: null },
  limits: DEFAULT_LIMITS,
  session: null,
  pollMs: 500,
  windowMin: 5,
  selectedCores: [],
  sampleCount: 0,
  alerts: [],
  workerErrors: [],
  bannerAlert: null,
  actionError: null,
  lastExport: null,
  noThermalConfig: null,

  async init() {
    const [info, latest, stress, sidecar, limits, session] = await Promise.all([
      ipc.getSystemInfo(),
      ipc.getLatestSample(),
      ipc.getStressStatus(),
      ipc.getSidecarStatus(),
      ipc.getSafetyLimits(),
      ipc.getSessionInfo(),
    ]);
    const selectedCores = Array.from({ length: info.logicalCores }, (_, i) => i < 4);
    if (latest) pushSampleToColumns(latest);
    set({
      ready: true,
      info,
      latest,
      stress,
      sidecar,
      limits,
      session,
      selectedCores,
      sampleCount: latest ? 1 : 0,
    });

    const unlisten = await subscribeAll({
      onSample: (s) => {
        pushSampleToColumns(s);
        let reconcile = false;
        set((st) => {
          const cur = st.stress;
          // Reconcile if backend snapshot shows the run finished while UI still shows an active/stopping phase.
          const needsTerminalSync =
            cur != null &&
            !TERMINAL_PHASES.has(cur.phase) &&
            TERMINAL_PHASES.has(s.stress.phase);
          if (needsTerminalSync) {
            reconcile = true;
          }
          const syncedStress =
            cur == null
              ? cur
              : needsTerminalSync
                ? { ...cur, phase: s.stress.phase, workersAlive: 0, totalErrors: Math.max(cur.totalErrors, s.stress.errors) }
                : s.stress.errors > cur.totalErrors
                  ? { ...cur, totalErrors: s.stress.errors }
                  : cur;
          return {
            latest: s,
            sampleCount: st.sampleCount + 1,
            session: st.session?.recording ? { ...st.session, samples: st.session.samples + 1 } : st.session,
            stress: syncedStress,
          };
        });
        if (reconcile) {
          void ipc.getStressStatus().then((full) =>
            set((st) => ({ stress: mergeStressStatus(st.stress, full) }))
          );
        }
      },
      onStressStatus: (s) => {
        set((st) => ({ stress: mergeStressStatus(st.stress, s) }));
        // Refresh session info whenever a run enters running or a terminal phase.
        if (s.phase === "running" || TERMINAL_PHASES.has(s.phase)) {
          void ipc.getSessionInfo().then((session) => set({ session }));
        }
      },
      onStressError: (e) => {
        set((st) => ({
          workerErrors: [e, ...st.workerErrors].slice(0, 100),
          stress:
            st.stress && e.kind !== "allocFailed" && e.kind !== "gpuDevice"
              ? { ...st.stress, totalErrors: Math.max(st.stress.totalErrors, st.workerErrors.length + 1) }
              : st.stress,
        }));
      },
      onAlert: (a) => {
        set((st) => ({
          alerts: [a, ...st.alerts].slice(0, 50),
          bannerAlert: a,
        }));
      },
      onSidecar: (sc) => set({ sidecar: sc }),
    });
    return unlisten;
  },

  setWindowMin(m) {
    set({ windowMin: m });
    for (const fn of tickListeners) fn();
  },

  async setPollMs(ms) {
    const applied = await ipc.setPollInterval(ms);
    set({ pollMs: applied });
  },

  toggleCore(id) {
    set((st) => {
      const next = [...st.selectedCores];
      if (id >= 0 && id < next.length) next[id] = !next[id];
      return { selectedCores: next };
    });
  },

  setAllCores(on) {
    set((st) => ({ selectedCores: st.selectedCores.map(() => on) }));
  },

  async startStress(config, ackNoThermal = false) {
    set({ actionError: null });
    try {
      const stress = await ipc.startStress(config, ackNoThermal);
      const session = await ipc.getSessionInfo();
      set({ stress, session, noThermalConfig: null, workerErrors: [], bannerAlert: null });
    } catch (e) {
      if (isAppError(e) && e.code === "thermalUnavailable") {
        set({ noThermalConfig: config });
      } else {
        set({ actionError: isAppError(e) ? e.message : String(e) });
      }
    }
  },

  async confirmNoThermal() {
    const cfg = get().noThermalConfig;
    if (!cfg) return;
    await get().startStress(cfg, true);
  },

  cancelNoThermal() {
    set({ noThermalConfig: null });
  },

  async stopStress() {
    set({ actionError: null });
    try {
      const stress = await ipc.stopStress();
      set((st) => ({ stress: mergeStressStatus(st.stress, stress) }));
    } catch (e) {
      set({ actionError: isAppError(e) ? e.message : String(e) });
    }
  },

  async emergencyAbort() {
    set({ actionError: null });
    try {
      const stress = await ipc.emergencyAbort();
      set((st) => ({ stress: mergeStressStatus(st.stress, stress) }));
    } catch (e) {
      set({ actionError: isAppError(e) ? e.message : String(e) });
    }
  },

  async retrySidecar() {
    set({ actionError: null });
    try {
      const sidecar = await ipc.launchSidecar();
      set({ sidecar });
    } catch (e) {
      set({ actionError: isAppError(e) ? e.message : String(e) });
    }
  },

  async updateLimits(limits) {
    try {
      const applied = await ipc.setSafetyLimits(limits);
      set({ limits: applied, actionError: null });
      return null;
    } catch (e) {
      return isAppError(e) ? e.message : String(e);
    }
  },

  async startManualSession(label) {
    const session = await ipc.sessionStart(label);
    set({ session, actionError: null });
  },

  async stopManualSession() {
    try {
      const session = await ipc.sessionStop();
      set({ session, actionError: null });
    } catch (e) {
      set({ actionError: isAppError(e) ? e.message : String(e) });
    }
  },

  async exportCsvTo(csvPath) {
    set({ actionError: null });
    try {
      const res = await ipc.exportSession(csvPath);
      set({ lastExport: res });
      return res;
    } catch (e) {
      set({ actionError: isAppError(e) ? e.message : String(e) });
      return null;
    }
  },

  dismissBanner() {
    set({ bannerAlert: null });
  },

  clearActionError() {
    set({ actionError: null });
  },
}));
