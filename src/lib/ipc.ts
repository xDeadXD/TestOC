import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
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
} from "./types";

/** Event names (keep in sync with src-tauri/src/state.rs). */
export const EV = {
  sample: "telemetry:sample",
  stressStatus: "stress:status",
  stressError: "stress:error",
  alert: "safety:alert",
  sidecar: "sidecar:status",
} as const;

/** Typed wrappers over the Tauri IPC command schema. */
export const ipc = {
  getSystemInfo: () => invoke<SystemInfo>("get_system_info"),
  getLatestSample: () => invoke<TelemetrySample | null>("get_latest_sample"),
  setPollInterval: (ms: number) => invoke<number>("set_poll_interval", { ms }),
  getSidecarStatus: () => invoke<SidecarStatus>("get_sidecar_status"),
  launchSidecar: () => invoke<SidecarStatus>("launch_sidecar"),
  startStress: (config: StressConfig, acknowledgeNoThermal = false) =>
    invoke<StressStatus>("start_stress", { config, acknowledgeNoThermal }),
  stopStress: () => invoke<StressStatus>("stop_stress"),
  emergencyAbort: () => invoke<StressStatus>("emergency_abort"),
  getStressStatus: () => invoke<StressStatus>("get_stress_status"),
  getSafetyLimits: () => invoke<SafetyLimits>("get_safety_limits"),
  setSafetyLimits: (limits: SafetyLimits) => invoke<SafetyLimits>("set_safety_limits", { limits }),
  sessionStart: (label?: string) => invoke<SessionInfo>("session_start", { label: label ?? null }),
  sessionStop: () => invoke<SessionInfo>("session_stop"),
  getSessionInfo: () => invoke<SessionInfo | null>("get_session_info"),
  exportSession: (csvPath: string) => invoke<ExportResult>("export_session", { csvPath }),
};

export interface Handlers {
  onSample: (s: TelemetrySample) => void;
  onStressStatus: (s: StressStatus) => void;
  onStressError: (e: WorkerError) => void;
  onAlert: (a: SafetyAlert) => void;
  onSidecar: (s: SidecarStatus) => void;
}

export async function subscribeAll(h: Handlers): Promise<UnlistenFn> {
  const fns = await Promise.all([
    listen<TelemetrySample>(EV.sample, (e) => h.onSample(e.payload)),
    listen<StressStatus>(EV.stressStatus, (e) => h.onStressStatus(e.payload)),
    listen<WorkerError>(EV.stressError, (e) => h.onStressError(e.payload)),
    listen<SafetyAlert>(EV.alert, (e) => h.onAlert(e.payload)),
    listen<SidecarStatus>(EV.sidecar, (e) => h.onSidecar(e.payload)),
  ]);
  return () => fns.forEach((f) => f());
}

export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}
