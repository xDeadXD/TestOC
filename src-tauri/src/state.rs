//! Application state shared by IPC commands and the background threads, plus the telemetry loop
//! (with the safety watchdog) and the dead-man's-switch heartbeat thread.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use tauri::{AppHandle, Emitter};

use crate::model::*;
use crate::session::SessionRecorder;
use crate::sidecar::SidecarHub;
use crate::stress::manager::{StressEvents, StressManager};
use crate::stress::safety::{self, WatchEvent, WatchInput, Watchdog};
use crate::telemetry::TelemetryEngine;
use crate::win;

pub const EV_SAMPLE: &str = "telemetry:sample";
pub const EV_STRESS_STATUS: &str = "stress:status";
pub const EV_STRESS_ERROR: &str = "stress:error";
pub const EV_ALERT: &str = "safety:alert";
pub const EV_SIDECAR: &str = "sidecar:status";

pub const DEFAULT_POLL_MS: u32 = 500;
pub const HEARTBEAT_STALL: Duration = Duration::from_secs(8);

pub struct AppShared {
    pub info: SystemInfo,
    pub latest: RwLock<Option<TelemetrySample>>,
    pub stress: StressManager,
    pub limits: RwLock<SafetyLimits>,
    pub recorder: Mutex<SessionRecorder>,
    pub sidecar: Option<SidecarHub>,
    pub sidecar_status: RwLock<SidecarStatus>,
    pub poll_ms: AtomicU32,
    pub last_tick_ms: AtomicU64,
    pub shutdown: AtomicBool,
}

/// Bridges stress-engine events to Tauri events (and the session recorder).
pub struct TauriEvents {
    pub app: AppHandle,
    pub shared: RwLock<Option<std::sync::Weak<AppShared>>>,
}

impl TauriEvents {
    fn shared(&self) -> Option<Arc<AppShared>> {
        self.shared.read().as_ref().and_then(|w| w.upgrade())
    }
}

impl StressEvents for TauriEvents {
    fn status(&self, s: &StressStatus) {
        let _ = self.app.emit(EV_STRESS_STATUS, s);
        if matches!(
            s.phase,
            StressPhase::Completed
                | StressPhase::Stopped
                | StressPhase::Aborted
                | StressPhase::Failed
        ) && s.workers_alive == 0
        {
            if let Some(sh) = self.shared() {
                let mut rec = sh.recorder.lock();
                rec.record_run(s);
                rec.stop_auto();
            }
        }
    }
    fn worker_error(&self, e: &WorkerError) {
        let _ = self.app.emit(EV_STRESS_ERROR, e);
    }
    fn alert(&self, a: &SafetyAlert) {
        let _ = self.app.emit(EV_ALERT, a);
        if let Some(sh) = self.shared() {
            sh.recorder.lock().add_alert(a.message.clone());
        }
    }
}

impl AppShared {
    pub fn thermal_ok(&self) -> bool {
        self.sidecar_status.read().state == SidecarState::Ok
            && self
                .latest
                .read()
                .as_ref()
                .is_some_and(|s| s.cpu.package_temp_c.is_some())
    }

    pub fn set_sidecar_status(&self, app: &AppHandle, st: SidecarStatus) {
        let changed = {
            let cur = self.sidecar_status.read();
            cur.state != st.state || cur.message != st.message
        };
        if changed {
            *self.sidecar_status.write() = st.clone();
            let _ = app.emit(EV_SIDECAR, &st);
        }
    }
}

/// The 500 ms telemetry loop. Runs on a dedicated ABOVE_NORMAL thread.
pub fn telemetry_loop(app: AppHandle, shared: Arc<AppShared>, mut engine: TelemetryEngine) {
    win::set_current_thread_above_normal();
    let mut watchdog = Watchdog::default();
    let mut buf = Vec::new();
    let mut last_run_ptr: usize = 0;
    let mut next = Instant::now();
    while !shared.shutdown.load(Ordering::Acquire) {
        // --- sensors ---
        let (sc_status, frame) = match &shared.sidecar {
            Some(hub) => hub.poll(&mut buf),
            None => (
                SidecarStatus {
                    state: SidecarState::Error,
                    message: Some("shared memory unavailable".into()),
                    writer_pid: None,
                },
                None,
            ),
        };
        let sc_state = sc_status.state;
        shared.set_sidecar_status(&app, sc_status);
        let run = shared.stress.current();
        let snapshot = shared.stress.snapshot();
        let sample = engine.poll(frame.as_ref(), sc_state, snapshot);

        // --- watchdog ---
        let run_ptr = run.as_ref().map(|r| Arc::as_ptr(r) as usize).unwrap_or(0);
        if run_ptr != last_run_ptr {
            watchdog.reset_run();
            last_run_ptr = run_ptr;
        }
        let active = run
            .as_ref()
            .is_some_and(|r| !r.is_done() && !r.is_aborted());
        let gpu_throttles: Vec<String> = sample
            .gpus
            .iter()
            .flat_map(|g| g.throttle_reasons.iter().cloned())
            .collect();
        let input = WatchInput {
            cpu_temp_c: sample.cpu.package_temp_c,
            gpu_temp_c: sample.gpus.iter().filter_map(|g| g.temp_c).reduce(f32::max),
            mem_used_pct: sample.memory.used_pct,
            mem_available_bytes: sample.memory.available_bytes,
            stress_active: active,
            stress_kind: run.as_ref().map(|r| r.kind),
            run_has_thermal: run.as_ref().is_some_and(|r| r.thermal_protection),
            sidecar_fresh: sc_state == SidecarState::Ok,
            severe_throttles: safety::severe_throttle_reasons(
                &sample.cpu.throttle_flags,
                &gpu_throttles,
            ),
        };
        let limits = *shared.limits.read();
        for ev in watchdog.evaluate(&input, &limits) {
            let ts = now_ms();
            let alert = match ev {
                WatchEvent::Alert(a) => a.into_alert(AlertAction::None, ts),
                WatchEvent::Abort(reason, a) => {
                    // Lock-free: flips the run's atomics directly.
                    let first = run.as_ref().is_some_and(|r| r.trigger_abort(reason));
                    if first {
                        log::warn!("watchdog abort: {reason:?} ({})", a.message);
                        if let Some(r) = &run {
                            let _ = app.emit(EV_STRESS_STATUS, &r.status());
                        }
                    }
                    a.into_alert(
                        if first {
                            AlertAction::Aborted
                        } else {
                            AlertAction::None
                        },
                        ts,
                    )
                }
            };
            let _ = app.emit(EV_ALERT, &alert);
            shared.recorder.lock().add_alert(alert.message.clone());
        }

        // --- publish ---
        shared.recorder.lock().push(&sample);
        let _ = app.emit(EV_SAMPLE, &sample);
        *shared.latest.write() = Some(sample);
        shared.last_tick_ms.store(now_ms(), Ordering::Release);

        // Fixed-rate scheduling: no drift, and skip ahead if we fell behind.
        let period = Duration::from_millis(shared.poll_ms.load(Ordering::Relaxed) as u64);
        next += period;
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        } else {
            next = now;
        }
    }
}

/// Dead-man's switch: if the telemetry loop stops ticking while a run is active, abort it.
pub fn heartbeat_loop(app: AppHandle, shared: Arc<AppShared>) {
    win::set_current_thread_above_normal();
    while !shared.shutdown.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(250));
        if shared.shutdown.load(Ordering::Acquire) {
            break;
        }
        let last = shared.last_tick_ms.load(Ordering::Acquire);
        if last == 0 {
            continue;
        }
        let age = now_ms().saturating_sub(last);
        let allowed =
            HEARTBEAT_STALL.as_millis() as u64 + shared.poll_ms.load(Ordering::Relaxed) as u64;
        if age > allowed {
            if let Some(r) = shared.stress.current() {
                if !r.is_done() && r.trigger_abort(AbortReason::SensorStall) {
                    log::error!("telemetry loop stalled for {age} ms; aborting stress");
                    let alert = SafetyAlert {
                        level: AlertLevel::Critical,
                        metric: AlertMetric::SensorStall,
                        value: age as f32 / 1000.0,
                        threshold: HEARTBEAT_STALL.as_secs_f32(),
                        action: AlertAction::Aborted,
                        timestamp_ms: now_ms(),
                        message: format!("Telemetry stalled for {:.1} s", age as f32 / 1000.0),
                    };
                    let _ = app.emit(EV_ALERT, &alert);
                    let _ = app.emit(EV_STRESS_STATUS, &r.status());
                }
            }
        }
    }
}

pub fn initial_sidecar_status() -> SidecarStatus {
    SidecarStatus {
        state: SidecarState::NotStarted,
        message: None,
        writer_pid: None,
    }
}

pub fn default_limits() -> SafetyLimits {
    safety::default_limits()
}
