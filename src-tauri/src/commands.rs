//! Tauri IPC commands. All return `Result<T, AppError>`; errors serialize as `{ code, message }`.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use tauri::{AppHandle, State};

use crate::error::{AppError, AppResult};
use crate::model::*;
use crate::session::{csv_export, summary};
use crate::state::AppShared;
use crate::stress::manager::StartContext;
use crate::stress::safety;

type S<'a> = State<'a, Arc<AppShared>>;

#[tauri::command]
pub fn get_system_info(state: S) -> SystemInfo {
    state.info.clone()
}

#[tauri::command]
pub fn get_latest_sample(state: S) -> Option<TelemetrySample> {
    state.latest.read().clone()
}

#[tauri::command]
pub fn set_poll_interval(state: S, ms: u32) -> u32 {
    let v = ms.clamp(250, 5000);
    state.poll_ms.store(v, Ordering::Relaxed);
    v
}

#[tauri::command]
pub fn get_sidecar_status(state: S) -> SidecarStatus {
    state.sidecar_status.read().clone()
}

/// Shows the UAC prompt again. Runs off the IPC thread because ShellExecuteExW blocks until the
/// user answers.
#[tauri::command]
pub async fn launch_sidecar(app: AppHandle, state: S<'_>) -> AppResult<SidecarStatus> {
    let shared = state.inner().clone();
    if shared.sidecar.is_none() {
        return Err(AppError::Internal("shared memory unavailable".into()));
    }
    let st = tauri::async_runtime::spawn_blocking(move || {
        let st = shared.sidecar.as_ref().unwrap().launch();
        shared.set_sidecar_status(&app, st.clone());
        st
    })
    .await
    .map_err(|e| AppError::Internal(e.to_string()))?;
    Ok(st)
}

#[tauri::command]
pub fn start_stress(
    state: S,
    config: StressConfig,
    acknowledge_no_thermal: Option<bool>,
) -> AppResult<StressStatus> {
    // Query live physical memory directly so startup timing never yields available_mem = 0.
    let (total_mem, available_mem) = crate::win::memory_status().unwrap_or_else(|| {
        let mem = state
            .latest
            .read()
            .as_ref()
            .map(|s| s.memory.clone())
            .unwrap_or_default();
        (
            if mem.total_bytes > 0 {
                mem.total_bytes
            } else {
                state.info.total_memory_bytes
            },
            mem.available_bytes,
        )
    });
    let limits = *state.limits.read();
    // Refuse to start if a relevant thermal abort limit is already exceeded (when enabled).
    if limits.thermal_abort_enabled {
        if let Some(s) = state.latest.read().as_ref() {
            if config.kind().loads_cpu() {
                if let Some(t) = s.cpu.package_temp_c.filter(|t| *t >= limits.cpu_abort_c) {
                    return Err(AppError::LimitExceeded(format!(
                        "CPU at {t:.1} °C ≥ {:.0} °C",
                        limits.cpu_abort_c
                    )));
                }
            }
            if let Some(t) = s
                .gpus
                .iter()
                .filter_map(|g| g.temp_c)
                .reduce(f32::max)
                .filter(|t| *t >= limits.gpu_abort_c)
            {
                return Err(AppError::LimitExceeded(format!(
                    "GPU at {t:.1} °C ≥ {:.0} °C",
                    limits.gpu_abort_c
                )));
            }
        }
    }
    // Live VRAM total & free from latest NVML sample (fallback handled inside StressManager).
    let (vram_total, vram_free) = state
        .latest
        .read()
        .as_ref()
        .and_then(|s| s.gpus.first())
        .map(|g| {
            (
                g.vram_total_bytes,
                g.vram_total_bytes
                    .zip(g.vram_used_bytes)
                    .map(|(t, u)| t.saturating_sub(u)),
            )
        })
        .unwrap_or((state.info.gpu_vram_total_bytes, None));
    let ctx = StartContext {
        physical_cores: state.info.physical_cores as usize,
        logical_cores: state.info.logical_cores as usize,
        isa: state.info.isa.clone(),
        total_mem,
        available_mem,
        vram_total,
        vram_free,
        thermal_ok: state.thermal_ok(),
        acknowledge_no_thermal: acknowledge_no_thermal.unwrap_or(false),
        limits,
    };
    // Auto-start a session before the run so the ramp-up is captured.
    let session = state.recorder.lock().start(None, true);
    match state.stress.start(config, ctx) {
        Ok(st) => Ok(st),
        Err(e) => {
            let mut rec = state.recorder.lock();
            if rec
                .info()
                .is_some_and(|i| i.id == session.id && i.auto && i.samples == 0)
            {
                rec.stop_auto();
            }
            Err(e)
        }
    }
}

#[tauri::command]
pub fn stop_stress(state: S) -> AppResult<StressStatus> {
    state.stress.stop()
}

#[tauri::command]
pub fn emergency_abort(state: S) -> StressStatus {
    state.stress.abort(AbortReason::User)
}

#[tauri::command]
pub fn get_stress_status(state: S) -> StressStatus {
    state.stress.status()
}

#[tauri::command]
pub fn get_safety_limits(state: S) -> SafetyLimits {
    *state.limits.read()
}

#[tauri::command]
pub fn set_safety_limits(state: S, limits: SafetyLimits) -> AppResult<SafetyLimits> {
    safety::validate(&limits).map_err(AppError::InvalidArgument)?;
    *state.limits.write() = limits;
    Ok(limits)
}

#[tauri::command]
pub fn session_start(state: S, label: Option<String>) -> SessionInfo {
    state.recorder.lock().start(label, false)
}

#[tauri::command]
pub fn session_stop(state: S) -> AppResult<SessionInfo> {
    state.recorder.lock().stop().ok_or(AppError::NoSession)
}

#[tauri::command]
pub fn get_session_info(state: S) -> Option<SessionInfo> {
    state.recorder.lock().info()
}

/// Writes `<csvPath>` and `<csvPath minus .csv>.summary.json`. The session is cloned under the
/// recorder lock and written after releasing it, so the telemetry loop never waits on disk I/O.
#[tauri::command]
pub async fn export_session(state: S<'_>, csv_path: String) -> AppResult<ExportResult> {
    let shared = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || export_to(&shared, &csv_path))
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
}

pub fn export_to(shared: &AppShared, csv_path: &str) -> AppResult<ExportResult> {
    let session = shared
        .recorder
        .lock()
        .export_snapshot()
        .ok_or(AppError::NoSession)?;
    if session.samples.is_empty() {
        return Err(AppError::NoSession);
    }
    let mut path = std::path::PathBuf::from(csv_path);
    if path
        .extension()
        .is_none_or(|e| !e.eq_ignore_ascii_case("csv"))
    {
        path.set_extension("csv");
    }
    let file = std::io::BufWriter::new(std::fs::File::create(&path)?);
    let rows = csv_export::write_csv(&session, file)?;
    let bytes = std::fs::metadata(&path)?.len();
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "session".into());
    let summary_path = path.with_file_name(format!("{stem}.summary.json"));
    let sum = summary::build(&session, &shared.info);
    std::fs::write(
        &summary_path,
        serde_json::to_vec_pretty(&sum).map_err(|e| AppError::Internal(e.to_string()))?,
    )?;
    Ok(ExportResult {
        csv_path: path.to_string_lossy().into_owned(),
        summary_path: summary_path.to_string_lossy().into_owned(),
        rows,
        bytes,
    })
}
