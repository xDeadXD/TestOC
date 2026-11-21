//! TestOC backend: telemetry engine, stress workers, safety watchdog, sensor sidecar bridge.

#[cfg(not(test))]
pub mod commands;
pub mod error;
pub mod model;
pub mod session;
pub mod sidecar;
#[cfg(not(test))]
pub mod state;
pub mod stress;
pub mod telemetry;
pub mod win;

#[cfg(not(test))]
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
#[cfg(not(test))]
use std::sync::Arc;
#[cfg(not(test))]
use std::time::Duration;

#[cfg(not(test))]
use parking_lot::{Mutex, RwLock};
#[cfg(not(test))]
use tauri::{Manager, RunEvent, WindowEvent};

#[cfg(not(test))]
use model::AbortReason;
#[cfg(not(test))]
use state::{AppShared, TauriEvents};

#[cfg(not(test))]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(log::LevelFilter::Info)
                .target(tauri_plugin_log::Target::new(
                    tauri_plugin_log::TargetKind::Stdout,
                ))
                .target(tauri_plugin_log::Target::new(
                    tauri_plugin_log::TargetKind::LogDir { file_name: None },
                ))
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let handle = app.handle().clone();
            let mut engine = telemetry::TelemetryEngine::new();
            let gpu = stress::gpu::GpuStress::probe().map(Arc::new);
            engine.set_fallback_gpu_name(gpu.as_ref().map(|g| g.adapter_name()));
            let info = engine.system_info(
                gpu.as_ref().map(|g| g.adapter_name().to_string()),
                gpu.as_ref().map(|g| g.fallback_vram_bytes()),
            );
            log::info!(
                "{} | {} logical / {} physical | ISA {:?}",
                info.cpu_brand,
                info.logical_cores,
                info.physical_cores,
                info.isa
            );

            let sidecar = match sidecar::SidecarHub::new() {
                Ok(h) => {
                    // When NVML already covers the GPU, skip LHM GPU polling to cut sidecar latency & RAM.
                    if info.nvml_available {
                        h.set_disable_gpu(true);
                    }
                    Some(h)
                }
                Err(e) => {
                    log::error!("sensor shared memory unavailable: {e}");
                    None
                }
            };
            let events = Arc::new(TauriEvents {
                app: handle.clone(),
                shared: RwLock::new(None),
            });
            let shared = Arc::new(AppShared {
                info,
                latest: RwLock::new(None),
                stress: stress::manager::StressManager::new(events.clone(), gpu),
                limits: RwLock::new(state::default_limits()),
                recorder: Mutex::new(session::SessionRecorder::default()),
                sidecar,
                sidecar_status: RwLock::new(state::initial_sidecar_status()),
                poll_ms: AtomicU32::new(state::DEFAULT_POLL_MS),
                last_tick_ms: AtomicU64::new(0),
                shutdown: AtomicBool::new(false),
            });
            *events.shared.write() = Some(Arc::downgrade(&shared));
            app.manage(shared.clone());

            {
                let (h, s) = (handle.clone(), shared.clone());
                std::thread::Builder::new()
                    .name("telemetry".into())
                    .spawn(move || state::telemetry_loop(h, s, engine))?;
            }
            {
                let (h, s) = (handle.clone(), shared.clone());
                std::thread::Builder::new()
                    .name("heartbeat".into())
                    .spawn(move || state::heartbeat_loop(h, s))?;
            }
            // Launch the elevated sensor sidecar automatically (one UAC prompt).
            // TESTOC_NO_SIDECAR=1 skips it; TESTOC_FAKE_SIDECAR=1 runs it unelevated in --fake mode.
            if std::env::var_os("TESTOC_NO_SIDECAR").is_none() {
                let (h, s) = (handle.clone(), shared.clone());
                std::thread::Builder::new()
                    .name("sidecar-launch".into())
                    .spawn(move || {
                        let Some(hub) = &s.sidecar else { return };
                        if std::env::var_os("TESTOC_FAKE_SIDECAR").is_some() {
                            if let Some(p) = sidecar::sidecar_path() {
                                let pid = std::process::id().to_string();
                                let args = [
                                    "--map",
                                    hub.map_name(),
                                    "--parent-pid",
                                    pid.as_str(),
                                    "--fake",
                                ];
                                match std::process::Command::new(p).args(args).spawn() {
                                    Ok(c) => {
                                        let st = hub.attach_child(c);
                                        s.set_sidecar_status(&h, st);
                                    }
                                    Err(e) => log::error!("fake sidecar spawn failed: {e}"),
                                }
                            }
                        } else {
                            let st = hub.launch();
                            s.set_sidecar_status(&h, st);
                        }
                    })?;
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                if let Some(s) = window.try_state::<Arc<AppShared>>() {
                    s.stress.abort(AbortReason::AppExit);
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_system_info,
            commands::get_latest_sample,
            commands::set_poll_interval,
            commands::get_sidecar_status,
            commands::launch_sidecar,
            commands::start_stress,
            commands::stop_stress,
            commands::emergency_abort,
            commands::get_stress_status,
            commands::get_safety_limits,
            commands::set_safety_limits,
            commands::session_start,
            commands::session_stop,
            commands::get_session_info,
            commands::export_session,
        ])
        .build(tauri::generate_context!())
        .expect("error while building TestOC");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            if let Some(s) = handle.try_state::<Arc<AppShared>>() {
                s.shutdown.store(true, Ordering::Release);
                let joined = s
                    .stress
                    .abort_and_wait(AbortReason::AppExit, Duration::from_secs(2));
                log::info!("exit: workers joined = {joined}");
                if let Some(hub) = &s.sidecar {
                    hub.shutdown();
                    hub.wait_exit(Duration::from_millis(1500));
                }
            }
        }
    });
}
