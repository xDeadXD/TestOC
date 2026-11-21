//! Elevated LibreHardwareMonitor sensor sidecar: launch, liveness and shared-memory reads.

pub mod layout;
pub mod shm;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use windows_sys::Win32::Foundation::{CloseHandle, ERROR_CANCELLED, HANDLE, WAIT_OBJECT_0};
use windows_sys::Win32::System::Com::{
    CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE,
};
use windows_sys::Win32::System::Threading::{
    GetExitCodeProcess, GetProcessId, WaitForSingleObject,
};
use windows_sys::Win32::UI::Shell::{
    ShellExecuteExW, SEE_MASK_FLAG_NO_UI, SEE_MASK_NOCLOSEPROCESS, SHELLEXECUTEINFOW,
};
use windows_sys::Win32::UI::WindowsAndMessaging::SW_HIDE;

use crate::model::{now_ms, SidecarState, SidecarStatus};
use crate::win::{last_error, wide};
use layout::{parse_frame, FrameError, SensorFrame, WriterStatus};
use shm::ShmRegion;

/// Heartbeat staleness threshold. Process crashes/exits are detected immediately on the next
/// 500 ms tick via `WaitForSingleObject`; 8 s only guards against a live hung sidecar while
/// absorbing heavy OS core-affinity scheduling stalls during 100% Absolut RAM / Combined bursts.
pub const STALE_AFTER_MS: u64 = 8000;
pub const SIDECAR_EXE: &str = "testoc-sensors.exe";

struct ProcHandle(HANDLE);
unsafe impl Send for ProcHandle {}
impl Drop for ProcHandle {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

enum LaunchState {
    NotStarted,
    Launching,
    Declined,
    Running(ProcHandle),
    Failed(String),
}

pub struct SidecarHub {
    region: ShmRegion,
    map_name: String,
    disable_gpu: AtomicBool,
    launch: Mutex<LaunchState>,
    last_frame: Mutex<Option<SensorFrame>>,
}

pub enum LaunchOutcome {
    Started(HANDLE),
    Declined,
    Failed(String),
}

/// Launch `exe` elevated through the UAC consent prompt. Blocks until the user answers.
pub fn launch_elevated(exe: &Path, params: &str) -> LaunchOutcome {
    let file = wide(exe.as_os_str());
    let params = wide(params);
    let verb = wide("runas");
    let dir = exe.parent().map(|p| wide(p.as_os_str()));
    unsafe {
        let com_hr = CoInitializeEx(
            std::ptr::null(),
            (COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) as u32,
        );
        let mut sei: SHELLEXECUTEINFOW = std::mem::zeroed();
        sei.cbSize = std::mem::size_of::<SHELLEXECUTEINFOW>() as u32;
        sei.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_FLAG_NO_UI;
        sei.lpVerb = verb.as_ptr();
        sei.lpFile = file.as_ptr();
        sei.lpParameters = params.as_ptr();
        sei.lpDirectory = dir.as_ref().map_or(std::ptr::null(), |d| d.as_ptr());
        sei.nShow = SW_HIDE;
        let ok = ShellExecuteExW(&mut sei);
        let err = if ok == 0 { last_error() } else { 0 };
        if com_hr >= 0 {
            CoUninitialize();
        }
        if ok == 0 {
            return if err == ERROR_CANCELLED {
                LaunchOutcome::Declined
            } else {
                LaunchOutcome::Failed(format!("ShellExecuteExW failed ({err})"))
            };
        }
        if sei.hProcess.is_null() {
            return LaunchOutcome::Failed("no process handle returned".into());
        }
        LaunchOutcome::Started(sei.hProcess)
    }
}

/// Locate the sidecar next to the running executable (dev: target/debug, installed: app dir).
pub fn sidecar_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let p = dir.join(SIDECAR_EXE);
    p.exists().then_some(p)
}

impl SidecarHub {
    pub fn new() -> Result<Self, String> {
        let mut tok = [0u8; 16];
        getrandom::fill(&mut tok).map_err(|e| format!("rng: {e}"))?;
        let hex: String = tok.iter().map(|b| format!("{b:02x}")).collect();
        let map_name = format!("Local\\TestOC.Sensors.{hex}");
        let region = ShmRegion::create(Some(&map_name))?;
        Ok(Self {
            region,
            map_name,
            disable_gpu: AtomicBool::new(false),
            launch: Mutex::new(LaunchState::NotStarted),
            last_frame: Mutex::new(None),
        })
    }

    /// Skip redundant LHM GPU polling (NVAPI/D3DKMT) when NVML already monitors NVIDIA GPUs.
    pub fn set_disable_gpu(&self, disable: bool) {
        self.disable_gpu.store(disable, Ordering::Relaxed);
    }

    pub fn map_name(&self) -> &str {
        &self.map_name
    }

    pub fn region(&self) -> &ShmRegion {
        &self.region
    }

    fn params(&self) -> String {
        let gpu_flag = if self.disable_gpu.load(Ordering::Relaxed) {
            " --no-gpu"
        } else {
            ""
        };
        format!(
            "--map \"{}\" --parent-pid {}{}",
            self.map_name,
            std::process::id(),
            gpu_flag
        )
    }

    /// Launch (or re-launch) the sidecar elevated. Blocks while the UAC prompt is shown,
    /// so call it from a background thread.
    pub fn launch(&self) -> SidecarStatus {
        {
            let mut st = self.launch.lock();
            match &*st {
                LaunchState::Launching => return self.status_from_state(&st, None),
                LaunchState::Running(h) if !process_exited(h).0 => {
                    return self.status_from_state(&st, None)
                }
                _ => {}
            }
            *st = LaunchState::Launching;
        }
        // Clear any stale frame and header from a previously crashed sidecar instance.
        *self.last_frame.lock() = None;
        self.region.reset_header();
        let outcome = match sidecar_path() {
            Some(p) => launch_elevated(&p, &self.params()),
            None => LaunchOutcome::Failed(format!(
                "{SIDECAR_EXE} not found next to the app executable"
            )),
        };
        let mut st = self.launch.lock();
        *st = match outcome {
            LaunchOutcome::Started(h) => LaunchState::Running(ProcHandle(h)),
            LaunchOutcome::Declined => LaunchState::Declined,
            LaunchOutcome::Failed(m) => LaunchState::Failed(m),
        };
        log::info!(
            "sidecar launch -> {:?}",
            self.status_from_state(&st, None).state
        );
        self.status_from_state(&st, None)
    }

    /// For tests / dev: launch an unelevated process (e.g. the sidecar in `--fake` mode).
    pub fn attach_child(&self, child: std::process::Child) -> SidecarStatus {
        use std::os::windows::io::IntoRawHandle;
        *self.last_frame.lock() = None;
        self.region.reset_header();
        let h = child.into_raw_handle() as HANDLE;
        let mut st = self.launch.lock();
        *st = LaunchState::Running(ProcHandle(h));
        self.status_from_state(&st, None)
    }

    pub fn fake_params(&self) -> String {
        format!("{} --fake", self.params())
    }

    /// Ask the sidecar to exit (it also exits when our process dies).
    pub fn shutdown(&self) {
        self.region.set_control(1);
    }

    /// Called by the telemetry thread every tick. Never blocks on the writer.
    pub fn poll(&self, buf: &mut Vec<u8>) -> (SidecarStatus, Option<SensorFrame>) {
        let st = self.launch.lock();
        if !matches!(&*st, LaunchState::Running(_)) {
            return (self.status_from_state(&st, None), None);
        }
        let mut frame_status = None;
        let fresh = match self.region.read_consistent(buf) {
            Some(_) => match parse_frame(buf) {
                Ok(f) => {
                    *self.last_frame.lock() = Some(f.clone());
                    Some(Ok(f))
                }
                Err(e) => Some(Err(e)),
            },
            None => None, // writer busy: reuse last frame
        };
        let frame = match fresh {
            Some(Ok(f)) => Some(f),
            Some(Err(e)) => {
                frame_status = Some(e);
                None
            }
            None => self.last_frame.lock().clone(),
        };
        let status = self.status_from_state(&st, Some((frame.as_ref(), frame_status)));
        let usable = matches!(status.state, SidecarState::Ok);
        (status, if usable { frame } else { None })
    }

    fn status_from_state(
        &self,
        st: &LaunchState,
        frame: Option<(Option<&SensorFrame>, Option<FrameError>)>,
    ) -> SidecarStatus {
        let mk = |state, message: Option<String>, pid| SidecarStatus {
            state,
            message,
            writer_pid: pid,
        };
        match st {
            LaunchState::NotStarted => mk(SidecarState::NotStarted, None, None),
            LaunchState::Launching => mk(
                SidecarState::Launching,
                Some("Waiting for UAC consent…".into()),
                None,
            ),
            LaunchState::Declined => mk(
                SidecarState::Declined,
                Some("Elevation was declined".into()),
                None,
            ),
            LaunchState::Failed(m) => mk(SidecarState::Error, Some(m.clone()), None),
            LaunchState::Running(h) => {
                let pid = unsafe { GetProcessId(h.0) };
                let (exited, code) = process_exited(h);
                if exited {
                    return mk(
                        SidecarState::Error,
                        Some(format!("sensor sidecar exited (code {code})")),
                        Some(pid),
                    );
                }
                match frame {
                    None | Some((None, None)) => mk(SidecarState::Initializing, None, Some(pid)),
                    Some((_, Some(FrameError::NotInitialized))) => {
                        mk(SidecarState::Initializing, None, Some(pid))
                    }
                    Some((_, Some(FrameError::Incompatible(m)))) => {
                        mk(SidecarState::Incompatible, Some(m), Some(pid))
                    }
                    Some((Some(f), None)) => {
                        if f.writer_pid != 0 && pid != 0 && f.writer_pid != pid {
                            return mk(SidecarState::Initializing, None, Some(pid));
                        }
                        let age = now_ms().saturating_sub(f.heartbeat_ms);
                        let text = (!f.status_text.is_empty()).then(|| f.status_text.clone());
                        match f.status {
                            WriterStatus::Initializing => {
                                mk(SidecarState::Initializing, text, Some(pid))
                            }
                            WriterStatus::PawnIoMissing => {
                                mk(SidecarState::PawnIoMissing, text, Some(pid))
                            }
                            WriterStatus::Error => mk(SidecarState::Error, text, Some(pid)),
                            WriterStatus::Ok if age > STALE_AFTER_MS => mk(
                                SidecarState::Stale,
                                Some(format!("no heartbeat for {:.1} s", age as f64 / 1000.0)),
                                Some(pid),
                            ),
                            WriterStatus::Ok => mk(SidecarState::Ok, text, Some(pid)),
                        }
                    }
                }
            }
        }
    }

    pub fn wait_exit(&self, timeout: Duration) {
        let st = self.launch.lock();
        if let LaunchState::Running(h) = &*st {
            unsafe {
                WaitForSingleObject(h.0, timeout.as_millis() as u32);
            }
        }
    }
}

fn process_exited(h: &ProcHandle) -> (bool, u32) {
    unsafe {
        if WaitForSingleObject(h.0, 0) == WAIT_OBJECT_0 {
            let mut code = 0u32;
            GetExitCodeProcess(h.0, &mut code);
            (true, code)
        } else {
            (false, 0)
        }
    }
}
