//! Stress run lifecycle and thread-safety core.
//!
//! * Workers never take locks: they only touch `Arc<RunShared>` atomics.
//! * `StressManager::control` (a short Mutex) serializes start/stop/abort requests.
//! * A per-run supervisor thread owns the worker `JoinHandle`s, enforces duration / time caps,
//!   drives the variable-load schedule, and publishes the final phase. `emergency_abort` therefore
//!   never waits for threads: it flips atomics and returns.

use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_utils::CachePadded;
use parking_lot::{Mutex, RwLock};

use super::cpu_kernel::{
    dataset_n, fft_ladder, fft_size_label, ntt_len, panel_rows, FftFailure, FftSet, Kernel, MatSet,
    NttSet,
};
use super::memory_kernel::{self as mk, Pattern};
use super::variable::{Pace, Throttle, VariableSchedule};
use crate::error::{AppError, AppResult};
use crate::model::*;
use crate::win;

/// Applied to CPU/combined runs started without a working CPU temperature sensor.
pub const NO_THERMAL_TIME_CAP: Duration = Duration::from_secs(600);
const SUPERVISOR_TICK: Duration = Duration::from_millis(5);
const MAX_ERROR_EVENTS_PER_SEC: u64 = 10;

/// Event sink so the engine is testable without Tauri.
pub trait StressEvents: Send + Sync {
    fn status(&self, s: &StressStatus);
    fn worker_error(&self, e: &WorkerError);
    fn alert(&self, a: &SafetyAlert);
}

pub struct NullEvents;
impl StressEvents for NullEvents {
    fn status(&self, _: &StressStatus) {}
    fn worker_error(&self, _: &WorkerError) {}
    fn alert(&self, _: &SafetyAlert) {}
}

type Counters = Box<[CachePadded<AtomicU64>]>;

fn counters(n: usize) -> Counters {
    (0..n)
        .map(|_| CachePadded::new(AtomicU64::new(0)))
        .collect()
}

fn sum(c: &Counters) -> u64 {
    c.iter().map(|x| x.load(Ordering::Relaxed)).sum()
}

#[derive(Default)]
struct RateState {
    last: Option<(Instant, u64, u64, u64)>,
}

/// State shared between the supervisor, workers, telemetry/watchdog and IPC commands.
pub struct RunShared {
    pub config: StressConfig,
    pub kind: StressKind,
    run: AtomicBool,
    abort: AtomicBool,
    phase: AtomicU8,
    abort_reason: AtomicU8,
    done: AtomicBool,
    failed: AtomicBool,
    completed_naturally: AtomicBool,
    /// CoreCycler active worker index (`u16::MAX` = all workers active).
    active_worker: AtomicU16,
    pub started: Instant,
    pub started_ms: u64,
    pub duration: Option<Duration>,
    pub time_cap: Option<Duration>,
    pub thermal_protection: bool,
    pub stop_on_error: bool,
    /// Effective critical error cutoff (1 when stop_on_error, or user max_errors > 0).
    pub error_cutoff: Option<u64>,
    pub cpu_threads: u16,
    pub mem_threads: u16,
    pub isa_used: Option<&'static str>,
    pub mem_target_bytes: u64,
    pub mem_allocated: AtomicU64,
    pub vram_target_bytes: u64,
    pub vram_allocated: AtomicU64,
    /// Per CPU worker FLOP counters.
    flops: Counters,
    /// Dedicated GPU compute FLOP counter (independent of CPU slots for Power mode).
    gpu_flops: AtomicU64,
    /// Per memory/VRAM worker byte counters.
    bytes: Counters,
    /// Per logical core compute-error counters.
    core_errors: Counters,
    gpu_errors: AtomicU64,
    total_errors: AtomicU64,
    pub load_level: AtomicU8,
    workers_alive: AtomicU16,
    finished_elapsed_ms: AtomicU64,
    gflops_bits: AtomicU64,
    gpu_gflops_bits: AtomicU64,
    mbps_bits: AtomicU64,
    rate: Mutex<RateState>,
    message: Mutex<Option<String>>,
    err_window: AtomicU64, // (second << 16) | count
    first_hw_error_announced: AtomicBool,
}

impl RunShared {
    #[inline]
    pub fn should_continue(&self) -> bool {
        self.run.load(Ordering::Relaxed) && !self.abort.load(Ordering::Relaxed)
    }

    pub fn is_aborted(&self) -> bool {
        self.abort.load(Ordering::Relaxed)
    }

    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    pub fn phase(&self) -> StressPhase {
        StressPhase::from_u8(self.phase.load(Ordering::Acquire))
    }

    pub fn abort_reason(&self) -> Option<AbortReason> {
        AbortReason::from_u8(self.abort_reason.load(Ordering::Acquire))
    }

    /// Instant, lock-free abort. Returns true for the first caller.
    pub fn trigger_abort(&self, reason: AbortReason) -> bool {
        if self.done.load(Ordering::Acquire) {
            return false;
        }
        if self.abort.swap(true, Ordering::SeqCst) {
            return false;
        }
        self.abort_reason.store(reason as u8, Ordering::SeqCst);
        self.phase
            .store(StressPhase::Aborted as u8, Ordering::SeqCst);
        true
    }

    /// Signal workers to finish after the current work block without regressing a terminal phase.
    pub fn request_stop(&self) -> bool {
        if self.done.load(Ordering::Acquire) || self.abort.load(Ordering::SeqCst) {
            return false;
        }
        let was = self.run.swap(false, Ordering::SeqCst);
        if was && !self.done.load(Ordering::Acquire) {
            // Only advance active non-stopping phases so a fast supervisor exit cannot be overwritten.
            for expected in [StressPhase::Starting as u8, StressPhase::Running as u8] {
                if self
                    .phase
                    .compare_exchange(
                        expected,
                        StressPhase::Stopping as u8,
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    break;
                }
            }
        }
        was
    }

    pub fn elapsed(&self) -> Duration {
        let f = self.finished_elapsed_ms.load(Ordering::Acquire);
        if f > 0 {
            Duration::from_millis(f)
        } else {
            self.started.elapsed()
        }
    }

    pub fn total_errors(&self) -> u64 {
        self.total_errors.load(Ordering::Relaxed)
    }

    fn set_message(&self, m: impl Into<String>) {
        let mut g = self.message.lock();
        if g.is_none() {
            *g = Some(m.into());
        }
    }

    /// Update live progress status message (e.g. active Absolut test or Core Cycler core) while running.
    fn update_message(&self, m: impl Into<String>) {
        if self.failed.load(Ordering::Relaxed) || self.is_aborted() {
            return;
        }
        *self.message.lock() = Some(m.into());
    }

    /// Recompute GFLOPS / GPU GFLOPS / MB/s from counter deltas (called by supervisor ~2 Hz).
    pub fn update_rates(&self) {
        let now = Instant::now();
        let (f, gf, b) = (
            sum(&self.flops),
            self.gpu_flops.load(Ordering::Relaxed),
            sum(&self.bytes),
        );
        let mut r = self.rate.lock();
        if let Some((t0, f0, gf0, b0)) = r.last {
            let dt = now.duration_since(t0).as_secs_f64();
            if dt > 0.05 {
                let g = (f.saturating_sub(f0)) as f64 / dt / 1e9;
                let gg = (gf.saturating_sub(gf0)) as f64 / dt / 1e9;
                let m = (b.saturating_sub(b0)) as f64 / dt / 1e6;
                self.gflops_bits.store(g.to_bits(), Ordering::Relaxed);
                self.gpu_gflops_bits.store(gg.to_bits(), Ordering::Relaxed);
                self.mbps_bits.store(m.to_bits(), Ordering::Relaxed);
                r.last = Some((now, f, gf, b));
            }
        } else {
            r.last = Some((now, f, gf, b));
        }
    }

    fn zero_rates(&self) {
        self.gflops_bits.store(0f64.to_bits(), Ordering::Relaxed);
        self.gpu_gflops_bits
            .store(0f64.to_bits(), Ordering::Relaxed);
        self.mbps_bits.store(0f64.to_bits(), Ordering::Relaxed);
    }

    pub fn gflops(&self) -> f64 {
        f64::from_bits(self.gflops_bits.load(Ordering::Relaxed))
    }

    pub fn gpu_gflops(&self) -> f64 {
        f64::from_bits(self.gpu_gflops_bits.load(Ordering::Relaxed))
    }

    pub fn mem_mbps(&self) -> f64 {
        f64::from_bits(self.mbps_bits.load(Ordering::Relaxed))
    }

    pub fn add_extra_gpu_errors(&self, extra: u64) {
        self.gpu_errors.fetch_add(extra, Ordering::Relaxed);
        self.total_errors.fetch_add(extra, Ordering::Relaxed);
    }

    /// Count a hardware error; emits a rate-limited event and honours `stop_on_error` / `max_errors`.
    pub fn record_error(
        &self,
        events: &dyn StressEvents,
        worker_id: usize,
        core_id: Option<usize>,
        kind: WorkerErrorKind,
        detail: String,
    ) {
        let is_hw_err = !matches!(
            kind,
            WorkerErrorKind::AllocFailed | WorkerErrorKind::GpuDevice
        );
        if is_hw_err {
            self.total_errors.fetch_add(1, Ordering::Relaxed);
            if matches!(
                kind,
                WorkerErrorKind::GpuMismatch | WorkerErrorKind::VramMismatch
            ) {
                self.gpu_errors.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(c) = core_id.and_then(|c| self.core_errors.get(c)) {
                c.fetch_add(1, Ordering::Relaxed);
            }
        }
        // Rate limit: at most N events per wall-clock second.
        let sec = now_ms() / 1000;
        let allowed = loop {
            let cur = self.err_window.load(Ordering::Relaxed);
            let (s, n) = (cur >> 16, cur & 0xFFFF);
            let next = if s == sec {
                (s << 16) | (n + 1).min(0xFFFF)
            } else {
                (sec << 16) | 1
            };
            if self
                .err_window
                .compare_exchange(cur, next, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break if s == sec {
                    n < MAX_ERROR_EVENTS_PER_SEC
                } else {
                    true
                };
            }
        };
        if allowed {
            events.worker_error(&WorkerError {
                worker_id: worker_id as u16,
                core_id: core_id.map(|c| c as u16),
                kind,
                detail: detail.clone(),
                timestamp_ms: now_ms(),
            });
            // Push updated error count to the UI immediately instead of waiting for the 1 s supervisor tick.
            if is_hw_err {
                events.status(&self.status());
            }
        }
        let errs = self.total_errors();
        let mut aborted_now = false;
        if let Some(cut) = self.error_cutoff {
            if is_hw_err && errs >= cut && self.trigger_abort(AbortReason::WorkerError) {
                aborted_now = true;
                self.first_hw_error_announced.store(true, Ordering::Relaxed);
                events.alert(&SafetyAlert {
                    level: AlertLevel::Critical,
                    metric: AlertMetric::WorkerError,
                    value: errs as f32,
                    threshold: cut as f32,
                    action: AlertAction::Aborted,
                    timestamp_ms: now_ms(),
                    message: if cut == 1 {
                        format!("Hardware error detected; stopped on first error ({detail})")
                    } else {
                        format!("Hardware error limit reached ({errs} ≥ {cut}); test aborted")
                    },
                });
                events.status(&self.status());
            }
        }
        // Surface the first hardware error immediately as a Critical banner even when stop-on-error is off.
        if is_hw_err && !aborted_now && !self.first_hw_error_announced.swap(true, Ordering::AcqRel)
        {
            events.alert(&SafetyAlert {
                level: AlertLevel::Critical,
                metric: AlertMetric::WorkerError,
                value: errs as f32,
                threshold: 1.0,
                action: AlertAction::None,
                timestamp_ms: now_ms(),
                message: format!("Hardware error detected: {detail}"),
            });
            events.status(&self.status());
        }
    }

    pub fn snapshot(&self) -> StressSnapshot {
        StressSnapshot {
            phase: self.phase(),
            kind: Some(self.kind),
            elapsed_s: self.elapsed().as_secs_f32(),
            gflops: self.gflops(),
            gpu_gflops: self.gpu_gflops(),
            mem_mbps: self.mem_mbps(),
            errors: self.total_errors(),
            load_level: self.load_level.load(Ordering::Relaxed),
        }
    }

    pub fn status(&self) -> StressStatus {
        StressStatus {
            phase: self.phase(),
            kind: Some(self.kind),
            config: Some(self.config.clone()),
            abort_reason: self.abort_reason(),
            message: self.message.lock().clone(),
            elapsed_s: self.elapsed().as_secs_f32(),
            duration_s: self.duration.map(|d| d.as_secs() as u32),
            time_cap_s: self.time_cap.map(|d| d.as_secs() as u32),
            thermal_protection: self.thermal_protection,
            cpu_threads: self.cpu_threads,
            mem_threads: self.mem_threads,
            isa_used: self.isa_used.map(str::to_string),
            mem_target_bytes: self.mem_target_bytes,
            mem_allocated_bytes: self.mem_allocated.load(Ordering::Relaxed),
            vram_target_bytes: self.vram_target_bytes,
            vram_allocated_bytes: self.vram_allocated.load(Ordering::Relaxed),
            gflops: self.gflops(),
            gpu_gflops: self.gpu_gflops(),
            mem_mbps: self.mem_mbps(),
            total_errors: self.total_errors(),
            gpu_errors: self.gpu_errors.load(Ordering::Relaxed),
            core_errors: self
                .core_errors
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect(),
            load_level: self.load_level.load(Ordering::Relaxed),
            workers_alive: self.workers_alive.load(Ordering::Relaxed),
        }
    }
}

/// Everything `start` needs to know about the machine right now.
#[derive(Debug, Clone)]
pub struct StartContext {
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub isa: IsaFlags,
    pub total_mem: u64,
    pub available_mem: u64,
    pub vram_total: Option<u64>,
    pub vram_free: Option<u64>,
    pub thermal_ok: bool,
    pub acknowledge_no_thermal: bool,
    pub limits: SafetyLimits,
}

pub struct StressManager {
    control: Mutex<()>,
    current: RwLock<Option<Arc<RunShared>>>,
    events: Arc<dyn StressEvents>,
    gpu: Option<Arc<super::gpu::GpuStress>>,
}

impl StressManager {
    pub fn new(events: Arc<dyn StressEvents>, gpu: Option<Arc<super::gpu::GpuStress>>) -> Self {
        Self {
            control: Mutex::new(()),
            current: RwLock::new(None),
            events,
            gpu,
        }
    }

    /// Current (or last) run, without taking the control lock.
    pub fn current(&self) -> Option<Arc<RunShared>> {
        self.current.read().clone()
    }

    pub fn status(&self) -> StressStatus {
        self.current()
            .map(|r| r.status())
            .unwrap_or_else(StressStatus::idle)
    }

    pub fn snapshot(&self) -> StressSnapshot {
        self.current()
            .map(|r| r.snapshot())
            .unwrap_or(StressSnapshot {
                phase: StressPhase::Idle,
                kind: None,
                elapsed_s: 0.0,
                gflops: 0.0,
                gpu_gflops: 0.0,
                mem_mbps: 0.0,
                errors: 0,
                load_level: 0,
            })
    }

    pub fn gpu_adapter_name(&self) -> Option<String> {
        self.gpu.as_ref().map(|g| g.adapter_name().to_string())
    }

    pub fn start(&self, config: StressConfig, ctx: StartContext) -> AppResult<StressStatus> {
        let _g = self.control.lock();
        if let Some(r) = self.current() {
            if !r.is_done() {
                return Err(AppError::Busy);
            }
        }
        let kind = config.kind();
        let logical = ctx.logical_cores.max(1);
        let physical = ctx.physical_cores.clamp(1, logical);
        let clamp_threads_max = |t: Option<u16>, default: usize, max: usize| -> AppResult<u16> {
            let t = t.map(|t| t as usize).unwrap_or(default);
            if t == 0 || t > max {
                return Err(AppError::InvalidArgument(format!(
                    "threads must be 1-{max}"
                )));
            }
            Ok(t as u16)
        };
        let clamp_threads = |t: Option<u16>, default: usize| clamp_threads_max(t, default, logical);
        let validate_var_range = |min_pct: u8, max_pct: u8| -> AppResult<(u8, u8)> {
            if !(10..=100).contains(&min_pct) || !(10..=100).contains(&max_pct) || min_pct > max_pct
            {
                return Err(AppError::InvalidArgument(format!(
                    "variable load range {min_pct}-{max_pct}% invalid (need 10 ≤ min ≤ max ≤ 100)"
                )));
            }
            Ok((min_pct, max_pct))
        };
        if let Some(d) = config.duration_s() {
            if d == 0 || d > 7 * 24 * 3600 {
                return Err(AppError::InvalidArgument(
                    "duration must be 1 s - 7 days".into(),
                ));
            }
        }

        // Thermal policy: CPU-loading tests (Cpu/Combined/Power) need the sensor or explicit ack.
        let mut time_cap = None;
        if kind.loads_cpu() && !ctx.thermal_ok {
            if !ctx.acknowledge_no_thermal {
                return Err(AppError::ThermalUnavailable);
            }
            time_cap = Some(NO_THERMAL_TIME_CAP);
        }

        let (mut cpu_threads, mut mem_threads, mut kernel, mut mem_target, mut dataset) =
            (0u16, 0u16, None, 0u64, Dataset::Small);
        let mut physical_only = false;
        let mut cpu_workload = CpuWorkload::Extreme;
        let mut cpu_fft_preset = FftPreset::Small;
        let mut cpu_var_range: Option<(u8, u8)> = None;
        let mut core_cycler: Option<(u16, u8)> = None;
        let mut mem_profile = MemoryProfile::Absolut;
        let mut mem_pattern = MemoryPattern::All;
        let mut mem_cycles: Option<u16> = None;
        let mut vram_pattern = MemoryPattern::All;
        let (mut vram_target, mut vram_chunk) = (0u64, 0u64);
        let mut gpu_opts: Option<super::gpu::GpuRunOptions> = None;
        let mut mem_pct = 0u8;
        match &config {
            StressConfig::Cpu {
                threads,
                physical_cores_only,
                isa,
                dataset: d,
                workload,
                fft_preset,
                mode: m,
                variable_min_pct,
                variable_max_pct,
                core_cycle_s,
                ..
            } => {
                // CoreCycler defaults to physical cores so each core hits max single-core Fmax boost.
                physical_only =
                    *physical_cores_only || (*m == LoadMode::CoreCycler && threads.is_none());
                let max_t = if physical_only { physical } else { logical };
                cpu_threads = clamp_threads_max(*threads, max_t, max_t)?;
                kernel = Some(Kernel::resolve(*isa, &ctx.isa)?);
                dataset = *d;
                cpu_workload = *workload;
                cpu_fft_preset = *fft_preset;
                if *m == LoadMode::Variable {
                    cpu_var_range = Some(validate_var_range(*variable_min_pct, *variable_max_pct)?);
                } else if *m == LoadMode::CoreCycler {
                    if !(1..=60).contains(core_cycle_s) {
                        return Err(AppError::InvalidArgument(
                            "core cycler interval must be 1-60 s".into(),
                        ));
                    }
                    core_cycler = Some((cpu_threads, *core_cycle_s));
                }
            }
            StressConfig::Memory {
                threads,
                target_pct,
                profile,
                pattern,
                cycles,
                ..
            } => {
                // Absolut saturates all logical threads by default for full dual-channel IMC pressure.
                let def_t = if *profile == MemoryProfile::Absolut {
                    logical
                } else {
                    (logical / 2).clamp(1, 8)
                };
                mem_threads = clamp_threads(*threads, def_t)?;
                mem_pct = *target_pct;
                mem_profile = *profile;
                mem_pattern = *pattern;
                if let Some(c) = cycles {
                    if !(1..=100).contains(c) {
                        return Err(AppError::InvalidArgument(
                            "memory cycle target must be 1-100".into(),
                        ));
                    }
                    mem_cycles = Some(*c);
                }
            }
            StressConfig::Combined {
                cpu_threads: t,
                cpu_isa,
                cpu_dataset,
                cpu_workload: cw,
                cpu_fft_preset: cf,
                mem_target_pct,
                mem_profile: mp,
                ..
            } => {
                cpu_threads = clamp_threads(*t, logical)?;
                mem_threads = 2.min(logical as u16);
                kernel = Some(Kernel::resolve(*cpu_isa, &ctx.isa)?);
                dataset = *cpu_dataset;
                cpu_workload = *cw;
                cpu_fft_preset = *cf;
                mem_pct = *mem_target_pct;
                mem_profile = *mp;
            }
            StressConfig::Gpu {
                mode: m,
                profile,
                intensity,
                variable_min_pct,
                variable_max_pct,
                variable_step_s,
                error_check,
                ..
            } => {
                if self.gpu.is_none() {
                    return Err(AppError::Unsupported(
                        "no Vulkan GPU compute adapter found".into(),
                    ));
                }
                if *m == LoadMode::CoreCycler {
                    return Err(AppError::InvalidArgument(
                        "Core Cycler mode is for CPU tests only".into(),
                    ));
                }
                if !(10..=100).contains(intensity) {
                    return Err(AppError::InvalidArgument("intensity must be 10-100".into()));
                }
                let (vmin, vmax) = validate_var_range(*variable_min_pct, *variable_max_pct)?;
                if !(1..=60).contains(variable_step_s) {
                    return Err(AppError::InvalidArgument(
                        "adaptive step interval must be 1-60 s".into(),
                    ));
                }
                gpu_opts = Some(super::gpu::GpuRunOptions {
                    profile: *profile,
                    mode: *m,
                    intensity: *intensity,
                    var_min_pct: vmin,
                    var_max_pct: vmax,
                    var_step_s: *variable_step_s,
                    error_check: *error_check,
                });
            }
            StressConfig::Vram {
                target_pct,
                pattern,
                ..
            } => {
                let gpu = self
                    .gpu
                    .as_ref()
                    .ok_or_else(|| AppError::Unsupported("no Vulkan GPU adapter found".into()))?;
                let total_vram = ctx.vram_total.unwrap_or_else(|| gpu.fallback_vram_bytes());
                let (t, c) = super::gpu::compute_vram_target_bytes(
                    total_vram,
                    ctx.vram_free,
                    *target_pct,
                    gpu.max_binding_size(),
                )
                .map_err(AppError::LimitExceeded)?;
                vram_target = t;
                vram_chunk = c;
                vram_pattern = *pattern;
            }
            StressConfig::Power {
                cpu_threads: t,
                cpu_isa,
                cpu_dataset,
                cpu_workload: cw,
                cpu_fft_preset: cf,
                gpu_profile,
                gpu_intensity: gi,
                ..
            } => {
                if self.gpu.is_none() {
                    return Err(AppError::Unsupported(
                        "no Vulkan GPU compute adapter found for Power test".into(),
                    ));
                }
                if !(10..=100).contains(gi) {
                    return Err(AppError::InvalidArgument(
                        "GPU intensity must be 10-100".into(),
                    ));
                }
                cpu_threads = clamp_threads(*t, logical)?;
                kernel = Some(Kernel::resolve(*cpu_isa, &ctx.isa)?);
                dataset = *cpu_dataset;
                cpu_workload = *cw;
                cpu_fft_preset = *cf;
                gpu_opts = Some(super::gpu::GpuRunOptions {
                    profile: *gpu_profile,
                    mode: LoadMode::Steady,
                    intensity: *gi,
                    var_min_pct: 20,
                    var_max_pct: 100,
                    var_step_s: 5,
                    error_check: true,
                });
            }
        }
        if mem_threads > 0 {
            mem_target = mk::compute_target_bytes(
                ctx.total_mem,
                ctx.available_mem,
                mem_pct,
                ctx.limits.mem_max_pct,
            )
            .map_err(AppError::LimitExceeded)?;
        }

        let isa_used = match kind {
            StressKind::Cpu => match (kernel, cpu_workload) {
                (Some(Kernel::Avx512), CpuWorkload::Extreme) => Some("AVX-512F + FFT + NTT64"),
                (Some(Kernel::Avx2Fma), CpuWorkload::Extreme) => Some("AVX2+FMA + FFT + NTT64"),
                (Some(Kernel::Scalar), CpuWorkload::Extreme) => Some("SSE2 + FFT + NTT64"),
                (Some(Kernel::Avx512), CpuWorkload::FftTorture) => Some("FFT Torture (AVX-512F)"),
                (Some(Kernel::Avx2Fma), CpuWorkload::FftTorture) => Some("FFT Torture (AVX2+FMA)"),
                (Some(Kernel::Scalar), CpuWorkload::FftTorture) => Some("FFT Torture (SSE2)"),
                (_, CpuWorkload::NttPrime) => Some("NTT64 (Goldilocks)"),
                (k, CpuWorkload::MatrixFma) => k.map(Kernel::name),
                (None, _) => None,
            },
            StressKind::Memory => Some(match mem_profile {
                MemoryProfile::Absolut if ctx.isa.avx2 => "Absolut 32-Step (AVX2 NT)",
                MemoryProfile::Absolut => "Absolut 32-Step",
                MemoryProfile::Standard => "RAM Standard",
            }),
            StressKind::Combined => Some(match (cpu_workload, mem_profile) {
                (CpuWorkload::FftTorture, MemoryProfile::Absolut) => "FFT Torture + Absolut",
                (CpuWorkload::FftTorture, MemoryProfile::Standard) => "FFT Torture + RAM",
                (_, MemoryProfile::Absolut) if ctx.isa.avx2 => "AVX2+FFT+NTT + Absolut",
                (_, MemoryProfile::Absolut) => "CPU + Absolut 32-Step",
                (_, MemoryProfile::Standard) => kernel.map(Kernel::name).unwrap_or("CPU + RAM"),
            }),
            StressKind::Gpu => gpu_opts.map(|o| match (o.profile, o.mode) {
                (GpuProfile::Extreme, LoadMode::Steady) => "Vulkan 3D Extreme",
                (GpuProfile::Extreme, LoadMode::Variable) => "Vulkan 3D Extreme (Adaptive)",
                (GpuProfile::Heavy, LoadMode::Steady) => "Vulkan Heavy Compute",
                (GpuProfile::Heavy, LoadMode::Variable) => "Vulkan Heavy Compute (Adaptive)",
                (GpuProfile::Standard, LoadMode::Steady) => "Vulkan FP32 FMA",
                (GpuProfile::Standard, LoadMode::Variable) => "Vulkan FP32 FMA (Adaptive)",
                (GpuProfile::Extreme, LoadMode::CoreCycler) => "Vulkan 3D Extreme",
                (GpuProfile::Heavy, LoadMode::CoreCycler) => "Vulkan Heavy Compute",
                (GpuProfile::Standard, LoadMode::CoreCycler) => "Vulkan FP32 FMA",
            }),
            StressKind::Vram => Some("Vulkan VRAM Absolut"),
            StressKind::Power => {
                let gp = gpu_opts.map(|o| o.profile).unwrap_or(GpuProfile::Extreme);
                Some(match (kernel, gp) {
                    (Some(Kernel::Avx512), GpuProfile::Extreme) => "AVX-512F + 3D Extreme",
                    (Some(Kernel::Avx512), GpuProfile::Heavy) => "AVX-512F + GPU Heavy",
                    (Some(Kernel::Avx512), GpuProfile::Standard) => "AVX-512F + GPU Standard",
                    (Some(Kernel::Avx2Fma), GpuProfile::Extreme) => "AVX2+FMA + 3D Extreme",
                    (Some(Kernel::Avx2Fma), GpuProfile::Heavy) => "AVX2+FMA + GPU Heavy",
                    (Some(Kernel::Avx2Fma), GpuProfile::Standard) => "AVX2+FMA + GPU Standard",
                    (_, GpuProfile::Extreme) => "Scalar + 3D Extreme",
                    (_, GpuProfile::Heavy) => "Scalar + GPU Heavy",
                    (_, GpuProfile::Standard) => "Scalar + GPU Standard",
                })
            }
        };
        let shared = Arc::new(RunShared {
            config: config.clone(),
            kind,
            run: AtomicBool::new(true),
            abort: AtomicBool::new(false),
            phase: AtomicU8::new(StressPhase::Starting as u8),
            abort_reason: AtomicU8::new(0),
            done: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            completed_naturally: AtomicBool::new(false),
            active_worker: AtomicU16::new(if core_cycler.is_some() { 0 } else { u16::MAX }),
            started: Instant::now(),
            started_ms: now_ms(),
            duration: config.duration_s().map(|s| Duration::from_secs(s as u64)),
            time_cap,
            thermal_protection: ctx.thermal_ok,
            stop_on_error: config.stop_on_error(),
            error_cutoff: config.error_cutoff(),
            cpu_threads,
            mem_threads,
            isa_used,
            mem_target_bytes: mem_target,
            mem_allocated: AtomicU64::new(0),
            vram_target_bytes: vram_target,
            vram_allocated: AtomicU64::new(0),
            flops: counters((cpu_threads as usize).max(1)),
            gpu_flops: AtomicU64::new(0),
            bytes: counters((mem_threads as usize).max(1)),
            core_errors: counters(logical),
            gpu_errors: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            load_level: AtomicU8::new(100),
            workers_alive: AtomicU16::new(0),
            finished_elapsed_ms: AtomicU64::new(0),
            gflops_bits: AtomicU64::new(0),
            gpu_gflops_bits: AtomicU64::new(0),
            mbps_bits: AtomicU64::new(0),
            rate: Mutex::new(RateState::default()),
            message: Mutex::new(None),
            err_window: AtomicU64::new(0),
            first_hw_error_announced: AtomicBool::new(false),
        });

        // Spawn workers.
        let mut handles: Vec<JoinHandle<()>> = Vec::new();
        let spawn = |name: String, f: Box<dyn FnOnce() + Send>| -> AppResult<JoinHandle<()>> {
            std::thread::Builder::new()
                .name(name)
                .stack_size(1 << 20)
                .spawn(f)
                .map_err(|e| AppError::Internal(format!("spawn failed: {e}")))
        };
        let mut spawn_err = None;
        if let Some(k) = kernel {
            let n = dataset_n(dataset);
            let ntt_n = ntt_len(dataset);
            for w in 0..cpu_threads as usize {
                // With physical_only or fewer threads than logical CPUs, spread across physical
                // cores first (Windows enumerates SMT siblings adjacently: 0/1 = core 0, 2/3 = core 1, ...).
                let half = logical / 2;
                let core = if physical_only && logical >= 2 {
                    (w * 2) % logical
                } else if cpu_threads as usize == logical || logical < 2 {
                    w
                } else if w < half {
                    w * 2
                } else {
                    (w - half) * 2 + 1
                };
                let (s, ev) = (shared.clone(), self.events.clone());
                s.workers_alive.fetch_add(1, Ordering::SeqCst);
                match spawn(
                    format!("cpu-worker-{w}"),
                    Box::new(move || {
                        cpu_worker(
                            s,
                            ev,
                            w,
                            core % logical,
                            k,
                            n,
                            ntt_n,
                            cpu_workload,
                            cpu_fft_preset,
                            core_cycler.is_some(),
                        )
                    }),
                ) {
                    Ok(h) => handles.push(h),
                    Err(e) => {
                        shared.workers_alive.fetch_sub(1, Ordering::SeqCst);
                        spawn_err = Some(e);
                        break;
                    }
                }
            }
        }
        if spawn_err.is_none() && mem_threads > 0 {
            let per = mem_target / mem_threads as u64;
            let coord = Arc::new(MemAllocCoord {
                total_workers: mem_threads as usize,
                stage1_done: AtomicU16::new(0),
                last_reclaim_ms: AtomicU64::new(0),
                last_swap_ms: AtomicU64::new(0),
                topup_lock: Mutex::new(()),
                target_pct: mem_pct,
                max_pct: ctx.limits.mem_max_pct,
            });
            for w in 0..mem_threads as usize {
                let (s, ev, co) = (shared.clone(), self.events.clone(), coord.clone());
                let pin_core = (kind == StressKind::Memory && logical > 0).then_some(w % logical);
                s.workers_alive.fetch_add(1, Ordering::SeqCst);
                match spawn(
                    format!("mem-worker-{w}"),
                    Box::new(move || {
                        mem_worker(
                            s,
                            ev,
                            w,
                            pin_core,
                            per,
                            mem_profile,
                            mem_pattern,
                            mem_cycles,
                            co,
                        )
                    }),
                ) {
                    Ok(h) => handles.push(h),
                    Err(e) => {
                        shared.workers_alive.fetch_sub(1, Ordering::SeqCst);
                        spawn_err = Some(e);
                        break;
                    }
                }
            }
        }
        if spawn_err.is_none() {
            if let (Some(opts), Some(gpu)) = (gpu_opts, &self.gpu) {
                let (s, ev, g) = (shared.clone(), self.events.clone(), gpu.clone());
                s.workers_alive.fetch_add(1, Ordering::SeqCst);
                match spawn(
                    "gpu-worker".into(),
                    Box::new(move || gpu_worker(s, ev, g, opts)),
                ) {
                    Ok(h) => handles.push(h),
                    Err(e) => {
                        shared.workers_alive.fetch_sub(1, Ordering::SeqCst);
                        spawn_err = Some(e);
                    }
                }
            } else if let (StressConfig::Vram { .. }, Some(gpu)) = (&config, &self.gpu) {
                let (s, ev, g) = (shared.clone(), self.events.clone(), gpu.clone());
                s.workers_alive.fetch_add(1, Ordering::SeqCst);
                match spawn(
                    "vram-worker".into(),
                    Box::new(move || vram_worker(s, ev, g, vram_target, vram_chunk, vram_pattern)),
                ) {
                    Ok(h) => handles.push(h),
                    Err(e) => {
                        shared.workers_alive.fetch_sub(1, Ordering::SeqCst);
                        spawn_err = Some(e);
                    }
                }
            }
        }
        if let Some(e) = spawn_err {
            shared.run.store(false, Ordering::SeqCst);
            for h in handles {
                let _ = h.join();
            }
            return Err(e);
        }

        *self.current.write() = Some(shared.clone());
        let (s, ev) = (shared.clone(), self.events.clone());
        std::thread::Builder::new()
            .name("stress-supervisor".into())
            .spawn(move || supervise(s, ev, handles, cpu_var_range, core_cycler))
            .map_err(|e| AppError::Internal(format!("supervisor spawn failed: {e}")))?;
        let st = shared.status();
        self.events.status(&st);
        Ok(st)
    }

    pub fn stop(&self) -> AppResult<StressStatus> {
        let r = {
            let _g = self.control.lock();
            let r = self
                .current()
                .filter(|r| !r.is_done())
                .ok_or(AppError::NotRunning)?;
            r.request_stop();
            let st = r.status();
            self.events.status(&st);
            r
        };
        // Wait briefly (with control lock released so abort() never blocks) so fast-exiting workers
        // return a terminal Stopped status directly instead of racing with the stopping event.
        let deadline = Instant::now() + Duration::from_millis(250);
        while !r.is_done() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(r.status())
    }

    /// Never blocks on worker threads. Safe to call from any thread, any time.
    pub fn abort(&self, reason: AbortReason) -> StressStatus {
        if let Some(r) = self.current() {
            if r.trigger_abort(reason) {
                let st = r.status();
                self.events.status(&st);
                return st;
            }
            return r.status();
        }
        StressStatus::idle()
    }

    /// Abort and wait (bounded) for workers to exit. Used on app exit.
    pub fn abort_and_wait(&self, reason: AbortReason, timeout: Duration) -> bool {
        self.abort(reason);
        let deadline = Instant::now() + timeout;
        while let Some(r) = self.current() {
            if r.is_done() {
                return true;
            }
            if Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        true
    }
}

fn supervise(
    shared: Arc<RunShared>,
    events: Arc<dyn StressEvents>,
    handles: Vec<JoinHandle<()>>,
    cpu_var_range: Option<(u8, u8)>,
    core_cycler: Option<(u16, u8)>,
) {
    win::set_current_thread_above_normal();
    if shared.should_continue() {
        shared
            .phase
            .store(StressPhase::Running as u8, Ordering::SeqCst);
    }
    if let Some((n, _)) = core_cycler {
        shared.update_message(format!("Core Cycler · Core 1/{n} (Peak Boost)"));
    }
    events.status(&shared.status());
    let mut sched =
        cpu_var_range.map(|(min, max)| VariableSchedule::with_range(shared.started_ms, min, max));
    let mut last_cycled_core: u16 = 0;
    let (mut last_rate, mut last_emit) = (Instant::now(), Instant::now());
    let mut completed = false;
    loop {
        std::thread::sleep(SUPERVISOR_TICK);
        if shared.is_aborted() || !shared.run.load(Ordering::SeqCst) {
            break;
        }
        if shared.workers_alive.load(Ordering::SeqCst) == 0 {
            if shared.completed_naturally.load(Ordering::Acquire) {
                completed = true;
            } else {
                shared.failed.store(true, Ordering::SeqCst);
                shared.set_message("all workers exited");
            }
            break;
        }
        let el = shared.started.elapsed();
        if let Some((n, step_s)) = core_cycler {
            let idx = ((el.as_secs() / (step_s.max(1) as u64)) % (n.max(1) as u64)) as u16;
            if idx != last_cycled_core {
                last_cycled_core = idx;
                shared.active_worker.store(idx, Ordering::Relaxed);
                shared.update_message(format!("Core Cycler · Core {}/{n} (Peak Boost)", idx + 1));
            }
        }
        if shared.duration.is_some_and(|d| el >= d) {
            completed = true;
            shared.run.store(false, Ordering::SeqCst);
            break;
        }
        if shared.time_cap.is_some_and(|c| el >= c) {
            if shared.trigger_abort(AbortReason::NoThermalTimeCap) {
                events.alert(&SafetyAlert {
                    level: AlertLevel::Critical,
                    metric: AlertMetric::TimeCap,
                    value: el.as_secs_f32(),
                    threshold: NO_THERMAL_TIME_CAP.as_secs_f32(),
                    action: AlertAction::Aborted,
                    timestamp_ms: now_ms(),
                    message: "10-minute cap reached (no CPU thermal protection)".into(),
                });
            }
            break;
        }
        if let Some(ref mut sc) = sched {
            shared.load_level.store(sc.level_at(el), Ordering::Relaxed);
        }
        if last_rate.elapsed() >= Duration::from_millis(500) {
            shared.update_rates();
            last_rate = Instant::now();
        }
        if last_emit.elapsed() >= Duration::from_secs(1) {
            events.status(&shared.status());
            last_emit = Instant::now();
        }
    }
    shared.run.store(false, Ordering::SeqCst);
    let join_start = Instant::now();
    for h in handles {
        if h.join().is_err() {
            shared.failed.store(true, Ordering::SeqCst);
            shared.set_message("a worker thread panicked");
        }
    }
    log::info!("stress workers joined in {:?}", join_start.elapsed());
    shared.finished_elapsed_ms.store(
        shared.started.elapsed().as_millis().max(1) as u64,
        Ordering::Release,
    );
    let final_phase = if shared.is_aborted() {
        StressPhase::Aborted
    } else if shared.failed.load(Ordering::SeqCst) {
        StressPhase::Failed
    } else if completed {
        StressPhase::Completed
    } else {
        StressPhase::Stopped
    };
    shared.phase.store(final_phase as u8, Ordering::SeqCst);
    shared.zero_rates();
    shared.load_level.store(0, Ordering::Relaxed);
    shared.done.store(true, Ordering::Release);
    events.status(&shared.status());
}

struct AliveGuard<'a>(&'a RunShared);
impl Drop for AliveGuard<'_> {
    fn drop(&mut self) {
        self.0.workers_alive.fetch_sub(1, Ordering::SeqCst);
    }
}

fn fft_preset_name(p: FftPreset) -> &'static str {
    match p {
        FftPreset::Smallest => "Smallest",
        FftPreset::Small => "Small",
        FftPreset::Large => "Large",
        FftPreset::Blend => "Blend",
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_worker(
    shared: Arc<RunShared>,
    events: Arc<dyn StressEvents>,
    worker: usize,
    core: usize,
    kernel: Kernel,
    n: usize,
    ntt_n: usize,
    workload: CpuWorkload,
    fft_preset: FftPreset,
    is_core_cycler: bool,
) {
    let _alive = AliveGuard(&shared);
    win::set_current_thread_below_normal();
    win::pin_current_thread(core);
    let keep = || shared.should_continue();

    // In Extreme mode, align auxiliary FMA & NTT footprints with the selected FFT preset so
    // Smallest (L1 Max Heat), Small (L2 Heavy), Large (>L3 IMC/RAM), and Blend feel unmistakably distinct.
    let (eff_mat_n, eff_ntt_n) = if workload == CpuWorkload::Extreme {
        match fft_preset {
            FftPreset::Smallest => (64, 4_096),
            FftPreset::Small => (128, 8_192),
            FftPreset::Large => (512, 131_072),
            FftPreset::Blend => (n, ntt_n),
        }
    } else {
        (n, ntt_n)
    };

    let mut mat = if matches!(workload, CpuWorkload::Extreme | CpuWorkload::MatrixFma) {
        match MatSet::new_interruptible(eff_mat_n, 0x5EED_0000 + worker as u64, keep) {
            Some(m) => Some(m),
            None => return,
        }
    } else {
        None
    };
    let mut ntt = if matches!(workload, CpuWorkload::Extreme | CpuWorkload::NttPrime) {
        match NttSet::new_interruptible(eff_ntt_n, 0xC0DE_0000 + worker as u64, keep) {
            Some(t) => Some(t),
            None => return,
        }
    } else {
        None
    };

    // FFT squaring ladder setup for FftTorture and Extreme modes.
    let uses_fft = matches!(workload, CpuWorkload::Extreme | CpuWorkload::FftTorture);
    let ladder = fft_ladder(fft_preset);
    let mut fft_idx = 0usize;
    let mut fft_set = if uses_fft {
        match FftSet::new_interruptible(ladder[0], kernel, keep) {
            Some(f) => Some(f),
            None => return,
        }
    } else {
        None
    };
    let mut fft_step_start = Instant::now();
    let mut fft_pass = 0u64;
    let mut outer_iter = 0u64;
    let mut last_fft_status = Instant::now() - Duration::from_secs(10);

    let panel = panel_rows(eff_mat_n);
    let flops_per_panel = 2 * (panel as u64) * (eff_mat_n as u64) * (eff_mat_n as u64);
    let mut throttle = Throttle::default();
    let mut ntt_pass = 0u64;
    let is_active_for_cycler = || {
        let active = shared.active_worker.load(Ordering::Relaxed);
        active == u16::MAX || active as usize == worker
    };

    'outer: loop {
        outer_iter = outer_iter.wrapping_add(1);
        // In CoreCycler mode, idle threads sleep so the active core boosts to max single-core Fmax.
        while !is_active_for_cycler() {
            if !shared.should_continue() {
                break 'outer;
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        // FFT squaring burst (Bailey L1D power-virus for Smallest, Fused Radix-4 for Small/Large/Blend).
        if let Some(fft) = fft_set.as_mut() {
            // Advance along the FFT ladder every 6 seconds to sweep all cache & RAM tiers.
            if ladder.len() > 1 && fft_step_start.elapsed() >= Duration::from_secs(6) {
                fft_idx = (fft_idx + 1) % ladder.len();
                let next_len = ladder[fft_idx];
                match FftSet::new_interruptible(next_len, kernel, keep) {
                    Some(new_f) => *fft = new_f,
                    None => break 'outer,
                }
                fft_step_start = Instant::now();
            }
            // Size bursts so L1 Bailey Smallest stays locked in tight FMA loops (~15-30ms) with minimal
            // throttle/atomic overhead while Large out-of-L3 FFTs still respond to abort in <20ms.
            let fft_reps = if fft.n <= 64 {
                512
            } else if fft.n <= 256 {
                256
            } else if fft.n <= 4_096 {
                64
            } else if fft.n <= 16_384 {
                16
            } else {
                2
            };
            let mut max_round = 0.0f64;
            let mut burst_flops = 0u64;
            let flops_each = fft.flops_per_pass();
            for rep in 0..fft_reps {
                // Only check pacing/throttle periodically on micro-sized L1 FFTs when at 100% steady load.
                let lvl = shared.load_level.load(Ordering::Relaxed);
                if lvl < 100 || (rep & 15) == 0 {
                    loop {
                        if !shared.should_continue() {
                            if burst_flops > 0 {
                                shared.flops[worker].fetch_add(burst_flops, Ordering::Relaxed);
                            }
                            break 'outer;
                        }
                        if !is_active_for_cycler() {
                            if burst_flops > 0 {
                                shared.flops[worker].fetch_add(burst_flops, Ordering::Relaxed);
                            }
                            continue 'outer;
                        }
                        let cur_lvl = shared.load_level.load(Ordering::Relaxed);
                        match throttle.pace(cur_lvl, keep) {
                            Pace::Work => break,
                            Pace::Skip => continue,
                            Pace::Stop => {
                                if burst_flops > 0 {
                                    shared.flops[worker].fetch_add(burst_flops, Ordering::Relaxed);
                                }
                                break 'outer;
                            }
                        }
                    }
                }
                let seed = (worker as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(fft_pass);
                match fft.run_and_verify(kernel, seed, || {
                    shared.should_continue() && is_active_for_cycler()
                }) {
                    None => {
                        if burst_flops > 0 {
                            shared.flops[worker].fetch_add(burst_flops, Ordering::Relaxed);
                        }
                        if !shared.should_continue() {
                            break 'outer;
                        }
                        continue 'outer;
                    }
                    Some(Ok(err)) => {
                        max_round = max_round.max(err);
                        burst_flops += flops_each;
                    }
                    Some(Err(FftFailure::Rounding { max_err, idx })) => {
                        burst_flops += flops_each;
                        let sz = fft_size_label(fft.n, fft.lanes);
                        shared.record_error(
                            &*events,
                            worker,
                            Some(core),
                            WorkerErrorKind::ComputeMismatch,
                            format!(
                                "FFT rounding error on logical CPU {core} ({sz}): rounding was {max_err:.4}, expected < 0.40 (idx {idx})"
                            ),
                        );
                    }
                    Some(Err(FftFailure::ConvolutionMismatch {
                        got_dc,
                        expect_dc,
                        got_nyq,
                        expect_nyq,
                    })) => {
                        burst_flops += flops_each;
                        let sz = fft_size_label(fft.n, fft.lanes);
                        shared.record_error(
                            &*events,
                            worker,
                            Some(core),
                            WorkerErrorKind::ComputeMismatch,
                            format!(
                                "FFT convolution sum mismatch on logical CPU {core} ({sz}): DC {got_dc} != {expect_dc}, Nyq {got_nyq} != {expect_nyq}"
                            ),
                        );
                    }
                }
                fft_pass = fft_pass.wrapping_add(1);
            }
            if burst_flops > 0 {
                shared.flops[worker].fetch_add(burst_flops, Ordering::Relaxed);
            }
            if worker == 0
                && !is_core_cycler
                && shared.kind != StressKind::Combined
                && last_fft_status.elapsed() >= Duration::from_millis(800)
            {
                last_fft_status = Instant::now();
                let sz = fft_size_label(fft.n, fft.lanes);
                let label = if workload == CpuWorkload::FftTorture {
                    format!("{} FFTs", fft_preset_name(fft_preset))
                } else {
                    format!("Extreme · {} FFTs", fft_preset_name(fft_preset))
                };
                shared.update_message(format!(
                    "{label} · {sz} ({}) · roundoff {max_round:.5}",
                    kernel.name()
                ));
            }
        }

        if let Some(m) = mat.as_mut() {
            // In Extreme + Smallest run tight L1 FMA matrix bursts to reinforce peak thermal soak;
            // in Extreme + Large keep auxiliary FMA sparse so >L3 FFT dominates IMC/RAM pressure.
            let run_mat = workload != CpuWorkload::Extreme
                || fft_preset != FftPreset::Large
                || outer_iter.is_multiple_of(4);
            let reps = if workload == CpuWorkload::Extreme && eff_mat_n <= 64 {
                8
            } else if workload == CpuWorkload::Extreme && eff_mat_n <= 128 {
                2
            } else {
                1
            };
            if run_mat {
                for _ in 0..reps {
                    for r0 in (0..eff_mat_n).step_by(panel) {
                        loop {
                            if !shared.should_continue() {
                                break 'outer;
                            }
                            if !is_active_for_cycler() {
                                continue 'outer;
                            }
                            let lvl = shared.load_level.load(Ordering::Relaxed);
                            match throttle.pace(lvl, keep) {
                                Pace::Work => break,
                                Pace::Skip => continue,
                                Pace::Stop => break 'outer,
                            }
                        }
                        m.compute_rows(kernel, r0, r0 + panel);
                        shared.flops[worker].fetch_add(flops_per_panel, Ordering::Relaxed);
                    }
                    if !shared.should_continue() {
                        break 'outer;
                    }
                    if !m.verify() {
                        shared.record_error(
                            &*events,
                            worker,
                            Some(core),
                            WorkerErrorKind::ComputeMismatch,
                            format!(
                                "dense-mantissa matrix checksum mismatch on logical CPU {core} ({})",
                                kernel.name()
                            ),
                        );
                    }
                    m.flip_sign();
                }
            }
        }

        if let Some(t) = ntt.as_mut() {
            // Keep scalar-integer NTT infrequent during Extreme+Smallest so L1 FMA power stays pegged.
            let run_ntt = workload != CpuWorkload::Extreme
                || match fft_preset {
                    FftPreset::Smallest => outer_iter.is_multiple_of(16),
                    FftPreset::Small => outer_iter.is_multiple_of(4),
                    FftPreset::Large | FftPreset::Blend => outer_iter.is_multiple_of(2),
                };
            if run_ntt {
                loop {
                    if !shared.should_continue() {
                        break 'outer;
                    }
                    if !is_active_for_cycler() {
                        continue 'outer;
                    }
                    let lvl = shared.load_level.load(Ordering::Relaxed);
                    match throttle.pace(lvl, keep) {
                        Pace::Work => break,
                        Pace::Skip => continue,
                        Pace::Stop => break 'outer,
                    }
                }
                let ok = t.run_and_verify(ntt_pass, || {
                    shared.should_continue() && is_active_for_cycler()
                });
                match ok {
                    None => {
                        if !shared.should_continue() {
                            break 'outer;
                        }
                        continue 'outer;
                    }
                    Some(true) => {
                        shared.flops[worker].fetch_add(t.ops_per_pass(), Ordering::Relaxed);
                    }
                    Some(false) => {
                        shared.flops[worker].fetch_add(t.ops_per_pass(), Ordering::Relaxed);
                        shared.record_error(
                            &*events,
                            worker,
                            Some(core),
                            WorkerErrorKind::ComputeMismatch,
                            format!(
                                "64-bit Goldilocks NTT avalanche mismatch on logical CPU {core} (N={eff_ntt_n}, pass {ntt_pass})"
                            ),
                        );
                    }
                }
                ntt_pass = ntt_pass.wrapping_add(1);
            }
        }
    }
}

struct MemAllocCoord {
    total_workers: usize,
    stage1_done: AtomicU16,
    last_reclaim_ms: AtomicU64,
    last_swap_ms: AtomicU64,
    topup_lock: Mutex<()>,
    target_pct: u8,
    max_pct: u8,
}

struct AllocatedChunks<'a> {
    shared: &'a RunShared,
    /// `(chunk, starting_word_offset)` so mixed-size initial + top-up chunks have unique word IDs.
    chunks: Vec<(Vec<u64>, u64)>,
    next_word: u64,
    bytes: u64,
}

impl AllocatedChunks<'_> {
    fn push(&mut self, c: Vec<u64>) {
        let words = c.len() as u64;
        let sz = words * 8;
        let start = self.next_word;
        self.next_word += words;
        self.bytes += sz;
        self.shared.mem_allocated.fetch_add(sz, Ordering::Relaxed);
        self.chunks.push((c, start));
    }

    fn pop(&mut self) {
        if let Some((c, _)) = self.chunks.pop() {
            let words = c.len() as u64;
            let sz = words * 8;
            self.next_word = self.next_word.saturating_sub(words);
            self.bytes = self.bytes.saturating_sub(sz);
            let prev = self.shared.mem_allocated.load(Ordering::Relaxed);
            self.shared
                .mem_allocated
                .fetch_sub(sz.min(prev), Ordering::Relaxed);
        }
    }
}

impl Drop for AllocatedChunks<'_> {
    fn drop(&mut self) {
        self.chunks.clear();
        self.chunks.shrink_to_fit();
        if self.bytes > 0 {
            let prev = self.shared.mem_allocated.load(Ordering::Relaxed);
            self.shared
                .mem_allocated
                .fetch_sub(self.bytes.min(prev), Ordering::Relaxed);
        }
    }
}

fn sleep_check(total: Duration, shared: &RunShared) {
    let t0 = Instant::now();
    while t0.elapsed() < total && shared.should_continue() {
        let rest = total.saturating_sub(t0.elapsed());
        std::thread::sleep(rest.min(Duration::from_millis(5)));
    }
}

/// Allocates newly freed physical RAM into `alloc` in 16 MiB (bulk) down to 4 MiB (fine tail)
/// chunks without pushing into pagefile swap, holding a constant 99%+ DRAM load at 100% target.
/// Returns `(added_any, hit_chunk_cap)`.
fn reclaim_free_ram(
    shared: &RunShared,
    coord: &MemAllocCoord,
    alloc: &mut AllocatedChunks<'_>,
    max_chunks: usize,
) -> (bool, bool) {
    let mut added_any = false;
    let mut chunks_grabbed = 0usize;
    let eff = coord.target_pct.min(coord.max_pct);

    while chunks_grabbed < max_chunks && shared.should_continue() {
        let grab_res = {
            let _lock = coord.topup_lock.lock();
            let Some((total, avail)) = win::memory_status() else {
                break;
            };
            let want_total = if coord.target_pct >= 100 {
                total
            } else {
                total / 100 * coord.target_pct as u64
            };
            let alloc_now = shared.mem_allocated.load(Ordering::Relaxed);
            if alloc_now >= want_total {
                break;
            }
            let used_pct_x10 = total.saturating_sub(avail) * 1000 / total.max(1);
            let limit_x10 = if eff >= 100 { 996 } else { eff as u64 * 10 };
            if used_pct_x10 >= limit_x10 || avail <= mk::MAX_FLOOR_BYTES {
                break;
            }
            let extra = mk::compute_topup_bytes(total, avail, coord.target_pct, coord.max_pct)
                .min(want_total.saturating_sub(alloc_now));
            if extra < (mk::FINE_TOPUP_CHUNK_BYTES as u64) {
                break;
            }
            // Use 16 MiB chunks when reclaiming large freed blocks, 4 MiB chunks to pin steady 99%+.
            let chunk_cap = if extra >= (64 << 20) {
                mk::TOPUP_CHUNK_BYTES
            } else {
                mk::FINE_TOPUP_CHUNK_BYTES
            };
            let grab = (extra as usize).min(chunk_cap);
            let words = (grab / 8).max(1);
            let ws_before = win::process_working_set_bytes();
            if let Some(c) = mk::try_alloc_chunk(words) {
                alloc.push(c);
                // Check physical residency delta below 256 MiB free: if Windows evicted existing
                // working-set pages to fit this chunk, revert it and back off briefly to avoid swap.
                let swapped = match (
                    win::memory_status(),
                    ws_before,
                    win::process_working_set_bytes(),
                ) {
                    (Some((_, avail_after)), Some(b), Some(a)) if avail_after < (256 << 20) => {
                        a < b + (grab as u64) / 2
                    }
                    _ => false,
                };
                if swapped {
                    alloc.pop();
                    coord.last_swap_ms.store(now_ms(), Ordering::Relaxed);
                    false
                } else {
                    added_any = true;
                    chunks_grabbed += 1;
                    true
                }
            } else {
                false
            }
        };
        if !grab_res {
            break;
        }
        // Yield briefly outside the lock so sibling memory workers grab chunks evenly.
        std::thread::yield_now();
    }
    if added_any {
        win::hint_working_set_bytes(shared.mem_allocated.load(Ordering::Relaxed));
    }
    (added_any, chunks_grabbed >= max_chunks)
}

#[allow(clippy::too_many_arguments)]
fn mem_worker(
    shared: Arc<RunShared>,
    events: Arc<dyn StressEvents>,
    worker: usize,
    pin_core: Option<usize>,
    bytes: u64,
    profile: MemoryProfile,
    pattern_pref: MemoryPattern,
    target_cycles: Option<u16>,
    coord: Arc<MemAllocCoord>,
) {
    let _alive = AliveGuard(&shared);
    win::set_current_thread_below_normal();
    if let Some(core) = pin_core {
        win::pin_current_thread(core);
    }
    let mut alloc = AllocatedChunks {
        shared: &shared,
        chunks: Vec::new(),
        next_word: 0,
        bytes: 0,
    };

    // Stage 1: Commit initial physical RAM target in parallel across workers (ensuring >= 2 disjoint
    // chunks per worker so cross-chunk R<->W turnarounds & opposing-end MirrorMove bypass L3 cache).
    let chunk_cap = mk::worker_chunk_bytes(bytes);
    let mut remaining = bytes as usize;
    while remaining > 0 {
        if !shared.should_continue() {
            return;
        }
        let words = (remaining.min(chunk_cap) / 8).max(1);
        match mk::try_alloc_chunk(words) {
            Some(c) => {
                alloc.push(c);
                remaining = remaining.saturating_sub(words * 8);
            }
            None => {
                let small_words = (remaining.min(4 << 20) / 8).max(1);
                if small_words < words {
                    if let Some(c) = mk::try_alloc_chunk(small_words) {
                        alloc.push(c);
                        remaining = remaining.saturating_sub(small_words * 8);
                        continue;
                    }
                }
                shared.record_error(
                    &*events,
                    worker,
                    None,
                    WorkerErrorKind::AllocFailed,
                    format!(
                        "allocation capped after {} MiB",
                        (bytes as usize - remaining) >> 20
                    ),
                );
                break;
            }
        }
    }

    // Stage 2: Fast standby-cache reclamation (2 short waves so T1 begins hammering DRAM in < 0.5 s).
    coord.stage1_done.fetch_add(1, Ordering::AcqRel);
    if coord.target_pct.min(coord.max_pct) >= 50 {
        while (coord.stage1_done.load(Ordering::Acquire) as usize) < coord.total_workers {
            if !shared.should_continue() {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        if worker == 0 {
            win::hint_working_set_bytes(shared.mem_allocated.load(Ordering::Relaxed));
        }
        for wave in 0..4 {
            if !shared.should_continue() {
                break;
            }
            sleep_check(
                Duration::from_millis(if wave == 0 { 35 } else { 25 }),
                &shared,
            );
            let (added, _) = reclaim_free_ram(&shared, &coord, &mut alloc, 64);
            if !added {
                break;
            }
        }
        coord.last_reclaim_ms.store(now_ms(), Ordering::Relaxed);
    }

    if alloc.chunks.is_empty() {
        shared.set_message("memory worker could not allocate");
        return;
    }
    let base_word = (worker as u64) << 40; // distinct address space per worker for address patterns
    let try_reclaim = |alloc: &mut AllocatedChunks<'_>| {
        if coord.target_pct.min(coord.max_pct) >= 50 {
            let now = now_ms();
            let last_swap = coord.last_swap_ms.load(Ordering::Relaxed);
            // Once physical DRAM ceiling is reached (working-set eviction detected), back off
            // 3 s to avoid alloc->trim->VirtualFree TLB shootdowns that stall sensors, unless
            // >= 128 MiB just freed up (e.g. user closed a browser or background app).
            if now.saturating_sub(last_swap) < 3000 {
                let big_free = win::memory_status().is_some_and(|(_, avail)| avail >= (128 << 20));
                if !big_free {
                    return;
                }
            }
            let prev = coord.last_reclaim_ms.load(Ordering::Relaxed);
            if now.saturating_sub(prev) >= 150
                && coord
                    .last_reclaim_ms
                    .compare_exchange(prev, now, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            {
                let (_, hit_cap) = reclaim_free_ram(&shared, &coord, alloc, 128);
                if hit_cap {
                    // More headroom remains; allow sibling workers to continue immediately.
                    coord.last_reclaim_ms.store(0, Ordering::Relaxed);
                }
            }
        }
    };

    if profile == MemoryProfile::Absolut {
        let keep = || shared.should_continue();
        let add_bytes = |b: u64| {
            shared.bytes[worker].fetch_add(b, Ordering::Relaxed);
        };
        let total_steps = mk::ABSOLUT_SEQUENCE.len();
        let mut cycle = 1u64;
        'absolut: loop {
            for (step_idx, &test_id) in mk::ABSOLUT_SEQUENCE.iter().enumerate() {
                let total_sub = mk::absolut_step_passes_runtime(test_id);
                for sub_pass in 0..total_sub {
                    if !shared.should_continue() {
                        break 'absolut;
                    }
                    // Reclaim newly freed RAM between every sub-pass (strictly before fill) to hold 99%+ flat.
                    try_reclaim(&mut alloc);
                    if worker == 0 {
                        let cyc_tag = match target_cycles {
                            Some(max_c) => format!("C{cycle}/{max_c}"),
                            None => format!("C{cycle}"),
                        };
                        let pass_tag = if total_sub > 1 {
                            format!(" ({}/{total_sub})", sub_pass + 1)
                        } else {
                            String::new()
                        };
                        shared.update_message(format!(
                            "Absolut {cyc_tag} · Step {}/{total_steps} · T{test_id} {}{pass_tag}",
                            step_idx + 1,
                            mk::absolut_test_name(test_id)
                        ));
                    }
                    let on_error = |bad: u64, mm: mk::MemMismatch| {
                        // Add remaining mismatches before record_error so total_errors and error_cutoff see exact count.
                        if bad > 1 {
                            shared.total_errors.fetch_add(bad - 1, Ordering::Relaxed);
                        }
                        let detail = format!(
                            "Absolut C{cycle} T{test_id} p{}/{total_sub} ({}): {bad} word(s) mismatched (chunk {}, offset {} B, exp=0x{:016X}, got=0x{:016X}, diff=0x{:016X})",
                            sub_pass + 1,
                            mk::absolut_test_name(test_id),
                            mm.chunk_idx,
                            mm.word_idx * 8,
                            mm.expected,
                            mm.actual,
                            mm.expected ^ mm.actual
                        );
                        shared.record_error(
                            &*events,
                            worker,
                            None,
                            WorkerErrorKind::MemoryMismatch,
                            detail,
                        );
                    };
                    let step_res = mk::run_absolut_step(
                        &mut alloc.chunks,
                        base_word,
                        cycle - 1,
                        step_idx,
                        sub_pass,
                        test_id,
                        &keep,
                        &add_bytes,
                        &on_error,
                    );
                    if step_res.aborted || !shared.should_continue() {
                        break 'absolut;
                    }
                }
            }
            if let Some(max_c) = target_cycles {
                if cycle >= max_c as u64 {
                    if worker == 0 {
                        shared.update_message(format!(
                            "Absolut completed {max_c} cycle(s) · {} error(s)",
                            shared.total_errors()
                        ));
                    }
                    shared.completed_naturally.store(true, Ordering::Release);
                    break 'absolut;
                }
            }
            cycle += 1;
        }
        return;
    }

    let mut pass = 0u64;
    'outer: loop {
        // Reclaim newly freed RAM (e.g. closed browser/apps) strictly at top of pass BEFORE fill,
        // so new chunks are always written before verification.
        try_reclaim(&mut alloc);

        let p = Pattern::select(pattern_pref, pass);
        // Fill all chunks before verifying so data leaves CPU caches into DRAM.
        for (chunk, chunk_start) in &mut alloc.chunks {
            for (si, step) in chunk.chunks_mut(mk::STEP_WORDS).enumerate() {
                if !shared.should_continue() {
                    break 'outer;
                }
                let base = base_word + *chunk_start + (si * mk::STEP_WORDS) as u64;
                mk::fill_slice(step, p, pass, base);
                shared.bytes[worker].fetch_add((step.len() * 8) as u64, Ordering::Relaxed);
            }
        }
        for (ci, (chunk, chunk_start)) in alloc.chunks.iter().enumerate() {
            for (si, step) in chunk.chunks(mk::STEP_WORDS).enumerate() {
                if !shared.should_continue() {
                    break 'outer;
                }
                let base = base_word + *chunk_start + (si * mk::STEP_WORDS) as u64;
                let (bad, first) = mk::verify_slice(step, p, pass, base);
                shared.bytes[worker].fetch_add((step.len() * 8) as u64, Ordering::Relaxed);
                if bad > 0 {
                    // record_error counts one; account for every mismatching word.
                    shared.total_errors.fetch_add(bad - 1, Ordering::Relaxed);
                    shared.record_error(
                        &*events,
                        worker,
                        None,
                        WorkerErrorKind::MemoryMismatch,
                        format!(
                            "{bad} word(s) mismatched ({}, pass {pass}), first at chunk {ci} offset {} B",
                            p.name(),
                            (si * mk::STEP_WORDS + first.unwrap_or(0)) * 8
                        ),
                    );
                }
            }
        }
        pass += 1;
    }
}

fn gpu_worker(
    shared: Arc<RunShared>,
    events: Arc<dyn StressEvents>,
    gpu: Arc<super::gpu::GpuStress>,
    opts: super::gpu::GpuRunOptions,
) {
    let _alive = AliveGuard(&shared);
    // Keep GPU command submission responsive when CPU stress runs simultaneously (Power mode).
    win::set_current_thread_above_normal();
    gpu.run_gpu(&shared, &*events, opts, |flops| {
        shared.gpu_flops.fetch_add(flops, Ordering::Relaxed);
    });
}

fn vram_worker(
    shared: Arc<RunShared>,
    events: Arc<dyn StressEvents>,
    gpu: Arc<super::gpu::GpuStress>,
    target_bytes: u64,
    chunk_bytes: u64,
    pattern: MemoryPattern,
) {
    let _alive = AliveGuard(&shared);
    gpu.run_vram(
        &shared,
        &*events,
        target_bytes,
        chunk_bytes,
        pattern,
        |bytes| {
            shared.bytes[0].fetch_add(bytes, Ordering::Relaxed);
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> StartContext {
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        StartContext {
            physical_cores: (logical / 2).max(1),
            logical_cores: logical,
            isa: super::super::cpu_kernel::detect_isa(),
            total_mem: 16 << 30,
            available_mem: 8 << 30,
            vram_total: Some(12 << 30),
            vram_free: Some(10 << 30),
            thermal_ok: true,
            acknowledge_no_thermal: false,
            limits: crate::stress::safety::default_limits(),
        }
    }

    fn cpu_cfg(mode: LoadMode, dataset: Dataset) -> StressConfig {
        StressConfig::Cpu {
            threads: None,
            physical_cores_only: false,
            isa: IsaChoice::Auto,
            dataset,
            workload: CpuWorkload::Extreme,
            fft_preset: FftPreset::Small,
            mode,
            variable_min_pct: 20,
            variable_max_pct: 100,
            core_cycle_s: 3,
            stop_on_error: false,
            max_errors: None,
            duration_s: None,
        }
    }

    fn wait_running(r: &RunShared) {
        let t = Instant::now();
        while r.phase() != StressPhase::Running && t.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    static STRESS_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    fn assert_abort_latency(cfg: StressConfig) {
        let _serial = STRESS_TEST_LOCK.lock();
        let m = StressManager::new(Arc::new(NullEvents), None);
        m.start(cfg, ctx()).unwrap();
        let r = m.current().unwrap();
        wait_running(&r);
        std::thread::sleep(Duration::from_millis(300));
        assert!(r.workers_alive.load(Ordering::SeqCst) > 0);
        let t0 = Instant::now();
        m.abort(AbortReason::User);
        while !r.is_done() {
            assert!(
                t0.elapsed() < Duration::from_millis(500),
                "workers did not stop in time"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let lat = t0.elapsed();
        eprintln!("abort -> all joined in {lat:?}");
        assert!(
            lat < Duration::from_millis(100),
            "abort latency {lat:?} >= 100 ms"
        );
        assert_eq!(r.phase(), StressPhase::Aborted);
        assert_eq!(r.abort_reason(), Some(AbortReason::User));
        assert_eq!(r.workers_alive.load(Ordering::SeqCst), 0);
        assert_eq!(r.total_errors(), 0, "healthy CPU should report no errors");
    }

    #[test]
    fn abort_latency_steady_small() {
        assert_abort_latency(cpu_cfg(LoadMode::Steady, Dataset::Small));
    }

    #[test]
    fn abort_latency_steady_medium() {
        assert_abort_latency(cpu_cfg(LoadMode::Steady, Dataset::Medium));
    }

    #[test]
    fn abort_latency_steady_large() {
        assert_abort_latency(cpu_cfg(LoadMode::Steady, Dataset::Large));
    }

    #[test]
    fn abort_latency_variable() {
        assert_abort_latency(cpu_cfg(LoadMode::Variable, Dataset::Small));
    }

    #[test]
    fn abort_latency_memory() {
        assert_abort_latency(StressConfig::Memory {
            threads: Some(2),
            target_pct: 3,
            profile: MemoryProfile::Absolut,
            pattern: MemoryPattern::All,
            cycles: None,
            stop_on_error: false,
            max_errors: None,
            duration_s: None,
        });
    }

    #[test]
    fn reclaim_helper_honors_target_and_releases_on_stop() {
        let _serial = STRESS_TEST_LOCK.lock();
        let m = StressManager::new(Arc::new(NullEvents), None);
        m.start(
            StressConfig::Memory {
                threads: Some(2),
                target_pct: 3,
                profile: MemoryProfile::Standard,
                pattern: MemoryPattern::All,
                cycles: None,
                stop_on_error: false,
                max_errors: None,
                duration_s: None,
            },
            ctx(),
        )
        .unwrap();
        let r = m.current().unwrap();
        wait_running(&r);
        let t = Instant::now();
        while r.mem_allocated.load(Ordering::Relaxed) == 0 && t.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(5));
        }
        let alloc = r.mem_allocated.load(Ordering::Relaxed);
        assert!(alloc > 0, "expected initial Stage-1 memory allocation > 0");
        assert!(alloc <= r.mem_target_bytes + (16 << 20));
        let st = m.stop().unwrap();
        assert_eq!(st.phase, StressPhase::Stopped);
        assert!(r.is_done());
        assert_eq!(r.mem_allocated.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn absolut_memory_cycle_completes_cleanly() {
        let _serial = STRESS_TEST_LOCK.lock();
        let m = StressManager::new(Arc::new(NullEvents), None);
        let mut c = ctx();
        c.total_mem = 4 << 30;
        c.available_mem = 2 << 30;
        m.start(
            StressConfig::Memory {
                threads: Some(c.logical_cores.min(8) as u16),
                target_pct: 4, // ~160 MiB split across up to 8 threads
                profile: MemoryProfile::Absolut,
                pattern: MemoryPattern::All,
                cycles: Some(1),
                stop_on_error: true,
                max_errors: None,
                duration_s: None,
            },
            c,
        )
        .unwrap();
        let r = m.current().unwrap();
        let t = Instant::now();
        while !r.is_done() && t.elapsed() < Duration::from_secs(30) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            r.is_done(),
            "Absolut 1-cycle run did not complete in time (msg={:?})",
            r.message.lock()
        );
        assert_eq!(r.phase(), StressPhase::Completed);
        assert_eq!(r.total_errors(), 0);
        assert_eq!(r.mem_allocated.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn cpu_core_cycler_and_ntt_verify_cleanly() {
        let _serial = STRESS_TEST_LOCK.lock();
        let m = StressManager::new(Arc::new(NullEvents), None);
        let cfg = StressConfig::Cpu {
            threads: Some(2),
            physical_cores_only: true,
            isa: IsaChoice::Auto,
            dataset: Dataset::Small,
            workload: CpuWorkload::Extreme,
            fft_preset: FftPreset::Small,
            mode: LoadMode::CoreCycler,
            variable_min_pct: 20,
            variable_max_pct: 100,
            core_cycle_s: 1,
            stop_on_error: true,
            max_errors: None,
            duration_s: None,
        };
        m.start(cfg, ctx()).unwrap();
        let r = m.current().unwrap();
        wait_running(&r);
        std::thread::sleep(Duration::from_millis(1200));
        assert!(m.abort_and_wait(AbortReason::User, Duration::from_secs(2)));
        assert_eq!(r.total_errors(), 0);
        assert!(sum(&r.flops) > 0);
    }

    #[test]
    fn fft_torture_presets_verify_cleanly() {
        let _serial = STRESS_TEST_LOCK.lock();
        let m = StressManager::new(Arc::new(NullEvents), None);
        for preset in [
            FftPreset::Smallest,
            FftPreset::Small,
            FftPreset::Large,
            FftPreset::Blend,
        ] {
            let cfg = StressConfig::Cpu {
                threads: Some(2),
                physical_cores_only: false,
                isa: IsaChoice::Auto,
                dataset: Dataset::Small,
                workload: CpuWorkload::FftTorture,
                fft_preset: preset,
                mode: LoadMode::Steady,
                variable_min_pct: 20,
                variable_max_pct: 100,
                core_cycle_s: 3,
                stop_on_error: true,
                max_errors: None,
                duration_s: None,
            };
            m.start(cfg, ctx()).unwrap();
            let r = m.current().unwrap();
            wait_running(&r);
            std::thread::sleep(Duration::from_millis(350));
            assert!(m.abort_and_wait(AbortReason::User, Duration::from_secs(2)));
            assert_eq!(
                r.total_errors(),
                0,
                "FFT torture preset {preset:?} had errors"
            );
            assert!(
                sum(&r.flops) > 0,
                "FFT torture preset {preset:?} executed zero FLOPs"
            );
        }
    }

    #[test]
    fn physical_cores_only_caps_thread_count() {
        let _serial = STRESS_TEST_LOCK.lock();
        let m = StressManager::new(Arc::new(NullEvents), None);
        let c = ctx();
        let cfg = StressConfig::Cpu {
            threads: None,
            physical_cores_only: true,
            isa: IsaChoice::Auto,
            dataset: Dataset::Small,
            workload: CpuWorkload::Extreme,
            fft_preset: FftPreset::Small,
            mode: LoadMode::Steady,
            variable_min_pct: 20,
            variable_max_pct: 100,
            core_cycle_s: 3,
            stop_on_error: false,
            max_errors: None,
            duration_s: None,
        };
        let st = m.start(cfg, c.clone()).unwrap();
        assert_eq!(st.cpu_threads as usize, c.physical_cores);
        assert!(m.abort_and_wait(AbortReason::User, Duration::from_secs(2)));
    }

    #[test]
    fn duration_completes_and_rejects_concurrent_start() {
        let m = StressManager::new(Arc::new(NullEvents), None);
        let cfg = StressConfig::Cpu {
            threads: Some(2),
            physical_cores_only: false,
            isa: IsaChoice::Auto,
            dataset: Dataset::Small,
            workload: CpuWorkload::Extreme,
            fft_preset: FftPreset::Small,
            mode: LoadMode::Steady,
            variable_min_pct: 20,
            variable_max_pct: 100,
            core_cycle_s: 3,
            stop_on_error: false,
            max_errors: None,
            duration_s: Some(1),
        };
        m.start(cfg.clone(), ctx()).unwrap();
        assert!(matches!(m.start(cfg.clone(), ctx()), Err(AppError::Busy)));
        let r = m.current().unwrap();
        let t = Instant::now();
        while !r.is_done() && t.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(r.phase(), StressPhase::Completed);
        // A new run can start once the previous one is done.
        m.start(cfg, ctx()).unwrap();
        m.abort(AbortReason::User);
    }

    #[test]
    fn graceful_stop_reports_stopped() {
        let _serial = STRESS_TEST_LOCK.lock();
        let m = StressManager::new(Arc::new(NullEvents), None);
        m.start(cpu_cfg(LoadMode::Steady, Dataset::Small), ctx())
            .unwrap();
        let r = m.current().unwrap();
        wait_running(&r);
        let t0 = Instant::now();
        let st = m.stop().unwrap();
        assert!(
            t0.elapsed() < Duration::from_millis(260),
            "stop took {:?}",
            t0.elapsed()
        );
        assert_eq!(st.phase, StressPhase::Stopped);
        assert!(r.is_done());
        assert_eq!(r.phase(), StressPhase::Stopped);
        assert!(m.stop().is_err());
    }

    #[test]
    fn thermal_policy_requires_ack_and_applies_cap() {
        let m = StressManager::new(Arc::new(NullEvents), None);
        let mut c = ctx();
        c.thermal_ok = false;
        assert!(matches!(
            m.start(cpu_cfg(LoadMode::Steady, Dataset::Small), c.clone()),
            Err(AppError::ThermalUnavailable)
        ));
        // Memory-only is allowed without ack.
        let mem = StressConfig::Memory {
            threads: Some(1),
            target_pct: 2,
            profile: MemoryProfile::Standard,
            pattern: MemoryPattern::WalkingOnes,
            cycles: None,
            stop_on_error: false,
            max_errors: None,
            duration_s: Some(1),
        };
        let st = m.start(mem, c.clone()).unwrap();
        assert_eq!(st.time_cap_s, None);
        m.abort_and_wait(AbortReason::User, Duration::from_secs(2));
        c.acknowledge_no_thermal = true;
        let st = m
            .start(cpu_cfg(LoadMode::Steady, Dataset::Small), c)
            .unwrap();
        assert_eq!(st.time_cap_s, Some(600));
        assert!(!st.thermal_protection);
        assert!(m.abort_and_wait(AbortReason::User, Duration::from_secs(2)));
    }

    #[test]
    fn invalid_threads_rejected() {
        let m = StressManager::new(Arc::new(NullEvents), None);
        let cfg = StressConfig::Cpu {
            threads: Some(0),
            physical_cores_only: false,
            isa: IsaChoice::Auto,
            dataset: Dataset::Small,
            workload: CpuWorkload::Extreme,
            fft_preset: FftPreset::Small,
            mode: LoadMode::Steady,
            variable_min_pct: 20,
            variable_max_pct: 100,
            core_cycle_s: 3,
            stop_on_error: false,
            max_errors: None,
            duration_s: None,
        };
        assert!(matches!(
            m.start(cfg, ctx()),
            Err(AppError::InvalidArgument(_))
        ));
    }

    #[test]
    fn max_errors_cutoff_aborts_at_threshold() {
        let m = StressManager::new(Arc::new(NullEvents), None);
        let cfg = StressConfig::Cpu {
            threads: Some(1),
            physical_cores_only: false,
            isa: IsaChoice::Auto,
            dataset: Dataset::Small,
            workload: CpuWorkload::Extreme,
            fft_preset: FftPreset::Small,
            mode: LoadMode::Steady,
            variable_min_pct: 20,
            variable_max_pct: 100,
            core_cycle_s: 3,
            stop_on_error: false,
            max_errors: Some(3),
            duration_s: None,
        };
        m.start(cfg, ctx()).unwrap();
        let r = m.current().unwrap();
        wait_running(&r);
        r.record_error(
            &NullEvents,
            0,
            Some(0),
            WorkerErrorKind::ComputeMismatch,
            "e1".into(),
        );
        assert!(!r.is_aborted());
        r.record_error(
            &NullEvents,
            0,
            Some(0),
            WorkerErrorKind::ComputeMismatch,
            "e2".into(),
        );
        assert!(!r.is_aborted());
        r.record_error(
            &NullEvents,
            0,
            Some(0),
            WorkerErrorKind::ComputeMismatch,
            "e3".into(),
        );
        assert!(r.is_aborted());
        assert_eq!(r.abort_reason(), Some(AbortReason::WorkerError));
        assert!(m.abort_and_wait(AbortReason::User, Duration::from_secs(2)));
    }

    #[test]
    fn gpu_profiles_and_vram_verify_and_abort_cleanly() {
        let _serial = STRESS_TEST_LOCK.lock();
        let Some(gpu) = super::super::gpu::GpuStress::probe().map(Arc::new) else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let m = StressManager::new(Arc::new(NullEvents), Some(gpu));
        for (profile, mode) in [
            (GpuProfile::Standard, LoadMode::Steady),
            (GpuProfile::Heavy, LoadMode::Steady),
            (GpuProfile::Extreme, LoadMode::Steady),
            (GpuProfile::Extreme, LoadMode::Variable),
        ] {
            let gcfg = StressConfig::Gpu {
                mode,
                profile,
                intensity: 100,
                variable_min_pct: 30,
                variable_max_pct: 100,
                variable_step_s: 2,
                error_check: true,
                stop_on_error: true,
                max_errors: None,
                duration_s: None,
            };
            m.start(gcfg, ctx()).unwrap();
            let r = m.current().unwrap();
            wait_running(&r);
            std::thread::sleep(Duration::from_millis(600));
            let t0 = Instant::now();
            assert!(m.abort_and_wait(AbortReason::User, Duration::from_secs(2)));
            assert!(
                t0.elapsed() < Duration::from_millis(350),
                "gpu {profile:?} abort took {:?}",
                t0.elapsed()
            );
            assert_eq!(
                r.phase(),
                StressPhase::Aborted,
                "gpu {profile:?} failed: {:?}",
                r.message.lock()
            );
            assert_eq!(r.total_errors(), 0, "gpu {profile:?} had errors");
            assert!(
                r.gpu_flops.load(Ordering::Relaxed) > 0,
                "gpu {profile:?} must execute real FLOPs"
            );
        }
        // VRAM (small 10% slice so unit test is quick)
        let vcfg = StressConfig::Vram {
            target_pct: 10,
            pattern: MemoryPattern::All,
            stop_on_error: true,
            max_errors: None,
            duration_s: None,
        };
        m.start(vcfg, ctx()).unwrap();
        let r = m.current().unwrap();
        wait_running(&r);
        std::thread::sleep(Duration::from_millis(800));
        assert!(m.abort_and_wait(AbortReason::User, Duration::from_secs(2)));
        assert_eq!(
            r.phase(),
            StressPhase::Aborted,
            "VRAM worker failed: {:?}",
            r.message.lock()
        );
        assert_eq!(r.total_errors(), 0);
        assert!(
            sum(&r.bytes) > 0,
            "VRAM worker must actually read/write bytes"
        );
        assert_eq!(
            r.vram_allocated.load(Ordering::Relaxed),
            0,
            "VRAM buffers must be released on exit"
        );
    }
}
