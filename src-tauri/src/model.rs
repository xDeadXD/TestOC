//! Serializable data model shared with the frontend.
//!
//! These types are the single source of truth for the IPC schema. `ts-rs` generates the
//! TypeScript mirrors into `src/bindings/` when `cargo test` runs (see `.cargo/config.toml`).
//! Every `u64` is exported as `number`; values stay well below 2^53 in practice.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

// System info

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct IsaFlags {
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
}

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    #[ts(type = "number | null")]
    pub vram_total_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct SystemInfo {
    pub app_version: String,
    pub hostname: String,
    pub os_version: String,
    pub cpu_brand: String,
    pub physical_cores: u16,
    pub logical_cores: u16,
    pub base_mhz: u32,
    pub isa: IsaFlags,
    #[ts(type = "number")]
    pub total_memory_bytes: u64,
    pub gpus: Vec<GpuInfo>,
    /// Name of the wgpu adapter used for the GPU stress test, if one was found.
    pub gpu_stress_adapter: Option<String>,
    #[ts(type = "number | null")]
    pub gpu_vram_total_bytes: Option<u64>,
    pub pdh_clocks_available: bool,
    pub nvml_available: bool,
}

// Telemetry

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum ClockSource {
    Lhm,
    Pdh,
    Base,
}

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct CoreMetrics {
    pub id: u16,
    pub load_pct: f32,
    pub effective_mhz: Option<f32>,
    pub clock_source: ClockSource,
    pub vid: Option<f32>,
}

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct CpuMetrics {
    pub total_load_pct: f32,
    pub avg_effective_mhz: Option<f32>,
    pub max_effective_mhz: Option<f32>,
    /// Tctl/Tdie from LibreHardwareMonitor: the safety-watchdog source.
    pub package_temp_c: Option<f32>,
    pub ccd_temp_c: Option<f32>,
    pub package_power_w: Option<f32>,
    pub core_voltage_v: Option<f32>,
    /// Active CPU thermal/power throttling indicators (from LHM / thermal proximity).
    pub throttle_flags: Vec<String>,
    pub cores: Vec<CoreMetrics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct BoardSensor {
    pub name: String,
    pub value: f32,
}

#[derive(Debug, Clone, Default, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct BoardMetrics {
    pub temps: Vec<BoardSensor>,
    pub fans: Vec<BoardSensor>,
}

#[derive(Debug, Clone, Default, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct MemoryMetrics {
    #[ts(type = "number")]
    pub total_bytes: u64,
    #[ts(type = "number")]
    pub used_bytes: u64,
    #[ts(type = "number")]
    pub available_bytes: u64,
    pub used_pct: f32,
    #[ts(type = "number")]
    pub swap_total_bytes: u64,
    #[ts(type = "number")]
    pub swap_used_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct GpuMetrics {
    pub index: u32,
    pub name: String,
    pub temp_c: Option<f32>,
    pub util_pct: Option<f32>,
    pub mem_util_pct: Option<f32>,
    pub core_mhz: Option<f32>,
    pub mem_mhz: Option<f32>,
    pub power_w: Option<f32>,
    pub power_limit_w: Option<f32>,
    #[ts(type = "number | null")]
    pub vram_used_bytes: Option<u64>,
    #[ts(type = "number | null")]
    pub vram_total_bytes: Option<u64>,
    pub fan_pct: Option<f32>,
    pub temp_slowdown_c: Option<f32>,
    /// Active non-idle NVML clock throttle reasons (e.g. "Power Cap", "Thermal Slowdown").
    pub throttle_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct StressSnapshot {
    pub phase: StressPhase,
    pub kind: Option<StressKind>,
    pub elapsed_s: f32,
    pub gflops: f64,
    pub gpu_gflops: f64,
    pub mem_mbps: f64,
    #[ts(type = "number")]
    pub errors: u64,
    /// Variable mode: current target load level (0-100). 100 for steady mode.
    pub load_level: u8,
}

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct TelemetrySample {
    #[ts(type = "number")]
    pub seq: u64,
    #[ts(type = "number")]
    pub timestamp_ms: u64,
    pub cpu: CpuMetrics,
    pub board: BoardMetrics,
    pub memory: MemoryMetrics,
    pub gpus: Vec<GpuMetrics>,
    pub stress: StressSnapshot,
    pub sidecar: SidecarState,
    pub poll_duration_us: u32,
}

// Sidecar

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum SidecarState {
    NotStarted,
    Launching,
    Declined,
    Initializing,
    Ok,
    Stale,
    PawnIoMissing,
    Incompatible,
    Error,
}

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct SidecarStatus {
    pub state: SidecarState,
    pub message: Option<String>,
    pub writer_pid: Option<u32>,
}

// Stress configuration & status

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum IsaChoice {
    Auto,
    Scalar,
    Avx2Fma,
    Avx512,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum Dataset {
    /// 128x128 f64 matrices, L2-resident: maximum heat / FPU load.
    #[default]
    Small,
    /// 512x512 f64 matrices, L3-resident: balances cache + FPU stress.
    Medium,
    /// 1024x1024 f64 matrices, streams from RAM: stresses the memory controller too.
    Large,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum LoadMode {
    #[default]
    Steady,
    Variable,
    /// Single-core peak-boost hopping across physical cores (exposes Curve Optimizer / Fmax V/F instability).
    CoreCycler,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum FftPreset {
    /// 4 KiB – 16 KiB/thread (L1D cache): Bailey 4-Step FMA power-virus for peak heat, PPT & Vcore stress.
    Smallest,
    /// 64 KiB – 256 KiB/thread (L2 cache): sustained Small FFTs thermal soak across L1/L2 + FMA.
    #[default]
    Small,
    /// 4 MiB – 16 MiB/thread (exceeds L3): Large FFTs hammering IMC, FCLK/UCLK & system RAM.
    Large,
    /// Full Smallest -> Large FFT ladder sweeping L1, L2, L3, IMC, and RAM together.
    Blend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum CpuWorkload {
    /// Interleaved FFT Torture + dense-mantissa FMA GEMM + 64-bit Goldilocks NTT for maximum fault coverage.
    #[default]
    Extreme,
    /// Pure AVX2+FMA / AVX-512F floating-point FFT squaring with < 0.40 rounding & convolution checks.
    FftTorture,
    /// Dense-mantissa FP64 FMA matrix multiply (peak FPU thermal/current draw).
    MatrixFma,
    /// Pure 64-bit Goldilocks Number Theoretic Transform (integer ALU/MUL & cache avalanche).
    NttPrime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum MemoryProfile {
    /// Full 32-step Absolut harsh DRAM & IMC sequence (NT-streaming, MirrorMove128, Refresh/Rowhammer, cross-checks).
    #[default]
    Absolut,
    /// Classic selectable bit-exact pattern runner (All 5 or single pattern).
    Standard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum MemoryPattern {
    /// Rotate all deterministic DRAM tests (Moving Inversions, Strided Bank-Conflict, Walking 1/0, Address, Random).
    #[default]
    All,
    /// Alternating 0x55/0xAA and 0x00/0xFF burst toggling (maximum DQ bus switching noise).
    MovingInversions,
    /// Coprime strided row/bank-conflict sweep defeating hardware prefetchers (stresses tRCD/tRP/tRFC/IMC).
    BlockStrided,
    WalkingOnes,
    AddressInAddress,
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum GpuProfile {
    /// Balanced verified FP32 FMA compute (moderate power).
    Standard,
    /// Double-buffered multi-unit compute: shared-mem tiled FMA + INT32 + GDDR streaming.
    Heavy,
    /// Full-TDP offscreen 3D render (2048x2048 4xMRT + texture + SFU/FMA + ROPs) + dual-frame pixel compare + FMA verify.
    #[default]
    Extreme,
}

fn default_true() -> bool {
    true
}
fn default_var_min() -> u8 {
    20
}
fn default_var_max() -> u8 {
    100
}
fn default_var_step_s() -> u8 {
    5
}
fn default_core_cycle_s() -> u8 {
    3
}
fn default_isa_auto() -> IsaChoice {
    IsaChoice::Auto
}
fn default_fft_blend() -> FftPreset {
    FftPreset::Blend
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
#[ts(export)]
pub enum StressConfig {
    Cpu {
        threads: Option<u16>,
        #[serde(default)]
        physical_cores_only: bool,
        isa: IsaChoice,
        dataset: Dataset,
        #[serde(default)]
        workload: CpuWorkload,
        #[serde(default)]
        fft_preset: FftPreset,
        mode: LoadMode,
        #[serde(default = "default_var_min")]
        variable_min_pct: u8,
        #[serde(default = "default_var_max")]
        variable_max_pct: u8,
        #[serde(default = "default_core_cycle_s")]
        core_cycle_s: u8,
        stop_on_error: bool,
        #[serde(default)]
        #[ts(type = "number | null")]
        max_errors: Option<u64>,
        duration_s: Option<u32>,
    },
    Memory {
        threads: Option<u16>,
        /// Percent of TOTAL RAM to allocate (further capped by safety limits).
        target_pct: u8,
        #[serde(default)]
        profile: MemoryProfile,
        #[serde(default)]
        pattern: MemoryPattern,
        #[serde(default)]
        cycles: Option<u16>,
        stop_on_error: bool,
        #[serde(default)]
        #[ts(type = "number | null")]
        max_errors: Option<u64>,
        duration_s: Option<u32>,
    },
    Combined {
        cpu_threads: Option<u16>,
        #[serde(default = "default_isa_auto")]
        cpu_isa: IsaChoice,
        #[serde(default)]
        cpu_dataset: Dataset,
        #[serde(default)]
        cpu_workload: CpuWorkload,
        #[serde(default = "default_fft_blend")]
        cpu_fft_preset: FftPreset,
        mem_target_pct: u8,
        #[serde(default)]
        mem_profile: MemoryProfile,
        stop_on_error: bool,
        #[serde(default)]
        #[ts(type = "number | null")]
        max_errors: Option<u64>,
        duration_s: Option<u32>,
    },
    Gpu {
        #[serde(default)]
        mode: LoadMode,
        #[serde(default)]
        profile: GpuProfile,
        /// 10-100: target load intensity in Steady mode.
        intensity: u8,
        #[serde(default = "default_var_min")]
        variable_min_pct: u8,
        #[serde(default = "default_var_max")]
        variable_max_pct: u8,
        #[serde(default = "default_var_step_s")]
        variable_step_s: u8,
        #[serde(default = "default_true")]
        error_check: bool,
        #[serde(default)]
        stop_on_error: bool,
        #[serde(default)]
        #[ts(type = "number | null")]
        max_errors: Option<u64>,
        duration_s: Option<u32>,
    },
    Vram {
        /// 10-100: percent of total VRAM to allocate (with lean DWM headroom & OOM clamping).
        target_pct: u8,
        #[serde(default)]
        pattern: MemoryPattern,
        #[serde(default)]
        stop_on_error: bool,
        #[serde(default)]
        #[ts(type = "number | null")]
        max_errors: Option<u64>,
        duration_s: Option<u32>,
    },
    /// Combined Power (PSU) test: simultaneous CPU workload + full-TDP GPU workload.
    Power {
        cpu_threads: Option<u16>,
        #[serde(default = "default_isa_auto")]
        cpu_isa: IsaChoice,
        #[serde(default)]
        cpu_dataset: Dataset,
        #[serde(default)]
        cpu_workload: CpuWorkload,
        #[serde(default)]
        cpu_fft_preset: FftPreset,
        #[serde(default)]
        gpu_profile: GpuProfile,
        /// 10-100: GPU intensity.
        gpu_intensity: u8,
        #[serde(default)]
        stop_on_error: bool,
        #[serde(default)]
        #[ts(type = "number | null")]
        max_errors: Option<u64>,
        duration_s: Option<u32>,
    },
}

impl StressConfig {
    pub fn kind(&self) -> StressKind {
        match self {
            StressConfig::Cpu { .. } => StressKind::Cpu,
            StressConfig::Memory { .. } => StressKind::Memory,
            StressConfig::Combined { .. } => StressKind::Combined,
            StressConfig::Gpu { .. } => StressKind::Gpu,
            StressConfig::Vram { .. } => StressKind::Vram,
            StressConfig::Power { .. } => StressKind::Power,
        }
    }
    pub fn duration_s(&self) -> Option<u32> {
        match self {
            StressConfig::Cpu { duration_s, .. }
            | StressConfig::Memory { duration_s, .. }
            | StressConfig::Combined { duration_s, .. }
            | StressConfig::Gpu { duration_s, .. }
            | StressConfig::Vram { duration_s, .. }
            | StressConfig::Power { duration_s, .. } => *duration_s,
        }
    }
    pub fn stop_on_error(&self) -> bool {
        match self {
            StressConfig::Cpu { stop_on_error, .. }
            | StressConfig::Memory { stop_on_error, .. }
            | StressConfig::Combined { stop_on_error, .. }
            | StressConfig::Gpu { stop_on_error, .. }
            | StressConfig::Vram { stop_on_error, .. }
            | StressConfig::Power { stop_on_error, .. } => *stop_on_error,
        }
    }
    /// Effective critical error cutoff: `Some(1)` if `stop_on_error`, else `max_errors` (> 0).
    pub fn error_cutoff(&self) -> Option<u64> {
        if self.stop_on_error() {
            return Some(1);
        }
        let max = match self {
            StressConfig::Cpu { max_errors, .. }
            | StressConfig::Memory { max_errors, .. }
            | StressConfig::Combined { max_errors, .. }
            | StressConfig::Gpu { max_errors, .. }
            | StressConfig::Vram { max_errors, .. }
            | StressConfig::Power { max_errors, .. } => *max_errors,
        };
        max.filter(|&n| n > 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum StressKind {
    Cpu,
    Memory,
    Combined,
    Gpu,
    Vram,
    Power,
}

impl StressKind {
    /// Whether this kind loads the CPU cores (and therefore needs CPU thermal protection).
    pub fn loads_cpu(self) -> bool {
        matches!(
            self,
            StressKind::Cpu | StressKind::Combined | StressKind::Power
        )
    }
    pub fn uses_memory(self) -> bool {
        matches!(self, StressKind::Memory | StressKind::Combined)
    }
    pub fn loads_gpu(self) -> bool {
        matches!(self, StressKind::Gpu | StressKind::Vram | StressKind::Power)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
#[repr(u8)]
pub enum StressPhase {
    Idle = 0,
    Starting = 1,
    Running = 2,
    Stopping = 3,
    Completed = 4,
    Stopped = 5,
    Aborted = 6,
    Failed = 7,
}

impl StressPhase {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Starting,
            2 => Self::Running,
            3 => Self::Stopping,
            4 => Self::Completed,
            5 => Self::Stopped,
            6 => Self::Aborted,
            7 => Self::Failed,
            _ => Self::Idle,
        }
    }
    pub fn is_active(self) -> bool {
        matches!(self, Self::Starting | Self::Running | Self::Stopping)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
#[repr(u8)]
pub enum AbortReason {
    User = 1,
    CpuTemp = 2,
    GpuTemp = 3,
    MemoryPressure = 4,
    SensorStall = 5,
    WorkerError = 6,
    NoThermalTimeCap = 7,
    AppExit = 8,
}

impl AbortReason {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            1 => Self::User,
            2 => Self::CpuTemp,
            3 => Self::GpuTemp,
            4 => Self::MemoryPressure,
            5 => Self::SensorStall,
            6 => Self::WorkerError,
            7 => Self::NoThermalTimeCap,
            8 => Self::AppExit,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct StressStatus {
    pub phase: StressPhase,
    pub kind: Option<StressKind>,
    pub config: Option<StressConfig>,
    pub abort_reason: Option<AbortReason>,
    pub message: Option<String>,
    pub elapsed_s: f32,
    pub duration_s: Option<u32>,
    /// Hard cap applied when thermal protection was unavailable at start.
    pub time_cap_s: Option<u32>,
    pub thermal_protection: bool,
    pub cpu_threads: u16,
    pub mem_threads: u16,
    pub isa_used: Option<String>,
    #[ts(type = "number")]
    pub mem_target_bytes: u64,
    #[ts(type = "number")]
    pub mem_allocated_bytes: u64,
    #[ts(type = "number")]
    pub vram_target_bytes: u64,
    #[ts(type = "number")]
    pub vram_allocated_bytes: u64,
    pub gflops: f64,
    pub gpu_gflops: f64,
    pub mem_mbps: f64,
    #[ts(type = "number")]
    pub total_errors: u64,
    #[ts(type = "number")]
    pub gpu_errors: u64,
    /// Per logical core (CPU workers) error counters.
    #[ts(type = "number[]")]
    pub core_errors: Vec<u64>,
    pub load_level: u8,
    pub workers_alive: u16,
}

impl StressStatus {
    pub fn idle() -> Self {
        Self {
            phase: StressPhase::Idle,
            kind: None,
            config: None,
            abort_reason: None,
            message: None,
            elapsed_s: 0.0,
            duration_s: None,
            time_cap_s: None,
            thermal_protection: false,
            cpu_threads: 0,
            mem_threads: 0,
            isa_used: None,
            mem_target_bytes: 0,
            mem_allocated_bytes: 0,
            vram_target_bytes: 0,
            vram_allocated_bytes: 0,
            gflops: 0.0,
            gpu_gflops: 0.0,
            mem_mbps: 0.0,
            total_errors: 0,
            gpu_errors: 0,
            core_errors: Vec::new(),
            load_level: 0,
            workers_alive: 0,
        }
    }
}

// Safety & alerts

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct SafetyLimits {
    pub cpu_warn_c: f32,
    pub cpu_abort_c: f32,
    pub gpu_warn_c: f32,
    pub gpu_abort_c: f32,
    /// Max system memory usage (percent of total, 10-100; 100 = uncapped) allowed during memory stress.
    pub mem_max_pct: u8,
    /// When false, temperature thresholds emit alerts only and do not auto-abort tests.
    #[serde(default = "default_true")]
    pub thermal_abort_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum AlertLevel {
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum AlertMetric {
    CpuTemp,
    GpuTemp,
    Memory,
    SensorStall,
    TimeCap,
    WorkerError,
    Throttle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum AlertAction {
    None,
    Aborted,
}

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct SafetyAlert {
    pub level: AlertLevel,
    pub metric: AlertMetric,
    pub value: f32,
    pub threshold: f32,
    pub action: AlertAction,
    #[ts(type = "number")]
    pub timestamp_ms: u64,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum WorkerErrorKind {
    ComputeMismatch,
    MemoryMismatch,
    AllocFailed,
    GpuMismatch,
    VramMismatch,
    GpuDevice,
}

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct WorkerError {
    pub worker_id: u16,
    pub core_id: Option<u16>,
    pub kind: WorkerErrorKind,
    pub detail: String,
    #[ts(type = "number")]
    pub timestamp_ms: u64,
}

// Sessions / export

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct SessionInfo {
    #[ts(type = "number")]
    pub id: u64,
    pub label: String,
    #[ts(type = "number")]
    pub started_ms: u64,
    #[ts(type = "number | null")]
    pub stopped_ms: Option<u64>,
    pub samples: u32,
    pub recording: bool,
    pub auto: bool,
}

#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ExportResult {
    pub csv_path: String,
    pub summary_path: String,
    pub rows: u32,
    #[ts(type = "number")]
    pub bytes: u64,
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
