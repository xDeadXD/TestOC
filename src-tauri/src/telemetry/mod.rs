//! Telemetry engine: assembles one `TelemetrySample` per tick from sysinfo, PDH, NVML and the
//! LibreHardwareMonitor sidecar frame. Every provider failure degrades to `None`, never a panic.

pub mod gpu_nvml;
pub mod lhm_map;
pub mod pdh_clock;

use std::time::Instant;

use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

use crate::model::*;
use crate::sidecar::layout::SensorFrame;
use gpu_nvml::NvmlGpus;
use pdh_clock::PdhClock;

pub struct TelemetryEngine {
    sys: System,
    pdh: Option<PdhClock>,
    nvml: Option<NvmlGpus>,
    fallback_gpu_name: Option<String>,
    logical: usize,
    physical: usize,
    base_mhz: u32,
    cpu_brand: String,
    seq: u64,
}

impl TelemetryEngine {
    pub fn new() -> Self {
        let mut sys = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        )
        .expect("failed to initialize sysinfo::System");
        // First usage refresh primes the deltas.
        sys.refresh_cpu_usage();
        let logical = sys.cpus().len().max(1);
        let physical = System::physical_core_count().unwrap_or(logical).max(1);
        let base_mhz = sys
            .cpus()
            .first()
            .map(|c| c.frequency() as u32)
            .unwrap_or(0);
        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .unwrap_or_default();
        Self {
            sys,
            pdh: PdhClock::new(),
            nvml: NvmlGpus::new(),
            fallback_gpu_name: None,
            logical,
            physical,
            base_mhz,
            cpu_brand,
            seq: 0,
        }
    }

    pub fn set_fallback_gpu_name(&mut self, adapter_name: Option<&str>) {
        self.fallback_gpu_name = adapter_name
            .map(|n| n.trim_end_matches(" (Vulkan)").trim().to_string())
            .filter(|n| !n.is_empty());
    }

    pub fn logical_cores(&self) -> usize {
        self.logical
    }

    pub fn system_info(
        &self,
        gpu_stress_adapter: Option<String>,
        fallback_vram_bytes: Option<u64>,
    ) -> SystemInfo {
        let mut gpus = self.nvml.as_ref().map(|n| n.info()).unwrap_or_default();
        if gpus.is_empty() {
            if let Some(ref adapter_name) = gpu_stress_adapter {
                gpus.push(GpuInfo {
                    index: 0,
                    name: adapter_name.trim_end_matches(" (Vulkan)").to_string(),
                    vram_total_bytes: fallback_vram_bytes,
                });
            }
        }
        let gpu_vram_total_bytes = gpus
            .first()
            .and_then(|g| g.vram_total_bytes)
            .or(fallback_vram_bytes);
        SystemInfo {
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            hostname: System::host_name().unwrap_or_default(),
            os_version: System::long_os_version().unwrap_or_default(),
            cpu_brand: self.cpu_brand.clone(),
            physical_cores: self.physical as u16,
            logical_cores: self.logical as u16,
            base_mhz: self.base_mhz,
            isa: crate::stress::cpu_kernel::detect_isa(),
            total_memory_bytes: self.sys.total_memory(),
            gpus,
            gpu_stress_adapter,
            gpu_vram_total_bytes,
            pdh_clocks_available: self.pdh.is_some(),
            nvml_available: self.nvml.is_some(),
        }
    }

    pub fn memory_now(&mut self) -> MemoryMetrics {
        self.sys.refresh_memory();
        self.memory_metrics()
    }

    fn memory_metrics(&self) -> MemoryMetrics {
        let total = self.sys.total_memory();
        let avail = self.sys.available_memory();
        let used = total.saturating_sub(avail);
        MemoryMetrics {
            total_bytes: total,
            used_bytes: used,
            available_bytes: avail,
            used_pct: if total > 0 {
                used as f32 * 100.0 / total as f32
            } else {
                0.0
            },
            swap_total_bytes: self.sys.total_swap(),
            swap_used_bytes: self.sys.used_swap(),
        }
    }

    pub fn poll(
        &mut self,
        frame: Option<&SensorFrame>,
        sidecar: SidecarState,
        stress: StressSnapshot,
    ) -> TelemetrySample {
        let t0 = Instant::now();
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();

        let pdh = self
            .pdh
            .as_mut()
            .map(|p| p.sample(self.logical))
            .unwrap_or_default();
        let lhm = frame.map(lhm_map::map_frame).unwrap_or_default();
        let smt = self.logical / self.physical.max(1);
        let lhm_clock_usable =
            !lhm.core_clocks.is_empty() && smt >= 1 && lhm.core_clocks.len() * smt == self.logical;

        let cores: Vec<CoreMetrics> = self
            .sys
            .cpus()
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let phys = i / smt.max(1);
                let (mhz, src) = if lhm_clock_usable && lhm.core_clocks[phys].is_some() {
                    (lhm.core_clocks[phys], ClockSource::Lhm)
                } else if let Some(Some(m)) = pdh.get(i) {
                    (Some(*m), ClockSource::Pdh)
                } else {
                    (Some(c.frequency() as f32), ClockSource::Base)
                };
                CoreMetrics {
                    id: i as u16,
                    load_pct: c.usage(),
                    effective_mhz: mhz,
                    clock_source: src,
                    vid: lhm.core_vids.get(phys).copied().flatten(),
                }
            })
            .collect();
        let mhz: Vec<f32> = cores.iter().filter_map(|c| c.effective_mhz).collect();
        let mut throttle_flags = lhm.cpu_throttle_flags;
        // Only emit a synthetic thermal throttle indicator at universal Tjmax proximity (>=100 °C)
        // when the CPU did not already expose a dedicated hardware Factor throttle sensor.
        if lhm.package_temp_c.is_some_and(|t| t >= 100.0)
            && !throttle_flags.iter().any(|f| f == "Thermal Limit")
        {
            throttle_flags.push("Thermal Limit".into());
        }
        let cpu = CpuMetrics {
            total_load_pct: self.sys.global_cpu_usage(),
            avg_effective_mhz: (!mhz.is_empty())
                .then(|| mhz.iter().sum::<f32>() / mhz.len() as f32),
            max_effective_mhz: mhz.iter().copied().reduce(f32::max),
            package_temp_c: lhm.package_temp_c,
            ccd_temp_c: lhm.ccd_temp_c,
            package_power_w: lhm.package_power_w,
            core_voltage_v: lhm.core_voltage_v,
            throttle_flags,
            cores,
        };
        let mut gpus = self.nvml.as_ref().map(|n| n.sample()).unwrap_or_default();
        if gpus.is_empty() {
            if let Some(mut lhm_gpu) = lhm.gpu {
                if let Some(ref name) = self.fallback_gpu_name {
                    lhm_gpu.name = name.clone();
                }
                gpus.push(lhm_gpu);
            }
        }
        self.seq += 1;
        TelemetrySample {
            seq: self.seq,
            timestamp_ms: now_ms(),
            cpu,
            board: BoardMetrics {
                temps: lhm.board_temps,
                fans: lhm.board_fans,
            },
            memory: self.memory_metrics(),
            gpus,
            stress,
            sidecar,
            poll_duration_us: t0.elapsed().as_micros().min(u32::MAX as u128) as u32,
        }
    }
}

impl Default for TelemetryEngine {
    fn default() -> Self {
        Self::new()
    }
}
