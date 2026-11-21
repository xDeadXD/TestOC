//! NVIDIA GPU telemetry through NVML (loads the driver's nvml.dll at runtime).

use nvml_wrapper::bitmasks::device::ThrottleReasons;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor, TemperatureThreshold};
use nvml_wrapper::Nvml;

use crate::model::{GpuInfo, GpuMetrics};

/// Decode NVML throttle reason bitmask into human-readable active non-idle flags.
pub fn decode_throttle_reasons(r: ThrottleReasons) -> Vec<String> {
    let mut out = Vec::new();
    if r.contains(ThrottleReasons::SW_THERMAL_SLOWDOWN)
        || r.contains(ThrottleReasons::HW_THERMAL_SLOWDOWN)
    {
        out.push("Thermal Slowdown".into());
    }
    if r.contains(ThrottleReasons::HW_POWER_BRAKE_SLOWDOWN) {
        out.push("Power Brake".into());
    }
    if r.contains(ThrottleReasons::HW_SLOWDOWN) {
        out.push("HW Slowdown".into());
    }
    if r.contains(ThrottleReasons::SW_POWER_CAP) {
        out.push("Power Cap".into());
    }
    if r.contains(ThrottleReasons::SYNC_BOOST) {
        out.push("Sync Boost".into());
    }
    if r.contains(ThrottleReasons::APPLICATIONS_CLOCKS_SETTING) {
        out.push("App Clocks".into());
    }
    out
}

pub struct NvmlGpus {
    nvml: Nvml,
    count: u32,
    names: Vec<String>,
}

impl NvmlGpus {
    pub fn new() -> Option<Self> {
        let nvml = Nvml::init()
            .map_err(|e| log::warn!("NVML unavailable: {e}"))
            .ok()?;
        let count = nvml.device_count().ok()?;
        let names = (0..count)
            .map(|i| {
                nvml.device_by_index(i)
                    .and_then(|d| d.name())
                    .unwrap_or_else(|_| format!("GPU {i}"))
            })
            .collect();
        Some(Self { nvml, count, names })
    }

    pub fn info(&self) -> Vec<GpuInfo> {
        (0..self.count)
            .map(|i| GpuInfo {
                index: i,
                name: self.names[i as usize].clone(),
                vram_total_bytes: self
                    .nvml
                    .device_by_index(i)
                    .ok()
                    .and_then(|d| d.memory_info().ok())
                    .map(|m| m.total),
            })
            .collect()
    }

    pub fn sample(&self) -> Vec<GpuMetrics> {
        (0..self.count)
            .map(|i| {
                let mut g = GpuMetrics {
                    index: i,
                    name: self.names[i as usize].clone(),
                    ..Default::default()
                };
                if let Ok(d) = self.nvml.device_by_index(i) {
                    g.temp_c = d.temperature(TemperatureSensor::Gpu).ok().map(|t| t as f32);
                    if let Ok(u) = d.utilization_rates() {
                        g.util_pct = Some(u.gpu as f32);
                        g.mem_util_pct = Some(u.memory as f32);
                    }
                    g.core_mhz = d.clock_info(Clock::Graphics).ok().map(|c| c as f32);
                    g.mem_mhz = d.clock_info(Clock::Memory).ok().map(|c| c as f32);
                    g.power_w = d.power_usage().ok().map(|mw| mw as f32 / 1000.0);
                    g.power_limit_w = d.enforced_power_limit().ok().map(|mw| mw as f32 / 1000.0);
                    if let Ok(m) = d.memory_info() {
                        g.vram_used_bytes = Some(m.used);
                        g.vram_total_bytes = Some(m.total);
                    }
                    g.fan_pct = d.fan_speed(0).ok().map(|f| f as f32);
                    g.temp_slowdown_c = d
                        .temperature_threshold(TemperatureThreshold::Slowdown)
                        .ok()
                        .map(|t| t as f32);
                    if let Ok(tr) = d.current_throttle_reasons() {
                        g.throttle_reasons = decode_throttle_reasons(tr);
                    }
                }
                g
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_throttle_bitmasks() {
        assert!(decode_throttle_reasons(ThrottleReasons::GPU_IDLE).is_empty());
        let r = decode_throttle_reasons(
            ThrottleReasons::SW_POWER_CAP | ThrottleReasons::HW_THERMAL_SLOWDOWN,
        );
        assert!(r.contains(&"Power Cap".to_string()));
        assert!(r.contains(&"Thermal Slowdown".to_string()));
    }
}
