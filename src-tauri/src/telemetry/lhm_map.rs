//! Map LibreHardwareMonitor sensors (from the shared-memory frame) to dashboard fields.
//!
//! Names follow LHM conventions, e.g. AMD Zen: "Core (Tctl/Tdie)", "CCD1 (Tdie)", "Package"
//! (power), "Core #3" (clock, per physical core), "Core #3 VID"; Intel: "CPU Package", "CPU Core #3".

use crate::model::{BoardSensor, GpuMetrics};
use crate::sidecar::layout::{HwKind, SensorFrame, SensorKind};

#[derive(Debug, Default, Clone, PartialEq)]
pub struct LhmData {
    pub package_temp_c: Option<f32>,
    pub ccd_temp_c: Option<f32>,
    pub package_power_w: Option<f32>,
    pub core_voltage_v: Option<f32>,
    /// Indexed by physical core (0-based).
    pub core_clocks: Vec<Option<f32>>,
    pub core_vids: Vec<Option<f32>>,
    pub cpu_throttle_flags: Vec<String>,
    pub board_temps: Vec<BoardSensor>,
    pub board_fans: Vec<BoardSensor>,
    /// Universal GPU telemetry fallback (AMD Radeon / Intel Arc / non-NVML GPUs).
    pub gpu: Option<GpuMetrics>,
}

/// "Core #3" / "CPU Core #3" -> Some(2); with a suffix (e.g. "Core #3 VID") -> None.
fn core_index(name: &str, suffix: &str) -> Option<usize> {
    let n = name.strip_prefix("CPU ").unwrap_or(name);
    let rest = n.strip_prefix("Core #")?;
    let rest = rest.strip_suffix(suffix)?;
    let idx: usize = rest.trim().parse().ok()?;
    idx.checked_sub(1)
}

fn set_at(v: &mut Vec<Option<f32>>, idx: usize, val: f32) {
    if idx < 256 {
        if v.len() <= idx {
            v.resize(idx + 1, None);
        }
        v[idx] = Some(val);
    }
}

pub fn map_frame(f: &SensorFrame) -> LhmData {
    let mut d = LhmData::default();
    let mut tctl_fallback = None;
    let mut vcore_fallback = None;
    for r in &f.records {
        if !r.value.is_finite() {
            continue;
        }
        let name = r.name.as_str();
        match (r.hw_kind, r.sensor_kind) {
            (HwKind::Cpu, SensorKind::Temperature) => {
                if name == "Core (Tctl/Tdie)" || name == "CPU Package" {
                    d.package_temp_c = Some(r.value);
                } else if name.starts_with("CCD") && name.contains("Tdie") {
                    d.ccd_temp_c = Some(d.ccd_temp_c.map_or(r.value, |c: f32| c.max(r.value)));
                } else if name.contains("Tctl")
                    || name == "Core (Tdie)"
                    || name == "Core Max"
                    || name == "Core Average"
                {
                    tctl_fallback.get_or_insert(r.value);
                }
            }
            (HwKind::Cpu, SensorKind::Power) => {
                if name == "Package" || name == "CPU Package" {
                    d.package_power_w = Some(r.value);
                }
            }
            (HwKind::Cpu, SensorKind::Voltage) => {
                if let Some(i) = core_index(name, " VID") {
                    set_at(&mut d.core_vids, i, r.value);
                } else if name == "Core (SVI2 TFN)"
                    || name == "CPU Core"
                    || name == "Core"
                    || name == "Core (SVI3 TFN)"
                    || name == "Vcore"
                    || name == "CPU Vcore"
                {
                    d.core_voltage_v = Some(r.value);
                }
            }
            (HwKind::SuperIo | HwKind::Motherboard, SensorKind::Voltage) => {
                if (name == "CPU VCore" || name == "Vcore" || name == "CPU Core")
                    && r.value > 0.3
                    && r.value < 2.2
                {
                    vcore_fallback.get_or_insert(r.value);
                }
            }
            (HwKind::Cpu, SensorKind::Clock) => {
                if let Some(i) = core_index(name, "") {
                    if r.value > 0.0 {
                        set_at(&mut d.core_clocks, i, r.value);
                    }
                }
            }
            (HwKind::Cpu, SensorKind::Factor) if r.value > 0.0 => {
                let lower = name.to_ascii_lowercase();
                if (lower.contains("thrott")
                    || lower.contains("prochot")
                    || lower.contains("thermal")
                    || lower.contains("limit"))
                    && !d.cpu_throttle_flags.iter().any(|f| f == name)
                {
                    d.cpu_throttle_flags.push(name.to_string());
                }
            }
            (HwKind::SuperIo | HwKind::Motherboard, SensorKind::Temperature) => {
                if r.value > 0.0 && r.value < 150.0 {
                    d.board_temps.push(BoardSensor {
                        name: r.name.clone(),
                        value: r.value,
                    });
                }
            }
            (HwKind::SuperIo | HwKind::Motherboard, SensorKind::Fan)
                if r.value >= 0.0 && r.value < 20000.0 =>
            {
                d.board_fans.push(BoardSensor {
                    name: r.name.clone(),
                    value: r.value,
                });
            }
            (HwKind::Gpu, sk) => {
                let g = d.gpu.get_or_insert_with(|| GpuMetrics {
                    index: 0,
                    name: if r.identifier.contains("amd") {
                        "AMD Radeon GPU".into()
                    } else if r.identifier.contains("intel") {
                        "Intel GPU".into()
                    } else {
                        "GPU 0".into()
                    },
                    ..Default::default()
                });
                match sk {
                    SensorKind::Temperature if name == "GPU Core" || g.temp_c.is_none() => {
                        if r.value > 0.0 && r.value < 150.0 {
                            g.temp_c = Some(r.value);
                        }
                    }
                    SensorKind::Load if name == "GPU Core" => g.util_pct = Some(r.value),
                    SensorKind::Load if name == "GPU Memory" || name == "GPU Memory Controller" => {
                        g.mem_util_pct = Some(r.value)
                    }
                    SensorKind::Clock if name == "GPU Core" => g.core_mhz = Some(r.value),
                    SensorKind::Clock if name == "GPU Memory" => g.mem_mhz = Some(r.value),
                    SensorKind::Power
                        if name == "GPU Package" || name == "GPU Power" || name == "GPU Core" =>
                    {
                        g.power_w = Some(r.value);
                    }
                    SensorKind::Fan if r.value >= 0.0 && r.value <= 100.0 => {
                        g.fan_pct = Some(r.value);
                    }
                    SensorKind::SmallData if name == "GPU Memory Used" => {
                        g.vram_used_bytes = Some((r.value.max(0.0) as u64) << 20);
                    }
                    SensorKind::SmallData if name == "GPU Memory Total" => {
                        g.vram_total_bytes = Some((r.value.max(0.0) as u64) << 20);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    if d.package_temp_c.is_none() {
        d.package_temp_c = tctl_fallback;
    }
    if d.core_voltage_v.is_none() {
        d.core_voltage_v = vcore_fallback;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sidecar::layout::{SensorRecord, WriterStatus};

    fn r(hw: HwKind, k: SensorKind, name: &str, v: f32) -> SensorRecord {
        SensorRecord {
            hw_kind: hw,
            sensor_kind: k,
            index: 0,
            value: v,
            min: v,
            max: v,
            identifier: String::new(),
            name: name.into(),
        }
    }

    #[test]
    fn maps_zen3_sensor_names() {
        use HwKind::*;
        use SensorKind::*;
        let f = SensorFrame {
            seq: 2,
            heartbeat_ms: 0,
            writer_pid: 1,
            status: WriterStatus::Ok,
            status_text: String::new(),
            records: vec![
                r(Cpu, Temperature, "Core (Tctl/Tdie)", 71.5),
                r(Cpu, Temperature, "CCD1 (Tdie)", 69.0),
                r(Cpu, Power, "Package", 88.2),
                r(Cpu, Power, "Core #1 (SMU)", 9.0),
                r(Cpu, Voltage, "Core (SVI2 TFN)", 1.25),
                r(Cpu, Voltage, "Core #2 VID", 1.3),
                r(Cpu, Clock, "Core #1", 4650.0),
                r(Cpu, Clock, "Core #8", 4500.0),
                r(Cpu, Clock, "Bus Speed", 100.0),
                r(SuperIo, Temperature, "System", 38.0),
                r(SuperIo, Temperature, "Bogus", -60.0),
                r(SuperIo, Fan, "Fan #1", 1200.0),
                r(Cpu, Temperature, "Core (Tctl/Tdie)", f32::NAN),
            ],
        };
        let d = map_frame(&f);
        assert_eq!(d.package_temp_c, Some(71.5));
        assert_eq!(d.ccd_temp_c, Some(69.0));
        assert_eq!(d.package_power_w, Some(88.2));
        assert_eq!(d.core_voltage_v, Some(1.25));
        assert_eq!(d.core_vids.get(1).copied().flatten(), Some(1.3));
        assert_eq!(d.core_clocks.len(), 8);
        assert_eq!(d.core_clocks[0], Some(4650.0));
        assert_eq!(d.core_clocks[7], Some(4500.0));
        assert_eq!(d.board_temps.len(), 1);
        assert_eq!(d.board_fans.len(), 1);
    }

    #[test]
    fn core_index_parsing() {
        assert_eq!(core_index("Core #1", ""), Some(0));
        assert_eq!(core_index("CPU Core #12", ""), Some(11));
        assert_eq!(core_index("Core #1 VID", ""), None);
        assert_eq!(core_index("Core #1 VID", " VID"), Some(0));
        assert_eq!(core_index("Core #0", ""), None);
    }
}
