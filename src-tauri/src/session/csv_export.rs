//! CSV export: one row per telemetry sample. Dot decimals, ISO-8601 timestamps, UTF-8 BOM so
//! Excel opens it cleanly regardless of locale.

use std::collections::BTreeSet;
use std::io::Write;

use super::Session;
use crate::model::*;

fn f(v: Option<f32>, prec: usize) -> String {
    v.filter(|x| x.is_finite())
        .map(|x| format!("{x:.prec$}"))
        .unwrap_or_default()
}

fn phase_str(p: StressPhase) -> &'static str {
    match p {
        StressPhase::Idle => "idle",
        StressPhase::Starting => "starting",
        StressPhase::Running => "running",
        StressPhase::Stopping => "stopping",
        StressPhase::Completed => "completed",
        StressPhase::Stopped => "stopped",
        StressPhase::Aborted => "aborted",
        StressPhase::Failed => "failed",
    }
}

fn sidecar_str(s: SidecarState) -> String {
    serde_json::to_value(s)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

pub fn write_csv<W: Write>(s: &Session, mut out: W) -> Result<u32, crate::error::AppError> {
    out.write_all(b"\xEF\xBB\xBF")?;
    let mut w = csv::Writer::from_writer(out);
    let cores = s
        .samples
        .iter()
        .map(|x| x.cpu.cores.len())
        .max()
        .unwrap_or(0);
    let gpus = s
        .samples
        .iter()
        .map(|x| x.gpus.len())
        .max()
        .unwrap_or(0)
        .min(4);
    let board_t: BTreeSet<String> = s
        .samples
        .iter()
        .flat_map(|x| x.board.temps.iter().map(|b| b.name.clone()))
        .collect();
    let board_f: BTreeSet<String> = s
        .samples
        .iter()
        .flat_map(|x| x.board.fans.iter().map(|b| b.name.clone()))
        .collect();

    let mut h: Vec<String> = [
        "timestamp_iso",
        "elapsed_s",
        "stress_state",
        "stress_kind",
        "load_level_pct",
        "sidecar",
        "cpu_load_pct",
        "cpu_temp_c",
        "cpu_ccd_temp_c",
        "cpu_power_w",
        "cpu_core_voltage_v",
        "cpu_avg_mhz",
        "cpu_max_mhz",
        "cpu_throttle",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for i in 0..cores {
        h.push(format!("core{i}_load_pct"));
    }
    for i in 0..cores {
        h.push(format!("core{i}_mhz"));
    }
    for i in 0..cores {
        h.push(format!("core{i}_vid_v"));
    }
    for n in &board_t {
        h.push(format!("board_temp_c[{n}]"));
    }
    for n in &board_f {
        h.push(format!("board_fan_rpm[{n}]"));
    }
    h.extend(
        [
            "mem_used_mb",
            "mem_used_pct",
            "mem_available_mb",
            "swap_used_mb",
        ]
        .map(String::from),
    );
    for g in 0..gpus {
        for c in [
            "temp_c",
            "util_pct",
            "core_mhz",
            "mem_mhz",
            "power_w",
            "vram_used_mb",
            "fan_pct",
            "throttle",
        ] {
            h.push(format!("gpu{g}_{c}"));
        }
    }
    h.extend(
        [
            "stress_gflops",
            "stress_gpu_gflops",
            "stress_mem_mbps",
            "stress_errors",
            "alert",
        ]
        .map(String::from),
    );
    w.write_record(&h)?;

    let t0 = s
        .samples
        .front()
        .map(|x| x.timestamp_ms)
        .unwrap_or(s.started_ms);
    let mb = |b: u64| format!("{:.0}", b as f64 / 1_048_576.0);
    for x in &s.samples {
        let mut r: Vec<String> = Vec::with_capacity(h.len());
        let ts = chrono::DateTime::from_timestamp_millis(x.timestamp_ms as i64)
            .map(|d| {
                d.with_timezone(&chrono::Local)
                    .to_rfc3339_opts(chrono::SecondsFormat::Millis, false)
            })
            .unwrap_or_default();
        r.push(ts);
        r.push(format!(
            "{:.1}",
            x.timestamp_ms.saturating_sub(t0) as f64 / 1000.0
        ));
        r.push(phase_str(x.stress.phase).into());
        r.push(
            x.stress
                .kind
                .map(|k| format!("{k:?}").to_lowercase())
                .unwrap_or_default(),
        );
        r.push(x.stress.load_level.to_string());
        r.push(sidecar_str(x.sidecar));
        r.push(f(Some(x.cpu.total_load_pct), 1));
        r.push(f(x.cpu.package_temp_c, 1));
        r.push(f(x.cpu.ccd_temp_c, 1));
        r.push(f(x.cpu.package_power_w, 1));
        r.push(f(x.cpu.core_voltage_v, 3));
        r.push(f(x.cpu.avg_effective_mhz, 0));
        r.push(f(x.cpu.max_effective_mhz, 0));
        r.push(x.cpu.throttle_flags.join(";"));
        for i in 0..cores {
            r.push(f(x.cpu.cores.get(i).map(|c| c.load_pct), 1));
        }
        for i in 0..cores {
            r.push(f(x.cpu.cores.get(i).and_then(|c| c.effective_mhz), 0));
        }
        for i in 0..cores {
            r.push(f(x.cpu.cores.get(i).and_then(|c| c.vid), 3));
        }
        for n in &board_t {
            r.push(f(
                x.board.temps.iter().find(|b| &b.name == n).map(|b| b.value),
                1,
            ));
        }
        for n in &board_f {
            r.push(f(
                x.board.fans.iter().find(|b| &b.name == n).map(|b| b.value),
                0,
            ));
        }
        r.push(mb(x.memory.used_bytes));
        r.push(f(Some(x.memory.used_pct), 1));
        r.push(mb(x.memory.available_bytes));
        r.push(mb(x.memory.swap_used_bytes));
        for g in 0..gpus {
            let gm = x.gpus.get(g);
            r.push(f(gm.and_then(|g| g.temp_c), 0));
            r.push(f(gm.and_then(|g| g.util_pct), 0));
            r.push(f(gm.and_then(|g| g.core_mhz), 0));
            r.push(f(gm.and_then(|g| g.mem_mhz), 0));
            r.push(f(gm.and_then(|g| g.power_w), 1));
            r.push(
                gm.and_then(|g| g.vram_used_bytes)
                    .map(mb)
                    .unwrap_or_default(),
            );
            r.push(f(gm.and_then(|g| g.fan_pct), 0));
            r.push(gm.map(|g| g.throttle_reasons.join(";")).unwrap_or_default());
        }
        r.push(format!("{:.2}", x.stress.gflops));
        r.push(format!("{:.2}", x.stress.gpu_gflops));
        r.push(format!("{:.0}", x.stress.mem_mbps));
        r.push(x.stress.errors.to_string());
        let alerts: Vec<&str> = s
            .alerts
            .iter()
            .filter(|(seq, _)| *seq == x.seq)
            .map(|(_, t)| t.as_str())
            .collect();
        r.push(alerts.join(" | "));
        w.write_record(&r)?;
    }
    w.flush()?;
    Ok(s.samples.len() as u32)
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn fixture_sample(seq: u64, temp: f32) -> TelemetrySample {
        TelemetrySample {
            seq,
            timestamp_ms: 1_700_000_000_000 + seq * 500,
            cpu: CpuMetrics {
                total_load_pct: 50.0 + seq as f32,
                avg_effective_mhz: Some(4500.0),
                max_effective_mhz: Some(4650.0),
                package_temp_c: Some(temp),
                ccd_temp_c: None,
                package_power_w: Some(80.0),
                core_voltage_v: Some(1.25),
                throttle_flags: Vec::new(),
                cores: (0..4)
                    .map(|i| CoreMetrics {
                        id: i,
                        load_pct: 50.0,
                        effective_mhz: Some(4500.0),
                        clock_source: ClockSource::Pdh,
                        vid: None,
                    })
                    .collect(),
            },
            board: BoardMetrics {
                temps: vec![BoardSensor {
                    name: "System".into(),
                    value: 35.0,
                }],
                fans: vec![],
            },
            memory: MemoryMetrics {
                total_bytes: 16 << 30,
                used_bytes: 8 << 30,
                available_bytes: 8 << 30,
                used_pct: 50.0,
                swap_total_bytes: 0,
                swap_used_bytes: 0,
            },
            gpus: vec![GpuMetrics {
                index: 0,
                name: "GPU".into(),
                temp_c: Some(45.0),
                power_w: Some(30.5),
                ..Default::default()
            }],
            stress: StressSnapshot {
                phase: StressPhase::Running,
                kind: Some(StressKind::Cpu),
                elapsed_s: seq as f32 * 0.5,
                gflops: 400.0,
                gpu_gflops: 0.0,
                mem_mbps: 0.0,
                errors: 0,
                load_level: 100,
            },
            sidecar: SidecarState::Ok,
            poll_duration_us: 1000,
        }
    }

    pub fn fixture_session(n: u64) -> Session {
        Session {
            id: 1,
            label: "fixture".into(),
            auto: true,
            started_ms: 1_700_000_000_000,
            stopped_ms: Some(1_700_000_000_000 + n * 500),
            samples: (1..=n)
                .map(|i| fixture_sample(i, 60.0 + i as f32))
                .collect(),
            alerts: vec![(3, "CPU warning".into())],
            runs: vec![],
            truncated: false,
        }
    }

    #[test]
    fn csv_header_and_rows() {
        let s = fixture_session(10);
        let mut buf = Vec::new();
        let rows = write_csv(&s, &mut buf).unwrap();
        assert_eq!(rows, 10);
        assert!(buf.starts_with(b"\xEF\xBB\xBF"));
        let text = String::from_utf8(buf[3..].to_vec()).unwrap();
        let mut rdr = csv::Reader::from_reader(text.as_bytes());
        let headers = rdr.headers().unwrap().clone();
        assert_eq!(&headers[0], "timestamp_iso");
        assert!(headers.iter().any(|h| h == "core3_mhz"));
        assert!(headers.iter().any(|h| h == "board_temp_c[System]"));
        assert!(headers.iter().any(|h| h == "gpu0_power_w"));
        let recs: Vec<_> = rdr.records().map(|r| r.unwrap()).collect();
        assert_eq!(recs.len(), 10);
        assert!(recs.iter().all(|r| r.len() == headers.len()));
        let temp_col = headers.iter().position(|h| h == "cpu_temp_c").unwrap();
        assert_eq!(&recs[0][temp_col], "61.0");
        let alert_col = headers.len() - 1;
        assert_eq!(&recs[2][alert_col], "CPU warning");
    }
}
