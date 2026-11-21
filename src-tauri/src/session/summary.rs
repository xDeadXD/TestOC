//! `<name>.summary.json`: min/avg/max of key metrics, run outcomes, errors, sensor availability.

use serde::Serialize;

use super::Session;
use crate::model::*;

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Stat {
    pub min: f32,
    pub avg: f32,
    pub max: f32,
    pub samples: u32,
}

pub fn stat(vals: impl Iterator<Item = Option<f32>>) -> Option<Stat> {
    let (mut mn, mut mx, mut sum, mut n) = (f32::INFINITY, f32::NEG_INFINITY, 0f64, 0u32);
    for v in vals.flatten().filter(|v| v.is_finite()) {
        mn = mn.min(v);
        mx = mx.max(v);
        sum += v as f64;
        n += 1;
    }
    (n > 0).then(|| Stat {
        min: mn,
        avg: (sum / n as f64) as f32,
        max: mx,
        samples: n,
    })
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Summary<'a> {
    pub label: &'a str,
    pub started: String,
    pub stopped: Option<String>,
    pub duration_s: f64,
    pub samples: usize,
    pub truncated: bool,
    pub machine: &'a SystemInfo,
    pub runs: Vec<RunSummary>,
    pub total_errors: u64,
    pub sidecar_availability_pct: f32,
    pub throttle_samples: u32,
    pub alerts: Vec<&'a str>,
    pub metrics: Metrics,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunSummary {
    pub kind: Option<StressKind>,
    pub config: Option<StressConfig>,
    pub outcome: StressPhase,
    pub abort_reason: Option<AbortReason>,
    pub elapsed_s: f32,
    pub isa: Option<String>,
    pub total_errors: u64,
    pub core_errors: Vec<u64>,
    pub thermal_protection: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    pub cpu_load_pct: Option<Stat>,
    pub cpu_temp_c: Option<Stat>,
    pub cpu_power_w: Option<Stat>,
    pub cpu_avg_mhz: Option<Stat>,
    pub mem_used_pct: Option<Stat>,
    pub gpu_temp_c: Option<Stat>,
    pub gpu_util_pct: Option<Stat>,
    pub gpu_power_w: Option<Stat>,
    pub stress_gflops: Option<Stat>,
    pub stress_gpu_gflops: Option<Stat>,
    pub stress_mem_mbps: Option<Stat>,
}

fn iso(ms: u64) -> String {
    chrono::DateTime::from_timestamp_millis(ms as i64)
        .map(|d| {
            d.with_timezone(&chrono::Local)
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, false)
        })
        .unwrap_or_default()
}

pub fn build<'a>(s: &'a Session, machine: &'a SystemInfo) -> Summary<'a> {
    let x = &s.samples;
    let running = |p: StressPhase| p.is_active();
    let end = s
        .stopped_ms
        .unwrap_or_else(|| x.back().map(|l| l.timestamp_ms).unwrap_or(s.started_ms));
    let ok = x.iter().filter(|v| v.sidecar == SidecarState::Ok).count();
    let throttle_samples = x
        .iter()
        .filter(|v| {
            !v.cpu.throttle_flags.is_empty()
                || v.gpus.iter().any(|g| !g.throttle_reasons.is_empty())
        })
        .count() as u32;
    Summary {
        label: &s.label,
        started: iso(s.started_ms),
        stopped: s.stopped_ms.map(iso),
        duration_s: end.saturating_sub(s.started_ms) as f64 / 1000.0,
        samples: x.len(),
        truncated: s.truncated,
        machine,
        runs: s
            .runs
            .iter()
            .map(|r| RunSummary {
                kind: r.kind,
                config: r.config.clone(),
                outcome: r.phase,
                abort_reason: r.abort_reason,
                elapsed_s: r.elapsed_s,
                isa: r.isa_used.clone(),
                total_errors: r.total_errors,
                core_errors: r.core_errors.clone(),
                thermal_protection: r.thermal_protection,
            })
            .collect(),
        total_errors: x
            .iter()
            .map(|v| v.stress.errors)
            .max()
            .unwrap_or(0)
            .max(s.runs.iter().map(|r| r.total_errors).sum()),
        sidecar_availability_pct: if x.is_empty() {
            0.0
        } else {
            ok as f32 * 100.0 / x.len() as f32
        },
        throttle_samples,
        alerts: s.alerts.iter().map(|(_, t)| t.as_str()).collect(),
        metrics: Metrics {
            cpu_load_pct: stat(x.iter().map(|v| Some(v.cpu.total_load_pct))),
            cpu_temp_c: stat(x.iter().map(|v| v.cpu.package_temp_c)),
            cpu_power_w: stat(x.iter().map(|v| v.cpu.package_power_w)),
            cpu_avg_mhz: stat(x.iter().map(|v| v.cpu.avg_effective_mhz)),
            mem_used_pct: stat(x.iter().map(|v| Some(v.memory.used_pct))),
            gpu_temp_c: stat(x.iter().map(|v| v.gpus.first().and_then(|g| g.temp_c))),
            gpu_util_pct: stat(x.iter().map(|v| v.gpus.first().and_then(|g| g.util_pct))),
            gpu_power_w: stat(x.iter().map(|v| v.gpus.first().and_then(|g| g.power_w))),
            stress_gflops: stat(
                x.iter()
                    .filter(|v| running(v.stress.phase))
                    .map(|v| Some(v.stress.gflops as f32)),
            ),
            stress_gpu_gflops: stat(
                x.iter()
                    .filter(|v| running(v.stress.phase))
                    .map(|v| Some(v.stress.gpu_gflops as f32)),
            ),
            stress_mem_mbps: stat(
                x.iter()
                    .filter(|v| running(v.stress.phase))
                    .map(|v| Some(v.stress.mem_mbps as f32)),
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::csv_export::tests::fixture_session;

    #[test]
    fn stat_basic() {
        let s = stat([Some(1.0), None, Some(3.0), Some(f32::NAN)].into_iter()).unwrap();
        assert_eq!(
            s,
            Stat {
                min: 1.0,
                avg: 2.0,
                max: 3.0,
                samples: 2
            }
        );
        assert!(stat(std::iter::empty()).is_none());
    }

    #[test]
    fn summary_min_avg_max() {
        let sess = fixture_session(10); // temps 61..=70
        let info = SystemInfo {
            app_version: "t".into(),
            hostname: "h".into(),
            os_version: "w".into(),
            cpu_brand: "c".into(),
            physical_cores: 2,
            logical_cores: 4,
            base_mhz: 3400,
            isa: IsaFlags {
                avx2: true,
                fma: true,
                avx512f: false,
            },
            total_memory_bytes: 16 << 30,
            gpus: vec![],
            gpu_stress_adapter: None,
            gpu_vram_total_bytes: None,
            pdh_clocks_available: true,
            nvml_available: true,
        };
        let s = build(&sess, &info);
        let t = s.metrics.cpu_temp_c.as_ref().unwrap();
        assert_eq!((t.min, t.max), (61.0, 70.0));
        assert!((t.avg - 65.5).abs() < 1e-4);
        assert_eq!(s.sidecar_availability_pct, 100.0);
        assert_eq!(s.alerts, vec!["CPU warning"]);
        assert!(serde_json::to_string(&s).unwrap().contains("\"cpuTempC\""));
    }
}
