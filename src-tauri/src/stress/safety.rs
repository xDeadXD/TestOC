//! Safety limits and the pure watchdog evaluation used by the telemetry loop.

use crate::model::*;

pub fn default_limits() -> SafetyLimits {
    // OC stability defaults: 100% RAM cap (uncapped, OOM-safe) and modern thermal ceilings.
    SafetyLimits {
        cpu_warn_c: 90.0,
        cpu_abort_c: 100.0,
        gpu_warn_c: 88.0,
        gpu_abort_c: 95.0,
        mem_max_pct: 100,
        thermal_abort_enabled: true,
    }
}

pub const CPU_ABORT_RANGE: (f32, f32) = (60.0, 115.0);
pub const GPU_ABORT_RANGE: (f32, f32) = (60.0, 110.0);
pub const MEM_RANGE: (u8, u8) = (10, 100);
/// Consecutive breaching samples required before aborting (500 ms polling, so ~1 s).
pub const DEBOUNCE: u8 = 2;
/// Warning re-arms only after the value drops this far below the threshold.
pub const HYSTERESIS_C: f32 = 2.0;
/// Lean OS reserve floor (128 MiB) so 95-99% RAM stability tests can run without false aborts.
pub const MIN_AVAILABLE_BYTES: u64 = 128 << 20;

pub fn validate(l: &SafetyLimits) -> Result<(), String> {
    let in_range = |v: f32, (lo, hi): (f32, f32)| v.is_finite() && v >= lo && v <= hi;
    if !in_range(l.cpu_abort_c, CPU_ABORT_RANGE) {
        return Err(format!(
            "CPU abort limit must be {}-{} °C",
            CPU_ABORT_RANGE.0, CPU_ABORT_RANGE.1
        ));
    }
    if !(l.cpu_warn_c.is_finite() && l.cpu_warn_c >= 40.0 && l.cpu_warn_c <= l.cpu_abort_c) {
        return Err("CPU warning must be between 40 °C and the abort limit".into());
    }
    if !in_range(l.gpu_abort_c, GPU_ABORT_RANGE) {
        return Err(format!(
            "GPU abort limit must be {}-{} °C",
            GPU_ABORT_RANGE.0, GPU_ABORT_RANGE.1
        ));
    }
    if !(l.gpu_warn_c.is_finite() && l.gpu_warn_c >= 40.0 && l.gpu_warn_c <= l.gpu_abort_c) {
        return Err("GPU warning must be between 40 °C and the abort limit".into());
    }
    if l.mem_max_pct < MEM_RANGE.0 || l.mem_max_pct > MEM_RANGE.1 {
        return Err(format!(
            "memory limit must be {}-{} %",
            MEM_RANGE.0, MEM_RANGE.1
        ));
    }
    Ok(())
}

/// Inputs for one watchdog evaluation (one telemetry sample).
#[derive(Debug, Clone, Default)]
pub struct WatchInput {
    pub cpu_temp_c: Option<f32>,
    pub gpu_temp_c: Option<f32>,
    pub mem_used_pct: f32,
    pub mem_available_bytes: u64,
    /// A run is active (not done) and not already aborted.
    pub stress_active: bool,
    pub stress_kind: Option<StressKind>,
    /// The run started with CPU thermal protection.
    pub run_has_thermal: bool,
    /// Sidecar state is `Ok` (fresh heartbeat).
    pub sidecar_fresh: bool,
    /// Severe (thermal/HW/power-brake) throttle labels currently active across CPU/GPU.
    pub severe_throttles: Vec<String>,
}

/// Filter raw CPU/GPU throttle reasons down to severe thermal/HW/power-brake conditions.
/// Informational flags such as GPU "Power Cap", "Sync Boost", and "App Clocks" are excluded.
pub fn severe_throttle_reasons<'a>(
    cpu_flags: &'a [String],
    gpu_flags: &'a [String],
) -> Vec<String> {
    let mut out = Vec::new();
    for f in cpu_flags {
        out.push(format!("CPU: {f}"));
    }
    for f in gpu_flags {
        if matches!(
            f.as_str(),
            "Thermal Slowdown" | "HW Slowdown" | "Power Brake"
        ) {
            out.push(format!("GPU: {f}"));
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub enum WatchEvent {
    Alert(SafetyAlertLite),
    Abort(AbortReason, SafetyAlertLite),
}

/// Alert without timestamp (filled in by the caller) so tests compare easily.
#[derive(Debug, Clone, PartialEq)]
pub struct SafetyAlertLite {
    pub level: AlertLevel,
    pub metric: AlertMetric,
    pub value: f32,
    pub threshold: f32,
    pub message: String,
}

impl SafetyAlertLite {
    pub fn into_alert(self, action: AlertAction, ts: u64) -> SafetyAlert {
        SafetyAlert {
            level: self.level,
            metric: self.metric,
            value: self.value,
            threshold: self.threshold,
            action,
            timestamp_ms: ts,
            message: self.message,
        }
    }
}

#[derive(Debug, Default)]
pub struct Watchdog {
    cpu_breach: u8,
    gpu_breach: u8,
    mem_breach: u8,
    throttle_breach: u8,
    throttle_clear: u8,
    throttle_warned: bool,
    cpu_warned: bool,
    gpu_warned: bool,
    cpu_crit_announced: bool,
    gpu_crit_announced: bool,
}

impl Watchdog {
    pub fn reset_run(&mut self) {
        self.cpu_breach = 0;
        self.gpu_breach = 0;
        self.mem_breach = 0;
        self.throttle_breach = 0;
        self.throttle_warned = false;
    }

    pub fn evaluate(&mut self, i: &WatchInput, l: &SafetyLimits) -> Vec<WatchEvent> {
        let mut out = Vec::new();
        let kind = i.stress_kind;
        let active = i.stress_active;

        // --- Sensor stall: a CPU-loading run that started with thermal protection lost it. ---
        if active
            && kind.is_some_and(StressKind::loads_cpu)
            && i.run_has_thermal
            && !i.sidecar_fresh
        {
            out.push(WatchEvent::Abort(
                AbortReason::SensorStall,
                SafetyAlertLite {
                    level: AlertLevel::Critical,
                    metric: AlertMetric::SensorStall,
                    value: 0.0,
                    threshold: crate::sidecar::STALE_AFTER_MS as f32 / 1000.0,
                    message: "CPU temperature sensor lost (sidecar stale or exited)".into(),
                },
            ));
            return out;
        }

        // --- CPU temperature ---
        self.temp_rule(
            &mut out,
            i.cpu_temp_c,
            l.cpu_warn_c,
            l.cpu_abort_c,
            l.thermal_abort_enabled && active && kind.is_some_and(StressKind::loads_cpu),
            AlertMetric::CpuTemp,
            AbortReason::CpuTemp,
            "CPU",
            true,
        );
        // --- GPU temperature (aborts any active run when thermal abort is enabled) ---
        self.temp_rule(
            &mut out,
            i.gpu_temp_c,
            l.gpu_warn_c,
            l.gpu_abort_c,
            l.thermal_abort_enabled && active,
            AlertMetric::GpuTemp,
            AbortReason::GpuTemp,
            "GPU",
            false,
        );

        // --- Memory pressure during memory tests (100% = uncapped, relies on OOM chunk guard) ---
        if active && kind.is_some_and(StressKind::uses_memory) && l.mem_max_pct < 100 {
            let over_pct = i.mem_used_pct > l.mem_max_pct as f32 + 3.0;
            let low_avail = l.mem_max_pct < 98
                && i.mem_available_bytes > 0
                && i.mem_available_bytes < MIN_AVAILABLE_BYTES;
            if over_pct || low_avail {
                self.mem_breach = self.mem_breach.saturating_add(1);
                if self.mem_breach >= DEBOUNCE {
                    out.push(WatchEvent::Abort(
                        AbortReason::MemoryPressure,
                        SafetyAlertLite {
                            level: AlertLevel::Critical,
                            metric: AlertMetric::Memory,
                            value: i.mem_used_pct,
                            threshold: l.mem_max_pct as f32,
                            message: if low_avail {
                                format!("Available memory below {} MiB", MIN_AVAILABLE_BYTES >> 20)
                            } else {
                                format!(
                                    "Memory usage {:.1}% above limit {}%",
                                    i.mem_used_pct, l.mem_max_pct
                                )
                            },
                        },
                    ));
                }
            } else {
                self.mem_breach = 0;
            }
        } else {
            self.mem_breach = 0;
        }

        // --- Severe throttling (non-aborting debounced Warning alert) ---
        if !i.severe_throttles.is_empty() {
            self.throttle_clear = 0;
            self.throttle_breach = self.throttle_breach.saturating_add(1);
            if self.throttle_breach >= DEBOUNCE && !self.throttle_warned {
                self.throttle_warned = true;
                out.push(WatchEvent::Alert(SafetyAlertLite {
                    level: AlertLevel::Warning,
                    metric: AlertMetric::Throttle,
                    value: i.severe_throttles.len() as f32,
                    threshold: 1.0,
                    message: format!(
                        "Hardware throttling active: {}",
                        i.severe_throttles.join(", ")
                    ),
                }));
            }
        } else {
            self.throttle_breach = 0;
            self.throttle_clear = self.throttle_clear.saturating_add(1);
            if self.throttle_clear >= DEBOUNCE {
                self.throttle_warned = false;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn temp_rule(
        &mut self,
        out: &mut Vec<WatchEvent>,
        temp: Option<f32>,
        warn: f32,
        abort: f32,
        can_abort: bool,
        metric: AlertMetric,
        reason: AbortReason,
        label: &str,
        is_cpu: bool,
    ) {
        let (breach, warned, crit) = if is_cpu {
            (
                &mut self.cpu_breach,
                &mut self.cpu_warned,
                &mut self.cpu_crit_announced,
            )
        } else {
            (
                &mut self.gpu_breach,
                &mut self.gpu_warned,
                &mut self.gpu_crit_announced,
            )
        };
        let Some(t) = temp.filter(|t| t.is_finite()) else {
            *breach = 0;
            return;
        };
        if t >= abort {
            *breach = breach.saturating_add(1);
            if *breach >= DEBOUNCE {
                let alert = SafetyAlertLite {
                    level: AlertLevel::Critical,
                    metric,
                    value: t,
                    threshold: abort,
                    message: format!(
                        "{label} temperature {t:.1} °C reached abort limit {abort:.0} °C"
                    ),
                };
                if can_abort {
                    out.push(WatchEvent::Abort(reason, alert));
                    *crit = true;
                } else if !*crit {
                    out.push(WatchEvent::Alert(alert));
                    *crit = true;
                }
            }
        } else {
            *breach = 0;
            if t < abort - HYSTERESIS_C {
                *crit = false;
            }
        }
        if t >= warn && !*warned {
            *warned = true;
            out.push(WatchEvent::Alert(SafetyAlertLite {
                level: AlertLevel::Warning,
                metric,
                value: t,
                threshold: warn,
                message: format!("{label} temperature {t:.1} °C above warning {warn:.0} °C"),
            }));
        } else if t < warn - HYSTERESIS_C {
            *warned = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_run(temp: f32) -> WatchInput {
        WatchInput {
            cpu_temp_c: Some(temp),
            gpu_temp_c: Some(50.0),
            mem_used_pct: 40.0,
            mem_available_bytes: 8 << 30,
            stress_active: true,
            stress_kind: Some(StressKind::Cpu),
            run_has_thermal: true,
            sidecar_fresh: true,
            severe_throttles: Vec::new(),
        }
    }

    fn aborts(ev: &[WatchEvent]) -> Vec<AbortReason> {
        ev.iter()
            .filter_map(|e| match e {
                WatchEvent::Abort(r, _) => Some(*r),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn defaults_are_valid() {
        validate(&default_limits()).unwrap();
    }

    #[test]
    fn validation_bounds() {
        let mut l = default_limits();
        l.cpu_abort_c = 116.0;
        assert!(validate(&l).is_err());
        l = default_limits();
        l.cpu_abort_c = 115.0;
        assert!(validate(&l).is_ok());
        l = default_limits();
        l.cpu_warn_c = 101.0;
        assert!(validate(&l).is_err());
        l = default_limits();
        l.gpu_abort_c = 111.0;
        assert!(validate(&l).is_err());
        l = default_limits();
        l.gpu_abort_c = 105.0;
        assert!(validate(&l).is_ok());
        l = default_limits();
        l.mem_max_pct = 100;
        assert!(validate(&l).is_ok());
        l.mem_max_pct = 9;
        assert!(validate(&l).is_err());
        l = default_limits();
        l.gpu_abort_c = f32::NAN;
        assert!(validate(&l).is_err());
    }

    #[test]
    fn single_breach_does_not_trip_two_do() {
        let l = default_limits();
        let mut w = Watchdog::default();
        assert!(aborts(&w.evaluate(&cpu_run(101.0), &l)).is_empty());
        assert_eq!(
            aborts(&w.evaluate(&cpu_run(101.0), &l)),
            vec![AbortReason::CpuTemp]
        );
    }

    #[test]
    fn thermal_abort_can_be_disabled_while_keeping_alerts() {
        let mut l = default_limits();
        l.thermal_abort_enabled = false;
        let mut w = Watchdog::default();
        w.evaluate(&cpu_run(102.0), &l);
        let ev = w.evaluate(&cpu_run(102.0), &l);
        assert!(
            aborts(&ev).is_empty(),
            "disabled thermal abort must not abort"
        );
        assert!(ev
            .iter()
            .any(|e| matches!(e, WatchEvent::Alert(a) if a.level == AlertLevel::Critical)));
    }

    #[test]
    fn recovered_reading_resets_debounce() {
        let l = default_limits();
        let mut w = Watchdog::default();
        w.evaluate(&cpu_run(101.0), &l);
        w.evaluate(&cpu_run(80.0), &l);
        assert!(aborts(&w.evaluate(&cpu_run(101.0), &l)).is_empty());
    }

    #[test]
    fn warning_is_edge_triggered_with_hysteresis() {
        let l = default_limits();
        let mut w = Watchdog::default();
        let warn_count = |ev: &[WatchEvent]| {
            ev.iter()
                .filter(|e| matches!(e, WatchEvent::Alert(a) if a.level == AlertLevel::Warning))
                .count()
        };
        assert_eq!(warn_count(&w.evaluate(&cpu_run(91.0), &l)), 1);
        assert_eq!(warn_count(&w.evaluate(&cpu_run(91.5), &l)), 0);
        assert_eq!(warn_count(&w.evaluate(&cpu_run(89.0), &l)), 0); // within hysteresis
        w.evaluate(&cpu_run(85.0), &l); // re-arm
        assert_eq!(warn_count(&w.evaluate(&cpu_run(91.0), &l)), 1);
    }

    #[test]
    fn idle_overheat_alerts_but_does_not_abort() {
        let l = default_limits();
        let mut w = Watchdog::default();
        let mut i = cpu_run(102.0);
        i.stress_active = false;
        w.evaluate(&i, &l);
        let ev = w.evaluate(&i, &l);
        assert!(aborts(&ev).is_empty());
        assert!(ev
            .iter()
            .any(|e| matches!(e, WatchEvent::Alert(a) if a.level == AlertLevel::Critical)));
    }

    #[test]
    fn sensor_stall_aborts_cpu_run_immediately() {
        let l = default_limits();
        let mut w = Watchdog::default();
        let mut i = cpu_run(60.0);
        i.sidecar_fresh = false;
        assert_eq!(aborts(&w.evaluate(&i, &l)), vec![AbortReason::SensorStall]);
        // ...but not a run that started without thermal protection (acknowledged + capped).
        i.run_has_thermal = false;
        assert!(aborts(&w.evaluate(&i, &l)).is_empty());
        // ...and not a memory-only run.
        i.run_has_thermal = true;
        i.stress_kind = Some(StressKind::Memory);
        assert!(aborts(&w.evaluate(&i, &l)).is_empty());
    }

    #[test]
    fn memory_pressure_aborts_memory_runs_unless_uncapped() {
        let mut l = default_limits();
        l.mem_max_pct = 95;
        let mut w = Watchdog::default();
        let mut i = cpu_run(50.0);
        i.stress_kind = Some(StressKind::Memory);
        i.mem_used_pct = 99.0; // > 95 + 3
        assert!(aborts(&w.evaluate(&i, &l)).is_empty());
        assert_eq!(
            aborts(&w.evaluate(&i, &l)),
            vec![AbortReason::MemoryPressure]
        );
        // Default (mem_max_pct == 100, uncapped): memory pressure does not spuriously abort.
        let uncapped = default_limits();
        assert_eq!(uncapped.mem_max_pct, 100);
        let mut w2 = Watchdog::default();
        w2.evaluate(&i, &uncapped);
        assert!(aborts(&w2.evaluate(&i, &uncapped)).is_empty());
    }

    #[test]
    fn gpu_overheat_aborts_any_run() {
        let l = default_limits(); // gpu_abort_c = 95
        let mut w = Watchdog::default();
        let mut i = cpu_run(50.0);
        i.stress_kind = Some(StressKind::Gpu);
        i.gpu_temp_c = Some(96.0);
        w.evaluate(&i, &l);
        assert_eq!(aborts(&w.evaluate(&i, &l)), vec![AbortReason::GpuTemp]);
    }

    #[test]
    fn severe_throttling_warns_debounced_while_power_cap_is_informational() {
        assert!(
            severe_throttle_reasons(&[], &["Power Cap".into(), "Sync Boost".into()]).is_empty()
        );
        let sev = severe_throttle_reasons(&[], &["Power Cap".into(), "Thermal Slowdown".into()]);
        assert_eq!(sev, vec!["GPU: Thermal Slowdown"]);

        let l = default_limits();
        let mut w = Watchdog::default();
        let mut i = cpu_run(60.0);
        i.severe_throttles = sev;
        // 1st sample: debounced
        assert!(w.evaluate(&i, &l).is_empty());
        // 2nd sample: non-aborting Warning
        let ev = w.evaluate(&i, &l);
        assert!(aborts(&ev).is_empty());
        assert!(ev.iter().any(|e| matches!(e, WatchEvent::Alert(a) if a.metric == AlertMetric::Throttle && a.level == AlertLevel::Warning)));
        // 3rd sample: edge-triggered (no spam)
        assert!(w.evaluate(&i, &l).is_empty());
        // Clear for 2 samples -> re-arms
        i.severe_throttles.clear();
        w.evaluate(&i, &l);
        w.evaluate(&i, &l);
        i.severe_throttles = vec!["CPU: Thermal Limit".into()];
        assert!(w.evaluate(&i, &l).is_empty());
        assert_eq!(w.evaluate(&i, &l).len(), 1);
    }
}
