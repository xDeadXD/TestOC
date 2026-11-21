//! Standalone stress CLI: exercises CPU, CoreCycler, Memory (Absolut 32-Step / Standard), CPU+RAM,
//! GPU, VRAM, and Power workloads on hardware with live telemetry, thermal guard, and abort latency.
//!
//! Usage: cargo run --release --example stress_cli -- --mode <cpu|variable|corecycler|mem|combined|gpu|vram|power>
//!        [--workload <extreme|fft|fma|ntt>] [--fft <smallest|small|large|blend>]
//!        [--mem-profile <absolut|standard>] [--cycles N] [--profile <standard|heavy|extreme>]
//!        [--adaptive] [--pattern <all|inv|stride|walk|addr|rand>] [--secs 10] [--pct 100]

use std::sync::Arc;
use std::time::{Duration, Instant};

use testoc_lib::model::*;
use testoc_lib::stress::gpu::GpuStress;
use testoc_lib::stress::manager::{NullEvents, StartContext, StressManager};
use testoc_lib::stress::safety;
use testoc_lib::telemetry::TelemetryEngine;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mode = "cpu".to_string();
    let mut profile = GpuProfile::Extreme;
    let mut cpu_workload = CpuWorkload::Extreme;
    let mut fft_preset = FftPreset::Small;
    let mut mem_profile = MemoryProfile::Absolut;
    let mut cycles: Option<u16> = None;
    let mut adaptive = false;
    let mut pattern = MemoryPattern::All;
    let mut secs = 10u32;
    let mut poll_ms = 1000u64;
    let mut pct = 100u8;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                i += 1;
                mode = args[i].clone();
            }
            "--profile" => {
                i += 1;
                profile = match args[i].as_str() {
                    "standard" => GpuProfile::Standard,
                    "heavy" => GpuProfile::Heavy,
                    "extreme" => GpuProfile::Extreme,
                    other => panic!("unknown profile {other}"),
                };
            }
            "--workload" => {
                i += 1;
                cpu_workload = match args[i].as_str() {
                    "extreme" => CpuWorkload::Extreme,
                    "fft" => CpuWorkload::FftTorture,
                    "fma" => CpuWorkload::MatrixFma,
                    "ntt" => CpuWorkload::NttPrime,
                    other => panic!("unknown workload {other}"),
                };
            }
            "--fft" => {
                i += 1;
                fft_preset = match args[i].as_str() {
                    "smallest" => FftPreset::Smallest,
                    "small" => FftPreset::Small,
                    "large" => FftPreset::Large,
                    "blend" => FftPreset::Blend,
                    other => panic!("unknown fft preset {other}"),
                };
            }
            "--mem-profile" => {
                i += 1;
                mem_profile = match args[i].as_str() {
                    "absolut" => MemoryProfile::Absolut,
                    "standard" => MemoryProfile::Standard,
                    other => panic!("unknown mem-profile {other}"),
                };
            }
            "--cycles" => {
                i += 1;
                cycles = Some(args[i].parse().unwrap());
            }
            "--adaptive" => {
                adaptive = true;
            }
            "--pattern" => {
                i += 1;
                pattern = match args[i].as_str() {
                    "all" => MemoryPattern::All,
                    "inv" => MemoryPattern::MovingInversions,
                    "stride" => MemoryPattern::BlockStrided,
                    "walk" => MemoryPattern::WalkingOnes,
                    "addr" => MemoryPattern::AddressInAddress,
                    "rand" => MemoryPattern::Random,
                    other => panic!("unknown pattern {other}"),
                };
            }
            "--secs" => {
                i += 1;
                secs = args[i].parse().unwrap();
            }
            "--poll-ms" => {
                i += 1;
                poll_ms = args[i].parse().unwrap();
            }
            "--pct" => {
                i += 1;
                pct = args[i].parse().unwrap();
            }
            _ => {}
        }
        i += 1;
    }

    let mut engine = TelemetryEngine::new();
    let gpu = GpuStress::probe().map(Arc::new);
    let info = engine.system_info(
        gpu.as_ref().map(|g| g.adapter_name().to_string()),
        gpu.as_ref().map(|g| g.fallback_vram_bytes()),
    );
    let mgr = StressManager::new(Arc::new(NullEvents), gpu);
    let snap0 = engine.poll(None, SidecarState::NotStarted, mgr.snapshot());
    let (total_mem, avail_mem) = testoc_lib::win::memory_status()
        .unwrap_or((snap0.memory.total_bytes, snap0.memory.available_bytes));
    let (vram_total, vram_free) = snap0
        .gpus
        .first()
        .map(|g| {
            (
                g.vram_total_bytes,
                g.vram_total_bytes
                    .zip(g.vram_used_bytes)
                    .map(|(t, u)| t.saturating_sub(u)),
            )
        })
        .unwrap_or((info.gpu_vram_total_bytes, None));

    let cfg = match mode.as_str() {
        "cpu" => StressConfig::Cpu {
            threads: None,
            physical_cores_only: false,
            isa: IsaChoice::Auto,
            dataset: Dataset::Small,
            workload: cpu_workload,
            fft_preset,
            mode: LoadMode::Steady,
            variable_min_pct: 20,
            variable_max_pct: 100,
            core_cycle_s: 3,
            stop_on_error: false,
            max_errors: None,
            duration_s: None,
        },
        "variable" => StressConfig::Cpu {
            threads: None,
            physical_cores_only: false,
            isa: IsaChoice::Auto,
            dataset: Dataset::Small,
            workload: cpu_workload,
            fft_preset,
            mode: LoadMode::Variable,
            variable_min_pct: 20,
            variable_max_pct: 100,
            core_cycle_s: 3,
            stop_on_error: false,
            max_errors: None,
            duration_s: None,
        },
        "corecycler" => StressConfig::Cpu {
            threads: None,
            physical_cores_only: true,
            isa: IsaChoice::Auto,
            dataset: Dataset::Small,
            workload: cpu_workload,
            fft_preset,
            mode: LoadMode::CoreCycler,
            variable_min_pct: 20,
            variable_max_pct: 100,
            core_cycle_s: 2,
            stop_on_error: false,
            max_errors: None,
            duration_s: None,
        },
        "combined" => StressConfig::Combined {
            cpu_threads: None,
            cpu_isa: IsaChoice::Auto,
            cpu_dataset: Dataset::Small,
            cpu_workload,
            cpu_fft_preset: fft_preset,
            mem_target_pct: pct,
            mem_profile,
            stop_on_error: false,
            max_errors: None,
            duration_s: None,
        },
        "mem" => StressConfig::Memory {
            threads: None,
            target_pct: pct,
            profile: mem_profile,
            pattern,
            cycles,
            stop_on_error: false,
            max_errors: None,
            duration_s: None,
        },
        "gpu" => StressConfig::Gpu {
            mode: if adaptive {
                LoadMode::Variable
            } else {
                LoadMode::Steady
            },
            profile,
            intensity: 100,
            variable_min_pct: 25,
            variable_max_pct: 100,
            variable_step_s: 3,
            error_check: true,
            stop_on_error: false,
            max_errors: None,
            duration_s: None,
        },
        "vram" => StressConfig::Vram {
            target_pct: pct,
            pattern,
            stop_on_error: false,
            max_errors: None,
            duration_s: None,
        },
        "power" => StressConfig::Power {
            cpu_threads: None,
            cpu_isa: IsaChoice::Auto,
            cpu_dataset: Dataset::Small,
            cpu_workload,
            cpu_fft_preset: fft_preset,
            gpu_profile: profile,
            gpu_intensity: 100,
            stop_on_error: false,
            max_errors: None,
            duration_s: None,
        },
        other => panic!("unknown mode {other}"),
    };

    let mut limits = safety::default_limits();
    limits.mem_max_pct = pct.clamp(limits.mem_max_pct, 100);
    let watch_limits = limits;
    let ctx = StartContext {
        physical_cores: info.physical_cores as usize,
        logical_cores: info.logical_cores as usize,
        isa: info.isa.clone(),
        total_mem,
        available_mem: avail_mem,
        vram_total,
        vram_free,
        thermal_ok: true,
        acknowledge_no_thermal: true,
        limits,
    };

    println!(
        "Starting {mode} for {secs}s (ISA={:?}, GPU={:?})...",
        info.isa, info.gpu_stress_adapter
    );
    let st = mgr.start(cfg, ctx).expect("start stress");
    println!(
        "Started: isa={:?} cpu_threads={} mem_threads={} mem_target={}MiB vram_target={}MiB",
        st.isa_used,
        st.cpu_threads,
        st.mem_threads,
        st.mem_target_bytes >> 20,
        st.vram_target_bytes >> 20
    );

    let mut watchdog = safety::Watchdog::default();
    let start = Instant::now();
    let mut min_gpu_util = 100.0f32;
    let mut peak_gpu_util = 0f32;
    let mut peak_gpu_temp = 0f32;
    let mut peak_gpu_pwr = 0f32;
    let mut peak_cpu_load = 0f32;
    let mut peak_gflops = 0f64;
    let mut peak_bw = 0f64;
    let mut min_ram_pct = 100.0f32;
    let mut max_ram_pct = 0.0f32;
    while start.elapsed() < Duration::from_secs(secs as u64) {
        std::thread::sleep(Duration::from_millis(poll_ms));
        let st = mgr.status();
        let sample = engine.poll(None, SidecarState::NotStarted, mgr.snapshot());
        let g = sample.gpus.first();
        peak_cpu_load = peak_cpu_load.max(sample.cpu.total_load_pct);
        let gu = g.and_then(|x| x.util_pct).unwrap_or(0.0);
        peak_gpu_util = peak_gpu_util.max(gu);
        if st.elapsed_s >= 2.0 {
            min_gpu_util = min_gpu_util.min(gu);
        }
        peak_gpu_temp = peak_gpu_temp.max(g.and_then(|x| x.temp_c).unwrap_or(0.0));
        peak_gpu_pwr = peak_gpu_pwr.max(g.and_then(|x| x.power_w).unwrap_or(0.0));
        peak_gflops = peak_gflops.max(st.gflops);
        peak_bw = peak_bw.max(st.mem_mbps);
        if st.elapsed_s >= 3.0 {
            min_ram_pct = min_ram_pct.min(sample.memory.used_pct);
            max_ram_pct = max_ram_pct.max(sample.memory.used_pct);
        }
        let events = watchdog.evaluate(
            &safety::WatchInput {
                cpu_temp_c: sample.cpu.package_temp_c,
                gpu_temp_c: g.and_then(|x| x.temp_c),
                mem_used_pct: sample.memory.used_pct,
                mem_available_bytes: sample.memory.available_bytes,
                stress_active: st.phase.is_active(),
                stress_kind: st.kind,
                run_has_thermal: false,
                sidecar_fresh: false,
                severe_throttles: safety::severe_throttle_reasons(
                    &sample.cpu.throttle_flags,
                    g.map(|x| x.throttle_reasons.as_slice()).unwrap_or(&[]),
                ),
            },
            &watch_limits,
        );
        if let Some(safety::WatchEvent::Abort(reason, alert)) = events
            .into_iter()
            .find(|e| matches!(e, safety::WatchEvent::Abort(..)))
        {
            eprintln!("WATCHDOG SAFETY ABORT ({reason:?}): {}", alert.message);
            mgr.abort(reason);
            break;
        }
        let msg = st.message.as_deref().unwrap_or("");
        println!(
            "t={:4.1}s lvl={:3}% | CPU={:5.1}% {:4.0}MHz {:6.1} GFLOPS | RAM={:5.1}% ({}MiB alloc) | GPU={:5.1}% {:6.1} GFLOPS clk={:?}MHz temp={:?}C pwr={:?}/{:?}W vram={:?}MiB | BW={:6.0} MB/s | errs={} | {msg}",
            st.elapsed_s,
            st.load_level,
            sample.cpu.total_load_pct,
            sample.cpu.avg_effective_mhz.unwrap_or(0.0),
            st.gflops,
            sample.memory.used_pct,
            st.mem_allocated_bytes >> 20,
            g.and_then(|x| x.util_pct).unwrap_or(0.0),
            st.gpu_gflops,
            g.and_then(|x| x.core_mhz),
            g.and_then(|x| x.temp_c),
            g.and_then(|x| x.power_w),
            g.and_then(|x| x.power_limit_w),
            g.and_then(|x| x.vram_used_bytes).map(|b| b >> 20),
            st.mem_mbps,
            st.total_errors,
        );
        if !st.phase.is_active() {
            break;
        }
    }

    let t0 = Instant::now();
    let joined = mgr.abort_and_wait(AbortReason::User, Duration::from_secs(2));
    let abort_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let final_st = mgr.status();
    println!(
        "Done: phase={:?} joined={joined} latency={abort_ms:.1}ms errors={} peak_cpu={peak_cpu_load:.1}% peak_gflops={peak_gflops:.1} peak_bw={peak_bw:.0}MB/s ram_steady={min_ram_pct:.1}%..{max_ram_pct:.1}% gpu_steady={min_gpu_util:.1}%..{peak_gpu_util:.1}% peak_gpu_temp={peak_gpu_temp:.1}C peak_gpu_pwr={peak_gpu_pwr:.1}W",
        final_st.phase,
        final_st.total_errors
    );
    assert!(joined, "workers did not join");
    assert!(
        matches!(
            final_st.phase,
            StressPhase::Aborted | StressPhase::Completed
        ),
        "unexpected phase {:?}: {:?}",
        final_st.phase,
        final_st.message
    );
    assert_eq!(final_st.total_errors, 0, "unexpected hardware errors");
    println!("STRESS_CLI PASS ({mode})");
}
