//! Live telemetry probe: prints 10 samples with poll latency and system info.

use std::time::Duration;

use testoc_lib::model::{SidecarState, StressPhase, StressSnapshot};
use testoc_lib::stress::gpu::GpuStress;
use testoc_lib::telemetry::TelemetryEngine;

fn main() {
    let mut engine = TelemetryEngine::new();
    let gpu = GpuStress::probe();
    let info = engine.system_info(
        gpu.as_ref().map(|g| g.adapter_name().to_string()),
        gpu.as_ref().map(|g| g.fallback_vram_bytes()),
    );
    println!(
        "CPU: {} ({}C/{}T, base {} MHz, ISA avx2={} fma={} avx512={})",
        info.cpu_brand,
        info.physical_cores,
        info.logical_cores,
        info.base_mhz,
        info.isa.avx2,
        info.isa.fma,
        info.isa.avx512f
    );
    println!(
        "RAM: {:.1} GiB | PDH={} | NVML={} | GPU stress={:?}",
        info.total_memory_bytes as f64 / (1u64 << 30) as f64,
        info.pdh_clocks_available,
        info.nvml_available,
        info.gpu_stress_adapter
    );
    let snap = StressSnapshot {
        phase: StressPhase::Idle,
        kind: None,
        elapsed_s: 0.0,
        gflops: 0.0,
        gpu_gflops: 0.0,
        mem_mbps: 0.0,
        errors: 0,
        load_level: 0,
    };
    let mut max_poll_us = 0u32;
    for i in 1..=10 {
        std::thread::sleep(Duration::from_millis(500));
        let s = engine.poll(None, SidecarState::NotStarted, snap.clone());
        max_poll_us = max_poll_us.max(s.poll_duration_us);
        let g = s.gpus.first();
        println!(
            "#{i:02} poll={:.1}ms | CPU load={:5.1}% avgMHz={:6.0} | RAM={:4.1}% ({:.1}/{:.1} GiB) | GPU temp={:?}C util={:?}% vram={:?}MiB pwr={:?}W throttle={:?}",
            s.poll_duration_us as f64 / 1000.0,
            s.cpu.total_load_pct,
            s.cpu.avg_effective_mhz.unwrap_or(0.0),
            s.memory.used_pct,
            s.memory.used_bytes as f64 / (1u64 << 30) as f64,
            s.memory.total_bytes as f64 / (1u64 << 30) as f64,
            g.and_then(|x| x.temp_c),
            g.and_then(|x| x.util_pct),
            g.and_then(|x| x.vram_used_bytes).map(|b| b >> 20),
            g.and_then(|x| x.power_w),
            g.map(|x| &x.throttle_reasons)
        );
    }
    println!(
        "Max poll latency: {:.2} ms (target < 50 ms)",
        max_poll_us as f64 / 1000.0
    );
    assert!(
        max_poll_us < 50_000,
        "poll exceeded 50 ms: {max_poll_us} us"
    );
    println!("PROBE PASS");
}
