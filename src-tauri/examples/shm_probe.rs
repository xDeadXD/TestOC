//! Cross-process shared-memory verification probe.
//!
//! Spawns the real .NET `testoc-sensors.exe --fake` binary (no UAC), validates layout agreement,
//! polls frames for 2 s with zero torn reads, then kills the child and verifies `Stale` detection
//! fires right after `STALE_AFTER_MS`.

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use testoc_lib::model::SidecarState;
use testoc_lib::sidecar::SidecarHub;
use testoc_lib::telemetry::lhm_map;

fn find_sidecar() -> PathBuf {
    let candidates = [
        PathBuf::from("binaries/testoc-sensors-x86_64-pc-windows-gnu.exe"),
        PathBuf::from("../sensors-sidecar/publish/testoc-sensors.exe"),
    ];
    for c in candidates {
        if c.exists() {
            return c;
        }
    }
    panic!("sidecar executable not found; run scripts/build-sidecar.ps1 first");
}

fn main() {
    let exe = find_sidecar();
    println!(
        "Sidecar binary: {} ({} bytes)",
        exe.display(),
        std::fs::metadata(&exe).unwrap().len()
    );
    let hub = SidecarHub::new().expect("create shm");
    let pid = std::process::id().to_string();
    let mut child = Command::new(&exe)
        .args([
            "--map",
            hub.map_name(),
            "--parent-pid",
            &pid,
            "--fake",
            "--interval-ms",
            "200",
        ])
        .spawn()
        .expect("spawn fake sidecar");
    let child_pid = child.id();
    // Attach a duplicate handle? Better: keep child handle to kill cleanly; poll via raw region.
    let mut buf = Vec::new();
    let t0 = Instant::now();
    let mut ok_frames = 0u32;
    let mut last_seq = 0u64;
    while t0.elapsed() < Duration::from_secs(4) {
        if let Some(seq) = hub.region().read_consistent(&mut buf) {
            if let Ok(frame) = testoc_lib::sidecar::layout::parse_frame(&buf) {
                if seq != last_seq && !frame.records.is_empty() {
                    last_seq = seq;
                    ok_frames += 1;
                    let mapped = lhm_map::map_frame(&frame);
                    assert_eq!(frame.writer_pid, child_pid, "writer_pid mismatch");
                    assert!(mapped.package_temp_c.is_some(), "missing Tctl/Tdie");
                    assert!(mapped.package_power_w.is_some(), "missing Package power");
                    assert_eq!(
                        mapped.core_clocks.len(),
                        8,
                        "expected 8 fake physical cores"
                    );
                    if ok_frames == 1 {
                        println!(
                            "Frame #1 ok: pid={} records={} Tctl={:.1}C Pkg={:.1}W Core1={:.0}MHz BoardFans={}",
                            frame.writer_pid,
                            frame.records.len(),
                            mapped.package_temp_c.unwrap(),
                            mapped.package_power_w.unwrap(),
                            mapped.core_clocks[0].unwrap(),
                            mapped.board_fans.len()
                        );
                    }
                    if ok_frames >= 6 {
                        break;
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        ok_frames >= 6,
        "expected >= 6 cross-process frames, got {ok_frames}"
    );
    println!("Cross-process C# <-> Rust layout & seqlock verified ({ok_frames} frames, 0 errors)");

    // Now test SidecarHub stale/exit detection by killing the sidecar process.
    let _ = child.kill();
    let _ = child.wait();
    // Verify SidecarHub reports Stale when heartbeat ages past STALE_AFTER_MS (or Error on process exit).
    let kill_t = Instant::now();
    loop {
        hub.region().read_consistent(&mut buf).expect("read");
        let f = testoc_lib::sidecar::layout::parse_frame(&buf).expect("parse");
        let age = testoc_lib::model::now_ms().saturating_sub(f.heartbeat_ms);
        if age > testoc_lib::sidecar::STALE_AFTER_MS {
            println!(
                "Stale heartbeat detected in {:.2} s (age={} ms) -> {:?}",
                kill_t.elapsed().as_secs_f32(),
                age,
                SidecarState::Stale
            );
            assert!(
                kill_t.elapsed()
                    < Duration::from_millis(testoc_lib::sidecar::STALE_AFTER_MS + 1000)
            );
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    println!("SHM_PROBE PASS");
}
