# TestOC

High-resolution desktop hardware stability testing and system telemetry suite for Windows. Built with **Tauri v2 (Rust)**, **React + TypeScript + Tailwind CSS v4**, **`uPlot` Canvas charting**, **`wgpu` (Vulkan)** GPU compute/3D workloads, and a read-only **LibreHardwareMonitor (.NET 8)** sensor sidecar.

Designed to expose even `0.001%` marginal CPU, IMC, DRAM, GPU, and GDDR overclock instabilities with zero subjective heuristics — every stress kernel performs mathematically deterministic verification.

## Architecture & Stress Engines

- **Memory — Absolut 32-Step (Default) + Standard Profile Switcher (`src-tauri/src/stress/memory_kernel.rs`):**
  - **Absolut (Default):** Full 32-step DRAM & IMC torture sequence starting immediately on `T1` (`SimpleTest` Mode 2 LCG `0x77777777 / 0x33333333`) with rotating **128-byte 100%-row-miss Write-Back RFO** (max FCLK / L3 dirty writeback / IMC queue pressure), **128-byte Non-Temporal WC-bypass**, and **4 KiB max-bandwidth simultaneous cross-chunk Read↔Write turnarounds** (`tRDWR`/`tWRRD`/`tWTR`/`ODT`), plus opposing-end cross-allocation non-temporal `MirrorMove128` & 4 KiB `MirrorMove` with full 64-bit transit inversion, cross-chunk NT `BlockMove`, `Modulo20` bank-conflict scatter, `_mm_clflush` rowhammer sweeps, instant per-slice error reporting, and constant `99%+` live RAM reclamation (`∞`, `1C`, `3C`, `6C`, `12C`).
  - **Standard Profile:** Preserves the selectable bit-exact 5-pattern runner (`All 5`, `Moving Inversions`, `Strided Bank-Conflict`, `Walking 1s/0s`, `Address-in-Address`, `SplitMix64 Random`).
- **CPU & CPU+RAM — FFT Torture (`Smallest` / `Small` / `Large` / `Blend`) + Extreme FMA + 64-bit Goldilocks NTT + Core Cycler (`src-tauri/src/stress/cpu_kernel.rs`):**
  - **Split-Complex Floating-Point FFT Torture (`AVX2+FMA`, `AVX-512F`, `Scalar`):** Dual-engine self-squaring cyclic FFT convolution (`IFFT(FFT(z)^2)`) combining a `2×2` register-blocked **Bailey 4-Step L1D FMA power-virus** (`Smallest FFTs`: `4K–16K` L1D, >95% pure `vfmadd231pd`/`vfnmadd231pd` for peak core heat/PPT) and a **zero-bit-reversal fused Radix-4 DIF/DIT** SIMD engine (`Small FFTs`: `64K–256K` L2 heavy soak; `Large FFTs`: `4M–16M`/thread = `64–256 MiB` across 16T busting L3 to hammer IMC/FCLK/RAM; `Blend FFTs`: full `L1 → RAM` sweep), verifying a strict `< 0.40` floating-point rounding bound plus exact DC & Nyquist complex convolution sums in both **CPU** and **CPU+RAM** modes.
  - **Dense-Mantissa FMA GEMM:** 16-bit dense integer inputs (`-32767..=32767`) in `f64` exercising 40+ mantissa bits across `Scalar`, `AVX2+FMA`, and `AVX-512F` with `i128` analytical row/column checksums and alternating sign polarity.
  - **64-bit Goldilocks Prime NTT (`p = 2^64 - 2^32 + 1`):** 64-bit radix-2 Number Theoretic Transform with frequency-domain cyclic shift and full-buffer modular avalanche verification across `Small` (L1/L2), `Medium` (L2/L3), and `Large` (IMC/RAM) footprints.
  - **Load Modes:** `Steady` (all-core max heat), `Variable` (Vdroop transients), and `Core Cycler` (single-core peak Fmax hopping across physical cores for Curve Optimizer & per-core V/F tuning).
- **GPU & VRAM (`src-tauri/src/stress/gpu.rs`, `wgpu` Vulkan):**
  - **Extreme 3D (Default):** Full-TDP offscreen `2048×2048` 4×MRT `Rgba16Float` render pass with **dual-frame bit-exact 16.7M-pixel MRT hardware artifact comparison** plus analytical FP32 FMA verification; also offers `Heavy` (multi-unit compute + shared memory + 128 MiB GDDR stream), `Standard` (FP32 FMA), and `3D Adaptive` V/F ramp modes.
  - **VRAM (`10–100%`, OOM-safe):** 8 deterministic GDDR patterns including Absolut `0x77/0x33` crosstalk, VDDQ ground-bounce, and GPU-side `MirrorMove128` bidirectional stripe reversal.
- **Power / PSU:** Simultaneous full-thread CPU workload + full-TDP GPU workload.
- **Privilege-Isolated Telemetry & Safety Watchdog (`sensors-sidecar/`, `src-tauri/src/stress/safety.rs`):**
  - Elevated `.NET 8` LibreHardwareMonitorLib sidecar (`testoc-sensors.exe`) publishing over a SID-restricted seqlock shared-memory mapping, paired with NVML GPU telemetry, PDH clocks, debounced thermal/stall/error safety limits, global `ABORT (Esc)`, and UTF-8 BOM CSV + JSON summary export.

## Prerequisites

- **Rust** (stable toolchain; tested with `x86_64-pc-windows-gnu` + MinGW-w64)
- **Node.js** 20+ & `npm`
- **.NET 8 SDK** (`win-x64` self-contained single-file sidecar build)
- *(Optional for NSIS installer bundling)* Place the signed official `PawnIO_setup.exe` in `src-tauri/nsis/PawnIO_setup.exe` before running `npm run tauri build`.

## Development & Verification

```powershell
# UI development against built-in Mock IPC (no Rust/sidecar build needed)
npm run dev:mock

# Typecheck & unit tests
npm run typecheck
npm test

# Rust checks & unit/Vulkan integration tests
cd src-tauri
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test

# Browser E2E suite (Microsoft Edge, Mock IPC)
cd ..
npm run e2e

# Hardware CLI benchmark tool (Absolut RAM / CPU Extreme / CoreCycler / GPU Extreme / VRAM / Power)
cd src-tauri
cargo run --release --example stress_cli -- --mode mem --mem-profile absolut --pct 100 --secs 15
cargo run --release --example stress_cli -- --mode gpu --profile extreme --secs 10

# Production build & NSIS installer
cd ..
npm run tauri build
```

## License

[MIT](LICENSE) — Copyright (c) 2026 xDead
