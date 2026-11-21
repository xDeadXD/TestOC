//! GPU Compute, 3D Render, & VRAM stress engines built on `wgpu` (Vulkan backend).
//!
//! Selectable thermal/load profiles:
//! * **`GpuProfile::Standard`:** Double-buffered integer-exact unimodular FP32 FMA compute
//!   (`M = [[2, -3], [1, -1]]`, `M^6 = I`) across 65,536 threads with independent GPU + CPU
//!   workgroup-digest verification.
//! * **`GpuProfile::Heavy`:** Double-buffered multi-unit compute saturating FP32 FMA, INT32 IMAD,
//!   SFU transcendentals, workgroup shared memory (`var<workgroup>`), and 128 MiB VRAM streaming,
//!   with exact unimodular + INT32 verification.
//! * **`GpuProfile::Extreme`:** Full-TDP offscreen `2048x2048` 4xMRT (`Rgba16Float`) 3D render pass
//!   (L2-resident TMU sampling, dense FMA + SFU fragment loop, blended ROP writes across 128 MiB
//!   of color attachments) with periodic deterministic FP32 FMA error verification.
//!
//! Supports `LoadMode::Steady` (10-100%) and `LoadMode::Variable` (Adaptive boost-curve stepping),
//! auto-calibrates each dispatch window below Windows' 2 s TDR timeout, and checks shutdown flags every frame.

use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::manager::{RunShared, StressEvents};
use super::variable::adaptive_level;
use crate::model::{GpuProfile, LoadMode, MemoryPattern, WorkerErrorKind};

const WORKGROUP_SIZE: u32 = 256;
const COMPUTE_WORKGROUPS: u32 = 256; // 65,536 threads (exact verification grid)
const COMPUTE_THREADS: u32 = WORKGROUP_SIZE * COMPUTE_WORKGROUPS;
const HEAVY_WORKGROUPS: u32 = 2048; // 524,288 threads for Heavy profile
const HEAVY_STREAM_BYTES: u64 = 128 << 20; // 128 MiB GDDR streaming pressure
const MAX_DEVICE_BUFFER: u64 = 256 << 20; // device limit ceiling for Heavy + VRAM chunks
const RENDER_DIM: u32 = 2048; // 2048x2048 x 4 MRT Rgba16Float = 128 MiB ROP traffic/frame
const OVERDRAW_DRAWS: u32 = 1;
const TEX_DIM: u32 = 1024; // 4 MiB noise texture stays L2-resident to avoid DRAM-stall bubbles
const TARGET_DISPATCH_MS: f64 = 22.0; // short frames + 3-deep queue keep GPU at flat 100% with <100ms abort
const MIN_VRAM_TARGET: u64 = 64 << 20;
const CHUNK_ALIGN: u64 = 16 << 20;
const MAX_CHUNK_BYTES: u64 = 64 << 20;

/// Options controlling one GPU compute / 3D run.
#[derive(Debug, Clone, Copy)]
pub struct GpuRunOptions {
    pub profile: GpuProfile,
    pub mode: LoadMode,
    pub intensity: u8,
    pub var_min_pct: u8,
    pub var_max_pct: u8,
    pub var_step_s: u8,
    pub error_check: bool,
}

// Profile 1: Unimodular integer-exact FP32 FMA verification shader
const COMPUTE_WGSL: &str = r#"
struct Params {
    m00: f32,
    m01: f32,
    m10: f32,
    m11: f32,
    zero: f32,
    iters: u32,
    pass_id: u32,
    rem: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> wg_digest: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>>;

fn init_x(gid: u32, pair: u32, pass_id: u32) -> vec4<f32> {
    let b = (gid + pair * 7u + pass_id * 3u);
    return vec4<f32>(
        f32(i32((b) & 15u) - 8),
        f32(i32((b + 3u) & 15u) - 8),
        f32(i32((b + 7u) & 15u) - 8),
        f32(i32((b + 11u) & 15u) - 8),
    );
}

fn init_y(gid: u32, pair: u32, pass_id: u32) -> vec4<f32> {
    let b = (gid * 3u + pair * 11u + pass_id * 5u);
    return vec4<f32>(
        f32(i32((b) & 15u) - 8),
        f32(i32((b + 5u) & 15u) - 8),
        f32(i32((b + 9u) & 15u) - 8),
        f32(i32((b + 13u) & 15u) - 8),
    );
}

fn step_pair(x: vec4<f32>, y: vec4<f32>, m00: vec4<f32>, m01: vec4<f32>, m10: vec4<f32>, m11: vec4<f32>, z: vec4<f32>) -> array<vec4<f32>, 2> {
    let nx = fma(m00, x, fma(m01, y, z));
    let ny = fma(m10, x, fma(m11, y, z));
    return array<vec4<f32>, 2>(nx, ny);
}

fn expect_xy(x0: vec4<f32>, y0: vec4<f32>, r: u32) -> array<vec4<f32>, 2> {
    var x = x0;
    var y = y0;
    for (var i = 0u; i < r; i = i + 1u) {
        let nx = 2.0 * x - 3.0 * y;
        let ny = x - y;
        x = nx;
        y = ny;
    }
    return array<vec4<f32>, 2>(x, y);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let gid = gid3.x;
    let m00 = vec4<f32>(p.m00);
    let m01 = vec4<f32>(p.m01);
    let m10 = vec4<f32>(p.m10);
    let m11 = vec4<f32>(p.m11);
    let z = vec4<f32>(p.zero);

    var x0 = init_x(gid, 0u, p.pass_id); var y0 = init_y(gid, 0u, p.pass_id);
    var x1 = init_x(gid, 1u, p.pass_id); var y1 = init_y(gid, 1u, p.pass_id);
    var x2 = init_x(gid, 2u, p.pass_id); var y2 = init_y(gid, 2u, p.pass_id);
    var x3 = init_x(gid, 3u, p.pass_id); var y3 = init_y(gid, 3u, p.pass_id);
    var x4 = init_x(gid, 4u, p.pass_id); var y4 = init_y(gid, 4u, p.pass_id);
    var x5 = init_x(gid, 5u, p.pass_id); var y5 = init_y(gid, 5u, p.pass_id);
    var x6 = init_x(gid, 6u, p.pass_id); var y6 = init_y(gid, 6u, p.pass_id);
    var x7 = init_x(gid, 7u, p.pass_id); var y7 = init_y(gid, 7u, p.pass_id);

    for (var i = 0u; i < p.iters; i = i + 1u) {
        var t = step_pair(x0, y0, m00, m01, m10, m11, z); x0 = t[0]; y0 = t[1];
        t = step_pair(x1, y1, m00, m01, m10, m11, z); x1 = t[0]; y1 = t[1];
        t = step_pair(x2, y2, m00, m01, m10, m11, z); x2 = t[0]; y2 = t[1];
        t = step_pair(x3, y3, m00, m01, m10, m11, z); x3 = t[0]; y3 = t[1];
        t = step_pair(x4, y4, m00, m01, m10, m11, z); x4 = t[0]; y4 = t[1];
        t = step_pair(x5, y5, m00, m01, m10, m11, z); x5 = t[0]; y5 = t[1];
        t = step_pair(x6, y6, m00, m01, m10, m11, z); x6 = t[0]; y6 = t[1];
        t = step_pair(x7, y7, m00, m01, m10, m11, z); x7 = t[0]; y7 = t[1];
    }

    var bad = 0u;
    var e = expect_xy(init_x(gid, 0u, p.pass_id), init_y(gid, 0u, p.pass_id), p.rem);
    if (any(x0 != e[0]) || any(y0 != e[1])) { bad = bad + 1u; }
    e = expect_xy(init_x(gid, 1u, p.pass_id), init_y(gid, 1u, p.pass_id), p.rem);
    if (any(x1 != e[0]) || any(y1 != e[1])) { bad = bad + 1u; }
    e = expect_xy(init_x(gid, 2u, p.pass_id), init_y(gid, 2u, p.pass_id), p.rem);
    if (any(x2 != e[0]) || any(y2 != e[1])) { bad = bad + 1u; }
    e = expect_xy(init_x(gid, 3u, p.pass_id), init_y(gid, 3u, p.pass_id), p.rem);
    if (any(x3 != e[0]) || any(y3 != e[1])) { bad = bad + 1u; }
    e = expect_xy(init_x(gid, 4u, p.pass_id), init_y(gid, 4u, p.pass_id), p.rem);
    if (any(x4 != e[0]) || any(y4 != e[1])) { bad = bad + 1u; }
    e = expect_xy(init_x(gid, 5u, p.pass_id), init_y(gid, 5u, p.pass_id), p.rem);
    if (any(x5 != e[0]) || any(y5 != e[1])) { bad = bad + 1u; }
    e = expect_xy(init_x(gid, 6u, p.pass_id), init_y(gid, 6u, p.pass_id), p.rem);
    if (any(x6 != e[0]) || any(y6 != e[1])) { bad = bad + 1u; }
    e = expect_xy(init_x(gid, 7u, p.pass_id), init_y(gid, 7u, p.pass_id), p.rem);
    if (any(x7 != e[0]) || any(y7 != e[1])) { bad = bad + 1u; }

    if (bad > 0u) {
        atomicAdd(&stats[0], bad);
    }
    let s = x0 + y0 + x1 + y1 + x2 + y2 + x3 + y3 + x4 + y4 + x5 + y5 + x6 + y6 + x7 + y7;
    let sum_i = i32(s.x) + i32(s.y) + i32(s.z) + i32(s.w);
    atomicAdd(&wg_digest[wg.x], bitcast<u32>(sum_i));
}
"#;

// Profile 2: Multi-unit compute shader (FP32 FMA + INT32 IMAD + SFU + shared memory + 128 MiB stream)
const HEAVY_WGSL: &str = r#"
struct HeavyParams {
    m00: f32,
    m01: f32,
    m10: f32,
    m11: f32,
    iters: u32,
    pass_id: u32,
    rem: u32,
    stream_mask: u32,
};

@group(0) @binding(0) var<uniform> p: HeavyParams;
@group(0) @binding(1) var<storage, read_write> stream: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>>;

var<workgroup> smem: array<vec4<f32>, 256>;

fn init_v(seed: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32((seed) & 15u) - 8),
        f32(i32((seed + 3u) & 15u) - 8),
        f32(i32((seed + 7u) & 15u) - 8),
        f32(i32((seed + 11u) & 15u) - 8),
    );
}

fn expect_xy(x0: vec4<f32>, y0: vec4<f32>, r: u32) -> array<vec4<f32>, 2> {
    var x = x0;
    var y = y0;
    for (var i = 0u; i < r; i = i + 1u) {
        let nx = 2.0 * x - 3.0 * y;
        let ny = x - y;
        x = nx;
        y = ny;
    }
    return array<vec4<f32>, 2>(x, y);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid3: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
) {
    let gid = gid3.x;
    let lid = lid3.x;
    let m00 = vec4<f32>(p.m00);
    let m01 = vec4<f32>(p.m01);
    let m10 = vec4<f32>(p.m10);
    let m11 = vec4<f32>(p.m11);

    // 4 exact unimodular FP32 pairs (checked deterministically at end).
    var x0 = init_v(gid + p.pass_id * 3u);
    var y0 = init_v(gid * 3u + p.pass_id * 5u);
    var x1 = init_v(gid + 17u + p.pass_id * 7u);
    var y1 = init_v(gid * 5u + 29u + p.pass_id * 11u);
    var x2 = init_v(gid + 41u + p.pass_id * 13u);
    var y2 = init_v(gid * 7u + 53u + p.pass_id * 17u);
    var x3 = init_v(gid + 71u + p.pass_id * 19u);
    var y3 = init_v(gid * 11u + 83u + p.pass_id * 23u);

    // Simultaneous SFU + GDDR7 streaming heat accumulator & dual-issue INT32 exact chain.
    let base_idx = (gid * 7u + p.pass_id * 1021u) & p.stream_mask;
    var sfu = stream[base_idx] * 0.001 + vec4<f32>(0.37, 0.71, 1.13, 1.61);
    var ia = gid ^ (p.pass_id * 0x9E3779B9u);
    var ib = (gid * 0x85EBCA6Bu) ^ 0xA5A55A5Au;
    let ia0 = ia;
    let ib0 = ib;

    smem[lid] = sfu;
    workgroupBarrier();

    for (var i = 0u; i < p.iters; i = i + 1u) {
        // Exact FP32 FMA (8 independent FMAs per step).
        let nx0 = fma(m00, x0, m01 * y0); let ny0 = fma(m10, x0, m11 * y0); x0 = nx0; y0 = ny0;
        let nx1 = fma(m00, x1, m01 * y1); let ny1 = fma(m10, x1, m11 * y1); x1 = nx1; y1 = ny1;
        let nx2 = fma(m00, x2, m01 * y2); let ny2 = fma(m10, x2, m11 * y2); x2 = nx2; y2 = ny2;
        let nx3 = fma(m00, x3, m01 * y3); let ny3 = fma(m10, x3, m11 * y3); x3 = nx3; y3 = ny3;

        // Dual-issue INT32 invertible Feistel step (period-checked via forward/reverse symmetry).
        ia = ia + (ib * 1664525u + 1013904223u);
        ib = ib ^ ((ia >> 7u) | (ia << 25u));

        // SFU + Shared-memory L1 bank stress.
        let nb = smem[(lid + i + 1u) & 255u];
        sfu = fma(sin(sfu + nb * 0.125), vec4<f32>(0.85), cos(sfu * 1.37 - nb * 0.25));
        sfu = fma(inverseSqrt(abs(sfu) + vec4<f32>(1.0)), vec4<f32>(0.5), sfu * 0.5);

        if ((i & 15u) == 0u) {
            let idx = (base_idx + i * 257u) & p.stream_mask;
            let mem = stream[idx];
            sfu = fma(mem, vec4<f32>(0.01), sfu);
            stream[idx] = sfu;
            smem[lid] = sfu;
        }
    }

    // Reverse the INT32 Feistel chain to verify zero INT32 ALU bit flips!
    for (var i = 0u; i < p.iters; i = i + 1u) {
        ib = ib ^ ((ia >> 7u) | (ia << 25u));
        ia = ia - (ib * 1664525u + 1013904223u);
    }

    var bad = 0u;
    if (ia != ia0 || ib != ib0) { bad = bad + 1u; }
    var e = expect_xy(init_v(gid + p.pass_id * 3u), init_v(gid * 3u + p.pass_id * 5u), p.rem);
    if (any(x0 != e[0]) || any(y0 != e[1])) { bad = bad + 1u; }
    e = expect_xy(init_v(gid + 17u + p.pass_id * 7u), init_v(gid * 5u + 29u + p.pass_id * 11u), p.rem);
    if (any(x1 != e[0]) || any(y1 != e[1])) { bad = bad + 1u; }
    e = expect_xy(init_v(gid + 41u + p.pass_id * 13u), init_v(gid * 7u + 53u + p.pass_id * 17u), p.rem);
    if (any(x2 != e[0]) || any(y2 != e[1])) { bad = bad + 1u; }
    e = expect_xy(init_v(gid + 71u + p.pass_id * 19u), init_v(gid * 11u + 83u + p.pass_id * 23u), p.rem);
    if (any(x3 != e[0]) || any(y3 != e[1])) { bad = bad + 1u; }
    if (any(sfu != sfu)) { bad = bad + 1u; } // NaN guard on SFU path

    // Commit final SFU result to GDDR7 so the compiler cannot dead-code eliminate it.
    stream[(base_idx + 1u) & p.stream_mask] = sfu;
    if (bad > 0u) {
        atomicAdd(&stats[0], bad);
    }
}
"#;

// Profile 3: Full-TDP 3D fragment shader balancing FP32 FMA, SFU, L2 TMU fetches, and 4xMRT ROPs.
// Register count stays low (<=32 live regs/thread via 4 vec4 accumulators) for full SM occupancy.
const RENDER_3D_WGSL: &str = r#"
struct RenderParams {
    time_s: f32,
    iters: u32,
    pass_id: u32,
    draw_id: u32,
};

@group(0) @binding(0) var<uniform> rp: RenderParams;
@group(0) @binding(1) var noise_tex: texture_2d<f32>;
@group(0) @binding(2) var noise_samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>( 3.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let p = positions[vi];
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * 0.5 + vec2<f32>(0.5);
    return out;
}

struct MrtOut {
    @location(0) c0: vec4<f32>,
    @location(1) c1: vec4<f32>,
    @location(2) c2: vec4<f32>,
    @location(3) c3: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut) -> MrtOut {
    let t = rp.time_s;
    let p0 = (in.uv - vec2<f32>(0.5)) * 2.0;

    // Keep register count compact (<=32 live regs/thread via 4 vec4 accumulators) for full SM occupancy.
    var a = vec4<f32>( 1.137 + p0.x, -0.893 + p0.y,  1.414 - p0.x, -1.732 - p0.y);
    var b = vec4<f32>(-1.271 - p0.y,  0.947 + p0.x, -1.618 + p0.y,  1.319 - p0.x);
    var c = vec4<f32>( 0.783 + p0.y, -1.531 - p0.x,  0.641 + p0.x, -1.193 + p0.y);
    var d = vec4<f32>(-0.917 + p0.x,  1.673 - p0.y, -0.829 - p0.y,  1.487 + p0.x);
    var u = (u32(in.pos.x) * 0x9E3779B1u) ^ (u32(in.pos.y) * 0x85EBCA77u) ^ (rp.pass_id * 0xC2B2AE3Du) ^ 0x55555555u;

    // Orthogonal sign-alternating rotation constants (bounded magnitude, high bit-toggle rate).
    let c0 = vec4<f32>( 0.8660254, -0.8660254,  0.8660254, -0.8660254);
    let s0 = vec4<f32>( 0.5,        0.5,       -0.5,       -0.5);
    let c1 = vec4<f32>(-0.7071068,  0.7071068, -0.7071068,  0.7071068);
    let s1 = vec4<f32>( 0.7071068,  0.7071068,  0.7071068,  0.7071068);
    var uv = in.uv + vec2<f32>(sin(t * 0.37) * 0.25, cos(t * 0.29) * 0.25);

    for (var i = 0u; i < rp.iters; i = i + 1u) {
        let fi = f32(i) * 0.0625 + t;

        // 2 L2-resident bilinear texture taps keep TMUs and L1/L2 caches active without DRAM stalls.
        let tx0 = textureSample(noise_tex, noise_samp, uv) * 2.0 - vec4<f32>(1.0);
        let tx1 = textureSample(noise_tex, noise_samp, uv * 1.73 + vec2<f32>(b.x * 0.11, b.y * 0.11)) * 2.0 - vec4<f32>(1.0);

        // 5 rounds of energy-preserving 2D rotations (160 FP32 FMAs/iter) scale to arbitrary GPU power limits.
        var na = fma(a, c0, b * s0); var nb = fma(b, c0, -a * s0); a = na; b = nb;
        var nc = fma(c, c1, d * s1); var nd = fma(d, c1, -c * s1); c = nc; d = nd;

        u = ((u * 0x9E3779B1u) + 0x85EBCA77u) ^ 0xAAAAAAAAu;

        na = fma(a, c1, c * s1); nc = fma(c, c1, -a * s1); a = na; c = nc;
        nb = fma(b, c0, d * s0); nd = fma(d, c0, -b * s0); b = nb; d = nd;

        u = ((u * 0xC2B2AE3Du) + 0x27D4EB2Fu) ^ 0x55555555u;

        na = fma(a, c0, d * s0); nd = fma(d, c0, -a * s0); a = na; d = nd;
        nb = fma(b, c1, c * s1); nc = fma(c, c1, -b * s1); b = nb; c = nc;

        u = u ^ ((u >> 11u) | (u << 21u));

        na = fma(a, c1, -b * s1); nb = fma(b, c1, a * s1); a = na; b = nb;
        nc = fma(c, c0, -d * s0); nd = fma(d, c0, c * s0); c = nc; d = nd;

        u = ((u * 0x165667B1u) + 0x9E3779B9u) ^ 0xAAAAAAAAu;

        na = fma(a, c0, -c * s0); nc = fma(c, c0, a * s0); a = na; c = nc;
        nb = fma(b, c1, -d * s1); nd = fma(d, c1, b * s1); b = nb; d = nd;

        // Inject L2 texture samples + balanced SFU ops alongside FP32 and INT32 paths.
        let sn = sin(a + vec4<f32>(fi));
        let cs = cos(b - vec4<f32>(fi * 1.13));
        let ir = inverseSqrt(dot(c, c) + 0.5);

        a = fma(sn, vec4<f32>(0.72), fma(tx0, vec4<f32>(0.68), a * vec4<f32>(-0.25)));
        b = fma(cs, vec4<f32>(-0.72), fma(tx1, vec4<f32>(0.68), b * vec4<f32>(0.25)));
        c = fma(c, vec4<f32>(ir * 0.9), fma(tx1, vec4<f32>(-0.62), d * vec4<f32>(0.35)));
        d = fma(d, vec4<f32>(-0.78), fma(tx0, vec4<f32>(0.62), sn * cs));

        uv = fract(uv + vec2<f32>(0.0137, 0.0193) + a.xy * 0.031);
    }

    let iu = f32(u & 0xFFFFu) * 0.000015258789;
    var out: MrtOut;
    out.c0 = vec4<f32>(abs(a.xyz) * 0.25 + vec3<f32>(iu), 0.5);
    out.c1 = vec4<f32>(abs(b.xyz) * 0.25 - vec3<f32>(iu), 0.5);
    out.c2 = vec4<f32>(abs(c.xyz) * 0.25 + vec3<f32>(iu), 0.5);
    out.c3 = vec4<f32>(abs(d.xyz) * 0.25 - vec3<f32>(iu), 0.5);
    return out;
}
"#;

// Dual-frame bit-exact 4xMRT pixel artifact comparator (verifies 16.7M Rgba16Float pixels at full TDP).
const MRT_COMPARE_WGSL: &str = r#"
@group(0) @binding(0) var a0: texture_2d<f32>;
@group(0) @binding(1) var a1: texture_2d<f32>;
@group(0) @binding(2) var a2: texture_2d<f32>;
@group(0) @binding(3) var a3: texture_2d<f32>;
@group(0) @binding(4) var b0: texture_2d<f32>;
@group(0) @binding(5) var b1: texture_2d<f32>;
@group(0) @binding(6) var b2: texture_2d<f32>;
@group(0) @binding(7) var b3: texture_2d<f32>;
@group(0) @binding(8) var<storage, read_write> stats: array<atomic<u32>>;

fn px_diff(x: vec4<f32>, y: vec4<f32>) -> u32 {
    let bx = bitcast<vec4<u32>>(x);
    let by = bitcast<vec4<u32>>(y);
    return select(0u, 1u, any(bx != by));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(a0);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    var bad = 0u;
    bad = bad + px_diff(textureLoad(a0, xy, 0), textureLoad(b0, xy, 0));
    bad = bad + px_diff(textureLoad(a1, xy, 0), textureLoad(b1, xy, 0));
    bad = bad + px_diff(textureLoad(a2, xy, 0), textureLoad(b2, xy, 0));
    bad = bad + px_diff(textureLoad(a3, xy, 0), textureLoad(b3, xy, 0));
    if (bad > 0u) {
        atomicAdd(&stats[1], bad);
    }
}
"#;

// VRAM pattern fill, MirrorMove128 permutation, & verification compute shader (8 deterministic patterns)
const VRAM_WGSL: &str = r#"
struct VramParams {
    word_count: u32,
    chunk_id: u32,
    pass_id: u32,
    pattern: u32,
};

@group(0) @binding(0) var<uniform> vp: VramParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> errs: array<atomic<u32>>;

fn hash32(x_in: u32) -> u32 {
    var x = x_in;
    x = x ^ (x >> 16u);
    x = x * 0x7FEB352Du;
    x = x ^ (x >> 15u);
    x = x * 0x846CA68Bu;
    return x ^ (x >> 16u);
}

fn expected_word(idx: u32, chunk_id: u32, pass_id: u32, pat: u32) -> u32 {
    if (pat == 0u) {
        // Walking 1s (even passes) & complementary Walking 0s (odd passes)
        let bit = 1u << ((idx + chunk_id + pass_id) & 31u);
        return select(~bit, bit, (pass_id & 1u) == 0u);
    } else if (pat == 1u) {
        // Address-in-Address
        return (chunk_id << 25u) ^ idx ^ (pass_id * 0x9E3779B9u);
    } else if (pat == 2u) {
        // PRNG Random (Murmur3 finalizer)
        return hash32(idx ^ (chunk_id * 0x85EBCA6Bu) ^ (pass_id * 0xC2B2AE35u));
    } else if (pat == 3u) {
        // Moving Inversions: alternating 0x55555555/0xAAAAAAAA and 0xFFFFFFFF/0x00000000
        let mode = (pass_id / 8u) & 3u;
        var base = 0x55555555u;
        if (mode == 1u) { base = 0xAAAAAAAAu; }
        else if (mode == 2u) { base = 0xFFFFFFFFu; }
        else if (mode == 3u) { base = 0x00000000u; }
        return select(~base, base, (idx & 1u) == 0u);
    } else if (pat == 4u) {
        // Strided Bank-Conflict high-entropy inverted cache-line pattern
        let v = hash32((idx * 4099u) ^ (chunk_id * 0x9E3779B9u) ^ (pass_id * 0x7FEB352Du));
        return select(~v, v, ((idx >> 3u) & 1u) == 0u);
    } else if (pat == 5u) {
        // Absolut M2 Crosstalk: 0x77777777 / 0x33333333 (inverted on odd passes)
        let a = select(0x88888888u, 0x77777777u, (pass_id & 1u) == 0u);
        let b = select(0xCCCCCCCCu, 0x33333333u, (pass_id & 1u) == 0u);
        return select(b, a, (idx & 1u) == 0u);
    } else if (pat == 6u) {
        // Absolut VDDQ Ground-Bounce: 0x00/0xFF on even 64B lines, 0x00FF/0xFF00 on odd 64B lines
        let odd_line = ((idx >> 4u) & 1u) != 0u;
        let odd_word = (idx & 1u) != 0u;
        if (!odd_line) {
            return select(0x00000000u, 0xFFFFFFFFu, odd_word);
        }
        return select(0x00FF00FFu, 0xFF00FF00u, odd_word);
    } else {
        // Absolut MirrorMove128 source signature (swapped in-place by mirror128 before verify)
        return hash32((idx ^ 0xA5A55A5Au) ^ (chunk_id * 0x9E3779B9u) ^ (pass_id * 0x27D4EB2Fu));
    }
}

@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let stride = 256u * 4096u;
    var i = gid3.x;
    while (i < vp.word_count) {
        data[i] = expected_word(i, vp.chunk_id, vp.pass_id, vp.pattern);
        i = i + stride;
    }
}

@compute @workgroup_size(256)
fn mirror128(@builtin(global_invocation_id) gid3: vec3<u32>) {
    // 128-byte stripe = 32 u32 words. Swap stripe s <-> (stripes - 1 - s) across opposing ends.
    let stripes = vp.word_count >> 5u;
    let half_words = (stripes >> 1u) << 5u;
    let stride = 256u * 4096u;
    var i = gid3.x;
    while (i < half_words) {
        let s = i >> 5u;
        let off = i & 31u;
        let j = ((stripes - 1u - s) << 5u) | off;
        let a = data[i];
        let b = data[j];
        data[i] = b;
        data[j] = a;
        i = i + stride;
    }
}

@compute @workgroup_size(256)
fn verify(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let stride = 256u * 4096u;
    let stripes = vp.word_count >> 5u;
    let active_words = stripes << 5u;
    var i = gid3.x;
    var local_bad = 0u;
    var first_idx = 0xFFFFFFFFu;
    while (i < vp.word_count) {
        let got = data[i];
        var src_idx = i;
        if (vp.pattern == 7u && i < active_words) {
            let s = i >> 5u;
            let off = i & 31u;
            src_idx = ((stripes - 1u - s) << 5u) | off;
        }
        let exp = expected_word(src_idx, vp.chunk_id, vp.pass_id, vp.pattern);
        if (got != exp) {
            local_bad = local_bad + 1u;
            if (first_idx == 0xFFFFFFFFu) {
                first_idx = i;
            }
        }
        i = i + stride;
    }
    if (local_bad > 0u) {
        atomicAdd(&errs[0], local_bad);
        atomicMin(&errs[1], first_idx);
    }
}
"#;

/// Universal VRAM sizing: scales all the way from 10% to 100% of total VRAM with lean DWM headroom
/// (64-256 MiB depending on target aggressiveness) and safe Vulkan OOM clamping per 64 MiB chunk.
/// Returns `(total_target_bytes, chunk_bytes)`.
pub fn compute_vram_target_bytes(
    total_vram: u64,
    free_vram: Option<u64>,
    target_pct: u8,
    max_binding_size: u64,
) -> Result<(u64, u64), String> {
    if !(10..=100).contains(&target_pct) {
        return Err(format!("VRAM target {target_pct}% out of range 10-100"));
    }
    if total_vram < 512 << 20 {
        return Err("detected VRAM too small for VRAM stress (< 512 MiB)".into());
    }
    let headroom = if target_pct >= 98 {
        64 << 20
    } else if target_pct >= 90 {
        128 << 20
    } else {
        (total_vram / 50).clamp(128 << 20, 256 << 20)
    };
    let want = if target_pct >= 100 {
        total_vram
    } else {
        total_vram / 100 * target_pct as u64
    };
    let avail_cap = match free_vram {
        Some(free) => free.saturating_sub(headroom),
        None => total_vram.saturating_sub(headroom.saturating_mul(2)),
    };
    let raw_chunk = max_binding_size.min(MAX_CHUNK_BYTES);
    let chunk = (raw_chunk / CHUNK_ALIGN) * CHUNK_ALIGN;
    if chunk < CHUNK_ALIGN {
        return Err("GPU max storage buffer binding size too small (< 16 MiB)".into());
    }
    let target = (want.min(avail_cap) / chunk) * chunk;
    if target < MIN_VRAM_TARGET.min(chunk) || target < 64 << 20 {
        return Err(format!(
            "not enough free VRAM after reserving {} MiB DWM headroom (available target: {} MiB)",
            headroom >> 20,
            want.min(avail_cap) >> 20
        ));
    }
    Ok((target, chunk))
}

/// CPU-side analytical expected workgroup digests for `COMPUTE_WGSL`.
/// Uses the exact period-16 symmetry of `init_x`/`init_y` across `WORKGROUP_SIZE = 256 = 16 * 16`
/// so verification computes in ~2 us instead of stalling the GPU submission thread for ~20 ms.
pub fn expected_compute_digests(pass_id: u32, rem: u32) -> Vec<u32> {
    let mut acc16: i32 = 0;
    for tid in 0..16u32 {
        for pair in 0..8u32 {
            let bx = tid.wrapping_add(pair * 7).wrapping_add(pass_id * 3);
            let by = tid
                .wrapping_mul(3)
                .wrapping_add(pair * 11)
                .wrapping_add(pass_id * 5);
            for lane in 0..4u32 {
                let ox = [0u32, 3, 7, 11][lane as usize];
                let oy = [0u32, 5, 9, 13][lane as usize];
                let mut x = ((bx.wrapping_add(ox) & 15) as i32) - 8;
                let mut y = ((by.wrapping_add(oy) & 15) as i32) - 8;
                for _ in 0..rem {
                    let nx = 2 * x - 3 * y;
                    let ny = x - y;
                    x = nx;
                    y = ny;
                }
                acc16 = acc16.wrapping_add(x).wrapping_add(y);
            }
        }
    }
    let wg_digest = acc16.wrapping_mul((WORKGROUP_SIZE / 16) as i32) as u32;
    vec![wg_digest; COMPUTE_WORKGROUPS as usize]
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ComputeParams {
    m00: f32,
    m01: f32,
    m10: f32,
    m11: f32,
    zero: f32,
    iters: u32,
    pass_id: u32,
    rem: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HeavyParams {
    m00: f32,
    m01: f32,
    m10: f32,
    m11: f32,
    iters: u32,
    pass_id: u32,
    rem: u32,
    stream_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RenderParams {
    time_s: f32,
    iters: u32,
    pass_id: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VramParams {
    word_count: u32,
    chunk_id: u32,
    pass_id: u32,
    pattern: u32,
}

fn as_bytes<T: Copy>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((v as *const T) as *const u8, std::mem::size_of::<T>()) }
}

pub struct GpuStress {
    adapter: wgpu::Adapter,
    adapter_name: String,
    max_binding_size: u64,
    fallback_vram_bytes: u64,
}

impl GpuStress {
    pub fn probe() -> Option<GpuStress> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| log::warn!("wgpu Vulkan adapter unavailable: {e}"))
        .ok()?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let name = format!("{} (Vulkan)", info.name);
        log::info!(
            "GPU stress adapter: {name} | max_binding={} MiB | max_buffer={} MiB",
            limits.max_storage_buffer_binding_size >> 20,
            limits.max_buffer_size >> 20
        );
        Some(GpuStress {
            adapter,
            adapter_name: name,
            max_binding_size: limits.max_storage_buffer_binding_size,
            fallback_vram_bytes: limits.max_buffer_size.clamp(1 << 30, 8 << 30),
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn max_binding_size(&self) -> u64 {
        self.max_binding_size
    }

    pub fn fallback_vram_bytes(&self) -> u64 {
        self.fallback_vram_bytes
    }

    fn create_device(&self) -> Result<(wgpu::Device, wgpu::Queue), String> {
        let mut limits = wgpu::Limits::default();
        let al = self.adapter.limits();
        limits.max_storage_buffer_binding_size =
            al.max_storage_buffer_binding_size.min(MAX_DEVICE_BUFFER);
        limits.max_buffer_size = al.max_buffer_size.min(MAX_DEVICE_BUFFER);
        limits.max_texture_dimension_2d = al.max_texture_dimension_2d.min(4096);
        limits.max_color_attachments = al.max_color_attachments.min(8);
        limits.max_color_attachment_bytes_per_sample =
            al.max_color_attachment_bytes_per_sample.min(64);
        pollster::block_on(self.adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("testoc-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("GPU device creation failed: {e}"))
    }

    /// Dispatch the configured GPU stress profile (Standard, Heavy, or Extreme).
    pub fn run_gpu(
        &self,
        shared: &RunShared,
        events: &dyn StressEvents,
        opts: GpuRunOptions,
        add_flops: impl Fn(u64),
    ) {
        match opts.profile {
            GpuProfile::Standard => self.run_standard(shared, events, opts, add_flops),
            GpuProfile::Heavy => self.run_heavy(shared, events, opts, add_flops),
            GpuProfile::Extreme => self.run_extreme_3d(shared, events, opts, add_flops),
        }
    }

    fn current_level(shared: &RunShared, opts: GpuRunOptions) -> u8 {
        let lvl = match opts.mode {
            LoadMode::Steady | LoadMode::CoreCycler => opts.intensity.clamp(10, 100),
            LoadMode::Variable => adaptive_level(
                shared.started.elapsed(),
                opts.var_min_pct,
                opts.var_max_pct,
                opts.var_step_s,
            ),
        };
        shared.load_level.store(lvl, Ordering::Relaxed);
        lvl
    }

    // Profile 1: Standard (double-buffered exact FP32 FMA compute)
    fn run_standard(
        &self,
        shared: &RunShared,
        events: &dyn StressEvents,
        opts: GpuRunOptions,
        add_flops: impl Fn(u64),
    ) {
        let (device, queue) = match self.create_device() {
            Ok(dq) => dq,
            Err(e) => {
                shared.record_error(events, 0, None, WorkerErrorKind::GpuDevice, e);
                return;
            }
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("testoc-compute"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("testoc-compute-pipe"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let digest_bytes = (COMPUTE_WORKGROUPS as u64) * 4;
        let bgl = pipeline.get_bind_group_layout(0);
        let mk_slot = |idx: usize| {
            let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compute-params"),
                size: std::mem::size_of::<ComputeParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let digest_gpu = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wg-digest"),
                size: digest_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let digest_cpu = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wg-digest-readback"),
                size: digest_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let stats_gpu = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compute-stats"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let stats_cpu = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compute-stats-readback"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(if idx == 0 {
                    "compute-bg-0"
                } else {
                    "compute-bg-1"
                }),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: digest_gpu.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: stats_gpu.as_entire_binding(),
                    },
                ],
            });
            (uniform, digest_gpu, digest_cpu, stats_gpu, stats_cpu, bg)
        };
        // 4-slot command ring: 3 frames ALWAYS queued ahead on the GPU so utilization never dips.
        let slots = [mk_slot(0), mk_slot(1), mk_slot(2), mk_slot(3)];

        struct FlightStd {
            si: usize,
            pass_id: u32,
            rem: u32,
            flops: u64,
            maps: Option<(PendingMap, PendingMap)>,
            sub: wgpu::SubmissionIndex,
        }

        let mut base_k: u32 = 300;
        let mut pass_id: u32 = 0;
        let mut in_flight: VecDeque<FlightStd> = VecDeque::with_capacity(4);
        let mut last_complete = Instant::now();

        let finish_std = |f: FlightStd| -> bool {
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: Some(f.sub),
                timeout: None,
            });
            if let Some((p_stats, p_digest)) = f.maps {
                let expected = expected_compute_digests(f.pass_id, f.rem);
                let (Some(stats_words), Some(digest_words)) = (
                    p_stats.finish(&device, &slots[f.si].4),
                    p_digest.finish(&device, &slots[f.si].2),
                ) else {
                    shared.record_error(
                        events,
                        0,
                        None,
                        WorkerErrorKind::GpuDevice,
                        "GPU readback failed".into(),
                    );
                    return false;
                };
                let lane_bad = stats_words.first().copied().unwrap_or(0) as u64;
                let digest_bad = digest_words
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a != b)
                    .count() as u64;
                let errs = lane_bad.max(digest_bad);
                if errs > 0 {
                    if errs > 1 {
                        shared.add_extra_gpu_errors(errs - 1);
                    }
                    shared.record_error(
                        events,
                        0,
                        None,
                        WorkerErrorKind::GpuMismatch,
                        format!(
                            "GPU compute mismatch (pass {}: lane={lane_bad}, digest={digest_bad})",
                            f.pass_id
                        ),
                    );
                }
            }
            add_flops(f.flops);
            true
        };

        while shared.should_continue() {
            let lvl = Self::current_level(shared, opts);
            if lvl < 100 {
                while let Some(f) = in_flight.pop_front() {
                    if !finish_std(f) {
                        let _ = device.poll(wgpu::PollType::wait_indefinitely());
                        return;
                    }
                }
            }

            let si = (pass_id as usize) % slots.len();
            let rem = (pass_id % 5) + 1;
            let iters = base_k * 6 + rem;
            let params = ComputeParams {
                m00: 2.0,
                m01: -3.0,
                m10: 1.0,
                m11: -1.0,
                zero: 0.0,
                iters,
                pass_id,
                rem,
            };
            let (uniform, digest_gpu, digest_cpu, stats_gpu, stats_cpu, bg) = &slots[si];
            queue.write_buffer(uniform, 0, as_bytes(&params));

            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.clear_buffer(digest_gpu, 0, None);
            enc.clear_buffer(stats_gpu, 0, None);
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, bg, &[]);
                cpass.dispatch_workgroups(COMPUTE_WORKGROUPS, 1, 1);
            }
            if opts.error_check {
                enc.copy_buffer_to_buffer(digest_gpu, 0, digest_cpu, 0, digest_bytes);
                enc.copy_buffer_to_buffer(stats_gpu, 0, stats_cpu, 0, 16);
            }
            let t_burst = Instant::now();
            let sub = queue.submit(Some(enc.finish()));
            // Start async maps IMMEDIATELY on this submission (before later slots submit).
            let maps = if opts.error_check {
                Some((PendingMap::start(stats_cpu), PendingMap::start(digest_cpu)))
            } else {
                None
            };
            let flops = (COMPUTE_THREADS as u64) * (iters as u64) * 256;
            let flight = FlightStd {
                si,
                pass_id,
                rem,
                flops,
                maps,
                sub,
            };

            if lvl < 100 {
                if !finish_std(flight) {
                    break;
                }
                let dt_ms = t_burst.elapsed().as_secs_f64() * 1000.0;
                base_k = adjust_iters_damped(base_k, dt_ms, 16.0, 16, 20_000);
                if shared.should_continue() {
                    let sleep_ms = dt_ms * ((100 - lvl) as f64 / lvl.max(10) as f64);
                    sleep_interruptible(
                        Duration::from_secs_f64((sleep_ms / 1000.0).min(0.25)),
                        || shared.should_continue(),
                    );
                }
                last_complete = Instant::now();
            } else {
                in_flight.push_back(flight);
                if in_flight.len() >= slots.len() {
                    if let Some(oldest) = in_flight.pop_front() {
                        if !finish_std(oldest) {
                            break;
                        }
                        let dt_ms = last_complete.elapsed().as_secs_f64() * 1000.0;
                        last_complete = Instant::now();
                        base_k = adjust_iters_damped(base_k, dt_ms, TARGET_DISPATCH_MS, 16, 20_000);
                    }
                } else {
                    last_complete = Instant::now();
                }
            }
            pass_id = pass_id.wrapping_add(1);
        }
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }

    // Profile 2: Heavy (double-buffered FP32+INT32+SFU+shared-memory+128 MiB VRAM compute)
    fn run_heavy(
        &self,
        shared: &RunShared,
        events: &dyn StressEvents,
        opts: GpuRunOptions,
        add_flops: impl Fn(u64),
    ) {
        let (device, queue) = match self.create_device() {
            Ok(dq) => dq,
            Err(e) => {
                shared.record_error(events, 0, None, WorkerErrorKind::GpuDevice, e);
                return;
            }
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("testoc-heavy"),
            source: wgpu::ShaderSource::Wgsl(HEAVY_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("testoc-heavy-pipe"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let stream_bytes = HEAVY_STREAM_BYTES.min(self.max_binding_size);
        // Round down to power-of-two vec4 count for cheap `& stream_mask` indexing.
        let vec4_count = (stream_bytes / 16).next_power_of_two() / 2;
        let alloc_bytes = (vec4_count * 16).max(16 << 20);
        let stream_mask = ((alloc_bytes / 16) - 1) as u32;

        let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let stream_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("heavy-stream"),
            size: alloc_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        if pollster::block_on(oom.pop()).is_some() {
            shared.record_error(
                events,
                0,
                None,
                WorkerErrorKind::AllocFailed,
                "Heavy GPU stream buffer OOM".into(),
            );
            return;
        }

        let bgl = pipeline.get_bind_group_layout(0);
        let mk_slot = || {
            let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("heavy-params"),
                size: std::mem::size_of::<HeavyParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let stats_gpu = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("heavy-stats"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let stats_cpu = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("heavy-stats-rb"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("heavy-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: stream_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: stats_gpu.as_entire_binding(),
                    },
                ],
            });
            (uniform, stats_gpu, stats_cpu, bg)
        };
        // 4-slot command ring: 3 frames ALWAYS queued ahead on the GPU so utilization stays flat 100%.
        let slots = [mk_slot(), mk_slot(), mk_slot(), mk_slot()];

        struct FlightHeavy {
            si: usize,
            pass_id: u32,
            flops: u64,
            map: Option<PendingMap>,
            sub: wgpu::SubmissionIndex,
        }

        let mut base_k: u32 = 24;
        let mut pass_id: u32 = 0;
        let mut in_flight: VecDeque<FlightHeavy> = VecDeque::with_capacity(4);
        let mut last_complete = Instant::now();

        let finish_heavy = |f: FlightHeavy| -> bool {
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: Some(f.sub),
                timeout: None,
            });
            if let Some(p_stats) = f.map {
                let Some(words) = p_stats.finish(&device, &slots[f.si].2) else {
                    shared.record_error(
                        events,
                        0,
                        None,
                        WorkerErrorKind::GpuDevice,
                        "Heavy GPU readback failed".into(),
                    );
                    return false;
                };
                let bad = words.first().copied().unwrap_or(0) as u64;
                if bad > 0 {
                    if bad > 1 {
                        shared.add_extra_gpu_errors(bad - 1);
                    }
                    shared.record_error(
                        events,
                        0,
                        None,
                        WorkerErrorKind::GpuMismatch,
                        format!(
                            "Heavy GPU verification error (pass {}: {bad} mismatches)",
                            f.pass_id
                        ),
                    );
                }
            }
            add_flops(f.flops);
            true
        };

        while shared.should_continue() {
            let lvl = Self::current_level(shared, opts);
            if lvl < 100 {
                while let Some(f) = in_flight.pop_front() {
                    if !finish_heavy(f) {
                        let _ = device.poll(wgpu::PollType::wait_indefinitely());
                        return;
                    }
                }
            }

            let si = (pass_id as usize) % slots.len();
            let rem = (pass_id % 5) + 1;
            let iters = base_k * 6 + rem;
            let params = HeavyParams {
                m00: 2.0,
                m01: -3.0,
                m10: 1.0,
                m11: -1.0,
                iters,
                pass_id,
                rem,
                stream_mask,
            };
            let (uniform, stats_gpu, stats_cpu, bg) = &slots[si];
            queue.write_buffer(uniform, 0, as_bytes(&params));

            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.clear_buffer(stats_gpu, 0, None);
            {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cp.set_pipeline(&pipeline);
                cp.set_bind_group(0, bg, &[]);
                cp.dispatch_workgroups(HEAVY_WORKGROUPS, 1, 1);
            }
            if opts.error_check {
                enc.copy_buffer_to_buffer(stats_gpu, 0, stats_cpu, 0, 16);
            }
            let t_burst = Instant::now();
            let sub = queue.submit(Some(enc.finish()));
            let map = if opts.error_check {
                Some(PendingMap::start(stats_cpu))
            } else {
                None
            };
            // ~180 FP32/SFU/INT ops per thread per iteration x 524,288 threads
            let flops = (HEAVY_WORKGROUPS as u64) * (WORKGROUP_SIZE as u64) * (iters as u64) * 180;
            let flight = FlightHeavy {
                si,
                pass_id,
                flops,
                map,
                sub,
            };

            if lvl < 100 {
                if !finish_heavy(flight) {
                    break;
                }
                let dt_ms = t_burst.elapsed().as_secs_f64() * 1000.0;
                base_k = adjust_iters_damped(base_k, dt_ms, 16.0, 4, 4000);
                if shared.should_continue() {
                    let sleep_ms = dt_ms * ((100 - lvl) as f64 / lvl.max(10) as f64);
                    sleep_interruptible(
                        Duration::from_secs_f64((sleep_ms / 1000.0).min(0.25)),
                        || shared.should_continue(),
                    );
                }
                last_complete = Instant::now();
            } else {
                in_flight.push_back(flight);
                if in_flight.len() >= slots.len() {
                    if let Some(oldest) = in_flight.pop_front() {
                        if !finish_heavy(oldest) {
                            break;
                        }
                        let dt_ms = last_complete.elapsed().as_secs_f64() * 1000.0;
                        last_complete = Instant::now();
                        base_k = adjust_iters_damped(base_k, dt_ms, TARGET_DISPATCH_MS, 4, 4000);
                    }
                } else {
                    last_complete = Instant::now();
                }
            }
            pass_id = pass_id.wrapping_add(1);
        }
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }

    // Profile 3: Extreme 3D (triple-buffered 4xMRT render pass + periodic FMA verification)
    fn run_extreme_3d(
        &self,
        shared: &RunShared,
        events: &dyn StressEvents,
        opts: GpuRunOptions,
        add_flops: impl Fn(u64),
    ) {
        let (device, queue) = match self.create_device() {
            Ok(dq) => dq,
            Err(e) => {
                shared.record_error(events, 0, None, WorkerErrorKind::GpuDevice, e);
                return;
            }
        };

        // Universal render target sizing: 2048x2048 on >=2 GiB adapters, 1024x1024 on tiny iGPUs.
        let render_dim = if device.limits().max_texture_dimension_2d >= RENDER_DIM
            && self.fallback_vram_bytes >= (2 << 30)
        {
            RENDER_DIM
        } else {
            1024
        };

        // 1024x1024 RGBA8 noise texture (4 MiB -> 100% L2-resident on all modern GPUs: drives TMUs
        // & L1/L2 texture caches at full bandwidth with zero DRAM-miss warp stalls!).
        let mut tex_data = vec![0u8; (TEX_DIM * TEX_DIM * 4) as usize];
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for chunk in tex_data.as_chunks_mut::<4>().0 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            chunk[0] = (s & 0xFF) as u8;
            chunk[1] = ((s >> 8) & 0xFF) as u8;
            chunk[2] = ((s >> 16) & 0xFF) as u8;
            chunk[3] = ((s >> 24) & 0xFF) as u8;
        }
        let noise_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("extreme-noise"),
            size: wgpu::Extent3d {
                width: TEX_DIM,
                height: TEX_DIM,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &noise_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &tex_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEX_DIM * 4),
                rows_per_image: Some(TEX_DIM),
            },
            wgpu::Extent3d {
                width: TEX_DIM,
                height: TEX_DIM,
                depth_or_array_layers: 1,
            },
        );
        let noise_view = noise_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("extreme-samp"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Dual 4xMRT Rgba16Float render target sets (A/B) for deterministic dual-frame bit-exact artifact compare.
        let mk_rt = || {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("extreme-mrt"),
                size: wgpu::Extent3d {
                    width: render_dim,
                    height: render_dim,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let v = t.create_view(&wgpu::TextureViewDescriptor::default());
            (t, v)
        };
        let rts_a = [mk_rt(), mk_rt(), mk_rt(), mk_rt()];
        let rts_b = [mk_rt(), mk_rt(), mk_rt(), mk_rt()];

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("testoc-3d"),
            source: wgpu::ShaderSource::Wgsl(RENDER_3D_WGSL.into()),
        });
        let blend_state = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let target_state = Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(blend_state),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let render_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("testoc-3d-pipe"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    target_state.clone(),
                    target_state.clone(),
                    target_state.clone(),
                    target_state,
                ],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Analytical FP32 FMA verification compute pipeline + dual-frame 4xMRT pixel comparator.
        let verify_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("testoc-3d-verify"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_WGSL.into()),
        });
        let verify_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("testoc-3d-verify-pipe"),
            layout: None,
            module: &verify_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let cmp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("testoc-3d-mrt-cmp"),
            source: wgpu::ShaderSource::Wgsl(MRT_COMPARE_WGSL.into()),
        });
        let cmp_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("testoc-3d-mrt-cmp-pipe"),
            layout: None,
            module: &cmp_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let digest_bytes = (COMPUTE_WORKGROUPS as u64) * 4;
        let rbgl = render_pipe.get_bind_group_layout(0);
        let cbgl = verify_pipe.get_bind_group_layout(0);
        let cmp_bgl = cmp_pipe.get_bind_group_layout(0);
        let mk_slot = || {
            let r_uni = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("3d-params"),
                size: std::mem::size_of::<RenderParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let r_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("3d-bg"),
                layout: &rbgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: r_uni.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&noise_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            let c_uni = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("3d-ver-params"),
                size: std::mem::size_of::<ComputeParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let digest_gpu = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("3d-digest"),
                size: digest_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let digest_cpu = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("3d-digest-rb"),
                size: digest_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let stats_gpu = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("3d-stats"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let stats_cpu = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("3d-stats-rb"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let c_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("3d-ver-bg"),
                layout: &cbgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: c_uni.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: digest_gpu.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: stats_gpu.as_entire_binding(),
                    },
                ],
            });
            let cmp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("3d-mrt-cmp-bg"),
                layout: &cmp_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&rts_a[0].1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&rts_a[1].1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&rts_a[2].1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&rts_a[3].1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&rts_b[0].1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&rts_b[1].1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&rts_b[2].1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&rts_b[3].1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: stats_gpu.as_entire_binding(),
                    },
                ],
            });
            (
                r_uni, r_bg, c_uni, digest_gpu, digest_cpu, stats_gpu, stats_cpu, c_bg, cmp_bg,
            )
        };
        // 4-slot command ring: 3 frames are ALWAYS queued ahead on the GPU so utilization never dips.
        let slots = [mk_slot(), mk_slot(), mk_slot(), mk_slot()];

        struct Flight3d {
            si: usize,
            pass_id: u32,
            rem: u32,
            flops: u64,
            maps: Option<(PendingMap, PendingMap)>,
            sub: wgpu::SubmissionIndex,
        }

        let mut frag_iters: u32 = 8;
        let mut pair_time: f32 = 0.0;
        let mut pair_iters: u32 = frag_iters;
        let mut pass_id: u32 = 0;
        let mut in_flight: VecDeque<Flight3d> = VecDeque::with_capacity(4);
        let t_start = Instant::now();
        let mut last_complete = Instant::now();

        let finish_flight = |f: Flight3d| -> bool {
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: Some(f.sub),
                timeout: None,
            });
            if let Some((p_stats, p_digest)) = f.maps {
                let expected = expected_compute_digests(f.pass_id, f.rem);
                let (Some(stats_words), Some(digest_words)) = (
                    p_stats.finish(&device, &slots[f.si].6),
                    p_digest.finish(&device, &slots[f.si].4),
                ) else {
                    shared.record_error(
                        events,
                        0,
                        None,
                        WorkerErrorKind::GpuDevice,
                        "Extreme 3D readback failed".into(),
                    );
                    return false;
                };
                let lane_bad = stats_words.first().copied().unwrap_or(0) as u64;
                let pixel_bad = stats_words.get(1).copied().unwrap_or(0) as u64;
                let digest_bad = digest_words
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a != b)
                    .count() as u64;
                let errs = lane_bad.max(digest_bad) + pixel_bad;
                if errs > 0 {
                    if errs > 1 {
                        shared.add_extra_gpu_errors(errs - 1);
                    }
                    shared.record_error(
                        events,
                        0,
                        None,
                        WorkerErrorKind::GpuMismatch,
                        format!(
                            "GPU Extreme 3D artifact/error (pass {}: pixel_diff={pixel_bad}, fma_lane={lane_bad}, digest={digest_bad})",
                            f.pass_id
                        ),
                    );
                }
            }
            add_flops(f.flops);
            true
        };

        while shared.should_continue() {
            let lvl = Self::current_level(shared, opts);
            if lvl < 100 {
                while let Some(f) = in_flight.pop_front() {
                    if !finish_flight(f) {
                        let _ = device.poll(wgpu::PollType::wait_indefinitely());
                        return;
                    }
                }
            }

            let si = (pass_id as usize) % slots.len();
            let rem = (pass_id % 5) + 1;
            let is_second_of_pair = (pass_id & 1) == 1;
            if !is_second_of_pair {
                pair_time = t_start.elapsed().as_secs_f32();
                pair_iters = frag_iters;
            }
            // Verify on the 2nd frame of every 4th pair (pass 1, 9, 17, ...): compares all 4xMRT
            // pixels between Frame A and Frame B bit-for-bit AND runs the analytical FMA verifier.
            let do_verify = opts.error_check && (pass_id & 7) == 1;
            let verify_iters = 60 * 6 + rem;
            let rp = RenderParams {
                time_s: pair_time,
                iters: pair_iters,
                pass_id: pass_id >> 1,
                _pad: 0,
            };
            let (r_uni, r_bg, c_uni, digest_gpu, digest_cpu, stats_gpu, stats_cpu, c_bg, cmp_bg) =
                &slots[si];
            queue.write_buffer(r_uni, 0, as_bytes(&rp));
            if do_verify {
                let cp = ComputeParams {
                    m00: 2.0,
                    m01: -3.0,
                    m10: 1.0,
                    m11: -1.0,
                    zero: 0.0,
                    iters: verify_iters,
                    pass_id,
                    rem,
                };
                queue.write_buffer(c_uni, 0, as_bytes(&cp));
            }

            let target_rts = if is_second_of_pair { &rts_b } else { &rts_a };
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            if do_verify {
                enc.clear_buffer(digest_gpu, 0, None);
                enc.clear_buffer(stats_gpu, 0, None);
            }
            {
                fn att(v: &wgpu::TextureView) -> Option<wgpu::RenderPassColorAttachment<'_>> {
                    Some(wgpu::RenderPassColorAttachment {
                        view: v,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })
                }
                let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("extreme-3d-pass"),
                    color_attachments: &[
                        att(&target_rts[0].1),
                        att(&target_rts[1].1),
                        att(&target_rts[2].1),
                        att(&target_rts[3].1),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rpass.set_pipeline(&render_pipe);
                rpass.set_bind_group(0, r_bg, &[]);
                for _ in 0..OVERDRAW_DRAWS {
                    rpass.draw(0..3, 0..1);
                }
            }
            if do_verify {
                let cmp_wgs = render_dim.div_ceil(16);
                {
                    let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&cmp_pipe);
                    cpass.set_bind_group(0, cmp_bg, &[]);
                    cpass.dispatch_workgroups(cmp_wgs, cmp_wgs, 1);
                    cpass.set_pipeline(&verify_pipe);
                    cpass.set_bind_group(0, c_bg, &[]);
                    cpass.dispatch_workgroups(COMPUTE_WORKGROUPS, 1, 1);
                }
                enc.copy_buffer_to_buffer(digest_gpu, 0, digest_cpu, 0, digest_bytes);
                enc.copy_buffer_to_buffer(stats_gpu, 0, stats_cpu, 0, 16);
            }
            let t_burst = Instant::now();
            let sub = queue.submit(Some(enc.finish()));
            // Register async map IMMEDIATELY after submit so finish_flight never drains newer frames.
            let maps = if do_verify {
                Some((PendingMap::start(stats_cpu), PendingMap::start(digest_cpu)))
            } else {
                None
            };

            // ~460 FP32/INT/SFU ops/pixel/iter x OVERDRAW_DRAWS x (render_dim^2) pixels
            let render_flops = (render_dim as u64)
                * (render_dim as u64)
                * (OVERDRAW_DRAWS as u64)
                * (pair_iters as u64)
                * 460;
            let verify_flops = if do_verify {
                (COMPUTE_THREADS as u64) * (verify_iters as u64) * 256
            } else {
                0
            };
            let flight = Flight3d {
                si,
                pass_id,
                rem,
                flops: render_flops + verify_flops,
                maps,
                sub,
            };

            if lvl < 100 {
                if !finish_flight(flight) {
                    break;
                }
                let dt_ms = t_burst.elapsed().as_secs_f64() * 1000.0;
                if !do_verify {
                    frag_iters = adjust_iters_damped(frag_iters, dt_ms, 16.0, 2, 512);
                }
                if shared.should_continue() {
                    let sleep_ms = dt_ms * ((100 - lvl) as f64 / lvl.max(10) as f64);
                    sleep_interruptible(
                        Duration::from_secs_f64((sleep_ms / 1000.0).min(0.25)),
                        || shared.should_continue(),
                    );
                }
                last_complete = Instant::now();
            } else {
                in_flight.push_back(flight);
                if in_flight.len() >= slots.len() {
                    if let Some(oldest) = in_flight.pop_front() {
                        let was_verify = oldest.maps.is_some();
                        if !finish_flight(oldest) {
                            break;
                        }
                        let dt_ms = last_complete.elapsed().as_secs_f64() * 1000.0;
                        last_complete = Instant::now();
                        // Calibrate only on pure render frames so verify-frame overhead never oscillates frag_iters.
                        if !was_verify {
                            frag_iters =
                                adjust_iters_damped(frag_iters, dt_ms, TARGET_DISPATCH_MS, 2, 512);
                        }
                    }
                } else {
                    last_complete = Instant::now();
                }
            }
            pass_id = pass_id.wrapping_add(1);
        }
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }

    // Chunked VRAM fill-then-verify pass with selectable deterministic patterns
    pub fn run_vram(
        &self,
        shared: &RunShared,
        events: &dyn StressEvents,
        target_bytes: u64,
        chunk_bytes: u64,
        pattern_pref: MemoryPattern,
        add_bytes: impl Fn(u64),
    ) {
        let (device, queue) = match self.create_device() {
            Ok(dq) => dq,
            Err(e) => {
                shared.record_error(events, 0, None, WorkerErrorKind::GpuDevice, e);
                return;
            }
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("testoc-vram"),
            source: wgpu::ShaderSource::Wgsl(VRAM_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vram-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vram-pipe-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let fill_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vram-fill"),
            layout: Some(&pipe_layout),
            module: &shader,
            entry_point: Some("fill"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let mirror_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vram-mirror128"),
            layout: Some(&pipe_layout),
            module: &shader,
            entry_point: Some("mirror128"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let verify_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vram-verify"),
            layout: Some(&pipe_layout),
            module: &shader,
            entry_point: Some("verify"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vram-params"),
            size: std::mem::size_of::<VramParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let errs_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vram-errs"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let errs_cpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vram-errs-readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        struct VramGuard<'a> {
            shared: &'a RunShared,
            bufs: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
            allocated: u64,
        }
        impl Drop for VramGuard<'_> {
            fn drop(&mut self) {
                self.bufs.clear();
                self.shared.vram_allocated.store(0, Ordering::Relaxed);
            }
        }

        let mut guard = VramGuard {
            shared,
            bufs: Vec::new(),
            allocated: 0,
        };
        let num_chunks = (target_bytes / chunk_bytes).max(1);
        for ci in 0..num_chunks {
            if !shared.should_continue() {
                return;
            }
            let oom_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let val_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vram-chunk"),
                size: chunk_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let v_err = pollster::block_on(val_scope.pop());
            let o_err = pollster::block_on(oom_scope.pop());
            if o_err.is_some() || v_err.is_some() {
                shared.record_error(
                    events,
                    0,
                    None,
                    WorkerErrorKind::AllocFailed,
                    format!(
                        "VRAM allocation capped at {} MiB (chunk {ci})",
                        guard.allocated >> 20
                    ),
                );
                break;
            }
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vram-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: errs_gpu.as_entire_binding(),
                    },
                ],
            });
            guard.allocated += chunk_bytes;
            shared
                .vram_allocated
                .store(guard.allocated, Ordering::Relaxed);
            guard.bufs.push((buf, bg));
        }

        if guard.bufs.is_empty() {
            return;
        }

        let word_count = (chunk_bytes / 4) as u32;
        let dispatch_wgs = (word_count / WORKGROUP_SIZE).clamp(1, 4096);
        let init_errs: [u32; 4] = [0, u32::MAX, 0, 0];
        let init_errs_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(init_errs.as_ptr() as *const u8, 16) };
        let mut pass_id: u32 = 0;

        'outer: loop {
            let pattern = match pattern_pref {
                MemoryPattern::All => pass_id % 8,
                MemoryPattern::WalkingOnes => 0,
                MemoryPattern::AddressInAddress => 1,
                MemoryPattern::Random => 2,
                MemoryPattern::MovingInversions => 3,
                MemoryPattern::BlockStrided => 4,
            };
            for (ci, (_, bg)) in guard.bufs.iter().enumerate() {
                if !shared.should_continue() {
                    break 'outer;
                }
                let p = VramParams {
                    word_count,
                    chunk_id: ci as u32,
                    pass_id,
                    pattern,
                };
                queue.write_buffer(&uniform, 0, as_bytes(&p));
                let mut enc =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(&fill_pipe);
                    cp.set_bind_group(0, bg, &[]);
                    cp.dispatch_workgroups(dispatch_wgs, 1, 1);
                }
                queue.submit(Some(enc.finish()));
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                add_bytes(chunk_bytes);
            }
            // Pattern 7 (Absolut MirrorMove128): bidirectionally swap 128B stripes across every chunk before verify.
            if pattern == 7 {
                for (ci, (_, bg)) in guard.bufs.iter().enumerate() {
                    if !shared.should_continue() {
                        break 'outer;
                    }
                    let p = VramParams {
                        word_count,
                        chunk_id: ci as u32,
                        pass_id,
                        pattern,
                    };
                    queue.write_buffer(&uniform, 0, as_bytes(&p));
                    let mut enc = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                        cp.set_pipeline(&mirror_pipe);
                        cp.set_bind_group(0, bg, &[]);
                        cp.dispatch_workgroups(dispatch_wgs, 1, 1);
                    }
                    queue.submit(Some(enc.finish()));
                    let _ = device.poll(wgpu::PollType::wait_indefinitely());
                    add_bytes(chunk_bytes * 2);
                }
            }
            for (ci, (_, bg)) in guard.bufs.iter().enumerate() {
                if !shared.should_continue() {
                    break 'outer;
                }
                let p = VramParams {
                    word_count,
                    chunk_id: ci as u32,
                    pass_id,
                    pattern,
                };
                queue.write_buffer(&uniform, 0, as_bytes(&p));
                queue.write_buffer(&errs_gpu, 0, init_errs_bytes);
                let mut enc =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(&verify_pipe);
                    cp.set_bind_group(0, bg, &[]);
                    cp.dispatch_workgroups(dispatch_wgs, 1, 1);
                }
                enc.copy_buffer_to_buffer(&errs_gpu, 0, &errs_cpu, 0, 16);
                let sub = queue.submit(Some(enc.finish()));
                let Some(words) = readback_u32(&device, &errs_cpu, Some(sub)) else {
                    shared.record_error(
                        events,
                        0,
                        None,
                        WorkerErrorKind::GpuDevice,
                        "VRAM readback failed".into(),
                    );
                    break 'outer;
                };
                add_bytes(chunk_bytes);
                let bad = words.first().copied().unwrap_or(0) as u64;
                if bad > 0 {
                    let first = words.get(1).copied().unwrap_or(0);
                    if bad > 1 {
                        shared.add_extra_gpu_errors(bad - 1);
                    }
                    shared.record_error(
                        events,
                        0,
                        None,
                        WorkerErrorKind::VramMismatch,
                        format!("{bad} VRAM word(s) mismatched (chunk {ci}, pattern {pattern}, pass {pass_id}, first word {first})"),
                    );
                }
            }
            pass_id = pass_id.wrapping_add(1);
        }
    }

    /// Execute one deterministic compute pass directly (used by unit tests).
    #[cfg(test)]
    fn single_compute_pass(
        &self,
        m00: f32,
        pass_id: u32,
        iters_k: u32,
    ) -> Result<(u64, u64), String> {
        use wgpu::util::DeviceExt;
        let (device, queue) = self.create_device()?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(COMPUTE_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let rem = (pass_id % 5) + 1;
        let params = ComputeParams {
            m00,
            m01: -3.0,
            m10: 1.0,
            m11: -1.0,
            zero: 0.0,
            iters: iters_k * 6 + rem,
            pass_id,
            rem,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: as_bytes(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let digest_bytes = (COMPUTE_WORKGROUPS as u64) * 4;
        let digest_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &vec![0u8; digest_bytes as usize],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let digest_cpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: digest_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stats_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0u8; 16],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let stats_cpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: digest_gpu.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: stats_gpu.as_entire_binding(),
                },
            ],
        });
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipeline);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(COMPUTE_WORKGROUPS, 1, 1);
        }
        enc.copy_buffer_to_buffer(&digest_gpu, 0, &digest_cpu, 0, digest_bytes);
        enc.copy_buffer_to_buffer(&stats_gpu, 0, &stats_cpu, 0, 16);
        let sub = queue.submit(Some(enc.finish()));
        let (stats, digest) =
            readback_two(&device, &stats_cpu, &digest_cpu, Some(sub)).ok_or("readback failed")?;
        let exp = expected_compute_digests(pass_id, rem);
        let digest_bad = digest.iter().zip(&exp).filter(|(a, b)| a != b).count() as u64;
        Ok((stats[0] as u64, digest_bad))
    }
}

/// Non-draining async staging-buffer readback handle.
/// Register `PendingMap::start(&buf)` IMMEDIATELY after `queue.submit(sub)` (before later frames
/// submit) so `finish()` waits ONLY for `sub` without ever draining newer in-flight GPU work.
struct PendingMap {
    rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl PendingMap {
    fn start(buf: &wgpu::Buffer) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        Self { rx }
    }

    fn finish(self, device: &wgpu::Device, buf: &wgpu::Buffer) -> Option<Vec<u32>> {
        // The caller already waited for this slot's SubmissionIndex; poll non-blockingly if a driver
        // needs one extra callback pump, never calling wait_indefinitely() which drains newer frames.
        for _ in 0..200 {
            match self.rx.try_recv() {
                Ok(res) => {
                    res.ok()?;
                    let slice = buf.slice(..);
                    let data = slice.get_mapped_range().ok()?;
                    let words: Vec<u32> = data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes(*c))
                        .collect();
                    drop(data);
                    buf.unmap();
                    return Some(words);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    let _ = device.poll(wgpu::PollType::Poll);
                    std::thread::yield_now();
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return None,
            }
        }
        // Last-resort fallback if a quirky driver delayed callback delivery past poll cycles.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        self.rx.recv().ok()?.ok()?;
        let slice = buf.slice(..);
        let data = slice.get_mapped_range().ok()?;
        let words: Vec<u32> = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        drop(data);
        buf.unmap();
        Some(words)
    }
}

/// Damped iteration-count controller: nudges frame duration smoothly toward `target_ms`
/// without overshooting or sawtoothing when an occasional frame varies in duration.
fn adjust_iters_damped(cur: u32, dt_ms: f64, target_ms: f64, min_v: u32, max_v: u32) -> u32 {
    if !(dt_ms.is_finite() && dt_ms > 1.0 && dt_ms < 500.0) {
        return cur;
    }
    let raw = (target_ms / dt_ms).clamp(0.70, 1.35);
    let damped = 1.0 + (raw - 1.0) * 0.30;
    ((cur as f64) * damped)
        .round()
        .clamp(min_v as f64, max_v as f64) as u32
}

fn readback_u32(
    device: &wgpu::Device,
    buf: &wgpu::Buffer,
    wait_sub: Option<wgpu::SubmissionIndex>,
) -> Option<Vec<u32>> {
    let pending = PendingMap::start(buf);
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: wait_sub,
        timeout: None,
    });
    pending.finish(device, buf)
}

#[cfg(test)]
fn readback_two(
    device: &wgpu::Device,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    wait_sub: Option<wgpu::SubmissionIndex>,
) -> Option<(Vec<u32>, Vec<u32>)> {
    let pa = PendingMap::start(a);
    let pb = PendingMap::start(b);
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: wait_sub,
        timeout: None,
    });
    Some((pa.finish(device, a)?, pb.finish(device, b)?))
}

fn sleep_interruptible(total: Duration, keep_going: impl Fn() -> bool) {
    let start = Instant::now();
    while start.elapsed() < total && keep_going() {
        let rest = total.saturating_sub(start.elapsed());
        std::thread::sleep(rest.min(Duration::from_millis(5)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    #[test]
    fn vram_sizing_universal_across_gpu_tiers() {
        // 12 GiB GPU at 98%: lean 64 MiB DWM headroom + 64 MiB chunks -> allocates >10.5 GiB
        let (t, c) = compute_vram_target_bytes(12 * GB, Some(10_800 << 20), 98, 1 << 30).unwrap();
        assert_eq!(c, 64 << 20);
        assert!(t <= (10_800 << 20) - (64 << 20));
        assert!(t >= 10_600 << 20);

        // 12 GiB GPU at 100%: fills all free VRAM minus 64 MiB headroom
        let (t100, _) =
            compute_vram_target_bytes(12 * GB, Some(11_200 << 20), 100, 1 << 30).unwrap();
        assert!(t100 >= 11_000 << 20);

        // 4 GiB GPU at 85%
        let (t4, _) = compute_vram_target_bytes(4 * GB, Some(3_400 << 20), 85, 128 << 20).unwrap();
        assert!(t4 <= (3_400 << 20) - (128 << 20));
        assert!(t4 >= 3 * GB);

        // 24 GiB GPU at 95% (above old 90% cap)
        let (t24, _) = compute_vram_target_bytes(24 * GB, Some(23 * GB), 95, 1 << 30).unwrap();
        assert!(t24 > 22 * GB);
        assert!(t24 <= 24 * GB / 100 * 95);

        // Fallback (no NVML free_vram) on 8 GiB GPU at 80%
        let (tf, _) = compute_vram_target_bytes(8 * GB, None, 80, 128 << 20).unwrap();
        assert!((5 * GB..=8 * GB / 100 * 80).contains(&tf));

        // Out of bounds
        assert!(compute_vram_target_bytes(12 * GB, Some(10 * GB), 101, 1 << 30).is_err());
        assert!(compute_vram_target_bytes(12 * GB, Some(10 * GB), 5, 1 << 30).is_err());
    }

    #[test]
    fn fast_compute_digests_match_bruteforce_exactly() {
        for pass_id in [0u32, 1, 7, 42] {
            for rem in 1..=5u32 {
                let fast = expected_compute_digests(pass_id, rem);
                assert_eq!(fast.len(), COMPUTE_WORKGROUPS as usize);
                for wg in [0u32, 1, 17, COMPUTE_WORKGROUPS - 1] {
                    let mut acc: i32 = 0;
                    let base_gid = wg * WORKGROUP_SIZE;
                    for tid in 0..WORKGROUP_SIZE {
                        let gid = base_gid + tid;
                        for pair in 0..8u32 {
                            let bx = gid.wrapping_add(pair * 7).wrapping_add(pass_id * 3);
                            let by = gid
                                .wrapping_mul(3)
                                .wrapping_add(pair * 11)
                                .wrapping_add(pass_id * 5);
                            for lane in 0..4u32 {
                                let ox = [0u32, 3, 7, 11][lane as usize];
                                let oy = [0u32, 5, 9, 13][lane as usize];
                                let mut x = ((bx.wrapping_add(ox) & 15) as i32) - 8;
                                let mut y = ((by.wrapping_add(oy) & 15) as i32) - 8;
                                for _ in 0..rem {
                                    let nx = 2 * x - 3 * y;
                                    let ny = x - y;
                                    x = nx;
                                    y = ny;
                                }
                                acc = acc.wrapping_add(x).wrapping_add(y);
                            }
                        }
                    }
                    assert_eq!(
                        fast[wg as usize], acc as u32,
                        "pass={pass_id} rem={rem} wg={wg}"
                    );
                }
            }
        }
    }

    #[test]
    fn vulkan_compute_verifies_and_detects_fault() {
        let Some(gpu) = GpuStress::probe() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        for pass in 0..3 {
            let (lane_bad, digest_bad) = gpu.single_compute_pass(2.0, pass, 10).unwrap();
            assert_eq!((lane_bad, digest_bad), (0, 0), "pass {pass}");
        }
        let (lane_bad, digest_bad) = gpu.single_compute_pass(3.0, 0, 1).unwrap();
        assert!(
            lane_bad > 0 && digest_bad > 0,
            "fault not caught: lane={lane_bad} digest={digest_bad}"
        );
    }
}
