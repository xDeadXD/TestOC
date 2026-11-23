//! Memory stress engine: Absolut 32-step DRAM/IMC suite + classic pattern runner.
//!
//! Every test is 100% deterministic and bit-exact (zero subjectivity, zero false positives).
//! Absolut uses pure-AVX2 zero-scalar-overhead pattern generators (8-lane 32-bit vector LCG
//! `0x77777777 / 0x33333333`, `0xFFFFFFFF / 0x55555555`, and BL8 64-pin DQ max-toggle rails)
//! driven through full-chunk bijective 64-byte / 128-byte coprime row-conflict in-place RMW
//! scatter (100% L1/L2/L3 miss + 100% DRAM row-buffer miss across channel, rank, bank-group,
//! bank, row, and column bits simultaneously), rotating cached Write-Back RFO dirty-eviction
//! storms with Non-Temporal WC-bypass (`_mm256_stream_si256`), cross-chunk simultaneous cold-DRAM
//! Read<->Write turnarounds (`tRDWR`/`tWRRD`/`tWTR`/`ODT`), coprime-scattered cross-allocation
//! `MirrorMove128` / `MirrorMove 4K`, and `_mm_clflush` aggressor rowhammer sweeps.

use std::sync::atomic::{compiler_fence, Ordering};

/// Max initial chunk size (64 MiB); `worker_chunk_bytes` ensures >= 2 chunks per worker so
/// cross-chunk simultaneous R<->W turnaround and opposing-end MirrorMove always stay cold in DRAM.
pub const CHUNK_BYTES: usize = 64 << 20;
/// Top-up chunk granularity used when reclaiming large freed blocks (e.g. closed apps).
pub const TOPUP_CHUNK_BYTES: usize = 16 << 20;
/// Fine-grain tail top-up chunk used to hold constant 99%+ physical RAM without dipping.
pub const FINE_TOPUP_CHUNK_BYTES: usize = 4 << 20;
/// Write/verify interrupt granularity: 1 MiB, stop flag checked after every step.
pub const STEP_WORDS: usize = (1 << 20) / 8;
/// One 4 KiB OS page in 64-bit words (64 cache lines).
const PAGE_WORDS: usize = 512;
/// One 64-byte CPU cache line / DDR BL8 burst in 64-bit words.
const CACHELINE_WORDS: usize = 8;
/// 128-byte dual-cache-line stripe in 64-bit words (for MirrorMove128 / dual-channel interleave).
const STRIPE128_WORDS: usize = 16;
/// Coprime stride divisors rotated across passes so every pass hits a fresh DRAM bank/row order.
const SCATTER_DIVS: &[usize] = &[13, 17, 19, 23, 29, 31, 37, 41];
pub const MIN_TARGET_BYTES: u64 = 128 << 20;
/// Standard OS safety reserve (128 MiB); drops to 64 MiB at >=98% and 24 MiB at 100% (OOM-safe).
pub const FREE_FLOOR_BYTES: u64 = 128 << 20;
pub const EXTREME_FLOOR_BYTES: u64 = 64 << 20;
pub const MAX_FLOOR_BYTES: u64 = 24 << 20;

/// Prime stride in 64-bit words (~32 KiB + 24 B): coprime to power-of-2 slice lengths, defeating
/// L1/L2 stride prefetchers and crossing DRAM banks.
const STRIDE_WORDS: usize = 4099;

/// Per-worker chunk sizing: caps at 64 MiB while guaranteeing at least 2 disjoint chunks whenever
/// a worker allocates >= 8 MiB (so cross-chunk R<->W and MirrorMove always span distinct regions).
pub fn worker_chunk_bytes(worker_bytes: u64) -> usize {
    let b = worker_bytes as usize;
    if b >= (8 << 20) {
        (b / 2).clamp(4 << 20, CHUNK_BYTES)
    } else {
        b.max(4096)
    }
}

/// Authentic Absolut 32-step DRAM cycle sequence:
/// Starts immediately on T1 (`SimpleTest M2` LCG `0x77777777 / 0x33333333`) followed by the full
/// interleaved MirrorMove / SimpleTest / T15 sequence and T0 RefreshStable retention pass.
pub const ABSOLUT_SEQUENCE: &[u8] = &[
    1, 4, 6, 15, 3, 2, 7, 15, 5, 2, 8, 15, 4, 2, 9, 15, 3, 2, 10, 15, 5, 2, 11, 15, 4, 2, 12, 15,
    5, 14, 15, 0,
];

/// Inner pass counts per Absolut sub-test, weighted to match authentic `Time (%)` proportions so
/// one full 32-step cycle provides sustained multi-pass soak scaling linearly with installed RAM.
pub fn absolut_step_passes(test_id: u8) -> u32 {
    match test_id {
        1 => 36,  // SimpleTest M2 LCG 0x77777777/0x33333333 (Time%=240)
        2 => 10,  // SimpleTest M1 Variable Polarity (Time%=8 x 7)
        3 => 20,  // MirrorMove128 Striped 2x Cross-Chunk NT (Time%=20 x 2)
        4 => 24,  // MirrorMove 4K Page Param-4 Cross-Chunk NT (Time%=20 x 3)
        5 => 18,  // MirrorMove128 Contiguous Cross-Chunk NT 2x (Time%=20 x 3)
        6 => 40,  // SimpleTest M0 4 MiB Moving Inv R<->W (Time%=240)
        7 => 36,  // SimpleTest M0 8 MiB DQ Single-Bit Aggressor R<->W (Time%=240)
        8 => 36,  // SimpleTest M0 16 MiB VDDQ Ground-Bounce R<->W (Time%=240)
        9 => 36,  // SimpleTest M0 32 MiB Full-Entropy PRNG R<->W (Time%=240)
        10 => 32, // SimpleTest M0 64 MiB Modulo20 Bank-Conflict R<->W (Time%=240)
        11 => 28, // SimpleTest M0 128 MiB Cross-Chunk NT BlockMove (Time%=240)
        12 => 36, // SimpleTest M0 256 MiB Address-Coupled CMD/ADDR R<->W (Time%=240)
        13 => 32, // SimpleTest M0 512 MiB Walking 64-Bit R<->W (Time%=240)
        14 => 24, // SimpleTest M1 Rowhammer Bit-Fade Retention (Time%=100)
        15 => 20, // SimpleTest M2 0xFFFFFFFF/0x55555555 R<->W (Time%=100 x 8)
        0 => 12,  // RefreshStable tREFI/tRFC Retention + Rowhammer (Time%=8)
        _ => 20,
    }
}

/// Effective runtime inner pass count: full cycle depth in production, 1 pass in unit tests.
pub fn absolut_step_passes_runtime(test_id: u8) -> u32 {
    #[cfg(test)]
    {
        let _ = test_id;
        1
    }
    #[cfg(not(test))]
    {
        absolut_step_passes(test_id)
    }
}

/// Human-readable label for each Absolut 32-step sub-test ID (0..=15).
pub fn absolut_test_name(test_id: u8) -> &'static str {
    match test_id {
        0 => "RefreshStable (tREFI/tRFC)",
        1 => "SimpleTest M2 (LCG 0x7777/0x3333)",
        2 => "SimpleTest M1 (Variable Polarity)",
        3 => "MirrorMove128 (Striped 2x NT)",
        4 => "MirrorMove (4K Page NT)",
        5 => "MirrorMove128 (Contiguous NT)",
        6 => "SimpleTest M0 (4M Moving Inv R/W)",
        7 => "SimpleTest M0 (8M DQ Aggressor R/W)",
        8 => "SimpleTest M0 (16M VDDQ Bounce R/W)",
        9 => "SimpleTest M0 (32M PRNG R/W)",
        10 => "SimpleTest M0 (64M Modulo20 R/W)",
        11 => "SimpleTest M0 (128M BlockMove NT)",
        12 => "SimpleTest M0 (256M CMD/ADDR Walk)",
        13 => "SimpleTest M0 (512M Walking 64-Bit)",
        14 => "Rowhammer Bit-Fade (clflush)",
        15 => "SimpleTest M2 (0xFFFF/0x5555 R/W)",
        _ => "Absolut Subtest",
    }
}

/// First mismatching word details reported by bit-exact DRAM verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemMismatch {
    pub chunk_idx: usize,
    pub word_idx: usize,
    pub expected: u64,
    pub actual: u64,
}

/// Aggregated verification result for one Absolut sub-test step.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepResult {
    pub bad_words: u64,
    pub first: Option<MemMismatch>,
    pub aborted: bool,
}

impl StepResult {
    fn record(
        &mut self,
        chunk_idx: usize,
        bad: u64,
        first: Option<(usize, u64, u64)>,
        on_error: &impl Fn(u64, MemMismatch),
    ) {
        if bad == 0 {
            return;
        }
        self.bad_words += bad;
        if let Some((word_idx, expected, actual)) = first {
            let mm = MemMismatch {
                chunk_idx,
                word_idx,
                expected,
                actual,
            };
            if self.first.is_none() {
                self.first = Some(mm);
            }
            on_error(bad, mm);
        }
    }
}

/// OS safety reserve bytes for the requested `(target_pct, max_pct)` pair.
pub fn reserve_floor_bytes(target_pct: u8, max_pct: u8) -> u64 {
    let eff = target_pct.min(max_pct);
    if eff >= 100 {
        MAX_FLOOR_BYTES
    } else if eff >= 98 {
        EXTREME_FLOOR_BYTES
    } else {
        FREE_FLOOR_BYTES
    }
}

/// How many bytes the memory test should initially allocate.
///
/// `min(target_pct x total, available - reserve, max_pct x total - used)`; errors if < 128 MiB.
pub fn compute_target_bytes(
    total: u64,
    available: u64,
    target_pct: u8,
    max_pct: u8,
) -> Result<u64, String> {
    if !(1..=100).contains(&target_pct) {
        return Err(format!("target {target_pct}% out of range 1-100"));
    }
    let used = total.saturating_sub(available);
    let want = if target_pct >= 100 {
        total
    } else {
        total / 100 * target_pct as u64
    };
    let floor = reserve_floor_bytes(target_pct, max_pct);
    let floor_cap = available.saturating_sub(floor);
    let pct_cap = if max_pct >= 100 {
        available.saturating_sub(floor)
    } else {
        (total / 100 * max_pct as u64).saturating_sub(used)
    };
    let t = want.min(floor_cap).min(pct_cap);
    if t < MIN_TARGET_BYTES {
        return Err(format!(
            "not enough free memory: would allocate {} MiB (need >= 128 MiB with {} MiB kept free and <= {max_pct}% total usage)",
            t >> 20,
            floor >> 20
        ));
    }
    Ok(t)
}

/// Compute additional bytes to allocate during dynamic top-up after Windows trims standby caches.
pub fn compute_topup_bytes(total: u64, available: u64, target_pct: u8, max_pct: u8) -> u64 {
    let eff = target_pct.min(max_pct);
    let floor = reserve_floor_bytes(target_pct, max_pct);
    let avail_headroom = available.saturating_sub(floor);
    if eff >= 100 {
        return avail_headroom;
    }
    let used = total.saturating_sub(available);
    let desired_used = total / 100 * eff as u64;
    desired_used.saturating_sub(used).min(avail_headroom)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pattern {
    MovingInversions,
    BlockStrided,
    WalkingOnes,
    AddressInAddress,
    Random,
}

impl Pattern {
    pub fn for_pass(pass: u64) -> Pattern {
        match pass % 5 {
            0 => Pattern::MovingInversions,
            1 => Pattern::BlockStrided,
            2 => Pattern::WalkingOnes,
            3 => Pattern::AddressInAddress,
            _ => Pattern::Random,
        }
    }
    pub fn select(pref: crate::model::MemoryPattern, pass: u64) -> Pattern {
        match pref {
            crate::model::MemoryPattern::All => Self::for_pass(pass),
            crate::model::MemoryPattern::MovingInversions => Pattern::MovingInversions,
            crate::model::MemoryPattern::BlockStrided => Pattern::BlockStrided,
            crate::model::MemoryPattern::WalkingOnes => Pattern::WalkingOnes,
            crate::model::MemoryPattern::AddressInAddress => Pattern::AddressInAddress,
            crate::model::MemoryPattern::Random => Pattern::Random,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Pattern::MovingInversions => "moving inversions",
            Pattern::BlockStrided => "strided bank-conflict",
            Pattern::WalkingOnes => "walking 1s/0s",
            Pattern::AddressInAddress => "address-in-address",
            Pattern::Random => "random",
        }
    }
}

#[inline(always)]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Exact scalar counterpart of the 8-lane AVX2 32-bit dual-round vector LCG with alternating
/// odd-word BL8 inversion (used for scalar fallback, tail words, and exact mismatch diagnostics).
#[inline(always)]
fn vec_lcg_word(seed32: u32, mul32: u32, add32: u32, idx: usize, invert: bool) -> u64 {
    let i0 = (idx as u32).wrapping_mul(2).wrapping_add(seed32);
    let i1 = i0.wrapping_add(1);
    let mut a = i0.wrapping_mul(mul32).wrapping_add(add32);
    a = a.wrapping_mul(mul32).wrapping_add(add32);
    let mut b = i1.wrapping_mul(mul32).wrapping_add(add32);
    b = b.wrapping_mul(mul32).wrapping_add(add32);
    let w = (a as u64) | ((b as u64) << 32);
    let flipped = if (idx & 1) == 0 { w } else { !w };
    if invert {
        !flipped
    } else {
        flipped
    }
}

/// Deterministic 32-bit per-chunk seed derived from worker base, chunk word offset, and step seed.
#[inline(always)]
fn chunk_seed32(base_word: u64, chunk_start: u64, seed: u64) -> u32 {
    let h = splitmix64(base_word ^ chunk_start.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed);
    (h as u32) ^ ((h >> 32) as u32)
}

/// Stateless expected value for a word in the classic Standard pattern runner.
#[inline(always)]
pub fn expected(p: Pattern, pass: u64, global_word: u64) -> u64 {
    match p {
        Pattern::MovingInversions => {
            let base = match (pass / 5) & 3 {
                0 => 0x5555_5555_5555_5555u64,
                1 => 0xAAAA_AAAA_AAAA_AAAAu64,
                2 => 0xFFFF_FFFF_FFFF_FFFFu64,
                _ => 0x0000_0000_0000_0000u64,
            };
            if (global_word & 1) == 0 {
                base
            } else {
                !base
            }
        }
        Pattern::BlockStrided => {
            let v = splitmix64(global_word ^ pass.wrapping_mul(0xD1B5_4A32_D192_ED03));
            if ((global_word >> 3) & 1) == 0 {
                v
            } else {
                !v
            }
        }
        Pattern::WalkingOnes => {
            let bit = 1u64.rotate_left(((global_word + pass) % 64) as u32);
            if (pass & 1) == 0 {
                bit
            } else {
                !bit
            }
        }
        Pattern::AddressInAddress => global_word ^ pass.wrapping_mul(0x9E37_79B9_7F4A_7C15),
        Pattern::Random => splitmix64(global_word ^ (pass << 40)),
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn coprime_stride(len: usize, desired: usize) -> usize {
    if len <= 1 {
        return 1;
    }
    let mut s = desired % len;
    if s == 0 {
        s = 1;
    }
    while gcd(s, len) != 1 {
        s = (s + 1) % len;
        if s == 0 {
            s = 1;
        }
    }
    s
}

/// Full-chunk coprime scatter stride (`~len / div`, odd) so consecutive 64B/128B accesses jump
/// multi-MiB + odd cache-line offsets, toggling channel/rank/bank-group/bank/row/column bits on
/// every single touch for 100% L3 & DRAM row-buffer misses.
fn pick_scatter_stride(len: usize, div: usize) -> usize {
    if len <= 1 {
        return 1;
    }
    let mut s = (len / div.max(2)).max(1) | 1;
    if s >= len {
        s = 1;
    }
    while gcd(s, len) != 1 {
        s += 2;
        if s >= len {
            return coprime_stride(len, 1);
        }
    }
    s
}

pub fn fill_slice(s: &mut [u64], p: Pattern, pass: u64, base_word: u64) {
    if p == Pattern::BlockStrided && s.len() > 1 {
        let len = s.len();
        let stride = coprime_stride(len, STRIDE_WORDS);
        let mut idx = 0usize;
        for _ in 0..len {
            s[idx] = expected(p, pass, base_word + idx as u64);
            idx += stride;
            if idx >= len {
                idx -= len;
            }
        }
        return;
    }
    for (i, w) in s.iter_mut().enumerate() {
        *w = expected(p, pass, base_word + i as u64);
    }
}

/// Returns (mismatch count, first mismatching index within the slice).
pub fn verify_slice(s: &[u64], p: Pattern, pass: u64, base_word: u64) -> (u64, Option<usize>) {
    let mut bad = 0u64;
    let mut first = None;
    if p == Pattern::BlockStrided && s.len() > 1 {
        let len = s.len();
        let stride = coprime_stride(len, STRIDE_WORDS);
        let mut idx = 0usize;
        for _ in 0..len {
            let v = unsafe { std::ptr::read_volatile(&s[idx]) };
            if v != expected(p, pass, base_word + idx as u64) {
                bad += 1;
                first.get_or_insert(idx);
            }
            idx += stride;
            if idx >= len {
                idx -= len;
            }
        }
        return (bad, first);
    }
    for (i, &w) in s.iter().enumerate() {
        let v = unsafe { std::ptr::read_volatile(&w) };
        if v != expected(p, pass, base_word + i as u64) {
            bad += 1;
            first.get_or_insert(i);
        }
    }
    (bad, first)
}

/// Try to allocate one zero-filled chunk without aborting on OOM.
pub fn try_alloc_chunk(words: usize) -> Option<Vec<u64>> {
    let mut v: Vec<u64> = Vec::new();
    v.try_reserve_exact(words).ok()?;
    v.resize(words, 0); // commits physical pages
    Some(v)
}

// ---------------------------------------------------------------------------
// Low-level AVX2 non-temporal streaming, NTA prefetch & simultaneous R<->W primitives
// ---------------------------------------------------------------------------

#[inline]
fn has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

/// Stream a 2-word repeating pattern directly to DRAM via AVX2 `_mm256_stream_si256` (bypasses L1/L2/L3).
fn nt_fill_pair(s: &mut [u64], even: u64, odd: u64) {
    if has_avx2() && s.len() >= 8 {
        unsafe { nt_fill_pair_avx2(s, even, odd) };
        return;
    }
    for (i, w) in s.iter_mut().enumerate() {
        let v = if (i & 1) == 0 { even } else { odd };
        unsafe { std::ptr::write_volatile(w, v) };
    }
}

#[target_feature(enable = "avx2")]
unsafe fn nt_fill_pair_avx2(s: &mut [u64], even: u64, odd: u64) {
    use std::arch::x86_64::*;
    let len = s.len();
    let ptr = s.as_mut_ptr();
    let mis = (ptr as usize) & 31;
    let prefix = if mis == 0 { 0 } else { (32 - mis) / 8 }.min(len);
    for i in 0..prefix {
        std::ptr::write_volatile(ptr.add(i), if (i & 1) == 0 { even } else { odd });
    }
    let (v0, v1) = if (prefix & 1) == 0 {
        (even, odd)
    } else {
        (odd, even)
    };
    let vec = _mm256_set_epi64x(v1 as i64, v0 as i64, v1 as i64, v0 as i64);
    let mut i = prefix;
    let end4 = prefix + ((len - prefix) & !3);
    while i < end4 {
        _mm256_stream_si256(ptr.add(i) as *mut __m256i, vec);
        i += 4;
    }
    while i < len {
        std::ptr::write_volatile(ptr.add(i), if (i & 1) == 0 { even } else { odd });
        i += 1;
    }
    _mm_sfence();
}

/// Write computed per-word values via AVX2 either non-temporally (`WRITE_BACK=false`, direct
/// WC-buffer bypass) or cached write-back (`WRITE_BACK=true`, forcing RFO + async L3 dirty writeback
/// pressure on Infinity Fabric / IMC queues across buffers >> L3).
#[inline]
fn fill_gen_nofence<const WRITE_BACK: bool>(s: &mut [u64], base: u64, gen: &impl Fn(u64) -> u64) {
    if has_avx2() && s.len() >= 8 {
        unsafe { fill_gen_avx2_nofence::<WRITE_BACK>(s, base, gen) };
        return;
    }
    for (i, w) in s.iter_mut().enumerate() {
        unsafe { std::ptr::write_volatile(w, gen(base + i as u64)) };
    }
}

#[target_feature(enable = "avx2")]
unsafe fn fill_gen_avx2_nofence<const WRITE_BACK: bool>(
    s: &mut [u64],
    base: u64,
    gen: &impl Fn(u64) -> u64,
) {
    use std::arch::x86_64::*;
    let len = s.len();
    let ptr = s.as_mut_ptr();
    let mis = (ptr as usize) & 31;
    let prefix = if mis == 0 { 0 } else { (32 - mis) / 8 }.min(len);
    for i in 0..prefix {
        std::ptr::write_volatile(ptr.add(i), gen(base + i as u64));
    }
    let mut i = prefix;
    let end4 = prefix + ((len - prefix) & !3);
    while i < end4 {
        let g0 = gen(base + i as u64);
        let g1 = gen(base + (i + 1) as u64);
        let g2 = gen(base + (i + 2) as u64);
        let g3 = gen(base + (i + 3) as u64);
        let v = _mm256_set_epi64x(g3 as i64, g2 as i64, g1 as i64, g0 as i64);
        if WRITE_BACK {
            _mm256_storeu_si256(ptr.add(i) as *mut __m256i, v);
        } else {
            _mm256_stream_si256(ptr.add(i) as *mut __m256i, v);
        }
        i += 4;
    }
    while i < len {
        std::ptr::write_volatile(ptr.add(i), gen(base + i as u64));
        i += 1;
    }
}

/// Non-temporal convenience wrapper.
#[inline]
fn nt_fill_gen_nofence(s: &mut [u64], base: u64, gen: &impl Fn(u64) -> u64) {
    fill_gen_nofence::<false>(s, base, gen);
}

#[inline(always)]
fn sfence_if_x86() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
}

/// Non-temporal streaming copy `src -> dst` (same length), optionally XOR-inverting each word.
fn nt_copy_slice(src: &[u64], dst: &mut [u64], invert: bool) {
    debug_assert_eq!(src.len(), dst.len());
    let mask = if invert { u64::MAX } else { 0 };
    if has_avx2() && dst.len() >= 8 {
        unsafe { nt_copy_slice_avx2(src, dst, mask) };
        return;
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let v = unsafe { std::ptr::read_volatile(s) } ^ mask;
        unsafe { std::ptr::write_volatile(d, v) };
    }
}

#[target_feature(enable = "avx2")]
unsafe fn nt_copy_slice_avx2(src: &[u64], dst: &mut [u64], mask: u64) {
    use std::arch::x86_64::*;
    let len = dst.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mis = (dp as usize) & 31;
    let prefix = if mis == 0 { 0 } else { (32 - mis) / 8 }.min(len);
    for i in 0..prefix {
        std::ptr::write_volatile(dp.add(i), std::ptr::read_volatile(sp.add(i)) ^ mask);
    }
    let mvec = _mm256_set1_epi64x(mask as i64);
    let mut i = prefix;
    let end4 = prefix + ((len - prefix) & !3);
    while i < end4 {
        if i + 32 < end4 {
            _mm_prefetch::<_MM_HINT_NTA>(sp.add(i + 32) as *const i8);
        }
        let v = _mm256_xor_si256(_mm256_loadu_si256(sp.add(i) as *const __m256i), mvec);
        _mm256_stream_si256(dp.add(i) as *mut __m256i, v);
        i += 4;
    }
    while i < len {
        std::ptr::write_volatile(dp.add(i), std::ptr::read_volatile(sp.add(i)) ^ mask);
        i += 1;
    }
    _mm_sfence();
}

/// Fast 256-bit NTA-prefetched verification against a per-word generator.
fn verify_gen(
    s: &[u64],
    base: u64,
    word_offset: usize,
    gen: &impl Fn(u64) -> u64,
) -> (u64, Option<(usize, u64, u64)>) {
    compiler_fence(Ordering::SeqCst);
    if has_avx2() && s.len() >= 8 {
        return unsafe { verify_gen_avx2(s, base, word_offset, gen) };
    }
    let mut bad = 0u64;
    let mut first = None;
    for (i, w) in s.iter().enumerate() {
        let got = unsafe { std::ptr::read_volatile(w) };
        let exp = gen(base + i as u64);
        if got != exp {
            bad += 1;
            first.get_or_insert((word_offset + i, exp, got));
        }
    }
    (bad, first)
}

#[target_feature(enable = "avx2")]
unsafe fn verify_gen_avx2(
    s: &[u64],
    base: u64,
    word_offset: usize,
    gen: &impl Fn(u64) -> u64,
) -> (u64, Option<(usize, u64, u64)>) {
    use std::arch::x86_64::*;
    let len = s.len();
    let ptr = s.as_ptr();
    let mut bad = 0u64;
    let mut first = None;
    let mut i = 0usize;
    let end4 = len & !3;
    while i < end4 {
        if i + 32 < end4 {
            _mm_prefetch::<_MM_HINT_NTA>(ptr.add(i + 32) as *const i8);
        }
        let e0 = gen(base + i as u64);
        let e1 = gen(base + (i + 1) as u64);
        let e2 = gen(base + (i + 2) as u64);
        let e3 = gen(base + (i + 3) as u64);
        let ev = _mm256_set_epi64x(e3 as i64, e2 as i64, e1 as i64, e0 as i64);
        let gv = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let cmp = _mm256_cmpeq_epi64(gv, ev);
        if _mm256_movemask_epi8(cmp) != -1i32 {
            let exps = [e0, e1, e2, e3];
            for (k, &exp) in exps.iter().enumerate() {
                let got = std::ptr::read_volatile(ptr.add(i + k));
                if got != exp {
                    bad += 1;
                    first.get_or_insert((word_offset + i + k, exp, got));
                }
            }
        }
        i += 4;
    }
    while i < len {
        let got = std::ptr::read_volatile(ptr.add(i));
        let exp = gen(base + i as u64);
        if got != exp {
            bad += 1;
            first.get_or_insert((word_offset + i, exp, got));
        }
        i += 1;
    }
    (bad, first)
}

/// Fast 256-bit verification for a 2-word repeating `(even, odd)` pattern with NTA prefetch.
fn verify_pair(
    s: &[u64],
    even: u64,
    odd: u64,
    word_offset: usize,
) -> (u64, Option<(usize, u64, u64)>) {
    compiler_fence(Ordering::SeqCst);
    if has_avx2() && s.len() >= 8 {
        return unsafe { verify_pair_avx2(s, even, odd, word_offset) };
    }
    let mut bad = 0u64;
    let mut first = None;
    for (i, w) in s.iter().enumerate() {
        let got = unsafe { std::ptr::read_volatile(w) };
        let exp = if (i & 1) == 0 { even } else { odd };
        if got != exp {
            bad += 1;
            first.get_or_insert((word_offset + i, exp, got));
        }
    }
    (bad, first)
}

#[target_feature(enable = "avx2")]
unsafe fn verify_pair_avx2(
    s: &[u64],
    even: u64,
    odd: u64,
    word_offset: usize,
) -> (u64, Option<(usize, u64, u64)>) {
    use std::arch::x86_64::*;
    let len = s.len();
    let ptr = s.as_ptr();
    let ev = _mm256_set_epi64x(odd as i64, even as i64, odd as i64, even as i64);
    let mut bad = 0u64;
    let mut first = None;
    let mut i = 0usize;
    let end4 = len & !3;
    while i < end4 {
        if i + 32 < end4 {
            _mm_prefetch::<_MM_HINT_NTA>(ptr.add(i + 32) as *const i8);
        }
        let gv = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        if _mm256_movemask_epi8(_mm256_cmpeq_epi64(gv, ev)) != -1i32 {
            for k in 0..4 {
                let got = std::ptr::read_volatile(ptr.add(i + k));
                let exp = if ((i + k) & 1) == 0 { even } else { odd };
                if got != exp {
                    bad += 1;
                    first.get_or_insert((word_offset + i + k, exp, got));
                }
            }
        }
        i += 4;
    }
    while i < len {
        let got = std::ptr::read_volatile(ptr.add(i));
        let exp = if (i & 1) == 0 { even } else { odd };
        if got != exp {
            bad += 1;
            first.get_or_insert((word_offset + i, exp, got));
        }
        i += 1;
    }
    (bad, first)
}

/// Simultaneous cold-DRAM Read-Verify on `src` + cold-DRAM Write on `dst` (either Non-Temporal
/// WC-bypass when `WRITE_BACK=false` or cached Write-Back RFO + async L3 dirty writeback when
/// `WRITE_BACK=true`). Forces back-to-back Read <-> Write bus turnarounds (`tRDWR`, `tWRRD`,
/// `tWTR_S/L`, `tRTP`, `tWR`, ODT, and Infinity Fabric / FCLK coherency) on every 32-byte vector.
#[allow(clippy::too_many_arguments)]
fn dual_verify_and_write_slice<const WRITE_BACK: bool>(
    src: &[u64],
    src_base: u64,
    src_word_offset: usize,
    src_gen: &impl Fn(u64) -> u64,
    dst: &mut [u64],
    dst_base: u64,
    dst_gen: &impl Fn(u64) -> u64,
) -> (u64, Option<(usize, u64, u64)>) {
    debug_assert_eq!(src.len(), dst.len());
    compiler_fence(Ordering::SeqCst);
    if has_avx2() && dst.len() >= 8 {
        return unsafe {
            dual_verify_and_write_avx2::<WRITE_BACK>(
                src,
                src_base,
                src_word_offset,
                src_gen,
                dst,
                dst_base,
                dst_gen,
            )
        };
    }
    let mut bad = 0u64;
    let mut first = None;
    for i in 0..src.len() {
        let got = unsafe { std::ptr::read_volatile(&src[i]) };
        let exp = src_gen(src_base + i as u64);
        if got != exp {
            bad += 1;
            first.get_or_insert((src_word_offset + i, exp, got));
        }
        unsafe { std::ptr::write_volatile(&mut dst[i], dst_gen(dst_base + i as u64)) };
    }
    (bad, first)
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn dual_verify_and_write_avx2<const WRITE_BACK: bool>(
    src: &[u64],
    src_base: u64,
    src_word_offset: usize,
    src_gen: &impl Fn(u64) -> u64,
    dst: &mut [u64],
    dst_base: u64,
    dst_gen: &impl Fn(u64) -> u64,
) -> (u64, Option<(usize, u64, u64)>) {
    use std::arch::x86_64::*;
    let len = dst.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mis = (dp as usize) & 31;
    let prefix = if mis == 0 { 0 } else { (32 - mis) / 8 }.min(len);
    let mut bad = 0u64;
    let mut first = None;
    for i in 0..prefix {
        let got = std::ptr::read_volatile(sp.add(i));
        let exp = src_gen(src_base + i as u64);
        if got != exp {
            bad += 1;
            first.get_or_insert((src_word_offset + i, exp, got));
        }
        std::ptr::write_volatile(dp.add(i), dst_gen(dst_base + i as u64));
    }
    let mut i = prefix;
    let end4 = prefix + ((len - prefix) & !3);
    while i < end4 {
        if i + 32 < end4 {
            _mm_prefetch::<_MM_HINT_NTA>(sp.add(i + 32) as *const i8);
        }
        // Cold DRAM READ + SIMD verify on `src`
        let e0 = src_gen(src_base + i as u64);
        let e1 = src_gen(src_base + (i + 1) as u64);
        let e2 = src_gen(src_base + (i + 2) as u64);
        let e3 = src_gen(src_base + (i + 3) as u64);
        let ev = _mm256_set_epi64x(e3 as i64, e2 as i64, e1 as i64, e0 as i64);
        let gv = _mm256_loadu_si256(sp.add(i) as *const __m256i);
        // Immediately interleave cold DRAM WRITE on `dst` (Read<->Write bus flip!)
        let w0 = dst_gen(dst_base + i as u64);
        let w1 = dst_gen(dst_base + (i + 1) as u64);
        let w2 = dst_gen(dst_base + (i + 2) as u64);
        let w3 = dst_gen(dst_base + (i + 3) as u64);
        let wv = _mm256_set_epi64x(w3 as i64, w2 as i64, w1 as i64, w0 as i64);
        if WRITE_BACK {
            _mm256_storeu_si256(dp.add(i) as *mut __m256i, wv);
        } else {
            _mm256_stream_si256(dp.add(i) as *mut __m256i, wv);
        }

        if _mm256_movemask_epi8(_mm256_cmpeq_epi64(gv, ev)) != -1i32 {
            let exps = [e0, e1, e2, e3];
            for (k, &exp) in exps.iter().enumerate() {
                let got = std::ptr::read_volatile(sp.add(i + k));
                if got != exp {
                    bad += 1;
                    first.get_or_insert((src_word_offset + i + k, exp, got));
                }
            }
        }
        i += 4;
    }
    while i < len {
        let got = std::ptr::read_volatile(sp.add(i));
        let exp = src_gen(src_base + i as u64);
        if got != exp {
            bad += 1;
            first.get_or_insert((src_word_offset + i, exp, got));
        }
        std::ptr::write_volatile(dp.add(i), dst_gen(dst_base + i as u64));
        i += 1;
    }
    (bad, first)
}

/// Swap two disjoint equal-length blocks AND bitwise-invert every word using AVX2 stores
/// (Non-Temporal `_mm256_stream_si256` when `WRITE_BACK=false`, or cached Write-Back RFO when
/// `WRITE_BACK=true`), flipping all 64 DQ lanes in transit.
#[inline]
fn swap_invert_blocks<const WRITE_BACK: bool>(a: &mut [u64], b: &mut [u64]) {
    debug_assert_eq!(a.len(), b.len());
    if has_avx2() && a.len() >= 4 {
        unsafe { swap_invert_blocks_avx2::<WRITE_BACK>(a, b) };
        return;
    }
    for (xa, xb) in a.iter_mut().zip(b.iter_mut()) {
        let va = unsafe { std::ptr::read_volatile(xa) };
        let vb = unsafe { std::ptr::read_volatile(xb) };
        unsafe {
            std::ptr::write_volatile(xa, !vb);
            std::ptr::write_volatile(xb, !va);
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn swap_invert_blocks_avx2<const WRITE_BACK: bool>(a: &mut [u64], b: &mut [u64]) {
    use std::arch::x86_64::*;
    let len = a.len();
    let ap = a.as_mut_ptr();
    let bp = b.as_mut_ptr();
    let use_nt = !WRITE_BACK && ((ap as usize) & 31) == 0 && ((bp as usize) & 31) == 0;
    let ones = _mm256_set1_epi64x(-1i64);
    let mut i = 0usize;
    let end4 = len & !3;
    if use_nt {
        while i < end4 {
            let va = _mm256_xor_si256(_mm256_loadu_si256(ap.add(i) as *const __m256i), ones);
            let vb = _mm256_xor_si256(_mm256_loadu_si256(bp.add(i) as *const __m256i), ones);
            _mm256_stream_si256(ap.add(i) as *mut __m256i, vb);
            _mm256_stream_si256(bp.add(i) as *mut __m256i, va);
            i += 4;
        }
    } else {
        while i < end4 {
            let va = _mm256_xor_si256(_mm256_loadu_si256(ap.add(i) as *const __m256i), ones);
            let vb = _mm256_xor_si256(_mm256_loadu_si256(bp.add(i) as *const __m256i), ones);
            _mm256_storeu_si256(ap.add(i) as *mut __m256i, vb);
            _mm256_storeu_si256(bp.add(i) as *mut __m256i, va);
            i += 4;
        }
    }
    while i < len {
        let va = std::ptr::read_volatile(ap.add(i));
        let vb = std::ptr::read_volatile(bp.add(i));
        std::ptr::write_volatile(ap.add(i), !vb);
        std::ptr::write_volatile(bp.add(i), !va);
        i += 1;
    }
}

/// Fast 4x-unrolled AVX2 cached fill for a 2-word repeating `(even, odd)` BL8 toggle pattern.
fn fill_pair_cached(s: &mut [u64], even: u64, odd: u64) {
    if has_avx2() && s.len() >= 16 {
        unsafe { fill_pair_cached_avx2(s, even, odd) };
        return;
    }
    for (i, w) in s.iter_mut().enumerate() {
        let v = if (i & 1) == 0 { even } else { odd };
        unsafe { std::ptr::write_volatile(w, v) };
    }
}

#[target_feature(enable = "avx2")]
unsafe fn fill_pair_cached_avx2(s: &mut [u64], even: u64, odd: u64) {
    use std::arch::x86_64::*;
    let len = s.len();
    let ptr = s.as_mut_ptr();
    let v = _mm256_set_epi64x(odd as i64, even as i64, odd as i64, even as i64);
    let mut i = 0usize;
    let end16 = len & !15;
    while i < end16 {
        _mm256_storeu_si256(ptr.add(i) as *mut __m256i, v);
        _mm256_storeu_si256(ptr.add(i + 4) as *mut __m256i, v);
        _mm256_storeu_si256(ptr.add(i + 8) as *mut __m256i, v);
        _mm256_storeu_si256(ptr.add(i + 12) as *mut __m256i, v);
        i += 16;
    }
    while i < len {
        std::ptr::write_volatile(ptr.add(i), if (i & 1) == 0 { even } else { odd });
        i += 1;
    }
}

/// Pure AVX2 8-lane 32-bit Vector LCG fill (`v = (v * mul32 + add32)` x 2 rounds) with
/// alternating odd-word BL8 inversion. Zero scalar ALU overhead (~40-50 GB/s).
fn fill_vec_lcg_slice<const WRITE_BACK: bool>(
    s: &mut [u64],
    word_offset: usize,
    seed32: u32,
    mul32: u32,
    add32: u32,
    invert: bool,
) {
    if has_avx2() && s.len() >= 8 {
        unsafe {
            fill_vec_lcg_avx2::<WRITE_BACK>(s, word_offset, seed32, mul32, add32, invert);
        }
        return;
    }
    for (i, w) in s.iter_mut().enumerate() {
        let v = vec_lcg_word(seed32, mul32, add32, word_offset + i, invert);
        unsafe { std::ptr::write_volatile(w, v) };
    }
}

#[target_feature(enable = "avx2")]
unsafe fn fill_vec_lcg_avx2<const WRITE_BACK: bool>(
    s: &mut [u64],
    word_offset: usize,
    seed32: u32,
    mul32: u32,
    add32: u32,
    invert: bool,
) {
    use std::arch::x86_64::*;
    let len = s.len();
    let ptr = s.as_mut_ptr();
    let mis = if WRITE_BACK { 0 } else { (ptr as usize) & 31 };
    let prefix = if mis == 0 { 0 } else { (32 - mis) / 8 }.min(len);
    for i in 0..prefix {
        std::ptr::write_volatile(
            ptr.add(i),
            vec_lcg_word(seed32, mul32, add32, word_offset + i, invert),
        );
    }
    let mul = _mm256_set1_epi32(mul32 as i32);
    let add = _mm256_set1_epi32(add32 as i32);
    let eff_invert = invert ^ (((word_offset + prefix) & 1) != 0);
    let odd_flip = if eff_invert {
        _mm256_set_epi64x(0, -1, 0, -1)
    } else {
        _mm256_set_epi64x(-1, 0, -1, 0)
    };
    let step = _mm256_set1_epi32(8);
    let b0 = ((word_offset + prefix) as u32)
        .wrapping_mul(2)
        .wrapping_add(seed32);
    let mut idx = _mm256_setr_epi32(
        b0 as i32,
        b0.wrapping_add(1) as i32,
        b0.wrapping_add(2) as i32,
        b0.wrapping_add(3) as i32,
        b0.wrapping_add(4) as i32,
        b0.wrapping_add(5) as i32,
        b0.wrapping_add(6) as i32,
        b0.wrapping_add(7) as i32,
    );
    let mut i = prefix;
    let end4 = prefix + ((len - prefix) & !3);
    while i < end4 {
        let v1 = _mm256_add_epi32(_mm256_mullo_epi32(idx, mul), add);
        let v2 = _mm256_xor_si256(_mm256_add_epi32(_mm256_mullo_epi32(v1, mul), add), odd_flip);
        if WRITE_BACK {
            _mm256_storeu_si256(ptr.add(i) as *mut __m256i, v2);
        } else {
            _mm256_stream_si256(ptr.add(i) as *mut __m256i, v2);
        }
        idx = _mm256_add_epi32(idx, step);
        i += 4;
    }
    while i < len {
        std::ptr::write_volatile(
            ptr.add(i),
            vec_lcg_word(seed32, mul32, add32, word_offset + i, invert),
        );
        i += 1;
    }
    if !WRITE_BACK {
        _mm_sfence();
    }
}

/// Pure AVX2 8-lane 32-bit Vector LCG verification (matches `fill_vec_lcg_slice` bit-for-bit).
fn verify_vec_lcg_slice(
    s: &[u64],
    word_offset: usize,
    seed32: u32,
    mul32: u32,
    add32: u32,
    invert: bool,
) -> (u64, Option<(usize, u64, u64)>) {
    compiler_fence(Ordering::SeqCst);
    if has_avx2() && s.len() >= 8 {
        return unsafe { verify_vec_lcg_avx2(s, word_offset, seed32, mul32, add32, invert) };
    }
    let mut bad = 0u64;
    let mut first = None;
    for (i, w) in s.iter().enumerate() {
        let got = unsafe { std::ptr::read_volatile(w) };
        let exp = vec_lcg_word(seed32, mul32, add32, word_offset + i, invert);
        if got != exp {
            bad += 1;
            first.get_or_insert((word_offset + i, exp, got));
        }
    }
    (bad, first)
}

#[target_feature(enable = "avx2")]
unsafe fn verify_vec_lcg_avx2(
    s: &[u64],
    word_offset: usize,
    seed32: u32,
    mul32: u32,
    add32: u32,
    invert: bool,
) -> (u64, Option<(usize, u64, u64)>) {
    use std::arch::x86_64::*;
    let len = s.len();
    let ptr = s.as_ptr();
    let mul = _mm256_set1_epi32(mul32 as i32);
    let add = _mm256_set1_epi32(add32 as i32);
    let eff_invert = invert ^ ((word_offset & 1) != 0);
    let odd_flip = if eff_invert {
        _mm256_set_epi64x(0, -1, 0, -1)
    } else {
        _mm256_set_epi64x(-1, 0, -1, 0)
    };
    let step = _mm256_set1_epi32(8);
    let b0 = (word_offset as u32).wrapping_mul(2).wrapping_add(seed32);
    let mut idx = _mm256_setr_epi32(
        b0 as i32,
        b0.wrapping_add(1) as i32,
        b0.wrapping_add(2) as i32,
        b0.wrapping_add(3) as i32,
        b0.wrapping_add(4) as i32,
        b0.wrapping_add(5) as i32,
        b0.wrapping_add(6) as i32,
        b0.wrapping_add(7) as i32,
    );
    let mut bad = 0u64;
    let mut first = None;
    let mut i = 0usize;
    let end4 = len & !3;
    while i < end4 {
        let v1 = _mm256_add_epi32(_mm256_mullo_epi32(idx, mul), add);
        let ev = _mm256_xor_si256(_mm256_add_epi32(_mm256_mullo_epi32(v1, mul), add), odd_flip);
        let gv = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        if _mm256_movemask_epi8(_mm256_cmpeq_epi64(gv, ev)) != -1i32 {
            for k in 0..4 {
                let got = std::ptr::read_volatile(ptr.add(i + k));
                let exp = vec_lcg_word(seed32, mul32, add32, word_offset + i + k, invert);
                if got != exp {
                    bad += 1;
                    first.get_or_insert((word_offset + i + k, exp, got));
                }
            }
        }
        idx = _mm256_add_epi32(idx, step);
        i += 4;
    }
    while i < len {
        let got = std::ptr::read_volatile(ptr.add(i));
        let exp = vec_lcg_word(seed32, mul32, add32, word_offset + i, invert);
        if got != exp {
            bad += 1;
            first.get_or_insert((word_offset + i, exp, got));
        }
        i += 1;
    }
    (bad, first)
}

/// Full-chunk 64-byte or 128-byte coprime row-conflict in-place RMW for a repeating `(even, odd)`
/// pair. Visits every block exactly once across the entire chunk (multi-MiB + odd cache-line jump),
/// forcing 100% L1/L2/L3 + DRAM row-buffer misses and writing exact `(!even, !odd)` (zero cascade).
#[allow(clippy::too_many_arguments)]
fn scatter_pair_rmw_chunk<const WRITE_BACK: bool>(
    chunk_idx: usize,
    chunk: &mut [u64],
    blk_words: usize,
    even: u64,
    odd: u64,
    stride_div: usize,
    reverse: bool,
    res: &mut StepResult,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
    on_error: &impl Fn(u64, MemMismatch),
) -> bool {
    debug_assert!(blk_words == CACHELINE_WORDS || blk_words == STRIPE128_WORDS);
    compiler_fence(Ordering::SeqCst);
    let blks = chunk.len() / blk_words;
    if blks > 0 {
        if has_avx2() {
            if !unsafe {
                scatter_pair_rmw_avx2::<WRITE_BACK>(
                    chunk_idx, chunk, blks, blk_words, even, odd, stride_div, reverse, res,
                    keep_going, add_bytes, on_error,
                )
            } {
                return false;
            }
        } else {
            let stride = pick_scatter_stride(blks, stride_div);
            let mut b = if reverse { blks - 1 } else { 0 };
            let mut done_words = 0usize;
            for _ in 0..blks {
                if done_words >= STEP_WORDS {
                    if !keep_going() {
                        return false;
                    }
                    add_bytes((done_words * 16) as u64);
                    done_words = 0;
                }
                let w = b * blk_words;
                let mut bad = 0u64;
                let mut first = None;
                for k in 0..blk_words {
                    let got = unsafe { std::ptr::read_volatile(&chunk[w + k]) };
                    let exp = if ((w + k) & 1) == 0 { even } else { odd };
                    if got != exp {
                        bad += 1;
                        first.get_or_insert((w + k, exp, got));
                    }
                    unsafe { std::ptr::write_volatile(&mut chunk[w + k], !exp) };
                }
                res.record(chunk_idx, bad, first, on_error);
                done_words += blk_words;
                if reverse {
                    b = if b >= stride {
                        b - stride
                    } else {
                        b + blks - stride
                    };
                } else {
                    b += stride;
                    if b >= blks {
                        b -= blks;
                    }
                }
            }
            if done_words > 0 {
                add_bytes((done_words * 16) as u64);
            }
        }
    }
    let tail = blks * blk_words;
    if tail < chunk.len() {
        let mut bad = 0u64;
        let mut first = None;
        for (k, slot) in chunk[tail..].iter_mut().enumerate() {
            let idx = tail + k;
            let got = unsafe { std::ptr::read_volatile(slot) };
            let exp = if (idx & 1) == 0 { even } else { odd };
            if got != exp {
                bad += 1;
                first.get_or_insert((idx, exp, got));
            }
            unsafe { std::ptr::write_volatile(slot, !exp) };
        }
        res.record(chunk_idx, bad, first, on_error);
        add_bytes(((chunk.len() - tail) * 16) as u64);
    }
    true
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn scatter_pair_rmw_avx2<const WRITE_BACK: bool>(
    chunk_idx: usize,
    chunk: &mut [u64],
    blks: usize,
    blk_words: usize,
    even: u64,
    odd: u64,
    stride_div: usize,
    reverse: bool,
    res: &mut StepResult,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
    on_error: &impl Fn(u64, MemMismatch),
) -> bool {
    use std::arch::x86_64::*;
    let stride = pick_scatter_stride(blks, stride_div);
    let ptr = chunk.as_mut_ptr();
    let use_nt = !WRITE_BACK && ((ptr as usize) & 31) == 0;
    let ev = _mm256_set_epi64x(odd as i64, even as i64, odd as i64, even as i64);
    let next_ev = _mm256_set_epi64x((!odd) as i64, (!even) as i64, (!odd) as i64, (!even) as i64);
    let mut b = if reverse { blks - 1 } else { 0 };
    let mut done_words = 0usize;
    for _ in 0..blks {
        if done_words >= STEP_WORDS {
            if use_nt {
                _mm_sfence();
            }
            if !keep_going() {
                return false;
            }
            add_bytes((done_words * 16) as u64);
            done_words = 0;
        }
        let w = b * blk_words;
        let mut off = 0usize;
        while off < blk_words {
            let g0 = _mm256_loadu_si256(ptr.add(w + off) as *const __m256i);
            let g1 = _mm256_loadu_si256(ptr.add(w + off + 4) as *const __m256i);
            let all = _mm256_and_si256(_mm256_cmpeq_epi64(g0, ev), _mm256_cmpeq_epi64(g1, ev));
            if _mm256_movemask_epi8(all) != -1i32 {
                let mut bad = 0u64;
                let mut first = None;
                for k in 0..8 {
                    let got = std::ptr::read_volatile(ptr.add(w + off + k));
                    let exp = if ((w + off + k) & 1) == 0 { even } else { odd };
                    if got != exp {
                        bad += 1;
                        first.get_or_insert((w + off + k, exp, got));
                    }
                }
                res.record(chunk_idx, bad, first, on_error);
            }
            if use_nt {
                _mm256_stream_si256(ptr.add(w + off) as *mut __m256i, next_ev);
                _mm256_stream_si256(ptr.add(w + off + 4) as *mut __m256i, next_ev);
            } else {
                _mm256_storeu_si256(ptr.add(w + off) as *mut __m256i, next_ev);
                _mm256_storeu_si256(ptr.add(w + off + 4) as *mut __m256i, next_ev);
            }
            off += 8;
        }
        done_words += blk_words;
        if reverse {
            b = if b >= stride {
                b - stride
            } else {
                b + blks - stride
            };
        } else {
            b += stride;
            if b >= blks {
                b -= blks;
            }
        }
    }
    if done_words > 0 {
        add_bytes((done_words * 16) as u64);
    }
    if use_nt {
        _mm_sfence();
    }
    true
}

/// Full-chunk 64-byte or 128-byte coprime row-conflict in-place RMW for pure AVX2 Vector LCG
/// (`mul32 / add32` + alternating odd-word BL8 inversion). Verifies exact `invert` state and
/// writes exact `!invert` state (zero error cascade) at ~30+ GB/s of random DRAM row-misses.
#[allow(clippy::too_many_arguments)]
fn scatter_vec_lcg_rmw_chunk<const WRITE_BACK: bool>(
    chunk_idx: usize,
    chunk: &mut [u64],
    blk_words: usize,
    seed32: u32,
    mul32: u32,
    add32: u32,
    invert: bool,
    stride_div: usize,
    reverse: bool,
    res: &mut StepResult,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
    on_error: &impl Fn(u64, MemMismatch),
) -> bool {
    debug_assert!(blk_words == CACHELINE_WORDS || blk_words == STRIPE128_WORDS);
    compiler_fence(Ordering::SeqCst);
    let blks = chunk.len() / blk_words;
    if blks > 0 {
        if has_avx2() {
            if !unsafe {
                scatter_vec_lcg_rmw_avx2::<WRITE_BACK>(
                    chunk_idx, chunk, blks, blk_words, seed32, mul32, add32, invert, stride_div,
                    reverse, res, keep_going, add_bytes, on_error,
                )
            } {
                return false;
            }
        } else {
            let stride = pick_scatter_stride(blks, stride_div);
            let mut b = if reverse { blks - 1 } else { 0 };
            let mut done_words = 0usize;
            for _ in 0..blks {
                if done_words >= STEP_WORDS {
                    if !keep_going() {
                        return false;
                    }
                    add_bytes((done_words * 16) as u64);
                    done_words = 0;
                }
                let w = b * blk_words;
                let mut bad = 0u64;
                let mut first = None;
                for k in 0..blk_words {
                    let got = unsafe { std::ptr::read_volatile(&chunk[w + k]) };
                    let exp = vec_lcg_word(seed32, mul32, add32, w + k, invert);
                    if got != exp {
                        bad += 1;
                        first.get_or_insert((w + k, exp, got));
                    }
                    unsafe { std::ptr::write_volatile(&mut chunk[w + k], !exp) };
                }
                res.record(chunk_idx, bad, first, on_error);
                done_words += blk_words;
                if reverse {
                    b = if b >= stride {
                        b - stride
                    } else {
                        b + blks - stride
                    };
                } else {
                    b += stride;
                    if b >= blks {
                        b -= blks;
                    }
                }
            }
            if done_words > 0 {
                add_bytes((done_words * 16) as u64);
            }
        }
    }
    let tail = blks * blk_words;
    if tail < chunk.len() {
        let mut bad = 0u64;
        let mut first = None;
        for (k, slot) in chunk[tail..].iter_mut().enumerate() {
            let idx = tail + k;
            let got = unsafe { std::ptr::read_volatile(slot) };
            let exp = vec_lcg_word(seed32, mul32, add32, idx, invert);
            if got != exp {
                bad += 1;
                first.get_or_insert((idx, exp, got));
            }
            unsafe { std::ptr::write_volatile(slot, !exp) };
        }
        res.record(chunk_idx, bad, first, on_error);
        add_bytes(((chunk.len() - tail) * 16) as u64);
    }
    true
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn scatter_vec_lcg_rmw_avx2<const WRITE_BACK: bool>(
    chunk_idx: usize,
    chunk: &mut [u64],
    blks: usize,
    blk_words: usize,
    seed32: u32,
    mul32: u32,
    add32: u32,
    invert: bool,
    stride_div: usize,
    reverse: bool,
    res: &mut StepResult,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
    on_error: &impl Fn(u64, MemMismatch),
) -> bool {
    use std::arch::x86_64::*;
    let stride = pick_scatter_stride(blks, stride_div);
    let ptr = chunk.as_mut_ptr();
    let use_nt = !WRITE_BACK && ((ptr as usize) & 31) == 0;
    let mul = _mm256_set1_epi32(mul32 as i32);
    let add = _mm256_set1_epi32(add32 as i32);
    let odd_flip = if invert {
        _mm256_set_epi64x(0, -1, 0, -1)
    } else {
        _mm256_set_epi64x(-1, 0, -1, 0)
    };
    let ones = _mm256_set1_epi64x(-1i64);
    let offs0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let offs1 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    let mut b = if reverse { blks - 1 } else { 0 };
    let mut done_words = 0usize;
    for _ in 0..blks {
        if done_words >= STEP_WORDS {
            if use_nt {
                _mm_sfence();
            }
            if !keep_going() {
                return false;
            }
            add_bytes((done_words * 16) as u64);
            done_words = 0;
        }
        let w = b * blk_words;
        let mut off = 0usize;
        while off < blk_words {
            let base32 =
                _mm256_set1_epi32(((w + off) as u32).wrapping_mul(2).wrapping_add(seed32) as i32);
            let idx0 = _mm256_add_epi32(base32, offs0);
            let idx1 = _mm256_add_epi32(base32, offs1);
            let t0 = _mm256_add_epi32(_mm256_mullo_epi32(idx0, mul), add);
            let e0 = _mm256_xor_si256(_mm256_add_epi32(_mm256_mullo_epi32(t0, mul), add), odd_flip);
            let t1 = _mm256_add_epi32(_mm256_mullo_epi32(idx1, mul), add);
            let e1 = _mm256_xor_si256(_mm256_add_epi32(_mm256_mullo_epi32(t1, mul), add), odd_flip);
            let g0 = _mm256_loadu_si256(ptr.add(w + off) as *const __m256i);
            let g1 = _mm256_loadu_si256(ptr.add(w + off + 4) as *const __m256i);
            let all = _mm256_and_si256(_mm256_cmpeq_epi64(g0, e0), _mm256_cmpeq_epi64(g1, e1));
            if _mm256_movemask_epi8(all) != -1i32 {
                let mut bad = 0u64;
                let mut first = None;
                for k in 0..8 {
                    let got = std::ptr::read_volatile(ptr.add(w + off + k));
                    let exp = vec_lcg_word(seed32, mul32, add32, w + off + k, invert);
                    if got != exp {
                        bad += 1;
                        first.get_or_insert((w + off + k, exp, got));
                    }
                }
                res.record(chunk_idx, bad, first, on_error);
            }
            let n0 = _mm256_xor_si256(e0, ones);
            let n1 = _mm256_xor_si256(e1, ones);
            if use_nt {
                _mm256_stream_si256(ptr.add(w + off) as *mut __m256i, n0);
                _mm256_stream_si256(ptr.add(w + off + 4) as *mut __m256i, n1);
            } else {
                _mm256_storeu_si256(ptr.add(w + off) as *mut __m256i, n0);
                _mm256_storeu_si256(ptr.add(w + off + 4) as *mut __m256i, n1);
            }
            off += 8;
        }
        done_words += blk_words;
        if reverse {
            b = if b >= stride {
                b - stride
            } else {
                b + blks - stride
            };
        } else {
            b += stride;
            if b >= blks {
                b -= blks;
            }
        }
    }
    if done_words > 0 {
        add_bytes((done_words * 16) as u64);
    }
    if use_nt {
        _mm_sfence();
    }
    true
}

/// Full-chunk 64-byte coprime row-conflict in-place RMW for any per-word generator `gen_exp`.
/// Verifies `gen_exp(base + idx)` and writes `!gen_exp(base + idx)` in-place (zero cascade).
#[allow(clippy::too_many_arguments)]
fn scatter_gen_rmw_chunk<const WRITE_BACK: bool>(
    chunk_idx: usize,
    chunk: &mut [u64],
    chunk_base: u64,
    stride_div: usize,
    reverse: bool,
    gen_exp: &impl Fn(u64) -> u64,
    res: &mut StepResult,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
    on_error: &impl Fn(u64, MemMismatch),
) -> bool {
    compiler_fence(Ordering::SeqCst);
    let blks = chunk.len() / CACHELINE_WORDS;
    if blks > 0 {
        if has_avx2() {
            if !unsafe {
                scatter_gen_rmw_avx2::<WRITE_BACK>(
                    chunk_idx, chunk, chunk_base, blks, stride_div, reverse, gen_exp, res,
                    keep_going, add_bytes, on_error,
                )
            } {
                return false;
            }
        } else {
            let stride = pick_scatter_stride(blks, stride_div);
            let mut b = if reverse { blks - 1 } else { 0 };
            let mut done_words = 0usize;
            for _ in 0..blks {
                if done_words >= STEP_WORDS {
                    if !keep_going() {
                        return false;
                    }
                    add_bytes((done_words * 16) as u64);
                    done_words = 0;
                }
                let w = b * CACHELINE_WORDS;
                let mut bad = 0u64;
                let mut first = None;
                for k in 0..CACHELINE_WORDS {
                    let got = unsafe { std::ptr::read_volatile(&chunk[w + k]) };
                    let exp = gen_exp(chunk_base + (w + k) as u64);
                    if got != exp {
                        bad += 1;
                        first.get_or_insert((w + k, exp, got));
                    }
                    unsafe { std::ptr::write_volatile(&mut chunk[w + k], !exp) };
                }
                res.record(chunk_idx, bad, first, on_error);
                done_words += CACHELINE_WORDS;
                if reverse {
                    b = if b >= stride {
                        b - stride
                    } else {
                        b + blks - stride
                    };
                } else {
                    b += stride;
                    if b >= blks {
                        b -= blks;
                    }
                }
            }
            if done_words > 0 {
                add_bytes((done_words * 16) as u64);
            }
        }
    }
    let tail = blks * CACHELINE_WORDS;
    if tail < chunk.len() {
        let mut bad = 0u64;
        let mut first = None;
        for (k, slot) in chunk[tail..].iter_mut().enumerate() {
            let idx = tail + k;
            let got = unsafe { std::ptr::read_volatile(slot) };
            let exp = gen_exp(chunk_base + idx as u64);
            if got != exp {
                bad += 1;
                first.get_or_insert((idx, exp, got));
            }
            unsafe { std::ptr::write_volatile(slot, !exp) };
        }
        res.record(chunk_idx, bad, first, on_error);
        add_bytes(((chunk.len() - tail) * 16) as u64);
    }
    true
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn scatter_gen_rmw_avx2<const WRITE_BACK: bool>(
    chunk_idx: usize,
    chunk: &mut [u64],
    chunk_base: u64,
    blks: usize,
    stride_div: usize,
    reverse: bool,
    gen_exp: &impl Fn(u64) -> u64,
    res: &mut StepResult,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
    on_error: &impl Fn(u64, MemMismatch),
) -> bool {
    use std::arch::x86_64::*;
    let stride = pick_scatter_stride(blks, stride_div);
    let ptr = chunk.as_mut_ptr();
    let use_nt = !WRITE_BACK && ((ptr as usize) & 31) == 0;
    let ones = _mm256_set1_epi64x(-1i64);
    let mut b = if reverse { blks - 1 } else { 0 };
    let mut done_words = 0usize;
    for _ in 0..blks {
        if done_words >= STEP_WORDS {
            if use_nt {
                _mm_sfence();
            }
            if !keep_going() {
                return false;
            }
            add_bytes((done_words * 16) as u64);
            done_words = 0;
        }
        let w = b * CACHELINE_WORDS;
        let gw = chunk_base + w as u64;
        let e0 = gen_exp(gw);
        let e1 = gen_exp(gw + 1);
        let e2 = gen_exp(gw + 2);
        let e3 = gen_exp(gw + 3);
        let e4 = gen_exp(gw + 4);
        let e5 = gen_exp(gw + 5);
        let e6 = gen_exp(gw + 6);
        let e7 = gen_exp(gw + 7);
        let ev0 = _mm256_set_epi64x(e3 as i64, e2 as i64, e1 as i64, e0 as i64);
        let ev1 = _mm256_set_epi64x(e7 as i64, e6 as i64, e5 as i64, e4 as i64);
        let g0 = _mm256_loadu_si256(ptr.add(w) as *const __m256i);
        let g1 = _mm256_loadu_si256(ptr.add(w + 4) as *const __m256i);
        let all = _mm256_and_si256(_mm256_cmpeq_epi64(g0, ev0), _mm256_cmpeq_epi64(g1, ev1));
        if _mm256_movemask_epi8(all) != -1i32 {
            let exps = [e0, e1, e2, e3, e4, e5, e6, e7];
            let mut bad = 0u64;
            let mut first = None;
            for (k, &exp) in exps.iter().enumerate() {
                let got = std::ptr::read_volatile(ptr.add(w + k));
                if got != exp {
                    bad += 1;
                    first.get_or_insert((w + k, exp, got));
                }
            }
            res.record(chunk_idx, bad, first, on_error);
        }
        let n0 = _mm256_xor_si256(ev0, ones);
        let n1 = _mm256_xor_si256(ev1, ones);
        if use_nt {
            _mm256_stream_si256(ptr.add(w) as *mut __m256i, n0);
            _mm256_stream_si256(ptr.add(w + 4) as *mut __m256i, n1);
        } else {
            _mm256_storeu_si256(ptr.add(w) as *mut __m256i, n0);
            _mm256_storeu_si256(ptr.add(w + 4) as *mut __m256i, n1);
        }
        done_words += CACHELINE_WORDS;
        if reverse {
            b = if b >= stride {
                b - stride
            } else {
                b + blks - stride
            };
        } else {
            b += stride;
            if b >= blks {
                b -= blks;
            }
        }
    }
    if done_words > 0 {
        add_bytes((done_words * 16) as u64);
    }
    if use_nt {
        _mm_sfence();
    }
    true
}

/// Flush a memory address from all CPU cache levels so the next read forces a physical DRAM row open.
#[inline(always)]
unsafe fn clflush_word(ptr: *const u64) {
    use std::arch::x86_64::{_mm_clflush, _mm_mfence};
    _mm_clflush(ptr as *const u8);
    _mm_mfence();
}

// ---------------------------------------------------------------------------
// Bank-strided 4 KiB page-jumping helpers (force DRAM row/bank ACT+PRE at full AVX2 speed)
// ---------------------------------------------------------------------------

/// Fill `chunk` in bijective coprime 4 KiB page-jumping order via non-temporal AVX2 stores.
/// Returns `false` if `keep_going` requested abort.
#[allow(clippy::too_many_arguments)]
fn nt_fill_chunk_strided(
    chunk: &mut [u64],
    chunk_base: u64,
    stride_hint: usize,
    reverse: bool,
    gen: &impl Fn(u64) -> u64,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
) -> bool {
    let pages = chunk.len() / PAGE_WORDS;
    if pages > 0 {
        let stride = coprime_stride(pages, stride_hint);
        let mut p = if reverse { pages - 1 } else { 0 };
        let mut done_words = 0usize;
        for _ in 0..pages {
            if done_words >= STEP_WORDS {
                sfence_if_x86();
                if !keep_going() {
                    return false;
                }
                add_bytes((done_words * 8) as u64);
                done_words = 0;
            }
            let w0 = p * PAGE_WORDS;
            nt_fill_gen_nofence(&mut chunk[w0..w0 + PAGE_WORDS], chunk_base + w0 as u64, gen);
            done_words += PAGE_WORDS;
            if reverse {
                p = if p >= stride {
                    p - stride
                } else {
                    p + pages - stride
                };
            } else {
                p += stride;
                if p >= pages {
                    p -= pages;
                }
            }
        }
        if done_words > 0 {
            add_bytes((done_words * 8) as u64);
        }
    }
    let tail = pages * PAGE_WORDS;
    if tail < chunk.len() {
        nt_fill_gen_nofence(&mut chunk[tail..], chunk_base + tail as u64, gen);
        add_bytes(((chunk.len() - tail) * 8) as u64);
    }
    sfence_if_x86();
    true
}

/// Verify `chunk` in bijective coprime 4 KiB page-jumping order with NTA prefetch & immediate error reporting.
/// Returns `false` if `keep_going` requested abort.
#[allow(clippy::too_many_arguments)]
fn verify_chunk_strided(
    chunk_idx: usize,
    chunk: &[u64],
    chunk_base: u64,
    stride_hint: usize,
    reverse: bool,
    gen: &impl Fn(u64) -> u64,
    res: &mut StepResult,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
    on_error: &impl Fn(u64, MemMismatch),
) -> bool {
    let pages = chunk.len() / PAGE_WORDS;
    if pages > 0 {
        let stride = coprime_stride(pages, stride_hint);
        let mut p = if reverse { pages - 1 } else { 0 };
        let mut done_words = 0usize;
        for _ in 0..pages {
            if done_words >= STEP_WORDS {
                if !keep_going() {
                    return false;
                }
                add_bytes((done_words * 8) as u64);
                done_words = 0;
            }
            let w0 = p * PAGE_WORDS;
            let (bad, first) =
                verify_gen(&chunk[w0..w0 + PAGE_WORDS], chunk_base + w0 as u64, w0, gen);
            res.record(chunk_idx, bad, first, on_error);
            done_words += PAGE_WORDS;
            if reverse {
                p = if p >= stride {
                    p - stride
                } else {
                    p + pages - stride
                };
            } else {
                p += stride;
                if p >= pages {
                    p -= pages;
                }
            }
        }
        if done_words > 0 {
            add_bytes((done_words * 8) as u64);
        }
    }
    let tail = pages * PAGE_WORDS;
    if tail < chunk.len() {
        let (bad, first) = verify_gen(&chunk[tail..], chunk_base + tail as u64, tail, gen);
        res.record(chunk_idx, bad, first, on_error);
        add_bytes(((chunk.len() - tail) * 8) as u64);
    }
    true
}

/// High-speed full-chunk coprime row-conflict torture runner for repeating `(even, odd)` BL8
/// patterns (e.g. `0x0000_0000_0000_0000 / 0xFFFF_FFFF_FFFF_FFFF` toggling all 64 DQ pins on
/// every DDR burst beat).
///
/// Phase 1: Full-bandwidth fill across ALL chunks (total allocation >> L3 -> cold DRAM).
/// Phase 2: Forward full-chunk 64B/128B coprime row-conflict RMW (`verify pat -> write !pat`).
/// Phase 3: Reverse full-chunk 64B/128B coprime row-conflict RMW (`verify !pat -> write pat`,
///          using a distinct coprime divisor to reverse every bank/row sequence).
/// Phase 4: Full-speed final verification across all chunks.
#[allow(clippy::too_many_arguments)]
fn run_scatter_pair_torture<const WRITE_BACK: bool>(
    chunks: &mut [(Vec<u64>, u64)],
    blk_words: usize,
    even: u64,
    odd: u64,
    div_idx: usize,
    res: &mut StepResult,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
    on_error: &impl Fn(u64, MemMismatch),
) {
    if chunks.is_empty() {
        return;
    }
    let div_fwd = SCATTER_DIVS[div_idx % SCATTER_DIVS.len()];
    let div_rev = SCATTER_DIVS[(div_idx + 3) % SCATTER_DIVS.len()];

    // Phase 1: High-speed fill across all chunks so every cache line reaches physical DRAM.
    for (chunk, _) in chunks.iter_mut() {
        for step in chunk.chunks_mut(STEP_WORDS) {
            if !keep_going() {
                res.aborted = true;
                return;
            }
            if WRITE_BACK {
                fill_pair_cached(step, even, odd);
            } else {
                nt_fill_pair(step, even, odd);
            }
            add_bytes((step.len() * 8) as u64);
        }
    }

    // Phase 2: Forward full-chunk coprime row-conflict in-place RMW (pat -> !pat).
    for (ci, (chunk, _)) in chunks.iter_mut().enumerate() {
        if !scatter_pair_rmw_chunk::<WRITE_BACK>(
            ci, chunk, blk_words, even, odd, div_fwd, false, res, keep_going, add_bytes, on_error,
        ) {
            res.aborted = true;
            return;
        }
    }

    // Phase 3: Reverse full-chunk coprime row-conflict in-place RMW (!pat -> pat).
    for (ci, (chunk, _)) in chunks.iter_mut().enumerate().rev() {
        if !scatter_pair_rmw_chunk::<WRITE_BACK>(
            ci, chunk, blk_words, !even, !odd, div_rev, true, res, keep_going, add_bytes, on_error,
        ) {
            res.aborted = true;
            return;
        }
    }

    // Phase 4: Final cold verification pass.
    for (ci, (chunk, _)) in chunks.iter().enumerate() {
        for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
            if !keep_going() {
                res.aborted = true;
                return;
            }
            let (bad, first) = verify_pair(step, even, odd, si * STEP_WORDS);
            res.record(ci, bad, first, on_error);
            add_bytes((step.len() * 8) as u64);
        }
    }
}

/// High-speed full-chunk coprime row-conflict torture runner for pure AVX2 8-lane Vector LCG
/// (`mul32 / add32` + alternating odd-word BL8 inversion). Zero scalar ALU bottleneck (~30+ GB/s).
#[allow(clippy::too_many_arguments)]
fn run_scatter_vec_lcg_torture<const WRITE_BACK: bool>(
    chunks: &mut [(Vec<u64>, u64)],
    base_word: u64,
    seed: u64,
    blk_words: usize,
    mul32: u32,
    add32: u32,
    div_idx: usize,
    res: &mut StepResult,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
    on_error: &impl Fn(u64, MemMismatch),
) {
    if chunks.is_empty() {
        return;
    }
    let div_fwd = SCATTER_DIVS[div_idx % SCATTER_DIVS.len()];
    let div_rev = SCATTER_DIVS[(div_idx + 3) % SCATTER_DIVS.len()];

    // Phase 1: Full-speed AVX2 Vector LCG fill across all chunks.
    for (chunk, chunk_start) in chunks.iter_mut() {
        let s32 = chunk_seed32(base_word, *chunk_start, seed);
        for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
            if !keep_going() {
                res.aborted = true;
                return;
            }
            fill_vec_lcg_slice::<WRITE_BACK>(step, si * STEP_WORDS, s32, mul32, add32, false);
            add_bytes((step.len() * 8) as u64);
        }
    }

    // Phase 2: Forward full-chunk coprime row-conflict in-place RMW (invert=false -> true).
    for (ci, (chunk, chunk_start)) in chunks.iter_mut().enumerate() {
        let s32 = chunk_seed32(base_word, *chunk_start, seed);
        if !scatter_vec_lcg_rmw_chunk::<WRITE_BACK>(
            ci, chunk, blk_words, s32, mul32, add32, false, div_fwd, false, res, keep_going,
            add_bytes, on_error,
        ) {
            res.aborted = true;
            return;
        }
    }

    // Phase 3: Reverse full-chunk coprime row-conflict in-place RMW (invert=true -> false).
    for (ci, (chunk, chunk_start)) in chunks.iter_mut().enumerate().rev() {
        let s32 = chunk_seed32(base_word, *chunk_start, seed);
        if !scatter_vec_lcg_rmw_chunk::<WRITE_BACK>(
            ci, chunk, blk_words, s32, mul32, add32, true, div_rev, true, res, keep_going,
            add_bytes, on_error,
        ) {
            res.aborted = true;
            return;
        }
    }

    // Phase 4: Final cold AVX2 Vector LCG verification pass.
    for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
        let s32 = chunk_seed32(base_word, *chunk_start, seed);
        for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
            if !keep_going() {
                res.aborted = true;
                return;
            }
            let (bad, first) =
                verify_vec_lcg_slice(step, si * STEP_WORDS, s32, mul32, add32, false);
            res.record(ci, bad, first, on_error);
            add_bytes((step.len() * 8) as u64);
        }
    }
}

/// Core Absolut SimpleTest multi-mode torture runner for arbitrary per-word pattern generators.
///
/// `rmw_mode` rotates through 3 complementary hardware stress modes across sub-passes:
///   * `0`: Full-chunk 64B coprime row-conflict cached Write-Back RMW (max DRAM ACT+PRE + L3 dirty evictions)
///   * `1`: Full-chunk 64B coprime row-conflict Non-Temporal RMW (direct WC-buffer DRAM row-miss storms)
///   * `2`: 4 KiB bank-strided simultaneous cross-chunk cold-DRAM Read-Verify + NT Write (`tRDWR`/`tWRRD`/`tWTR`)
#[allow(clippy::too_many_arguments)]
fn run_simple_dual_rmw(
    chunks: &mut [(Vec<u64>, u64)],
    base_word: u64,
    stride_hint: usize,
    rmw_mode: u32,
    gen_fwd: &impl Fn(u64) -> u64,
    res: &mut StepResult,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
    on_error: &impl Fn(u64, MemMismatch),
) {
    let gen_inv = |g: u64| !gen_fwd(g);
    let n = chunks.len();
    if n == 0 {
        return;
    }

    // Modes 0 (Write-Back RFO) & 1 (Non-Temporal): Full-chunk 64-byte coprime row-conflict RMW!
    if rmw_mode != 2 {
        let write_back = rmw_mode == 0;
        let div_fwd = SCATTER_DIVS[stride_hint % SCATTER_DIVS.len()];
        let div_rev = SCATTER_DIVS[(stride_hint + 3) % SCATTER_DIVS.len()];
        for (chunk, chunk_start) in chunks.iter_mut() {
            let cb = base_word + *chunk_start;
            for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
                if !keep_going() {
                    res.aborted = true;
                    return;
                }
                let b = cb + (si * STEP_WORDS) as u64;
                if write_back {
                    fill_gen_nofence::<true>(step, b, gen_fwd);
                } else {
                    fill_gen_nofence::<false>(step, b, gen_fwd);
                }
                add_bytes((step.len() * 8) as u64);
            }
            if !write_back {
                sfence_if_x86();
            }
        }
        for (ci, (chunk, chunk_start)) in chunks.iter_mut().enumerate() {
            let cb = base_word + *chunk_start;
            let ok = if write_back {
                scatter_gen_rmw_chunk::<true>(
                    ci, chunk, cb, div_fwd, false, gen_fwd, res, keep_going, add_bytes, on_error,
                )
            } else {
                scatter_gen_rmw_chunk::<false>(
                    ci, chunk, cb, div_fwd, false, gen_fwd, res, keep_going, add_bytes, on_error,
                )
            };
            if !ok {
                res.aborted = true;
                return;
            }
        }
        for (ci, (chunk, chunk_start)) in chunks.iter_mut().enumerate().rev() {
            let cb = base_word + *chunk_start;
            let ok = if write_back {
                scatter_gen_rmw_chunk::<true>(
                    ci, chunk, cb, div_rev, true, &gen_inv, res, keep_going, add_bytes, on_error,
                )
            } else {
                scatter_gen_rmw_chunk::<false>(
                    ci, chunk, cb, div_rev, true, &gen_inv, res, keep_going, add_bytes, on_error,
                )
            };
            if !ok {
                res.aborted = true;
                return;
            }
        }
        for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
            let cb = base_word + *chunk_start;
            for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                if !keep_going() {
                    res.aborted = true;
                    return;
                }
                let w0 = si * STEP_WORDS;
                let (bad, first) = verify_gen(step, cb + w0 as u64, w0, gen_fwd);
                res.record(ci, bad, first, on_error);
                add_bytes((step.len() * 8) as u64);
            }
        }
        return;
    }

    // Mode 2: 4 KiB bank-strided simultaneous cross-chunk cold-DRAM Read-Verify + NT Write turnaround.
    for (chunk, chunk_start) in chunks.iter_mut() {
        if !nt_fill_chunk_strided(
            chunk,
            base_word + *chunk_start,
            stride_hint,
            false,
            gen_fwd,
            keep_going,
            add_bytes,
        ) {
            res.aborted = true;
            return;
        }
    }

    if n == 1 {
        let (chunk, chunk_start) = &mut chunks[0];
        let b = base_word + *chunk_start;
        if !verify_chunk_strided(
            0,
            chunk,
            b,
            stride_hint,
            true,
            gen_fwd,
            res,
            keep_going,
            add_bytes,
            on_error,
        ) {
            res.aborted = true;
            return;
        }
        if !nt_fill_chunk_strided(chunk, b, stride_hint, true, &gen_inv, keep_going, add_bytes) {
            res.aborted = true;
            return;
        }
    } else {
        {
            let (c0, s0) = &chunks[0];
            if !verify_chunk_strided(
                0,
                c0,
                base_word + *s0,
                stride_hint,
                false,
                gen_fwd,
                res,
                keep_going,
                add_bytes,
                on_error,
            ) {
                res.aborted = true;
                return;
            }
        }
        for k in 1..n {
            let (left, right) = chunks.split_at_mut(k);
            let (dst_vec, dst_start) = &mut left[k - 1];
            let (src_vec, src_start) = &right[0];
            let dst_base = base_word + *dst_start;
            let src_base = base_word + *src_start;
            let common = src_vec.len().min(dst_vec.len());
            let common_pages = common / PAGE_WORDS;
            if common_pages > 0 {
                let stride = coprime_stride(common_pages, stride_hint);
                let mut p = common_pages - 1;
                let mut done_words = 0usize;
                for _ in 0..common_pages {
                    if done_words >= STEP_WORDS {
                        sfence_if_x86();
                        if !keep_going() {
                            res.aborted = true;
                            return;
                        }
                        add_bytes((done_words * 16) as u64);
                        done_words = 0;
                    }
                    let w0 = p * PAGE_WORDS;
                    let (bad, first) = dual_verify_and_write_slice::<false>(
                        &src_vec[w0..w0 + PAGE_WORDS],
                        src_base + w0 as u64,
                        w0,
                        gen_fwd,
                        &mut dst_vec[w0..w0 + PAGE_WORDS],
                        dst_base + w0 as u64,
                        &gen_inv,
                    );
                    res.record(k, bad, first, on_error);
                    done_words += PAGE_WORDS;
                    p = if p >= stride {
                        p - stride
                    } else {
                        p + common_pages - stride
                    };
                }
                if done_words > 0 {
                    add_bytes((done_words * 16) as u64);
                }
                sfence_if_x86();
            }
            let covered = common_pages * PAGE_WORDS;
            if covered < src_vec.len() {
                let (bad, first) = verify_gen(
                    &src_vec[covered..],
                    src_base + covered as u64,
                    covered,
                    gen_fwd,
                );
                res.record(k, bad, first, on_error);
                add_bytes(((src_vec.len() - covered) * 8) as u64);
            }
            if covered < dst_vec.len()
                && !nt_fill_chunk_strided(
                    &mut dst_vec[covered..],
                    dst_base + covered as u64,
                    stride_hint,
                    true,
                    &gen_inv,
                    keep_going,
                    add_bytes,
                )
            {
                res.aborted = true;
                return;
            }
        }
        let (last_vec, last_start) = &mut chunks[n - 1];
        if !nt_fill_chunk_strided(
            last_vec,
            base_word + *last_start,
            stride_hint,
            true,
            &gen_inv,
            keep_going,
            add_bytes,
        ) {
            res.aborted = true;
            return;
        }
    }

    for (ci, (chunk, chunk_start)) in chunks.iter().enumerate().rev() {
        if !verify_chunk_strided(
            ci,
            chunk,
            base_word + *chunk_start,
            stride_hint,
            true,
            &gen_inv,
            res,
            keep_going,
            add_bytes,
            on_error,
        ) {
            res.aborted = true;
            return;
        }
    }
}

/// Cross-allocation coprime-scattered MirrorMove with full 64-bit transit inversion.
///
/// Pairs opposing-end chunks (`chunks[c]` <-> `chunks[n - 1 - c]`) and traverses block pairs in
/// bijective coprime row-conflict order (`j = (j + stride) % pair_cnt`) so every 128-byte stripe
/// or 4 KiB page triggers a physical DRAM row-buffer miss across opposite regions of RAM.
/// Rotates between cached Write-Back RFO (`WRITE_BACK=true`) and Non-Temporal WC-bypass (`false`).
#[allow(clippy::too_many_arguments)]
fn cross_chunk_mirror<const WRITE_BACK: bool>(
    chunks: &mut [(Vec<u64>, u64)],
    block_words: usize,
    striped_parity: Option<usize>,
    stride_div: usize,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
) -> bool {
    let n = chunks.len();
    let chunk_pairs = n / 2;
    for c in 0..chunk_pairs {
        let hi_idx = n - 1 - c;
        let (left, right) = chunks.split_at_mut(hi_idx);
        let a = &mut left[c].0;
        let b = &mut right[0].0;
        let pair_cnt = (a.len().min(b.len()) / block_words) / 2;
        if pair_cnt == 0 {
            continue;
        }
        let stride = pick_scatter_stride(pair_cnt, stride_div);
        let mut j = 0usize;
        let mut done_words = 0usize;
        for _ in 0..pair_cnt {
            for parity in 0..2usize {
                if let Some(want) = striped_parity {
                    if parity != want {
                        continue;
                    }
                }
                if done_words >= STEP_WORDS {
                    if !WRITE_BACK {
                        sfence_if_x86();
                    }
                    if !keep_going() {
                        return false;
                    }
                    add_bytes((done_words * 16) as u64);
                    done_words = 0;
                }
                let lo = (2 * j + parity) * block_words;
                let hi = (2 * (pair_cnt - 1 - j) + parity) * block_words;
                swap_invert_blocks::<WRITE_BACK>(
                    &mut a[lo..lo + block_words],
                    &mut b[hi..hi + block_words],
                );
                done_words += block_words;
            }
            j += stride;
            if j >= pair_cnt {
                j -= pair_cnt;
            }
        }
        if done_words > 0 {
            add_bytes((done_words * 16) as u64);
        }
        if !WRITE_BACK {
            sfence_if_x86();
        }
    }
    // If odd chunk count, mirror the middle chunk from opposite ends in coprime row-miss order.
    if (n & 1) == 1 {
        let mid = &mut chunks[chunk_pairs].0;
        let pair_cnt = (mid.len() / block_words) / 2;
        let half = pair_cnt / 2;
        if half > 0 {
            let stride = pick_scatter_stride(half, stride_div);
            let mut j = 0usize;
            let mut done_words = 0usize;
            for _ in 0..half {
                for parity in 0..2usize {
                    if let Some(want) = striped_parity {
                        if parity != want {
                            continue;
                        }
                    }
                    if done_words >= STEP_WORDS {
                        if !WRITE_BACK {
                            sfence_if_x86();
                        }
                        if !keep_going() {
                            return false;
                        }
                        add_bytes((done_words * 16) as u64);
                        done_words = 0;
                    }
                    let lo = (2 * j + parity) * block_words;
                    let hi = (2 * (pair_cnt - 1 - j) + parity) * block_words;
                    let (l_part, r_part) = mid.split_at_mut(hi);
                    swap_invert_blocks::<WRITE_BACK>(
                        &mut l_part[lo..lo + block_words],
                        &mut r_part[..block_words],
                    );
                    done_words += block_words;
                }
                j += stride;
                if j >= half {
                    j -= half;
                }
            }
            if done_words > 0 {
                add_bytes((done_words * 16) as u64);
            }
            if !WRITE_BACK {
                sfence_if_x86();
            }
        }
    }
    true
}

/// Verify cross-allocation mirrored+inverted image (`even_swapped`, `odd_swapped`).
#[allow(clippy::too_many_arguments)]
fn verify_cross_chunk_mirror(
    chunks: &[(Vec<u64>, u64)],
    base_word: u64,
    block_words: usize,
    even_swapped: bool,
    odd_swapped: bool,
    tag_fn: &impl Fn(u64) -> u64,
    res: &mut StepResult,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
    on_error: &impl Fn(u64, MemMismatch),
) -> bool {
    let n = chunks.len();
    let chunk_pairs = n / 2;
    for ci in 0..n {
        let (chunk, chunk_start) = &chunks[ci];
        let is_mid = (n & 1) == 1 && ci == chunk_pairs;
        let partner_idx = n - 1 - ci;
        let partner_start = chunks[partner_idx].1;
        let pair_cnt = if is_mid {
            (chunk.len() / block_words) / 2
        } else {
            (chunk.len().min(chunks[partner_idx].0.len()) / block_words) / 2
        };
        let active_words = pair_cnt * 2 * block_words;
        for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
            if !keep_going() {
                return false;
            }
            let w0 = si * STEP_WORDS;
            let (bad, first) = verify_gen(step, w0 as u64, w0, &|idx_u64| {
                let idx = idx_u64 as usize;
                if idx < active_words {
                    let blk = idx / block_words;
                    let off = idx % block_words;
                    let j = blk / 2;
                    let parity = blk & 1;
                    let want_swap = if parity == 1 {
                        odd_swapped
                    } else {
                        even_swapped
                    };
                    if want_swap {
                        let partner_j = pair_cnt - 1 - j;
                        if !is_mid || partner_j != j {
                            let partner_blk = 2 * partner_j + parity;
                            let orig = partner_blk * block_words + off;
                            let src_start = if is_mid { *chunk_start } else { partner_start };
                            return !tag_fn(base_word + src_start + orig as u64);
                        }
                    }
                }
                tag_fn(base_word + *chunk_start + idx as u64)
            });
            res.record(ci, bad, first, on_error);
            add_bytes((step.len() * 8) as u64);
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Absolut 16-algorithm execution engine
// ---------------------------------------------------------------------------

#[inline(always)]
fn absolut_tag(seed: u64, global_word: u64) -> u64 {
    splitmix64(global_word ^ seed)
}

#[inline(always)]
fn cacheline_word(i: usize, l0e: u64, l0o: u64, l1e: u64, l1o: u64) -> u64 {
    let odd_line = ((i >> 3) & 1) != 0;
    let odd_word = (i & 1) != 0;
    match (odd_line, odd_word) {
        (false, false) => l0e,
        (false, true) => l0o,
        (true, false) => l1e,
        (true, true) => l1o,
    }
}

/// Execute one inner pass (`sub_pass`) of an Absolut sub-test (`test_id` 0..=15) across all
/// `chunks` owned by one worker. Calls `on_error` IMMEDIATELY upon any single-bit mismatch.
#[allow(clippy::too_many_arguments)]
pub fn run_absolut_step(
    chunks: &mut [(Vec<u64>, u64)],
    base_word: u64,
    cycle: u64,
    step_idx: usize,
    sub_pass: u32,
    test_id: u8,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
    on_error: &impl Fn(u64, MemMismatch),
) -> StepResult {
    let mut res = StepResult::default();
    let iter_idx = cycle.wrapping_mul(256).wrapping_add(sub_pass as u64);
    let seed = splitmix64(
        (cycle << 24)
            ^ ((step_idx as u64) << 16)
            ^ ((sub_pass as u64) << 4)
            ^ (test_id as u64)
            ^ 0xA5A5_5A5A_1234_5678,
    );

    // Rotate complementary DRAM/IMC/FCLK stress modes across sub-passes:
    //   0 = 64B full-chunk coprime row-conflict Cached Write-Back RMW (max ACT+PRE + L3 dirty evictions)
    //   1 = 64B full-chunk coprime row-conflict Non-Temporal RMW (WC-bypass direct DRAM bursts)
    //   2 = 4 KiB cross-chunk simultaneous cold-DRAM Read-Verify + NT Write turnaround (tRDWR/tWRRD/tWTR)
    let rmw_mode = sub_pass % 3;
    let div_idx = (step_idx + sub_pass as usize + cycle as usize) % SCATTER_DIVS.len();
    let mirror_div = SCATTER_DIVS[div_idx];
    let mirror_wb = (sub_pass & 1) == 0;
    match test_id {
        // T1 (FIRST STEP IN SEQUENCE): SimpleTest M2 — Full-chunk 64B/128B coprime row-conflict
        // BL8 64-pin DQ max-toggle + pure AVX2 8-lane Vector LCG (0x77777777 / 0x33333333).
        1 => match sub_pass % 5 {
            0 => run_scatter_pair_torture::<true>(
                chunks,
                CACHELINE_WORDS,
                0x0000_0000_0000_0000,
                0xFFFF_FFFF_FFFF_FFFF,
                div_idx,
                &mut res,
                keep_going,
                add_bytes,
                on_error,
            ),
            1 => run_scatter_vec_lcg_torture::<true>(
                chunks,
                base_word,
                seed,
                CACHELINE_WORDS,
                0x7777_7777,
                0x3333_3333,
                div_idx,
                &mut res,
                keep_going,
                add_bytes,
                on_error,
            ),
            2 => run_scatter_pair_torture::<false>(
                chunks,
                CACHELINE_WORDS,
                0xFFFF_FFFF_FFFF_FFFF,
                0x0000_0000_0000_0000,
                div_idx,
                &mut res,
                keep_going,
                add_bytes,
                on_error,
            ),
            3 => run_scatter_vec_lcg_torture::<false>(
                chunks,
                base_word,
                seed,
                CACHELINE_WORDS,
                0x7777_7777,
                0x3333_3333,
                div_idx,
                &mut res,
                keep_going,
                add_bytes,
                on_error,
            ),
            _ => run_scatter_vec_lcg_torture::<true>(
                chunks,
                base_word,
                seed,
                STRIPE128_WORDS,
                0x7777_7777,
                0x3333_3333,
                div_idx,
                &mut res,
                keep_going,
                add_bytes,
                on_error,
            ),
        },

        // T2: SimpleTest M1 — Rotating variable cache-line DQ polarity flips + 64B row-conflict RMW.
        2 => {
            let (l0e, l0o, l1e, l1o) = match (step_idx + sub_pass as usize) & 3 {
                0 => (
                    0xAAAA_AAAA_5555_5555u64,
                    0x5555_5555_AAAA_AAAAu64,
                    0x5555_5555_AAAA_AAAAu64,
                    0xAAAA_AAAA_5555_5555u64,
                ),
                1 => (
                    0xF0F0_F0F0_0F0F_0F0Fu64,
                    0x0F0F_0F0F_F0F0_F0F0u64,
                    0x0F0F_0F0F_F0F0_F0F0u64,
                    0xF0F0_F0F0_0F0F_0F0Fu64,
                ),
                2 => (
                    0x00FF_00FF_FF00_FF00u64,
                    0xFF00_FF00_00FF_00FFu64,
                    0xFF00_FF00_00FF_00FFu64,
                    0x00FF_00FF_FF00_FF00u64,
                ),
                _ => (
                    0x0000_0000_FFFF_FFFFu64,
                    0xFFFF_FFFF_0000_0000u64,
                    0xFFFF_FFFF_0000_0000u64,
                    0x0000_0000_FFFF_FFFFu64,
                ),
            };
            let gen = |g: u64| cacheline_word(g as usize, l0e, l0o, l1e, l1o);
            run_simple_dual_rmw(
                chunks, base_word, 17, rmw_mode, &gen, &mut res, keep_going, add_bytes, on_error,
            );
        }

        // T3: MirrorMove128 (Param 2) — Coprime row-conflict striped 128B mirror + 64-bit invert.
        3 => {
            let tag = |g: u64| absolut_tag(seed, g);
            for (chunk, chunk_start) in chunks.iter_mut() {
                if !nt_fill_chunk_strided(
                    chunk,
                    base_word + *chunk_start,
                    17,
                    false,
                    &tag,
                    keep_going,
                    add_bytes,
                ) {
                    res.aborted = true;
                    return res;
                }
            }
            // Pass A: Mirror + invert EVEN 128-byte stripes across opposite ends of worker RAM.
            let ok_a = if mirror_wb {
                cross_chunk_mirror::<true>(chunks, 16, Some(0), mirror_div, keep_going, add_bytes)
            } else {
                cross_chunk_mirror::<false>(chunks, 16, Some(0), mirror_div, keep_going, add_bytes)
            };
            if !ok_a {
                res.aborted = true;
                return res;
            }
            if !verify_cross_chunk_mirror(
                chunks, base_word, 16, true, false, &tag, &mut res, keep_going, add_bytes, on_error,
            ) {
                res.aborted = true;
                return res;
            }
            // Pass B: Mirror + invert ODD 128-byte stripes across opposite ends of worker RAM.
            let ok_b = if mirror_wb {
                cross_chunk_mirror::<true>(chunks, 16, Some(1), mirror_div, keep_going, add_bytes)
            } else {
                cross_chunk_mirror::<false>(chunks, 16, Some(1), mirror_div, keep_going, add_bytes)
            };
            if !ok_b {
                res.aborted = true;
                return res;
            }
            if !verify_cross_chunk_mirror(
                chunks, base_word, 16, true, true, &tag, &mut res, keep_going, add_bytes, on_error,
            ) {
                res.aborted = true;
                return res;
            }
        }

        // T4: MirrorMove (Param 4) — Coprime row-conflict 4 KiB page mirror + 64-bit invert.
        4 => {
            let tag = |g: u64| absolut_tag(seed, g);
            for (chunk, chunk_start) in chunks.iter_mut() {
                if !nt_fill_chunk_strided(
                    chunk,
                    base_word + *chunk_start,
                    33,
                    false,
                    &tag,
                    keep_going,
                    add_bytes,
                ) {
                    res.aborted = true;
                    return res;
                }
            }
            let ok_m = if mirror_wb {
                cross_chunk_mirror::<true>(
                    chunks, PAGE_WORDS, None, mirror_div, keep_going, add_bytes,
                )
            } else {
                cross_chunk_mirror::<false>(
                    chunks, PAGE_WORDS, None, mirror_div, keep_going, add_bytes,
                )
            };
            if !ok_m {
                res.aborted = true;
                return res;
            }
            if !verify_cross_chunk_mirror(
                chunks, base_word, PAGE_WORDS, true, true, &tag, &mut res, keep_going, add_bytes,
                on_error,
            ) {
                res.aborted = true;
                return res;
            }
        }

        // T5: MirrorMove128 (Param 1) — Coprime row-conflict contiguous 128B mirror + invert + restore.
        5 => {
            let tag = |g: u64| absolut_tag(seed, g);
            for (chunk, chunk_start) in chunks.iter_mut() {
                if !nt_fill_chunk_strided(
                    chunk,
                    base_word + *chunk_start,
                    29,
                    false,
                    &tag,
                    keep_going,
                    add_bytes,
                ) {
                    res.aborted = true;
                    return res;
                }
            }
            // Mirror + invert pass 1.
            let ok_1 = if mirror_wb {
                cross_chunk_mirror::<true>(chunks, 16, None, mirror_div, keep_going, add_bytes)
            } else {
                cross_chunk_mirror::<false>(chunks, 16, None, mirror_div, keep_going, add_bytes)
            };
            if !ok_1 {
                res.aborted = true;
                return res;
            }
            if !verify_cross_chunk_mirror(
                chunks, base_word, 16, true, true, &tag, &mut res, keep_going, add_bytes, on_error,
            ) {
                res.aborted = true;
                return res;
            }
            // Mirror + invert pass 2 (restores exact original un-inverted data).
            let ok_2 = if mirror_wb {
                cross_chunk_mirror::<true>(chunks, 16, None, mirror_div, keep_going, add_bytes)
            } else {
                cross_chunk_mirror::<false>(chunks, 16, None, mirror_div, keep_going, add_bytes)
            };
            if !ok_2 {
                res.aborted = true;
                return res;
            }
            if !verify_cross_chunk_mirror(
                chunks, base_word, 16, false, false, &tag, &mut res, keep_going, add_bytes,
                on_error,
            ) {
                res.aborted = true;
                return res;
            }
        }

        // T6: SimpleTest M0 (4M) — High-speed BL8 Moving Inversions in 64B/128B coprime row-conflict RMW.
        6 => {
            let pat = match iter_idx & 3 {
                0 => 0x0000_0000_0000_0000u64,
                1 => 0xFFFF_FFFF_FFFF_FFFFu64,
                2 => 0x5555_5555_5555_5555u64,
                _ => 0xAAAA_AAAA_AAAA_AAAAu64,
            };
            match rmw_mode {
                0 => run_scatter_pair_torture::<true>(
                    chunks,
                    CACHELINE_WORDS,
                    pat,
                    !pat,
                    div_idx,
                    &mut res,
                    keep_going,
                    add_bytes,
                    on_error,
                ),
                1 => run_scatter_pair_torture::<false>(
                    chunks,
                    CACHELINE_WORDS,
                    pat,
                    !pat,
                    div_idx,
                    &mut res,
                    keep_going,
                    add_bytes,
                    on_error,
                ),
                _ => run_scatter_pair_torture::<true>(
                    chunks,
                    STRIPE128_WORDS,
                    pat,
                    !pat,
                    div_idx,
                    &mut res,
                    keep_going,
                    add_bytes,
                    on_error,
                ),
            }
        }

        // T7: SimpleTest M0 (8M) — Rotating Single-Bit DQ Aggressor / Victim across all 8 bit lanes.
        7 => {
            let shift = (iter_idx & 7) as u32;
            let gen = |g: u64| {
                let bit =
                    0x0101_0101_0101_0101u64.rotate_left(((g as u32).wrapping_add(shift)) & 7);
                if ((g >> 3) & 1) == 0 {
                    bit
                } else {
                    !bit
                }
            };
            run_simple_dual_rmw(
                chunks, base_word, 19, rmw_mode, &gen, &mut res, keep_going, add_bytes, on_error,
            );
        }

        // T8: SimpleTest M0 (16M) — VDDQ Ground-Bounce / Max di/dt Rail Switching.
        8 => {
            let (l0e, l0o, l1e, l1o) = if (iter_idx & 1) == 0 {
                (
                    0x0000_0000_0000_0000u64,
                    0xFFFF_FFFF_FFFF_FFFFu64,
                    0x00FF_00FF_00FF_00FFu64,
                    0xFF00_FF00_FF00_FF00u64,
                )
            } else {
                (
                    0xFFFF_FFFF_FFFF_FFFFu64,
                    0x0000_0000_0000_0000u64,
                    0xFF00_FF00_FF00_FF00u64,
                    0x00FF_00FF_00FF_00FFu64,
                )
            };
            let gen = |g: u64| cacheline_word(g as usize, l0e, l0o, l1e, l1o);
            run_simple_dual_rmw(
                chunks, base_word, 37, rmw_mode, &gen, &mut res, keep_going, add_bytes, on_error,
            );
        }

        // T9: SimpleTest M0 (32M) — Pure AVX2 Vector PRNG high-entropy 64B/128B row-conflict RMW.
        9 => match rmw_mode {
            0 => run_scatter_vec_lcg_torture::<true>(
                chunks,
                base_word,
                seed,
                CACHELINE_WORDS,
                0x85EB_CA6B,
                0xC2B2_AE35,
                div_idx,
                &mut res,
                keep_going,
                add_bytes,
                on_error,
            ),
            1 => run_scatter_vec_lcg_torture::<false>(
                chunks,
                base_word,
                seed,
                CACHELINE_WORDS,
                0x85EB_CA6B,
                0xC2B2_AE35,
                div_idx,
                &mut res,
                keep_going,
                add_bytes,
                on_error,
            ),
            _ => run_scatter_vec_lcg_torture::<true>(
                chunks,
                base_word,
                seed,
                STRIPE128_WORDS,
                0x85EB_CA6B,
                0xC2B2_AE35,
                div_idx,
                &mut res,
                keep_going,
                add_bytes,
                on_error,
            ),
        },

        // T10: SimpleTest M0 (64M) — Modulo20 Bank-Conflict Scatter at full AVX2 bandwidth.
        10 => {
            let phase = iter_idx % 20;
            let pat_hit = 0xD555_AAAA_5555_D5AAu64 ^ seed;
            let pat_bg = !pat_hit;
            let gen = |g: u64| {
                if (g.wrapping_add(phase)).is_multiple_of(20) {
                    pat_hit ^ g.rotate_left(13)
                } else {
                    pat_bg
                }
            };
            run_simple_dual_rmw(
                chunks, base_word, 129, rmw_mode, &gen, &mut res, keep_going, add_bytes, on_error,
            );
        }

        // T11: SimpleTest M0 (128M) — Opposing-End Cross-Chunk 64 KiB NT BlockMove + inverted copy-back.
        11 => {
            let n_chunks = chunks.len();
            let tag = |g: u64| absolut_tag(seed, g);
            for (chunk, chunk_start) in chunks.iter_mut() {
                if !nt_fill_chunk_strided(
                    chunk,
                    base_word + *chunk_start,
                    33,
                    false,
                    &tag,
                    keep_going,
                    add_bytes,
                ) {
                    res.aborted = true;
                    return res;
                }
            }
            let pairs = n_chunks / 2;
            for j in 0..pairs {
                let hi_idx = n_chunks - 1 - j;
                let (left, right) = chunks.split_at_mut(hi_idx);
                let (src_vec, src_start) = &mut left[j];
                let (dst_vec, _) = &mut right[0];
                let common = src_vec.len().min(dst_vec.len());
                let blk = 8192usize; // 64 KiB
                let mut off = 0usize;
                let mut done = 0usize;
                while off < common {
                    if done >= STEP_WORDS {
                        if !keep_going() {
                            res.aborted = true;
                            return res;
                        }
                        add_bytes((done * 16) as u64);
                        done = 0;
                    }
                    let end = (off + blk).min(common);
                    nt_copy_slice(&src_vec[off..end], &mut dst_vec[off..end], false);
                    done += end - off;
                    off = end;
                }
                if done > 0 {
                    add_bytes((done * 16) as u64);
                }
                for (si, step) in dst_vec[..common].chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *src_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, &tag);
                    res.record(hi_idx, bad, first, on_error);
                    add_bytes((step.len() * 8) as u64);
                }
                off = 0;
                done = 0;
                while off < common {
                    if done >= STEP_WORDS {
                        if !keep_going() {
                            res.aborted = true;
                            return res;
                        }
                        add_bytes((done * 16) as u64);
                        done = 0;
                    }
                    let end = (off + blk).min(common);
                    nt_copy_slice(&dst_vec[off..end], &mut src_vec[off..end], true);
                    done += end - off;
                    off = end;
                }
                if done > 0 {
                    add_bytes((done * 16) as u64);
                }
                let inv_tag = |g: u64| !absolut_tag(seed, g);
                for (si, step) in src_vec[..common].chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *src_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, &inv_tag);
                    res.record(j, bad, first, on_error);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            if (n_chunks & 1) == 1 {
                let ci = pairs;
                let (chunk, chunk_start) = &mut chunks[ci];
                let half = chunk.len() / 2;
                if half > 0 {
                    let (lo, hi) = chunk.split_at_mut(half);
                    nt_copy_slice(lo, &mut hi[..half], true);
                    add_bytes((half * 16) as u64);
                    let inv_tag = |g: u64| !absolut_tag(seed, g);
                    let (bad, first) =
                        verify_gen(&hi[..half], base_word + *chunk_start, half, &inv_tag);
                    res.record(ci, bad, first, on_error);
                    add_bytes((half * 8) as u64);
                }
            }
        }

        // T12: SimpleTest M0 (256M) — Address-Coupled Forward/Backward Inverse Walk (CMD/ADDR, 1T/2T CR).
        12 => {
            let magic = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let gen = |g: u64| g ^ magic ^ (g.rotate_left(29));
            run_simple_dual_rmw(
                chunks, base_word, 41, rmw_mode, &gen, &mut res, keep_going, add_bytes, on_error,
            );
        }

        // T13: SimpleTest M0 (512M) — Walking 64-bit 1s & 0s + 64B row-conflict RMW.
        13 => {
            let gen = |g: u64| {
                let bit = 1u64.rotate_left(((g + iter_idx) & 63) as u32);
                if (iter_idx & 1) == 0 {
                    bit
                } else {
                    !bit
                }
            };
            run_simple_dual_rmw(
                chunks, base_word, 53, rmw_mode, &gen, &mut res, keep_going, add_bytes, on_error,
            );
        }

        // T14: Rowhammer / Bit-Fade Retention — high/zero charge row bands + double-sided clflush.
        14 => {
            let gen = |g: u64| {
                let band = (g >> 10) & 1;
                let base_v = if band == 0 {
                    0xFFFF_FFFF_FFFF_FFFFu64
                } else {
                    0x0000_0000_0000_0000u64
                };
                base_v ^ (0x0001_0001_0001_0001u64.wrapping_mul((g ^ seed) & 0xF))
            };
            for (chunk, chunk_start) in chunks.iter_mut() {
                if !nt_fill_chunk_strided(
                    chunk,
                    base_word + *chunk_start,
                    33,
                    false,
                    &gen,
                    keep_going,
                    add_bytes,
                ) {
                    res.aborted = true;
                    return res;
                }
            }
            for (chunk, _) in chunks.iter() {
                let len = chunk.len();
                let stride = 16384usize; // 128 KiB
                if len >= stride * 3 {
                    let mut p = 0usize;
                    let limit = len.min(stride * 32);
                    while p + stride * 2 < limit {
                        if !keep_going() {
                            res.aborted = true;
                            return res;
                        }
                        let a0 = unsafe { chunk.as_ptr().add(p) };
                        let a1 = unsafe { chunk.as_ptr().add(p + stride * 2) };
                        for _ in 0..256 {
                            unsafe {
                                let _ = std::ptr::read_volatile(a0);
                                let _ = std::ptr::read_volatile(a1);
                                clflush_word(a0);
                                clflush_word(a1);
                            }
                        }
                        p += stride * 3;
                    }
                }
            }
            for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
                if !verify_chunk_strided(
                    ci,
                    chunk,
                    base_word + *chunk_start,
                    33,
                    true,
                    &gen,
                    &mut res,
                    keep_going,
                    add_bytes,
                    on_error,
                ) {
                    res.aborted = true;
                    return res;
                }
            }
        }

        // T0: RefreshStable — NT alternating checkerboard + aggressor rowhammer + retention verify.
        0 => {
            let (e0, o0) = if (iter_idx & 1) == 0 {
                (0xAAAA_AAAA_AAAA_AAAAu64, 0x5555_5555_5555_5555u64)
            } else {
                (0x5555_5555_5555_5555u64, 0xAAAA_AAAA_AAAA_AAAAu64)
            };
            for (chunk, _) in chunks.iter_mut() {
                for step in chunk.chunks_mut(STEP_WORDS) {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    nt_fill_pair(step, e0, o0);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            for (chunk, _) in chunks.iter() {
                let len = chunk.len();
                let row_words = 8192usize; // 64 KiB stride across DRAM rows/banks
                if len >= row_words * 4 {
                    let mut p = 0usize;
                    let limit = len.min(row_words * 64);
                    while p + row_words * 2 < limit {
                        if !keep_going() {
                            res.aborted = true;
                            return res;
                        }
                        let a0 = unsafe { chunk.as_ptr().add(p) };
                        let a1 = unsafe { chunk.as_ptr().add(p + row_words * 2) };
                        for _ in 0..128 {
                            unsafe {
                                let _ = std::ptr::read_volatile(a0);
                                let _ = std::ptr::read_volatile(a1);
                                clflush_word(a0);
                                clflush_word(a1);
                            }
                        }
                        p += row_words * 4;
                    }
                }
            }
            for (ci, (chunk, _)) in chunks.iter().enumerate() {
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let (bad, first) = verify_pair(step, e0, o0, si * STEP_WORDS);
                    res.record(ci, bad, first, on_error);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T15 (default, runs 8x per cycle): SimpleTest M2 — 0xFFFFFFFF / 0x55555555 64B coprime RMW.
        _ => match rmw_mode {
            0 => run_scatter_pair_torture::<true>(
                chunks,
                CACHELINE_WORDS,
                0xFFFF_FFFF_FFFF_FFFF,
                0x5555_5555_5555_5555,
                div_idx,
                &mut res,
                keep_going,
                add_bytes,
                on_error,
            ),
            1 => run_scatter_vec_lcg_torture::<true>(
                chunks,
                base_word,
                seed,
                CACHELINE_WORDS,
                0x9E37_79B9,
                0x5555_5555,
                div_idx,
                &mut res,
                keep_going,
                add_bytes,
                on_error,
            ),
            _ => run_scatter_pair_torture::<false>(
                chunks,
                CACHELINE_WORDS,
                0x0000_0000_0000_0000,
                0xFFFF_FFFF_FFFF_FFFF,
                div_idx,
                &mut res,
                keep_going,
                add_bytes,
                on_error,
            ),
        },
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    const GB: u64 = 1 << 30;

    #[test]
    fn target_respects_floor_and_cap() {
        let t = compute_target_bytes(16 * GB, 10 * GB, 95, 95).unwrap();
        assert!(t <= 10 * GB - FREE_FLOOR_BYTES, "keeps reserve free");
        assert!(6 * GB + t <= 16 * GB / 100 * 95 + 1, "<= 95% total");
        assert!(t > 9 * GB, "should allocate >9 GiB");
        let t100 = compute_target_bytes(16 * GB, 10 * GB, 100, 100).unwrap();
        assert_eq!(t100, 10 * GB - MAX_FLOOR_BYTES);
        assert_eq!(
            compute_topup_bytes(16 * GB, 2 * GB, 100, 100),
            2 * GB - MAX_FLOOR_BYTES
        );
        let t = compute_target_bytes(16 * GB, 10 * GB, 80, 50).unwrap();
        assert!(6 * GB + t <= 16 * GB / 100 * 50 + 1);
        assert!(compute_target_bytes(16 * GB, 150 << 20, 50, 95).is_err());
        assert!(compute_target_bytes(16 * GB, 8 * GB, 0, 95).is_err());
        assert!(compute_target_bytes(16 * GB, 8 * GB, 101, 100).is_err());
    }

    #[test]
    fn sweep_never_violates_rules() {
        for total_gb in [4u64, 8, 16, 32, 64] {
            let total = total_gb * GB;
            for avail_pct in (5..=95).step_by(5) {
                let available = total / 100 * avail_pct;
                for pct in [10u8, 50, 80, 90, 95, 99, 100] {
                    for maxp in [50u8, 80, 90, 95, 99, 100] {
                        if let Ok(t) = compute_target_bytes(total, available, pct, maxp) {
                            let floor = reserve_floor_bytes(pct, maxp);
                            assert!(t >= MIN_TARGET_BYTES);
                            assert!(available - t >= floor);
                            let used = total - available;
                            if maxp < 100 {
                                assert!(used + t <= total / 100 * maxp as u64);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn patterns_roundtrip() {
        let mut buf = vec![0u64; 16_384];
        for pass in 0..10u64 {
            let p = Pattern::for_pass(pass);
            fill_slice(&mut buf, p, pass, 1000);
            assert_eq!(verify_slice(&buf, p, pass, 1000), (0, None), "pass {pass}");
        }
    }

    #[test]
    fn walking_ones_and_zeros_have_exact_bit_counts() {
        for w in 0..64u64 {
            assert_eq!(expected(Pattern::WalkingOnes, 0, w).count_ones(), 1);
            assert_eq!(expected(Pattern::WalkingOnes, 1, w).count_zeros(), 1);
        }
    }

    #[test]
    fn verifier_detects_injected_fault() {
        let mut buf = vec![0u64; 4096];
        for pass in 0..5u64 {
            let p = Pattern::for_pass(pass);
            fill_slice(&mut buf, p, pass, 0);
            buf[1234] ^= 1 << 17;
            assert_eq!(
                verify_slice(&buf, p, pass, 0),
                (1, Some(1234)),
                "pass {pass}"
            );
        }
    }

    #[test]
    fn absolut_all_16_tests_roundtrip_even_odd_and_mixed_chunks() {
        // Test 1 chunk, 2 equal chunks, and 3 mismatched chunks (simulating 64M + 16M + 4M top-up).
        let layouts: &[&[usize]] = &[
            &[32_768],
            &[32_768, 32_768],
            &[32_768, 12_345, 24_576],
            &[519, 17],
        ];
        for sizes in layouts {
            let mut off = 0u64;
            let mut chunks: Vec<(Vec<u64>, u64)> = sizes
                .iter()
                .map(|&n| {
                    let s = off;
                    off += n as u64;
                    (vec![0u64; n], s)
                })
                .collect();
            for test_id in 0..=15u8 {
                for sub_pass in 0..5u32 {
                    let errs = RefCell::new(Vec::new());
                    let res = run_absolut_step(
                        &mut chunks,
                        1u64 << 40,
                        0,
                        test_id as usize,
                        sub_pass,
                        test_id,
                        &|| true,
                        &|_| {},
                        &|n, mm| errs.borrow_mut().push((n, mm)),
                    );
                    assert!(
                        !res.aborted,
                        "test {test_id} p{sub_pass} aborted unexpectedly"
                    );
                    assert_eq!(
                        res.bad_words,
                        0,
                        "layout {sizes:?} test {test_id} p{sub_pass} ({}) reported errors: {:?}",
                        absolut_test_name(test_id),
                        res.first
                    );
                    assert!(errs.borrow().is_empty());
                }
            }
        }
    }

    #[test]
    fn vec_lcg_avx2_matches_scalar_on_unaligned_and_odd_slices() {
        let mut backing = vec![0u64; 523];
        for start in 0..4usize {
            for invert in [false, true] {
                let sub = &mut backing[start..start + 517];
                fill_vec_lcg_slice::<false>(sub, 13, 0x1234_5678, 0x7777_7777, 0x3333_3333, invert);
                for (i, &w) in sub.iter().enumerate() {
                    let exp = vec_lcg_word(0x1234_5678, 0x7777_7777, 0x3333_3333, 13 + i, invert);
                    assert_eq!(w, exp, "start={start} invert={invert} i={i}");
                }
                assert_eq!(
                    verify_vec_lcg_slice(sub, 13, 0x1234_5678, 0x7777_7777, 0x3333_3333, invert).0,
                    0
                );
            }
        }
    }

    #[test]
    fn scatter_rmw_and_mirror_detect_single_bit_fault_with_zero_cascade() {
        // 1) 64B Cached & NT pair scatter RMW detect a single flipped bit and write clean !pat.
        for use_wb in [true, false] {
            let mut buf = vec![0u64; 4099];
            fill_pair_cached(&mut buf, 0, u64::MAX);
            buf[2117] ^= 1u64 << 21;
            let mut res = StepResult::default();
            let caught = RefCell::new(Vec::new());
            let ok = if use_wb {
                scatter_pair_rmw_chunk::<true>(
                    0,
                    &mut buf,
                    CACHELINE_WORDS,
                    0,
                    u64::MAX,
                    13,
                    false,
                    &mut res,
                    &|| true,
                    &|_| {},
                    &|n, mm| caught.borrow_mut().push((n, mm)),
                )
            } else {
                scatter_pair_rmw_chunk::<false>(
                    0,
                    &mut buf,
                    CACHELINE_WORDS,
                    0,
                    u64::MAX,
                    13,
                    false,
                    &mut res,
                    &|| true,
                    &|_| {},
                    &|n, mm| caught.borrow_mut().push((n, mm)),
                )
            };
            assert!(ok);
            assert_eq!(res.bad_words, 1);
            let mm = caught.borrow()[0].1;
            assert_eq!(mm.word_idx, 2117);
            assert_eq!(mm.expected ^ mm.actual, 1u64 << 21);
            // Zero-cascade invariant: subsequent pass verifies clean without repeating the old fault.
            assert_eq!(verify_pair(&buf, u64::MAX, 0, 0).0, 0);
        }

        // 2) 64B Vector-LCG scatter RMW catches a single flipped bit and writes clean !lcg.
        let mut lbuf = vec![0u64; 4103];
        fill_vec_lcg_slice::<true>(&mut lbuf, 0, 0xCAFE_1234, 0x7777_7777, 0x3333_3333, false);
        lbuf[1907] ^= 1u64 << 23;
        let mut lres = StepResult::default();
        let lcaught = RefCell::new(Vec::new());
        assert!(scatter_vec_lcg_rmw_chunk::<true>(
            2,
            &mut lbuf,
            CACHELINE_WORDS,
            0xCAFE_1234,
            0x7777_7777,
            0x3333_3333,
            false,
            17,
            false,
            &mut lres,
            &|| true,
            &|_| {},
            &|n, mm| lcaught.borrow_mut().push((n, mm)),
        ));
        assert_eq!(lres.bad_words, 1);
        let lmm = lcaught.borrow()[0].1;
        assert_eq!(lmm.chunk_idx, 2);
        assert_eq!(lmm.word_idx, 1907);
        assert_eq!(lmm.expected ^ lmm.actual, 1u64 << 23);
        assert_eq!(
            verify_vec_lcg_slice(&lbuf, 0, 0xCAFE_1234, 0x7777_7777, 0x3333_3333, true).0,
            0
        );

        // 3) Cross-chunk coprime mirror verifier catches a single bit flip and invokes on_error immediately.
        let tag = |g: u64| absolut_tag(0xCAFE_BABE, g);
        let mut chunks = vec![(vec![0u64; 2048], 0u64), (vec![0u64; 2048], 2048u64)];
        for (c, s) in &mut chunks {
            nt_fill_gen_nofence(c, *s, &tag);
        }
        sfence_if_x86();
        assert!(cross_chunk_mirror::<true>(
            &mut chunks,
            16,
            None,
            13,
            &|| true,
            &|_| {}
        ));
        chunks[1].0[513] ^= 1u64 << 9;
        let mut res = StepResult::default();
        let caught = RefCell::new(Vec::new());
        assert!(verify_cross_chunk_mirror(
            &chunks,
            0,
            16,
            true,
            true,
            &tag,
            &mut res,
            &|| true,
            &|_| {},
            &|n, mm| caught.borrow_mut().push((n, mm)),
        ));
        assert_eq!(res.bad_words, 1);
        assert_eq!(caught.borrow().len(), 1);
        let mm = caught.borrow()[0].1;
        assert_eq!(mm.chunk_idx, 1);
        assert_eq!(mm.word_idx, 513);
        assert_eq!(mm.expected ^ mm.actual, 1u64 << 9);
    }

    #[test]
    fn absolut_unaligned_slices_and_single_bit_faults_detected() {
        let mut backing = vec![0u64; 519];
        let sub = &mut backing[1..518];
        nt_fill_pair(sub, 0x7777_7777_3333_3333, 0x3333_3333_7777_7777);
        assert_eq!(
            verify_pair(sub, 0x7777_7777_3333_3333, 0x3333_3333_7777_7777, 0).0,
            0
        );
        sub[211] ^= 1u64 << 39;
        let (bad, first) = verify_pair(sub, 0x7777_7777_3333_3333, 0x3333_3333_7777_7777, 0);
        assert_eq!(bad, 1);
        let (idx, exp, got) = first.unwrap();
        assert_eq!(idx, 211);
        assert_eq!(exp ^ got, 1u64 << 39);
    }

    #[test]
    fn absolut_sequence_starts_with_t1_and_matches_32_step_layout() {
        assert_eq!(ABSOLUT_SEQUENCE.len(), 32);
        assert_eq!(ABSOLUT_SEQUENCE[0], 1);
        assert_eq!(ABSOLUT_SEQUENCE[31], 0);
        assert_eq!(
            &ABSOLUT_SEQUENCE[..31],
            &[
                1, 4, 6, 15, 3, 2, 7, 15, 5, 2, 8, 15, 4, 2, 9, 15, 3, 2, 10, 15, 5, 2, 11, 15, 4,
                2, 12, 15, 5, 14, 15
            ]
        );
    }
}
