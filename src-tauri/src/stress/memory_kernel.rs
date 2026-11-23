//! Memory stress engine: Absolut 32-step DRAM/IMC suite + classic pattern runner.
//!
//! Every test is 100% deterministic and bit-exact (zero subjectivity, zero false positives).
//! Absolut starts immediately on T1 (`SimpleTest` Mode 2 LCG `0x77777777 / 0x33333333`) using
//! AVX2 256-bit non-temporal streaming stores (`_mm256_stream_si256` + `_mm_sfence`),
//! non-temporal prefetch loads (`_MM_HINT_NTA`), bijective bank-strided 4 KiB DRAM row-conflict
//! page jumping, simultaneous cold-DRAM Read-Verify + opposing-chunk NT-Write bus turnarounds
//! (`tRDWR`/`tWRRD`/`tWTR`/`ODT`), cross-allocation non-temporal `MirrorMove128` / `MirrorMove 4K`
//! with full 64-bit transit inversion, and `_mm_clflush` aggressor rowhammer sweeps.

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
/// 128-byte dual-cache-line stripe in 64-bit words (used for 100% DRAM row-miss bank thrashing).
const STRIPE128_WORDS: usize = 16;
/// 4 MiB superblock in 4 KiB pages (1024 pages fit inside L2 TLB while 16T x 4 MiB = 64 MiB > L3).
const BLOCK4M_PAGES: usize = 1024;
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

/// Dual 32-bit LCG generator implementing Absolut T1 (`Pattern Param0=0x77777777, Param1=0x33333333`)
/// with cross-lane rotation and alternating odd-word inversion for maximum 64-bit DQ crosstalk/SSN.
#[inline(always)]
fn absolut_lcg_t1(seed: u64, g: u64) -> u64 {
    let mut lo = ((g ^ seed) as u32)
        .wrapping_mul(0x7777_7777)
        .wrapping_add(0x3333_3333);
    lo = lo
        .rotate_left(13)
        .wrapping_mul(0x7777_7777)
        .wrapping_add(0x3333_3333);
    let mut hi = (((g >> 32) ^ (seed >> 32) ^ (g.wrapping_mul(0x9E37_79B9) >> 16)) as u32)
        .wrapping_mul(0x7777_7777)
        .wrapping_add(0x3333_3333);
    hi = hi
        .rotate_left(17)
        .wrapping_mul(0x7777_7777)
        .wrapping_add(0x3333_3333);
    let w = (lo as u64) | ((hi as u64) << 32);
    if (g & 1) == 0 {
        w
    } else {
        !w
    }
}

/// Dual 32-bit LCG generator implementing Absolut T15 (`Pattern Param0=0xFFFFFFFF, Param1=0x55555555`)
/// with alternating 64-byte cache-line inversion so adjacent beats hammer both `0x55/0xAA` and random rails.
#[inline(always)]
fn absolut_lcg_t15(seed: u64, g: u64) -> u64 {
    let x = splitmix64(g ^ seed);
    let lo = (x as u32)
        .wrapping_mul(0xFFFF_FFFF)
        .wrapping_add(0x5555_5555);
    let hi = ((x >> 32) as u32)
        .wrapping_mul(0xFFFF_FFFF)
        .wrapping_add(0x5555_5555);
    let w = (lo as u64) | ((hi as u64) << 32);
    if ((g >> 3) & 1) == 0 {
        w
    } else {
        !w
    }
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

/// Swap two disjoint equal-length blocks AND bitwise-invert every word using non-temporal AVX2
/// stores (`_mm256_stream_si256`), preventing L3 cache masking and flipping all 64 DQ lanes in transit.
#[inline]
fn nt_swap_invert_blocks(a: &mut [u64], b: &mut [u64]) {
    debug_assert_eq!(a.len(), b.len());
    if has_avx2() && a.len() >= 4 {
        unsafe { nt_swap_invert_blocks_avx2(a, b) };
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
unsafe fn nt_swap_invert_blocks_avx2(a: &mut [u64], b: &mut [u64]) {
    use std::arch::x86_64::*;
    let len = a.len();
    let ap = a.as_mut_ptr();
    let bp = b.as_mut_ptr();
    let aligned_both = ((ap as usize) & 31) == 0 && ((bp as usize) & 31) == 0;
    let ones = _mm256_set1_epi64x(-1i64);
    let mut i = 0usize;
    let end4 = len & !3;
    if aligned_both {
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

/// Fill `chunk` with 100% DRAM row-miss 128-byte stripes jumping +68 KiB across 4 MiB TLB-resident
/// superblocks. When `WRITE_BACK=true`, uses cached write-back stores to trigger RFO + async L3
/// dirty evictions (hammers Infinity Fabric / FCLK / IMC queues); when `false`, uses NT WC-bypass.
#[allow(clippy::too_many_arguments)]
fn fill_chunk_row_thrash_128b<const WRITE_BACK: bool>(
    chunk: &mut [u64],
    chunk_base: u64,
    stride_hint: usize,
    reverse: bool,
    gen: &impl Fn(u64) -> u64,
    keep_going: &impl Fn() -> bool,
    add_bytes: &impl Fn(u64),
) -> bool {
    let total_pages = chunk.len() / PAGE_WORDS;
    let mut page_off = 0usize;
    let mut done_words = 0usize;
    while page_off < total_pages {
        let blk_pages = (total_pages - page_off).min(BLOCK4M_PAGES);
        let blk_base_word = page_off * PAGE_WORDS;
        let stride = coprime_stride(blk_pages, stride_hint);
        let stripes_per_page = PAGE_WORDS / STRIPE128_WORDS; // 32
        for s_idx in 0..stripes_per_page {
            let s = if reverse {
                stripes_per_page - 1 - s_idx
            } else {
                s_idx
            };
            let s_off = s * STRIPE128_WORDS;
            let mut p = if reverse { blk_pages - 1 } else { 0 };
            for _ in 0..blk_pages {
                if done_words >= STEP_WORDS {
                    if !WRITE_BACK {
                        sfence_if_x86();
                    }
                    if !keep_going() {
                        return false;
                    }
                    add_bytes((done_words * 8) as u64);
                    done_words = 0;
                }
                let w0 = blk_base_word + p * PAGE_WORDS + s_off;
                fill_gen_nofence::<WRITE_BACK>(
                    &mut chunk[w0..w0 + STRIPE128_WORDS],
                    chunk_base + w0 as u64,
                    gen,
                );
                done_words += STRIPE128_WORDS;
                if reverse {
                    p = if p >= stride {
                        p - stride
                    } else {
                        p + blk_pages - stride
                    };
                } else {
                    p += stride;
                    if p >= blk_pages {
                        p -= blk_pages;
                    }
                }
            }
        }
        page_off += blk_pages;
    }
    if done_words > 0 {
        add_bytes((done_words * 8) as u64);
    }
    let tail = total_pages * PAGE_WORDS;
    if tail < chunk.len() {
        fill_gen_nofence::<WRITE_BACK>(&mut chunk[tail..], chunk_base + tail as u64, gen);
        add_bytes(((chunk.len() - tail) * 8) as u64);
    }
    if !WRITE_BACK {
        sfence_if_x86();
    }
    true
}

/// Verify `chunk` with 100% DRAM row-miss 128-byte stripes across 4 MiB superblocks + immediate `on_error`.
#[allow(clippy::too_many_arguments)]
fn verify_chunk_row_thrash_128b(
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
    let total_pages = chunk.len() / PAGE_WORDS;
    let mut page_off = 0usize;
    let mut done_words = 0usize;
    while page_off < total_pages {
        let blk_pages = (total_pages - page_off).min(BLOCK4M_PAGES);
        let blk_base_word = page_off * PAGE_WORDS;
        let stride = coprime_stride(blk_pages, stride_hint);
        let stripes_per_page = PAGE_WORDS / STRIPE128_WORDS;
        for s_idx in 0..stripes_per_page {
            let s = if reverse {
                stripes_per_page - 1 - s_idx
            } else {
                s_idx
            };
            let s_off = s * STRIPE128_WORDS;
            let mut p = if reverse { blk_pages - 1 } else { 0 };
            for _ in 0..blk_pages {
                if done_words >= STEP_WORDS {
                    if !keep_going() {
                        return false;
                    }
                    add_bytes((done_words * 8) as u64);
                    done_words = 0;
                }
                let w0 = blk_base_word + p * PAGE_WORDS + s_off;
                let (bad, first) = verify_gen(
                    &chunk[w0..w0 + STRIPE128_WORDS],
                    chunk_base + w0 as u64,
                    w0,
                    gen,
                );
                res.record(chunk_idx, bad, first, on_error);
                done_words += STRIPE128_WORDS;
                if reverse {
                    p = if p >= stride {
                        p - stride
                    } else {
                        p + blk_pages - stride
                    };
                } else {
                    p += stride;
                    if p >= blk_pages {
                        p -= blk_pages;
                    }
                }
            }
        }
        page_off += blk_pages;
    }
    if done_words > 0 {
        add_bytes((done_words * 8) as u64);
    }
    let tail = total_pages * PAGE_WORDS;
    if tail < chunk.len() {
        let (bad, first) = verify_gen(&chunk[tail..], chunk_base + tail as u64, tail, gen);
        res.record(chunk_idx, bad, first, on_error);
        add_bytes(((chunk.len() - tail) * 8) as u64);
    }
    true
}

/// Core Absolut SimpleTest 3-phase torture runner:
///   Phase 1: Bank-strided fill of ALL chunks with `gen_fwd` (total worker RAM >> L3 -> cold DRAM).
///   Phase 2: Cross-chunk simultaneous cold-DRAM Read-Verify (`chunks[k]`) + cold-DRAM Write
///            (`chunks[k-1]` with `!gen_fwd`) — hammers `tRDWR`, `tWRRD`, `tWTR`, `tRCD`, `tRP`,
///            `tFAW`, `tRRD`, ODT, and Infinity Fabric / FCLK simultaneously.
///   Phase 3: Reverse bank-strided NTA verify of `!gen_fwd` across ALL chunks.
///
/// `rmw_mode` rotates through 3 complementary hardware stress modes across sub-passes:
///   * `0`: 128B 100%-row-miss bank thrash + cached Write-Back RFO (max FCLK / L3 dirty writeback / IMC queue collisions)
///   * `1`: 128B 100%-row-miss bank thrash + Non-Temporal WC-bypass (`_mm256_stream_si256` direct DRAM bursts)
///   * `2`: 4 KiB bank-strided max-bandwidth streaming + Non-Temporal (peak sustained DRAM heat & rail droop)
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
    let row_thrash_128b = rmw_mode != 2;
    let write_back_rfo = rmw_mode == 0;

    // Phase 1: Bank-strided fill across all chunks (total allocation >> L3 so every line reaches DRAM).
    for (chunk, chunk_start) in chunks.iter_mut() {
        let ok = if !row_thrash_128b {
            nt_fill_chunk_strided(
                chunk,
                base_word + *chunk_start,
                stride_hint,
                false,
                gen_fwd,
                keep_going,
                add_bytes,
            )
        } else if write_back_rfo {
            fill_chunk_row_thrash_128b::<true>(
                chunk,
                base_word + *chunk_start,
                stride_hint,
                false,
                gen_fwd,
                keep_going,
                add_bytes,
            )
        } else {
            fill_chunk_row_thrash_128b::<false>(
                chunk,
                base_word + *chunk_start,
                stride_hint,
                false,
                gen_fwd,
                keep_going,
                add_bytes,
            )
        };
        if !ok {
            res.aborted = true;
            return;
        }
    }

    // Phase 2: Simultaneous cross-chunk cold-DRAM Read-Verify + cold-DRAM Write invert.
    if n == 1 {
        let (chunk, chunk_start) = &mut chunks[0];
        let b = base_word + *chunk_start;
        let ok_v = if row_thrash_128b {
            verify_chunk_row_thrash_128b(
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
            )
        } else {
            verify_chunk_strided(
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
            )
        };
        if !ok_v {
            res.aborted = true;
            return;
        }
        let ok_w = if !row_thrash_128b {
            nt_fill_chunk_strided(chunk, b, stride_hint, true, &gen_inv, keep_going, add_bytes)
        } else if write_back_rfo {
            fill_chunk_row_thrash_128b::<true>(
                chunk,
                b,
                stride_hint,
                true,
                &gen_inv,
                keep_going,
                add_bytes,
            )
        } else {
            fill_chunk_row_thrash_128b::<false>(
                chunk,
                b,
                stride_hint,
                true,
                &gen_inv,
                keep_going,
                add_bytes,
            )
        };
        if !ok_w {
            res.aborted = true;
            return;
        }
    } else {
        // Step 2a: Verify chunks[0] so it becomes a cold verified target for simultaneous write.
        {
            let (c0, s0) = &chunks[0];
            let ok = if row_thrash_128b {
                verify_chunk_row_thrash_128b(
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
                )
            } else {
                verify_chunk_strided(
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
                )
            };
            if !ok {
                res.aborted = true;
                return;
            }
        }
        // Step 2b: For k = 1..n, read-verify chunks[k] while simultaneously writing !gen_fwd to chunks[k-1].
        for k in 1..n {
            let (left, right) = chunks.split_at_mut(k);
            let (dst_vec, dst_start) = &mut left[k - 1];
            let (src_vec, src_start) = &right[0];
            let dst_base = base_word + *dst_start;
            let src_base = base_word + *src_start;
            let common = src_vec.len().min(dst_vec.len());
            let common_pages = common / PAGE_WORDS;
            if common_pages > 0 {
                if row_thrash_128b {
                    // 128-byte simultaneous cold-DRAM READ + WRITE jumping across DRAM rows!
                    let mut page_off = 0usize;
                    let mut done_words = 0usize;
                    let stripes_per_page = PAGE_WORDS / STRIPE128_WORDS;
                    while page_off < common_pages {
                        let blk_pages = (common_pages - page_off).min(BLOCK4M_PAGES);
                        let blk_base_word = page_off * PAGE_WORDS;
                        let stride = coprime_stride(blk_pages, stride_hint);
                        for s_idx in 0..stripes_per_page {
                            let s_off = (stripes_per_page - 1 - s_idx) * STRIPE128_WORDS;
                            let mut p = blk_pages - 1;
                            for _ in 0..blk_pages {
                                if done_words >= STEP_WORDS {
                                    if !write_back_rfo {
                                        sfence_if_x86();
                                    }
                                    if !keep_going() {
                                        res.aborted = true;
                                        return;
                                    }
                                    add_bytes((done_words * 16) as u64);
                                    done_words = 0;
                                }
                                let w0 = blk_base_word + p * PAGE_WORDS + s_off;
                                let (bad, first) = if write_back_rfo {
                                    dual_verify_and_write_slice::<true>(
                                        &src_vec[w0..w0 + STRIPE128_WORDS],
                                        src_base + w0 as u64,
                                        w0,
                                        gen_fwd,
                                        &mut dst_vec[w0..w0 + STRIPE128_WORDS],
                                        dst_base + w0 as u64,
                                        &gen_inv,
                                    )
                                } else {
                                    dual_verify_and_write_slice::<false>(
                                        &src_vec[w0..w0 + STRIPE128_WORDS],
                                        src_base + w0 as u64,
                                        w0,
                                        gen_fwd,
                                        &mut dst_vec[w0..w0 + STRIPE128_WORDS],
                                        dst_base + w0 as u64,
                                        &gen_inv,
                                    )
                                };
                                res.record(k, bad, first, on_error);
                                done_words += STRIPE128_WORDS;
                                p = if p >= stride {
                                    p - stride
                                } else {
                                    p + blk_pages - stride
                                };
                            }
                        }
                        page_off += blk_pages;
                    }
                    if done_words > 0 {
                        add_bytes((done_words * 16) as u64);
                    }
                    if !write_back_rfo {
                        sfence_if_x86();
                    }
                } else {
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
        // Step 2c: Write !gen_fwd to the final chunk chunks[n-1] (already verified in Step 2b).
        let (last_vec, last_start) = &mut chunks[n - 1];
        let ok = if !row_thrash_128b {
            nt_fill_chunk_strided(
                last_vec,
                base_word + *last_start,
                stride_hint,
                true,
                &gen_inv,
                keep_going,
                add_bytes,
            )
        } else if write_back_rfo {
            fill_chunk_row_thrash_128b::<true>(
                last_vec,
                base_word + *last_start,
                stride_hint,
                true,
                &gen_inv,
                keep_going,
                add_bytes,
            )
        } else {
            fill_chunk_row_thrash_128b::<false>(
                last_vec,
                base_word + *last_start,
                stride_hint,
                true,
                &gen_inv,
                keep_going,
                add_bytes,
            )
        };
        if !ok {
            res.aborted = true;
            return;
        }
    }

    // Phase 3: Reverse bank-strided verify of !gen_fwd across all chunks (cold from physical DRAM).
    for (ci, (chunk, chunk_start)) in chunks.iter().enumerate().rev() {
        let ok = if row_thrash_128b {
            verify_chunk_row_thrash_128b(
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
            )
        } else {
            verify_chunk_strided(
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
            )
        };
        if !ok {
            res.aborted = true;
            return;
        }
    }
}

/// Cross-allocation non-temporal MirrorMove with full 64-bit transit inversion.
///
/// Pairs opposing-end chunks (`chunks[c]` <-> `chunks[n - 1 - c]`) so every 128-byte stripe or
/// 4 KiB page crosses the full worker allocation (hundreds of MiB apart, never trapped in L3).
/// Uses parity-preserving mirrored block indices `2*(pairs - 1 - j) + parity` so striped even/odd
/// passes (`Some(0)`, `Some(1)`) are cleanly involutive on both opposing chunk pairs and odd middle chunks.
#[allow(clippy::too_many_arguments)]
fn cross_chunk_nt_mirror(
    chunks: &mut [(Vec<u64>, u64)],
    block_words: usize,
    striped_parity: Option<usize>,
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
        let mut done_words = 0usize;
        for j in 0..pair_cnt {
            for parity in 0..2usize {
                if let Some(want) = striped_parity {
                    if parity != want {
                        continue;
                    }
                }
                if done_words >= STEP_WORDS {
                    sfence_if_x86();
                    if !keep_going() {
                        return false;
                    }
                    add_bytes((done_words * 16) as u64);
                    done_words = 0;
                }
                let lo = (2 * j + parity) * block_words;
                let hi = (2 * (pair_cnt - 1 - j) + parity) * block_words;
                nt_swap_invert_blocks(&mut a[lo..lo + block_words], &mut b[hi..hi + block_words]);
                done_words += block_words;
            }
        }
        if done_words > 0 {
            add_bytes((done_words * 16) as u64);
        }
        sfence_if_x86();
    }
    // If odd chunk count, mirror the middle chunk from opposite ends using the exact same parity pairing.
    if (n & 1) == 1 {
        let mid = &mut chunks[chunk_pairs].0;
        let pair_cnt = (mid.len() / block_words) / 2;
        let half = pair_cnt / 2;
        let mut done_words = 0usize;
        for j in 0..half {
            for parity in 0..2usize {
                if let Some(want) = striped_parity {
                    if parity != want {
                        continue;
                    }
                }
                if done_words >= STEP_WORDS {
                    sfence_if_x86();
                    if !keep_going() {
                        return false;
                    }
                    add_bytes((done_words * 16) as u64);
                    done_words = 0;
                }
                let lo = (2 * j + parity) * block_words;
                let hi = (2 * (pair_cnt - 1 - j) + parity) * block_words;
                let (l_part, r_part) = mid.split_at_mut(hi);
                nt_swap_invert_blocks(
                    &mut l_part[lo..lo + block_words],
                    &mut r_part[..block_words],
                );
                done_words += block_words;
            }
        }
        if done_words > 0 {
            add_bytes((done_words * 16) as u64);
        }
        sfence_if_x86();
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

    // Rotate 3 complementary DRAM/IMC/FCLK stress modes:
    //   0 = 128B row-miss + Write-Back RFO (max FCLK/L3 dirty writeback/IMC collisions)
    //   1 = 128B row-miss + Non-Temporal WC-bypass
    //   2 = 4 KiB max-bandwidth + Non-Temporal
    let rmw_mode = sub_pass % 3;
    match test_id {
        // T1 (FIRST STEP IN SEQUENCE): SimpleTest M2 — True LCG (0x77777777 / 0x33333333)
        1 => {
            let gen = |g: u64| absolut_lcg_t1(seed, g);
            run_simple_dual_rmw(
                chunks, base_word, 33, rmw_mode, &gen, &mut res, keep_going, add_bytes, on_error,
            );
        }

        // T2: SimpleTest M1 — Rotating variable cache-line DQ polarity flips + R<->W turnaround.
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

        // T3: MirrorMove128 (Param 2) — Cross-allocation striped 128B NT mirror + 64-bit invert.
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
            if !cross_chunk_nt_mirror(chunks, 16, Some(0), keep_going, add_bytes) {
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
            if !cross_chunk_nt_mirror(chunks, 16, Some(1), keep_going, add_bytes) {
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

        // T4: MirrorMove (Param 4) — Cross-allocation 4 KiB page NT mirror + 64-bit invert.
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
            if !cross_chunk_nt_mirror(chunks, PAGE_WORDS, None, keep_going, add_bytes) {
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

        // T5: MirrorMove128 (Param 1) — Cross-allocation contiguous 128B NT mirror + invert + restore.
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
            if !cross_chunk_nt_mirror(chunks, 16, None, keep_going, add_bytes) {
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
            if !cross_chunk_nt_mirror(chunks, 16, None, keep_going, add_bytes) {
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

        // T6: SimpleTest M0 (4M) — Heavy moving inversions + cross-chunk bank-strided R<->W.
        6 => {
            let pat = match iter_idx & 3 {
                0 => 0x0000_0000_0000_0000u64,
                1 => 0xFFFF_FFFF_FFFF_FFFFu64,
                2 => 0x5555_5555_5555_5555u64,
                _ => 0xAAAA_AAAA_AAAA_AAAAu64,
            };
            let gen = |g: u64| if (g & 1) == 0 { pat } else { !pat };
            run_simple_dual_rmw(
                chunks, base_word, 33, rmw_mode, &gen, &mut res, keep_going, add_bytes, on_error,
            );
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

        // T9: SimpleTest M0 (32M) — LCG + SplitMix64 high-entropy bus toggle R<->W.
        9 => {
            let mul = 0x5851_F42D_4C95_7F2Du64;
            let add = seed | 1;
            let gen = |g: u64| splitmix64(g.wrapping_mul(mul).wrapping_add(add));
            run_simple_dual_rmw(
                chunks, base_word, 65, rmw_mode, &gen, &mut res, keep_going, add_bytes, on_error,
            );
        }

        // T10: SimpleTest M0 (64M) — Modulo20 Bank-Conflict Scatter at full AVX2 NT bandwidth.
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

        // T13: SimpleTest M0 (512M) — Walking 64-bit 1s & 0s + bank-strided R<->W.
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

        // T15 (default, runs 8x per cycle): SimpleTest M2 — 0xFFFFFFFF / 0x55555555 R<->W turnaround.
        _ => {
            let gen = |g: u64| absolut_lcg_t15(seed, g);
            run_simple_dual_rmw(
                chunks, base_word, 29, rmw_mode, &gen, &mut res, keep_going, add_bytes, on_error,
            );
        }
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
        let layouts: &[&[usize]] = &[&[32_768], &[32_768, 32_768], &[32_768, 12_345, 24_576]];
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
                for sub_pass in 0..3u32 {
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
    fn absolut_dual_rmw_and_mirror_detect_single_bit_fault_immediately() {
        // 1) Verify dual_verify_and_nt_write_slice catches a single flipped bit with exact XOR diff.
        let base = 0x1000u64;
        let gen = |g: u64| absolut_lcg_t1(0xDEAD_BEEF, g);
        let inv = |g: u64| !gen(g);
        let mut src = vec![0u64; 4096];
        let mut dst = vec![0u64; 4096];
        nt_fill_gen_nofence(&mut src, base, &gen);
        sfence_if_x86();
        src[777] ^= 1u64 << 43;
        let (bad, first) =
            dual_verify_and_write_slice::<false>(&src, base, 0, &gen, &mut dst, base + 4096, &inv);
        assert_eq!(bad, 1);
        let (idx, exp, got) = first.unwrap();
        assert_eq!(idx, 777);
        assert_eq!(exp ^ got, 1u64 << 43);
        // Verify Write-Back RFO path catches the same single bit flip as well.
        let (bad_wb, first_wb) =
            dual_verify_and_write_slice::<true>(&src, base, 0, &gen, &mut dst, base + 4096, &inv);
        assert_eq!(bad_wb, 1);
        assert_eq!(first_wb, Some((777, exp, got)));
        // Verify dst still received exact inverted stream.
        assert_eq!(verify_gen(&dst, base + 4096, 0, &inv).0, 0);

        // 2) Verify cross-chunk mirror verifier catches a single bit flip and invokes on_error immediately.
        let tag = |g: u64| absolut_tag(0xCAFE_BABE, g);
        let mut chunks = vec![(vec![0u64; 2048], 0u64), (vec![0u64; 2048], 2048u64)];
        for (c, s) in &mut chunks {
            nt_fill_gen_nofence(c, *s, &tag);
        }
        sfence_if_x86();
        assert!(cross_chunk_nt_mirror(
            &mut chunks,
            16,
            None,
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
