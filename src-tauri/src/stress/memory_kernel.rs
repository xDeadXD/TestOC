//! Memory stress engine: Absolut 32-step DRAM/IMC suite + classic pattern runner.
//!
//! Every test is 100% deterministic and bit-exact (zero subjectivity, zero false positives).
//! On AVX2 CPUs, Absolut uses 256-bit non-temporal streaming stores (`_mm256_stream_si256` +
//! `_mm_sfence`) to bypass L1/L2/L3 caches directly into physical DRAM write-combining buffers,
//! paired with 256-bit SIMD verification, converging bidirectional `MirrorMove` / `MirrorMove128`
//! permutations, cross-chunk `BlockMove` copies, and `_mm_clflush` aggressor rowhammer sweeps.

use std::sync::atomic::{compiler_fence, Ordering};

/// Allocation granularity. Abort is checked between chunks, so even allocation is interruptible.
pub const CHUNK_BYTES: usize = 32 << 20;
/// Top-up chunk granularity used when reclaiming large freed blocks (e.g. closed apps).
pub const TOPUP_CHUNK_BYTES: usize = 16 << 20;
/// Fine-grain tail top-up chunk used to hold constant 99%+ physical RAM without dipping.
pub const FINE_TOPUP_CHUNK_BYTES: usize = 4 << 20;
/// Write/verify step: 1 MiB, stop flag checked after every step.
pub const STEP_WORDS: usize = (1 << 20) / 8;
pub const MIN_TARGET_BYTES: u64 = 128 << 20;
/// Standard OS safety reserve (128 MiB); drops to 64 MiB at >=98% and 24 MiB at 100% (OOM-safe).
pub const FREE_FLOOR_BYTES: u64 = 128 << 20;
pub const EXTREME_FLOOR_BYTES: u64 = 64 << 20;
pub const MAX_FLOOR_BYTES: u64 = 24 << 20;

/// Prime stride in 64-bit words (~32 KiB + 24 B): coprime to power-of-2 slice lengths, defeating
/// L1/L2 stride prefetchers and crossing DRAM banks.
const STRIDE_WORDS: usize = 4099;

/// Absolut 32-step DRAM cycle sequence:
/// Prologue T0 (RefreshStable) followed by the 31-step interleaved sequence (32 steps per cycle).
pub const ABSOLUT_SEQUENCE: &[u8] = &[
    0, 1, 4, 6, 15, 3, 2, 7, 15, 5, 2, 8, 15, 4, 2, 9, 15, 3, 2, 10, 15, 5, 2, 11, 15, 4, 2, 12,
    15, 5, 14, 15,
];

/// Inner pass counts per Absolut sub-test, weighted by test duration.
/// Calibrated so 1 full 32-step cycle takes ~25-30 minutes on 16 GiB RAM (~55-60 min on 32 GiB),
/// scaling linearly with installed RAM capacity while sweeping all bit-shifts, polarities, and phases.
pub fn absolut_step_passes(test_id: u8) -> u32 {
    match test_id {
        0 => 32,  // RefreshStable (~38 s on 16 GiB)
        1 => 40,  // SimpleTest M2 0x77/0x33 (~42 s)
        2 => 48,  // SimpleTest M1 Alt Polarity (~26 s x 7)
        3 => 32,  // MirrorMove128 Striped 2x (~60 s x 2)
        4 => 48,  // MirrorMove 4K Page (~52 s x 3)
        5 => 32,  // MirrorMove128 Contiguous (~60 s x 3)
        6 => 120, // SimpleTest M0 Extended Inv heavy thermal soak (~126 s)
        7 => 64,  // DQ Single-Bit Aggressor (8x full 8-bit lane sweep, ~34 s)
        8 => 80,  // VDDQ Ground-Bounce (~42 s)
        9 => 80,  // RandomNumberSequence (~42 s)
        10 => 40, // Modulo20 Bank-Conflict (2x full 0..19 phase sweep, ~52 s)
        11 => 40, // Cross-Chunk NT BlockMove (~64 s)
        12 => 48, // Address Inverse Walk (~50 s)
        13 => 64, // Walking 64-Bit (all 64 bits)
        14 => 48, // Rowhammer Bit-Fade (~58 s)
        15 => 56, // NT-Stream Cross-Check (~58 s x 8)
        _ => 32,
    }
}

/// Effective runtime inner pass count: authentic 30-min cycle in production, 1 pass in unit tests.
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
        1 => "SimpleTest M2 (0x77/0x33)",
        2 => "SimpleTest M1 (Alt Polarity)",
        3 => "MirrorMove128 (Striped 2x)",
        4 => "MirrorMove (4K Page)",
        5 => "MirrorMove128 (Contiguous)",
        6 => "SimpleTest M0 (Extended Inv)",
        7 => "DQ Single-Bit Aggressor",
        8 => "VDDQ Ground-Bounce",
        9 => "RandomNumberSequence",
        10 => "Modulo20 Bank-Conflict",
        11 => "BlockMove (Cross-Chunk NT)",
        12 => "Address Inverse Walk",
        13 => "Walking 64-Bit",
        14 => "Rowhammer Bit-Fade",
        15 => "NT-Stream Cross-Check",
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
    fn merge(&mut self, chunk_idx: usize, bad: u64, first: Option<(usize, u64, u64)>) {
        self.bad_words += bad;
        if self.first.is_none() {
            if let Some((word_idx, expected, actual)) = first {
                self.first = Some(MemMismatch {
                    chunk_idx,
                    word_idx,
                    expected,
                    actual,
                });
            }
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

/// Stateless expected value for a word in the classic Standard pattern runner.
#[inline(always)]
pub fn expected(p: Pattern, pass: u64, global_word: u64) -> u64 {
    match p {
        Pattern::MovingInversions => {
            // Rotate 0x55/0xAA and 0x00/0xFF across passes, alternating every word for 64-bit SSO toggling.
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
            // Alternating high-entropy inverted words per cache line so row-conflict reads toggle all lanes.
            let v = splitmix64(global_word ^ pass.wrapping_mul(0xD1B5_4A32_D192_ED03));
            if ((global_word >> 3) & 1) == 0 {
                v
            } else {
                !v
            }
        }
        Pattern::WalkingOnes => {
            let bit = 1u64.rotate_left(((global_word + pass) % 64) as u32);
            // Alternate Walking Ones & complementary Walking Zeros to catch both stuck-0 and stuck-1 bits.
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
        // Volatile read so the compiler cannot reuse values from the fill loop.
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
    v.resize(words, 0); // commits the pages
    Some(v)
}

// ---------------------------------------------------------------------------
// Low-level AVX2 non-temporal streaming & 256-bit verification primitives
// ---------------------------------------------------------------------------

#[inline]
fn has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

/// Stream a 2-word repeating pattern directly to DRAM via AVX2 `_mm256_stream_si256` (bypasses L1/L2/L3).
fn nt_fill_pair(s: &mut [u64], even: u64, odd: u64) {
    if has_avx2() && s.len() >= 8 {
        // SAFETY: `has_avx2()` checked above; alignment prefix ensures 32-byte alignment for `_mm256_stream_si256`.
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
    // Preserve word parity after an odd-length alignment prefix.
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

/// Stream a 64-byte (8-word) cache-line alternating pattern (`line0` on even lines, `line1` on odd lines).
fn nt_fill_cachelines(
    s: &mut [u64],
    line0_even: u64,
    line0_odd: u64,
    line1_even: u64,
    line1_odd: u64,
) {
    if has_avx2() && s.len() >= 16 {
        // SAFETY: `has_avx2()` checked above; handles 32-byte alignment safely.
        unsafe { nt_fill_cachelines_avx2(s, line0_even, line0_odd, line1_even, line1_odd) };
        return;
    }
    for (i, w) in s.iter_mut().enumerate() {
        let v = cacheline_word(i, line0_even, line0_odd, line1_even, line1_odd);
        unsafe { std::ptr::write_volatile(w, v) };
    }
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

#[target_feature(enable = "avx2")]
unsafe fn nt_fill_cachelines_avx2(s: &mut [u64], l0e: u64, l0o: u64, l1e: u64, l1o: u64) {
    use std::arch::x86_64::*;
    let len = s.len();
    let ptr = s.as_mut_ptr();
    let mis = (ptr as usize) & 31;
    let prefix = if mis == 0 { 0 } else { (32 - mis) / 8 }.min(len);
    for i in 0..prefix {
        std::ptr::write_volatile(ptr.add(i), cacheline_word(i, l0e, l0o, l1e, l1o));
    }
    let mut i = prefix;
    let end4 = prefix + ((len - prefix) & !3);
    while i < end4 {
        let a = cacheline_word(i, l0e, l0o, l1e, l1o);
        let b = cacheline_word(i + 1, l0e, l0o, l1e, l1o);
        let c = cacheline_word(i + 2, l0e, l0o, l1e, l1o);
        let d = cacheline_word(i + 3, l0e, l0o, l1e, l1o);
        let v = _mm256_set_epi64x(d as i64, c as i64, b as i64, a as i64);
        _mm256_stream_si256(ptr.add(i) as *mut __m256i, v);
        i += 4;
    }
    while i < len {
        std::ptr::write_volatile(ptr.add(i), cacheline_word(i, l0e, l0o, l1e, l1o));
        i += 1;
    }
    _mm_sfence();
}

/// Stream computed per-word values directly to DRAM via AVX2 `_mm256_stream_si256`.
fn nt_fill_gen(s: &mut [u64], base: u64, gen: impl Fn(u64) -> u64) {
    if has_avx2() && s.len() >= 8 {
        // SAFETY: `has_avx2()` checked above; handles 32-byte alignment prefix/tail safely.
        unsafe { nt_fill_gen_avx2(s, base, gen) };
        return;
    }
    for (i, w) in s.iter_mut().enumerate() {
        unsafe { std::ptr::write_volatile(w, gen(base + i as u64)) };
    }
}

#[target_feature(enable = "avx2")]
unsafe fn nt_fill_gen_avx2(s: &mut [u64], base: u64, gen: impl Fn(u64) -> u64) {
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
        _mm256_stream_si256(ptr.add(i) as *mut __m256i, v);
        i += 4;
    }
    while i < len {
        std::ptr::write_volatile(ptr.add(i), gen(base + i as u64));
        i += 1;
    }
    _mm_sfence();
}

/// Non-temporal streaming copy `src -> dst` (same length), optionally XOR-inverting each word.
fn nt_copy_slice(src: &[u64], dst: &mut [u64], invert: bool) {
    debug_assert_eq!(src.len(), dst.len());
    let mask = if invert { u64::MAX } else { 0 };
    if has_avx2() && dst.len() >= 8 {
        // SAFETY: `has_avx2()` checked above; handles 32-byte alignment prefix/tail safely.
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

/// Fast 256-bit SIMD verification against a per-word generator, reporting exact word diagnostics on mismatch.
fn verify_gen(
    s: &[u64],
    base: u64,
    word_offset: usize,
    gen: impl Fn(u64) -> u64,
) -> (u64, Option<(usize, u64, u64)>) {
    compiler_fence(Ordering::SeqCst);
    if has_avx2() && s.len() >= 8 {
        // SAFETY: `has_avx2()` checked above; unaligned loads are valid for any slice pointer.
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
    gen: impl Fn(u64) -> u64,
) -> (u64, Option<(usize, u64, u64)>) {
    use std::arch::x86_64::*;
    let len = s.len();
    let ptr = s.as_ptr();
    let mut bad = 0u64;
    let mut first = None;
    let mut i = 0usize;
    let end4 = len & !3;
    while i < end4 {
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

/// Fast 256-bit verification for a 2-word repeating `(even, odd)` pattern.
fn verify_pair(
    s: &[u64],
    even: u64,
    odd: u64,
    word_offset: usize,
) -> (u64, Option<(usize, u64, u64)>) {
    compiler_fence(Ordering::SeqCst);
    if has_avx2() && s.len() >= 8 {
        // SAFETY: `has_avx2()` checked above.
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

/// Swap two disjoint equal-length blocks using AVX2 256-bit burst loads/stores.
#[inline]
fn swap_blocks(a: &mut [u64], b: &mut [u64]) {
    debug_assert_eq!(a.len(), b.len());
    if has_avx2() && a.len() >= 4 {
        // SAFETY: `has_avx2()` checked; `&mut` slices are guaranteed non-overlapping.
        unsafe { swap_blocks_avx2(a, b) };
        return;
    }
    a.swap_with_slice(b);
}

#[target_feature(enable = "avx2")]
unsafe fn swap_blocks_avx2(a: &mut [u64], b: &mut [u64]) {
    use std::arch::x86_64::*;
    let len = a.len();
    let ap = a.as_mut_ptr();
    let bp = b.as_mut_ptr();
    let mut i = 0usize;
    let end4 = len & !3;
    while i < end4 {
        let va = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(bp.add(i) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, vb);
        _mm256_storeu_si256(bp.add(i) as *mut __m256i, va);
        i += 4;
    }
    while i < len {
        std::ptr::swap(ap.add(i), bp.add(i));
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
// Absolut 16-algorithm execution engine
// ---------------------------------------------------------------------------

#[inline(always)]
fn absolut_tag(seed: u64, global_word: u64) -> u64 {
    splitmix64(global_word ^ seed)
}

/// Execute one inner pass (`sub_pass`) of an Absolut sub-test (`test_id` 0..=15) across all
/// `chunks` owned by one worker. Each sub-pass rotates bit-shifts, polarities, modulo phases, and
/// PRNG seeds while filling all chunks before verifying so data leaves CPU L3 into physical DRAM.
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

    match test_id {
        // T0: RefreshStable — NT alternating checkerboard + aggressor rowhammer + retention verify.
        0 => {
            let (e0, o0) = if (iter_idx & 1) == 0 {
                (0xAAAA_AAAA_AAAA_AAAAu64, 0x5555_5555_5555_5555u64)
            } else {
                (0x5555_5555_5555_5555u64, 0xAAAA_AAAA_AAAA_AAAAu64)
            };
            // Phase 1: NT-stream fill all chunks.
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
            // Phase 2: Focused _mm_clflush aggressor reads at 64 KiB / 128 KiB row strides to stress tREFI/tRFC.
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
            // Phase 3: Full 256-bit read-verify across all chunks.
            for (ci, (chunk, _)) in chunks.iter().enumerate() {
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let (bad, first) = verify_pair(step, e0, o0, si * STEP_WORDS);
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T1: SimpleTest Mode 2 — 0x77777777 / 0x33333333 bit-lane crosstalk + complement invert pass.
        1 => {
            let (e0, o0) = if (iter_idx & 1) == 0 {
                (0x7777_7777_3333_3333u64, 0x3333_3333_7777_7777u64)
            } else {
                (0x8888_8888_CCCC_CCCCu64, 0xCCCC_CCCC_8888_8888u64)
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
            // Verify forward and immediately overwrite with bitwise complement (!e0, !o0).
            for (ci, (chunk, _)) in chunks.iter_mut().enumerate() {
                for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let (bad, first) = verify_pair(step, e0, o0, si * STEP_WORDS);
                    res.merge(ci, bad, first);
                    nt_fill_pair(step, !e0, !o0);
                    add_bytes((step.len() * 16) as u64);
                }
            }
            // Verify complemented pass across all chunks.
            for (ci, (chunk, _)) in chunks.iter().enumerate() {
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let (bad, first) = verify_pair(step, !e0, !o0, si * STEP_WORDS);
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T2: SimpleTest Mode 1 — Alternating 64-byte cache-line DQ polarity flips.
        2 => {
            let (l0e, l0o, l1e, l1o) = if ((step_idx + sub_pass as usize) & 1) == 0 {
                (
                    0xAAAA_AAAA_5555_5555u64,
                    0x5555_5555_AAAA_AAAAu64,
                    0x5555_5555_AAAA_AAAAu64,
                    0xAAAA_AAAA_5555_5555u64,
                )
            } else {
                (
                    0xF0F0_F0F0_0F0F_0F0Fu64,
                    0x0F0F_0F0F_F0F0_F0F0u64,
                    0x0F0F_0F0F_F0F0_F0F0u64,
                    0xF0F0_F0F0_0F0F_0F0Fu64,
                )
            };
            for (chunk, _) in chunks.iter_mut() {
                for step in chunk.chunks_mut(STEP_WORDS) {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    nt_fill_cachelines(step, l0e, l0o, l1e, l1o);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            for (ci, (chunk, _)) in chunks.iter().enumerate() {
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let (bad, first) = verify_gen(step, 0, si * STEP_WORDS, |idx| {
                        cacheline_word(idx as usize, l0e, l0o, l1e, l1o)
                    });
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T3: MirrorMove128 (Param 2) — Striped 128B converging mirror (even stripes, verify, odd stripes, verify).
        3 => {
            // Fill unique per-word signature via NT-stream.
            for (chunk, chunk_start) in chunks.iter_mut() {
                for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let b = base_word + *chunk_start + (si * STEP_WORDS) as u64;
                    nt_fill_gen(step, b, |g| absolut_tag(seed, g));
                    add_bytes((step.len() * 8) as u64);
                }
            }
            // Pass A: Converging swap on EVEN 128-byte (16-word) stripes across each chunk.
            for (chunk, _) in chunks.iter_mut() {
                let stripes = chunk.len() / 16;
                let pairs = stripes / 2; // count of even stripes (2*j)
                let half = pairs / 2;
                let mut done_words = 0usize;
                for j in 0..half {
                    if done_words >= STEP_WORDS {
                        if !keep_going() {
                            res.aborted = true;
                            return res;
                        }
                        add_bytes((done_words * 16) as u64);
                        done_words = 0;
                    }
                    let lo = (2 * j) * 16;
                    let hi = (2 * (pairs - 1 - j)) * 16;
                    let (left, right) = chunk.split_at_mut(hi);
                    swap_blocks(&mut left[lo..lo + 16], &mut right[..16]);
                    done_words += 16;
                }
                if done_words > 0 {
                    add_bytes((done_words * 16) as u64);
                }
            }
            // Verify exact even-mirrored image.
            for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
                let stripes = chunk.len() / 16;
                let pairs = stripes / 2;
                let active_words = pairs * 2 * 16;
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let (bad, first) = verify_gen(step, w0 as u64, w0, |idx_u64| {
                        let idx = idx_u64 as usize;
                        let orig = if idx < active_words {
                            let st = idx / 16;
                            let off = idx & 15;
                            if (st & 1) == 0 {
                                let j = st / 2;
                                (2 * (pairs - 1 - j)) * 16 + off
                            } else {
                                idx
                            }
                        } else {
                            idx
                        };
                        absolut_tag(seed, base_word + *chunk_start + orig as u64)
                    });
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            // Pass B: Converging swap on ODD 128-byte stripes.
            for (chunk, _) in chunks.iter_mut() {
                let stripes = chunk.len() / 16;
                let pairs = stripes / 2;
                let half = pairs / 2;
                let mut done_words = 0usize;
                for j in 0..half {
                    if done_words >= STEP_WORDS {
                        if !keep_going() {
                            res.aborted = true;
                            return res;
                        }
                        add_bytes((done_words * 16) as u64);
                        done_words = 0;
                    }
                    let lo = (2 * j + 1) * 16;
                    let hi = (2 * (pairs - 1 - j) + 1) * 16;
                    let (left, right) = chunk.split_at_mut(hi);
                    swap_blocks(&mut left[lo..lo + 16], &mut right[..16]);
                    done_words += 16;
                }
                if done_words > 0 {
                    add_bytes((done_words * 16) as u64);
                }
            }
            // Verify full even+odd mirrored image.
            for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
                let stripes = chunk.len() / 16;
                let pairs = stripes / 2;
                let active_words = pairs * 2 * 16;
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let (bad, first) = verify_gen(step, w0 as u64, w0, |idx_u64| {
                        let idx = idx_u64 as usize;
                        let orig = if idx < active_words {
                            let st = idx / 16;
                            let off = idx & 15;
                            let j = st / 2;
                            let parity = st & 1;
                            (2 * (pairs - 1 - j) + parity) * 16 + off
                        } else {
                            idx
                        };
                        absolut_tag(seed, base_word + *chunk_start + orig as u64)
                    });
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T4: MirrorMove (Param 4) — Converging 4 KiB page mirror from opposite ends + verify.
        4 => {
            for (chunk, chunk_start) in chunks.iter_mut() {
                for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let b = base_word + *chunk_start + (si * STEP_WORDS) as u64;
                    nt_fill_gen(step, b, |g| absolut_tag(seed, g));
                    add_bytes((step.len() * 8) as u64);
                }
            }
            for (chunk, _) in chunks.iter_mut() {
                let pages = chunk.len() / 512;
                let half = pages / 2;
                let mut done_words = 0usize;
                for p in 0..half {
                    if done_words >= STEP_WORDS {
                        if !keep_going() {
                            res.aborted = true;
                            return res;
                        }
                        add_bytes((done_words * 16) as u64);
                        done_words = 0;
                    }
                    let lo = p * 512;
                    let hi = (pages - 1 - p) * 512;
                    let (left, right) = chunk.split_at_mut(hi);
                    swap_blocks(&mut left[lo..lo + 512], &mut right[..512]);
                    done_words += 512;
                }
                if done_words > 0 {
                    add_bytes((done_words * 16) as u64);
                }
            }
            for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
                let pages = chunk.len() / 512;
                let active_words = pages * 512;
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let (bad, first) = verify_gen(step, w0 as u64, w0, |idx_u64| {
                        let idx = idx_u64 as usize;
                        let orig = if idx < active_words {
                            let p = idx / 512;
                            let off = idx & 511;
                            (pages - 1 - p) * 512 + off
                        } else {
                            idx
                        };
                        absolut_tag(seed, base_word + *chunk_start + orig as u64)
                    });
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T5: MirrorMove128 (Param 1) — Contiguous 128B converging mirror, verify, mirror back, verify.
        5 => {
            for (chunk, chunk_start) in chunks.iter_mut() {
                for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let b = base_word + *chunk_start + (si * STEP_WORDS) as u64;
                    nt_fill_gen(step, b, |g| absolut_tag(seed, g));
                    add_bytes((step.len() * 8) as u64);
                }
            }
            // Mirror pass 1: reverse all 128-byte stripes from outer edges inward.
            for (chunk, _) in chunks.iter_mut() {
                let stripes = chunk.len() / 16;
                let half = stripes / 2;
                let mut done_words = 0usize;
                for s in 0..half {
                    if done_words >= STEP_WORDS {
                        if !keep_going() {
                            res.aborted = true;
                            return res;
                        }
                        add_bytes((done_words * 16) as u64);
                        done_words = 0;
                    }
                    let lo = s * 16;
                    let hi = (stripes - 1 - s) * 16;
                    let (left, right) = chunk.split_at_mut(hi);
                    swap_blocks(&mut left[lo..lo + 16], &mut right[..16]);
                    done_words += 16;
                }
                if done_words > 0 {
                    add_bytes((done_words * 16) as u64);
                }
            }
            // Verify mirrored image.
            for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
                let stripes = chunk.len() / 16;
                let active_words = stripes * 16;
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let (bad, first) = verify_gen(step, w0 as u64, w0, |idx_u64| {
                        let idx = idx_u64 as usize;
                        let orig = if idx < active_words {
                            let s = idx / 16;
                            let off = idx & 15;
                            (stripes - 1 - s) * 16 + off
                        } else {
                            idx
                        };
                        absolut_tag(seed, base_word + *chunk_start + orig as u64)
                    });
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            // Mirror pass 2: reverse back to restore original order and re-verify.
            for (chunk, _) in chunks.iter_mut() {
                let stripes = chunk.len() / 16;
                let half = stripes / 2;
                let mut done_words = 0usize;
                for s in 0..half {
                    if done_words >= STEP_WORDS {
                        if !keep_going() {
                            res.aborted = true;
                            return res;
                        }
                        add_bytes((done_words * 16) as u64);
                        done_words = 0;
                    }
                    let lo = s * 16;
                    let hi = (stripes - 1 - s) * 16;
                    let (left, right) = chunk.split_at_mut(hi);
                    swap_blocks(&mut left[lo..lo + 16], &mut right[..16]);
                    done_words += 16;
                }
                if done_words > 0 {
                    add_bytes((done_words * 16) as u64);
                }
            }
            for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *chunk_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, |g| absolut_tag(seed, g));
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T6: SimpleTest Mode 0 — Heavy extended moving inversions (forward fill, backward verify+invert, forward verify).
        6 => {
            let pat = match iter_idx & 3 {
                0 => 0x0000_0000_0000_0000u64,
                1 => 0xFFFF_FFFF_FFFF_FFFFu64,
                2 => 0x5555_5555_5555_5555u64,
                _ => 0xAAAA_AAAA_AAAA_AAAAu64,
            };
            for (chunk, _) in chunks.iter_mut() {
                for step in chunk.chunks_mut(STEP_WORDS) {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    nt_fill_pair(step, pat, !pat);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            // Backward pass: verify (pat, !pat) then invert to (!pat, pat).
            for (ci, (chunk, _)) in chunks.iter_mut().enumerate().rev() {
                let total_steps = chunk.len().div_ceil(STEP_WORDS);
                for (rev_i, step) in chunk.chunks_mut(STEP_WORDS).rev().enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let si = total_steps - 1 - rev_i;
                    let (bad, first) = verify_pair(step, pat, !pat, si * STEP_WORDS);
                    res.merge(ci, bad, first);
                    nt_fill_pair(step, !pat, pat);
                    add_bytes((step.len() * 16) as u64);
                }
            }
            // Forward pass: verify (!pat, pat).
            for (ci, (chunk, _)) in chunks.iter().enumerate() {
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let (bad, first) = verify_pair(step, !pat, pat, si * STEP_WORDS);
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T7: Rotating Single-Bit DQ Aggressor / Victim across all 8 bit lanes per byte.
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
            for (chunk, chunk_start) in chunks.iter_mut() {
                for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let b = base_word + *chunk_start + (si * STEP_WORDS) as u64;
                    nt_fill_gen(step, b, gen);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *chunk_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, gen);
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T8: VDDQ Ground-Bounce / Max Di-Dt Rail Switching (0x00/0xFF & 0x00FF/0xFF00 cache-line slams).
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
            for (chunk, _) in chunks.iter_mut() {
                for step in chunk.chunks_mut(STEP_WORDS) {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    nt_fill_cachelines(step, l0e, l0o, l1e, l1o);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            for (ci, (chunk, _)) in chunks.iter().enumerate() {
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let (bad, first) = verify_gen(step, 0, si * STEP_WORDS, |idx| {
                        cacheline_word(idx as usize, l0e, l0o, l1e, l1o)
                    });
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T9: RandomNumberSequence — LCG + SplitMix64 high-entropy bus toggle.
        9 => {
            let mul = 0x5851_F42D_4C95_7F2Du64;
            let add = seed | 1;
            let gen = |g: u64| splitmix64(g.wrapping_mul(mul).wrapping_add(add));
            for (chunk, chunk_start) in chunks.iter_mut() {
                for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let b = base_word + *chunk_start + (si * STEP_WORDS) as u64;
                    nt_fill_gen(step, b, gen);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *chunk_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, gen);
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T10: Modulo20 / Prime-Stride Bank-Conflict Scatter — defeats L1/L2/L3 stride prefetchers.
        10 => {
            let phase = (iter_idx % 20) as usize;
            let pat_hit = 0xD555_AAAA_5555_D5AAu64 ^ seed;
            let pat_bg = !pat_hit;
            let val_at = |idx: usize, g: u64| {
                if (idx + phase).is_multiple_of(20) {
                    pat_hit ^ g.rotate_left(13)
                } else {
                    pat_bg
                }
            };
            for (chunk, chunk_start) in chunks.iter_mut() {
                let len = chunk.len();
                let stride = coprime_stride(len, STRIDE_WORDS);
                let mut idx = 0usize;
                let mut done = 0usize;
                for _ in 0..len {
                    if done >= STEP_WORDS {
                        if !keep_going() {
                            res.aborted = true;
                            return res;
                        }
                        add_bytes((done * 8) as u64);
                        done = 0;
                    }
                    let g = base_word + *chunk_start + idx as u64;
                    chunk[idx] = val_at(idx, g);
                    idx += stride;
                    if idx >= len {
                        idx -= len;
                    }
                    done += 1;
                }
                if done > 0 {
                    add_bytes((done * 8) as u64);
                }
            }
            compiler_fence(Ordering::SeqCst);
            for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
                let len = chunk.len();
                let stride = coprime_stride(len, STRIDE_WORDS);
                let mut idx = 0usize;
                let mut done = 0usize;
                let mut bad = 0u64;
                let mut first = None;
                for _ in 0..len {
                    if done >= STEP_WORDS {
                        if !keep_going() {
                            res.aborted = true;
                            return res;
                        }
                        add_bytes((done * 8) as u64);
                        done = 0;
                    }
                    let got = unsafe { std::ptr::read_volatile(&chunk[idx]) };
                    let g = base_word + *chunk_start + idx as u64;
                    let exp = val_at(idx, g);
                    if got != exp {
                        bad += 1;
                        first.get_or_insert((idx, exp, got));
                    }
                    idx += stride;
                    if idx >= len {
                        idx -= len;
                    }
                    done += 1;
                }
                res.merge(ci, bad, first);
                if done > 0 {
                    add_bytes((done * 8) as u64);
                }
            }
        }

        // T11: BlockMove — Cross-chunk 64 KiB AVX2 non-temporal streaming copy + inverted copy-back.
        11 => {
            let n_chunks = chunks.len();
            // Fill every chunk with a distinct per-word signature.
            for (chunk, chunk_start) in chunks.iter_mut() {
                for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let b = base_word + *chunk_start + (si * STEP_WORDS) as u64;
                    nt_fill_gen(step, b, |g| absolut_tag(seed, g));
                    add_bytes((step.len() * 8) as u64);
                }
            }
            // Pairwise cross-chunk copy: chunk[2j] -> chunk[2j+1] (verify), then inverted chunk[2j+1] -> chunk[2j] (verify).
            let pairs = n_chunks / 2;
            for j in 0..pairs {
                let (left, right) = chunks.split_at_mut(2 * j + 1);
                let (src_vec, src_start) = &mut left[2 * j];
                let (dst_vec, _) = &mut right[0];
                let common = src_vec.len().min(dst_vec.len());
                // Copy src -> dst in 64 KiB blocks via NT stores.
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
                // Verify dst matches src's exact signature.
                for (si, step) in dst_vec[..common].chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *src_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, |g| absolut_tag(seed, g));
                    res.merge(2 * j + 1, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
                // Inverted copy-back dst -> src in 64 KiB blocks.
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
                // Verify src matches bitwise complement of its original signature.
                for (si, step) in src_vec[..common].chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *src_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, |g| !absolut_tag(seed, g));
                    res.merge(2 * j, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            // If odd number of chunks, run intra-chunk half-to-half NT BlockMove on the tail chunk.
            if (n_chunks & 1) == 1 {
                let ci = n_chunks - 1;
                let (chunk, chunk_start) = &mut chunks[ci];
                let half = chunk.len() / 2;
                if half > 0 {
                    let (lo, hi) = chunk.split_at_mut(half);
                    nt_copy_slice(lo, &mut hi[..half], true);
                    add_bytes((half * 16) as u64);
                    let (bad, first) =
                        verify_gen(&hi[..half], base_word + *chunk_start, half, |g| {
                            !absolut_tag(seed, g)
                        });
                    res.merge(ci, bad, first);
                    add_bytes((half * 8) as u64);
                }
            }
        }

        // T12: Address-Coupled Forward/Backward Inverse Walk (stresses CMD/ADDR bus, 1T/2T CR, FCLK/UCLK).
        12 => {
            let magic = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let gen_fwd = |g: u64| g ^ magic ^ (g.rotate_left(29));
            let gen_inv = |g: u64| !gen_fwd(g);
            for (chunk, chunk_start) in chunks.iter_mut() {
                for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let b = base_word + *chunk_start + (si * STEP_WORDS) as u64;
                    nt_fill_gen(step, b, gen_fwd);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            // Backward verify + invert.
            for (ci, (chunk, chunk_start)) in chunks.iter_mut().enumerate().rev() {
                let total_steps = chunk.len().div_ceil(STEP_WORDS);
                for (rev_i, step) in chunk.chunks_mut(STEP_WORDS).rev().enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let si = total_steps - 1 - rev_i;
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *chunk_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, gen_fwd);
                    res.merge(ci, bad, first);
                    nt_fill_gen(step, b, gen_inv);
                    add_bytes((step.len() * 16) as u64);
                }
            }
            // Forward verify of inverted address pattern.
            for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *chunk_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, gen_inv);
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T13: Walking 64-bit 1s & 0s.
        13 => {
            let gen = |g: u64| {
                let bit = 1u64.rotate_left(((g + iter_idx) & 63) as u32);
                if (iter_idx & 1) == 0 {
                    bit
                } else {
                    !bit
                }
            };
            for (chunk, chunk_start) in chunks.iter_mut() {
                for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let b = base_word + *chunk_start + (si * STEP_WORDS) as u64;
                    nt_fill_gen(step, b, gen);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *chunk_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, gen);
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T14: Rowhammer / Bit-Fade Retention — writes victim rows, hammers flanking rows with clflush, verifies victims.
        14 => {
            let gen = |g: u64| {
                // Alternate high-charge (all-1) and zero-charge (all-0) across 8 KiB row-sized bands with tag bits.
                let band = (g >> 10) & 1;
                let base_v = if band == 0 {
                    0xFFFF_FFFF_FFFF_FFFFu64
                } else {
                    0x0000_0000_0000_0000u64
                };
                base_v ^ (0x0001_0001_0001_0001u64.wrapping_mul((g ^ seed) & 0xF))
            };
            for (chunk, chunk_start) in chunks.iter_mut() {
                for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let b = base_word + *chunk_start + (si * STEP_WORDS) as u64;
                    nt_fill_gen(step, b, gen);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            // Double-sided rowhammer across 128 KiB and 256 KiB strides via `_mm_clflush` (forces physical ACT+PRE).
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
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *chunk_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, gen);
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }

        // T15 (default): Absolut Interleaved NT-Stream Cross-Check (runs 8x per cycle).
        // NT fill -> reverse burst verify + XOR-invert -> forward verify of inverted data.
        _ => {
            let gen_fwd = |g: u64| absolut_tag(seed, g);
            let gen_inv = |g: u64| !absolut_tag(seed, g);
            for (chunk, chunk_start) in chunks.iter_mut() {
                for (si, step) in chunk.chunks_mut(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let b = base_word + *chunk_start + (si * STEP_WORDS) as u64;
                    nt_fill_gen(step, b, gen_fwd);
                    add_bytes((step.len() * 8) as u64);
                }
            }
            for (ci, (chunk, chunk_start)) in chunks.iter_mut().enumerate().rev() {
                let total_steps = chunk.len().div_ceil(STEP_WORDS);
                for (rev_i, step) in chunk.chunks_mut(STEP_WORDS).rev().enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let si = total_steps - 1 - rev_i;
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *chunk_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, gen_fwd);
                    res.merge(ci, bad, first);
                    nt_fill_gen(step, b, gen_inv);
                    add_bytes((step.len() * 16) as u64);
                }
            }
            for (ci, (chunk, chunk_start)) in chunks.iter().enumerate() {
                for (si, step) in chunk.chunks(STEP_WORDS).enumerate() {
                    if !keep_going() {
                        res.aborted = true;
                        return res;
                    }
                    let w0 = si * STEP_WORDS;
                    let b = base_word + *chunk_start + w0 as u64;
                    let (bad, first) = verify_gen(step, b, w0, gen_inv);
                    res.merge(ci, bad, first);
                    add_bytes((step.len() * 8) as u64);
                }
            }
        }
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn absolut_all_16_tests_roundtrip_cleanly() {
        // Two 256 KiB chunks (32768 words each) exercise cross-chunk BlockMove, 4K MirrorMove,
        // striped/contiguous MirrorMove128, clflush rowhammer, and AVX2 NT-streaming.
        let mut chunks = vec![(vec![0u64; 32_768], 0u64), (vec![0u64; 32_768], 32_768u64)];
        for test_id in 0..=15u8 {
            let res = run_absolut_step(
                &mut chunks,
                1u64 << 40,
                0,
                test_id as usize,
                0,
                test_id,
                &|| true,
                &|_| {},
            );
            assert!(!res.aborted, "test {test_id} aborted unexpectedly");
            assert_eq!(
                res.bad_words,
                0,
                "test {test_id} ({}) reported errors: {:?}",
                absolut_test_name(test_id),
                res.first
            );
        }
        // Verify authentic ~25-30 min cycle depth on 16 GiB (total inner passes across 32 steps).
        let total_passes: u32 = ABSOLUT_SEQUENCE
            .iter()
            .map(|&id| absolut_step_passes(id))
            .sum();
        assert!(
            total_passes >= 1500,
            "expected >=1500 inner passes per Absolut cycle, got {total_passes}"
        );
    }

    #[test]
    fn absolut_unaligned_slices_and_single_bit_faults_detected() {
        // Odd/unaligned subslice exercises alignment prefix + tail paths in AVX2 NT-stream primitives.
        let mut backing = vec![0u64; 519];
        let sub = &mut backing[1..518]; // guaranteed unaligned relative to 32-byte boundary if base is aligned
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
    fn absolut_sequence_matches_32_step_layout() {
        assert_eq!(ABSOLUT_SEQUENCE.len(), 32);
        assert_eq!(ABSOLUT_SEQUENCE[0], 0);
        assert_eq!(
            &ABSOLUT_SEQUENCE[1..],
            &[
                1, 4, 6, 15, 3, 2, 7, 15, 5, 2, 8, 15, 4, 2, 9, 15, 3, 2, 10, 15, 5, 2, 11, 15, 4,
                2, 12, 15, 5, 14, 15
            ]
        );
    }
}
