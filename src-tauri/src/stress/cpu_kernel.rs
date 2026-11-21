//! CPU stability stress kernels with analytical & avalanche verification.
//!
//! 1. **`MatSet` (Dense-Mantissa FMA GEMM):**
//!    Inputs are dense 16-bit signed integers (-32767..=32767) stored as `f64`, so products
//!    (< 2^30) and partial sums (< 2^40 for n=1024) stay strictly inside `f64`'s exact 53-bit
//!    integer range while exercising 40+ mantissa bits, full Booth-multiplier bit-slices, sign
//!    toggles, and wide exponent shifts. Results are bit-identical across Scalar, AVX2+FMA, and
//!    AVX-512, verified analytically with two `i128` weighted checksums:
//!    `sum_ij (i+1) * C_ij = sum_k (sum_i (i+1) A_ik) * (sum_j B_kj)` and
//!    `sum_ij (j+1) * C_ij = sum_k (sum_i A_ik) * (sum_j (j+1) B_kj)`.
//!
//! 2. **`NttSet` (64-bit Goldilocks Prime Number Theoretic Transform):**
//!    64-bit radix-2 NTT over F_p (p = 2^64 - 2^32 + 1). Executes forward NTT -> cyclic-shift
//!    frequency multiply -> inverse NTT -> exact element-by-element comparison. Every butterfly
//!    exercises 64x64->128-bit integer MUL, carry chains, and cache/IMC bandwidth with full
//!    avalanche error propagation across all N elements.
//!
//! 3. **`FftSet` (Split-Complex Floating-Point FFT Squaring Torture):**
//!    Dual-engine AVX2+FMA / AVX-512F / Scalar split-complex cyclic FFT convolution
//!    (`IFFT(FFT(z)^2)`) with irrational trigonometric twiddles (`cos`/`sin`) filling all 52
//!    IEEE-754 mantissa bits. Supports `Smallest` (L1D Bailey FMA), `Small` (L2 Radix-4),
//!    `Large` (>L3 IMC/RAM), and `Blend` presets, verifying both a strict `< 0.40` floating-point
//!    rounding bound and exact complex DC/Nyquist convolution sum identities on every pass.

use crate::error::{AppError, AppResult};
use crate::model::{Dataset, FftPreset, IsaChoice, IsaFlags};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    Avx2Fma,
    Avx512,
}

impl Kernel {
    pub fn resolve(choice: IsaChoice, isa: &IsaFlags) -> AppResult<Kernel> {
        match choice {
            IsaChoice::Scalar => Ok(Kernel::Scalar),
            IsaChoice::Avx2Fma if isa.avx2 && isa.fma => Ok(Kernel::Avx2Fma),
            IsaChoice::Avx2Fma => Err(AppError::Unsupported("AVX2/FMA".into())),
            IsaChoice::Avx512 if isa.avx512f => Ok(Kernel::Avx512),
            IsaChoice::Avx512 => Err(AppError::Unsupported(
                "AVX-512F (not available on this CPU)".into(),
            )),
            IsaChoice::Auto if isa.avx512f => Ok(Kernel::Avx512),
            IsaChoice::Auto if isa.avx2 && isa.fma => Ok(Kernel::Avx2Fma),
            IsaChoice::Auto => Ok(Kernel::Scalar),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kernel::Scalar => "Scalar (SSE2)",
            Kernel::Avx2Fma => "AVX2 + FMA",
            Kernel::Avx512 => "AVX-512F",
        }
    }
}

pub fn detect_isa() -> IsaFlags {
    IsaFlags {
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        fma: std::arch::is_x86_feature_detected!("fma"),
        avx512f: std::arch::is_x86_feature_detected!("avx512f"),
    }
}

pub fn dataset_n(d: Dataset) -> usize {
    match d {
        Dataset::Small => 128,
        Dataset::Medium => 512,
        Dataset::Large => 1024,
    }
}

/// Power-of-2 NTT length matching cache/IMC footprint of `Dataset`.
pub fn ntt_len(d: Dataset) -> usize {
    match d {
        Dataset::Small => 8_192, // 64 KiB -> L1/L2 resident (max core ALU/MUL burn)
        Dataset::Medium => 65_536, // 512 KiB -> L2/L3 resident (64-bit modular avalanche)
        Dataset::Large => 524_288, // 4 MiB/thread -> exceeds L3, stresses IMC/FCLK/RAM
    }
}

/// Rows of C computed per work block. Small: whole matrix (~4 MFLOP, ~0.3 ms with AVX2).
/// Medium: 16-row panels (~8.4 MFLOP). Large: 4-row panels (~8.4 MFLOP) so abort stays instant.
pub fn panel_rows(n: usize) -> usize {
    if n <= 128 {
        n
    } else if n <= 512 {
        16
    } else {
        4
    }
}

pub struct MatSet {
    pub n: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    pub c: Vec<f64>,
    expect_row_w: i128,
    expect_col_w: i128,
}

fn xorshift(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

impl MatSet {
    pub fn new(n: usize, seed: u64) -> Self {
        Self::new_interruptible(n, seed, || true).expect("uninterrupted")
    }

    /// Build dense-mantissa matrices while polling `keep_going`; returns `None` if aborted during init.
    pub fn new_interruptible(n: usize, seed: u64, keep_going: impl Fn() -> bool) -> Option<Self> {
        assert!(n.is_multiple_of(32), "n must be a multiple of 32");
        let mut s = seed | 1;
        let mut gen = |len: usize| -> Option<Vec<f64>> {
            let mut v = Vec::with_capacity(len);
            for chunk in 0..(len / 16384).max(1) {
                if !keep_going() {
                    return None;
                }
                let start = chunk * 16384;
                let end = (start + 16384).min(len);
                for _ in start..end {
                    // Dense 16-bit signed values in [-32767, 32767]: products < 2^30, sums over
                    // n=1024 stay < 2^40 << 2^53 (exact f64 integer arithmetic across all ISAs).
                    let raw = (xorshift(&mut s) & 0xFFFF) as i32 - 32_768;
                    let clamped = raw.clamp(-32_767, 32_767);
                    v.push(clamped as f64);
                }
            }
            Some(v)
        };
        let a = gen(n * n)?;
        let b = gen(n * n)?;
        if !keep_going() {
            return None;
        }
        let mut m = MatSet {
            n,
            a,
            b,
            c: vec![0.0; n * n],
            expect_row_w: 0,
            expect_col_w: 0,
        };
        if !m.compute_expected(&keep_going) {
            return None;
        }
        Some(m)
    }

    fn compute_expected(&mut self, keep_going: impl Fn() -> bool) -> bool {
        let n = self.n;
        let (mut rw, mut cw) = (0i128, 0i128);
        for k in 0..n {
            if k.is_multiple_of(64) && !keep_going() {
                return false;
            }
            let mut a_w = 0i128; // sum_i (i+1) A_ik
            let mut a_s = 0i128; // sum_i A_ik
            for i in 0..n {
                let v = self.a[i * n + k] as i128;
                a_w += (i as i128 + 1) * v;
                a_s += v;
            }
            let mut b_s = 0i128; // sum_j B_kj
            let mut b_w = 0i128; // sum_j (j+1) B_kj
            for j in 0..n {
                let v = self.b[k * n + j] as i128;
                b_s += v;
                b_w += (j as i128 + 1) * v;
            }
            rw += a_w * b_s;
            cw += a_s * b_w;
        }
        self.expect_row_w = rw;
        self.expect_col_w = cw;
        true
    }

    /// Flip sign polarity of A and analytical checksums between passes to toggle sign/carry paths.
    pub fn flip_sign(&mut self) {
        for v in &mut self.a {
            *v = -*v;
        }
        self.expect_row_w = -self.expect_row_w;
        self.expect_col_w = -self.expect_col_w;
    }

    /// C[r0..r1, :] = A[r0..r1, :] * B. `r1 - r0` must be a multiple of 4.
    pub fn compute_rows(&mut self, kernel: Kernel, r0: usize, r1: usize) {
        debug_assert!((r1 - r0).is_multiple_of(4) && r1 <= self.n);
        let n = self.n;
        match kernel {
            Kernel::Scalar => rows_scalar(&self.a, &self.b, &mut self.c, n, r0, r1),
            // SAFETY: Kernel::resolve only returns SIMD kernels when the CPU supports them.
            Kernel::Avx2Fma => unsafe { rows_avx2_fma(&self.a, &self.b, &mut self.c, n, r0, r1) },
            Kernel::Avx512 => unsafe { rows_avx512(&self.a, &self.b, &mut self.c, n, r0, r1) },
        }
    }

    pub fn compute_all(&mut self, kernel: Kernel) {
        let n = self.n;
        self.compute_rows(kernel, 0, n);
    }

    /// Returns the (row-weighted, col-weighted) checksum of the current C.
    pub fn checksums(&self) -> (i128, i128) {
        let n = self.n;
        let (mut rw, mut cw) = (0i128, 0i128);
        for i in 0..n {
            let row = &self.c[i * n..(i + 1) * n];
            let mut rs = 0i128;
            for (j, &v) in row.iter().enumerate() {
                let iv = v as i128;
                rs += iv;
                cw += (j as i128 + 1) * iv;
            }
            rw += (i as i128 + 1) * rs;
        }
        (rw, cw)
    }

    /// Single-pass weighted checksum + finite/exact-integer verification so AVX2 duty cycle stays maximized.
    pub fn verify(&self) -> bool {
        let n = self.n;
        let (mut rw, mut cw) = (0i128, 0i128);
        for i in 0..n {
            let row = &self.c[i * n..(i + 1) * n];
            let mut rs = 0i128;
            for (j, &v) in row.iter().enumerate() {
                if !v.is_finite() {
                    return false;
                }
                let iv = v as i64;
                if (iv as f64) != v {
                    return false;
                }
                let iv128 = iv as i128;
                rs += iv128;
                cw += (j as i128 + 1) * iv128;
            }
            rw += (i as i128 + 1) * rs;
        }
        rw == self.expect_row_w && cw == self.expect_col_w
    }

    pub fn flops_per_full(&self) -> u64 {
        2 * (self.n as u64).pow(3)
    }
}

fn rows_scalar(a: &[f64], b: &[f64], c: &mut [f64], n: usize, r0: usize, r1: usize) {
    for i in r0..r1 {
        let crow = &mut c[i * n..(i + 1) * n];
        crow.fill(0.0);
        for k in 0..n {
            let av = a[i * n + k];
            let brow = &b[k * n..(k + 1) * n];
            for (cv, bv) in crow.iter_mut().zip(brow) {
                *cv += av * bv;
            }
        }
    }
}

/// 4x8 register-blocked micro-kernel: 8 independent FMA chains hide 4-cycle FMA latency
/// across dual 256-bit FMA execution pipes.
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_avx2_fma(a: &[f64], b: &[f64], c: &mut [f64], n: usize, r0: usize, r1: usize) {
    use std::arch::x86_64::*;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();
    let mut i = r0;
    while i < r1 {
        let mut j = 0;
        while j < n {
            let mut c00 = _mm256_setzero_pd();
            let mut c01 = _mm256_setzero_pd();
            let mut c10 = _mm256_setzero_pd();
            let mut c11 = _mm256_setzero_pd();
            let mut c20 = _mm256_setzero_pd();
            let mut c21 = _mm256_setzero_pd();
            let mut c30 = _mm256_setzero_pd();
            let mut c31 = _mm256_setzero_pd();
            for k in 0..n {
                let brow = bp.add(k * n + j);
                let b0 = _mm256_loadu_pd(brow);
                let b1 = _mm256_loadu_pd(brow.add(4));
                let a0 = _mm256_broadcast_sd(&*ap.add(i * n + k));
                let a1 = _mm256_broadcast_sd(&*ap.add((i + 1) * n + k));
                let a2 = _mm256_broadcast_sd(&*ap.add((i + 2) * n + k));
                let a3 = _mm256_broadcast_sd(&*ap.add((i + 3) * n + k));
                c00 = _mm256_fmadd_pd(a0, b0, c00);
                c01 = _mm256_fmadd_pd(a0, b1, c01);
                c10 = _mm256_fmadd_pd(a1, b0, c10);
                c11 = _mm256_fmadd_pd(a1, b1, c11);
                c20 = _mm256_fmadd_pd(a2, b0, c20);
                c21 = _mm256_fmadd_pd(a2, b1, c21);
                c30 = _mm256_fmadd_pd(a3, b0, c30);
                c31 = _mm256_fmadd_pd(a3, b1, c31);
            }
            let o = cp.add(i * n + j);
            _mm256_storeu_pd(o, c00);
            _mm256_storeu_pd(o.add(4), c01);
            let o = cp.add((i + 1) * n + j);
            _mm256_storeu_pd(o, c10);
            _mm256_storeu_pd(o.add(4), c11);
            let o = cp.add((i + 2) * n + j);
            _mm256_storeu_pd(o, c20);
            _mm256_storeu_pd(o.add(4), c21);
            let o = cp.add((i + 3) * n + j);
            _mm256_storeu_pd(o, c30);
            _mm256_storeu_pd(o.add(4), c31);
            j += 8;
        }
        i += 4;
    }
}

/// 4x16 AVX-512 variant (selected only when the CPU reports avx512f).
#[target_feature(enable = "avx512f")]
unsafe fn rows_avx512(a: &[f64], b: &[f64], c: &mut [f64], n: usize, r0: usize, r1: usize) {
    use std::arch::x86_64::*;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();
    let mut i = r0;
    while i < r1 {
        let mut j = 0;
        while j < n {
            let mut acc = [_mm512_setzero_pd(); 8];
            for k in 0..n {
                let brow = bp.add(k * n + j);
                let b0 = _mm512_loadu_pd(brow);
                let b1 = _mm512_loadu_pd(brow.add(8));
                for r in 0..4 {
                    let av = _mm512_set1_pd(*ap.add((i + r) * n + k));
                    acc[2 * r] = _mm512_fmadd_pd(av, b0, acc[2 * r]);
                    acc[2 * r + 1] = _mm512_fmadd_pd(av, b1, acc[2 * r + 1]);
                }
            }
            for r in 0..4 {
                let o = cp.add((i + r) * n + j);
                _mm512_storeu_pd(o, acc[2 * r]);
                _mm512_storeu_pd(o.add(8), acc[2 * r + 1]);
            }
            j += 16;
        }
        i += 4;
    }
}

// ---------------------------------------------------------------------------
// 64-bit Goldilocks Prime Number Theoretic Transform (Modular Avalanche)
// ---------------------------------------------------------------------------

/// Goldilocks prime p = 2^64 - 2^32 + 1 (supports exact radix-2 NTT for all N = 2^k <= 2^32).
const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;
/// Primitive root modulo GOLDILOCKS_P.
const GOLDILOCKS_G: u64 = 7;

#[inline(always)]
fn add_mod(a: u64, b: u64) -> u64 {
    let s = (a as u128) + (b as u128);
    if s >= GOLDILOCKS_P as u128 {
        (s - GOLDILOCKS_P as u128) as u64
    } else {
        s as u64
    }
}

#[inline(always)]
fn sub_mod(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        GOLDILOCKS_P - (b - a)
    }
}

#[inline(always)]
fn mul_mod(a: u64, b: u64) -> u64 {
    (((a as u128) * (b as u128)) % (GOLDILOCKS_P as u128)) as u64
}

fn pow_mod(mut base: u64, mut exp: u64) -> u64 {
    let mut acc = 1u64;
    while exp > 0 {
        if (exp & 1) != 0 {
            acc = mul_mod(acc, base);
        }
        base = mul_mod(base, base);
        exp >>= 1;
    }
    acc
}

/// Exact radix-2 NTT stress workspace with full-buffer avalanche verification.
pub struct NttSet {
    pub n: usize,
    log_n: u32,
    input: Vec<u64>,
    pub work: Vec<u64>,
    /// Stage roots of unity for forward NTT: `fwd_roots[stage]` is the primitive 2^(stage+1)-th root.
    fwd_roots: Vec<u64>,
    /// Stage roots of unity for inverse NTT.
    inv_roots: Vec<u64>,
    /// Primitive N-th root of unity (for frequency-domain cyclic shift).
    root_n: u64,
    /// Multiplicative inverse of N modulo GOLDILOCKS_P.
    n_inv: u64,
}

impl NttSet {
    pub fn new(n: usize, seed: u64) -> Self {
        Self::new_interruptible(n, seed, || true).expect("uninterrupted")
    }

    pub fn new_interruptible(n: usize, seed: u64, keep_going: impl Fn() -> bool) -> Option<Self> {
        assert!(n.is_power_of_two() && n >= 64);
        let log_n = n.trailing_zeros();
        let mut s = seed ^ 0xD1B5_4A32_D192_ED03 | 1;
        let mut input = Vec::with_capacity(n);
        for chunk in 0..(n / 16384).max(1) {
            if !keep_going() {
                return None;
            }
            let start = chunk * 16384;
            let end = (start + 16384).min(n);
            for _ in start..end {
                let mut v = xorshift(&mut s);
                if v >= GOLDILOCKS_P {
                    v -= GOLDILOCKS_P;
                }
                input.push(v);
            }
        }
        let root_n = pow_mod(GOLDILOCKS_G, (GOLDILOCKS_P - 1) / (n as u64));
        let inv_root_n = pow_mod(root_n, GOLDILOCKS_P - 2);
        let n_inv = pow_mod(n as u64, GOLDILOCKS_P - 2);
        let mut fwd_roots = Vec::with_capacity(log_n as usize);
        let mut inv_roots = Vec::with_capacity(log_n as usize);
        for s_idx in 0..log_n {
            let step = 1u64 << (log_n - 1 - s_idx);
            fwd_roots.push(pow_mod(root_n, step));
            inv_roots.push(pow_mod(inv_root_n, step));
        }
        Some(Self {
            n,
            log_n,
            input,
            work: vec![0u64; n],
            fwd_roots,
            inv_roots,
            root_n,
            n_inv,
        })
    }

    /// Execute one complete forward NTT -> frequency-domain cyclic shift -> inverse NTT roundtrip
    /// and verify EVERY output element bit-for-bit. Returns `Some(true)` on exact pass, `Some(false)`
    /// on hardware mismatch, or `None` if interrupted by `keep_going`.
    pub fn run_and_verify(&mut self, pass: u64, keep_going: impl Fn() -> bool) -> Option<bool> {
        let n = self.n;
        let shift = ((pass as usize) * 13 + 7) & (n - 1);
        // Seed work buffer with pass-varied affine offset (preserves cyclic convolution exactness).
        let salt = mul_mod(pass.wrapping_add(1), 0x9E37_79B9_7F4A_7C15 % GOLDILOCKS_P);
        for (dst, &src) in self.work.iter_mut().zip(&self.input) {
            *dst = add_mod(src, salt);
        }
        if !ntt_inplace(&mut self.work, &self.fwd_roots, self.log_n, &keep_going) {
            return None;
        }
        // Multiply frequency bin k by w_N^{k * shift} -> cyclic shift by +shift in time domain (y[i] = x[(i - shift) mod N]).
        let step_w = pow_mod(self.root_n, shift as u64);
        let mut cur = 1u64;
        for w in &mut self.work {
            *w = mul_mod(*w, cur);
            cur = mul_mod(cur, step_w);
        }
        if !ntt_inplace(&mut self.work, &self.inv_roots, self.log_n, &keep_going) {
            return None;
        }
        // Scale by N^-1 and verify every single element against exact expected shifted input.
        let n_inv = self.n_inv;
        let mask = n - 1;
        for i in 0..n {
            if i.is_multiple_of(8192) && !keep_going() {
                return None;
            }
            let got = mul_mod(self.work[i], n_inv);
            let exp = add_mod(self.input[(i + n - shift) & mask], salt);
            if got != exp {
                return Some(false);
            }
        }
        Some(true)
    }

    /// Equivalent scalar op count per verified NTT+INTT roundtrip (for throughput reporting).
    pub fn ops_per_pass(&self) -> u64 {
        // 2 transforms * (N/2 * log2(N) butterflies * ~8 arithmetic ops)
        2 * (self.n as u64) * (self.log_n as u64) * 5
    }
}

/// Iterative bit-reversed Cooley-Tukey radix-2 NTT over F_p.
fn ntt_inplace(
    data: &mut [u64],
    stage_roots: &[u64],
    log_n: u32,
    keep_going: &impl Fn() -> bool,
) -> bool {
    let n = data.len();
    // Bit-reversal permutation.
    let shift = usize::BITS - log_n;
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if i < j {
            data.swap(i, j);
        }
    }
    // Butterfly stages.
    for (stage, &w_len) in stage_roots.iter().enumerate() {
        if !keep_going() {
            return false;
        }
        let half = 1usize << stage;
        let len = half << 1;
        let mut base = 0usize;
        while base < n {
            if base.is_multiple_of(32_768) && !keep_going() {
                return false;
            }
            let mut w = 1u64;
            for j in 0..half {
                let u = data[base + j];
                let v = mul_mod(data[base + j + half], w);
                data[base + j] = add_mod(u, v);
                data[base + j + half] = sub_mod(u, v);
                w = mul_mod(w, w_len);
            }
            base += len;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Split-Complex Floating-Point FFT Squaring Torture (AVX2+FMA / AVX-512)
// ---------------------------------------------------------------------------
//
// Two specialized zero-bit-reversal engines, both mathematically executing exact
// cyclic complex squaring `IFFT(FFT(z)^2)` across W=4 (AVX2+FMA / Scalar) or
// W=8 (AVX-512F) independent SIMD lanes in lockstep with a strict < 0.40
// floating-point rounding check and exact DC/Nyquist complex convolution sums:
//
// 1. Bailey 4-Step L1D FMA-Saturated FFT (`N <= 256`, used by `Smallest` &
//    `Blend`'s L1 steps): factors N = R x C (8x8, 8x16, 16x16) into 2x2
//    register-blocked complex DFT GEMMs + irrational twiddles (>95% pure
//    `vfmadd231pd`/`vfnmadd231pd` inside 32 KiB L1D for peak core heat/PPT).
// 2. Zero-Bit-Reversal Fused Radix-4 DIF/DIT FFT (`N >= 512`, used by `Small`
//    L2, `Large` out-of-L3 IMC/RAM, and `Blend`): Gentleman-Sande DIF forward
//    -> fused middle-stage + pointwise square -> Cooley-Tukey DIT inverse, with
//    100% packed SIMD butterflies across every single stage (zero scalar stages,
//    zero bit-reversal permutation, pure-SIMD rounding & checksum verification).

/// FFT complex-element length ladder per preset.
/// Each thread runs `W` batched lanes (`W=4` AVX2 -> `64*N` bytes; `W=8` AVX-512 -> `128*N` bytes).
pub fn fft_ladder(preset: FftPreset) -> &'static [usize] {
    match preset {
        // 4K – 16K/thread (8–32K per physical core w/ SMT): 100% L1D resident Bailey FMA power-virus
        FftPreset::Smallest => &[64, 128, 256],
        // 64K – 256K/thread (128–512K per physical core): saturates L1<->L2 fill buffers + FMA
        FftPreset::Small => &[1_024, 2_048, 4_096],
        // 4M – 16M/thread (64–256 MiB across 16T): busts 32–96 MiB L3 cache -> hammers IMC/FCLK/RAM
        FftPreset::Large => &[65_536, 131_072, 262_144],
        // Full L1 -> L2 -> L3 -> Out-of-L3 RAM sweep for transient Vdroop & CPU+RAM Blend
        FftPreset::Blend => &[64, 256, 1_024, 4_096, 16_384, 65_536, 131_072],
    }
}

/// Working-set size in KiB per thread (`re + im` across `lanes` batched complex streams).
pub fn fft_working_kib(n: usize, lanes: usize) -> usize {
    (n * lanes * 16) >> 10
}

/// Human-readable FFT working-set size + cache-tier label (e.g. `8K L1`, `256K L2`, `8M >L3`).
pub fn fft_size_label(n: usize, lanes: usize) -> String {
    let kib = fft_working_kib(n, lanes);
    let tier = if kib <= 32 {
        "L1"
    } else if kib <= 512 {
        "L2"
    } else if kib <= 2048 {
        "L3"
    } else {
        ">L3 RAM"
    };
    if kib >= 1024 {
        format!("{}M {tier}", kib >> 10)
    } else {
        format!("{kib}K {tier}")
    }
}

/// Strict floating-point FFT rounding-error failure threshold (healthy AVX2/FMA f64 roundoff stays < 1e-6).
pub const FFT_MAX_ROUNDOFF: f64 = 0.40;

#[derive(Debug, Clone)]
pub enum FftFailure {
    /// Floating-point rounding-error trigger (`max_err >= 0.40` or NaN/Inf).
    Rounding { max_err: f64, idx: usize },
    /// Exact integer convolution DC or Nyquist sum mismatch.
    ConvolutionMismatch {
        got_dc: i128,
        expect_dc: i128,
        got_nyq: i128,
        expect_nyq: i128,
    },
}

struct Radix4Stage {
    quarter: usize,
    w1_re: Vec<f64>,
    w1_im: Vec<f64>,
    w2_re: Vec<f64>,
    w2_im: Vec<f64>,
    w3_re: Vec<f64>,
    w3_im: Vec<f64>,
}

struct BaileyPlan {
    r: usize,
    c: usize,
    wr_fwd_re: Vec<f64>,
    wr_fwd_im: Vec<f64>,
    wr_inv_re: Vec<f64>,
    wr_inv_im: Vec<f64>,
    wc_fwd_re: Vec<f64>,
    wc_fwd_im: Vec<f64>,
    wc_inv_re: Vec<f64>,
    wc_inv_im: Vec<f64>,
    tw_fwd_re: Vec<f64>,
    tw_fwd_im: Vec<f64>,
    tw_inv_re: Vec<f64>,
    tw_inv_im: Vec<f64>,
    tmp_re: Vec<f64>,
    tmp_im: Vec<f64>,
}

enum FftPlan {
    Bailey(Box<BaileyPlan>),
    Radix4 {
        stages: Vec<Radix4Stage>,
        odd_radix2: bool,
    },
}

/// SIMD-batched split-complex FFT squaring workspace (`W=4` AVX2/Scalar, `W=8` AVX-512).
pub struct FftSet {
    pub n: usize,
    pub lanes: usize,
    log_n: u32,
    pub re: Vec<f64>,
    pub im: Vec<f64>,
    base_re: Vec<f64>,
    base_im: Vec<f64>,
    base_dc_re: [f64; 8],
    base_dc_im: [f64; 8],
    base_nyq_re: [f64; 8],
    base_nyq_im: [f64; 8],
    plan: FftPlan,
}

impl FftSet {
    pub fn new(n: usize, kernel: Kernel) -> Self {
        Self::new_interruptible(n, kernel, || true).expect("uninterrupted")
    }

    pub fn new_interruptible(
        n: usize,
        kernel: Kernel,
        keep_going: impl Fn() -> bool,
    ) -> Option<Self> {
        assert!(n.is_power_of_two() && n >= 64);
        let lanes = if kernel == Kernel::Avx512 { 8 } else { 4 };
        let log_n = n.trailing_zeros();

        let total = n * lanes;
        let mut base_re = vec![0.0f64; total];
        let mut base_im = vec![0.0f64; total];
        let mut base_dc_re = [0.0f64; 8];
        let mut base_dc_im = [0.0f64; 8];
        let mut base_nyq_re = [0.0f64; 8];
        let mut base_nyq_im = [0.0f64; 8];

        // Deterministic balanced signed integers in [-7, 7] across both real & imaginary channels.
        let mut s = (n as u64).wrapping_mul(0xD1B5_4A32_D192_ED03) ^ 0x9E37_79B9_7F4A_7C15 | 1;
        for k in 0..n {
            if k.is_multiple_of(16_384) && !keep_going() {
                return None;
            }
            let alt = if (k & 1) == 0 { 1.0 } else { -1.0 };
            for l in 0..lanes {
                let r_v = ((xorshift(&mut s) % 15) as i64 - 7) as f64;
                let i_v = ((xorshift(&mut s) % 15) as i64 - 7) as f64;
                base_re[k * lanes + l] = r_v;
                base_im[k * lanes + l] = i_v;
                base_dc_re[l] += r_v;
                base_dc_im[l] += i_v;
                base_nyq_re[l] += alt * r_v;
                base_nyq_im[l] += alt * i_v;
            }
        }

        let plan = if n <= 256 {
            let (r, c) = match n {
                64 => (8, 8),
                128 => (8, 16),
                256 => (16, 16),
                _ => unreachable!("power-of-two n in 64..=256"),
            };
            let (wr_fwd_re, wr_fwd_im, wr_inv_re, wr_inv_im) = build_dft_matrix(r);
            let (wc_fwd_re, wc_fwd_im, wc_inv_re, wc_inv_im) = build_dft_matrix(c);
            let mut tw_fwd_re = Vec::with_capacity(n);
            let mut tw_fwd_im = Vec::with_capacity(n);
            let mut tw_inv_re = Vec::with_capacity(n);
            let mut tw_inv_im = Vec::with_capacity(n);
            let step = std::f64::consts::TAU / (n as f64);
            for ri in 0..r {
                for ci in 0..c {
                    let ang = step * ((ri * ci) as f64);
                    let (sv, cv) = ang.sin_cos();
                    tw_fwd_re.push(cv);
                    tw_fwd_im.push(-sv);
                    tw_inv_re.push(cv);
                    tw_inv_im.push(sv);
                }
            }
            FftPlan::Bailey(Box::new(BaileyPlan {
                r,
                c,
                wr_fwd_re,
                wr_fwd_im,
                wr_inv_re,
                wr_inv_im,
                wc_fwd_re,
                wc_fwd_im,
                wc_inv_re,
                wc_inv_im,
                tw_fwd_re,
                tw_fwd_im,
                tw_inv_re,
                tw_inv_im,
                tmp_re: vec![0.0; total],
                tmp_im: vec![0.0; total],
            }))
        } else {
            let mut stages = Vec::new();
            let mut len = n;
            while len >= 8 {
                if !keep_going() {
                    return None;
                }
                let q = len >> 2;
                let step = std::f64::consts::TAU / (len as f64);
                let mut w1_re = Vec::with_capacity(q);
                let mut w1_im = Vec::with_capacity(q);
                let mut w2_re = Vec::with_capacity(q);
                let mut w2_im = Vec::with_capacity(q);
                let mut w3_re = Vec::with_capacity(q);
                let mut w3_im = Vec::with_capacity(q);
                for j in 0..q {
                    if j.is_multiple_of(16_384) && !keep_going() {
                        return None;
                    }
                    let a1 = step * (j as f64);
                    let (s1, c1) = a1.sin_cos();
                    let (s2, c2) = (2.0 * a1).sin_cos();
                    let (s3, c3) = (3.0 * a1).sin_cos();
                    w1_re.push(c1);
                    w1_im.push(-s1);
                    w2_re.push(c2);
                    w2_im.push(-s2);
                    w3_re.push(c3);
                    w3_im.push(-s3);
                }
                stages.push(Radix4Stage {
                    quarter: q,
                    w1_re,
                    w1_im,
                    w2_re,
                    w2_im,
                    w3_re,
                    w3_im,
                });
                len >>= 2;
            }
            // len is now 4 (even log_n -> fused q=1 radix-4 middle) or 2 (odd log_n -> fused radix-2 middle).
            let odd_radix2 = len == 2;
            FftPlan::Radix4 { stages, odd_radix2 }
        };

        Some(Self {
            n,
            lanes,
            log_n,
            re: vec![0.0; total],
            im: vec![0.0; total],
            base_re,
            base_im,
            base_dc_re,
            base_dc_im,
            base_nyq_re,
            base_nyq_im,
            plan,
        })
    }

    /// Execute one verified `IFFT(FFT(z)^2)` pass across all `lanes` independent streams.
    /// Returns `None` if aborted mid-pass, `Some(Ok(max_roundoff))` on clean verification, or
    /// `Some(Err(FftFailure))` on FFT rounding (`>= 0.40` / NaN) or convolution sum mismatch.
    pub fn run_and_verify(
        &mut self,
        kernel: Kernel,
        pass_seed: u64,
        keep_going: impl Fn() -> bool,
    ) -> Option<Result<f64, FftFailure>> {
        let lanes = self.lanes;
        let n = self.n;
        let nf = n as f64;

        // Per-lane pass-varying sign flip & bounded DC/Nyquist integer salts (zero scalar modulo).
        let mut sign = [1.0f64; 8];
        let mut even_re = [0.0f64; 8];
        let mut even_im = [0.0f64; 8];
        let mut odd_re = [0.0f64; 8];
        let mut odd_im = [0.0f64; 8];
        let mut exp_dc_re = [0.0f64; 8];
        let mut exp_dc_im = [0.0f64; 8];
        let mut exp_nyq_re = [0.0f64; 8];
        let mut exp_nyq_im = [0.0f64; 8];
        const SALT_TAB: [f64; 4] = [-2.0, -1.0, 1.0, 2.0];

        let mut h = pass_seed ^ 0x9E37_79B9_7F4A_7C15 | 1;
        for l in 0..lanes {
            let bits = xorshift(&mut h);
            let sg = if (bits & 1) == 0 { 1.0 } else { -1.0 };
            let sr = SALT_TAB[((bits >> 1) & 3) as usize];
            let si = SALT_TAB[((bits >> 3) & 3) as usize];
            let ar = if ((bits >> 5) & 1) == 0 { 1.0 } else { -1.0 };
            let ai = if ((bits >> 6) & 1) == 0 { 1.0 } else { -1.0 };
            sign[l] = sg;
            even_re[l] = sr + ar;
            even_im[l] = si + ai;
            odd_re[l] = sr - ar;
            odd_im[l] = si - ai;

            let dc_r = sg * self.base_dc_re[l] + nf * sr;
            let dc_i = sg * self.base_dc_im[l] + nf * si;
            let nyq_r = sg * self.base_nyq_re[l] + nf * ar;
            let nyq_i = sg * self.base_nyq_im[l] + nf * ai;
            exp_dc_re[l] = dc_r * dc_r - dc_i * dc_i;
            exp_dc_im[l] = 2.0 * dc_r * dc_i;
            exp_nyq_re[l] = nyq_r * nyq_r - nyq_i * nyq_i;
            exp_nyq_im[l] = 2.0 * nyq_r * nyq_i;
        }

        // Vectorized input preparation (alternating even/odd salts).
        self.prepare_input(kernel, &sign, &even_re, &even_im, &odd_re, &odd_im);

        if !self.execute_square_transform(kernel, &keep_going) {
            return None;
        }

        Some(self.verify_output(kernel, &exp_dc_re, &exp_dc_im, &exp_nyq_re, &exp_nyq_im))
    }

    /// Execute the full `IFFT(FFT(z)^2)` transform in place on `(self.re, self.im)`.
    pub fn execute_square_transform(
        &mut self,
        kernel: Kernel,
        keep_going: &impl Fn() -> bool,
    ) -> bool {
        match &mut self.plan {
            FftPlan::Bailey(b) => {
                if !keep_going() {
                    return false;
                }
                bailey_square_inplace(kernel, self.lanes, &mut self.re, &mut self.im, b);
                true
            }
            FftPlan::Radix4 { stages, odd_radix2 } => {
                for st in stages.iter() {
                    if !keep_going() {
                        return false;
                    }
                    radix4_dif_stage(kernel, self.lanes, &mut self.re, &mut self.im, st);
                }
                if !keep_going() {
                    return false;
                }
                if *odd_radix2 {
                    mid_radix2_square_fused(kernel, self.lanes, &mut self.re, &mut self.im);
                } else {
                    mid_radix4_square_fused(kernel, self.lanes, &mut self.re, &mut self.im);
                }
                for st in stages.iter().rev() {
                    if !keep_going() {
                        return false;
                    }
                    radix4_dit_stage(kernel, self.lanes, &mut self.re, &mut self.im, st);
                }
                true
            }
        }
    }

    /// True FLOP count executed per verified pass across all `lanes` independent FFTs.
    pub fn flops_per_pass(&self) -> u64 {
        let l = self.lanes as u64;
        let n = self.n as u64;
        match &self.plan {
            FftPlan::Bailey(b) => {
                // 2 * (8 * R^2 * C + 6 * N + 8 * R * C^2) + 6 * N
                l * (16 * n * ((b.r + b.c) as u64) + 18 * n)
            }
            FftPlan::Radix4 { .. } => {
                // Forward + Inverse FFT (2 * 5 * N * log2(N)) + complex square (6 * N)
                l * (10 * n * (self.log_n as u64) + 6 * n)
            }
        }
    }

    fn prepare_input(
        &mut self,
        kernel: Kernel,
        sign: &[f64; 8],
        even_re: &[f64; 8],
        even_im: &[f64; 8],
        odd_re: &[f64; 8],
        odd_im: &[f64; 8],
    ) {
        match kernel {
            Kernel::Avx512 if self.lanes == 8 => unsafe {
                prepare_input_avx512(
                    &self.base_re,
                    &self.base_im,
                    &mut self.re,
                    &mut self.im,
                    sign,
                    even_re,
                    even_im,
                    odd_re,
                    odd_im,
                );
            },
            Kernel::Avx2Fma | Kernel::Avx512 if self.lanes == 4 => unsafe {
                prepare_input_avx2(
                    &self.base_re,
                    &self.base_im,
                    &mut self.re,
                    &mut self.im,
                    sign,
                    even_re,
                    even_im,
                    odd_re,
                    odd_im,
                );
            },
            _ => {
                let lanes = self.lanes;
                for k in (0..self.n).step_by(2) {
                    let e = k * lanes;
                    let o = (k + 1) * lanes;
                    for l in 0..lanes {
                        self.re[e + l] = sign[l].mul_add(self.base_re[e + l], even_re[l]);
                        self.im[e + l] = sign[l].mul_add(self.base_im[e + l], even_im[l]);
                        self.re[o + l] = sign[l].mul_add(self.base_re[o + l], odd_re[l]);
                        self.im[o + l] = sign[l].mul_add(self.base_im[o + l], odd_im[l]);
                    }
                }
            }
        }
    }

    fn verify_output(
        &self,
        kernel: Kernel,
        exp_dc_re: &[f64; 8],
        exp_dc_im: &[f64; 8],
        exp_nyq_re: &[f64; 8],
        exp_nyq_im: &[f64; 8],
    ) -> Result<f64, FftFailure> {
        let inv_n = 1.0 / (self.n as f64);
        let mut max_err_lanes = [0.0f64; 8];
        let mut got_dc_re = [0.0f64; 8];
        let mut got_dc_im = [0.0f64; 8];
        let mut got_nyq_re = [0.0f64; 8];
        let mut got_nyq_im = [0.0f64; 8];

        match kernel {
            Kernel::Avx512 if self.lanes == 8 => unsafe {
                verify_output_avx512(
                    &self.re,
                    &self.im,
                    inv_n,
                    &mut max_err_lanes,
                    &mut got_dc_re,
                    &mut got_dc_im,
                    &mut got_nyq_re,
                    &mut got_nyq_im,
                );
            },
            Kernel::Avx2Fma | Kernel::Avx512 if self.lanes == 4 => unsafe {
                verify_output_avx2(
                    &self.re,
                    &self.im,
                    inv_n,
                    &mut max_err_lanes,
                    &mut got_dc_re,
                    &mut got_dc_im,
                    &mut got_nyq_re,
                    &mut got_nyq_im,
                );
            },
            _ => {
                let lanes = self.lanes;
                for k in (0..self.n).step_by(2) {
                    let e = k * lanes;
                    let o = (k + 1) * lanes;
                    for l in 0..lanes {
                        let re_e = self.re[e + l] * inv_n;
                        let im_e = self.im[e + l] * inv_n;
                        let re_ei = re_e.round();
                        let im_ei = im_e.round();
                        let err_e = (re_e - re_ei).abs().max((im_e - im_ei).abs());

                        let re_o = self.re[o + l] * inv_n;
                        let im_o = self.im[o + l] * inv_n;
                        let re_oi = re_o.round();
                        let im_oi = im_o.round();
                        let err_o = (re_o - re_oi).abs().max((im_o - im_oi).abs());

                        let err = if err_e.is_nan() || err_o.is_nan() {
                            f64::NAN
                        } else {
                            err_e.max(err_o)
                        };
                        if err.is_nan() || err > max_err_lanes[l] {
                            max_err_lanes[l] = err;
                        }
                        got_dc_re[l] += re_ei + re_oi;
                        got_dc_im[l] += im_ei + im_oi;
                        got_nyq_re[l] += re_ei - re_oi;
                        got_nyq_im[l] += im_ei - im_oi;
                    }
                }
            }
        }

        let mut overall_max = 0.0f64;
        for (l, &e) in max_err_lanes.iter().enumerate().take(self.lanes) {
            // Also check got_dc_re/im finiteness: x86 vmaxpd drops a NaN in operand 1 on subsequent
            // iterations, whereas IEEE-754 addition keeps NaN/Inf sticky in dc_re/dc_im forever.
            if !e.is_finite()
                || e >= FFT_MAX_ROUNDOFF
                || !got_dc_re[l].is_finite()
                || !got_dc_im[l].is_finite()
            {
                // Locate exact failing element index for diagnostics (error path only).
                let mut fail_idx = 0usize;
                let mut fail_val = e;
                for k in 0..self.n {
                    let r = self.re[k * self.lanes + l] * inv_n;
                    let im_v = self.im[k * self.lanes + l] * inv_n;
                    if !r.is_finite() || !im_v.is_finite() {
                        fail_idx = k;
                        fail_val = f64::INFINITY;
                        break;
                    }
                    let d = (r - r.round()).abs().max((im_v - im_v.round()).abs());
                    if d >= FFT_MAX_ROUNDOFF {
                        fail_idx = k;
                        fail_val = d;
                        break;
                    }
                }
                return Err(FftFailure::Rounding {
                    max_err: fail_val,
                    idx: fail_idx,
                });
            }
            if e > overall_max {
                overall_max = e;
            }
        }

        for l in 0..self.lanes {
            if got_dc_re[l] != exp_dc_re[l]
                || got_dc_im[l] != exp_dc_im[l]
                || got_nyq_re[l] != exp_nyq_re[l]
                || got_nyq_im[l] != exp_nyq_im[l]
            {
                return Err(FftFailure::ConvolutionMismatch {
                    got_dc: got_dc_re[l] as i128,
                    expect_dc: exp_dc_re[l] as i128,
                    got_nyq: got_nyq_re[l] as i128,
                    expect_nyq: exp_nyq_re[l] as i128,
                });
            }
        }

        Ok(overall_max)
    }
}

fn build_dft_matrix(m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut fwd_re = Vec::with_capacity(m * m);
    let mut fwd_im = Vec::with_capacity(m * m);
    let mut inv_re = Vec::with_capacity(m * m);
    let mut inv_im = Vec::with_capacity(m * m);
    let step = std::f64::consts::TAU / (m as f64);
    for r in 0..m {
        for c in 0..m {
            let ang = step * ((r * c) as f64);
            let (s, cv) = ang.sin_cos();
            fwd_re.push(cv);
            fwd_im.push(-s);
            inv_re.push(cv);
            inv_im.push(s);
        }
    }
    (fwd_re, fwd_im, inv_re, inv_im)
}

// ---------------------------------------------------------------------------
// Input Prep & FFT Rounding Verification SIMD Kernels
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn prepare_input_avx2(
    base_re: &[f64],
    base_im: &[f64],
    re: &mut [f64],
    im: &mut [f64],
    sign: &[f64; 8],
    even_re: &[f64; 8],
    even_im: &[f64; 8],
    odd_re: &[f64; 8],
    odd_im: &[f64; 8],
) {
    use std::arch::x86_64::*;
    let sg = _mm256_loadu_pd(sign.as_ptr());
    let er = _mm256_loadu_pd(even_re.as_ptr());
    let ei = _mm256_loadu_pd(even_im.as_ptr());
    let or_ = _mm256_loadu_pd(odd_re.as_ptr());
    let oi = _mm256_loadu_pd(odd_im.as_ptr());
    let brp = base_re.as_ptr();
    let bip = base_im.as_ptr();
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let total = re.len();
    let mut idx = 0usize;
    while idx < total {
        let vr0 = _mm256_loadu_pd(brp.add(idx));
        let vi0 = _mm256_loadu_pd(bip.add(idx));
        let vr1 = _mm256_loadu_pd(brp.add(idx + 4));
        let vi1 = _mm256_loadu_pd(bip.add(idx + 4));
        _mm256_storeu_pd(rp.add(idx), _mm256_fmadd_pd(sg, vr0, er));
        _mm256_storeu_pd(ip.add(idx), _mm256_fmadd_pd(sg, vi0, ei));
        _mm256_storeu_pd(rp.add(idx + 4), _mm256_fmadd_pd(sg, vr1, or_));
        _mm256_storeu_pd(ip.add(idx + 4), _mm256_fmadd_pd(sg, vi1, oi));
        idx += 8;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
unsafe fn prepare_input_avx512(
    base_re: &[f64],
    base_im: &[f64],
    re: &mut [f64],
    im: &mut [f64],
    sign: &[f64; 8],
    even_re: &[f64; 8],
    even_im: &[f64; 8],
    odd_re: &[f64; 8],
    odd_im: &[f64; 8],
) {
    use std::arch::x86_64::*;
    let sg = _mm512_loadu_pd(sign.as_ptr());
    let er = _mm512_loadu_pd(even_re.as_ptr());
    let ei = _mm512_loadu_pd(even_im.as_ptr());
    let or_ = _mm512_loadu_pd(odd_re.as_ptr());
    let oi = _mm512_loadu_pd(odd_im.as_ptr());
    let brp = base_re.as_ptr();
    let bip = base_im.as_ptr();
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let total = re.len();
    let mut idx = 0usize;
    while idx < total {
        let vr0 = _mm512_loadu_pd(brp.add(idx));
        let vi0 = _mm512_loadu_pd(bip.add(idx));
        let vr1 = _mm512_loadu_pd(brp.add(idx + 8));
        let vi1 = _mm512_loadu_pd(bip.add(idx + 8));
        _mm512_storeu_pd(rp.add(idx), _mm512_fmadd_pd(sg, vr0, er));
        _mm512_storeu_pd(ip.add(idx), _mm512_fmadd_pd(sg, vi0, ei));
        _mm512_storeu_pd(rp.add(idx + 8), _mm512_fmadd_pd(sg, vr1, or_));
        _mm512_storeu_pd(ip.add(idx + 8), _mm512_fmadd_pd(sg, vi1, oi));
        idx += 16;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn verify_output_avx2(
    re: &[f64],
    im: &[f64],
    inv_n: f64,
    max_err_out: &mut [f64; 8],
    dc_re_out: &mut [f64; 8],
    dc_im_out: &mut [f64; 8],
    nyq_re_out: &mut [f64; 8],
    nyq_im_out: &mut [f64; 8],
) {
    use std::arch::x86_64::*;
    let inv = _mm256_set1_pd(inv_n);
    let sign_bit = _mm256_set1_pd(-0.0);
    let mut max_err = _mm256_setzero_pd();
    let mut dc_re = _mm256_setzero_pd();
    let mut dc_im = _mm256_setzero_pd();
    let mut nyq_re = _mm256_setzero_pd();
    let mut nyq_im = _mm256_setzero_pd();

    let rp = re.as_ptr();
    let ip = im.as_ptr();
    let total = re.len();
    let mut idx = 0usize;
    while idx < total {
        let re_e = _mm256_mul_pd(_mm256_loadu_pd(rp.add(idx)), inv);
        let im_e = _mm256_mul_pd(_mm256_loadu_pd(ip.add(idx)), inv);
        let re_ei = _mm256_round_pd(re_e, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let im_ei = _mm256_round_pd(im_e, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let d_re_e = _mm256_andnot_pd(sign_bit, _mm256_sub_pd(re_e, re_ei));
        let d_im_e = _mm256_andnot_pd(sign_bit, _mm256_sub_pd(im_e, im_ei));
        // Second operand propagates NaN in x86 vmaxpd.
        max_err = _mm256_max_pd(max_err, d_re_e);
        max_err = _mm256_max_pd(max_err, d_im_e);

        let re_o = _mm256_mul_pd(_mm256_loadu_pd(rp.add(idx + 4)), inv);
        let im_o = _mm256_mul_pd(_mm256_loadu_pd(ip.add(idx + 4)), inv);
        let re_oi = _mm256_round_pd(re_o, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let im_oi = _mm256_round_pd(im_o, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let d_re_o = _mm256_andnot_pd(sign_bit, _mm256_sub_pd(re_o, re_oi));
        let d_im_o = _mm256_andnot_pd(sign_bit, _mm256_sub_pd(im_o, im_oi));
        max_err = _mm256_max_pd(max_err, d_re_o);
        max_err = _mm256_max_pd(max_err, d_im_o);

        dc_re = _mm256_add_pd(dc_re, _mm256_add_pd(re_ei, re_oi));
        dc_im = _mm256_add_pd(dc_im, _mm256_add_pd(im_ei, im_oi));
        nyq_re = _mm256_add_pd(nyq_re, _mm256_sub_pd(re_ei, re_oi));
        nyq_im = _mm256_add_pd(nyq_im, _mm256_sub_pd(im_ei, im_oi));
        idx += 8;
    }

    _mm256_storeu_pd(max_err_out.as_mut_ptr(), max_err);
    _mm256_storeu_pd(dc_re_out.as_mut_ptr(), dc_re);
    _mm256_storeu_pd(dc_im_out.as_mut_ptr(), dc_im);
    _mm256_storeu_pd(nyq_re_out.as_mut_ptr(), nyq_re);
    _mm256_storeu_pd(nyq_im_out.as_mut_ptr(), nyq_im);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
unsafe fn verify_output_avx512(
    re: &[f64],
    im: &[f64],
    inv_n: f64,
    max_err_out: &mut [f64; 8],
    dc_re_out: &mut [f64; 8],
    dc_im_out: &mut [f64; 8],
    nyq_re_out: &mut [f64; 8],
    nyq_im_out: &mut [f64; 8],
) {
    use std::arch::x86_64::*;
    let inv = _mm512_set1_pd(inv_n);
    let mut max_err = _mm512_setzero_pd();
    let mut dc_re = _mm512_setzero_pd();
    let mut dc_im = _mm512_setzero_pd();
    let mut nyq_re = _mm512_setzero_pd();
    let mut nyq_im = _mm512_setzero_pd();

    let rp = re.as_ptr();
    let ip = im.as_ptr();
    let total = re.len();
    let mut idx = 0usize;
    while idx < total {
        let re_e = _mm512_mul_pd(_mm512_loadu_pd(rp.add(idx)), inv);
        let im_e = _mm512_mul_pd(_mm512_loadu_pd(ip.add(idx)), inv);
        let re_ei = _mm512_roundscale_pd::<0x08>(re_e);
        let im_ei = _mm512_roundscale_pd::<0x08>(im_e);
        let d_re_e = _mm512_abs_pd(_mm512_sub_pd(re_e, re_ei));
        let d_im_e = _mm512_abs_pd(_mm512_sub_pd(im_e, im_ei));
        max_err = _mm512_max_pd(max_err, d_re_e);
        max_err = _mm512_max_pd(max_err, d_im_e);

        let re_o = _mm512_mul_pd(_mm512_loadu_pd(rp.add(idx + 8)), inv);
        let im_o = _mm512_mul_pd(_mm512_loadu_pd(ip.add(idx + 8)), inv);
        let re_oi = _mm512_roundscale_pd::<0x08>(re_o);
        let im_oi = _mm512_roundscale_pd::<0x08>(im_o);
        let d_re_o = _mm512_abs_pd(_mm512_sub_pd(re_o, re_oi));
        let d_im_o = _mm512_abs_pd(_mm512_sub_pd(im_o, im_oi));
        max_err = _mm512_max_pd(max_err, d_re_o);
        max_err = _mm512_max_pd(max_err, d_im_o);

        dc_re = _mm512_add_pd(dc_re, _mm512_add_pd(re_ei, re_oi));
        dc_im = _mm512_add_pd(dc_im, _mm512_add_pd(im_ei, im_oi));
        nyq_re = _mm512_add_pd(nyq_re, _mm512_sub_pd(re_ei, re_oi));
        nyq_im = _mm512_add_pd(nyq_im, _mm512_sub_pd(im_ei, im_oi));
        idx += 16;
    }

    _mm512_storeu_pd(max_err_out.as_mut_ptr(), max_err);
    _mm512_storeu_pd(dc_re_out.as_mut_ptr(), dc_re);
    _mm512_storeu_pd(dc_im_out.as_mut_ptr(), dc_im);
    _mm512_storeu_pd(nyq_re_out.as_mut_ptr(), nyq_re);
    _mm512_storeu_pd(nyq_im_out.as_mut_ptr(), nyq_im);
}

// ---------------------------------------------------------------------------
// Engine 1: Bailey 4-Step L1D FMA-Saturated FFT (Smallest FFTs Power-Virus)
// ---------------------------------------------------------------------------

fn bailey_square_inplace(
    kernel: Kernel,
    lanes: usize,
    re: &mut [f64],
    im: &mut [f64],
    b: &mut BaileyPlan,
) {
    match kernel {
        Kernel::Avx512 if lanes == 8 => unsafe {
            bailey_col_dft_avx512(
                re,
                im,
                &mut b.tmp_re,
                &mut b.tmp_im,
                &b.wr_fwd_re,
                &b.wr_fwd_im,
                b.r,
                b.c,
            );
            bailey_twiddle_avx512(&mut b.tmp_re, &mut b.tmp_im, &b.tw_fwd_re, &b.tw_fwd_im);
            bailey_row_dft_avx512(
                &b.tmp_re,
                &b.tmp_im,
                re,
                im,
                &b.wc_fwd_re,
                &b.wc_fwd_im,
                b.r,
                b.c,
            );
            square_complex_avx512(re, im);
            bailey_row_dft_avx512(
                re,
                im,
                &mut b.tmp_re,
                &mut b.tmp_im,
                &b.wc_inv_re,
                &b.wc_inv_im,
                b.r,
                b.c,
            );
            bailey_twiddle_avx512(&mut b.tmp_re, &mut b.tmp_im, &b.tw_inv_re, &b.tw_inv_im);
            bailey_col_dft_avx512(
                &b.tmp_re,
                &b.tmp_im,
                re,
                im,
                &b.wr_inv_re,
                &b.wr_inv_im,
                b.r,
                b.c,
            );
        },
        Kernel::Avx2Fma | Kernel::Avx512 if lanes == 4 => unsafe {
            bailey_col_dft_avx2(
                re,
                im,
                &mut b.tmp_re,
                &mut b.tmp_im,
                &b.wr_fwd_re,
                &b.wr_fwd_im,
                b.r,
                b.c,
            );
            bailey_twiddle_avx2(&mut b.tmp_re, &mut b.tmp_im, &b.tw_fwd_re, &b.tw_fwd_im);
            bailey_row_dft_avx2(
                &b.tmp_re,
                &b.tmp_im,
                re,
                im,
                &b.wc_fwd_re,
                &b.wc_fwd_im,
                b.r,
                b.c,
            );
            square_complex_avx2(re, im);
            bailey_row_dft_avx2(
                re,
                im,
                &mut b.tmp_re,
                &mut b.tmp_im,
                &b.wc_inv_re,
                &b.wc_inv_im,
                b.r,
                b.c,
            );
            bailey_twiddle_avx2(&mut b.tmp_re, &mut b.tmp_im, &b.tw_inv_re, &b.tw_inv_im);
            bailey_col_dft_avx2(
                &b.tmp_re,
                &b.tmp_im,
                re,
                im,
                &b.wr_inv_re,
                &b.wr_inv_im,
                b.r,
                b.c,
            );
        },
        _ => {
            bailey_col_dft_scalar(
                re,
                im,
                &mut b.tmp_re,
                &mut b.tmp_im,
                &b.wr_fwd_re,
                &b.wr_fwd_im,
                b.r,
                b.c,
                lanes,
            );
            bailey_twiddle_scalar(
                &mut b.tmp_re,
                &mut b.tmp_im,
                &b.tw_fwd_re,
                &b.tw_fwd_im,
                lanes,
            );
            bailey_row_dft_scalar(
                &b.tmp_re,
                &b.tmp_im,
                re,
                im,
                &b.wc_fwd_re,
                &b.wc_fwd_im,
                b.r,
                b.c,
                lanes,
            );
            square_complex_scalar(re, im);
            bailey_row_dft_scalar(
                re,
                im,
                &mut b.tmp_re,
                &mut b.tmp_im,
                &b.wc_inv_re,
                &b.wc_inv_im,
                b.r,
                b.c,
                lanes,
            );
            bailey_twiddle_scalar(
                &mut b.tmp_re,
                &mut b.tmp_im,
                &b.tw_inv_re,
                &b.tw_inv_im,
                lanes,
            );
            bailey_col_dft_scalar(
                &b.tmp_re,
                &b.tmp_im,
                re,
                im,
                &b.wr_inv_re,
                &b.wr_inv_im,
                b.r,
                b.c,
                lanes,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bailey_col_dft_scalar(
    src_re: &[f64],
    src_im: &[f64],
    dst_re: &mut [f64],
    dst_im: &mut [f64],
    w_re: &[f64],
    w_im: &[f64],
    r: usize,
    c: usize,
    lanes: usize,
) {
    for ri in 0..r {
        for ci in 0..c {
            let out = (ri * c + ci) * lanes;
            for l in 0..lanes {
                let (mut ar, mut ai) = (0.0f64, 0.0f64);
                for k in 0..r {
                    let idx = (k * c + ci) * lanes + l;
                    let xr = src_re[idx];
                    let xi = src_im[idx];
                    let wr = w_re[ri * r + k];
                    let wi = w_im[ri * r + k];
                    ar = xr.mul_add(wr, ar);
                    ar = (-xi).mul_add(wi, ar);
                    ai = xr.mul_add(wi, ai);
                    ai = xi.mul_add(wr, ai);
                }
                dst_re[out + l] = ar;
                dst_im[out + l] = ai;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bailey_row_dft_scalar(
    src_re: &[f64],
    src_im: &[f64],
    dst_re: &mut [f64],
    dst_im: &mut [f64],
    w_re: &[f64],
    w_im: &[f64],
    r: usize,
    c: usize,
    lanes: usize,
) {
    for ri in 0..r {
        let row_base = ri * c;
        for m in 0..c {
            let out = (row_base + m) * lanes;
            for l in 0..lanes {
                let (mut ar, mut ai) = (0.0f64, 0.0f64);
                for ci in 0..c {
                    let idx = (row_base + ci) * lanes + l;
                    let xr = src_re[idx];
                    let xi = src_im[idx];
                    let wr = w_re[m * c + ci];
                    let wi = w_im[m * c + ci];
                    ar = xr.mul_add(wr, ar);
                    ar = (-xi).mul_add(wi, ar);
                    ai = xr.mul_add(wi, ai);
                    ai = xi.mul_add(wr, ai);
                }
                dst_re[out + l] = ar;
                dst_im[out + l] = ai;
            }
        }
    }
}

fn bailey_twiddle_scalar(
    re: &mut [f64],
    im: &mut [f64],
    tw_re: &[f64],
    tw_im: &[f64],
    lanes: usize,
) {
    for (idx, (&wr, &wi)) in tw_re.iter().zip(tw_im.iter()).enumerate() {
        let base = idx * lanes;
        for l in 0..lanes {
            let xr = re[base + l];
            let xi = im[base + l];
            re[base + l] = xr.mul_add(wr, -(xi * wi));
            im[base + l] = xr.mul_add(wi, xi * wr);
        }
    }
}

/// 2x2 register-blocked pure-FMA column DFT (`dst[r, c] = sum_k W[r, k] * src[k, c]`).
/// Holds 8 independent 256-bit accumulator registers to keep both AVX2 FMA pipes 100% saturated.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn bailey_col_dft_avx2(
    src_re: &[f64],
    src_im: &[f64],
    dst_re: &mut [f64],
    dst_im: &mut [f64],
    w_re: &[f64],
    w_im: &[f64],
    r: usize,
    c: usize,
) {
    use std::arch::x86_64::*;
    let srp = src_re.as_ptr();
    let sip = src_im.as_ptr();
    let drp = dst_re.as_mut_ptr();
    let dip = dst_im.as_mut_ptr();
    let wrp = w_re.as_ptr();
    let wip = w_im.as_ptr();
    let row_stride = c * 4;

    let mut ri = 0usize;
    while ri < r {
        let w0_off = ri * r;
        let w1_off = (ri + 1) * r;
        let mut ci = 0usize;
        while ci < c {
            let mut a00r = _mm256_setzero_pd();
            let mut a00i = _mm256_setzero_pd();
            let mut a01r = _mm256_setzero_pd();
            let mut a01i = _mm256_setzero_pd();
            let mut a10r = _mm256_setzero_pd();
            let mut a10i = _mm256_setzero_pd();
            let mut a11r = _mm256_setzero_pd();
            let mut a11i = _mm256_setzero_pd();

            let mut src_off = ci * 4;
            for k in 0..r {
                let x0r = _mm256_loadu_pd(srp.add(src_off));
                let x0i = _mm256_loadu_pd(sip.add(src_off));
                let x1r = _mm256_loadu_pd(srp.add(src_off + 4));
                let x1i = _mm256_loadu_pd(sip.add(src_off + 4));

                let w0r = _mm256_set1_pd(*wrp.add(w0_off + k));
                let w0i = _mm256_set1_pd(*wip.add(w0_off + k));
                a00r = _mm256_fmadd_pd(x0r, w0r, a00r);
                a00r = _mm256_fnmadd_pd(x0i, w0i, a00r);
                a00i = _mm256_fmadd_pd(x0r, w0i, a00i);
                a00i = _mm256_fmadd_pd(x0i, w0r, a00i);
                a01r = _mm256_fmadd_pd(x1r, w0r, a01r);
                a01r = _mm256_fnmadd_pd(x1i, w0i, a01r);
                a01i = _mm256_fmadd_pd(x1r, w0i, a01i);
                a01i = _mm256_fmadd_pd(x1i, w0r, a01i);

                let w1r = _mm256_set1_pd(*wrp.add(w1_off + k));
                let w1i = _mm256_set1_pd(*wip.add(w1_off + k));
                a10r = _mm256_fmadd_pd(x0r, w1r, a10r);
                a10r = _mm256_fnmadd_pd(x0i, w1i, a10r);
                a10i = _mm256_fmadd_pd(x0r, w1i, a10i);
                a10i = _mm256_fmadd_pd(x0i, w1r, a10i);
                a11r = _mm256_fmadd_pd(x1r, w1r, a11r);
                a11r = _mm256_fnmadd_pd(x1i, w1i, a11r);
                a11i = _mm256_fmadd_pd(x1r, w1i, a11i);
                a11i = _mm256_fmadd_pd(x1i, w1r, a11i);

                src_off += row_stride;
            }

            let d0 = (ri * c + ci) * 4;
            let d1 = ((ri + 1) * c + ci) * 4;
            _mm256_storeu_pd(drp.add(d0), a00r);
            _mm256_storeu_pd(dip.add(d0), a00i);
            _mm256_storeu_pd(drp.add(d0 + 4), a01r);
            _mm256_storeu_pd(dip.add(d0 + 4), a01i);
            _mm256_storeu_pd(drp.add(d1), a10r);
            _mm256_storeu_pd(dip.add(d1), a10i);
            _mm256_storeu_pd(drp.add(d1 + 4), a11r);
            _mm256_storeu_pd(dip.add(d1 + 4), a11i);
            ci += 2;
        }
        ri += 2;
    }
}

/// 2x2 register-blocked pure-FMA row DFT (`dst[r, m] = sum_c src[r, c] * W[m, c]`).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn bailey_row_dft_avx2(
    src_re: &[f64],
    src_im: &[f64],
    dst_re: &mut [f64],
    dst_im: &mut [f64],
    w_re: &[f64],
    w_im: &[f64],
    r: usize,
    c: usize,
) {
    use std::arch::x86_64::*;
    let srp = src_re.as_ptr();
    let sip = src_im.as_ptr();
    let drp = dst_re.as_mut_ptr();
    let dip = dst_im.as_mut_ptr();
    let wrp = w_re.as_ptr();
    let wip = w_im.as_ptr();

    let mut ri = 0usize;
    while ri < r {
        let r0_base = ri * c * 4;
        let r1_base = (ri + 1) * c * 4;
        let mut m = 0usize;
        while m < c {
            let mut a00r = _mm256_setzero_pd();
            let mut a00i = _mm256_setzero_pd();
            let mut a01r = _mm256_setzero_pd();
            let mut a01i = _mm256_setzero_pd();
            let mut a10r = _mm256_setzero_pd();
            let mut a10i = _mm256_setzero_pd();
            let mut a11r = _mm256_setzero_pd();
            let mut a11i = _mm256_setzero_pd();

            let w0_off = m * c;
            let w1_off = (m + 1) * c;
            for ci in 0..c {
                let x0r = _mm256_loadu_pd(srp.add(r0_base + ci * 4));
                let x0i = _mm256_loadu_pd(sip.add(r0_base + ci * 4));
                let x1r = _mm256_loadu_pd(srp.add(r1_base + ci * 4));
                let x1i = _mm256_loadu_pd(sip.add(r1_base + ci * 4));

                let w0r = _mm256_set1_pd(*wrp.add(w0_off + ci));
                let w0i = _mm256_set1_pd(*wip.add(w0_off + ci));
                a00r = _mm256_fmadd_pd(x0r, w0r, a00r);
                a00r = _mm256_fnmadd_pd(x0i, w0i, a00r);
                a00i = _mm256_fmadd_pd(x0r, w0i, a00i);
                a00i = _mm256_fmadd_pd(x0i, w0r, a00i);
                a10r = _mm256_fmadd_pd(x1r, w0r, a10r);
                a10r = _mm256_fnmadd_pd(x1i, w0i, a10r);
                a10i = _mm256_fmadd_pd(x1r, w0i, a10i);
                a10i = _mm256_fmadd_pd(x1i, w0r, a10i);

                let w1r = _mm256_set1_pd(*wrp.add(w1_off + ci));
                let w1i = _mm256_set1_pd(*wip.add(w1_off + ci));
                a01r = _mm256_fmadd_pd(x0r, w1r, a01r);
                a01r = _mm256_fnmadd_pd(x0i, w1i, a01r);
                a01i = _mm256_fmadd_pd(x0r, w1i, a01i);
                a01i = _mm256_fmadd_pd(x0i, w1r, a01i);
                a11r = _mm256_fmadd_pd(x1r, w1r, a11r);
                a11r = _mm256_fnmadd_pd(x1i, w1i, a11r);
                a11i = _mm256_fmadd_pd(x1r, w1i, a11i);
                a11i = _mm256_fmadd_pd(x1i, w1r, a11i);
            }

            _mm256_storeu_pd(drp.add(r0_base + m * 4), a00r);
            _mm256_storeu_pd(dip.add(r0_base + m * 4), a00i);
            _mm256_storeu_pd(drp.add(r0_base + (m + 1) * 4), a01r);
            _mm256_storeu_pd(dip.add(r0_base + (m + 1) * 4), a01i);
            _mm256_storeu_pd(drp.add(r1_base + m * 4), a10r);
            _mm256_storeu_pd(dip.add(r1_base + m * 4), a10i);
            _mm256_storeu_pd(drp.add(r1_base + (m + 1) * 4), a11r);
            _mm256_storeu_pd(dip.add(r1_base + (m + 1) * 4), a11i);
            m += 2;
        }
        ri += 2;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn bailey_twiddle_avx2(re: &mut [f64], im: &mut [f64], tw_re: &[f64], tw_im: &[f64]) {
    use std::arch::x86_64::*;
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let wrp = tw_re.as_ptr();
    let wip = tw_im.as_ptr();
    for idx in 0..tw_re.len() {
        let off = idx * 4;
        let xr = _mm256_loadu_pd(rp.add(off));
        let xi = _mm256_loadu_pd(ip.add(off));
        let wr = _mm256_set1_pd(*wrp.add(idx));
        let wi = _mm256_set1_pd(*wip.add(idx));
        let nr = _mm256_fmsub_pd(xr, wr, _mm256_mul_pd(xi, wi));
        let ni = _mm256_fmadd_pd(xr, wi, _mm256_mul_pd(xi, wr));
        _mm256_storeu_pd(rp.add(off), nr);
        _mm256_storeu_pd(ip.add(off), ni);
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
unsafe fn bailey_col_dft_avx512(
    src_re: &[f64],
    src_im: &[f64],
    dst_re: &mut [f64],
    dst_im: &mut [f64],
    w_re: &[f64],
    w_im: &[f64],
    r: usize,
    c: usize,
) {
    use std::arch::x86_64::*;
    let srp = src_re.as_ptr();
    let sip = src_im.as_ptr();
    let drp = dst_re.as_mut_ptr();
    let dip = dst_im.as_mut_ptr();
    let wrp = w_re.as_ptr();
    let wip = w_im.as_ptr();
    let row_stride = c * 8;

    let mut ri = 0usize;
    while ri < r {
        let w0_off = ri * r;
        let w1_off = (ri + 1) * r;
        let mut ci = 0usize;
        while ci < c {
            let mut a00r = _mm512_setzero_pd();
            let mut a00i = _mm512_setzero_pd();
            let mut a01r = _mm512_setzero_pd();
            let mut a01i = _mm512_setzero_pd();
            let mut a10r = _mm512_setzero_pd();
            let mut a10i = _mm512_setzero_pd();
            let mut a11r = _mm512_setzero_pd();
            let mut a11i = _mm512_setzero_pd();

            let mut src_off = ci * 8;
            for k in 0..r {
                let x0r = _mm512_loadu_pd(srp.add(src_off));
                let x0i = _mm512_loadu_pd(sip.add(src_off));
                let x1r = _mm512_loadu_pd(srp.add(src_off + 8));
                let x1i = _mm512_loadu_pd(sip.add(src_off + 8));

                let w0r = _mm512_set1_pd(*wrp.add(w0_off + k));
                let w0i = _mm512_set1_pd(*wip.add(w0_off + k));
                a00r = _mm512_fmadd_pd(x0r, w0r, a00r);
                a00r = _mm512_fnmadd_pd(x0i, w0i, a00r);
                a00i = _mm512_fmadd_pd(x0r, w0i, a00i);
                a00i = _mm512_fmadd_pd(x0i, w0r, a00i);
                a01r = _mm512_fmadd_pd(x1r, w0r, a01r);
                a01r = _mm512_fnmadd_pd(x1i, w0i, a01r);
                a01i = _mm512_fmadd_pd(x1r, w0i, a01i);
                a01i = _mm512_fmadd_pd(x1i, w0r, a01i);

                let w1r = _mm512_set1_pd(*wrp.add(w1_off + k));
                let w1i = _mm512_set1_pd(*wip.add(w1_off + k));
                a10r = _mm512_fmadd_pd(x0r, w1r, a10r);
                a10r = _mm512_fnmadd_pd(x0i, w1i, a10r);
                a10i = _mm512_fmadd_pd(x0r, w1i, a10i);
                a10i = _mm512_fmadd_pd(x0i, w1r, a10i);
                a11r = _mm512_fmadd_pd(x1r, w1r, a11r);
                a11r = _mm512_fnmadd_pd(x1i, w1i, a11r);
                a11i = _mm512_fmadd_pd(x1r, w1i, a11i);
                a11i = _mm512_fmadd_pd(x1i, w1r, a11i);

                src_off += row_stride;
            }

            let d0 = (ri * c + ci) * 8;
            let d1 = ((ri + 1) * c + ci) * 8;
            _mm512_storeu_pd(drp.add(d0), a00r);
            _mm512_storeu_pd(dip.add(d0), a00i);
            _mm512_storeu_pd(drp.add(d0 + 8), a01r);
            _mm512_storeu_pd(dip.add(d0 + 8), a01i);
            _mm512_storeu_pd(drp.add(d1), a10r);
            _mm512_storeu_pd(dip.add(d1), a10i);
            _mm512_storeu_pd(drp.add(d1 + 8), a11r);
            _mm512_storeu_pd(dip.add(d1 + 8), a11i);
            ci += 2;
        }
        ri += 2;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
unsafe fn bailey_row_dft_avx512(
    src_re: &[f64],
    src_im: &[f64],
    dst_re: &mut [f64],
    dst_im: &mut [f64],
    w_re: &[f64],
    w_im: &[f64],
    r: usize,
    c: usize,
) {
    use std::arch::x86_64::*;
    let srp = src_re.as_ptr();
    let sip = src_im.as_ptr();
    let drp = dst_re.as_mut_ptr();
    let dip = dst_im.as_mut_ptr();
    let wrp = w_re.as_ptr();
    let wip = w_im.as_ptr();

    let mut ri = 0usize;
    while ri < r {
        let r0_base = ri * c * 8;
        let r1_base = (ri + 1) * c * 8;
        let mut m = 0usize;
        while m < c {
            let mut a00r = _mm512_setzero_pd();
            let mut a00i = _mm512_setzero_pd();
            let mut a01r = _mm512_setzero_pd();
            let mut a01i = _mm512_setzero_pd();
            let mut a10r = _mm512_setzero_pd();
            let mut a10i = _mm512_setzero_pd();
            let mut a11r = _mm512_setzero_pd();
            let mut a11i = _mm512_setzero_pd();

            let w0_off = m * c;
            let w1_off = (m + 1) * c;
            for ci in 0..c {
                let x0r = _mm512_loadu_pd(srp.add(r0_base + ci * 8));
                let x0i = _mm512_loadu_pd(sip.add(r0_base + ci * 8));
                let x1r = _mm512_loadu_pd(srp.add(r1_base + ci * 8));
                let x1i = _mm512_loadu_pd(sip.add(r1_base + ci * 8));

                let w0r = _mm512_set1_pd(*wrp.add(w0_off + ci));
                let w0i = _mm512_set1_pd(*wip.add(w0_off + ci));
                a00r = _mm512_fmadd_pd(x0r, w0r, a00r);
                a00r = _mm512_fnmadd_pd(x0i, w0i, a00r);
                a00i = _mm512_fmadd_pd(x0r, w0i, a00i);
                a00i = _mm512_fmadd_pd(x0i, w0r, a00i);
                a10r = _mm512_fmadd_pd(x1r, w0r, a10r);
                a10r = _mm512_fnmadd_pd(x1i, w0i, a10r);
                a10i = _mm512_fmadd_pd(x1r, w0i, a10i);
                a10i = _mm512_fmadd_pd(x1i, w0r, a10i);

                let w1r = _mm512_set1_pd(*wrp.add(w1_off + ci));
                let w1i = _mm512_set1_pd(*wip.add(w1_off + ci));
                a01r = _mm512_fmadd_pd(x0r, w1r, a01r);
                a01r = _mm512_fnmadd_pd(x0i, w1i, a01r);
                a01i = _mm512_fmadd_pd(x0r, w1i, a01i);
                a01i = _mm512_fmadd_pd(x0i, w1r, a01i);
                a11r = _mm512_fmadd_pd(x1r, w1r, a11r);
                a11r = _mm512_fnmadd_pd(x1i, w1i, a11r);
                a11i = _mm512_fmadd_pd(x1r, w1i, a11i);
                a11i = _mm512_fmadd_pd(x1i, w1r, a11i);
            }

            _mm512_storeu_pd(drp.add(r0_base + m * 8), a00r);
            _mm512_storeu_pd(dip.add(r0_base + m * 8), a00i);
            _mm512_storeu_pd(drp.add(r0_base + (m + 1) * 8), a01r);
            _mm512_storeu_pd(dip.add(r0_base + (m + 1) * 8), a01i);
            _mm512_storeu_pd(drp.add(r1_base + m * 8), a10r);
            _mm512_storeu_pd(dip.add(r1_base + m * 8), a10i);
            _mm512_storeu_pd(drp.add(r1_base + (m + 1) * 8), a11r);
            _mm512_storeu_pd(dip.add(r1_base + (m + 1) * 8), a11i);
            m += 2;
        }
        ri += 2;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn bailey_twiddle_avx512(re: &mut [f64], im: &mut [f64], tw_re: &[f64], tw_im: &[f64]) {
    use std::arch::x86_64::*;
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let wrp = tw_re.as_ptr();
    let wip = tw_im.as_ptr();
    for idx in 0..tw_re.len() {
        let off = idx * 8;
        let xr = _mm512_loadu_pd(rp.add(off));
        let xi = _mm512_loadu_pd(ip.add(off));
        let wr = _mm512_set1_pd(*wrp.add(idx));
        let wi = _mm512_set1_pd(*wip.add(idx));
        let nr = _mm512_fmsub_pd(xr, wr, _mm512_mul_pd(xi, wi));
        let ni = _mm512_fmadd_pd(xr, wi, _mm512_mul_pd(xi, wr));
        _mm512_storeu_pd(rp.add(off), nr);
        _mm512_storeu_pd(ip.add(off), ni);
    }
}

fn square_complex_scalar(re: &mut [f64], im: &mut [f64]) {
    for (r, i) in re.iter_mut().zip(im.iter_mut()) {
        let rv = *r;
        let iv = *i;
        *r = rv.mul_add(rv, -(iv * iv));
        *i = 2.0 * rv * iv;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn square_complex_avx2(re: &mut [f64], im: &mut [f64]) {
    use std::arch::x86_64::*;
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let two = _mm256_set1_pd(2.0);
    let mut k = 0usize;
    while k < re.len() {
        let r = _mm256_loadu_pd(rp.add(k));
        let i = _mm256_loadu_pd(ip.add(k));
        _mm256_storeu_pd(rp.add(k), _mm256_fmsub_pd(r, r, _mm256_mul_pd(i, i)));
        _mm256_storeu_pd(ip.add(k), _mm256_mul_pd(two, _mm256_mul_pd(r, i)));
        k += 4;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn square_complex_avx512(re: &mut [f64], im: &mut [f64]) {
    use std::arch::x86_64::*;
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let two = _mm512_set1_pd(2.0);
    let mut k = 0usize;
    while k < re.len() {
        let r = _mm512_loadu_pd(rp.add(k));
        let i = _mm512_loadu_pd(ip.add(k));
        _mm512_storeu_pd(rp.add(k), _mm512_fmsub_pd(r, r, _mm512_mul_pd(i, i)));
        _mm512_storeu_pd(ip.add(k), _mm512_mul_pd(two, _mm512_mul_pd(r, i)));
        k += 8;
    }
}

// ---------------------------------------------------------------------------
// Engine 2: Zero-Bit-Reversal Fused Radix-4 DIF/DIT FFT (Small / Large / Blend)
// ---------------------------------------------------------------------------

fn radix4_dif_stage(
    kernel: Kernel,
    lanes: usize,
    re: &mut [f64],
    im: &mut [f64],
    st: &Radix4Stage,
) {
    match kernel {
        Kernel::Avx512 if lanes == 8 => unsafe { radix4_dif_avx512(re, im, st) },
        Kernel::Avx2Fma | Kernel::Avx512 if lanes == 4 => unsafe { radix4_dif_avx2(re, im, st) },
        _ => radix4_dif_scalar(re, im, st, lanes),
    }
}

fn radix4_dit_stage(
    kernel: Kernel,
    lanes: usize,
    re: &mut [f64],
    im: &mut [f64],
    st: &Radix4Stage,
) {
    match kernel {
        Kernel::Avx512 if lanes == 8 => unsafe { radix4_dit_avx512(re, im, st) },
        Kernel::Avx2Fma | Kernel::Avx512 if lanes == 4 => unsafe { radix4_dit_avx2(re, im, st) },
        _ => radix4_dit_scalar(re, im, st, lanes),
    }
}

fn mid_radix4_square_fused(kernel: Kernel, lanes: usize, re: &mut [f64], im: &mut [f64]) {
    match kernel {
        Kernel::Avx512 if lanes == 8 => unsafe { mid_radix4_square_avx512(re, im) },
        Kernel::Avx2Fma | Kernel::Avx512 if lanes == 4 => unsafe { mid_radix4_square_avx2(re, im) },
        _ => mid_radix4_square_scalar(re, im, lanes),
    }
}

fn mid_radix2_square_fused(kernel: Kernel, lanes: usize, re: &mut [f64], im: &mut [f64]) {
    match kernel {
        Kernel::Avx512 if lanes == 8 => unsafe { mid_radix2_square_avx512(re, im) },
        Kernel::Avx2Fma | Kernel::Avx512 if lanes == 4 => unsafe { mid_radix2_square_avx2(re, im) },
        _ => mid_radix2_square_scalar(re, im, lanes),
    }
}

fn radix4_dif_scalar(re: &mut [f64], im: &mut [f64], st: &Radix4Stage, lanes: usize) {
    let n = re.len() / lanes;
    let q = st.quarter;
    let len = q << 2;
    let mut base = 0usize;
    while base < n {
        for j in 0..q {
            let w1r = st.w1_re[j];
            let w1i = st.w1_im[j];
            let w2r = st.w2_re[j];
            let w2i = st.w2_im[j];
            let w3r = st.w3_re[j];
            let w3i = st.w3_im[j];
            let i0 = (base + j) * lanes;
            let i1 = i0 + q * lanes;
            let i2 = i1 + q * lanes;
            let i3 = i2 + q * lanes;
            for l in 0..lanes {
                let (ar, ai) = (re[i0 + l], im[i0 + l]);
                let (br, bi) = (re[i1 + l], im[i1 + l]);
                let (cr, ci) = (re[i2 + l], im[i2 + l]);
                let (dr, di) = (re[i3 + l], im[i3 + l]);

                let (t0r, t0i) = (ar + cr, ai + ci);
                let (t1r, t1i) = (ar - cr, ai - ci);
                let (t2r, t2i) = (br + dr, bi + di);
                let (t3r, t3i) = (br - dr, bi - di);

                re[i0 + l] = t0r + t2r;
                im[i0 + l] = t0i + t2i;

                let (u1r, u1i) = (t0r - t2r, t0i - t2i);
                re[i1 + l] = u1r.mul_add(w2r, -(u1i * w2i));
                im[i1 + l] = u1r.mul_add(w2i, u1i * w2r);

                let (u2r, u2i) = (t1r + t3i, t1i - t3r);
                re[i2 + l] = u2r.mul_add(w1r, -(u2i * w1i));
                im[i2 + l] = u2r.mul_add(w1i, u2i * w1r);

                let (u3r, u3i) = (t1r - t3i, t1i + t3r);
                re[i3 + l] = u3r.mul_add(w3r, -(u3i * w3i));
                im[i3 + l] = u3r.mul_add(w3i, u3i * w3r);
            }
        }
        base += len;
    }
}

fn radix4_dit_scalar(re: &mut [f64], im: &mut [f64], st: &Radix4Stage, lanes: usize) {
    let n = re.len() / lanes;
    let q = st.quarter;
    let len = q << 2;
    let mut base = 0usize;
    while base < n {
        for j in 0..q {
            let w1r = st.w1_re[j];
            let w1i = st.w1_im[j];
            let w2r = st.w2_re[j];
            let w2i = st.w2_im[j];
            let w3r = st.w3_re[j];
            let w3i = st.w3_im[j];
            let i0 = (base + j) * lanes;
            let i1 = i0 + q * lanes;
            let i2 = i1 + q * lanes;
            let i3 = i2 + q * lanes;
            for l in 0..lanes {
                let (p0r, p0i) = (re[i0 + l], im[i0 + l]);
                let (y1r, y1i) = (re[i1 + l], im[i1 + l]);
                let (y2r, y2i) = (re[i2 + l], im[i2 + l]);
                let (y3r, y3i) = (re[i3 + l], im[i3 + l]);

                // Multiply by conjugate twiddles W* = (wr, -wi).
                let p1r = y1r.mul_add(w2r, y1i * w2i);
                let p1i = y1i.mul_add(w2r, -(y1r * w2i));
                let p2r = y2r.mul_add(w1r, y2i * w1i);
                let p2i = y2i.mul_add(w1r, -(y2r * w1i));
                let p3r = y3r.mul_add(w3r, y3i * w3i);
                let p3i = y3i.mul_add(w3r, -(y3r * w3i));

                let (u0r, u0i) = (p0r + p1r, p0i + p1i);
                let (u1r, u1i) = (p0r - p1r, p0i - p1i);
                let (u2r, u2i) = (p2r + p3r, p2i + p3i);
                let (u3r, u3i) = (p2r - p3r, p2i - p3i);

                re[i0 + l] = u0r + u2r;
                im[i0 + l] = u0i + u2i;
                re[i1 + l] = u1r - u3i;
                im[i1 + l] = u1i + u3r;
                re[i2 + l] = u0r - u2r;
                im[i2 + l] = u0i - u2i;
                re[i3 + l] = u1r + u3i;
                im[i3 + l] = u1i - u3r;
            }
        }
        base += len;
    }
}

fn mid_radix4_square_scalar(re: &mut [f64], im: &mut [f64], lanes: usize) {
    let n = re.len() / lanes;
    let mut base = 0usize;
    while base < n {
        let i0 = base * lanes;
        let i1 = i0 + lanes;
        let i2 = i1 + lanes;
        let i3 = i2 + lanes;
        for l in 0..lanes {
            let (ar, ai) = (re[i0 + l], im[i0 + l]);
            let (br, bi) = (re[i1 + l], im[i1 + l]);
            let (cr, ci) = (re[i2 + l], im[i2 + l]);
            let (dr, di) = (re[i3 + l], im[i3 + l]);

            let (t0r, t0i) = (ar + cr, ai + ci);
            let (t1r, t1i) = (ar - cr, ai - ci);
            let (t2r, t2i) = (br + dr, bi + di);
            let (t3r, t3i) = (br - dr, bi - di);

            let (y0r, y0i) = (t0r + t2r, t0i + t2i);
            let (y1r, y1i) = (t0r - t2r, t0i - t2i);
            let (y2r, y2i) = (t1r + t3i, t1i - t3r);
            let (y3r, y3i) = (t1r - t3i, t1i + t3r);

            let (p0r, p0i) = (y0r.mul_add(y0r, -(y0i * y0i)), 2.0 * y0r * y0i);
            let (p1r, p1i) = (y1r.mul_add(y1r, -(y1i * y1i)), 2.0 * y1r * y1i);
            let (p2r, p2i) = (y2r.mul_add(y2r, -(y2i * y2i)), 2.0 * y2r * y2i);
            let (p3r, p3i) = (y3r.mul_add(y3r, -(y3i * y3i)), 2.0 * y3r * y3i);

            let (u0r, u0i) = (p0r + p1r, p0i + p1i);
            let (u1r, u1i) = (p0r - p1r, p0i - p1i);
            let (u2r, u2i) = (p2r + p3r, p2i + p3i);
            let (u3r, u3i) = (p2r - p3r, p2i - p3i);

            re[i0 + l] = u0r + u2r;
            im[i0 + l] = u0i + u2i;
            re[i1 + l] = u1r - u3i;
            im[i1 + l] = u1i + u3r;
            re[i2 + l] = u0r - u2r;
            im[i2 + l] = u0i - u2i;
            re[i3 + l] = u1r + u3i;
            im[i3 + l] = u1i - u3r;
        }
        base += 4;
    }
}

fn mid_radix2_square_scalar(re: &mut [f64], im: &mut [f64], lanes: usize) {
    let n = re.len() / lanes;
    let mut base = 0usize;
    while base < n {
        let i0 = base * lanes;
        let i1 = i0 + lanes;
        for l in 0..lanes {
            let (ar, ai) = (re[i0 + l], im[i0 + l]);
            let (br, bi) = (re[i1 + l], im[i1 + l]);
            let (y0r, y0i) = (ar + br, ai + bi);
            let (y1r, y1i) = (ar - br, ai - bi);
            let (p0r, p0i) = (y0r.mul_add(y0r, -(y0i * y0i)), 2.0 * y0r * y0i);
            let (p1r, p1i) = (y1r.mul_add(y1r, -(y1i * y1i)), 2.0 * y1r * y1i);
            re[i0 + l] = p0r + p1r;
            im[i0 + l] = p0i + p1i;
            re[i1 + l] = p0r - p1r;
            im[i1 + l] = p0i - p1i;
        }
        base += 2;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn radix4_dif_avx2(re: &mut [f64], im: &mut [f64], st: &Radix4Stage) {
    use std::arch::x86_64::*;
    let n = re.len() >> 2;
    let q = st.quarter;
    let q4 = q << 2;
    let len = q << 2;
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let w1rp = st.w1_re.as_ptr();
    let w1ip = st.w1_im.as_ptr();
    let w2rp = st.w2_re.as_ptr();
    let w2ip = st.w2_im.as_ptr();
    let w3rp = st.w3_re.as_ptr();
    let w3ip = st.w3_im.as_ptr();

    let mut base = 0usize;
    while base < n {
        let base4 = base << 2;
        for j in 0..q {
            let i0 = base4 + (j << 2);
            let i1 = i0 + q4;
            let i2 = i1 + q4;
            let i3 = i2 + q4;

            let ar = _mm256_loadu_pd(rp.add(i0));
            let ai = _mm256_loadu_pd(ip.add(i0));
            let br = _mm256_loadu_pd(rp.add(i1));
            let bi = _mm256_loadu_pd(ip.add(i1));
            let cr = _mm256_loadu_pd(rp.add(i2));
            let ci = _mm256_loadu_pd(ip.add(i2));
            let dr = _mm256_loadu_pd(rp.add(i3));
            let di = _mm256_loadu_pd(ip.add(i3));

            let t0r = _mm256_add_pd(ar, cr);
            let t0i = _mm256_add_pd(ai, ci);
            let t1r = _mm256_sub_pd(ar, cr);
            let t1i = _mm256_sub_pd(ai, ci);
            let t2r = _mm256_add_pd(br, dr);
            let t2i = _mm256_add_pd(bi, di);
            let t3r = _mm256_sub_pd(br, dr);
            let t3i = _mm256_sub_pd(bi, di);

            _mm256_storeu_pd(rp.add(i0), _mm256_add_pd(t0r, t2r));
            _mm256_storeu_pd(ip.add(i0), _mm256_add_pd(t0i, t2i));

            let u1r = _mm256_sub_pd(t0r, t2r);
            let u1i = _mm256_sub_pd(t0i, t2i);
            let w2r = _mm256_set1_pd(*w2rp.add(j));
            let w2i = _mm256_set1_pd(*w2ip.add(j));
            _mm256_storeu_pd(
                rp.add(i1),
                _mm256_fmsub_pd(u1r, w2r, _mm256_mul_pd(u1i, w2i)),
            );
            _mm256_storeu_pd(
                ip.add(i1),
                _mm256_fmadd_pd(u1r, w2i, _mm256_mul_pd(u1i, w2r)),
            );

            let u2r = _mm256_add_pd(t1r, t3i);
            let u2i = _mm256_sub_pd(t1i, t3r);
            let w1r = _mm256_set1_pd(*w1rp.add(j));
            let w1i = _mm256_set1_pd(*w1ip.add(j));
            _mm256_storeu_pd(
                rp.add(i2),
                _mm256_fmsub_pd(u2r, w1r, _mm256_mul_pd(u2i, w1i)),
            );
            _mm256_storeu_pd(
                ip.add(i2),
                _mm256_fmadd_pd(u2r, w1i, _mm256_mul_pd(u2i, w1r)),
            );

            let u3r = _mm256_sub_pd(t1r, t3i);
            let u3i = _mm256_add_pd(t1i, t3r);
            let w3r = _mm256_set1_pd(*w3rp.add(j));
            let w3i = _mm256_set1_pd(*w3ip.add(j));
            _mm256_storeu_pd(
                rp.add(i3),
                _mm256_fmsub_pd(u3r, w3r, _mm256_mul_pd(u3i, w3i)),
            );
            _mm256_storeu_pd(
                ip.add(i3),
                _mm256_fmadd_pd(u3r, w3i, _mm256_mul_pd(u3i, w3r)),
            );
        }
        base += len;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn radix4_dit_avx2(re: &mut [f64], im: &mut [f64], st: &Radix4Stage) {
    use std::arch::x86_64::*;
    let n = re.len() >> 2;
    let q = st.quarter;
    let q4 = q << 2;
    let len = q << 2;
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let w1rp = st.w1_re.as_ptr();
    let w1ip = st.w1_im.as_ptr();
    let w2rp = st.w2_re.as_ptr();
    let w2ip = st.w2_im.as_ptr();
    let w3rp = st.w3_re.as_ptr();
    let w3ip = st.w3_im.as_ptr();

    let mut base = 0usize;
    while base < n {
        let base4 = base << 2;
        for j in 0..q {
            let i0 = base4 + (j << 2);
            let i1 = i0 + q4;
            let i2 = i1 + q4;
            let i3 = i2 + q4;

            let p0r = _mm256_loadu_pd(rp.add(i0));
            let p0i = _mm256_loadu_pd(ip.add(i0));
            let y1r = _mm256_loadu_pd(rp.add(i1));
            let y1i = _mm256_loadu_pd(ip.add(i1));
            let y2r = _mm256_loadu_pd(rp.add(i2));
            let y2i = _mm256_loadu_pd(ip.add(i2));
            let y3r = _mm256_loadu_pd(rp.add(i3));
            let y3i = _mm256_loadu_pd(ip.add(i3));

            let w2r = _mm256_set1_pd(*w2rp.add(j));
            let w2i = _mm256_set1_pd(*w2ip.add(j));
            let p1r = _mm256_fmadd_pd(y1r, w2r, _mm256_mul_pd(y1i, w2i));
            let p1i = _mm256_fmsub_pd(y1i, w2r, _mm256_mul_pd(y1r, w2i));

            let w1r = _mm256_set1_pd(*w1rp.add(j));
            let w1i = _mm256_set1_pd(*w1ip.add(j));
            let p2r = _mm256_fmadd_pd(y2r, w1r, _mm256_mul_pd(y2i, w1i));
            let p2i = _mm256_fmsub_pd(y2i, w1r, _mm256_mul_pd(y2r, w1i));

            let w3r = _mm256_set1_pd(*w3rp.add(j));
            let w3i = _mm256_set1_pd(*w3ip.add(j));
            let p3r = _mm256_fmadd_pd(y3r, w3r, _mm256_mul_pd(y3i, w3i));
            let p3i = _mm256_fmsub_pd(y3i, w3r, _mm256_mul_pd(y3r, w3i));

            let u0r = _mm256_add_pd(p0r, p1r);
            let u0i = _mm256_add_pd(p0i, p1i);
            let u1r = _mm256_sub_pd(p0r, p1r);
            let u1i = _mm256_sub_pd(p0i, p1i);
            let u2r = _mm256_add_pd(p2r, p3r);
            let u2i = _mm256_add_pd(p2i, p3i);
            let u3r = _mm256_sub_pd(p2r, p3r);
            let u3i = _mm256_sub_pd(p2i, p3i);

            _mm256_storeu_pd(rp.add(i0), _mm256_add_pd(u0r, u2r));
            _mm256_storeu_pd(ip.add(i0), _mm256_add_pd(u0i, u2i));
            _mm256_storeu_pd(rp.add(i1), _mm256_sub_pd(u1r, u3i));
            _mm256_storeu_pd(ip.add(i1), _mm256_add_pd(u1i, u3r));
            _mm256_storeu_pd(rp.add(i2), _mm256_sub_pd(u0r, u2r));
            _mm256_storeu_pd(ip.add(i2), _mm256_sub_pd(u0i, u2i));
            _mm256_storeu_pd(rp.add(i3), _mm256_add_pd(u1r, u3i));
            _mm256_storeu_pd(ip.add(i3), _mm256_sub_pd(u1i, u3r));
        }
        base += len;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn mid_radix4_square_avx2(re: &mut [f64], im: &mut [f64]) {
    use std::arch::x86_64::*;
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let two = _mm256_set1_pd(2.0);
    let total = re.len();
    let mut off = 0usize;
    while off < total {
        let ar = _mm256_loadu_pd(rp.add(off));
        let ai = _mm256_loadu_pd(ip.add(off));
        let br = _mm256_loadu_pd(rp.add(off + 4));
        let bi = _mm256_loadu_pd(ip.add(off + 4));
        let cr = _mm256_loadu_pd(rp.add(off + 8));
        let ci = _mm256_loadu_pd(ip.add(off + 8));
        let dr = _mm256_loadu_pd(rp.add(off + 12));
        let di = _mm256_loadu_pd(ip.add(off + 12));

        let t0r = _mm256_add_pd(ar, cr);
        let t0i = _mm256_add_pd(ai, ci);
        let t1r = _mm256_sub_pd(ar, cr);
        let t1i = _mm256_sub_pd(ai, ci);
        let t2r = _mm256_add_pd(br, dr);
        let t2i = _mm256_add_pd(bi, di);
        let t3r = _mm256_sub_pd(br, dr);
        let t3i = _mm256_sub_pd(bi, di);

        let y0r = _mm256_add_pd(t0r, t2r);
        let y0i = _mm256_add_pd(t0i, t2i);
        let y1r = _mm256_sub_pd(t0r, t2r);
        let y1i = _mm256_sub_pd(t0i, t2i);
        let y2r = _mm256_add_pd(t1r, t3i);
        let y2i = _mm256_sub_pd(t1i, t3r);
        let y3r = _mm256_sub_pd(t1r, t3i);
        let y3i = _mm256_add_pd(t1i, t3r);

        let p0r = _mm256_fmsub_pd(y0r, y0r, _mm256_mul_pd(y0i, y0i));
        let p0i = _mm256_mul_pd(two, _mm256_mul_pd(y0r, y0i));
        let p1r = _mm256_fmsub_pd(y1r, y1r, _mm256_mul_pd(y1i, y1i));
        let p1i = _mm256_mul_pd(two, _mm256_mul_pd(y1r, y1i));
        let p2r = _mm256_fmsub_pd(y2r, y2r, _mm256_mul_pd(y2i, y2i));
        let p2i = _mm256_mul_pd(two, _mm256_mul_pd(y2r, y2i));
        let p3r = _mm256_fmsub_pd(y3r, y3r, _mm256_mul_pd(y3i, y3i));
        let p3i = _mm256_mul_pd(two, _mm256_mul_pd(y3r, y3i));

        let u0r = _mm256_add_pd(p0r, p1r);
        let u0i = _mm256_add_pd(p0i, p1i);
        let u1r = _mm256_sub_pd(p0r, p1r);
        let u1i = _mm256_sub_pd(p0i, p1i);
        let u2r = _mm256_add_pd(p2r, p3r);
        let u2i = _mm256_add_pd(p2i, p3i);
        let u3r = _mm256_sub_pd(p2r, p3r);
        let u3i = _mm256_sub_pd(p2i, p3i);

        _mm256_storeu_pd(rp.add(off), _mm256_add_pd(u0r, u2r));
        _mm256_storeu_pd(ip.add(off), _mm256_add_pd(u0i, u2i));
        _mm256_storeu_pd(rp.add(off + 4), _mm256_sub_pd(u1r, u3i));
        _mm256_storeu_pd(ip.add(off + 4), _mm256_add_pd(u1i, u3r));
        _mm256_storeu_pd(rp.add(off + 8), _mm256_sub_pd(u0r, u2r));
        _mm256_storeu_pd(ip.add(off + 8), _mm256_sub_pd(u0i, u2i));
        _mm256_storeu_pd(rp.add(off + 12), _mm256_add_pd(u1r, u3i));
        _mm256_storeu_pd(ip.add(off + 12), _mm256_sub_pd(u1i, u3r));
        off += 16;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn mid_radix2_square_avx2(re: &mut [f64], im: &mut [f64]) {
    use std::arch::x86_64::*;
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let two = _mm256_set1_pd(2.0);
    let total = re.len();
    let mut off = 0usize;
    while off < total {
        let ar = _mm256_loadu_pd(rp.add(off));
        let ai = _mm256_loadu_pd(ip.add(off));
        let br = _mm256_loadu_pd(rp.add(off + 4));
        let bi = _mm256_loadu_pd(ip.add(off + 4));

        let y0r = _mm256_add_pd(ar, br);
        let y0i = _mm256_add_pd(ai, bi);
        let y1r = _mm256_sub_pd(ar, br);
        let y1i = _mm256_sub_pd(ai, bi);

        let p0r = _mm256_fmsub_pd(y0r, y0r, _mm256_mul_pd(y0i, y0i));
        let p0i = _mm256_mul_pd(two, _mm256_mul_pd(y0r, y0i));
        let p1r = _mm256_fmsub_pd(y1r, y1r, _mm256_mul_pd(y1i, y1i));
        let p1i = _mm256_mul_pd(two, _mm256_mul_pd(y1r, y1i));

        _mm256_storeu_pd(rp.add(off), _mm256_add_pd(p0r, p1r));
        _mm256_storeu_pd(ip.add(off), _mm256_add_pd(p0i, p1i));
        _mm256_storeu_pd(rp.add(off + 4), _mm256_sub_pd(p0r, p1r));
        _mm256_storeu_pd(ip.add(off + 4), _mm256_sub_pd(p0i, p1i));
        off += 8;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn radix4_dif_avx512(re: &mut [f64], im: &mut [f64], st: &Radix4Stage) {
    use std::arch::x86_64::*;
    let n = re.len() >> 3;
    let q = st.quarter;
    let q8 = q << 3;
    let len = q << 2;
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let w1rp = st.w1_re.as_ptr();
    let w1ip = st.w1_im.as_ptr();
    let w2rp = st.w2_re.as_ptr();
    let w2ip = st.w2_im.as_ptr();
    let w3rp = st.w3_re.as_ptr();
    let w3ip = st.w3_im.as_ptr();

    let mut base = 0usize;
    while base < n {
        let base8 = base << 3;
        for j in 0..q {
            let i0 = base8 + (j << 3);
            let i1 = i0 + q8;
            let i2 = i1 + q8;
            let i3 = i2 + q8;

            let ar = _mm512_loadu_pd(rp.add(i0));
            let ai = _mm512_loadu_pd(ip.add(i0));
            let br = _mm512_loadu_pd(rp.add(i1));
            let bi = _mm512_loadu_pd(ip.add(i1));
            let cr = _mm512_loadu_pd(rp.add(i2));
            let ci = _mm512_loadu_pd(ip.add(i2));
            let dr = _mm512_loadu_pd(rp.add(i3));
            let di = _mm512_loadu_pd(ip.add(i3));

            let t0r = _mm512_add_pd(ar, cr);
            let t0i = _mm512_add_pd(ai, ci);
            let t1r = _mm512_sub_pd(ar, cr);
            let t1i = _mm512_sub_pd(ai, ci);
            let t2r = _mm512_add_pd(br, dr);
            let t2i = _mm512_add_pd(bi, di);
            let t3r = _mm512_sub_pd(br, dr);
            let t3i = _mm512_sub_pd(bi, di);

            _mm512_storeu_pd(rp.add(i0), _mm512_add_pd(t0r, t2r));
            _mm512_storeu_pd(ip.add(i0), _mm512_add_pd(t0i, t2i));

            let u1r = _mm512_sub_pd(t0r, t2r);
            let u1i = _mm512_sub_pd(t0i, t2i);
            let w2r = _mm512_set1_pd(*w2rp.add(j));
            let w2i = _mm512_set1_pd(*w2ip.add(j));
            _mm512_storeu_pd(
                rp.add(i1),
                _mm512_fmsub_pd(u1r, w2r, _mm512_mul_pd(u1i, w2i)),
            );
            _mm512_storeu_pd(
                ip.add(i1),
                _mm512_fmadd_pd(u1r, w2i, _mm512_mul_pd(u1i, w2r)),
            );

            let u2r = _mm512_add_pd(t1r, t3i);
            let u2i = _mm512_sub_pd(t1i, t3r);
            let w1r = _mm512_set1_pd(*w1rp.add(j));
            let w1i = _mm512_set1_pd(*w1ip.add(j));
            _mm512_storeu_pd(
                rp.add(i2),
                _mm512_fmsub_pd(u2r, w1r, _mm512_mul_pd(u2i, w1i)),
            );
            _mm512_storeu_pd(
                ip.add(i2),
                _mm512_fmadd_pd(u2r, w1i, _mm512_mul_pd(u2i, w1r)),
            );

            let u3r = _mm512_sub_pd(t1r, t3i);
            let u3i = _mm512_add_pd(t1i, t3r);
            let w3r = _mm512_set1_pd(*w3rp.add(j));
            let w3i = _mm512_set1_pd(*w3ip.add(j));
            _mm512_storeu_pd(
                rp.add(i3),
                _mm512_fmsub_pd(u3r, w3r, _mm512_mul_pd(u3i, w3i)),
            );
            _mm512_storeu_pd(
                ip.add(i3),
                _mm512_fmadd_pd(u3r, w3i, _mm512_mul_pd(u3i, w3r)),
            );
        }
        base += len;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn radix4_dit_avx512(re: &mut [f64], im: &mut [f64], st: &Radix4Stage) {
    use std::arch::x86_64::*;
    let n = re.len() >> 3;
    let q = st.quarter;
    let q8 = q << 3;
    let len = q << 2;
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let w1rp = st.w1_re.as_ptr();
    let w1ip = st.w1_im.as_ptr();
    let w2rp = st.w2_re.as_ptr();
    let w2ip = st.w2_im.as_ptr();
    let w3rp = st.w3_re.as_ptr();
    let w3ip = st.w3_im.as_ptr();

    let mut base = 0usize;
    while base < n {
        let base8 = base << 3;
        for j in 0..q {
            let i0 = base8 + (j << 3);
            let i1 = i0 + q8;
            let i2 = i1 + q8;
            let i3 = i2 + q8;

            let p0r = _mm512_loadu_pd(rp.add(i0));
            let p0i = _mm512_loadu_pd(ip.add(i0));
            let y1r = _mm512_loadu_pd(rp.add(i1));
            let y1i = _mm512_loadu_pd(ip.add(i1));
            let y2r = _mm512_loadu_pd(rp.add(i2));
            let y2i = _mm512_loadu_pd(ip.add(i2));
            let y3r = _mm512_loadu_pd(rp.add(i3));
            let y3i = _mm512_loadu_pd(ip.add(i3));

            let w2r = _mm512_set1_pd(*w2rp.add(j));
            let w2i = _mm512_set1_pd(*w2ip.add(j));
            let p1r = _mm512_fmadd_pd(y1r, w2r, _mm512_mul_pd(y1i, w2i));
            let p1i = _mm512_fmsub_pd(y1i, w2r, _mm512_mul_pd(y1r, w2i));

            let w1r = _mm512_set1_pd(*w1rp.add(j));
            let w1i = _mm512_set1_pd(*w1ip.add(j));
            let p2r = _mm512_fmadd_pd(y2r, w1r, _mm512_mul_pd(y2i, w1i));
            let p2i = _mm512_fmsub_pd(y2i, w1r, _mm512_mul_pd(y2r, w1i));

            let w3r = _mm512_set1_pd(*w3rp.add(j));
            let w3i = _mm512_set1_pd(*w3ip.add(j));
            let p3r = _mm512_fmadd_pd(y3r, w3r, _mm512_mul_pd(y3i, w3i));
            let p3i = _mm512_fmsub_pd(y3i, w3r, _mm512_mul_pd(y3r, w3i));

            let u0r = _mm512_add_pd(p0r, p1r);
            let u0i = _mm512_add_pd(p0i, p1i);
            let u1r = _mm512_sub_pd(p0r, p1r);
            let u1i = _mm512_sub_pd(p0i, p1i);
            let u2r = _mm512_add_pd(p2r, p3r);
            let u2i = _mm512_add_pd(p2i, p3i);
            let u3r = _mm512_sub_pd(p2r, p3r);
            let u3i = _mm512_sub_pd(p2i, p3i);

            _mm512_storeu_pd(rp.add(i0), _mm512_add_pd(u0r, u2r));
            _mm512_storeu_pd(ip.add(i0), _mm512_add_pd(u0i, u2i));
            _mm512_storeu_pd(rp.add(i1), _mm512_sub_pd(u1r, u3i));
            _mm512_storeu_pd(ip.add(i1), _mm512_add_pd(u1i, u3r));
            _mm512_storeu_pd(rp.add(i2), _mm512_sub_pd(u0r, u2r));
            _mm512_storeu_pd(ip.add(i2), _mm512_sub_pd(u0i, u2i));
            _mm512_storeu_pd(rp.add(i3), _mm512_add_pd(u1r, u3i));
            _mm512_storeu_pd(ip.add(i3), _mm512_sub_pd(u1i, u3r));
        }
        base += len;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn mid_radix4_square_avx512(re: &mut [f64], im: &mut [f64]) {
    use std::arch::x86_64::*;
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let two = _mm512_set1_pd(2.0);
    let total = re.len();
    let mut off = 0usize;
    while off < total {
        let ar = _mm512_loadu_pd(rp.add(off));
        let ai = _mm512_loadu_pd(ip.add(off));
        let br = _mm512_loadu_pd(rp.add(off + 8));
        let bi = _mm512_loadu_pd(ip.add(off + 8));
        let cr = _mm512_loadu_pd(rp.add(off + 16));
        let ci = _mm512_loadu_pd(ip.add(off + 16));
        let dr = _mm512_loadu_pd(rp.add(off + 24));
        let di = _mm512_loadu_pd(ip.add(off + 24));

        let t0r = _mm512_add_pd(ar, cr);
        let t0i = _mm512_add_pd(ai, ci);
        let t1r = _mm512_sub_pd(ar, cr);
        let t1i = _mm512_sub_pd(ai, ci);
        let t2r = _mm512_add_pd(br, dr);
        let t2i = _mm512_add_pd(bi, di);
        let t3r = _mm512_sub_pd(br, dr);
        let t3i = _mm512_sub_pd(bi, di);

        let y0r = _mm512_add_pd(t0r, t2r);
        let y0i = _mm512_add_pd(t0i, t2i);
        let y1r = _mm512_sub_pd(t0r, t2r);
        let y1i = _mm512_sub_pd(t0i, t2i);
        let y2r = _mm512_add_pd(t1r, t3i);
        let y2i = _mm512_sub_pd(t1i, t3r);
        let y3r = _mm512_sub_pd(t1r, t3i);
        let y3i = _mm512_add_pd(t1i, t3r);

        let p0r = _mm512_fmsub_pd(y0r, y0r, _mm512_mul_pd(y0i, y0i));
        let p0i = _mm512_mul_pd(two, _mm512_mul_pd(y0r, y0i));
        let p1r = _mm512_fmsub_pd(y1r, y1r, _mm512_mul_pd(y1i, y1i));
        let p1i = _mm512_mul_pd(two, _mm512_mul_pd(y1r, y1i));
        let p2r = _mm512_fmsub_pd(y2r, y2r, _mm512_mul_pd(y2i, y2i));
        let p2i = _mm512_mul_pd(two, _mm512_mul_pd(y2r, y2i));
        let p3r = _mm512_fmsub_pd(y3r, y3r, _mm512_mul_pd(y3i, y3i));
        let p3i = _mm512_mul_pd(two, _mm512_mul_pd(y3r, y3i));

        let u0r = _mm512_add_pd(p0r, p1r);
        let u0i = _mm512_add_pd(p0i, p1i);
        let u1r = _mm512_sub_pd(p0r, p1r);
        let u1i = _mm512_sub_pd(p0i, p1i);
        let u2r = _mm512_add_pd(p2r, p3r);
        let u2i = _mm512_add_pd(p2i, p3i);
        let u3r = _mm512_sub_pd(p2r, p3r);
        let u3i = _mm512_sub_pd(p2i, p3i);

        _mm512_storeu_pd(rp.add(off), _mm512_add_pd(u0r, u2r));
        _mm512_storeu_pd(ip.add(off), _mm512_add_pd(u0i, u2i));
        _mm512_storeu_pd(rp.add(off + 8), _mm512_sub_pd(u1r, u3i));
        _mm512_storeu_pd(ip.add(off + 8), _mm512_add_pd(u1i, u3r));
        _mm512_storeu_pd(rp.add(off + 16), _mm512_sub_pd(u0r, u2r));
        _mm512_storeu_pd(ip.add(off + 16), _mm512_sub_pd(u0i, u2i));
        _mm512_storeu_pd(rp.add(off + 24), _mm512_add_pd(u1r, u3i));
        _mm512_storeu_pd(ip.add(off + 24), _mm512_sub_pd(u1i, u3r));
        off += 32;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn mid_radix2_square_avx512(re: &mut [f64], im: &mut [f64]) {
    use std::arch::x86_64::*;
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let two = _mm512_set1_pd(2.0);
    let total = re.len();
    let mut off = 0usize;
    while off < total {
        let ar = _mm512_loadu_pd(rp.add(off));
        let ai = _mm512_loadu_pd(ip.add(off));
        let br = _mm512_loadu_pd(rp.add(off + 8));
        let bi = _mm512_loadu_pd(ip.add(off + 8));

        let y0r = _mm512_add_pd(ar, br);
        let y0i = _mm512_add_pd(ai, bi);
        let y1r = _mm512_sub_pd(ar, br);
        let y1i = _mm512_sub_pd(ai, bi);

        let p0r = _mm512_fmsub_pd(y0r, y0r, _mm512_mul_pd(y0i, y0i));
        let p0i = _mm512_mul_pd(two, _mm512_mul_pd(y0r, y0i));
        let p1r = _mm512_fmsub_pd(y1r, y1r, _mm512_mul_pd(y1i, y1i));
        let p1i = _mm512_mul_pd(two, _mm512_mul_pd(y1r, y1i));

        _mm512_storeu_pd(rp.add(off), _mm512_add_pd(p0r, p1r));
        _mm512_storeu_pd(ip.add(off), _mm512_add_pd(p0i, p1i));
        _mm512_storeu_pd(rp.add(off + 8), _mm512_sub_pd(p0r, p1r));
        _mm512_storeu_pd(ip.add(off + 8), _mm512_sub_pd(p0i, p1i));
        off += 16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_result_verifies() {
        let mut m = MatSet::new(64, 42);
        m.compute_all(Kernel::Scalar);
        assert!(m.verify());
        m.flip_sign();
        m.compute_all(Kernel::Scalar);
        assert!(m.verify());
    }

    #[test]
    fn avx2_is_bit_identical_to_scalar() {
        let isa = detect_isa();
        if !(isa.avx2 && isa.fma) {
            eprintln!("skipping: no AVX2/FMA");
            return;
        }
        for &n in &[128usize, 256] {
            let mut s = MatSet::new(n, 7);
            s.compute_all(Kernel::Scalar);
            let mut v = MatSet::new(n, 7);
            v.compute_all(Kernel::Avx2Fma);
            assert!(
                s.c.iter()
                    .zip(&v.c)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "n={n}"
            );
            assert!(v.verify());
        }
    }

    #[test]
    fn avx512_matches_when_available() {
        if !detect_isa().avx512f {
            eprintln!("skipping: no AVX-512F");
            return;
        }
        let mut s = MatSet::new(128, 9);
        s.compute_all(Kernel::Scalar);
        let mut v = MatSet::new(128, 9);
        v.compute_all(Kernel::Avx512);
        assert_eq!(s.c, v.c);
    }

    #[test]
    fn panels_compose_to_full_result() {
        let isa = detect_isa();
        let k = if isa.avx2 && isa.fma {
            Kernel::Avx2Fma
        } else {
            Kernel::Scalar
        };
        let mut m = MatSet::new(256, 3);
        for r0 in (0..256).step_by(16) {
            m.compute_rows(k, r0, r0 + 16);
        }
        assert!(m.verify());
    }

    #[test]
    fn checksum_detects_single_flipped_element() {
        let mut m = MatSet::new(128, 11);
        m.compute_all(Kernel::Scalar);
        assert!(m.verify());
        for idx in [0usize, 5000, 128 * 128 - 1] {
            let orig = m.c[idx];
            m.c[idx] = orig + 1.0;
            assert!(!m.verify(), "missed +1 at {idx}");
            m.c[idx] = f64::from_bits(orig.to_bits() ^ (1 << 40));
            assert!(!m.verify(), "missed bit flip at {idx}");
            m.c[idx] = orig;
        }
        assert!(m.verify());
    }

    #[test]
    fn ntt_goldilocks_roundtrip_and_fault_detection() {
        // Verify primitive root order: g^((P-1)/2) == P - 1 (mod P)
        assert_eq!(
            pow_mod(GOLDILOCKS_G, (GOLDILOCKS_P - 1) / 2),
            GOLDILOCKS_P - 1
        );
        let mut ntt = NttSet::new(1024, 99);
        for pass in 0..4u64 {
            assert_eq!(ntt.run_and_verify(pass, || true), Some(true), "pass {pass}");
        }
        // Inject a 1-bit corruption mid-transform and confirm avalanche failure.
        let mut bad_buf = ntt.input.clone();
        assert!(ntt_inplace(
            &mut bad_buf,
            &ntt.fwd_roots,
            ntt.log_n,
            &|| true
        ));
        bad_buf[317] ^= 1u64 << 23;
        assert!(ntt_inplace(
            &mut bad_buf,
            &ntt.inv_roots,
            ntt.log_n,
            &|| true
        ));
        let mismatches = bad_buf
            .iter()
            .zip(&ntt.input)
            .filter(|(&g, &e)| mul_mod(g, ntt.n_inv) != e)
            .count();
        assert!(
            mismatches > 512,
            "expected avalanche error across array, got {mismatches}"
        );
    }

    #[test]
    fn fft_torture_verifies_rounding_and_detects_faults() {
        let isa = detect_isa();
        let k = if isa.avx2 && isa.fma {
            Kernel::Avx2Fma
        } else {
            Kernel::Scalar
        };
        // Cover Bailey 4-Step L1 (64/128/256) and Fused Radix-4 even+odd log_n (1024/2048/4096).
        for &n in &[64usize, 128, 256, 1024, 2048, 4096] {
            let mut fft_s = FftSet::new(n, Kernel::Scalar);
            let mut fft_v = FftSet::new(n, k);
            let rs = fft_s.run_and_verify(Kernel::Scalar, 123, || true).unwrap();
            let rv = fft_v.run_and_verify(k, 123, || true).unwrap();
            let err_s = rs.expect("scalar FFT pass");
            let err_v = rv.expect("SIMD FFT pass");
            assert!(err_s < 1e-5, "unexpected scalar roundoff {err_s} for n={n}");
            assert!(err_v < 1e-5, "unexpected SIMD roundoff {err_v} for n={n}");
            // Rounded integer real & imag outputs from Scalar and AVX2+FMA must match bit-for-bit.
            let inv_n = 1.0 / (n as f64);
            for i in 0..(n * fft_s.lanes) {
                assert_eq!(
                    (fft_s.re[i] * inv_n).round() as i64,
                    (fft_v.re[i] * inv_n).round() as i64,
                    "re mismatch n={n} idx={i}"
                );
                assert_eq!(
                    (fft_s.im[i] * inv_n).round() as i64,
                    (fft_v.im[i] * inv_n).round() as i64,
                    "im mismatch n={n} idx={i}"
                );
            }
        }
        // Direct O(N^2) schoolbook cyclic complex convolution check against Bailey 4-Step (N=64).
        let n = 64usize;
        let mut fft = FftSet::new(n, k);
        let lanes = fft.lanes;
        let orig_re: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
        let orig_im: Vec<f64> = (0..n).map(|i| ((i * 5) % 9) as f64 - 4.0).collect();
        for i in 0..n {
            for l in 0..lanes {
                fft.re[i * lanes + l] = orig_re[i];
                fft.im[i * lanes + l] = orig_im[i];
            }
        }
        assert!(fft.execute_square_transform(k, &|| true));
        for m in 0..n {
            let (mut er, mut ei) = (0.0f64, 0.0f64);
            for j in 0..n {
                let p = (m + n - j) % n;
                er += orig_re[j] * orig_re[p] - orig_im[j] * orig_im[p];
                ei += orig_re[j] * orig_im[p] + orig_im[j] * orig_re[p];
            }
            for l in 0..lanes {
                let gr = fft.re[m * lanes + l] / (n as f64);
                let gi = fft.im[m * lanes + l] / (n as f64);
                assert!(
                    (gr - er).abs() < 1e-7,
                    "Bailey re m={m} l={l}: {gr} vs {er}"
                );
                assert!(
                    (gi - ei).abs() < 1e-7,
                    "Bailey im m={m} l={l}: {gi} vs {ei}"
                );
            }
        }
        // Fault injection: rounding >= 0.40, NaN propagation, and exact integer DC shift.
        let mut fft_f = FftSet::new(1024, k);
        let _ = fft_f.run_and_verify(k, 55, || true).unwrap().unwrap();
        let zero = [0.0f64; 8];
        fft_f.re[213 * fft_f.lanes] += 1024.0 * 0.45;
        assert!(matches!(
            fft_f.verify_output(k, &zero, &zero, &zero, &zero),
            Err(FftFailure::Rounding { max_err, idx }) if max_err >= FFT_MAX_ROUNDOFF && idx == 213
        ));
        fft_f.re[213 * fft_f.lanes] = f64::NAN;
        assert!(matches!(
            fft_f.verify_output(k, &zero, &zero, &zero, &zero),
            Err(FftFailure::Rounding { idx, .. }) if idx == 213
        ));
        // Clean pass followed by +1.0 integer shift -> 0.0000 roundoff, caught by exact DC/Nyquist sum.
        let _ = fft_f.run_and_verify(k, 55, || true).unwrap().unwrap();
        fft_f.re[0] += 1024.0;
        let err = fft_f.verify_output(k, &zero, &zero, &zero, &zero);
        assert!(matches!(err, Err(FftFailure::ConvolutionMismatch { .. })));
    }

    #[test]
    fn auto_never_selects_unsupported_isa() {
        let none = IsaFlags {
            avx2: false,
            fma: false,
            avx512f: false,
        };
        assert_eq!(
            Kernel::resolve(IsaChoice::Auto, &none).unwrap(),
            Kernel::Scalar
        );
        assert!(Kernel::resolve(IsaChoice::Avx512, &none).is_err());
        let zen3 = IsaFlags {
            avx2: true,
            fma: true,
            avx512f: false,
        };
        assert_eq!(
            Kernel::resolve(IsaChoice::Auto, &zen3).unwrap(),
            Kernel::Avx2Fma
        );
    }
}
