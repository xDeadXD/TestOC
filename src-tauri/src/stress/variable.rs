//! Variable-load mode: a deterministic pseudo-random schedule of load levels, plus a per-worker
//! duty-cycle throttle that keeps abort latency low (sleeps in <= 5 ms slices).

use std::time::{Duration, Instant};

pub const LEVELS: [u8; 6] = [100, 50, 0, 100, 75, 25];
pub const WINDOW: Duration = Duration::from_millis(20);
const SLEEP_SLICE: Duration = Duration::from_millis(5);

/// Every 2-10 s the whole worker pool switches level in sync, producing sharp current transients.
pub struct VariableSchedule {
    state: u64,
    segment_end: Duration,
    level: u8,
    min_pct: u8,
    max_pct: u8,
}

impl VariableSchedule {
    pub fn new(seed: u64) -> Self {
        Self::with_range(seed, 0, 100)
    }

    pub fn with_range(seed: u64, min_pct: u8, max_pct: u8) -> Self {
        let lo = min_pct.min(100);
        let hi = max_pct.clamp(lo, 100);
        let mut s = Self {
            state: seed | 1,
            segment_end: Duration::ZERO,
            level: hi,
            min_pct: lo,
            max_pct: hi,
        };
        s.advance();
        s
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn pick_level(&mut self) -> u8 {
        if self.min_pct == 0 && self.max_pct == 100 {
            return LEVELS[(self.next() % LEVELS.len() as u64) as usize];
        }
        let span = (self.max_pct - self.min_pct) as u64;
        if span == 0 {
            return self.max_pct;
        }
        let step = self.next() % 5;
        (self.min_pct as u64 + (span * step + 2) / 4).min(self.max_pct as u64) as u8
    }

    fn advance(&mut self) {
        let len_ms = 2000 + self.next() % 8001; // 2-10 s
        self.segment_end += Duration::from_millis(len_ms);
        let mut lvl = self.pick_level();
        if lvl == self.level && self.max_pct > self.min_pct {
            lvl = self.pick_level();
        }
        self.level = lvl;
    }

    pub fn level_at(&mut self, elapsed: Duration) -> u8 {
        while elapsed >= self.segment_end {
            self.advance();
        }
        self.level
    }
}

/// Adaptive GPU ramp: steps smoothly from `min_pct` to `max_pct` in 10% increments every
/// `step_s` seconds, then wraps back to `min_pct` to sweep the GPU V/F boost curve.
pub fn adaptive_level(elapsed: Duration, min_pct: u8, max_pct: u8, step_s: u8) -> u8 {
    let lo = min_pct.clamp(10, 100);
    let hi = max_pct.clamp(lo, 100);
    if hi == lo {
        return hi;
    }
    let step_secs = (step_s as u64).clamp(1, 60);
    let steps = ((hi - lo) as u64 / 10).max(1) + 1;
    let idx = (elapsed.as_secs() / step_secs) % steps;
    (lo as u64 + idx * 10).min(hi as u64) as u8
}

/// Action returned by `Throttle::pace`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pace {
    /// Proceed with the next work block.
    Work,
    /// Load level is 0%: slept one window, re-read `load_level` without doing work.
    Skip,
    /// Stop/abort requested while sleeping.
    Stop,
}

/// Per-worker duty-cycle throttle: within each 20 ms window, work for `level`% then sleep.
pub struct Throttle {
    window_start: Instant,
}

impl Default for Throttle {
    fn default() -> Self {
        Self {
            window_start: Instant::now(),
        }
    }
}

impl Throttle {
    /// Call before each work block. Sleeps if the busy budget for this window is used up.
    /// Never recurses: at `level == 0` it sleeps at most one `WINDOW` and returns `Pace::Skip`
    /// so the caller re-reads the current schedule level.
    pub fn pace(&mut self, level: u8, keep_going: impl Fn() -> bool) -> Pace {
        if !keep_going() {
            return Pace::Stop;
        }
        if level >= 100 {
            self.window_start = Instant::now();
            return Pace::Work;
        }
        if level == 0 {
            let start = Instant::now();
            while start.elapsed() < WINDOW {
                if !keep_going() {
                    return Pace::Stop;
                }
                let rest = WINDOW.saturating_sub(start.elapsed());
                std::thread::sleep(rest.min(SLEEP_SLICE));
            }
            self.window_start = Instant::now();
            return Pace::Skip;
        }
        let budget = WINDOW.mul_f32(level as f32 / 100.0);
        if self.window_start.elapsed() < budget {
            return Pace::Work;
        }
        while self.window_start.elapsed() < WINDOW {
            if !keep_going() {
                return Pace::Stop;
            }
            let rest = WINDOW.saturating_sub(self.window_start.elapsed());
            std::thread::sleep(rest.min(SLEEP_SLICE));
        }
        self.window_start = Instant::now();
        Pace::Work
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_is_deterministic_and_bounded() {
        let mut a = VariableSchedule::new(5);
        let mut b = VariableSchedule::new(5);
        let mut seen = std::collections::HashSet::new();
        for s in 0..600 {
            let t = Duration::from_millis(s * 500);
            let la = a.level_at(t);
            assert_eq!(la, b.level_at(t));
            assert!(LEVELS.contains(&la));
            seen.insert(la);
        }
        assert!(
            seen.len() >= 3,
            "schedule should visit several levels: {seen:?}"
        );
    }

    #[test]
    fn throttle_level0_skips_and_resumes_without_recursion() {
        let mut t = Throttle::default();
        assert_eq!(t.pace(0, || true), Pace::Skip);
        assert_eq!(t.pace(100, || true), Pace::Work);
    }

    #[test]
    fn throttle_level0_exits_promptly_on_stop() {
        let mut t = Throttle::default();
        let start = Instant::now();
        let stop_at = start + Duration::from_millis(30);
        while t.pace(0, || Instant::now() < stop_at) != Pace::Stop {}
        assert!(start.elapsed() < Duration::from_millis(60));
    }
}
