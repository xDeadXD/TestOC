//! Live per-logical-CPU effective clocks from Windows PDH:
//! `Processor Frequency x % Processor Performance / 100`. No admin rights required.

use windows_sys::Win32::System::Performance::{
    PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW,
    PdhOpenQueryW, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE, PDH_MORE_DATA,
};

use crate::win::{from_wide_ptr, wide};

type Handle = *mut core::ffi::c_void;

pub struct PdhClock {
    query: Handle,
    perf: Handle,
    freq: Handle,
    primed: bool,
    buf: Vec<u8>,
}

unsafe impl Send for PdhClock {}

impl PdhClock {
    pub fn new() -> Option<Self> {
        unsafe {
            let mut query: Handle = std::ptr::null_mut();
            if PdhOpenQueryW(std::ptr::null(), 0, &mut query as *mut _ as _) != 0 {
                return None;
            }
            let mut perf: Handle = std::ptr::null_mut();
            let mut freq: Handle = std::ptr::null_mut();
            let p1 = wide("\\Processor Information(*)\\% Processor Performance");
            let p2 = wide("\\Processor Information(*)\\Processor Frequency");
            if PdhAddEnglishCounterW(query as _, p1.as_ptr(), 0, &mut perf as *mut _ as _) != 0
                || PdhAddEnglishCounterW(query as _, p2.as_ptr(), 0, &mut freq as *mut _ as _) != 0
            {
                PdhCloseQuery(query as _);
                return None;
            }
            PdhCollectQueryData(query as _);
            Some(Self {
                query,
                perf,
                freq,
                primed: false,
                buf: Vec::new(),
            })
        }
    }

    /// Returns effective MHz per logical CPU index (group 0). Empty until two samples exist.
    pub fn sample(&mut self, logical: usize) -> Vec<Option<f32>> {
        unsafe {
            if PdhCollectQueryData(self.query as _) != 0 {
                return vec![None; logical];
            }
        }
        if !self.primed {
            self.primed = true;
            return vec![None; logical];
        }
        let perf = self.read(self.perf);
        let freq = self.read(self.freq);
        let mut out = vec![None; logical];
        for (idx, p) in perf {
            if let Some((_, f)) = freq.iter().find(|(i, _)| *i == idx) {
                if idx < logical && *f > 0.0 {
                    out[idx] = Some((f * p / 100.0) as f32);
                }
            }
        }
        out
    }

    fn read(&mut self, counter: Handle) -> Vec<(usize, f64)> {
        unsafe {
            let mut size = 0u32;
            let mut count = 0u32;
            let r = PdhGetFormattedCounterArrayW(
                counter as _,
                PDH_FMT_DOUBLE,
                &mut size,
                &mut count,
                std::ptr::null_mut(),
            );
            if r as u32 != PDH_MORE_DATA || size == 0 {
                return Vec::new();
            }
            self.buf.resize(size as usize, 0);
            let items = self.buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
            if PdhGetFormattedCounterArrayW(
                counter as _,
                PDH_FMT_DOUBLE,
                &mut size,
                &mut count,
                items,
            ) != 0
            {
                return Vec::new();
            }
            let mut out = Vec::with_capacity(count as usize);
            for i in 0..count as usize {
                let it = &*items.add(i);
                if it.FmtValue.CStatus != 0 {
                    continue;
                }
                let name = from_wide_ptr(it.szName);
                if let Some(idx) = parse_instance(&name) {
                    out.push((idx, it.FmtValue.Anonymous.doubleValue));
                }
            }
            out
        }
    }
}

/// "0,5" -> 5 (group 0 only); "_Total" / "0,_Total" -> None.
pub fn parse_instance(name: &str) -> Option<usize> {
    let (g, i) = name.split_once(',')?;
    let g: usize = g.parse().ok()?;
    let i: usize = i.parse().ok()?;
    (g == 0).then_some(i)
}

impl Drop for PdhClock {
    fn drop(&mut self) {
        unsafe {
            PdhCloseQuery(self.query as _);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_parsing() {
        assert_eq!(parse_instance("0,5"), Some(5));
        assert_eq!(parse_instance("0,_Total"), None);
        assert_eq!(parse_instance("_Total"), None);
        assert_eq!(parse_instance("1,3"), None);
    }

    #[test]
    fn pdh_reports_plausible_clocks() {
        let Some(mut p) = PdhClock::new() else {
            eprintln!("PDH unavailable");
            return;
        };
        let n = std::thread::available_parallelism().unwrap().get();
        p.sample(n);
        std::thread::sleep(std::time::Duration::from_millis(300));
        let v = p.sample(n);
        let got: Vec<f32> = v.iter().flatten().copied().collect();
        assert!(!got.is_empty(), "no clocks");
        assert!(got.iter().all(|&m| m > 100.0 && m < 8000.0), "{got:?}");
    }
}
