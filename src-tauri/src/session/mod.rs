//! Session recording (auto-started with each stress run, or manual) and export.

pub mod csv_export;
pub mod summary;

use std::collections::VecDeque;

use crate::model::*;

/// 4 h at 2 Hz.
pub const MAX_SAMPLES: usize = 28_800;

#[derive(Debug, Clone)]
pub struct Session {
    pub id: u64,
    pub label: String,
    pub auto: bool,
    pub started_ms: u64,
    pub stopped_ms: Option<u64>,
    pub samples: VecDeque<TelemetrySample>,
    /// (sample seq, alert text)
    pub alerts: Vec<(u64, String)>,
    pub runs: Vec<StressStatus>,
    pub truncated: bool,
}

impl Session {
    pub fn info(&self) -> SessionInfo {
        SessionInfo {
            id: self.id,
            label: self.label.clone(),
            started_ms: self.started_ms,
            stopped_ms: self.stopped_ms,
            samples: self.samples.len() as u32,
            recording: self.stopped_ms.is_none(),
            auto: self.auto,
        }
    }
}

#[derive(Default)]
pub struct SessionRecorder {
    active: Option<Session>,
    last: Option<Session>,
    next_id: u64,
    last_seq: u64,
}

impl SessionRecorder {
    pub fn start(&mut self, label: Option<String>, auto: bool) -> SessionInfo {
        if let Some(s) = &self.active {
            // An auto start (stress run) never replaces an active session, and a manual start
            // keeps an active manual session. Only "manual start while auto-recording" restarts.
            if auto || !s.auto {
                return s.info();
            }
        }
        self.stop();
        self.next_id += 1;
        let now = now_ms();
        let label = label.filter(|l| !l.trim().is_empty()).unwrap_or_else(|| {
            chrono::Local::now()
                .format(if auto {
                    "stress-%Y%m%d-%H%M%S"
                } else {
                    "session-%Y%m%d-%H%M%S"
                })
                .to_string()
        });
        let s = Session {
            id: self.next_id,
            label,
            auto,
            started_ms: now,
            stopped_ms: None,
            samples: VecDeque::new(),
            alerts: Vec::new(),
            runs: Vec::new(),
            truncated: false,
        };
        let info = s.info();
        self.active = Some(s);
        info
    }

    pub fn stop(&mut self) -> Option<SessionInfo> {
        let mut s = self.active.take()?;
        s.stopped_ms = Some(now_ms());
        let info = s.info();
        self.last = Some(s);
        Some(info)
    }

    /// Stop only if the active session was auto-started (end of a stress run).
    pub fn stop_auto(&mut self) -> Option<SessionInfo> {
        if self.active.as_ref().is_some_and(|s| s.auto) {
            self.stop()
        } else {
            None
        }
    }

    pub fn push(&mut self, sample: &TelemetrySample) {
        self.last_seq = sample.seq;
        if let Some(s) = &mut self.active {
            if s.samples.len() >= MAX_SAMPLES {
                s.samples.pop_front();
                s.truncated = true;
            }
            s.samples.push_back(sample.clone());
        }
    }

    pub fn add_alert(&mut self, text: String) {
        if let Some(s) = &mut self.active {
            s.alerts.push((self.last_seq, text));
        }
    }

    pub fn record_run(&mut self, st: &StressStatus) {
        if let Some(s) = &mut self.active {
            s.runs.push(st.clone());
        }
    }

    pub fn info(&self) -> Option<SessionInfo> {
        self.active
            .as_ref()
            .or(self.last.as_ref())
            .map(Session::info)
    }

    /// Clone of the session to export (active, else last). Cloned under the lock; the caller
    /// writes files after releasing it.
    pub fn export_snapshot(&self) -> Option<Session> {
        self.active.as_ref().or(self.last.as_ref()).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_session_does_not_replace_manual_one() {
        let mut r = SessionRecorder::default();
        let m = r.start(Some("manual".into()), false);
        let a = r.start(None, true);
        assert_eq!(m.id, a.id);
        assert!(r.stop_auto().is_none());
        assert!(r.stop().is_some());
        let a = r.start(None, true);
        assert!(a.auto);
        assert!(r.stop_auto().is_some());
        assert_eq!(r.info().unwrap().id, a.id);
    }
}
