//! Named shared-memory mapping with a lock-free seqlock protocol.
//!
//! The unelevated Rust app *creates* the mapping with a DACL restricted to the current user SID;
//! the elevated sidecar opens and writes it. Reads never block: if the writer is mid-update we
//! retry a few times and otherwise keep the previous frame.

use std::sync::atomic::{fence, AtomicU32, AtomicU64, Ordering};

use windows_sys::Win32::Foundation::{CloseHandle, LocalFree, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
use windows_sys::Win32::Security::{
    GetTokenInformation, TokenUser, PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES, TOKEN_QUERY,
    TOKEN_USER,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_ALL_ACCESS,
    MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

use super::layout::{off, MAPPING_SIZE};
use crate::win::{from_wide_ptr, last_error, wide};

pub struct ShmRegion {
    handle: HANDLE,
    view: *mut u8,
}

// The view is plain shared memory accessed only through atomics / volatile ops.
unsafe impl Send for ShmRegion {}
unsafe impl Sync for ShmRegion {}

/// Returns the string SID (e.g. `S-1-5-21-...`) of the current process user.
pub fn current_user_sid() -> Result<String, String> {
    unsafe {
        let mut token: HANDLE = std::ptr::null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return Err(format!("OpenProcessToken failed ({})", last_error()));
        }
        let mut len = 0u32;
        GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut len);
        let mut buf = vec![0u8; len as usize];
        let ok = GetTokenInformation(token, TokenUser, buf.as_mut_ptr().cast(), len, &mut len);
        CloseHandle(token);
        if ok == 0 {
            return Err(format!("GetTokenInformation failed ({})", last_error()));
        }
        let tu = &*(buf.as_ptr() as *const TOKEN_USER);
        let mut sid_str: *mut u16 = std::ptr::null_mut();
        if ConvertSidToStringSidW(tu.User.Sid, &mut sid_str) == 0 {
            return Err(format!("ConvertSidToStringSidW failed ({})", last_error()));
        }
        let s = from_wide_ptr(sid_str);
        LocalFree(sid_str.cast());
        Ok(s)
    }
}

impl ShmRegion {
    /// Create a new mapping. `name = None` creates an anonymous mapping (tests).
    /// The DACL grants GENERIC_ALL only to the current user; nobody else can open it.
    pub fn create(name: Option<&str>) -> Result<Self, String> {
        let sid = current_user_sid()?;
        let sddl = wide(format!("D:P(A;;GA;;;{sid})"));
        unsafe {
            let mut sd: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
            if ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut sd,
                std::ptr::null_mut(),
            ) == 0
            {
                return Err(format!("SDDL conversion failed ({})", last_error()));
            }
            let sa = SECURITY_ATTRIBUTES {
                nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: sd,
                bInheritHandle: 0,
            };
            let wname = name.map(wide);
            let handle = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                &sa,
                PAGE_READWRITE,
                0,
                MAPPING_SIZE as u32,
                wname.as_ref().map_or(std::ptr::null(), |w| w.as_ptr()),
            );
            let err = last_error();
            LocalFree(sd);
            if handle.is_null() {
                return Err(format!("CreateFileMappingW failed ({err})"));
            }
            if name.is_some() && err == windows_sys::Win32::Foundation::ERROR_ALREADY_EXISTS {
                CloseHandle(handle);
                return Err("mapping name collision".into());
            }
            Self::map(handle)
        }
    }

    /// Open an existing named mapping (used by the probe / test writer).
    pub fn open(name: &str) -> Result<Self, String> {
        let wname = wide(name);
        unsafe {
            let handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, wname.as_ptr());
            if handle.is_null() {
                return Err(format!("OpenFileMappingW failed ({})", last_error()));
            }
            Self::map(handle)
        }
    }

    unsafe fn map(handle: HANDLE) -> Result<Self, String> {
        let view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, MAPPING_SIZE);
        if view.Value.is_null() {
            let e = last_error();
            CloseHandle(handle);
            return Err(format!("MapViewOfFile failed ({e})"));
        }
        Ok(Self {
            handle,
            view: view.Value as *mut u8,
        })
    }

    fn seq(&self) -> &AtomicU64 {
        unsafe { &*(self.view.add(off::SEQ) as *const AtomicU64) }
    }

    fn u32_at(&self, o: usize) -> &AtomicU32 {
        unsafe { &*(self.view.add(o) as *const AtomicU32) }
    }

    /// Seqlock read of the active header + `record_count` records into `out`.
    /// Returns the (even) sequence number on success, or `None` if the writer was busy.
    pub fn read_consistent(&self, out: &mut Vec<u8>) -> Option<u64> {
        use super::layout::{CAPACITY, HEADER_SIZE, RECORD_SIZE};
        out.resize(MAPPING_SIZE, 0);
        for _ in 0..8 {
            // Spin/yield briefly if the writer currently holds the odd sequence number.
            let s1 = self.seq().load(Ordering::Acquire);
            if s1 & 1 == 1 {
                std::hint::spin_loop();
                std::thread::yield_now();
                continue;
            }
            unsafe {
                let src = self.view as *const u64;
                let dst = out.as_mut_ptr() as *mut u64;
                // Copy the 256-byte header first, then only the populated records.
                for i in 0..(HEADER_SIZE / 8) {
                    dst.add(i)
                        .write_unaligned(std::ptr::read_volatile(src.add(i)));
                }
                let count = (u32::from_le_bytes(
                    out[off::RECORD_COUNT..off::RECORD_COUNT + 4]
                        .try_into()
                        .unwrap(),
                ) as usize)
                    .min(CAPACITY);
                let rec_words = (count * RECORD_SIZE) / 8;
                let base = HEADER_SIZE / 8;
                for i in 0..rec_words {
                    dst.add(base + i)
                        .write_unaligned(std::ptr::read_volatile(src.add(base + i)));
                }
            }
            // Reject the copy if the writer incremented `seq` while we were reading.
            fence(Ordering::Acquire);
            if self.seq().load(Ordering::Relaxed) == s1 {
                return Some(s1);
            }
        }
        None
    }

    /// Reader → writer control word (1 = shutdown requested).
    pub fn set_control(&self, v: u32) {
        self.u32_at(off::CONTROL).store(v, Ordering::SeqCst);
    }

    pub fn control(&self) -> u32 {
        self.u32_at(off::CONTROL).load(Ordering::SeqCst)
    }

    /// Zero the 256-byte header before relaunching the sidecar (no live writer exists).
    pub fn reset_header(&self) {
        let zeros = [0u8; super::layout::HEADER_SIZE];
        self.write_bytes(0, &zeros);
        fence(Ordering::SeqCst);
    }

    // ----- writer side (used by tests, the probe example and `--fake` emulation) -----

    pub fn begin_write(&self) {
        self.seq().fetch_add(1, Ordering::Relaxed);
        fence(Ordering::Release);
    }

    pub fn end_write(&self) {
        self.seq().fetch_add(1, Ordering::Release);
    }

    /// Volatile write at `offset` (uses 8-byte words when aligned for speed).
    pub fn write_bytes(&self, offset: usize, data: &[u8]) {
        assert!(offset + data.len() <= MAPPING_SIZE);
        unsafe {
            if offset.is_multiple_of(8) && data.len().is_multiple_of(8) {
                let dst = self.view.add(offset) as *mut u64;
                let src = data.as_ptr() as *const u64;
                for i in 0..(data.len() / 8) {
                    std::ptr::write_volatile(dst.add(i), src.add(i).read_unaligned());
                }
            } else {
                for (i, b) in data.iter().enumerate() {
                    std::ptr::write_volatile(self.view.add(offset + i), *b);
                }
            }
        }
    }
}

impl Drop for ShmRegion {
    fn drop(&mut self) {
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.view.cast(),
            });
            CloseHandle(self.handle);
        }
    }
}

/// Test/emulation writer that produces frames in the exact v1 layout.
pub struct FrameWriter<'a> {
    pub region: &'a ShmRegion,
}

impl<'a> FrameWriter<'a> {
    pub fn write_header_static(&self, pid: u32) {
        use super::layout::*;
        let r = self.region;
        r.write_bytes(off::VERSION, &LAYOUT_VERSION.to_le_bytes());
        r.write_bytes(off::HEADER_SIZE, &(HEADER_SIZE as u16).to_le_bytes());
        r.write_bytes(off::RECORD_SIZE, &(RECORD_SIZE as u16).to_le_bytes());
        r.write_bytes(off::CAPACITY, &(CAPACITY as u16).to_le_bytes());
        r.write_bytes(off::WRITER_PID, &pid.to_le_bytes());
        r.write_bytes(off::MAGIC, &MAGIC.to_le_bytes());
    }

    /// Write one full frame inside a seqlock window.
    pub fn write_frame(
        &self,
        status: u32,
        text: &str,
        heartbeat_ms: u64,
        records: &[super::layout::SensorRecord],
    ) {
        use super::layout::*;
        let r = self.region;
        r.begin_write();
        let n = records.len().min(CAPACITY);
        for (i, rec) in records.iter().take(n).enumerate() {
            let base = HEADER_SIZE + i * RECORD_SIZE;
            let mut buf = [0u8; RECORD_SIZE];
            buf[off::R_HW_KIND] = rec.hw_kind as u8;
            buf[off::R_SENSOR_KIND] = rec.sensor_kind as u8;
            buf[off::R_INDEX..off::R_INDEX + 2].copy_from_slice(&rec.index.to_le_bytes());
            buf[off::R_VALUE..off::R_VALUE + 4].copy_from_slice(&rec.value.to_le_bytes());
            buf[off::R_MIN..off::R_MIN + 4].copy_from_slice(&rec.min.to_le_bytes());
            buf[off::R_MAX..off::R_MAX + 4].copy_from_slice(&rec.max.to_le_bytes());
            put_str(
                &mut buf[off::R_IDENT..off::R_IDENT + off::R_STR_LEN],
                &rec.identifier,
            );
            put_str(
                &mut buf[off::R_NAME..off::R_NAME + off::R_STR_LEN],
                &rec.name,
            );
            r.write_bytes(base, &buf);
        }
        r.write_bytes(off::RECORD_COUNT, &(n as u32).to_le_bytes());
        r.write_bytes(off::HEARTBEAT, &heartbeat_ms.to_le_bytes());
        r.write_bytes(off::STATUS, &status.to_le_bytes());
        let mut t = [0u8; off::STATUS_TEXT_LEN];
        put_str(&mut t, text);
        r.write_bytes(off::STATUS_TEXT, &t);
        r.end_write();
    }
}

fn put_str(dst: &mut [u8], s: &str) {
    // Truncate on a char boundary, keep at least one NUL.
    let max = dst.len() - 1;
    let mut end = s.len().min(max);
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    dst[..end].copy_from_slice(&s.as_bytes()[..end]);
}

#[cfg(test)]
mod tests {
    use super::super::layout::*;
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    fn rec(i: usize, v: f32) -> SensorRecord {
        SensorRecord {
            hw_kind: HwKind::Cpu,
            sensor_kind: SensorKind::Temperature,
            index: i as u16,
            value: v,
            min: v,
            max: v,
            identifier: format!("/test/{i}"),
            name: format!("Sensor {i}"),
        }
    }

    #[test]
    fn roundtrip_frame() {
        let region = ShmRegion::create(None).unwrap();
        let w = FrameWriter { region: &region };
        w.write_header_static(4242);
        let recs: Vec<_> = (0..10).map(|i| rec(i, i as f32 * 1.5)).collect();
        w.write_frame(1, "ok", 123_456, &recs);
        let mut buf = Vec::new();
        let seq = region.read_consistent(&mut buf).unwrap();
        assert_eq!(seq, 2);
        let f = parse_frame(&buf).unwrap();
        assert_eq!(f.writer_pid, 4242);
        assert_eq!(f.heartbeat_ms, 123_456);
        assert_eq!(f.status, WriterStatus::Ok);
        assert_eq!(f.status_text, "ok");
        assert_eq!(f.records, recs);
    }

    #[test]
    fn named_mapping_is_openable_by_same_user() {
        let name = format!("Local\\TestOC.Test.{}", std::process::id());
        let owner = ShmRegion::create(Some(&name)).unwrap();
        let other = ShmRegion::open(&name).unwrap();
        FrameWriter { region: &other }.write_header_static(7);
        let mut buf = Vec::new();
        owner.read_consistent(&mut buf).unwrap();
        assert_eq!(parse_frame(&buf).unwrap().writer_pid, 7);
        other.set_control(1);
        assert_eq!(owner.control(), 1);
    }

    /// Writer hammers frames where every record value == frame number; the reader must never
    /// observe a mix of two frames.
    #[test]
    fn seqlock_never_returns_torn_frames() {
        let region = Arc::new(ShmRegion::create(None).unwrap());
        FrameWriter { region: &region }.write_header_static(1);
        let stop = Arc::new(AtomicBool::new(false));
        let writer = {
            let (region, stop) = (region.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut n = 0u64;
                let mut recs: Vec<_> = (0..64).map(|i| rec(i, 0.0)).collect();
                while !stop.load(Ordering::Relaxed) {
                    n += 1;
                    for r in recs.iter_mut() {
                        r.value = n as f32;
                    }
                    FrameWriter { region: &region }.write_frame(1, "", n, &recs);
                    // Keep the writer pounding at >1 kHz while leaving real even-seq windows in debug builds.
                    for _ in 0..200 {
                        std::hint::spin_loop();
                    }
                    std::thread::yield_now();
                }
                n
            })
        };
        let mut buf = Vec::new();
        let (mut ok, mut busy) = (0u32, 0u32);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while ok < 2_000 && std::time::Instant::now() < deadline {
            match region.read_consistent(&mut buf) {
                Some(_) => {
                    let f = parse_frame(&buf).unwrap();
                    if f.records.is_empty() {
                        continue;
                    }
                    let hb = f.heartbeat_ms as f32;
                    assert!(
                        f.records.iter().all(|r| r.value == hb),
                        "torn read at frame {}",
                        f.heartbeat_ms
                    );
                    ok += 1;
                }
                None => busy += 1,
            }
        }
        stop.store(true, Ordering::Relaxed);
        let frames = writer.join().unwrap();
        assert!(
            ok >= 1_000,
            "too few successful reads: ok={ok} busy={busy} frames={frames}"
        );
    }
}
