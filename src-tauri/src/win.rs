//! Small Win32 helpers shared across modules.

use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;

/// UTF-16, NUL-terminated.
pub fn wide(s: impl AsRef<OsStr>) -> Vec<u16> {
    s.as_ref().encode_wide().chain(std::iter::once(0)).collect()
}

/// Read a NUL-terminated UTF-16 string.
///
/// # Safety
/// `p` must point to a valid NUL-terminated UTF-16 string.
pub unsafe fn from_wide_ptr(p: *const u16) -> String {
    if p.is_null() {
        return String::new();
    }
    let mut len = 0usize;
    while *p.add(len) != 0 {
        len += 1;
    }
    String::from_utf16_lossy(std::slice::from_raw_parts(p, len))
}

pub fn last_error() -> u32 {
    unsafe { windows_sys::Win32::Foundation::GetLastError() }
}

/// Lower the calling thread's priority (stress workers).
pub fn set_current_thread_below_normal() {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
    };
    unsafe {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

/// Raise the calling thread's priority (telemetry / heartbeat).
pub fn set_current_thread_above_normal() {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
    };
    unsafe {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    }
}

/// Pin the calling thread to one logical processor (processor group 0).
pub fn pin_current_thread(logical_cpu: usize) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
    if logical_cpu < usize::BITS as usize {
        unsafe {
            SetThreadAffinityMask(GetCurrentThread(), 1usize << logical_cpu);
        }
    }
}

/// Live `(total_physical_bytes, available_physical_bytes)` via `GlobalMemoryStatusEx`.
pub fn memory_status() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    unsafe {
        let mut st: MEMORYSTATUSEX = std::mem::zeroed();
        st.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut st) != 0 {
            Some((st.ullTotalPhys, st.ullAvailPhys))
        } else {
            None
        }
    }
}

/// Current physical RAM (working set) actually resident in DRAM for this process.
/// Used to stop 100% RAM top-up the instant Windows starts paging our test buffers to disk.
pub fn process_working_set_bytes() -> Option<u64> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        let mut pmc: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        pmc.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
            Some(pmc.WorkingSetSize as u64)
        } else {
            None
        }
    }
}

/// Hint Windows' Balance Set Manager to keep our allocated DRAM test chunks resident in the
/// working set instead of trimming them out to Standby/Modified pages between passes.
/// Uses soft limits (`flags = 0`) so it works unelevated and never triggers hard OOM crashes.
pub fn hint_working_set_bytes(allocated_bytes: u64) {
    use windows_sys::Win32::System::Memory::SetProcessWorkingSetSizeEx;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let min_ws = (allocated_bytes.saturating_add(128 << 20)) as usize;
    let max_ws = (allocated_bytes.saturating_add(512 << 20)) as usize;
    unsafe {
        let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), min_ws, max_ws, 0);
    }
}
