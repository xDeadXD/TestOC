//! Shared-memory layout v1 shared between the Rust reader and the .NET sensor sidecar.
//!
//! ```text
//! Header (256 B)
//!   0  u32 magic = 0x4D534C53 ('SLSM')   4  u16 layout_version = 1   6  u16 header_size = 256
//!   8  u16 record_size = 128            10  u16 capacity = 400       12  u32 record_count
//!  16  u64 seq        (seqlock: odd = write in progress)
//!  24  u64 heartbeat_unix_ms             32  u32 writer_pid           36  u32 status
//!  40  u32 control    (written by reader: 1 = shutdown requested)    44  u32 reserved
//!  48  u8[208] status_text (UTF-8, NUL-padded)
//! Record (128 B) x capacity
//!   0  u8 hw_kind   1  u8 sensor_kind   2  u16 index   4  f32 value   8  f32 min   12  f32 max
//!  16  u8[56] identifier (UTF-8)        72  u8[56] name (UTF-8)
//! ```
//! Keep in sync with `sensors-sidecar/SharedMemoryWriter.cs`.

pub const MAGIC: u32 = 0x4D53_4C53;
pub const LAYOUT_VERSION: u16 = 1;
pub const HEADER_SIZE: usize = 256;
pub const RECORD_SIZE: usize = 128;
pub const CAPACITY: usize = 400;
pub const MAPPING_SIZE: usize = HEADER_SIZE + RECORD_SIZE * CAPACITY; // 51_456 B

pub mod off {
    pub const MAGIC: usize = 0;
    pub const VERSION: usize = 4;
    pub const HEADER_SIZE: usize = 6;
    pub const RECORD_SIZE: usize = 8;
    pub const CAPACITY: usize = 10;
    pub const RECORD_COUNT: usize = 12;
    pub const SEQ: usize = 16;
    pub const HEARTBEAT: usize = 24;
    pub const WRITER_PID: usize = 32;
    pub const STATUS: usize = 36;
    pub const CONTROL: usize = 40;
    pub const STATUS_TEXT: usize = 48;
    pub const STATUS_TEXT_LEN: usize = 208;

    pub const R_HW_KIND: usize = 0;
    pub const R_SENSOR_KIND: usize = 1;
    pub const R_INDEX: usize = 2;
    pub const R_VALUE: usize = 4;
    pub const R_MIN: usize = 8;
    pub const R_MAX: usize = 12;
    pub const R_IDENT: usize = 16;
    pub const R_NAME: usize = 72;
    pub const R_STR_LEN: usize = 56;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WriterStatus {
    Initializing = 0,
    Ok = 1,
    PawnIoMissing = 2,
    Error = 3,
}

impl WriterStatus {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Ok,
            2 => Self::PawnIoMissing,
            3 => Self::Error,
            _ => Self::Initializing,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HwKind {
    Other = 0,
    Cpu = 1,
    Motherboard = 2,
    SuperIo = 3,
    Gpu = 4,
}

impl HwKind {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Cpu,
            2 => Self::Motherboard,
            3 => Self::SuperIo,
            4 => Self::Gpu,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SensorKind {
    Other = 0,
    Temperature = 1,
    Clock = 2,
    Power = 3,
    Voltage = 4,
    Fan = 5,
    Load = 6,
    Factor = 7,
    SmallData = 8,
}

impl SensorKind {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Temperature,
            2 => Self::Clock,
            3 => Self::Power,
            4 => Self::Voltage,
            5 => Self::Fan,
            6 => Self::Load,
            7 => Self::Factor,
            8 => Self::SmallData,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensorRecord {
    pub hw_kind: HwKind,
    pub sensor_kind: SensorKind,
    pub index: u16,
    pub value: f32,
    pub min: f32,
    pub max: f32,
    pub identifier: String,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct SensorFrame {
    pub seq: u64,
    pub heartbeat_ms: u64,
    pub writer_pid: u32,
    pub status: WriterStatus,
    pub status_text: String,
    pub records: Vec<SensorRecord>,
}

/// Errors detected while validating a snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    /// Writer hasn't initialized the header yet.
    NotInitialized,
    /// Magic/version/geometry mismatch.
    Incompatible(String),
}

fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn rd_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn rd_str(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

/// Parse a consistent byte snapshot of the whole mapping.
pub fn parse_frame(buf: &[u8]) -> Result<SensorFrame, FrameError> {
    if buf.len() < MAPPING_SIZE {
        return Err(FrameError::Incompatible(format!(
            "mapping too small: {}",
            buf.len()
        )));
    }
    let magic = rd_u32(buf, off::MAGIC);
    if magic == 0 {
        return Err(FrameError::NotInitialized);
    }
    if magic != MAGIC {
        return Err(FrameError::Incompatible(format!("bad magic 0x{magic:08X}")));
    }
    let ver = rd_u16(buf, off::VERSION);
    let hs = rd_u16(buf, off::HEADER_SIZE) as usize;
    let rs = rd_u16(buf, off::RECORD_SIZE) as usize;
    let cap = rd_u16(buf, off::CAPACITY) as usize;
    if ver != LAYOUT_VERSION || hs != HEADER_SIZE || rs != RECORD_SIZE || cap != CAPACITY {
        return Err(FrameError::Incompatible(format!(
            "layout v{ver} hs={hs} rs={rs} cap={cap}, expected v{LAYOUT_VERSION} hs={HEADER_SIZE} rs={RECORD_SIZE} cap={CAPACITY}"
        )));
    }
    let count = (rd_u32(buf, off::RECORD_COUNT) as usize).min(CAPACITY);
    let mut records = Vec::with_capacity(count);
    for i in 0..count {
        let r = &buf[HEADER_SIZE + i * RECORD_SIZE..HEADER_SIZE + (i + 1) * RECORD_SIZE];
        records.push(SensorRecord {
            hw_kind: HwKind::from_u8(r[off::R_HW_KIND]),
            sensor_kind: SensorKind::from_u8(r[off::R_SENSOR_KIND]),
            index: rd_u16(r, off::R_INDEX),
            value: rd_f32(r, off::R_VALUE),
            min: rd_f32(r, off::R_MIN),
            max: rd_f32(r, off::R_MAX),
            identifier: rd_str(&r[off::R_IDENT..off::R_IDENT + off::R_STR_LEN]),
            name: rd_str(&r[off::R_NAME..off::R_NAME + off::R_STR_LEN]),
        });
    }
    Ok(SensorFrame {
        seq: rd_u64(buf, off::SEQ),
        heartbeat_ms: rd_u64(buf, off::HEARTBEAT),
        writer_pid: rd_u32(buf, off::WRITER_PID),
        status: WriterStatus::from_u32(rd_u32(buf, off::STATUS)),
        status_text: rd_str(&buf[off::STATUS_TEXT..off::STATUS_TEXT + off::STATUS_TEXT_LEN]),
        records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_is_consistent() {
        assert_eq!(off::STATUS_TEXT + off::STATUS_TEXT_LEN, HEADER_SIZE);
        assert_eq!(off::R_NAME + off::R_STR_LEN, RECORD_SIZE);
        assert_eq!(off::R_IDENT + off::R_STR_LEN, off::R_NAME);
        assert_eq!(
            off::SEQ % 8,
            0,
            "seq must be 8-byte aligned for atomic access"
        );
        assert_eq!(MAPPING_SIZE, 51_456);
    }

    #[test]
    fn zeroed_mapping_is_not_initialized() {
        let buf = vec![0u8; MAPPING_SIZE];
        assert_eq!(parse_frame(&buf).unwrap_err(), FrameError::NotInitialized);
    }
}
