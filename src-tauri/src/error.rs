//! Application error type. Serialized to the frontend as `{ code, message }`.

use serde::{ser::SerializeStruct, Serialize, Serializer};

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("a stress test is already running")]
    Busy,
    #[error("no stress test is running")]
    NotRunning,
    #[error("CPU thermal protection is unavailable (sensor sidecar not running); acknowledgement required")]
    ThermalUnavailable,
    #[error("safety limit already exceeded: {0}")]
    LimitExceeded(String),
    #[error("not supported on this machine: {0}")]
    Unsupported(String),
    #[error("no session data to export")]
    NoSession,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("{0}")]
    Internal(String),
}

impl AppError {
    pub fn code(&self) -> &'static str {
        match self {
            AppError::InvalidArgument(_) => "invalidArgument",
            AppError::Busy => "busy",
            AppError::NotRunning => "notRunning",
            AppError::ThermalUnavailable => "thermalUnavailable",
            AppError::LimitExceeded(_) => "limitExceeded",
            AppError::Unsupported(_) => "unsupported",
            AppError::NoSession => "noSession",
            AppError::Io(_) => "io",
            AppError::Csv(_) => "csv",
            AppError::Internal(_) => "internal",
        }
    }
}

impl Serialize for AppError {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("AppError", 2)?;
        s.serialize_field("code", self.code())?;
        s.serialize_field("message", &self.to_string())?;
        s.end()
    }
}

pub type AppResult<T> = Result<T, AppError>;
