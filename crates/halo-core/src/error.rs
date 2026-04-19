//! Unified error type.
//!
//! Every fallible entry point in `halo-core` returns `Result<_, HaloError>`.
//! Variants are chosen to match the failure modes of the underlying C++
//! loaders (`rcpp_status_t`): invalid arg, unsupported version, I/O, and a
//! small number of format-specific corruption cases.

use std::io;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum HaloError {
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("I/O error: {0}")]
    RawIo(#[from] io::Error),

    #[error("bad magic: expected {expected:?}, got {got:?}")]
    BadMagic { expected: [u8; 4], got: [u8; 4] },

    #[error("unsupported format version {version} (supported: {min}..={max})")]
    UnsupportedVersion { version: i32, min: i32, max: i32 },

    #[error("truncated file: needed {needed} bytes at offset {offset}, only {have} available")]
    Truncated {
        offset: usize,
        needed: usize,
        have: usize,
    },

    #[error("invalid h1b config: {0}")]
    InvalidConfig(&'static str),

    #[error("tokenizer piece is unknown: {0:?}")]
    UnknownBytePiece(Vec<u8>),

    #[error("sampler error: {0}")]
    Sampler(&'static str),
}

impl HaloError {
    pub fn io_at(path: impl Into<PathBuf>, source: io::Error) -> Self {
        HaloError::Io {
            path: path.into(),
            source,
        }
    }
}
