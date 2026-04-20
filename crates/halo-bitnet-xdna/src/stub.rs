//! stub.rs — no-op backend used when the `real-xrt` feature is off.
//!
//! The crate still type-checks and links without libxrt on the host. Every
//! public entry point that would otherwise touch the NPU returns
//! [`XdnaError::UnsupportedStub`] so callers can distinguish "no NPU on
//! this build" from "NPU is there but kernel failed".
//!
//! This file is `mod`'d from `lib.rs` only when `cfg(not(feature = "real-xrt"))`.

use std::path::Path;

use crate::XdnaError;

/// Opaque marker — the stub backend holds no real state, but we keep an
/// empty struct so the public API has the same shape as the real backend.
#[derive(Debug)]
pub(crate) struct StubDeviceInner {
    /// Preserved only so constructors can record the index the caller tried
    /// to open (useful for logging).
    pub(crate) _bdf_idx: u32,
    /// Whether [`load_xclbin`](super::XdnaDevice::load_xclbin) was called.
    /// Tracked so path-sanity tests can observe that stub mode still does
    /// its file-existence pre-check before reporting UnsupportedStub.
    pub(crate) xclbin_path: Option<std::path::PathBuf>,
}

impl StubDeviceInner {
    pub(crate) fn open(_bdf_idx: u32) -> Result<Self, XdnaError> {
        // Stub mode: the "hardware" doesn't exist. Return a clear error
        // rather than a silent success, so callers that forgot to enable
        // real-xrt fail loudly instead of silently skipping NPU work.
        Err(XdnaError::UnsupportedStub)
    }

    pub(crate) fn load_xclbin(&mut self, path: &Path) -> Result<(), XdnaError> {
        // File-existence pre-check runs even in stub mode, so the error
        // distinguishes "path typo" from "no NPU hardware".
        if !path.exists() {
            return Err(XdnaError::NotFound(path.display().to_string()));
        }
        self.xclbin_path = Some(path.to_path_buf());
        Err(XdnaError::UnsupportedStub)
    }

    pub(crate) fn run_kernel(
        &mut self,
        _name: &str,
        _input: &[u8],
        _output: &mut [u8],
    ) -> Result<(), XdnaError> {
        Err(XdnaError::UnsupportedStub)
    }
}
