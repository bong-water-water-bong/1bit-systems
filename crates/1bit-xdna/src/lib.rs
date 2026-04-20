//! 1bit-xdna — Rust wrapper around AMD XRT for dispatching kernels
//! to the XDNA 2 NPU (Strix Halo's on-die AIE array).
//!
//! ## Feature flags
//!
//! - `stub` (default): no libxrt linkage. Every public call returns
//!   [`XdnaError::UnsupportedStub`]. Lets CI on non-NPU hosts build the
//!   workspace without a functioning XRT install.
//! - `real-xrt`: compile the C++ shim in `cpp/shim.cpp` against XRT headers
//!   and link `libxrt_coreutil` + `libxrt_core`. Required on the strixhalo
//!   box once Peano produces a real AIE xclbin.
//!
//! ## Safety model
//!
//! The public API is 100% safe Rust. All `unsafe` blocks live inside this
//! file at the FFI boundary and are gated behind `cfg(feature = "real-xrt")`.
//! The C++ shim catches every exception at the extern-C boundary, so Rust
//! never sees a C++ unwind (which would be UB).
//!
//! ## Why no bindgen?
//!
//! XRT's public headers are C++ (xrt::device, xrt::kernel, xrt::bo). They
//! use std::filesystem, std::shared_ptr, std::optional, and several layers
//! of PIMPL. bindgen on these produces either opaque blobs with no useful
//! methods, or a 50-KLOC mess that breaks on every XRT point release. The
//! hand-rolled extern "C" + C++ glue in `cpp/shim.cpp` is ~100 lines,
//! compiles cleanly, and keeps the FFI boundary narrow.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::path::Path;

// ----- Feature-gated backend modules -----------------------------------------

#[cfg(feature = "real-xrt")]
mod ffi;

#[cfg(not(feature = "real-xrt"))]
mod stub;

// ----- Error type ------------------------------------------------------------

/// Errors returned by the XDNA backend.
#[derive(Debug, thiserror::Error)]
pub enum XdnaError {
    /// The `real-xrt` feature is off, so NPU work is not available in this
    /// build. Not a bug — it's a signal that the caller should fall back
    /// to the HIP (iGPU) path.
    #[error("1bit-xdna: built without real-xrt feature (stub mode)")]
    UnsupportedStub,

    /// The requested file (xclbin) does not exist on disk.
    #[error("1bit-xdna: file not found: {0}")]
    NotFound(String),

    /// Caller-side precondition violated (null pointer, mismatched lengths,
    /// bad UTF-8 path, etc.).
    #[error("1bit-xdna: invalid argument: {0}")]
    InvalidArg(&'static str),

    /// XRT rejected the device open or xclbin load. Usually means no XDNA
    /// silicon, wrong xclbin target, or the XRT runtime is misconfigured.
    #[error("1bit-xdna: XRT device error")]
    Device,

    /// The loaded xclbin does not export the requested kernel name, or no
    /// xclbin has been loaded yet.
    #[error("1bit-xdna: kernel not found in loaded xclbin")]
    Kernel,

    /// Any other XRT failure during dispatch (buffer allocation, kernel
    /// run, memcpy, etc.).
    #[error("1bit-xdna: internal XRT error")]
    Internal,
}

#[cfg(feature = "real-xrt")]
impl XdnaError {
    /// Map a `HALO_XRT_*` status code (from `cpp/shim.h`) to an `XdnaError`.
    fn from_status(code: i32) -> Self {
        match code {
            ffi::HALO_XRT_E_INVALID => Self::InvalidArg("shim rejected argument"),
            ffi::HALO_XRT_E_NOT_FOUND => Self::NotFound("(shim-reported)".into()),
            ffi::HALO_XRT_E_DEVICE => Self::Device,
            ffi::HALO_XRT_E_KERNEL => Self::Kernel,
            ffi::HALO_XRT_E_INTERNAL => Self::Internal,
            _ => Self::Internal,
        }
    }
}

// ----- Public device handle --------------------------------------------------

/// Owning handle to an XDNA NPU device. Constructed via [`XdnaDevice::open`].
///
/// On drop, closes the underlying XRT device handle (real backend) or
/// does nothing (stub backend).
pub struct XdnaDevice {
    #[cfg(feature = "real-xrt")]
    raw: *mut ffi::HaloXrtDevice,
    #[cfg(feature = "real-xrt")]
    xclbin_loaded: bool,
    #[cfg(not(feature = "real-xrt"))]
    inner: stub::StubDeviceInner,
}

// SAFETY: on the real backend the raw handle is not thread-safe per XRT's
// own docs, so we provide neither Send nor Sync. Callers that need to
// share a device between threads wrap it in a Mutex.
// (No unsafe impl Send / Sync — rely on the pointer default: !Send + !Sync.)

/// A named kernel belonging to a loaded xclbin.
///
/// This is a thin marker right now; the current MVP dispatches kernels by
/// name directly from [`XdnaDevice::run_kernel`]. Once Peano produces real
/// xclbins we'll cache `xrt::kernel` handles here to avoid re-resolving
/// the kernel on every call.
pub struct XdnaKernel {
    name: String,
}

impl XdnaKernel {
    /// Name as registered in the xclbin.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl XdnaDevice {
    /// Open the XDNA device at BDF index `idx` (usually 0 on Strix Halo).
    ///
    /// Returns [`XdnaError::UnsupportedStub`] in stub mode.
    pub fn open(idx: u32) -> Result<Self, XdnaError> {
        #[cfg(feature = "real-xrt")]
        {
            // SAFETY: onebit_xrt_open either returns NULL or a valid pointer
            // to an opaque struct allocated by the C++ shim. The pointer
            // is owned by the returned XdnaDevice and freed in Drop.
            let raw = unsafe { ffi::onebit_xrt_open(idx) };
            if raw.is_null() {
                tracing::warn!(bdf_idx = idx, "onebit_xrt_open returned NULL");
                return Err(XdnaError::Device);
            }
            Ok(Self {
                raw,
                xclbin_loaded: false,
            })
        }
        #[cfg(not(feature = "real-xrt"))]
        {
            let _ = tracing::trace_span!("stub_open", idx = idx).entered();
            stub::StubDeviceInner::open(idx).map(|inner| Self { inner })
        }
    }

    /// Load an AIE xclbin from disk. The xclbin stays resident for the
    /// lifetime of this device handle.
    pub fn load_xclbin(&mut self, path: &Path) -> Result<(), XdnaError> {
        #[cfg(feature = "real-xrt")]
        {
            // Pre-check on the Rust side so the error carries the actual
            // path (the shim's E_NOT_FOUND answer loses the string).
            if !path.exists() {
                return Err(XdnaError::NotFound(path.display().to_string()));
            }
            let path_cstr = std::ffi::CString::new(
                path.to_str()
                    .ok_or(XdnaError::InvalidArg("non-UTF-8 xclbin path"))?,
            )
            .map_err(|_| XdnaError::InvalidArg("xclbin path contained NUL byte"))?;

            // SAFETY: self.raw is a non-null handle from onebit_xrt_open;
            // path_cstr is a valid NUL-terminated C string that outlives
            // the call.
            let code = unsafe { ffi::onebit_xrt_load_xclbin(self.raw, path_cstr.as_ptr()) };
            if code == ffi::HALO_XRT_OK {
                self.xclbin_loaded = true;
                Ok(())
            } else if code == ffi::HALO_XRT_E_NOT_FOUND {
                Err(XdnaError::NotFound(path.display().to_string()))
            } else {
                Err(XdnaError::from_status(code))
            }
        }
        #[cfg(not(feature = "real-xrt"))]
        {
            self.inner.load_xclbin(path)
        }
    }

    /// Resolve a kernel by name. Requires [`load_xclbin`](Self::load_xclbin)
    /// to have succeeded first. In stub mode this always fails with
    /// [`XdnaError::UnsupportedStub`].
    pub fn kernel(&self, name: &str) -> Result<XdnaKernel, XdnaError> {
        #[cfg(feature = "real-xrt")]
        {
            if !self.xclbin_loaded {
                return Err(XdnaError::Kernel);
            }
            // Real resolution happens inside run_kernel for now; this
            // method exists so the API can cache handles later without
            // breaking callers.
            Ok(XdnaKernel {
                name: name.to_string(),
            })
        }
        #[cfg(not(feature = "real-xrt"))]
        {
            let _ = name;
            Err(XdnaError::UnsupportedStub)
        }
    }

    /// Dispatch a named kernel with one input buffer and one output buffer.
    ///
    /// This is the MVP surface — one input, one output, fully synchronous.
    /// Rich multi-buffer dispatch will follow when the Peano-generated
    /// xclbins land and we know what shapes the AIE kernels actually want.
    pub fn run_kernel(
        &mut self,
        name: &str,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<(), XdnaError> {
        #[cfg(feature = "real-xrt")]
        {
            if !self.xclbin_loaded {
                return Err(XdnaError::Kernel);
            }
            let name_cstr = std::ffi::CString::new(name)
                .map_err(|_| XdnaError::InvalidArg("kernel name contained NUL byte"))?;
            // SAFETY:
            //  - self.raw is a non-null handle from onebit_xrt_open
            //  - name_cstr is a valid NUL-terminated C string
            //  - input / output slices are valid for their declared lengths
            //    (Rust slice invariant); the shim reads input[0..in_len]
            //    and writes output[0..out_len].
            //  - input and output point at disjoint memory (one &, one &mut).
            let code = unsafe {
                ffi::onebit_xrt_run_kernel(
                    self.raw,
                    name_cstr.as_ptr(),
                    input.as_ptr(),
                    input.len(),
                    output.as_mut_ptr(),
                    output.len(),
                )
            };
            if code == ffi::HALO_XRT_OK {
                Ok(())
            } else {
                Err(XdnaError::from_status(code))
            }
        }
        #[cfg(not(feature = "real-xrt"))]
        {
            self.inner.run_kernel(name, input, output)
        }
    }
}

impl Drop for XdnaDevice {
    fn drop(&mut self) {
        #[cfg(feature = "real-xrt")]
        {
            if !self.raw.is_null() {
                // SAFETY: self.raw was produced by onebit_xrt_open and has
                // not been freed yet. onebit_xrt_close is NUL-safe anyway.
                unsafe { ffi::onebit_xrt_close(self.raw) };
                self.raw = std::ptr::null_mut();
            }
        }
        // stub backend: nothing to free.
    }
}

impl std::fmt::Debug for XdnaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("XdnaDevice");
        #[cfg(feature = "real-xrt")]
        {
            s.field("raw_is_null", &self.raw.is_null())
                .field("xclbin_loaded", &self.xclbin_loaded);
        }
        #[cfg(not(feature = "real-xrt"))]
        {
            s.field("backend", &"stub")
                .field("xclbin_path", &self.inner.xclbin_path);
        }
        s.finish()
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub mode: opening a device must fail cleanly (no panic).
    /// Gated on NOT real-xrt so `cargo test -p 1bit-xdna` on a real
    /// NPU box still exercises the real-backend path.
    #[test]
    #[cfg(not(feature = "real-xrt"))]
    fn stub_open_returns_unsupported() {
        let err = XdnaDevice::open(0).unwrap_err();
        assert!(
            matches!(err, XdnaError::UnsupportedStub),
            "expected UnsupportedStub, got {err:?}",
        );
    }

    /// Feature-gate sanity: when NOT on real-xrt, the ffi module is not
    /// compiled in. We assert via cfg — if someone accidentally makes ffi
    /// unconditional, this test gets deleted and the build breaks elsewhere.
    #[test]
    fn feature_flag_gate_is_coherent() {
        // At least one of these must hold. If both, Cargo itself would
        // have rejected the feature set at resolve time.
        let stub_on = cfg!(feature = "stub");
        let real_on = cfg!(feature = "real-xrt");
        assert!(
            stub_on || real_on,
            "at least one of `stub` / `real-xrt` must be enabled"
        );

        // The `ffi` module exists iff real-xrt is on. We can't name it
        // from a test in the `not(feature = "real-xrt")` case; instead we
        // verify the fallback path is reachable.
        #[cfg(not(feature = "real-xrt"))]
        {
            // stub::StubDeviceInner::open must exist and must refuse.
            assert!(stub::StubDeviceInner::open(0).is_err());
        }
    }

    /// Path sanity: loading a non-existent xclbin must return NotFound,
    /// not panic, not segfault, regardless of backend.
    ///
    /// In stub mode we build the inner struct manually (open() itself
    /// refuses in stub mode); in real-xrt mode we go through open() and
    /// skip if hardware isn't present.
    #[test]
    fn load_xclbin_nonexistent_is_not_found() {
        #[cfg(not(feature = "real-xrt"))]
        {
            // Bypass open()'s UnsupportedStub to reach load_xclbin's
            // file-existence check. Simulates what a caller on real
            // hardware would hit.
            let mut inner = stub::StubDeviceInner {
                _bdf_idx: 0,
                xclbin_path: None,
            };
            let err = inner
                .load_xclbin(Path::new("/tmp/halo-xdna-nonexistent-xclbin-probe.xclbin"))
                .unwrap_err();
            assert!(
                matches!(err, XdnaError::NotFound(_)),
                "expected NotFound, got {err:?}",
            );
        }

        #[cfg(feature = "real-xrt")]
        {
            // On a real box without the NPU device, open() itself fails
            // with Device. That's an acceptable path-sanity result — we
            // never panicked.
            let mut dev = match XdnaDevice::open(0) {
                Ok(d) => d,
                Err(XdnaError::Device) => {
                    eprintln!("no XDNA device available; skipping");
                    return;
                }
                Err(e) => panic!("unexpected error: {e:?}"),
            };
            let err = dev
                .load_xclbin(Path::new("/tmp/halo-xdna-nonexistent-xclbin-probe.xclbin"))
                .unwrap_err();
            assert!(
                matches!(err, XdnaError::NotFound(_)),
                "expected NotFound, got {err:?}",
            );
        }
    }

    /// Debug impl must not leak internal pointer values but should be
    /// stable enough for tracing-style logging.
    #[test]
    #[cfg(not(feature = "real-xrt"))]
    fn debug_format_is_stable_in_stub() {
        let inner = stub::StubDeviceInner {
            _bdf_idx: 0,
            xclbin_path: None,
        };
        let dev = XdnaDevice { inner };
        let s = format!("{dev:?}");
        assert!(s.contains("XdnaDevice"), "unexpected debug output: {s}");
        assert!(s.contains("stub"), "expected stub marker: {s}");
    }
}
