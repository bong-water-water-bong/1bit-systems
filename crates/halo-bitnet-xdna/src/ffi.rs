//! ffi.rs — extern "C" declarations mirroring `cpp/shim.h`.
//!
//! Compiled into the binary ONLY when the `real-xrt` feature is enabled
//! (see `lib.rs`'s cfg-gated `mod ffi;`). Without the feature the crate
//! falls through to `stub.rs` and never tries to resolve these symbols.

use std::ffi::{c_char, c_int};

/// Opaque handle matching the C++ `struct HaloXrtDevice` in `cpp/shim.cpp`.
/// We never dereference this from Rust; the C++ side owns the layout.
#[repr(C)]
pub(crate) struct HaloXrtDevice {
    _private: [u8; 0],
}

// Status codes — must match the `HALO_XRT_*` constants in `cpp/shim.h`.
pub(crate) const HALO_XRT_OK:          i32 = 0;
pub(crate) const HALO_XRT_E_INVALID:   i32 = -1;
pub(crate) const HALO_XRT_E_NOT_FOUND: i32 = -2;
pub(crate) const HALO_XRT_E_DEVICE:    i32 = -3;
pub(crate) const HALO_XRT_E_KERNEL:    i32 = -4;
pub(crate) const HALO_XRT_E_INTERNAL:  i32 = -5;

unsafe extern "C" {
    pub(crate) fn halo_xrt_open(bdf_idx: u32) -> *mut HaloXrtDevice;
    pub(crate) fn halo_xrt_close(dev: *mut HaloXrtDevice);
    pub(crate) fn halo_xrt_load_xclbin(
        dev: *mut HaloXrtDevice,
        path: *const c_char,
    ) -> c_int;
    pub(crate) fn halo_xrt_run_kernel(
        dev: *mut HaloXrtDevice,
        name: *const c_char,
        r#in: *const u8,
        in_len: usize,
        out: *mut u8,
        out_len: usize,
    ) -> c_int;
}
