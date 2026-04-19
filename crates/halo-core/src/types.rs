//! Shared small types used across the crate.

/// Token id. The C++ side uses `int32_t` for ids; we match that so the FFI
/// boundary to `halo-bitnet-hip` is a trivial transmute when needed.
pub type TokenId = i32;

/// Minimum `.h1b` format version this crate can parse.
pub const MIN_SUPPORTED_VERSION: i32 = 1;

/// Maximum `.h1b` format version this crate can parse.
///
/// v1 = original halo-1bit export (2 bpw, no rope/eps in header).
/// v2 = same payload as v1 but with `rope_theta` + `rms_norm_eps` in header.
/// v3 = Sherry 1.25 bpw ternary packing (`cols % 32 == 0`).
/// v4 = TQ1 base-3 packing, 1.6 bpw, lossless (`cols` padded to mult. of 20).
pub const MAX_SUPPORTED_VERSION: i32 = 4;

/// BitNet defaults for v1 files that don't carry explicit values.
/// Matches the fallbacks in `rocm-cpp/src/h1b_loader.cpp`.
pub const DEFAULT_ROPE_THETA: f32 = 500_000.0;
pub const DEFAULT_RMS_NORM_EPS: f32 = 1e-5;
