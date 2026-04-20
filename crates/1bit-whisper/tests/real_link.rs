//! real_link.rs — compile-time smoke test for the `real-whisper` feature.
//!
//! When built with `--features real-whisper --no-default-features` this
//! test references the FFI entry points just enough to require the
//! linker to resolve them against libwhisper. We don't actually call
//! the functions (that would need a real model file on disk), just name
//! them so the compile + link steps must succeed.
//!
//! Under the default `stub` feature this test compiles to a no-op.

#[cfg(feature = "real-whisper")]
#[test]
fn real_whisper_ffi_symbols_link() {
    use onebit_whisper::ffi;

    // Take function pointers to prove the symbols were resolved at link
    // time. Casting to `usize` keeps the optimizer from stripping them.
    let a = ffi::onebit_whisper_init as *const () as usize;
    let b = ffi::onebit_whisper_free as *const () as usize;
    let c = ffi::onebit_whisper_feed as *const () as usize;
    let d = ffi::onebit_whisper_drain as *const () as usize;
    assert!(a != 0 && b != 0 && c != 0 && d != 0);
}

#[cfg(not(feature = "real-whisper"))]
#[test]
fn real_whisper_feature_off_is_noop() {
    // Nothing to verify on the stub path; this test exists so `cargo
    // test` reports a stable test count across feature configurations.
}
