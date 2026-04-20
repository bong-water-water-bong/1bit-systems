//! build.rs — compile the C++ shim and link against libwhisper, but **only**
//! when the `real-whisper` feature is active.
//!
//! With the default `stub` feature we emit zero link directives: the crate
//! is pure-Rust and compiles on any host without whisper.cpp installed.
//!
//! When `real-whisper` is on we:
//!   1. Compile `cpp/shim.cpp` into a static library `libonebit_whisper_shim.a`
//!      via the `cc` crate. The shim exposes the C-linkage `onebit_whisper_*`
//!      surface declared in `cpp/shim.h`, on top of `whisper.h`.
//!   2. Emit `cargo:rustc-link-lib=whisper` so the final binary resolves the
//!      `whisper_*` symbols our shim calls into. The system whisper.cpp 1.8.x
//!      install on this box provides `/usr/lib/libwhisper.so.1.8.3` plus
//!      `/usr/include/whisper.h`.
//!
//! There is no header-search-path override: we assume whisper.h is on the
//! default compiler include path (`/usr/include`). If that breaks on a host,
//! set `CXXFLAGS=-I/path/to/whisper/include` before `cargo build`.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-changed=cpp/shim.h");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_REAL_WHISPER");

    let real_whisper = env::var_os("CARGO_FEATURE_REAL_WHISPER").is_some();
    if !real_whisper {
        // Stub mode: nothing to compile, nothing to link.
        return;
    }

    // 1. Build the C-linkage shim over whisper.h.
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/shim.cpp")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .flag_if_supported("-Wno-unused-parameter")
        .compile("onebit_whisper_shim");

    // 2. Link against system libwhisper. /usr/lib is on the default linker
    //    search path, so no extra `-L` is needed; if packaging moves it we
    //    can add `cargo:rustc-link-search=native=/path`.
    println!("cargo:rustc-link-lib=whisper");
}
