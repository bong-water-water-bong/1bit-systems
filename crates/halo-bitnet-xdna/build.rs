//! build.rs — wire `halo-bitnet-xdna` to XRT (libxrt_coreutil + libxrt_core)
//! and compile the thin C++ shim in `cpp/shim.cpp` that wraps the XRT C++
//! API into a C-linkage surface callable from Rust.
//!
//! # Feature gating
//!
//! - `stub` (default): no XRT on the build path. We emit no link directives
//!   and do NOT invoke the cc compiler. The crate builds against the
//!   no-op paths in `src/stub.rs`. This keeps CI on non-NPU boxes green.
//! - `real-xrt`: compile `cpp/shim.cpp` against `/usr/include/xrt/*`,
//!   link `libxrt_coreutil` + `libxrt_core`. Requires XRT ≥ 2.21.x.
//!
//! We deliberately avoid bindgen on XRT's C++ headers — they pull in
//! std::filesystem, std::shared_ptr, and a lot of template machinery that
//! bindgen cannot translate cleanly. The C shim approach keeps the unsafe
//! surface ~10 symbols wide.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=XRT_ROOT");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-changed=cpp/shim.h");

    let real_xrt = env::var_os("CARGO_FEATURE_REAL_XRT").is_some();
    if !real_xrt {
        println!(
            "cargo:warning=halo-bitnet-xdna built without `real-xrt` — \
             every NPU call will return XdnaError::UnsupportedStub."
        );
        return;
    }

    // 1. Locate XRT headers. Arch pkg installs them at /usr/include/xrt.
    let xrt_include = find_xrt_include().expect(
        "halo-bitnet-xdna: real-xrt feature enabled but XRT headers not found. \
         Install XRT (Arch: pacman -S xrt) or set XRT_ROOT=/path/to/xrt.",
    );
    // 2. Locate XRT libs. Arch pkg installs them at /usr/lib.
    let xrt_libdir = find_xrt_libdir().expect(
        "halo-bitnet-xdna: real-xrt feature enabled but libxrt_coreutil.so \
         not found. Install XRT (Arch: pacman -S xrt) or set XRT_ROOT.",
    );

    // 3. Compile the C++ shim against XRT headers. -std=c++17 is what XRT's
    //    own headers expect (they use std::filesystem, std::optional).
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/shim.cpp")
        .include(&xrt_include)
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("halo_xrt_shim");

    // 4. Link XRT. Both libs have unversioned .so symlinks on a standard
    //    install; the Rust linker resolves via `-lxrt_coreutil -lxrt_core`.
    println!("cargo:rustc-link-search=native={}", xrt_libdir.display());
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    println!("cargo:rustc-link-lib=dylib=xrt_core");
    // XRT's uuid handling calls libuuid's uuid_clear/uuid_copy; needed
    // explicitly because XRT's .so exposes these as public ABI.
    println!("cargo:rustc-link-lib=dylib=uuid");
    // libstdc++ is what the C++ shim needs; cc normally handles this but
    // we're explicit so the test binary link works even when cc's default
    // runtime detection misfires.
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // 5. rpath so `cargo test` finds the libs without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", xrt_libdir.display());
}

fn find_xrt_include() -> Option<PathBuf> {
    if let Some(root) = env::var_os("XRT_ROOT") {
        let p = PathBuf::from(root).join("include/xrt");
        if p.join("xrt_device.h").exists() {
            return Some(PathBuf::from(&p).parent().unwrap().to_path_buf());
        }
    }
    for candidate in ["/usr/include", "/opt/xilinx/xrt/include"] {
        let p = PathBuf::from(candidate).join("xrt");
        if p.join("xrt_device.h").exists() {
            return Some(PathBuf::from(candidate));
        }
    }
    None
}

fn find_xrt_libdir() -> Option<PathBuf> {
    if let Some(root) = env::var_os("XRT_ROOT") {
        let p = PathBuf::from(root).join("lib");
        if p.join("libxrt_coreutil.so").exists() {
            return Some(p);
        }
    }
    for candidate in ["/usr/lib", "/usr/lib64", "/opt/xilinx/xrt/lib"] {
        let p = PathBuf::from(candidate);
        if p.join("libxrt_coreutil.so").exists() {
            return Some(p);
        }
    }
    None
}
