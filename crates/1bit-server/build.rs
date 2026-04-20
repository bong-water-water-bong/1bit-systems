//! build.rs — propagate 1bit-hip's rpath to the final binary.
//!
//! `cargo:rustc-link-arg` from a library build script applies only to that
//! rlib's link step (a no-op for rlibs). `rustc-link-arg-bins` / -tests
//! are only honoured from crates that *have* bin / test targets — which is
//! us. We read the resolved rocm-cpp lib dir from 1bit-hip via the
//! `DEP_ROCM_CPP_ROCM_CPP_LIB_DIR` env var (Cargo sets this for direct
//! dependents of a crate with `links = "rocm_cpp"`) and re-emit it here.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DEP_ROCM_CPP_ROCM_CPP_LIB_DIR");

    // Only active when the `real-backend` feature is on — otherwise
    // 1bit-hip isn't pulled in at all and the env var will be
    // absent.
    let Some(lib_dir) = env::var_os("DEP_ROCM_CPP_ROCM_CPP_LIB_DIR") else {
        return;
    };
    let lib_dir = lib_dir.to_string_lossy().into_owned();
    emit_rpath(&lib_dir);
    emit_rpath("/opt/rocm/lib");
}

fn emit_rpath(dir: &str) {
    let arg = format!("-Wl,-rpath,{dir}");
    // `-bins` applies to the `1bit-server` binary target.
    println!("cargo:rustc-link-arg-bins={arg}");
}
