//! build.rs — propagate halo-bitnet-hip's rpath into router integration
//! test binaries. See crates/halo-server/build.rs for the long version.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DEP_ROCM_CPP_ROCM_CPP_LIB_DIR");

    let Some(lib_dir) = env::var_os("DEP_ROCM_CPP_ROCM_CPP_LIB_DIR") else {
        return;
    };
    let lib_dir = lib_dir.to_string_lossy().into_owned();
    emit_rpath(&lib_dir);
    emit_rpath("/opt/rocm/lib");
}

fn emit_rpath(dir: &str) {
    let arg = format!("-Wl,-rpath,{dir}");
    println!("cargo:rustc-link-arg-tests={arg}");
}
