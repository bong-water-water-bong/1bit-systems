// aie_hello.cpp — minimal "hello tile" kernel for AIE2P (Strix Halo XDNA2).
//
// Build:
//   /opt/peano/bin/clang --target=aie2p-none-unknown-elf \
//       -O2 -c aie_hello.cpp -o aie_hello.o
//
// This is scalar C++ only — no AIE intrinsics, no AIE-API. Just enough to
// prove the Peano toolchain can lower C++ to an AIE2P object file.
// Running it requires IRON / xrt_coreutil / a full MLIR-AIE host stack,
// which is explicitly out of scope for this checkpoint.

extern "C" int mac_tile(int a, int b, int c) {
    // One multiply-accumulate. The AIE tile has a native MAC unit; clang
    // should pick it during instruction selection with -O2.
    return a * b + c;
}
