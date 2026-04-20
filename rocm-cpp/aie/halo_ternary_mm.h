// halo_ternary_mm.h — ternary (int2) x int8 GEMM core for AIE2P (Strix Halo XDNA2).
//
// This header is consumed by Peano-compiled tile code (halo_ternary_mm.cpp)
// AND by any host-side build that needs to know the core entry signature for
// an MLIR wrapper or a parity reference. Keep it free of AIE-API types so it
// can be included from either side.
//
// Compile site: Peano clang, target aie2p-none-unknown-elf.
//   /opt/peano/bin/clang --target=aie2p-none-unknown-elf -O2 -c \
//       halo_ternary_mm.cpp -o halo_ternary_mm.o
//
// Source refs:
//   docs/wiki/NPU-Kernel-Design.md          — tile grid, DMA pattern, L1 budget
//   docs/wiki/Ternary-on-AIE-Pack-Plan.md   — 2-bit encoding, unpack fn shape

#ifndef HALO_TERNARY_MM_H
#define HALO_TERNARY_MM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tile-core entry, signature matches IRON's per-core wrapper in
// aie_kernels/aie2p/mm.cc:396-410 modulo the A-path type.
//
//   packed_A : ternary weights, 2 bits/elem, 4 elems/byte, packed into
//              uint64_t words (32 ternary weights per word). Pre-tiled by the
//              host requantizer into the 4D (m/r, k/s, r, s) order that the
//              mem-tile L2->L1 BD expects. See NPU-Kernel-Design.md §DMA.
//   B        : int8 activations, row-major, s-by-t tiles stored in (k/s, n/t,
//              s, t) order per IRON design.py:399.
//   C        : int32 accumulator out, row-major r-by-t tiles in
//              (m/r, n/t, r, t) order (c_row_maj=true).
//   M, N, K  : logical dims in elements. Must satisfy M%16==0, N%16==0,
//              K%8==0 (static_assert in the 2x2 mmul unroll, mm.cc:372-374).
//
// This is the hot per-core kernel. The host-side xclbin is responsible for
// L3->L2->L1 ObjectFifo wiring and runtime ND-DMA descriptor dispatch; this
// function assumes its three buffers are already in L1 ping-pong scratch.
void halo_ternary_mm_core(const uint64_t* packed_A,
                          const int8_t*   B,
                          int32_t*        C,
                          int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif // HALO_TERNARY_MM_H
