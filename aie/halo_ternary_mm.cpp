// halo_ternary_mm.cpp — ternary (int2) x int8 GEMM, AIE2P tile core.
//
// Compiled by Peano clang (target aie2p-none-unknown-elf). This file is
// deliberately scalar-first: the reference path below is the parity oracle
// for the vectorized aie::mmul path we will light up once the AIE-API
// headers (ship with mlir-aie) are on the include search path.
//
// Honest scope note: the scalar path compiles today with -O2 and an
// AIE2P-targeted clang. The vectorized path is stubbed behind HAVE_AIE_API.
//
// DMA descriptor requirements (host-side xclbin territory, NOT this file):
//   * A-path 4D reorder: (m/r, r*k), (k/s, s), (r, k), (s, 1)
//     baked into the requantizer so shim->mem BD can stream it flat.
//     Ref: iron/operators/gemm/design.py:368-375.
//   * B-path 4D reorder: (k/s, s*n), (n/t, t), (s, n), (t, 1)
//     row-major activations. Ref: design.py:397-399.
//   * C-drain inverse reorder: (m/r, r*n), (r, t), (n/t, r*t), (t, 1)
//     joins 4 r-by-t fragments back to one 64x64 slab. Ref: design.py:413-415.
//   * Alternating shim placement on 8-col NPU2: Tile(2*i, 1), not Tile(i, 1).
//     Ref: design.py:385.
// None of the above is this file's job. Host MLIR owns it.

#include "halo_ternary_mm.h"

namespace {

// Tile geometry. r=s=t=8, 2x2 unroll -> 16x16 int32 C per core step.
// Matches IRON's matmul_vectorized_8x8x8_i8_i32 (mm.cc:396-410).
constexpr int kR = 8;
constexpr int kS = 8;
constexpr int kT = 8;

// Unpack 32 ternary weights (2 bits each) from one uint64_t into 32 int8s.
// Encoding (docs/wiki/Ternary-on-AIE-Pack-Plan.md):
//   0b00 -> 0, 0b01 -> +1, 0b10 -> -1, 0b11 -> reserved (treated as 0).
// Keep this inline so the vectorized path can copy-paste it into an
// aie::vector<int8, 32> producer.
inline void unpack_ternary_u64(uint64_t packed, int8_t out[32]) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < 32; ++i) {
        const uint8_t b = static_cast<uint8_t>((packed >> (2 * i)) & 0x3u);
        // Branchless: map 00/01/10/11 -> 0/+1/-1/0.
        // (b == 1) - (b == 2) covers the three live cases; 0b11 falls to 0.
        out[i] = static_cast<int8_t>((b == 1) ? 1 : (b == 2) ? -1 : 0);
    }
}

} // namespace

extern "C" void halo_ternary_mm_core(const uint64_t* packed_A,
                                     const int8_t*   B,
                                     int32_t*        C,
                                     int M, int N, int K) {
    // Scalar reference. Correct but slow. Use as the bit-exact oracle when
    // the vectorized path lands. This walks the logical (M,K)x(K,N) matmul
    // as if A were already unpacked to int8 row-major; the unpack fills a
    // 32-elem scratch window into packed_A on demand.
    //
    // NOTE on layout: this scalar path treats packed_A as a flat bit-stream
    // of M*K ternary weights row-major. The production path will ingest the
    // 4D-reordered bytes described above and mirror IRON's 2x2 mmul tiling.
    // Keeping the scalar oracle flat makes it trivial to diff against the
    // HIP reference in rocm-cpp/kernels/ternary_gemv_halo.hip.

    const int64_t a_total = static_cast<int64_t>(M) * K;   // ternary elems
    int8_t a_scratch[32];
    int64_t cached_word = -1;  // invalid sentinel

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                const int64_t a_idx   = static_cast<int64_t>(m) * K + k;
                const int64_t word_ix = a_idx >> 5;           // /32
                const int     lane    = static_cast<int>(a_idx & 31);
                if (word_ix != cached_word) {
                    // Bounds-safe: caller guarantees M*K fits in the mmap'd
                    // weight blob; we still avoid OOB by clamping.
                    if (a_idx < a_total) {
                        unpack_ternary_u64(packed_A[word_ix], a_scratch);
                        cached_word = word_ix;
                    }
                }
                const int8_t  a = a_scratch[lane];
                const int8_t  b = B[static_cast<int64_t>(k) * N + n];
                acc += static_cast<int32_t>(a) * static_cast<int32_t>(b);
            }
            C[static_cast<int64_t>(m) * N + n] = acc;
        }
    }
}

// -----------------------------------------------------------------------------
// Vectorized path (stub). Enable once mlir-aie's aie_api/aie.hpp is on the
// Peano include path. Intentionally NOT the default so this TU builds today
// with Peano alone.
//
// TODO(halo-ternary-mm): port matmul_vectorized_2x2_mmul from
//   /tmp/IRON-read/aie_kernels/aie2p/mm.cc:83-208 with A-load replaced by
//   unpack_ternary_u64 feeding an aie::vector<int8, 32>. Four mmul<8,8,8>
//   accumulators (C00/C01/C10/C11) carry a 16x16 int32 fragment across the
//   K reduction. VPERM (unpack) issues on a separate VLIW slot from MUL/MAC;
//   ping-pong an i8 B' scratch to keep MAC fed.
//
// #ifdef HAVE_AIE_API
// #include <aie_api/aie.hpp>
// template <unsigned rowA, unsigned colA, unsigned colB>
// static inline void halo_ternary_mm_vec(const uint64_t* __restrict pA_packed,
//                                        const int8_t*   __restrict pB,
//                                        int32_t*        __restrict pC) {
//     using MMUL = aie::mmul<kR, kS, kT, int8, int8, aie::accauto>;
//     // ... 2x2 unroll, unpack_ternary_u64 -> aie::vector<int8, 32> per A load ...
//     // ... mirror mm.cc:147-177 ...
// }
// #endif  // HAVE_AIE_API
