// Standalone FP16 x pk_i4(ternary) -> FP16 GEMM for gfx1151.
// No CK headers, no CK symbols. Correctness-first; performance arrives in
// Phases 2-4.
//
// Layout (matches CK path exactly):
//   A      : FP16 row-major [M, K]   stride K
//   B_pack : pk_i4 bytes, WMMA-permuted for the same CK-style layout:
//              block-reshape K -> [K0, N, K1]  with K1 = 32
//              within-8 nibble permute (01234567 -> 20643175)
//              HIGH nibble of each byte holds the first (lower-index) element
//              (matches CK_USE_PK4_LAYOUT_SHUFFLE = 1 in ck.hpp)
//   C      : FP16 row-major [M, N]   stride N
//
// Nibble-to-FP16 decode matches CK: value = nibble - 8 (so 0x7->-1, 0x8->0, 0x9->+1)
//
// This TU intentionally has zero ck/ includes. Link: only HIP runtime.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#include "rocm_cpp/ck_gemm.h"   // C API — shared between CK and standalone backends

namespace {

constexpr int K1       = 32;  // KPerBlock in CK parlance
constexpr int K1_BYTES = K1 / 2;

// Forward permute (output slot -> input slot), per the CK upstream pk_i4 example:
//   slot 0 <- input 2   slot 4 <- input 3
//   slot 1 <- input 0   slot 5 <- input 1
//   slot 2 <- input 6   slot 6 <- input 7
//   slot 3 <- input 4   slot 7 <- input 5
// Inverse (input slot -> output slot) = {1, 5, 0, 4, 3, 7, 2, 6}.
__device__ __constant__ int kInversePerm[8] = {1, 5, 0, 4, 3, 7, 2, 6};

// Naive GEMM: one thread per output element. A loop of K inside each thread.
// BLOCK_M x BLOCK_N = 16 x 16 output tile per block, 256 threads per block.
__global__ __launch_bounds__(256)
void standalone_ternary_gemm_naive(
    const __half* __restrict__ A,       // [M, K] row-major
    const uint8_t* __restrict__ B,      // [K*N/2] WMMA-permuted pk_i4
    __half* __restrict__ C,             // [M, N] row-major
    int M, int N, int K)
{
    const int m = blockIdx.y * 16 + threadIdx.y;
    const int n = blockIdx.x * 16 + threadIdx.x;
    if(m >= M || n >= N) return;

    const int K0 = K / K1;
    float acc = 0.0f;

    for(int j = 0; j < K0; ++j) {
        // Base offset into B for (block j, column n): j*N*K1/2 + n*K1/2 bytes.
        const int b_base = j * N * K1_BYTES + n * K1_BYTES;

        for(int jj = 0; jj < K1; ++jj) {
            const int k_orig = j * K1 + jj;
            const int group_base = (jj / 8) * 8;
            const int slot_in    = jj & 7;
            const int perm_slot  = kInversePerm[slot_in];
            const int perm_jj    = group_base + perm_slot;

            // Byte inside this (j, n) block of K1/2 bytes, plus nibble select.
            const int byte_idx  = b_base + (perm_jj >> 1);
            const uint8_t byte  = B[byte_idx];
            const uint8_t nib   = ((perm_jj & 1) == 0)   // even perm_jj -> HIGH nibble
                                    ? ((byte >> 4) & 0xF)
                                    : (byte & 0xF);
            const float bval = (float)((int)nib - 8);   // -8 bias

            acc += (float)A[m * K + k_orig] * bval;
        }
    }

    C[m * N + n] = __float2half(acc);
}

}  // namespace

// Exposed via an internal extern "C" entry; the C API (ck_gemm.cpp) will add a
// backend-switch once this kernel is correct. For Phase 1 we expose a direct
// launcher so tests/test_standalone.cpp can call it without modifying the C API.
extern "C" {

void rcpp_standalone_launch(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(16, 16, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_naive,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

}  // extern "C"
