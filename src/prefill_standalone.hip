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

// Phase 2: LDS-tiled GEMM.
//   Block tile: BLOCK_M x BLOCK_N = 16 x 16 output elements
//   K tile     : K1 = 32 per step (matches the block-permute K1)
//   Threads    : 256 per block (16 x 16, one output per thread)
//   LDS        : A_tile [16, 32] FP16 = 1 KiB
//                B_tile [32, 16] FP16 = 1 KiB  (decoded on-the-fly at load)
// Each outer K step corresponds exactly to one "j" block in CK's K0 iteration.
// The within-8 nibble permute is unwound during the LDS load so the LDS-side
// B_tile[kk, n] is naturally K-contiguous for the accumulator loop.
__global__ __launch_bounds__(256)
void standalone_ternary_gemm_lds(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    constexpr int TILE_K = 32;   // == K1

    __shared__ __half A_lds[TILE_M * TILE_K];   // [TILE_M, TILE_K] row-major
    __shared__ __half B_lds[TILE_K * TILE_N];   // [TILE_K, TILE_N] row-major

    const int tx = threadIdx.x;       // 0..TILE_N-1
    const int ty = threadIdx.y;       // 0..TILE_M-1
    const int tid = ty * TILE_N + tx; // 0..255

    const int m0 = blockIdx.y * TILE_M;   // block's row origin
    const int n0 = blockIdx.x * TILE_N;   // block's col origin
    const int m  = m0 + ty;
    const int n  = n0 + tx;

    float acc = 0.0f;
    const int K0 = K / TILE_K;

    // 512 LDS elements per side, 256 threads -> 2 elements per thread per side.
    // Strided cooperative load keeps coalescing-friendly lanes.
    for(int j = 0; j < K0; ++j) {
        // --- Load A tile: [TILE_M, TILE_K] = 16 x 32 ---
        // Covers A[m0..m0+16, j*32..j*32+32]. Load 2 elements per thread.
        {
            const int base = m0 * K + j * TILE_K;
            // Layout: row r in 0..15, col c in 0..31 -> index r*32 + c
            const int slot0 = tid;             // 0..255
            const int slot1 = tid + 256;       // 256..511
            const int r0 = slot0 / TILE_K;
            const int c0 = slot0 % TILE_K;
            const int r1 = slot1 / TILE_K;
            const int c1 = slot1 % TILE_K;
            A_lds[slot0] = ((m0 + r0) < M)
                ? A[(m0 + r0) * K + j * TILE_K + c0] : (__half)0;
            A_lds[slot1] = ((m0 + r1) < M)
                ? A[(m0 + r1) * K + j * TILE_K + c1] : (__half)0;
        }

        // --- Load B tile: [TILE_K, TILE_N] = 32 x 16 (decoded FP16) ---
        // B on-device layout for block j, columns [n0..n0+16]:
        //   byte base = j * N * K1_BYTES + col * K1_BYTES, K1_BYTES = 16
        // Each column brings 32 nibbles = 16 bytes.
        // 16 cols x 16 bytes = 256 bytes — perfect for 256 threads (1 byte each).
        {
            const int col_in_tile = tid / K1_BYTES;    // 0..15 -> which of TILE_N cols
            const int byte_in_col = tid % K1_BYTES;    // 0..15 -> which of 16 bytes
            const int global_col  = n0 + col_in_tile;
            if(global_col < N) {
                const int byte_idx = j * N * K1_BYTES + global_col * K1_BYTES + byte_in_col;
                const uint8_t b    = B[byte_idx];
                // Byte holds two nibbles at perm_jj = byte_in_col*2 (HIGH) and *2+1 (LOW).
                // Reverse the within-8 permute to map perm_jj back to original jj,
                // which is the K-axis index (= kk in the LDS layout).
                //   forward_perm[s] = input slot for output slot s — {2,0,6,4,3,1,7,5}
                const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};
                #pragma unroll
                for(int nibble = 0; nibble < 2; ++nibble) {
                    const int perm_jj = byte_in_col * 2 + nibble;  // 0..31
                    const int group_base = (perm_jj / 8) * 8;
                    const int slot_out   = perm_jj & 7;
                    const int orig_slot  = forward_perm[slot_out];
                    const int kk         = group_base + orig_slot;  // original K within block
                    const uint8_t nib = (nibble == 0)
                        ? ((b >> 4) & 0xF)   // HIGH nibble = first element in byte
                        : (b & 0xF);
                    const float v = (float)((int)nib - 8);
                    B_lds[kk * TILE_N + col_in_tile] = __float2half(v);
                }
            } else {
                #pragma unroll
                for(int nibble = 0; nibble < 2; ++nibble) {
                    const int perm_jj = byte_in_col * 2 + nibble;
                    const int group_base = (perm_jj / 8) * 8;
                    const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};
                    const int kk = group_base + forward_perm[perm_jj & 7];
                    B_lds[kk * TILE_N + col_in_tile] = (__half)0;
                }
            }
        }

        __syncthreads();

        // --- Accumulate TILE_K FMAs from LDS ---
        if(m < M && n < N) {
            #pragma unroll
            for(int kk = 0; kk < TILE_K; ++kk) {
                const float a = (float)A_lds[ty * TILE_K + kk];
                const float b = (float)B_lds[kk * TILE_N + tx];
                acc += a * b;
            }
        }

        __syncthreads();
    }

    if(m < M && n < N) {
        C[m * N + n] = __float2half(acc);
    }
}

// Phase 3: WMMA-backed GEMM on gfx1151.
//   One wave (32 threads) = one block = one 16x16 output tile.
//   K tile = 16 per WMMA step.
//   Accumulator: FP32 (f32-accum variant avoids the OpSel dance).
//
// RDNA 3 Wave32 WMMA fragment layout (documented in AMD ISA):
//   A[16, 16] FP16: each lane holds half16_t. Lanes 0..15 hold row (lane_id)
//     of A (all 16 cols). Lanes 16..31 duplicate the same data.
//   B[16, 16] FP16: each lane holds half16_t. Lanes 0..15 hold col (lane_id)
//     of B (all 16 rows). Lanes 16..31 duplicate.
//   C[16, 16] FP32: each lane holds float8_t = 8 outputs.
//     lane i, slot r in 0..7 -> C[2*r + (i >= 16), i % 16]
//     (interleaved rows: lanes 0..15 own even rows, lanes 16..31 own odd rows.)
//
// Builtin:
//   float8_t __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(half16_t A, half16_t B, float8_t C);
using half16_t  = __attribute__((ext_vector_type(16))) __fp16;
using float8_t  = __attribute__((ext_vector_type(8))) float;

__global__ __launch_bounds__(32)
void standalone_ternary_gemm_wmma(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    constexpr int WMMA_K = 16;  // WMMA K step; also K1/2

    const int lane = threadIdx.x;      // 0..31
    const int row_lane = lane & 15;    // 0..15 — row index for A / col index for B
    const int half = lane >> 4;        // 0 or 1 — duplicate half

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    // Accumulator
    float8_t acc = {};

    const int K0 = K / K1;             // number of j blocks
    const int steps_per_block = K1 / WMMA_K;  // 2 WMMA steps per K1-block
    const int jj_per_step     = WMMA_K;       // 16 original K per step

    const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};

    for(int j = 0; j < K0; ++j) {
        // Each K1 block (32 elements along K axis) = 2 WMMA steps of 16.
        for(int step = 0; step < steps_per_block; ++step) {
            const int jj_lo = step * WMMA_K;            // 0 or 16
            // Original K range for this step: [j*K1 + jj_lo, j*K1 + jj_lo + 16)

            // ------- Build A fragment -------
            // Each of 32 lanes holds row (lane % 16) of A, all 16 cols for
            // this K step. Lanes 16..31 duplicate lanes 0..15.
            half16_t A_frag;
            {
                const int m_row = m0 + row_lane;
                if(m_row < M) {
                    #pragma unroll
                    for(int c = 0; c < 16; ++c) {
                        const int k_orig = j * K1 + jj_lo + c;
                        A_frag[c] = (__fp16)(float)A[m_row * K + k_orig];
                    }
                } else {
                    #pragma unroll
                    for(int c = 0; c < 16; ++c) A_frag[c] = (__fp16)0;
                }
            }

            // ------- Build B fragment -------
            // Each of 32 lanes holds col (lane % 16) of B for this K step.
            // B on device: for block j, col (n0 + row_lane), 16 bytes per col.
            half16_t B_frag;
            {
                const int n_col = n0 + row_lane;
                if(n_col < N) {
                    const int b_base = j * N * K1_BYTES + n_col * K1_BYTES;
                    // We need the 16 original-K values [jj_lo, jj_lo+16) from the
                    // permuted bytes. Reverse the within-8 permute.
                    #pragma unroll
                    for(int r = 0; r < 16; ++r) {
                        const int jj_orig  = jj_lo + r;                     // 0..31
                        const int grp_base = (jj_orig / 8) * 8;
                        const int slot_out = jj_orig & 7;
                        const int orig_s   = forward_perm[slot_out];        // WRONG direction; see below
                        (void)orig_s;
                        // B_lds we built in Phase 2 used forward_perm (output->input). Here
                        // we want: given desired ORIGINAL position jj_orig, where is it in
                        // the permuted byte buffer?
                        const int inv_perm[8] = {1, 5, 0, 4, 3, 7, 2, 6};
                        const int perm_slot = inv_perm[slot_out];
                        const int perm_jj   = grp_base + perm_slot;         // position in permuted bytes
                        const uint8_t byte  = B[b_base + (perm_jj >> 1)];
                        const uint8_t nib   = ((perm_jj & 1) == 0)
                            ? ((byte >> 4) & 0xF)
                            : (byte & 0xF);
                        B_frag[r] = (__fp16)((float)((int)nib - 8));
                    }
                } else {
                    #pragma unroll
                    for(int r = 0; r < 16; ++r) B_frag[r] = (__fp16)0;
                }
            }

            // ------- WMMA step -------
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A_frag, B_frag, acc);
        }
    }

    // ------- Store C -------
    // lane i slot r in 0..7 -> C[r + 8*(i >= 16), i % 16]
    #pragma unroll
    for(int r = 0; r < 8; ++r) {
        const int m_out = m0 + 2 * r + half;
        const int n_out = n0 + row_lane;
        if(m_out < M && n_out < N) {
            C[m_out * N + n_out] = __float2half(acc[r]);
        }
    }
}

// Phase 4: 2 waves per block (64 threads) computing a 32x16 output tile.
// LDS staging for A and B — both waves share the K loads:
//   A_lds : [32, 32] FP16 = 2 KiB  (two 16-row halves, one per wave)
//   B_lds : [32, 16] FP16 = 1 KiB  (shared by both waves)
// Each wave then loads its fragments from LDS and does 2 WMMAs per K1 step.
__global__ __launch_bounds__(64)
void standalone_ternary_gemm_wmma_2wave(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 32;   // 2 waves x 16 rows
    constexpr int TILE_N = 16;
    constexpr int WMMA_K = 16;

    __shared__ __half A_lds[TILE_M * K1];   // [32, 32] FP16 row-major
    __shared__ __half B_lds[K1 * TILE_N];   // [32, 16] FP16 row-major

    const int tid  = threadIdx.x;       // 0..63
    const int wave = tid >> 5;          // 0 or 1
    const int lane = tid & 31;          // 0..31
    const int row_lane = lane & 15;
    const int half     = lane >> 4;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};
    const int K0 = K / K1;

    // Two float8 accumulators per wave, one per WMMA step along K1.
    // Waves both target their own 16x16 output rows; N is shared.
    float8_t acc = {};

    for(int j = 0; j < K0; ++j) {
        // ---- Cooperatively load A tile [32, 32] into LDS ----
        // 32*32 = 1024 halves, 64 threads -> 16 halves per thread.
        // Use a simple row-strided pattern for coalescing.
        #pragma unroll
        for(int e = 0; e < 16; ++e) {
            const int slot = tid * 16 + e;   // 0..1023
            const int r = slot / 32;         // 0..31
            const int c = slot % 32;         // 0..31
            const int m_row = m0 + r;
            A_lds[slot] = (m_row < M)
                ? A[m_row * K + j * K1 + c]
                : (__half)0;
        }

        // ---- Cooperatively load+decode B tile [32, 16] into LDS ----
        // 32*16 = 512 halves, 64 threads -> 8 halves per thread.
        // The byte layout supplies 16 cols x 16 bytes per j-block (256 bytes).
        // With 64 threads, each loads 4 bytes / 8 nibbles.
        {
            const int bytes_per_col = K1_BYTES;               // 16
            const int bytes_total   = TILE_N * bytes_per_col; // 256
            const int my_byte0 = tid * 4;                     // 0, 4, 8, ..., 252
            if(my_byte0 < bytes_total) {
                const int col_start_byte = my_byte0 % bytes_per_col;
                const int col_idx        = my_byte0 / bytes_per_col;
                (void)col_start_byte;
                (void)col_idx;
                #pragma unroll
                for(int bb = 0; bb < 4; ++bb) {
                    const int byte_off   = my_byte0 + bb;
                    const int col        = byte_off / bytes_per_col;
                    const int byte_in_col = byte_off % bytes_per_col;
                    const int n_col      = n0 + col;
                    if(n_col < N) {
                        const int b_idx = j * N * bytes_per_col + n_col * bytes_per_col + byte_in_col;
                        const uint8_t bb_val = B[b_idx];
                        #pragma unroll
                        for(int nibble = 0; nibble < 2; ++nibble) {
                            const int perm_jj    = byte_in_col * 2 + nibble;
                            const int grp_base   = (perm_jj / 8) * 8;
                            const int orig_slot  = forward_perm[perm_jj & 7];
                            const int kk         = grp_base + orig_slot;     // original K (0..31)
                            const uint8_t nib = (nibble == 0)
                                ? ((bb_val >> 4) & 0xF)
                                : (bb_val & 0xF);
                            B_lds[kk * TILE_N + col] = __float2half((float)((int)nib - 8));
                        }
                    } else {
                        // zero-fill guard for out-of-range cols handled on read
                    }
                }
            }
        }

        __syncthreads();

        // ---- 2 WMMAs per j block (K1 = 32 = 2 * WMMA_K) ----
        #pragma unroll
        for(int step = 0; step < K1 / WMMA_K; ++step) {
            const int kk0 = step * WMMA_K;

            half16_t A_frag;
            half16_t B_frag;

            // A: this wave's 16 rows [wave*16 .. wave*16+15] of A_lds,
            //    16 K cols [kk0 .. kk0+15].
            const int my_a_row = wave * 16 + row_lane;  // 0..31 in A_lds
            #pragma unroll
            for(int c = 0; c < 16; ++c) {
                A_frag[c] = (__fp16)(float)A_lds[my_a_row * K1 + kk0 + c];
            }

            // B: 16 K rows [kk0 .. kk0+15], col row_lane.
            #pragma unroll
            for(int r = 0; r < 16; ++r) {
                B_frag[r] = (__fp16)(float)B_lds[(kk0 + r) * TILE_N + row_lane];
            }

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A_frag, B_frag, acc);
        }

        __syncthreads();
    }

    // Store: each wave owns rows [wave*16 .. wave*16+15], interleaved within
    //        the 16x16 subtile.
    const int wave_m0 = m0 + wave * 16;
    #pragma unroll
    for(int r = 0; r < 8; ++r) {
        const int m_out = wave_m0 + 2 * r + half;
        const int n_out = n0 + row_lane;
        if(m_out < M && n_out < N) {
            C[m_out * N + n_out] = __float2half(acc[r]);
        }
    }
}

}  // namespace

// Exposed via an internal extern "C" entry; the C API (ck_gemm.cpp) will add a
// backend-switch once this kernel is correct. For Phase 1 we expose a direct
// launcher so tests/test_standalone.cpp can call it without modifying the C API.
extern "C" {

// Phase 1 naive launcher — kept as the correctness reference.
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

// Phase 2 LDS-tiled launcher.
void rcpp_standalone_launch_lds(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(16, 16, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_lds,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 3 WMMA launcher — one wave per 16x16 output tile.
void rcpp_standalone_launch_wmma(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_wmma,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 4 WMMA + LDS + 2 waves per block (32x16 output tile).
void rcpp_standalone_launch_wmma_2wave(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(64, 1, 1);
    dim3 grid((N + 15) / 16, (M + 31) / 32, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_wmma_2wave,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

}  // extern "C"
