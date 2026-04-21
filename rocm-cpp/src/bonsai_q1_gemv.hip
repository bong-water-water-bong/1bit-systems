// bonsai_q1_gemv.hip — Bonsai Q1_0_g128 1-bit GEMV for gfx1151 (wave32).
//
// API (see include/rocm_cpp/bonsai.h):
//   bonsai_q1_gemv_launch(packed, act_fp16, out_fp16, N_out, K_in, stream)
//
// Layout (AoS, block-interleaved, 18 bytes per g128 block):
//   [FP16 d : 2][qs[16] : 16 sign bits LSB-first]
//   w[i] = (qs[i/8] >> (i%8)) & 1 ? +d : -d
//
// Thread model (wave32 gfx1151):
//   - One workgroup = 128 threads (4 waves × 32 lanes) per output row.
//   - Each block (128 weights) is consumed in one pass: thread `tid`
//     handles weight `tid` within the block, so the 128 threads cover
//     the block exactly. For K_in > 128 the workgroup iterates over
//     `K_in / 128` blocks sequentially, accumulating into a single fp32
//     register per thread.
//   - Activations for the current block are staged into 128-entry LDS
//     once per block; all 4 waves read from that slab. This is the
//     "LDS-stage activations per-block (256 B), reuse across waves"
//     line in the task spec.
//   - FP32 accumulation. Intra-wave reduction via __shfl_xor butterfly;
//     inter-wave reduction via 4-entry LDS scratch + wave-0 fold. fp16
//     clamp+cast epilogue matches sherry_gemv.hip exactly.
//
// Weight load pattern:
//   Each thread reads its block's FP16 scale `d` (broadcast-friendly —
//   the first 2 bytes of the block). Then each thread reads sign bit
//   `qs[tid/8] >> (tid%8) & 1`. The 32 lanes of a wave read bytes
//   `qs[0..3]` (lanes 0..7 read qs[0], lanes 8..15 read qs[1], etc.)
//   which is a dword-coalesced fetch. Over a block, the whole workgroup
//   touches the full 18-byte row (global memory coalesces the 128-thread
//   load as one 32-byte transaction for qs + one small transaction for
//   d). Cold-cache sizing: 18 bytes per block × (K/128) blocks × N_out
//   rows. Scales broadcast via intra-wave shuffle is NOT a win here
//   because every lane needs `d` anyway — the compiler emits a
//   scalar-load pattern via uniform address propagation.
//
// LDS usage:
//   128 × fp16 (= 256 B) activation slab + 4 × fp32 (= 16 B) wave-reduce
//   scratch = 272 B per workgroup. Negligible (LDS budget on gfx1151 is
//   64 KiB/workgroup); occupancy is VGPR-gated, not LDS-gated.
//
// Bank conflicts:
//   128 fp16 in LDS = 256 B = 4 LDS rows (64 B per row, 32 banks × 4 B).
//   Each wave reads x_lds[0..127] pattern: lane L reads x_lds[wave_base +
//   wave_lane_off + L] where base is uniform across the wave. With 128
//   lanes total across 4 waves, each wave reads 32 consecutive fp16 =
//   64 B = exactly one bank-row. No conflict.
//
// Register pressure:
//   Expected ≤ 24 VGPRs (one fp32 acc, a handful of temporaries, small
//   constants hoisted to SGPRs by the compiler). Target is ≤ 64 to stay
//   above 10 waves/SIMD residency.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

#include "rocm_cpp/bonsai.h"

namespace {

// Block constants — wave32 gfx1151.
constexpr int kBlockSize    = 128;  // threads per workgroup
constexpr int kWaveSize     = 32;
constexpr int kWaves        = kBlockSize / kWaveSize;  // 4
constexpr int kWeightsPerB  = 128;                     // one g128 block
constexpr int kQ1BlockBytes = 18;

// Packed block layout — matches oxibonsai BlockQ1_0G128 byte-for-byte.
// `qs` is 16 bytes (128 sign bits). We never deref this as a struct —
// all loads go through __builtin_memcpy from the raw byte pointer to
// avoid unaligned-access UB on the 18-byte stride.

__device__ __forceinline__
uint32_t load_qs_byte(const uint8_t* __restrict__ qs_ptr, int byte_idx) {
    return (uint32_t)qs_ptr[byte_idx];
}

__device__ __forceinline__
__half load_scale_fp16(const uint8_t* __restrict__ block_ptr) {
    uint16_t u;
    __builtin_memcpy(&u, block_ptr, sizeof(u));
    __half h;
    __builtin_memcpy(&h, &u, sizeof(h));
    return h;
}

}  // namespace

__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
__global__ void bonsai_q1_gemv_kernel(
    const uint8_t* __restrict__ packed,
    const __half*  __restrict__ act,
    __half*        __restrict__ out,
    int N_out, int K_in)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (kWaveSize - 1);
    const int wave = tid >> 5;
    const int row  = blockIdx.x;
    if (row >= N_out) return;

    const int num_blocks  = K_in >> 7;               // K_in / 128
    const size_t row_bytes = (size_t)num_blocks * (size_t)kQ1BlockBytes;
    const uint8_t* __restrict__ wrow = packed + (size_t)row * row_bytes;

    // LDS: activation tile (one g128 block) + wave reduction scratch.
    __shared__ __half x_lds[kWeightsPerB];
    __shared__ float  wave_sums[kWaves];

    float acc = 0.0f;

    // Block-outer loop. Each iteration stages the next 128 activations
    // into LDS and accumulates one block's contribution.
    for (int b = 0; b < num_blocks; ++b) {
        // Coop load 128 fp16 acts. 128 threads × 1 fp16 each = perfect fit.
        __syncthreads();
        x_lds[tid] = act[b * kWeightsPerB + tid];
        __syncthreads();

        const uint8_t* __restrict__ blk = wrow + (size_t)b * kQ1BlockBytes;

        // Broadcast the scale `d`. Every thread loads the same 2 bytes —
        // compiler typically emits a single scalar load; if not, the L1
        // hit is negligible vs the sign-bit traffic.
        const __half d_h = load_scale_fp16(blk);
        const float  d   = __half2float(d_h);

        // Each thread handles weight `tid` within the block. qs[] starts
        // at offset 2 (after the 2-byte fp16 scale).
        const int byte_idx = tid >> 3;                 // tid / 8, range 0..15
        const int bit_idx  = tid & 7;                  // tid % 8
        const uint32_t qbyte = load_qs_byte(blk + 2, byte_idx);
        const uint32_t bit   = (qbyte >> bit_idx) & 1u;
        const float sign = bit ? 1.0f : -1.0f;         // w = sign * d

        const float a = __half2float(x_lds[tid]);
        acc = fmaf(sign * d, a, acc);
    }

    // ── Reduce across the workgroup ──
    // Intra-wave xor butterfly.
    #pragma unroll
    for (int off = kWaveSize / 2; off > 0; off >>= 1) {
        acc += __shfl_xor(acc, off);
    }

    // Inter-wave: lane 0 of each wave → LDS; wave 0 folds.
    if (lane == 0) wave_sums[wave] = acc;
    __syncthreads();

    if (wave == 0) {
        float v = (lane < kWaves) ? wave_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = kWaveSize / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off);
        }
        if (lane == 0) {
            // fp16 clamp+cast — same guard as sherry_gemv.hip. Post-hoc
            // ternary × large acts can overflow fp16 on outlier rows;
            // downstream RMSNorm / softmax poisons on +Inf.
            constexpr float FP16_MAX = 65504.0f;
            if (v >  FP16_MAX) v =  FP16_MAX;
            if (v < -FP16_MAX) v = -FP16_MAX;
            out[row] = __float2half(v);
        }
    }
}

// ── C API ───────────────────────────────────────────────────────────────────

extern "C" void bonsai_q1_gemv_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream)
{
    if (N_out <= 0 || K_in <= 0)            return;
    if ((K_in & (kWeightsPerB - 1)) != 0)   return;  // K_in % 128 == 0

    dim3 grid(N_out, 1, 1);
    dim3 block(kBlockSize, 1, 1);
    hipLaunchKernelGGL(bonsai_q1_gemv_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_weights,
                       reinterpret_cast<const __half*>(act_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       N_out, K_in);
}
