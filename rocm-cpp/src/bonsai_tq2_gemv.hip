// bonsai_tq2_gemv.hip — Bonsai TQ2_0_g128 ternary GEMV for gfx1151 (wave32).
//
// API (see include/rocm_cpp/bonsai.h):
//   bonsai_tq2_gemv_launch(packed, act_fp16, out_fp16, N_out, K_in, stream)
//
// Layout (AoS, block-interleaved, 34 bytes per g128 block):
//   [FP16 d : 2 bytes][qs[32] : 32 bytes of 2-bit codes]
//   (d-first, NOT d-last — verified against the raw PrismML
//   Ternary-Bonsai-1.7B-Q2_0.gguf file. The Bonsai-Kernel-Spec.md doc
//   originally claimed d-last; it was wrong. Empirical: with d@0 no
//   block emits the reserved 0b11 code; with d@32 1-2 per block do,
//   and d@32 varies wildly across 10 orders of magnitude while d@0
//   stays in a tight per-row band.)
//   Code map per 2-bit code: 00→-1, 01→0, 10→+1, 11→0 (reserved).
//   4 codes per byte, LSB-first. Weight `j` within the block:
//     byte_idx = 2 + j/4, shift = (j%4)*2, code = (qs[byte_idx] >> shift) & 3.
//
// Thread model (wave32 gfx1151):
//   - One workgroup = 128 threads (4 waves × 32 lanes) per output row.
//   - One block (128 weights) consumed per iteration: thread `tid`
//     handles weight `tid` in the block. For K_in > 128 the workgroup
//     iterates over `K_in / 128` blocks sequentially.
//   - Activations staged into 128-entry LDS once per block, reused by
//     all 4 waves (256 B slab). FP32 accumulation. Intra-wave __shfl_xor
//     butterfly + inter-wave LDS fold into wave 0's lane 0.
//
// 2-bit decode — branch-free, no LUT:
//   For code c ∈ {0, 1, 2, 3} with the mapping above, the ternary value
//   v(c) ∈ {-1, 0, +1, 0}. Key observation:
//     v(c) = (c == 0) ? -1 : ((c == 2) ? +1 : 0)
//          = float(c == 2) - float(c == 0)
//   Each `c == k` lowers to a single v_cmp_u32 + v_cvt_f32_u32 pair on
//   RDNA3.5 and the compiler fuses the subtraction into one FMA. The
//   0b11 reserved code decodes to 0 naturally (both comparisons false).
//   No branch, no cndmask storm, no LUT.
//
// Load pattern:
//   Each thread reads the 1 byte containing its own 2-bit code:
//     qs_byte = qs[tid / 4]  // absolute block offset: 2 + tid/4
//   tid / 4 spans 0..31 across 128 threads → 32 bytes. A wave of 32
//   lanes reads 8 consecutive bytes (lanes 0..3 read byte 0, lanes 4..7
//   read byte 1, …). Coalesced as one 32-byte global transaction.
//   Scale `d` at offset 0 is broadcast-loaded by all threads.
//
// LDS usage: 128 × fp16 (256 B) act slab + 4 × fp32 (16 B) wave scratch
// = 272 B per workgroup. Register pressure target ≤ 64 VGPR.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

#include "rocm_cpp/bonsai.h"

namespace {

constexpr int kBlockSize     = 128;
constexpr int kWaveSize      = 32;
constexpr int kWaves         = kBlockSize / kWaveSize;
constexpr int kWeightsPerB   = 128;
constexpr int kTQ2BlockBytes = 34;
constexpr int kScaleOffset   = 0;    // `d` first, then 32 bytes of codes
constexpr int kCodesOffset   = 2;    // codes[0..32) live at blk + 2

__device__ __forceinline__
uint32_t load_qs_byte(const uint8_t* __restrict__ qs_ptr, int byte_idx) {
    return (uint32_t)qs_ptr[byte_idx];
}

__device__ __forceinline__
__half load_scale_fp16(const uint8_t* __restrict__ block_ptr) {
    uint16_t u;
    __builtin_memcpy(&u, block_ptr + kScaleOffset, sizeof(u));
    __half h;
    __builtin_memcpy(&h, &u, sizeof(h));
    return h;
}

// Branch-free 2-bit decode. Input `code` is in {0,1,2,3}; output is the
// ternary weight in {-1, 0, +1}. Reserved 0b11 lands on 0 without a
// guard — both comparisons below evaluate false.
__device__ __forceinline__
float decode_tq2(uint32_t code) {
    // (c == 2) → +1; (c == 0) → -1; else 0.
    const float pos = (code == 2u) ? 1.0f : 0.0f;
    const float neg = (code == 0u) ? 1.0f : 0.0f;
    return pos - neg;
}

}  // namespace

__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
__global__ void bonsai_tq2_gemv_kernel(
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

    const int num_blocks   = K_in >> 7;
    const size_t row_bytes = (size_t)num_blocks * (size_t)kTQ2BlockBytes;
    const uint8_t* __restrict__ wrow = packed + (size_t)row * row_bytes;

    __shared__ __half x_lds[kWeightsPerB];
    __shared__ float  wave_sums[kWaves];

    float acc = 0.0f;

    for (int b = 0; b < num_blocks; ++b) {
        // Coop load 128 fp16 acts.
        __syncthreads();
        x_lds[tid] = act[b * kWeightsPerB + tid];
        __syncthreads();

        const uint8_t* __restrict__ blk = wrow + (size_t)b * kTQ2BlockBytes;

        // Broadcast scale `d` (at offset 0).
        const __half d_h = load_scale_fp16(blk);
        const float  d   = __half2float(d_h);

        // Decode the tid-th 2-bit code. qs[] starts at block offset 2.
        const int byte_idx = kCodesOffset + (tid >> 2); // range 2..33
        const int shift    = (tid & 3) << 1;            // (tid % 4) * 2
        const uint32_t qbyte = load_qs_byte(blk, byte_idx);
        const uint32_t code  = (qbyte >> shift) & 3u;
        const float    tern  = decode_tq2(code);       // ∈ {-1, 0, +1}
        const float    w     = tern * d;

        const float a = __half2float(x_lds[tid]);
        acc = fmaf(w, a, acc);
    }

    // Intra-wave xor butterfly.
    #pragma unroll
    for (int off = kWaveSize / 2; off > 0; off >>= 1) {
        acc += __shfl_xor(acc, off);
    }

    if (lane == 0) wave_sums[wave] = acc;
    __syncthreads();

    if (wave == 0) {
        float v = (lane < kWaves) ? wave_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = kWaveSize / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off);
        }
        if (lane == 0) {
            constexpr float FP16_MAX = 65504.0f;
            if (v >  FP16_MAX) v =  FP16_MAX;
            if (v < -FP16_MAX) v = -FP16_MAX;
            out[row] = __float2half(v);
        }
    }
}

// ── C API ───────────────────────────────────────────────────────────────────

extern "C" void bonsai_tq2_gemv_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream)
{
    if (N_out <= 0 || K_in <= 0)            return;
    if ((K_in & (kWeightsPerB - 1)) != 0)   return;

    dim3 grid(N_out, 1, 1);
    dim3 block(kBlockSize, 1, 1);
    hipLaunchKernelGGL(bonsai_tq2_gemv_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_weights,
                       reinterpret_cast<const __half*>(act_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       N_out, K_in);
}
