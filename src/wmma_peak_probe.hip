// gfx1151 WMMA throughput ceiling probe — no global memory on hot path.
// Each wave loads A and B fragments from tiny global init once, then spins
// for N_INNER WMMA iterations using register-resident fragments. Timed.
// Gives the hardware ceiling for f32_16x16x16_f16_w32 on Strix Halo.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half16_t = __attribute__((ext_vector_type(16))) __fp16;
using float8_t = __attribute__((ext_vector_type(8))) float;

namespace {

// 16 accumulators (like Phase 4g/h), pure compute in inner loop.
__global__ __launch_bounds__(32)
void wmma_peak_inner(
    const __half* __restrict__ A_seed,
    const __half* __restrict__ B_seed,
    float* __restrict__ sink,
    int n_inner)
{
    half16_t A_frag[4];
    half16_t B_frag[4];
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        #pragma unroll
        for(int e = 0; e < 16; ++e) {
            A_frag[i][e] = (__fp16)(float)A_seed[(threadIdx.x * 4 + i) * 16 + e];
            B_frag[i][e] = (__fp16)(float)B_seed[(threadIdx.x * 4 + i) * 16 + e];
        }
    }

    float8_t acc[4][4] = {};
    for(int it = 0; it < n_inner; ++it) {
        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            #pragma unroll
            for(int j = 0; j < 4; ++j) {
                acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    A_frag[i], B_frag[j], acc[i][j]);
            }
        }
    }

    // Prevent DCE — write one value per lane from every acc
    float s = 0.0f;
    #pragma unroll
    for(int i = 0; i < 4; ++i)
        #pragma unroll
        for(int j = 0; j < 4; ++j)
            s += acc[i][j][0];
    sink[blockIdx.x * 32 + threadIdx.x] = s;
}

}  // namespace

extern "C" void rcpp_peak_probe_launch(
    const void* A_seed, const void* B_seed, void* sink,
    int n_blocks, int n_inner, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid(n_blocks, 1, 1);
    hipLaunchKernelGGL(wmma_peak_inner,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_seed),
                       static_cast<const __half*>(B_seed),
                       static_cast<float*>(sink),
                       n_inner);
}
