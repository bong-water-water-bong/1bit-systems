// Primitive kernels for MLX-free BitNet decode on gfx1151.
//
// Five small kernels that together with Phase 5 GEMV + Phase 4i prefill
// complete the BitNet decode math path without pulling in MLX:
//
//   1. rcpp_quantize_fp16_to_i8  — activation quantization (FP16 -> INT8 + scale)
//   2. rcpp_rmsnorm_fp16          — RMSNorm over the last dim
//   3. rcpp_rope_fp16             — rotary position embedding on Q/K
//   4. rcpp_silu_glu_fp16         — gate * SiLU(up) for SwiGLU FFN
//   5. rcpp_embedding_lookup_fp16 — token id -> FP16 embedding vector
//
// All kernels are single-block-per-output-item. Target: batch=1 decode.
// Batched variants can come later; decode is the primary use case.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>

#include "rocm_cpp/ck_gemm.h"

namespace {

constexpr int BLOCK = 256;
constexpr int WARP  = 32;

// ============================================================================
// 1. Activation quantization — FP16 [K] -> INT8 [K] + scalar scale
//    scale is computed as max(|x|) / 127 so INT8 values stay in [-127, +127].
// ============================================================================

__global__ void quantize_fp16_to_i8_kernel(
    const __half* __restrict__ x_fp16,
    int8_t*       __restrict__ x_i8,
    float*        __restrict__ out_scale,   // [1]
    int K)
{
    __shared__ float warp_max[BLOCK / WARP];
    __shared__ float scale_s;

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    // Block-wide max(|x|) reduction
    float local = 0.0f;
    for (int k = tid; k < K; k += BLOCK) {
        float v = std::fabsf((float)x_fp16[k]);
        if (v > local) local = v;
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) {
        float other = __shfl_xor(local, o);
        if (other > local) local = other;
    }
    if (lane == 0) warp_max[wid] = local;
    __syncthreads();

    if (wid == 0) {
        float v = (lane < BLOCK / WARP) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) {
            float other = __shfl_xor(v, o);
            if (other > v) v = other;
        }
        if (lane == 0) {
            float s = v / 127.0f;
            if (s < 1e-8f) s = 1e-8f;
            scale_s = s;
            *out_scale = s;
        }
    }
    __syncthreads();

    const float inv = 1.0f / scale_s;
    for (int k = tid; k < K; k += BLOCK) {
        float v = (float)x_fp16[k] * inv;
        int q   = (int)std::round(v);
        if (q >  127) q =  127;
        if (q < -127) q = -127;
        x_i8[k] = (int8_t)q;
    }
}

// ============================================================================
// 2. RMSNorm — y = (x / sqrt(mean(x^2) + eps)) * weight
//    Single block per output vector. Batch=1 decode.
// ============================================================================

__global__ void rmsnorm_fp16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half*       __restrict__ y,
    float                       eps,
    int K)
{
    __shared__ float warp_sum[BLOCK / WARP];
    __shared__ float rms_inv_s;

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    // Sum of squares
    float local = 0.0f;
    for (int k = tid; k < K; k += BLOCK) {
        float v = (float)x[k];
        local += v * v;
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) local += __shfl_xor(local, o);
    if (lane == 0) warp_sum[wid] = local;
    __syncthreads();

    if (wid == 0) {
        float s = (lane < BLOCK / WARP) ? warp_sum[lane] : 0.0f;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) s += __shfl_xor(s, o);
        if (lane == 0) {
            float mean = s / (float)K;
            rms_inv_s = 1.0f / std::sqrtf(mean + eps);
        }
    }
    __syncthreads();

    const float inv = rms_inv_s;
    for (int k = tid; k < K; k += BLOCK) {
        float v = (float)x[k] * inv * (float)weight[k];
        y[k] = __float2half(v);
    }
}

// ============================================================================
// 3. RoPE — apply rotary embedding to [num_heads, head_dim] at position pos.
//    Pair (x[2i], x[2i+1]) -> (cos*x0 - sin*x1, sin*x0 + cos*x1)
//    freq[i] = 1 / theta^(2i / head_dim)
//    Single block handles one (head, position) in the vector.
// ============================================================================

__global__ void rope_fp16_kernel(
    __half*  __restrict__ x,               // [num_heads, head_dim]
    int                   pos,
    float                 theta,
    int                   num_heads,
    int                   head_dim)
{
    const int h = blockIdx.x;
    if (h >= num_heads) return;
    const int tid = threadIdx.x;
    const int half_dim = head_dim / 2;

    __half* row = x + (size_t)h * head_dim;
    for (int i = tid; i < half_dim; i += blockDim.x) {
        const float freq = 1.0f / std::powf(theta, 2.0f * (float)i / (float)head_dim);
        const float angle = (float)pos * freq;
        const float c = std::cosf(angle);
        const float s = std::sinf(angle);
        const float x0 = (float)row[2 * i + 0];
        const float x1 = (float)row[2 * i + 1];
        row[2 * i + 0] = __float2half(c * x0 - s * x1);
        row[2 * i + 1] = __float2half(s * x0 + c * x1);
    }
}

// ============================================================================
// 4. SiLU-gated — y[i] = silu(up[i]) * gate[i]
//    silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
// ============================================================================

__global__ void silu_glu_fp16_kernel(
    const __half* __restrict__ up,
    const __half* __restrict__ gate,
    __half*       __restrict__ y,
    int                       N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    const float u = (float)up[tid];
    const float g = (float)gate[tid];
    const float silu = u / (1.0f + std::expf(-u));
    y[tid] = __float2half(silu * g);
}

// ============================================================================
// 6. Residual add — y[i] += src[i]  (in-place)
// ============================================================================

__global__ void residual_add_fp16_kernel(
    __half* __restrict__ y,
    const __half* __restrict__ src,
    int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    y[tid] = __float2half((float)y[tid] + (float)src[tid]);
}

// ============================================================================
// 8. Softmax (FP32) with optional temperature — in place on logits.
//    y[v] = exp((x[v] - max) / T) / sum
// ============================================================================

__global__ void softmax_fp32_kernel(
    float* __restrict__ logits,
    int V, float inv_T)
{
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_val[BLOCK / WARP];

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    // 1) max reduction
    float lmax = -INFINITY;
    for (int v = tid; v < V; v += BLOCK) {
        float x = logits[v] * inv_T;
        if (x > lmax) lmax = x;
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) {
        float ov = __shfl_xor(lmax, o);
        if (ov > lmax) lmax = ov;
    }
    if (lane == 0) s_val[wid] = lmax;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < BLOCK / WARP) ? s_val[lane] : -INFINITY;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) {
            float ov = __shfl_xor(v, o);
            if (ov > v) v = ov;
        }
        if (lane == 0) s_max = v;
    }
    __syncthreads();

    // 2) exp + sum
    float lsum = 0.0f;
    const float m = s_max;
    for (int v = tid; v < V; v += BLOCK) {
        float e = std::expf(logits[v] * inv_T - m);
        logits[v] = e;   // reuse buffer
        lsum += e;
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) lsum += __shfl_xor(lsum, o);
    if (lane == 0) s_val[wid] = lsum;
    __syncthreads();
    if (wid == 0) {
        float s = (lane < BLOCK / WARP) ? s_val[lane] : 0.0f;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) s += __shfl_xor(s, o);
        if (lane == 0) s_sum = s;
    }
    __syncthreads();

    // 3) normalize
    const float inv = 1.0f / s_sum;
    for (int v = tid; v < V; v += BLOCK) logits[v] *= inv;
}

// ============================================================================
// 9. Multinomial sample — given probs[V] and a uniform r ∈ [0, 1),
//    return the smallest v where cumsum(probs[0..v]) > r.
//    Linear scan on one thread — correct but could be faster (inverse-CDF
//    binary search). Good enough for one-token-per-step decode.
// ============================================================================

__global__ void sample_multinomial_fp32_kernel(
    const float* __restrict__ probs,
    float                     r,
    int* __restrict__ out_idx,
    int V)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float cumu = 0.0f;
    int picked = V - 1;
    for (int v = 0; v < V; ++v) {
        cumu += probs[v];
        if (cumu >= r) { picked = v; break; }
    }
    *out_idx = picked;
}

// ============================================================================
// 7. Argmax sampling — returns index of max element in logits[V]
//    Single block, block-wide reduction.
// ============================================================================

__global__ void argmax_fp32_kernel(
    const float* __restrict__ logits,
    int* __restrict__ out_idx,
    int V)
{
    __shared__ float s_val[BLOCK / WARP];
    __shared__ int   s_idx[BLOCK / WARP];

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    float best_v = -INFINITY;
    int   best_i = 0;
    for (int v = tid; v < V; v += BLOCK) {
        float x = logits[v];
        if (x > best_v) { best_v = x; best_i = v; }
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) {
        float ov = __shfl_xor(best_v, o);
        int   oi = __shfl_xor(best_i, o);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) { s_val[wid] = best_v; s_idx[wid] = best_i; }
    __syncthreads();

    if (wid == 0) {
        float v = (lane < BLOCK / WARP) ? s_val[lane] : -INFINITY;
        int   i = (lane < BLOCK / WARP) ? s_idx[lane] : 0;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) {
            float ov = __shfl_xor(v, o);
            int   oi = __shfl_xor(i, o);
            if (ov > v) { v = ov; i = oi; }
        }
        if (lane == 0) *out_idx = i;
    }
}

// ============================================================================
// 5. Embedding lookup — y[k] = E[token_id, k]  for k in 0..hidden-1
// ============================================================================

__global__ void embedding_lookup_fp16_kernel(
    const __half* __restrict__ embedding,   // [vocab, hidden]
    int                         token_id,
    __half*       __restrict__ y,            // [hidden]
    int                         hidden)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden) return;
    y[tid] = embedding[(size_t)token_id * hidden + tid];
}

}  // namespace

extern "C" {

rcpp_status_t
rcpp_quantize_fp16_to_i8(const void* x_fp16_dev, void* x_i8_dev,
                         float* scale_dev, int K, void* stream)
{
    if (!x_fp16_dev || !x_i8_dev || !scale_dev || K <= 0) return RCPP_INVALID_ARG;
    dim3 grid(1, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(quantize_fp16_to_i8_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(x_fp16_dev),
                       static_cast<int8_t*>(x_i8_dev),
                       scale_dev, K);
    return RCPP_OK;
}

rcpp_status_t
rcpp_rmsnorm_fp16(const void* x_dev, const void* weight_dev, void* y_dev,
                  float eps, int K, void* stream)
{
    if (!x_dev || !weight_dev || !y_dev || K <= 0) return RCPP_INVALID_ARG;
    dim3 grid(1, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(rmsnorm_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(x_dev),
                       static_cast<const __half*>(weight_dev),
                       static_cast<__half*>(y_dev),
                       eps, K);
    return RCPP_OK;
}

rcpp_status_t
rcpp_rope_fp16(void* x_dev, int pos, float theta,
               int num_heads, int head_dim, void* stream)
{
    if (!x_dev || num_heads <= 0 || head_dim <= 0 || (head_dim & 1)) return RCPP_INVALID_ARG;
    dim3 grid(num_heads, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(rope_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<__half*>(x_dev),
                       pos, theta, num_heads, head_dim);
    return RCPP_OK;
}

rcpp_status_t
rcpp_silu_glu_fp16(const void* up_dev, const void* gate_dev, void* y_dev,
                   int N, void* stream)
{
    if (!up_dev || !gate_dev || !y_dev || N <= 0) return RCPP_INVALID_ARG;
    dim3 grid((N + BLOCK - 1) / BLOCK, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(silu_glu_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(up_dev),
                       static_cast<const __half*>(gate_dev),
                       static_cast<__half*>(y_dev),
                       N);
    return RCPP_OK;
}

rcpp_status_t
rcpp_embedding_lookup_fp16(const void* embedding_dev, int token_id,
                           void* y_dev, int hidden, void* stream)
{
    if (!embedding_dev || !y_dev || token_id < 0 || hidden <= 0) return RCPP_INVALID_ARG;
    dim3 grid((hidden + BLOCK - 1) / BLOCK, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(embedding_lookup_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(embedding_dev),
                       token_id,
                       static_cast<__half*>(y_dev),
                       hidden);
    return RCPP_OK;
}

rcpp_status_t
rcpp_residual_add_fp16(void* y_dev, const void* src_dev, int N, void* stream)
{
    if (!y_dev || !src_dev || N <= 0) return RCPP_INVALID_ARG;
    dim3 grid((N + BLOCK - 1) / BLOCK, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(residual_add_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<__half*>(y_dev),
                       static_cast<const __half*>(src_dev),
                       N);
    return RCPP_OK;
}

rcpp_status_t
rcpp_argmax_fp32(const void* logits_dev, void* out_idx_dev, int V, void* stream)
{
    if (!logits_dev || !out_idx_dev || V <= 0) return RCPP_INVALID_ARG;
    dim3 grid(1, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(argmax_fp32_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const float*>(logits_dev),
                       static_cast<int*>(out_idx_dev),
                       V);
    return RCPP_OK;
}

rcpp_status_t
rcpp_softmax_fp32(void* logits_dev, int V, float temperature, void* stream)
{
    if (!logits_dev || V <= 0 || temperature <= 0.0f) return RCPP_INVALID_ARG;
    dim3 grid(1, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(softmax_fp32_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<float*>(logits_dev),
                       V, 1.0f / temperature);
    return RCPP_OK;
}

rcpp_status_t
rcpp_sample_multinomial_fp32(const void* probs_dev, float r,
                             void* out_idx_dev, int V, void* stream)
{
    if (!probs_dev || !out_idx_dev || V <= 0) return RCPP_INVALID_ARG;
    if (r < 0.0f) r = 0.0f;
    if (r >= 1.0f) r = 1.0f - 1e-7f;
    dim3 grid(1, 1, 1), block(1, 1, 1);
    hipLaunchKernelGGL(sample_multinomial_fp32_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const float*>(probs_dev),
                       r,
                       static_cast<int*>(out_idx_dev),
                       V);
    return RCPP_OK;
}

}  // extern "C"
