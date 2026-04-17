// rocm-cpp C API — BitNet-style ternary GEMM on gfx1151
//
// Batched prefill: FP16 activations [M, K] × ternary weights [K, N] -> FP16 [M, N].
// Weights are pre-packed once at model load via rcpp_ternary_pack_pk_i4 and
// stored in WMMA-permuted pk_i4 layout (K * N / 2 bytes, half of FP16).
//
// Consumers do NOT pull in CK or any HIP templates — only this C header.
// Link: librocm_cpp.so (+ libhip64, HIP runtime).

#ifndef ROCM_CPP_CK_GEMM_H
#define ROCM_CPP_CK_GEMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RCPP_OK           = 0,
    RCPP_INVALID_ARG  = 1,
    RCPP_UNSUPPORTED  = 2,
    RCPP_HIP_ERROR    = 3,
    RCPP_INTERNAL     = 4,
} rcpp_status_t;

typedef struct rcpp_ck_gemm_handle rcpp_ck_gemm_handle_t;

// Create a handle for FP16 x pk_i4 -> FP16 GEMM at shape (M, N, K).
// Picks the CK instance internally; handle is reusable across stream launches.
// Returns RCPP_UNSUPPORTED if no CK instance accepts this shape (pad if needed).
rcpp_status_t
rcpp_ck_gemm_create(int M, int N, int K, rcpp_ck_gemm_handle_t** handle_out);

void
rcpp_ck_gemm_destroy(rcpp_ck_gemm_handle_t* handle);

// C[M, N] = A[M, K] * B[K, N] on the given HIP stream.
//   A_dev         : FP16 row-major [M, K] on device, stride K
//   B_dev_packed  : pk_i4 WMMA-permuted bytes [K*N/2] on device (from rcpp_ternary_pack_pk_i4)
//   C_dev         : FP16 row-major [M, N] on device, stride N
//   stream        : hipStream_t (void* to avoid a HIP include in the header)
rcpp_status_t
rcpp_ck_gemm_run(rcpp_ck_gemm_handle_t* handle,
                 const void* A_dev, const void* B_dev_packed, void* C_dev,
                 void* stream);

// Offline weight packer (host side).
//   ternary_host  : int8 values in {-1, 0, +1}, col-major [K, N], size K*N bytes
//   packed_host   : output, pk_i4 WMMA-permuted, size K*N/2 bytes
// Requires K % 32 == 0 and K % 8 == 0 (BitNet FFN / attention shapes satisfy this).
rcpp_status_t
rcpp_ternary_pack_pk_i4(const int8_t* ternary_host,
                        int8_t* packed_host,
                        int K, int N);

// Informational — returns CK's instance type string (or a stub if not built).
// Lifetime: tied to the handle.
const char*
rcpp_ck_gemm_instance_string(const rcpp_ck_gemm_handle_t* handle);

// -----------------------------------------------------------------------------
// Phase 5 decode GEMV — ternary × INT8 activations.
//
// For batch=1 decode (1 output vector = 1 token). Takes INT8 activations
// (user pre-quantizes with per-vector scale), packed ternary weights in v1
// format (2 bits per value, 16 values per uint32), per-row weight scales,
// and writes FP32 output.
//
// Uses the v_dot4_i32_iu8 builtin (gfx11 dot8-insts) with 8 rows per block.
// Benchmarked at 2.4-7.1× faster than rocBLAS FP16 GEMV across all measured
// shapes on gfx1151.
//
// Shape constraints: K must be a multiple of 16 (for the packed encoding) and
// ideally a multiple of LDS_TILE_I8 = 2048 for best perf (tail path exists).
rcpp_status_t
rcpp_ternary_gemv(const void* packed_weights_dev,   // [M, K/16] uint32
                  const void* activations_i8_dev,   // [K] int8
                  float       activation_scale,     // scalar — real_a = i8 * scale
                  const void* row_scales_dev,       // [M] float — per-row weight scale
                  void*       output_dev,           // [M] float — post-dequant output
                  int M, int K,
                  void*       stream);              // hipStream_t (nullable)

// Halo-1bit-encoded variant (uint8 [M, (K+3)/4] packed, code: 0->-1, 1->0, 2->+1).
// Buffer can be reinterpret-cast from halo's uint8 pack directly when K % 16 == 0.
rcpp_status_t
rcpp_ternary_gemv_halo(const void* packed_weights_dev,
                       const void* activations_i8_dev,
                       float       activation_scale,
                       const void* row_scales_dev,
                       void*       output_dev,
                       int M, int K,
                       void*       stream);

// -----------------------------------------------------------------------------
// Primitive kernels — support math so consumers don't have to write their own.
// All are batch=1 (decode). Batched variants come with Phase 6 KV cache work.

// Quantize FP16 activations to INT8 with per-vector scale.
//   scale = max(|x|) / 127, clamped to >= 1e-8
//   scale_dev is a single FP32 location the kernel writes.
rcpp_status_t
rcpp_quantize_fp16_to_i8(const void* x_fp16_dev, void* x_i8_dev,
                         float* scale_dev, int K, void* stream);

// RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight
rcpp_status_t
rcpp_rmsnorm_fp16(const void* x_dev, const void* weight_dev, void* y_dev,
                  float eps, int K, void* stream);

// Rotary position embedding on [num_heads, head_dim] at the given position.
// head_dim must be even.
rcpp_status_t
rcpp_rope_fp16(void* x_dev, int pos, float theta,
               int num_heads, int head_dim, void* stream);

// SiLU-gated elementwise: y[i] = silu(up[i]) * gate[i]
rcpp_status_t
rcpp_silu_glu_fp16(const void* up_dev, const void* gate_dev, void* y_dev,
                   int N, void* stream);

// Embedding lookup: y[k] = embedding[token_id, k] for k in 0..hidden-1
rcpp_status_t
rcpp_embedding_lookup_fp16(const void* embedding_dev, int token_id,
                           void* y_dev, int hidden, void* stream);

// -----------------------------------------------------------------------------
// Residual add: y[i] += src[i]
rcpp_status_t
rcpp_residual_add_fp16(void* y_dev, const void* src_dev, int N, void* stream);

// Argmax over FP32 logits — writes the max-index to *out_idx_dev.
// Caller allocates one int on the device for out_idx_dev.
rcpp_status_t
rcpp_argmax_fp32(const void* logits_dev, void* out_idx_dev, int V, void* stream);

// FP16 × FP16 → FP32 GEMV. For LM head on tied-embedding BitNet models
// (embedding matrix IS the LM head, stored as FP16). y[m] = sum_k W[m,k]*x[k].
rcpp_status_t
rcpp_fp16_gemv(const void* W_dev, const void* x_dev, void* y_dev,
               int M, int K, void* stream);

// In-place FP32 -> FP16 cast, useful to chain ternary GEMV output
// (FP32) into the next kernel that wants FP16 input.
rcpp_status_t
rcpp_fp32_to_fp16(const void* x_fp32_dev, void* y_fp16_dev, int N, void* stream);

// Top-k logit filter — in-place. Keeps only the k largest logit values; the
// rest are set to -INFINITY so subsequent softmax zeroes them out. Caller
// sequence for top-k sampling: top_k -> softmax -> multinomial.
// Max k = 512.
rcpp_status_t
rcpp_top_k_fp32(void* logits_dev, int k, int V, void* stream);

// Softmax with optional temperature — in-place. logits[] becomes probs[].
//   probs[v] = exp(logits[v] / T - max) / sum
// temperature must be > 0. Pass 1.0 for standard softmax.
rcpp_status_t
rcpp_softmax_fp32(void* logits_dev, int V, float temperature, void* stream);

// Multinomial sample — given a probability distribution probs[V] (from the
// softmax above), and a uniform random r in [0,1), returns the smallest index
// v such that cumsum(probs[0..v]) > r. Caller generates r on CPU (any PRNG).
rcpp_status_t
rcpp_sample_multinomial_fp32(const void* probs_dev, float r,
                             void* out_idx_dev, int V, void* stream);

// -----------------------------------------------------------------------------
// Phase 6 — KV cache attention for batch=1 decode (Flash-Decoding style).
//
// Computes: out[h] = softmax(scale * Q[h] · K[*, h//gqa_ratio]) · V[*, h//gqa_ratio]
// where gqa_ratio = num_q_heads / num_kv_heads.
//
// Layouts (FP16):
//   Q:       [num_q_heads, head_dim]
//   K_cache: [seq_len, num_kv_heads, head_dim]
//   V_cache: [seq_len, num_kv_heads, head_dim]
//   out:     [num_q_heads, head_dim]
//
// Online softmax state held per-block in registers; no seq_len-size scratch.
// Supports head_dim up to 256.
rcpp_status_t
rcpp_kv_cache_attn_decode(const void* Q_dev, const void* K_dev, const void* V_dev,
                          void* out_dev,
                          int num_q_heads, int num_kv_heads, int head_dim,
                          int seq_len, float scale, void* stream);

// Prefill attention — multi-token, causal mask. Each output position t attends
// only to K/V[0..t]. All tensors in [seq_len, heads, head_dim] layout (Q uses
// num_q_heads, K/V use num_kv_heads). Grid scales as seq_len * num_q_heads.
rcpp_status_t
rcpp_kv_cache_attn_prefill(const void* Q_dev, const void* K_dev, const void* V_dev,
                           void* out_dev,
                           int num_q_heads, int num_kv_heads, int head_dim,
                           int seq_len, float scale, void* stream);

// -----------------------------------------------------------------------------
// Standalone (CK-free) prefill launcher.
//
// Same inputs as rcpp_ck_gemm_run. Produces bit-identical output to the CK
// backend on BitNet-realistic shapes; reaches 94% of CK's tuned WMMA perf on
// gfx1151 with ZERO ck/ includes in this TU (see src/prefill_standalone.hip,
// docs/11-de-ck-plan.md).
//
// Use this when you want the library to ship without the CK template surface —
// e.g., for a binary distribution that should not depend on TheRock being
// pre-built on the consumer's machine.
//
// Stateless: no handle needed. M, N, K must satisfy M % 64 == 0, N % 64 == 0,
// K % 32 == 0 for the 64x64 output-tile kernel; callers with arbitrary shapes
// should pad or fall back to the CK backend.
rcpp_status_t
rcpp_standalone_gemm(const void* A_dev, const void* B_dev_packed, void* C_dev,
                     int M, int N, int K, void* stream);

#ifdef __cplusplus
}
#endif
#endif  // ROCM_CPP_CK_GEMM_H
