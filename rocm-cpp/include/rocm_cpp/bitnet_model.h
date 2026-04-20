// BitNet model loader — pure C ABI.
// Reads .h1b files (halo-1bit format) and returns device pointers
// suitable for direct use with rocm_cpp kernels (rcpp_ternary_gemv,
// rcpp_rmsnorm_fp16, etc).

#ifndef ROCM_CPP_BITNET_MODEL_H
#define ROCM_CPP_BITNET_MODEL_H

#include "rocm_cpp/ck_gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // RMSNorm weights (FP16). BitNet-b1.58 has four per layer:
    //   input_norm    [hs] — pre Q/K/V
    //   attn_sub_norm [hs] — on attn output, before O proj
    //   post_attn_norm[hs] — pre gate/up
    //   ffn_sub_norm  [is] — on silu(gate)*up, before down proj
    void* input_norm_dev;
    void* post_attn_norm_dev;
    void* attn_sub_norm_dev;
    void* ffn_sub_norm_dev;

    // Ternary linear layers — halo-encoded uint8 packed (reinterpret as uint32
    // bytewise for rcpp_ternary_gemv_halo) + per-row FP32 scales.
    void* q_packed_dev;     float* q_scales_dev;      // [nh*hd, hs]
    void* k_packed_dev;     float* k_scales_dev;      // [nkv*hd, hs]
    void* v_packed_dev;     float* v_scales_dev;      // [nkv*hd, hs]
    void* o_packed_dev;     float* o_scales_dev;      // [hs, nh*hd]
    void* gate_packed_dev;  float* gate_scales_dev;   // [is, hs]
    void* up_packed_dev;    float* up_scales_dev;     // [is, hs]
    void* down_packed_dev;  float* down_scales_dev;   // [hs, is]
} rcpp_bitnet_layer_t;

typedef struct {
    int hidden_size;         // hs
    int intermediate_size;   // is
    int num_layers;
    int num_heads;           // nh
    int num_kv_heads;        // nkv  (GQA)
    int vocab_size;
    int max_seq_len;
    int tie_embeddings;      // non-zero => LM head = embedding.T

    float rope_theta;
    float rms_norm_eps;

    int format_version;      // 1/2 = halo v2 (2 bpw); 3 = Sherry v3 (1.25 bpw)

    void* embedding_dev;            // FP16 [vocab, hidden]
    void* final_norm_weight_dev;    // FP16 [hidden]
    rcpp_bitnet_layer_t* layers;    // [num_layers]
} rcpp_bitnet_model_t;

// Load a .h1b model file and upload all weights to the GPU.
// Caller eventually frees with rcpp_bitnet_free.
rcpp_status_t
rcpp_bitnet_load_h1b(const char* path, rcpp_bitnet_model_t* out_model);

// Release all device buffers + layer array.
void
rcpp_bitnet_free(rcpp_bitnet_model_t* model);

#ifdef __cplusplus
}
#endif
#endif  // ROCM_CPP_BITNET_MODEL_H
