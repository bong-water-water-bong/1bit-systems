// bitnet_decode — minimal end-to-end BitNet-2B-4T forward pass using only
// librocm_cpp. No MLX, no halo-1bit, no external ML framework.
//
// Usage: bitnet_decode <model.h1b> [start_token_id=1] [num_new_tokens=16]
//
// Greedy (argmax) decode. Prints generated token IDs + per-token latency.
// Doesn't include a tokenizer — caller provides token IDs directly.
//
// Pipeline matches BitNet-b1.58:
//   input_norm → QKV proj → RoPE → attention → attn_sub_norm → O proj
//   → residual → post_attn_norm → gate/up proj → relu² GLU → ffn_sub_norm
//   → down proj → residual → final_norm → tied LM head → argmax
//
// Status: full pipeline runs at ~85 tok/s on Strix Halo (gfx1151) against
// the corrected absmean-quantized .h1b. Logits are finite and vary with
// input, but residual stream magnitude drifts across 30 layers (±3 → ±13k)
// so decoded tokens loop on fixed points. Remaining gap lives in one of:
//   (a) input_layernorm/attn_sub_norm convention (tiny weights ~0.02)
//   (b) activation scaling mismatch with reference BitLinear path
// Comparing per-stage magnitudes against a PyTorch reference is the next
// debugging step when the architect resumes the work.

#include "rocm_cpp/ck_gemm.h"
#include "rocm_cpp/bitnet_model.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP %d %s:%d\n",_s,__FILE__,__LINE__); return 1;}} while(0)
#define RC_OK(e)  do { auto _s=(e); if(_s!=RCPP_OK){fprintf(stderr,"rcpp err %d at %s:%d\n",(int)_s,__FILE__,__LINE__); return 1;}} while(0)

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/home/bcloud/halo-1bit/models/halo-1bit-2b.h1b";
    const int   start_tok  = argc > 2 ? std::atoi(argv[2]) : 1;
    const int   num_tokens = argc > 3 ? std::atoi(argv[3]) : 16;

    rcpp_bitnet_model_t m;
    if (rcpp_bitnet_load_h1b(path, &m) != RCPP_OK) {
        fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }

    const int hs  = m.hidden_size;
    const int is  = m.intermediate_size;
    const int nh  = m.num_heads;
    const int nkv = m.num_kv_heads;
    const int hd  = hs / nh;
    const int L   = m.num_layers;
    const int V   = m.vocab_size;
    const int max_len = num_tokens + 1;   // small KV cache: prompt token + generated
    const float scale = 1.0f / std::sqrt((float)hd);

    fprintf(stderr, "[bitnet_decode] start_tok=%d num_tokens=%d max_ctx=%d\n",
            start_tok, num_tokens, max_len);

    // ---- Scratch buffers on device ----
    _Float16 *x, *normed, *x_i8_scratch_fp16;
    int8_t   *x_i8;
    float    *x_scale_dev;
    float    *q_raw, *k_raw, *v_raw, *o_raw, *gate_raw, *up_raw, *down_raw;
    _Float16 *q_fp16, *k_fp16, *v_fp16, *o_fp16, *gate_fp16, *up_fp16, *down_fp16;
    _Float16 *silu_out;
    int8_t   *silu_i8;
    float    *silu_scale_dev;
    float    *logits;
    int      *next_tok_dev;

    HIP_OK(hipMalloc(&x,             hs * 2));
    HIP_OK(hipMalloc(&normed,        hs * 2));
    HIP_OK(hipMalloc(&x_i8_scratch_fp16, hs * 2));  // unused slot, kept for parity
    HIP_OK(hipMalloc(&x_i8,          hs));
    HIP_OK(hipMalloc(&x_scale_dev,   4));
    HIP_OK(hipMalloc(&q_raw,         nh * hd * 4));
    HIP_OK(hipMalloc(&k_raw,         nkv * hd * 4));
    HIP_OK(hipMalloc(&v_raw,         nkv * hd * 4));
    HIP_OK(hipMalloc(&q_fp16,        nh * hd * 2));
    HIP_OK(hipMalloc(&k_fp16,        nkv * hd * 2));
    HIP_OK(hipMalloc(&v_fp16,        nkv * hd * 2));
    HIP_OK(hipMalloc(&o_raw,         hs * 4));
    HIP_OK(hipMalloc(&o_fp16,        hs * 2));
    HIP_OK(hipMalloc(&gate_raw,      is * 4));
    HIP_OK(hipMalloc(&up_raw,        is * 4));
    HIP_OK(hipMalloc(&down_raw,      hs * 4));
    HIP_OK(hipMalloc(&gate_fp16,     is * 2));
    HIP_OK(hipMalloc(&up_fp16,       is * 2));
    HIP_OK(hipMalloc(&down_fp16,     hs * 2));
    HIP_OK(hipMalloc(&silu_out,      is * 2));
    HIP_OK(hipMalloc(&silu_i8,       is));
    HIP_OK(hipMalloc(&silu_scale_dev, 4));
    HIP_OK(hipMalloc(&logits,        V * 4));
    HIP_OK(hipMalloc(&next_tok_dev,  4));

    // ---- KV cache (per layer) ----
    std::vector<_Float16*> K_caches(L, nullptr), V_caches(L, nullptr);
    const size_t kv_size = (size_t)max_len * nkv * hd * sizeof(_Float16);
    for (int l = 0; l < L; ++l) {
        HIP_OK(hipMalloc(&K_caches[l], kv_size));
        HIP_OK(hipMalloc(&V_caches[l], kv_size));
    }

    // ---- Forward pass for one token at position pos ----
    auto forward_token = [&](int token_id, int pos) -> int {
        RC_OK(rcpp_embedding_lookup_fp16(m.embedding_dev, token_id, x, hs, nullptr));

        for (int l = 0; l < L; ++l) {
            rcpp_bitnet_layer_t& ly = m.layers[l];

            // --- Attention block ---
            _Float16* residual = x;
            RC_OK(rcpp_rmsnorm_fp16(x, ly.input_norm_dev, normed, m.rms_norm_eps, hs, nullptr));
            RC_OK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs, nullptr));
            float x_scale;
            HIP_OK(hipMemcpy(&x_scale, x_scale_dev, 4, hipMemcpyDeviceToHost));

            // Q/K/V projections
            RC_OK(rcpp_ternary_gemv_halo(ly.q_packed_dev, x_i8, x_scale, ly.q_scales_dev, q_raw, nh*hd,  hs, nullptr));
            RC_OK(rcpp_ternary_gemv_halo(ly.k_packed_dev, x_i8, x_scale, ly.k_scales_dev, k_raw, nkv*hd, hs, nullptr));
            RC_OK(rcpp_ternary_gemv_halo(ly.v_packed_dev, x_i8, x_scale, ly.v_scales_dev, v_raw, nkv*hd, hs, nullptr));

            // FP32 -> FP16 for RoPE + attention
            RC_OK(rcpp_fp32_to_fp16(q_raw, q_fp16, nh*hd,  nullptr));
            RC_OK(rcpp_fp32_to_fp16(k_raw, k_fp16, nkv*hd, nullptr));
            RC_OK(rcpp_fp32_to_fp16(v_raw, v_fp16, nkv*hd, nullptr));

            // RoPE on Q and K
            RC_OK(rcpp_rope_fp16(q_fp16, pos, m.rope_theta, nh,  hd, nullptr));
            RC_OK(rcpp_rope_fp16(k_fp16, pos, m.rope_theta, nkv, hd, nullptr));

            // Append this token's K/V to the per-layer cache at slot 'pos'
            HIP_OK(hipMemcpy(K_caches[l] + (size_t)pos * nkv * hd, k_fp16, nkv*hd*2, hipMemcpyDeviceToDevice));
            HIP_OK(hipMemcpy(V_caches[l] + (size_t)pos * nkv * hd, v_fp16, nkv*hd*2, hipMemcpyDeviceToDevice));

            // Attention — decode kernel, attending to positions 0..pos
            RC_OK(rcpp_kv_cache_attn_decode(q_fp16, K_caches[l], V_caches[l],
                                            o_fp16, nh, nkv, hd, pos+1, scale, nullptr));

            // BitNet b1.58: attn_sub_norm on attention output before O proj
            RC_OK(rcpp_rmsnorm_fp16(o_fp16, ly.attn_sub_norm_dev, normed,
                                    m.rms_norm_eps, hs, nullptr));
            RC_OK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs, nullptr));
            HIP_OK(hipMemcpy(&x_scale, x_scale_dev, 4, hipMemcpyDeviceToHost));
            RC_OK(rcpp_ternary_gemv_halo(ly.o_packed_dev, x_i8, x_scale, ly.o_scales_dev, o_raw, hs, nh*hd, nullptr));
            RC_OK(rcpp_fp32_to_fp16(o_raw, o_fp16, hs, nullptr));
            RC_OK(rcpp_residual_add_fp16(residual, o_fp16, hs, nullptr));   // x += o_fp16

            // --- FFN block ---
            RC_OK(rcpp_rmsnorm_fp16(x, ly.post_attn_norm_dev, normed, m.rms_norm_eps, hs, nullptr));
            RC_OK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs, nullptr));
            HIP_OK(hipMemcpy(&x_scale, x_scale_dev, 4, hipMemcpyDeviceToHost));

            RC_OK(rcpp_ternary_gemv_halo(ly.gate_packed_dev, x_i8, x_scale, ly.gate_scales_dev, gate_raw, is, hs, nullptr));
            RC_OK(rcpp_ternary_gemv_halo(ly.up_packed_dev,   x_i8, x_scale, ly.up_scales_dev,   up_raw,   is, hs, nullptr));
            RC_OK(rcpp_fp32_to_fp16(gate_raw, gate_fp16, is, nullptr));
            RC_OK(rcpp_fp32_to_fp16(up_raw,   up_fp16,   is, nullptr));

            // BitNet-b1.58 uses ReLU² GLU (hidden_act=relu2): relu(gate)² * up
            RC_OK(rcpp_relu2_glu_fp16(gate_fp16, up_fp16, silu_out, is, nullptr));
            // ffn_sub_norm on act_out before down proj.
            RC_OK(rcpp_rmsnorm_fp16(silu_out, ly.ffn_sub_norm_dev, up_fp16,
                                    m.rms_norm_eps, is, nullptr));
            RC_OK(rcpp_quantize_fp16_to_i8(up_fp16, silu_i8, silu_scale_dev, is, nullptr));
            float silu_scale;
            HIP_OK(hipMemcpy(&silu_scale, silu_scale_dev, 4, hipMemcpyDeviceToHost));

            RC_OK(rcpp_ternary_gemv_halo(ly.down_packed_dev, silu_i8, silu_scale, ly.down_scales_dev, down_raw, hs, is, nullptr));
            RC_OK(rcpp_fp32_to_fp16(down_raw, down_fp16, hs, nullptr));
            RC_OK(rcpp_residual_add_fp16(x, down_fp16, hs, nullptr));
        }

        // Final norm + LM head
        RC_OK(rcpp_rmsnorm_fp16(x, m.final_norm_weight_dev, normed, m.rms_norm_eps, hs, nullptr));
        RC_OK(rcpp_fp16_gemv(m.embedding_dev, normed, logits, V, hs, nullptr));

        // Greedy sample
        RC_OK(rcpp_argmax_fp32(logits, next_tok_dev, V, nullptr));
        int next_tok;
        HIP_OK(hipMemcpy(&next_tok, next_tok_dev, 4, hipMemcpyDeviceToHost));
        return next_tok;
    };

    // ---- Generation loop ----
    int cur_tok = start_tok;
    printf("[bitnet_decode] tokens:");

    double total_ms = 0.0;
    for (int step = 0; step < num_tokens; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int next_tok = forward_token(cur_tok, step);
        HIP_OK(hipDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        printf(" %d", next_tok);
        fflush(stdout);
        cur_tok = next_tok;
    }
    printf("\n");
    printf("[bitnet_decode] %d tokens in %.2f ms  (%.2f ms/tok, %.1f tok/s)\n",
           num_tokens, total_ms, total_ms/num_tokens, 1000.0 * num_tokens / total_ms);

    // Cleanup
    for (int l = 0; l < L; ++l) { hipFree(K_caches[l]); hipFree(V_caches[l]); }
    rcpp_bitnet_free(&m);
    return 0;
}
