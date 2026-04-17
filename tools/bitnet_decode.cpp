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
//   → residual → post_attn_norm → gate/up proj → fused relu² GLU + ffn_sub_norm
//   → down proj → residual → final_norm → tied LM head → argmax
//
// Residual stream is FP32 throughout. The raw relu²(gate)*up intermediate
// reaches ~1e9 on real weights, so the ReLU² GLU is fused with its
// ffn_sub_norm inside the kernel (FP32 internal, FP16 output). The PyTorch
// reference (absmean quant, 1/mean(|W|) scale, ReLU² GLU, sub_norms) gives
// the exact same top-5 tokens for every input we've checked.

#include "rocm_cpp/ck_gemm.h"
#include "rocm_cpp/bitnet_model.h"
#include "rocm_cpp/tokenizer.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP %d %s:%d\n",_s,__FILE__,__LINE__); return 1;}} while(0)
#define RC_OK(e)  do { auto _s=(e); if(_s!=RCPP_OK){fprintf(stderr,"rcpp err %d at %s:%d\n",(int)_s,__FILE__,__LINE__); return 1;}} while(0)

// Load a whitespace-separated list of token IDs from a stream.
// Used when --prompt arg starts with @ (file path) or "-" (stdin).
static std::vector<int> read_token_ids(std::istream& in) {
    std::vector<int> ids;
    int t;
    while (in >> t) ids.push_back(t);
    return ids;
}

int main(int argc, char** argv) {
    // CLI:
    //   bitnet_decode <model.h1b> <prompt> <num_new_tokens> [tokenizer.htok]
    //   bitnet_decode <model.h1b>                                 # defaults
    //
    // <prompt> forms:
    //   --text "your prompt"      — encode via librocm_cpp tokenizer (.htok)
    //   @file.toks                — whitespace-separated ints from file
    //   -                         — whitespace-separated ints from stdin
    //   <int>                     — single start_tok (legacy)
    const char* path = argc > 1 ? argv[1] : "/home/bcloud/halo-1bit/models/halo-1bit-2b.h1b";
    const char* prompt_arg = argc > 2 ? argv[2] : "1";
    int num_tokens = 16;
    const char* tok_path = "/home/bcloud/halo-1bit/models/halo-1bit-2b.htok";

    std::vector<int> prompt_ids;
    if (std::string(prompt_arg) == "--text") {
        // layout: bitnet_decode <model> --text "<prompt>" <num_new> [tokenizer.htok]
        if (argc < 4) { fprintf(stderr, "usage: --text \"<prompt text>\" <num_new> [tokenizer.htok]\n"); return 1; }
        const char* text = argv[3];
        num_tokens = argc > 4 ? std::atoi(argv[4]) : 32;
        if (argc > 5) tok_path = argv[5];
        rcpp_tokenizer_t* tok = nullptr;
        if (rcpp_tokenizer_load(tok_path, &tok) != RCPP_OK) {
            fprintf(stderr, "cannot load tokenizer .htok: %s\n", tok_path); return 1;
        }
        size_t count = 0;
        std::vector<int> buf(4096);
        rcpp_tokenizer_encode(tok, text, std::strlen(text), /*add_bos=*/1,
                              buf.data(), buf.size(), &count);
        if (count > buf.size()) { buf.resize(count);
            rcpp_tokenizer_encode(tok, text, std::strlen(text), 1, buf.data(), buf.size(), &count);
        }
        buf.resize(count);
        prompt_ids = buf;
        rcpp_tokenizer_free(tok);
        fprintf(stderr, "[tokenizer] \"%s\" -> %zu tokens\n", text, count);
    } else if (prompt_arg[0] == '@') {
        num_tokens = argc > 3 ? std::atoi(argv[3]) : 16;
        std::ifstream f(prompt_arg + 1);
        if (!f) { fprintf(stderr, "cannot open prompt file: %s\n", prompt_arg + 1); return 1; }
        prompt_ids = read_token_ids(f);
    } else if (std::string(prompt_arg) == "-") {
        num_tokens = argc > 3 ? std::atoi(argv[3]) : 16;
        prompt_ids = read_token_ids(std::cin);
    } else {
        num_tokens = argc > 3 ? std::atoi(argv[3]) : 16;
        prompt_ids = { std::atoi(prompt_arg) };
    }
    if (prompt_ids.empty()) { fprintf(stderr, "prompt is empty\n"); return 1; }
    const int start_tok = prompt_ids.front();

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
    const int prompt_len = (int)prompt_ids.size();
    const int max_len = prompt_len + num_tokens;
    const float scale = 1.0f / std::sqrt((float)hd);

    fprintf(stderr, "[bitnet_decode] prompt_len=%d new_tokens=%d max_ctx=%d\n",
            prompt_len, num_tokens, max_len);
    (void)start_tok;  // preserved above for logging continuity

    // ---- Scratch buffers on device ----
    // x_fp32 is the FP32 residual stream (the dominant numerical-stability
    // knob in deep transformers). Sublayer math and KV cache stay FP16.
    float    *x_fp32;
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

    HIP_OK(hipMalloc(&x_fp32,        hs * 4));
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
        // Seed the FP32 residual stream from the FP16 embedding.
        RC_OK(rcpp_embedding_lookup_fp16(m.embedding_dev, token_id, x, hs, nullptr));
        HIP_OK(hipMemsetAsync(x_fp32, 0, hs * sizeof(float), nullptr));
        RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, x, hs, nullptr));

        for (int l = 0; l < L; ++l) {
            rcpp_bitnet_layer_t& ly = m.layers[l];

            // --- Attention block ---
            RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, ly.input_norm_dev, normed,
                                                m.rms_norm_eps, hs, nullptr));
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
            RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, o_fp16, hs, nullptr));

            // --- FFN block ---
            RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, ly.post_attn_norm_dev, normed,
                                                m.rms_norm_eps, hs, nullptr));
            RC_OK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs, nullptr));
            HIP_OK(hipMemcpy(&x_scale, x_scale_dev, 4, hipMemcpyDeviceToHost));

            RC_OK(rcpp_ternary_gemv_halo(ly.gate_packed_dev, x_i8, x_scale, ly.gate_scales_dev, gate_raw, is, hs, nullptr));
            RC_OK(rcpp_ternary_gemv_halo(ly.up_packed_dev,   x_i8, x_scale, ly.up_scales_dev,   up_raw,   is, hs, nullptr));
            RC_OK(rcpp_fp32_to_fp16(gate_raw, gate_fp16, is, nullptr));
            RC_OK(rcpp_fp32_to_fp16(up_raw,   up_fp16,   is, nullptr));

            // BitNet-b1.58 FFN activation: relu²(gate) * up — fused with
            // ffn_sub_norm in FP32 to avoid FP16 overflow of the raw product
            // (magnitude reaches ~1e9 on real weights; FP16 max is 6.5e4).
            RC_OK(rcpp_relu2_glu_rmsnorm_fp16(gate_fp16, up_fp16, ly.ffn_sub_norm_dev,
                                              silu_out, m.rms_norm_eps, is, nullptr));
            RC_OK(rcpp_quantize_fp16_to_i8(silu_out, silu_i8, silu_scale_dev, is, nullptr));
            float silu_scale;
            HIP_OK(hipMemcpy(&silu_scale, silu_scale_dev, 4, hipMemcpyDeviceToHost));

            RC_OK(rcpp_ternary_gemv_halo(ly.down_packed_dev, silu_i8, silu_scale, ly.down_scales_dev, down_raw, hs, is, nullptr));
            RC_OK(rcpp_fp32_to_fp16(down_raw, down_fp16, hs, nullptr));
            RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, down_fp16, hs, nullptr));
        }

        // Final norm reads FP32 residual, emits FP16 → tied LM head GEMV.
        RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, m.final_norm_weight_dev, normed,
                                            m.rms_norm_eps, hs, nullptr));
        RC_OK(rcpp_fp16_gemv(m.embedding_dev, normed, logits, V, hs, nullptr));

        // Greedy sample
        RC_OK(rcpp_argmax_fp32(logits, next_tok_dev, V, nullptr));
        int next_tok;
        HIP_OK(hipMemcpy(&next_tok, next_tok_dev, 4, hipMemcpyDeviceToHost));
        return next_tok;
    };

    // ---- Prefill: feed prompt tokens[0..prompt_len-2] through the cache,
    //      producing the logits for position prompt_len-1 at the last step.
    //      Then generate num_tokens new tokens greedily.
    double prefill_ms = 0.0;
    int next_tok = 0;
    for (int step = 0; step < prompt_len; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();
        next_tok = forward_token(prompt_ids[step], step);
        HIP_OK(hipDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        prefill_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    fprintf(stderr, "[bitnet_decode] prefill %d tok in %.2f ms (%.1f tok/s)\n",
            prompt_len, prefill_ms,
            prompt_len > 0 ? 1000.0 * prompt_len / prefill_ms : 0.0);

    // Stop tokens for LLaMA-3-family models (what BitNet ships with):
    //   128001 = <|end_of_text|>
    //   128009 = <|eot_id|>  (chat turn boundary)
    // Picking these up early means the CLI terminates naturally on
    // model-generated sentence endings instead of burning through num_tokens.
    const int stop_a = 128001, stop_b = 128009;
    std::vector<int> generated;
    generated.reserve(num_tokens);
    printf("[bitnet_decode] tokens:");
    double decode_ms = 0.0;
    int cur_tok = next_tok;
    bool hit_eos = false;
    for (int step = 0; step < num_tokens; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();
        next_tok = forward_token(cur_tok, prompt_len + step);
        HIP_OK(hipDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        decode_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf(" %d", next_tok);
        fflush(stdout);
        generated.push_back(next_tok);
        if (next_tok == stop_a || next_tok == stop_b) { hit_eos = true; break; }
        cur_tok = next_tok;
    }
    printf("\n");
    if (hit_eos) {
        fprintf(stderr, "[bitnet_decode] EOS (%d) after %zu new tokens\n",
                next_tok, generated.size());
    }
    if (num_tokens > 0) {
        printf("[bitnet_decode] decode %d tok in %.2f ms  (%.2f ms/tok, %.1f tok/s)\n",
               num_tokens, decode_ms, decode_ms/num_tokens, 1000.0 * num_tokens / decode_ms);
    }

    // Detokenize prompt + generated tokens back to text.
    rcpp_tokenizer_t* dec_tok = nullptr;
    if (rcpp_tokenizer_load(tok_path, &dec_tok) == RCPP_OK) {
        std::vector<int> full;
        full.reserve(prompt_len + generated.size());
        full.insert(full.end(), prompt_ids.begin(), prompt_ids.end());
        full.insert(full.end(), generated.begin(), generated.end());
        std::vector<char> text(16 * 1024);
        size_t tlen = 0;
        rcpp_tokenizer_decode(dec_tok, full.data(), full.size(),
                              text.data(), text.size(), &tlen);
        if (tlen > text.size()) tlen = text.size();
        printf("[bitnet_decode] text:\n%.*s\n", (int)tlen, text.data());
        rcpp_tokenizer_free(dec_tok);
    }

    // Cleanup
    for (int l = 0; l < L; ++l) { hipFree(K_caches[l]); hipFree(V_caches[l]); }
    rcpp_bitnet_free(&m);
    return 0;
}
