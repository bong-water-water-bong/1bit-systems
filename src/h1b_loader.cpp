// Minimal .h1b model loader — pure C++ + HIP, no MLX, no halo-1bit deps.
//
// Reads the .h1b format that halo-1bit writes (magic "H1B", v1, 9-int32 config,
// then embedding + per-layer weights). Uploads everything to the GPU and
// returns raw device pointers the inference loop can feed to rocm-cpp kernels.

#include "rocm_cpp/bitnet_model.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#define HIP_CHECK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP %d %s:%d\n",_s,__FILE__,__LINE__); return -1;}} while(0)

namespace {

// Read a block of FP16 values into a device buffer.
int read_fp16(std::ifstream& f, size_t n, __half** out) {
    std::vector<_Float16> tmp(n);
    f.read(reinterpret_cast<char*>(tmp.data()), n * sizeof(_Float16));
    HIP_CHECK(hipMalloc(out, n * sizeof(_Float16)));
    HIP_CHECK(hipMemcpy(*out, tmp.data(), n * sizeof(_Float16), hipMemcpyHostToDevice));
    return 0;
}

// Read a packed ternary weight (halo-1bit format: uint8[rows, (cols+3)/4] + float[rows] scales).
int read_ternary(std::ifstream& f, int rows, int cols, void** packed_out, void** scales_out) {
    const int packed_cols = (cols + 3) / 4;
    std::vector<uint8_t> packed((size_t)rows * packed_cols);
    f.read(reinterpret_cast<char*>(packed.data()), packed.size());
    std::vector<float> scales(rows);
    f.read(reinterpret_cast<char*>(scales.data()), rows * sizeof(float));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(packed_out), packed.size()));
    HIP_CHECK(hipMemcpy(*packed_out, packed.data(), packed.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(scales_out), rows * sizeof(float)));
    HIP_CHECK(hipMemcpy(*scales_out, scales.data(), rows * sizeof(float), hipMemcpyHostToDevice));
    return 0;
}

}  // namespace

extern "C" rcpp_status_t
rcpp_bitnet_load_h1b(const char* path, rcpp_bitnet_model_t* out_model) {
    if (!path || !out_model) return RCPP_INVALID_ARG;
    std::memset(out_model, 0, sizeof(*out_model));

    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return RCPP_INVALID_ARG; }

    char magic[4];
    f.read(magic, 4);
    if (std::strncmp(magic, "H1B", 3) != 0) {
        fprintf(stderr, "Bad .h1b magic\n");
        return RCPP_INVALID_ARG;
    }

    int32_t version;
    f.read(reinterpret_cast<char*>(&version), 4);
    if (version != 1) {
        fprintf(stderr, "Unsupported .h1b version: %d\n", version);
        return RCPP_UNSUPPORTED;
    }

    int32_t cfg[9];
    f.read(reinterpret_cast<char*>(cfg), sizeof(cfg));
    out_model->hidden_size       = cfg[0];
    out_model->intermediate_size = cfg[1];
    out_model->num_layers        = cfg[2];
    out_model->num_heads         = cfg[3];
    out_model->num_kv_heads      = cfg[4];
    out_model->vocab_size        = cfg[5];
    out_model->max_seq_len       = cfg[6];
    out_model->tie_embeddings    = cfg[7];
    // cfg[8] reserved
    // Rope theta / norm eps come from config.json in halo-1bit; we hard-code to
    // BitNet-2B-4T defaults since the .h1b binary doesn't store them.
    out_model->rope_theta   = 500000.0f;
    out_model->rms_norm_eps = 1e-5f;

    const int hs  = out_model->hidden_size;
    const int is_ = out_model->intermediate_size;
    const int nh  = out_model->num_heads;
    const int nkv = out_model->num_kv_heads;
    const int hd  = hs / nh;

    fprintf(stderr, "[rocm-cpp] loading .h1b: hs=%d is=%d L=%d nh=%d nkv=%d hd=%d vocab=%d\n",
            hs, is_, out_model->num_layers, nh, nkv, hd, out_model->vocab_size);

    // Embeddings + final norm
    if (read_fp16(f, (size_t)out_model->vocab_size * hs,
                  reinterpret_cast<__half**>(&out_model->embedding_dev)) != 0) return RCPP_HIP_ERROR;
    if (read_fp16(f, hs, reinterpret_cast<__half**>(&out_model->final_norm_weight_dev)) != 0) return RCPP_HIP_ERROR;

    // Layers
    out_model->layers = static_cast<rcpp_bitnet_layer_t*>(
        std::calloc(out_model->num_layers, sizeof(rcpp_bitnet_layer_t)));
    if (!out_model->layers) return RCPP_INTERNAL;

    for (int l = 0; l < out_model->num_layers; ++l) {
        rcpp_bitnet_layer_t& L = out_model->layers[l];

        // 4 RMSNorm weights: input_norm, post_attn_norm, q_ln, k_ln
        if (read_fp16(f, hs, reinterpret_cast<__half**>(&L.input_norm_dev))     != 0) return RCPP_HIP_ERROR;
        if (read_fp16(f, hs, reinterpret_cast<__half**>(&L.post_attn_norm_dev)) != 0) return RCPP_HIP_ERROR;
        if (read_fp16(f, hs, reinterpret_cast<__half**>(&L.q_ln_dev))           != 0) return RCPP_HIP_ERROR;
        if (read_fp16(f, hs, reinterpret_cast<__half**>(&L.k_ln_dev))           != 0) return RCPP_HIP_ERROR;

        // 7 ternary linear layers: Q K V O gate up down
        if (read_ternary(f, nh * hd,  hs,    &L.q_packed_dev,    reinterpret_cast<void**>(&L.q_scales_dev))    != 0) return RCPP_HIP_ERROR;
        if (read_ternary(f, nkv * hd, hs,    &L.k_packed_dev,    reinterpret_cast<void**>(&L.k_scales_dev))    != 0) return RCPP_HIP_ERROR;
        if (read_ternary(f, nkv * hd, hs,    &L.v_packed_dev,    reinterpret_cast<void**>(&L.v_scales_dev))    != 0) return RCPP_HIP_ERROR;
        if (read_ternary(f, hs,       nh*hd, &L.o_packed_dev,    reinterpret_cast<void**>(&L.o_scales_dev))    != 0) return RCPP_HIP_ERROR;
        if (read_ternary(f, is_,      hs,    &L.gate_packed_dev, reinterpret_cast<void**>(&L.gate_scales_dev)) != 0) return RCPP_HIP_ERROR;
        if (read_ternary(f, is_,      hs,    &L.up_packed_dev,   reinterpret_cast<void**>(&L.up_scales_dev))   != 0) return RCPP_HIP_ERROR;
        if (read_ternary(f, hs,       is_,   &L.down_packed_dev, reinterpret_cast<void**>(&L.down_scales_dev)) != 0) return RCPP_HIP_ERROR;
    }

    // LM head: tied to embedding by default in BitNet. If not tied, halo-1bit
    // stores a separate packed ternary in .h1b; we support only the tied case
    // for this MVP.
    if (!out_model->tie_embeddings) {
        // TODO: read separate LM head ternary
        fprintf(stderr, "[rocm-cpp] WARN: untied LM head not supported in MVP loader\n");
    }

    return RCPP_OK;
}

extern "C" void
rcpp_bitnet_free(rcpp_bitnet_model_t* m) {
    if (!m) return;
    auto f = [](void* p) { if (p) hipFree(p); };
    f(m->embedding_dev);
    f(m->final_norm_weight_dev);
    for (int l = 0; l < m->num_layers; ++l) {
        rcpp_bitnet_layer_t& L = m->layers[l];
        f(L.input_norm_dev);  f(L.post_attn_norm_dev); f(L.q_ln_dev); f(L.k_ln_dev);
        f(L.q_packed_dev);    f(L.q_scales_dev);
        f(L.k_packed_dev);    f(L.k_scales_dev);
        f(L.v_packed_dev);    f(L.v_scales_dev);
        f(L.o_packed_dev);    f(L.o_scales_dev);
        f(L.gate_packed_dev); f(L.gate_scales_dev);
        f(L.up_packed_dev);   f(L.up_scales_dev);
        f(L.down_packed_dev); f(L.down_scales_dev);
    }
    std::free(m->layers);
    std::memset(m, 0, sizeof(*m));
}
