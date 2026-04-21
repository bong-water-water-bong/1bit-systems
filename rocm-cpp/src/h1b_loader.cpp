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

// Read FP32 from disk, cast to FP16, upload to device (the .h1b format
// stores norms and embeddings as float32; kernels consume FP16).
int read_fp32_as_fp16(std::ifstream& f, size_t n, __half** out) {
    std::vector<float> src(n);
    f.read(reinterpret_cast<char*>(src.data()), n * sizeof(float));
    std::vector<_Float16> dst(n);
    for (size_t i = 0; i < n; ++i) dst[i] = (_Float16)src[i];
    HIP_CHECK(hipMalloc(out, n * sizeof(_Float16)));
    HIP_CHECK(hipMemcpy(*out, dst.data(), n * sizeof(_Float16), hipMemcpyHostToDevice));
    return 0;
}

// Skip a block of float32 values we don't need (e.g., the duplicated
// attn_sub_norm copies the exporter writes 4× for legacy reasons).
void skip_fp32(std::ifstream& f, size_t n) {
    f.seekg(n * sizeof(float), std::ios::cur);
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

// Sherry v3: uint8[rows * cols * 5 / 32] + float[rows] scales. 1.25 bpw.
// cols must be a multiple of 32.
int read_ternary_sherry(std::ifstream& f, int rows, int cols, void** packed_out, void** scales_out) {
    if (cols % 32 != 0) {
        fprintf(stderr, "[rocm-cpp] v3 load: cols=%d not divisible by 32\n", cols);
        return -1;
    }
    const size_t row_bytes = (size_t)cols * 5 / 32;
    std::vector<uint8_t> packed((size_t)rows * row_bytes);
    f.read(reinterpret_cast<char*>(packed.data()), packed.size());
    std::vector<float> scales(rows);
    f.read(reinterpret_cast<char*>(scales.data()), rows * sizeof(float));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(packed_out), packed.size()));
    HIP_CHECK(hipMemcpy(*packed_out, packed.data(), packed.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(scales_out), rows * sizeof(float)));
    HIP_CHECK(hipMemcpy(*scales_out, scales.data(), rows * sizeof(float), hipMemcpyHostToDevice));
    return 0;
}

// TQ1 v4: uint8[rows * cols_padded / 5] + float[rows] scales. 1.6 bpw.
// cols is padded up to multiple of 20 (requantizer handles the padding).
int read_ternary_tq1(std::ifstream& f, int rows, int cols, void** packed_out, void** scales_out) {
    const int cols_padded = ((cols + 19) / 20) * 20;
    const size_t row_bytes = (size_t)cols_padded / 5;
    std::vector<uint8_t> packed((size_t)rows * row_bytes);
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
    if (version != 1 && version != 2 && version != 3 && version != 4) {
        fprintf(stderr, "Unsupported .h1b version: %d\n", version);
        return RCPP_UNSUPPORTED;
    }
    out_model->format_version = version;
    const bool use_sherry = (version == 3);
    const bool use_tq1    = (version == 4);

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
    // cfg[8] — formerly "reserved", now flag bits. H1B_FLAG_* live in
    // include/rocm_cpp/bitnet_model.h. Legacy v1/v2 writers zero this field.
    out_model->flags             = static_cast<unsigned int>(cfg[8]);

    const bool sherry_fp16 = use_sherry
        && (out_model->flags & H1B_FLAG_SHERRY_FP16) != 0;

    // Resolve the dispatch tag once at load — the inference loop reads this
    // rather than replaying the (version, flags) logic on every GEMV call.
    if (use_tq1) {
        out_model->weight_format = RCPP_WEIGHT_FORMAT_TQ1;
    } else if (sherry_fp16) {
        out_model->weight_format = RCPP_WEIGHT_FORMAT_SHERRY_FP16;
    } else if (use_sherry) {
        out_model->weight_format = RCPP_WEIGHT_FORMAT_SHERRY_I8;
    } else {
        out_model->weight_format = RCPP_WEIGHT_FORMAT_HALO_V2;
    }

    // v1 .h1b files don't carry rope_theta / rms_norm_eps — fall back to the
    // BitNet-b1.58-2B-4T defaults. v2 adds them explicitly so variants with
    // different values (1B / 4B / 8B checkpoints, other BitNet derivatives)
    // Just Work without a code change.
    if (version >= 2) {
        float extras[2] = {0.0f, 0.0f};
        f.read(reinterpret_cast<char*>(extras), sizeof(extras));
        out_model->rope_theta   = extras[0] > 0 ? extras[0] : 500000.0f;
        out_model->rms_norm_eps = extras[1] > 0 ? extras[1] : 1e-5f;
    } else {
        out_model->rope_theta   = 500000.0f;
        out_model->rms_norm_eps = 1e-5f;
    }
    if (sherry_fp16) {
        fprintf(stderr, "[rocm-cpp] .h1b v3 + SHERRY_FP16 flag — dispatching through fp16-in/fp16-out sherry_ternary_gemv_launch.\n");
    } else if (use_sherry) {
        fprintf(stderr, "[rocm-cpp] .h1b v3 (Sherry 1.25 bpw, int8-act) — dispatching ternary GEMVs through sherry decoder.\n");
    }
    if (use_tq1) {
        fprintf(stderr, "[rocm-cpp] .h1b v4 (TQ1 base-3, 1.6 bpw, lossless) — dispatching through tq1-halo kernel.\n");
    }
    fprintf(stderr, "[rocm-cpp] .h1b v%d flags=0x%x: rope_theta=%.1f rms_norm_eps=%.1e\n",
            version, out_model->flags, out_model->rope_theta, out_model->rms_norm_eps);

    const int hs  = out_model->hidden_size;
    const int is_ = out_model->intermediate_size;
    const int nh  = out_model->num_heads;
    const int nkv = out_model->num_kv_heads;
    const int hd  = hs / nh;

    fprintf(stderr, "[rocm-cpp] loading .h1b: hs=%d is=%d L=%d nh=%d nkv=%d hd=%d vocab=%d\n",
            hs, is_, out_model->num_layers, nh, nkv, hd, out_model->vocab_size);

    // Embeddings + final norm (both stored as FP32 on disk → cast to FP16)
    if (read_fp32_as_fp16(f, (size_t)out_model->vocab_size * hs,
                          reinterpret_cast<__half**>(&out_model->embedding_dev)) != 0) return RCPP_HIP_ERROR;
    if (read_fp32_as_fp16(f, hs, reinterpret_cast<__half**>(&out_model->final_norm_weight_dev)) != 0) return RCPP_HIP_ERROR;

    // Layers
    out_model->layers = static_cast<rcpp_bitnet_layer_t*>(
        std::calloc(out_model->num_layers, sizeof(rcpp_bitnet_layer_t)));
    if (!out_model->layers) return RCPP_INTERNAL;

    for (int l = 0; l < out_model->num_layers; ++l) {
        rcpp_bitnet_layer_t& L = out_model->layers[l];

        // Sequence written by scripts/export_base_h1b.py (float32):
        //   input_layernorm           [hs]
        //   post_attention_layernorm  [hs]
        //   attn_sub_norm             [hs] × 4 (exporter duplicates for q/k/v/o slots)
        //   ffn_sub_norm[:hs]         [hs] × 2 (truncated gate/up slots)
        //   ffn_sub_norm              [is] × 1 (full, used before down_proj)
        if (read_fp32_as_fp16(f, hs, reinterpret_cast<__half**>(&L.input_norm_dev))     != 0) return RCPP_HIP_ERROR;
        if (read_fp32_as_fp16(f, hs, reinterpret_cast<__half**>(&L.post_attn_norm_dev)) != 0) return RCPP_HIP_ERROR;
        // attn_sub_norm: keep the first copy, skip three duplicates.
        if (read_fp32_as_fp16(f, hs, reinterpret_cast<__half**>(&L.attn_sub_norm_dev))  != 0) return RCPP_HIP_ERROR;
        skip_fp32(f, hs);
        skip_fp32(f, hs);
        skip_fp32(f, hs);
        // Two [hs] truncated ffn_sub copies (gate/up slots) — discarded.
        skip_fp32(f, hs);
        skip_fp32(f, hs);
        // Full [is] ffn_sub_norm — kept, applied on silu_out before down_proj.
        if (read_fp32_as_fp16(f, is_, reinterpret_cast<__half**>(&L.ffn_sub_norm_dev))  != 0) return RCPP_HIP_ERROR;

        // 7 ternary linear layers: Q K V O gate up down
        auto rt = use_tq1    ? read_ternary_tq1
               : use_sherry ? read_ternary_sherry
               :              read_ternary;
        if (rt(f, nh * hd,  hs,    &L.q_packed_dev,    reinterpret_cast<void**>(&L.q_scales_dev))    != 0) return RCPP_HIP_ERROR;
        if (rt(f, nkv * hd, hs,    &L.k_packed_dev,    reinterpret_cast<void**>(&L.k_scales_dev))    != 0) return RCPP_HIP_ERROR;
        if (rt(f, nkv * hd, hs,    &L.v_packed_dev,    reinterpret_cast<void**>(&L.v_scales_dev))    != 0) return RCPP_HIP_ERROR;
        if (rt(f, hs,       nh*hd, &L.o_packed_dev,    reinterpret_cast<void**>(&L.o_scales_dev))    != 0) return RCPP_HIP_ERROR;
        if (rt(f, is_,      hs,    &L.gate_packed_dev, reinterpret_cast<void**>(&L.gate_scales_dev)) != 0) return RCPP_HIP_ERROR;
        if (rt(f, is_,      hs,    &L.up_packed_dev,   reinterpret_cast<void**>(&L.up_scales_dev))   != 0) return RCPP_HIP_ERROR;
        if (rt(f, hs,       is_,   &L.down_packed_dev, reinterpret_cast<void**>(&L.down_scales_dev)) != 0) return RCPP_HIP_ERROR;
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
        f(L.input_norm_dev);  f(L.post_attn_norm_dev);
        f(L.attn_sub_norm_dev); f(L.ffn_sub_norm_dev);
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
