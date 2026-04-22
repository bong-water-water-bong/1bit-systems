// attn_fd_sweep — Split-KV Flash-Decoding attention sweep for
// {512, 1024, 2048, 4096, 8192} × {single-block, split-KV FD} on a single arch.
//
// Emits a JSON object on stdout keyed by arch:
//   { "<arch>": { "512": {single_ms, fd_ms, speedup, max_abs_diff}, ... } }
//
// Driver (host) merges results from both arches into the final JSON.
//
// Usage: attn_fd_sweep <arch-label>
//   arch-label: e.g. "gfx1151" or "gfx1201" — used as the JSON top-level key.
//
// 3 warmup + 5 timed per cell; reports median wall-ns per call.
// Correctness: byte-exact FP16 compare vs single-block reference.

#include "rocm_cpp/ck_gemm.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { auto _s = (x); if (_s != hipSuccess) { \
    fprintf(stderr, "HIP err %d %s:%d\n", _s, __FILE__, __LINE__); std::exit(1); } } while (0)
#define RCPP_CHECK(x) do { auto _s = (x); if (_s != RCPP_OK) { \
    fprintf(stderr, "rcpp err %d %s:%d\n", (int)_s, __FILE__, __LINE__); std::exit(1); } } while (0)

struct HipBuf {
    void* p = nullptr;
    explicit HipBuf(size_t bytes) { HIP_CHECK(hipMalloc(&p, bytes)); }
    ~HipBuf() { if (p) hipFree(p); }
    HipBuf(const HipBuf&) = delete;
    HipBuf& operator=(const HipBuf&) = delete;
};

// Median of an in-place-sortable vector (returns by value; swaps input).
static double median_ns(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1u) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Time N_TIMED launches of fn back-to-back on the default stream, return per-launch ns.
// Runs 3 warmups outside the timed region.
template <typename Fn>
static double time_single_ns(Fn fn) {
    const int N_WARM = 3;
    for (int i = 0; i < N_WARM; ++i) fn();
    HIP_CHECK(hipDeviceSynchronize());

    auto t0 = std::chrono::steady_clock::now();
    fn();
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: attn_fd_sweep <arch-label>\n");
        return 1;
    }
    const std::string arch = argv[1];

    // BitNet-2B-4T shape.
    const int NH  = 20;
    const int NKV = 5;
    const int HD  = 128;
    const std::vector<int> seqs = {512, 1024, 2048, 4096, 8192};
    const int N_TIMED = 5;

    std::mt19937 rng(0xC0DEF00D);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Build JSON incrementally — simple hand-written, no deps beyond stdlib.
    std::string json_body;
    json_body.reserve(1024);

    for (size_t si = 0; si < seqs.size(); ++si) {
        const int L = seqs[si];

        const size_t Q_bytes   = (size_t)NH  * HD * sizeof(__half);
        const size_t KV_bytes  = (size_t)L * NKV * HD * sizeof(__half);
        const size_t OUT_bytes = (size_t)NH  * HD * sizeof(__half);

        std::vector<__half> h_Q(NH  * HD);
        std::vector<__half> h_K(L * NKV * HD);
        std::vector<__half> h_V(L * NKV * HD);
        for (auto& x : h_Q) x = __float2half(nd(rng));
        for (auto& x : h_K) x = __float2half(nd(rng));
        for (auto& x : h_V) x = __float2half(nd(rng));

        HipBuf d_Q(Q_bytes), d_K(KV_bytes), d_V(KV_bytes);
        HipBuf d_out_single(OUT_bytes), d_out_fd(OUT_bytes);

        HIP_CHECK(hipMemcpy(d_Q.p, h_Q.data(), Q_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_K.p, h_K.data(), KV_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_V.p, h_V.data(), KV_bytes, hipMemcpyHostToDevice));

        const float scale = 1.0f / std::sqrt((float)HD);

        // --- Correctness (one-shot) ---
        RCPP_CHECK(rcpp_kv_cache_attn_decode(
            d_Q.p, d_K.p, d_V.p, d_out_single.p,
            NH, NKV, HD, L, scale, nullptr));
        RCPP_CHECK(rcpp_kv_cache_attn_decode_fd(
            d_Q.p, d_K.p, d_V.p, d_out_fd.p,
            NH, NKV, HD, L, scale, nullptr));
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<__half> h_out_single(NH * HD), h_out_fd(NH * HD);
        HIP_CHECK(hipMemcpy(h_out_single.data(), d_out_single.p, OUT_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_out_fd.data(),     d_out_fd.p,     OUT_bytes, hipMemcpyDeviceToHost));

        double max_abs = 0.0;
        for (size_t i = 0; i < h_out_single.size(); ++i) {
            double a = (double)(float)h_out_single[i];
            double b = (double)(float)h_out_fd[i];
            double d = std::fabs(a - b);
            if (d > max_abs) max_abs = d;
        }

        // --- Timing: N_TIMED samples each, median ---
        std::vector<double> single_samples(N_TIMED), fd_samples(N_TIMED);

        // Warm single once, then sample.
        auto single_fn = [&] {
            RCPP_CHECK(rcpp_kv_cache_attn_decode(
                d_Q.p, d_K.p, d_V.p, d_out_single.p,
                NH, NKV, HD, L, scale, nullptr));
        };
        auto fd_fn = [&] {
            RCPP_CHECK(rcpp_kv_cache_attn_decode_fd(
                d_Q.p, d_K.p, d_V.p, d_out_fd.p,
                NH, NKV, HD, L, scale, nullptr));
        };

        // 3 warmups each, then 5 individually-timed launches.
        for (int i = 0; i < 3; ++i) single_fn();
        HIP_CHECK(hipDeviceSynchronize());
        for (int i = 0; i < N_TIMED; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            single_fn();
            HIP_CHECK(hipDeviceSynchronize());
            auto t1 = std::chrono::steady_clock::now();
            single_samples[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }

        for (int i = 0; i < 3; ++i) fd_fn();
        HIP_CHECK(hipDeviceSynchronize());
        for (int i = 0; i < N_TIMED; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            fd_fn();
            HIP_CHECK(hipDeviceSynchronize());
            auto t1 = std::chrono::steady_clock::now();
            fd_samples[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }

        double single_ns_med = median_ns(single_samples);
        double fd_ns_med     = median_ns(fd_samples);
        double single_ms     = single_ns_med / 1.0e6;
        double fd_ms         = fd_ns_med     / 1.0e6;
        double speedup       = single_ns_med / fd_ns_med;

        // Per-L stderr progress line for humans.
        fprintf(stderr,
                "[%s L=%5d] single=%8.3f us  fd=%8.3f us  speedup=%.2fx  max|diff|=%.6f\n",
                arch.c_str(), L, single_ns_med / 1.0e3, fd_ns_med / 1.0e3, speedup, max_abs);

        char cell[512];
        std::snprintf(cell, sizeof(cell),
            "    \"%d\": {\"single_ms\": %.6f, \"fd_ms\": %.6f, "
            "\"speedup\": %.6f, \"max_abs_diff\": %.6f}",
            L, single_ms, fd_ms, speedup, max_abs);
        if (!json_body.empty()) json_body += ",\n";
        json_body += cell;
    }

    // Final JSON on stdout.
    std::printf("{\n  \"%s\": {\n%s\n  }\n}\n", arch.c_str(), json_body.c_str());
    return 0;
}
