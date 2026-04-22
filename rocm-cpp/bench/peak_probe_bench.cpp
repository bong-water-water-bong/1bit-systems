// peak_probe_bench.cpp — driver for rcpp_peak_probe_launch{_i8}.
//
// Measures WMMA compute ceiling (FP16 → FP32, INT8 → INT32) by launching
// the register-resident probe kernel in librocm_cpp.so. No hipBLAS, no
// global-memory hot path — just the WMMA pipe. Reports TFLOPS / TOPS.
//
// Build:
//   hipcc -O3 --offload-arch=gfx1151 peak_probe_bench.cpp -o peak_probe_bench \
//         -L/usr/local/lib -lrocm_cpp
//
// Usage:
//   ./peak_probe_bench [--n-blocks N] [--n-inner M] [--json]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                                    \
    do {                                                                                   \
        hipError_t _e = (expr);                                                            \
        if (_e != hipSuccess) {                                                            \
            std::fprintf(stderr, "HIP error %s at %s:%d: %s\n",                            \
                         hipGetErrorString(_e), __FILE__, __LINE__, #expr);                \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

// Symbols exported by librocm_cpp.so (wmma_peak_probe.hip).
extern "C" void rcpp_peak_probe_launch(
    const void* A, const void* B, void* sink, int n_blocks, int n_inner, void* stream);
extern "C" void rcpp_peak_probe_launch_i8(
    const void* A, const void* B, void* sink, int n_blocks, int n_inner, void* stream);

namespace {

// One 16x16x16 WMMA = 2 * 16*16*16 = 8192 MAC-FLOPs.
// Per wave per inner iter: 4x4 = 16 WMMAs = 131,072 FLOPs.
// Per launch: n_blocks * n_inner * 131,072. One wave per block.
constexpr uint64_t FLOPS_PER_WAVE_ITER = 4ULL * 4ULL * 2ULL * 16ULL * 16ULL * 16ULL;

struct Stats {
    double median_ms;
    double min_ms;
    double tflops_median;
    double tflops_peak;  // from min_ms
};

Stats run_probe(bool int8_variant,
                const void* d_A, const void* d_B, void* d_sink,
                int n_blocks, int n_inner,
                int n_warmup, int n_timed)
{
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    auto launch = [&]() {
        if (int8_variant) {
            rcpp_peak_probe_launch_i8(d_A, d_B, d_sink, n_blocks, n_inner, (void*)stream);
        } else {
            rcpp_peak_probe_launch(d_A, d_B, d_sink, n_blocks, n_inner, (void*)stream);
        }
    };

    for (int i = 0; i < n_warmup; ++i) launch();
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> ms_samples;
    ms_samples.reserve(n_timed);

    for (int i = 0; i < n_timed; ++i) {
        HIP_CHECK(hipStreamSynchronize(stream));
        auto t0 = std::chrono::steady_clock::now();
        launch();
        HIP_CHECK(hipStreamSynchronize(stream));
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ms_samples.push_back(ms);
    }

    HIP_CHECK(hipStreamDestroy(stream));

    std::vector<double> sorted = ms_samples;
    std::sort(sorted.begin(), sorted.end());
    double median_ms = sorted[sorted.size() / 2];
    double min_ms = sorted.front();

    uint64_t total_ops =
        static_cast<uint64_t>(n_blocks) * static_cast<uint64_t>(n_inner) * FLOPS_PER_WAVE_ITER;

    Stats s;
    s.median_ms = median_ms;
    s.min_ms = min_ms;
    s.tflops_median = (double)total_ops / (median_ms * 1e-3) / 1e12;
    s.tflops_peak   = (double)total_ops / (min_ms    * 1e-3) / 1e12;
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    int n_blocks = 1024;
    int n_inner  = 2000;
    int n_warmup = 3;
    int n_timed  = 5;
    bool emit_json = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--n-blocks" && i + 1 < argc) n_blocks = std::atoi(argv[++i]);
        else if (a == "--n-inner" && i + 1 < argc) n_inner = std::atoi(argv[++i]);
        else if (a == "--n-warmup" && i + 1 < argc) n_warmup = std::atoi(argv[++i]);
        else if (a == "--n-timed" && i + 1 < argc) n_timed = std::atoi(argv[++i]);
        else if (a == "--json") emit_json = true;
    }

    int device = 0;
    HIP_CHECK(hipSetDevice(device));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    // Seed buffers only need enough for one wave (threadIdx.x * 4 + i = up to 127
    // * WMMA_K_ELEMS up to 16 = 2048 elements). Give them a generous slab.
    const size_t seed_elems = 4096;
    __half* d_A_h16 = nullptr;
    __half* d_B_h16 = nullptr;
    int8_t* d_A_i8  = nullptr;
    int8_t* d_B_i8  = nullptr;
    float* d_sink_f = nullptr;
    int32_t* d_sink_i = nullptr;

    HIP_CHECK(hipMalloc(&d_A_h16, seed_elems * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_B_h16, seed_elems * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_A_i8,  seed_elems * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&d_B_i8,  seed_elems * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&d_sink_f, (size_t)n_blocks * 32 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_sink_i, (size_t)n_blocks * 32 * sizeof(int32_t)));

    // Seed: small values to avoid NaN/Inf in fp16 accumulation; INT8 small too.
    std::vector<__half> h_A_h16(seed_elems), h_B_h16(seed_elems);
    std::vector<int8_t> h_A_i8(seed_elems), h_B_i8(seed_elems);
    for (size_t i = 0; i < seed_elems; ++i) {
        float v = (float)((i % 7) - 3) * 0.0625f;
        h_A_h16[i] = __float2half(v);
        h_B_h16[i] = __float2half(v * 0.5f);
        h_A_i8[i]  = (int8_t)((i % 5) - 2);
        h_B_i8[i]  = (int8_t)((i % 3) - 1);
    }
    HIP_CHECK(hipMemcpy(d_A_h16, h_A_h16.data(), seed_elems * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_h16, h_B_h16.data(), seed_elems * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_A_i8,  h_A_i8.data(),  seed_elems * sizeof(int8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_i8,  h_B_i8.data(),  seed_elems * sizeof(int8_t), hipMemcpyHostToDevice));

    Stats f16 = run_probe(false, d_A_h16, d_B_h16, d_sink_f, n_blocks, n_inner, n_warmup, n_timed);
    Stats i8  = run_probe(true,  d_A_i8,  d_B_i8,  d_sink_i, n_blocks, n_inner, n_warmup, n_timed);

    const uint64_t total_ops =
        (uint64_t)n_blocks * (uint64_t)n_inner * FLOPS_PER_WAVE_ITER;

    if (emit_json) {
        // Machine-readable line. Caller assembles the outer JSON.
        std::printf(
            "{"
            "\"arch\":\"%s\","
            "\"device\":\"%s\","
            "\"n_blocks\":%d,\"n_inner\":%d,\"n_warmup\":%d,\"n_timed\":%d,"
            "\"total_ops_per_launch\":%llu,"
            "\"fp16\":{\"median_ms\":%.6f,\"min_ms\":%.6f,\"tflops_median\":%.4f,\"tflops_peak\":%.4f},"
            "\"int8\":{\"median_ms\":%.6f,\"min_ms\":%.6f,\"tops_median\":%.4f,\"tops_peak\":%.4f}"
            "}\n",
            prop.gcnArchName, prop.name,
            n_blocks, n_inner, n_warmup, n_timed,
            (unsigned long long)total_ops,
            f16.median_ms, f16.min_ms, f16.tflops_median, f16.tflops_peak,
            i8.median_ms,  i8.min_ms,  i8.tflops_median,  i8.tflops_peak);
    } else {
        std::printf("device: %s (%s)\n", prop.name, prop.gcnArchName);
        std::printf("n_blocks=%d n_inner=%d  warmup=%d timed=%d\n",
                    n_blocks, n_inner, n_warmup, n_timed);
        std::printf("total FLOPs per launch: %llu (%.3f G)\n",
                    (unsigned long long)total_ops, total_ops / 1e9);
        std::printf("FP16:  median %.3f ms  min %.3f ms  =>  %.2f TFLOPS (median)  %.2f TFLOPS (peak)\n",
                    f16.median_ms, f16.min_ms, f16.tflops_median, f16.tflops_peak);
        std::printf("INT8:  median %.3f ms  min %.3f ms  =>  %.2f TOPS (median)  %.2f TOPS (peak)\n",
                    i8.median_ms, i8.min_ms, i8.tflops_median, i8.tflops_peak);
    }

    HIP_CHECK(hipFree(d_A_h16));
    HIP_CHECK(hipFree(d_B_h16));
    HIP_CHECK(hipFree(d_A_i8));
    HIP_CHECK(hipFree(d_B_i8));
    HIP_CHECK(hipFree(d_sink_f));
    HIP_CHECK(hipFree(d_sink_i));
    return 0;
}
