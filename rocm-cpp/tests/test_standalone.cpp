// Phase 1 correctness: standalone HIP kernel vs CK kernel on the SAME packed
// weights. If both produce the same output (within FP16 rounding), the strip
// is valid and we can wire the standalone path into the C API.

#include "rocm_cpp/ck_gemm.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" void rcpp_standalone_launch                (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_lds            (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_wmma           (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_wmma_2wave     (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_wmma_4x        (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_wmma_2x4       (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_wmma_2x4_ldsB  (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_wmma_4x4       (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_wmma_4x4_cached(const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_wmma_4x4_vec   (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_wmma_2x2wave   (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_lds_pingpong   (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_4c_pipelined   (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_fp16b          (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_decode_pk_i4_to_fp16_launch      (const void*, void*, int, int, void*);

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP err %d %s:%d\n",_s,__FILE__,__LINE__); std::abort();}} while(0)
#define RC_OK(e)  do { auto _s=(e); if(_s!=RCPP_OK){fprintf(stderr,"rcpp err %d %s:%d\n",(int)_s,__FILE__,__LINE__); std::abort();}} while(0)

int main(int argc, char** argv) {
    // 128 chosen to land on every tile shape cleanly: 64x64 (4g/4h/4i/fp16b),
    // 64x32 (pingpong), 32x64 (4c/4j/4d), 16x64 (4x).
    int M = 128, N = 128, K = 512;
    if(argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }

    printf("=== standalone (Phase 1 naive) vs CK — same packed weights ===\n");
    printf("Shape: M=%d N=%d K=%d\n", M, N, K);

    std::mt19937 rng(0x1b1fe4e4);  // "1bit fever" bytes
    std::uniform_real_distribution<float> rd(-0.25f, 0.25f);
    std::uniform_int_distribution<int>    rt(-1, 1);

    // Generate random FP16 A + ternary B, pack B via the C API.
    std::vector<_Float16> A((size_t)M * K);
    for(auto& v : A) v = (_Float16)rd(rng);

    std::vector<int8_t> B_ternary((size_t)K * N);
    for(auto& v : B_ternary) v = (int8_t)rt(rng);

    std::vector<int8_t> B_packed((size_t)K * N / 2);
    RC_OK(rcpp_ternary_pack_pk_i4(B_ternary.data(), B_packed.data(), K, N));

    // Device buffers
    _Float16* dA = nullptr;
    int8_t*   dB = nullptr;
    _Float16* dB_fp16 = nullptr;   // decoded B for fp16b kernel
    _Float16* dC_ck   = nullptr;
    _Float16* dC_out  = nullptr;   // reused per-kernel scratch
    HIP_OK(hipMalloc(&dA,      A.size() * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dB,      B_packed.size()));
    HIP_OK(hipMalloc(&dB_fp16, (size_t)K * N * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dC_ck,   (size_t)M * N * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dC_out,  (size_t)M * N * sizeof(_Float16)));
    HIP_OK(hipMemcpy(dA, A.data(),        A.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dB, B_packed.data(), B_packed.size(),             hipMemcpyHostToDevice));

    // Run CK path
    rcpp_ck_gemm_handle_t* h = nullptr;
    RC_OK(rcpp_ck_gemm_create(M, N, K, &h));
    RC_OK(rcpp_ck_gemm_run(h, dA, dB, dC_ck, nullptr));
    HIP_OK(hipDeviceSynchronize());

    // Decode B pk_i4 -> FP16 col-major for the fp16b kernel.
    rcpp_decode_pk_i4_to_fp16_launch(dB, dB_fp16, K, N, nullptr);
    HIP_OK(hipDeviceSynchronize());

    std::vector<_Float16> C_ck((size_t)M * N);
    HIP_OK(hipMemcpy(C_ck.data(), dC_ck, C_ck.size() * sizeof(_Float16), hipMemcpyDeviceToHost));

    auto run_and_diff = [&](const char* label, auto launch) -> float {
        HIP_OK(hipMemset(dC_out, 0, (size_t)M * N * sizeof(_Float16)));
        launch();
        HIP_OK(hipDeviceSynchronize());
        std::vector<_Float16> C_got((size_t)M * N);
        HIP_OK(hipMemcpy(C_got.data(), dC_out, C_got.size() * sizeof(_Float16), hipMemcpyDeviceToHost));
        float max_abs = 0.0f;
        for(size_t i = 0; i < C_ck.size(); ++i) {
            float d = std::fabs((float)C_ck[i] - (float)C_got[i]);
            max_abs = std::max(max_abs, d);
        }
        printf("  %-30s  max abs = %.6f\n", label, max_abs);
        return max_abs;
    };

    printf("Diffs vs CK:\n");
    float max_abs        = run_and_diff("Phase 1 naive",     [&](){ rcpp_standalone_launch                (dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_lds    = run_and_diff("Phase 2 LDS",       [&](){ rcpp_standalone_launch_lds            (dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_wmma   = run_and_diff("Phase 3 WMMA",      [&](){ rcpp_standalone_launch_wmma           (dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_2wave  = run_and_diff("Phase 4  2wave",    [&](){ rcpp_standalone_launch_wmma_2wave     (dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_4x     = run_and_diff("Phase 4b 4x",       [&](){ rcpp_standalone_launch_wmma_4x        (dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_2x4    = run_and_diff("Phase 4c 2x4",      [&](){ rcpp_standalone_launch_wmma_2x4       (dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_2x4ldb = run_and_diff("Phase 4d 2x4+ldsB", [&](){ rcpp_standalone_launch_wmma_2x4_ldsB  (dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_4x4    = run_and_diff("Phase 4g 4x4",      [&](){ rcpp_standalone_launch_wmma_4x4       (dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_4h     = run_and_diff("Phase 4h 4x4 cached",[&](){ rcpp_standalone_launch_wmma_4x4_cached(dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_4i     = run_and_diff("Phase 4i 4x4 vec",  [&](){ rcpp_standalone_launch_wmma_4x4_vec   (dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_4k     = run_and_diff("Phase 4k pingpong", [&](){ rcpp_standalone_launch_lds_pingpong   (dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_4j     = run_and_diff("Phase 4j pipelined",[&](){ rcpp_standalone_launch_4c_pipelined   (dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_4f     = run_and_diff("Phase 4f 2x2wave",  [&](){ rcpp_standalone_launch_wmma_2x2wave   (dA, dB,      dC_out, M, N, K, nullptr); });
    float max_abs_fp16b  = run_and_diff("fp16b (decoded B)", [&](){ rcpp_standalone_launch_fp16b          (dA, dB_fp16, dC_out, M, N, K, nullptr); });

    // Perf sanity for Phase 3 — the one the task pins as bit-exact.
    const int runs = 20;
    hipEvent_t e0, e1; HIP_OK(hipEventCreate(&e0)); HIP_OK(hipEventCreate(&e1));
    double flops = 2.0 * (double)M * N * K;

    auto time_ms = [&](auto launch) -> double {
        for(int w = 0; w < 3; ++w) launch();
        HIP_OK(hipDeviceSynchronize());
        HIP_OK(hipEventRecord(e0, nullptr));
        for(int r = 0; r < runs; ++r) launch();
        HIP_OK(hipEventRecord(e1, nullptr));
        HIP_OK(hipEventSynchronize(e1));
        float ms = 0.0f; HIP_OK(hipEventElapsedTime(&ms, e0, e1));
        return (double)ms / runs;
    };

    double ms_ck   = time_ms([&](){ RC_OK(rcpp_ck_gemm_run(h, dA, dB, dC_ck, nullptr)); });
    double ms_wmma = time_ms([&](){ rcpp_standalone_launch_wmma(dA, dB, dC_out, M, N, K, nullptr); });

    printf("Perf:\n");
    printf("  %-22s  %.3f ms  %6.2f TFlops   (1.0x)\n",
           "CK reference",   ms_ck,   flops / (ms_ck   * 1e-3) / 1e12);
    printf("  %-22s  %.3f ms  %6.2f TFlops  (%.1fx vs CK)\n",
           "Phase 3 WMMA",   ms_wmma, flops / (ms_wmma * 1e-3) / 1e12, ms_wmma / ms_ck);

    const float pass_abs = 0.25f;
    const bool pass =
        (max_abs        < pass_abs) && (max_abs_lds    < pass_abs) &&
        (max_abs_wmma   < pass_abs) && (max_abs_2wave  < pass_abs) &&
        (max_abs_4x     < pass_abs) && (max_abs_2x4    < pass_abs) &&
        (max_abs_2x4ldb < pass_abs) && (max_abs_4x4    < pass_abs) &&
        (max_abs_4h     < pass_abs) && (max_abs_4i     < pass_abs) &&
        (max_abs_4k     < pass_abs) && (max_abs_4j     < pass_abs) &&
        (max_abs_4f     < pass_abs) && (max_abs_fp16b  < pass_abs);
    printf("Verdict: %s (threshold max_abs < %.3f)\n", pass ? "PASS" : "FAIL", pass_abs);

    rcpp_ck_gemm_destroy(h);
    HIP_OK(hipFree(dA));     HIP_OK(hipFree(dB));
    HIP_OK(hipFree(dB_fp16)); HIP_OK(hipFree(dC_ck)); HIP_OK(hipFree(dC_out));
    HIP_OK(hipEventDestroy(e0)); HIP_OK(hipEventDestroy(e1));
    return pass ? 0 : 1;
}
