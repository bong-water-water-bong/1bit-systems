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

extern "C" void rcpp_standalone_launch(const void*, const void*, void*, int, int, int, void*);

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP err %d %s:%d\n",_s,__FILE__,__LINE__); std::abort();}} while(0)
#define RC_OK(e)  do { auto _s=(e); if(_s!=RCPP_OK){fprintf(stderr,"rcpp err %d %s:%d\n",(int)_s,__FILE__,__LINE__); std::abort();}} while(0)

int main(int argc, char** argv) {
    int M = 256, N = 256, K = 512;   // small default — naive kernel is slow
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
    _Float16* dC_ck   = nullptr;
    _Float16* dC_std  = nullptr;
    HIP_OK(hipMalloc(&dA,      A.size() * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dB,      B_packed.size()));
    HIP_OK(hipMalloc(&dC_ck,   (size_t)M * N * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dC_std,  (size_t)M * N * sizeof(_Float16)));
    HIP_OK(hipMemcpy(dA, A.data(),        A.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dB, B_packed.data(), B_packed.size(),             hipMemcpyHostToDevice));

    // Run CK path
    rcpp_ck_gemm_handle_t* h = nullptr;
    RC_OK(rcpp_ck_gemm_create(M, N, K, &h));
    RC_OK(rcpp_ck_gemm_run(h, dA, dB, dC_ck, nullptr));
    HIP_OK(hipDeviceSynchronize());

    // Run standalone path
    rcpp_standalone_launch(dA, dB, dC_std, M, N, K, nullptr);
    HIP_OK(hipDeviceSynchronize());

    // Diff
    std::vector<_Float16> C_ck((size_t)M * N), C_std((size_t)M * N);
    HIP_OK(hipMemcpy(C_ck.data(),  dC_ck,  C_ck.size()  * sizeof(_Float16), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(C_std.data(), dC_std, C_std.size() * sizeof(_Float16), hipMemcpyDeviceToHost));

    float max_abs = 0.0f, max_rel = 0.0f;
    double sum_abs = 0.0;
    for(size_t i = 0; i < C_ck.size(); ++i) {
        float a = (float)C_ck[i];
        float b = (float)C_std[i];
        float d = std::fabs(a - b);
        max_abs = std::max(max_abs, d);
        float denom = std::max(std::fabs(a), 1e-3f);
        max_rel = std::max(max_rel, d / denom);
        sum_abs += d;
    }
    double mean_abs = sum_abs / C_ck.size();
    printf("Standalone vs CK output diff:\n");
    printf("  max abs  : %.6f\n", max_abs);
    printf("  max rel  : %.6f\n", max_rel);
    printf("  mean abs : %.6f\n", mean_abs);

    // Perf sanity (not a goal at Phase 1)
    const int runs = 10;
    hipEvent_t e0, e1; HIP_OK(hipEventCreate(&e0)); HIP_OK(hipEventCreate(&e1));

    HIP_OK(hipEventRecord(e0, nullptr));
    for(int r = 0; r < runs; ++r) rcpp_standalone_launch(dA, dB, dC_std, M, N, K, nullptr);
    HIP_OK(hipEventRecord(e1, nullptr));
    HIP_OK(hipEventSynchronize(e1));
    float ms_std = 0.0f; HIP_OK(hipEventElapsedTime(&ms_std, e0, e1));
    ms_std /= runs;

    HIP_OK(hipEventRecord(e0, nullptr));
    for(int r = 0; r < runs; ++r) RC_OK(rcpp_ck_gemm_run(h, dA, dB, dC_ck, nullptr));
    HIP_OK(hipEventRecord(e1, nullptr));
    HIP_OK(hipEventSynchronize(e1));
    float ms_ck = 0.0f; HIP_OK(hipEventElapsedTime(&ms_ck, e0, e1));
    ms_ck /= runs;

    double flops = 2.0 * (double)M * N * K;
    printf("Perf (expected: standalone much slower at Phase 1):\n");
    printf("  CK         : %.3f ms  %.2f TFlops\n", ms_ck,  flops / (ms_ck  * 1e-3) / 1e12);
    printf("  Standalone : %.3f ms  %.2f TFlops  (%.1fx vs CK)\n",
           ms_std, flops / (ms_std * 1e-3) / 1e12, ms_std / ms_ck);

    const float pass_abs = 0.25f;
    const int pass = (max_abs < pass_abs);
    printf("Verdict: %s (threshold max_abs < %.3f)\n", pass ? "PASS" : "FAIL", pass_abs);

    rcpp_ck_gemm_destroy(h);
    HIP_OK(hipFree(dA)); HIP_OK(hipFree(dB));
    HIP_OK(hipFree(dC_ck)); HIP_OK(hipFree(dC_std));
    HIP_OK(hipEventDestroy(e0)); HIP_OK(hipEventDestroy(e1));
    return pass ? 0 : 1;
}
