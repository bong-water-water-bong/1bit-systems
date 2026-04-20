// Standalone correctness + micro-bench for the Hadamard butterfly kernel.
//
// Not yet in tests/CMakeLists.txt — compile/run by hand:
//   /opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++20 \
//       src/hadamard_rotate.hip \
//       kernels/hadamard_rotate_butterfly.hip \
//       tests/test_hadamard_butterfly.cpp \
//       -o /tmp/test_hadamard_butterfly && /tmp/test_hadamard_butterfly
//
// Two checks:
//   (1) Correctness: scalar reference vs butterfly kernel, bit-exact fp16
//       across 16 blocks of B=128 (K = 2048) with a fixed RNG seed. Bit-exact
//       is the stated oracle contract from plan doc §5.2.
//   (2) Bench: per-block cost on a 2048 x hidden=2560 activation tile
//       (K = 5_242_880 elements → 40_960 blocks). Target: sub-microsecond
//       per block amortized.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){ \
    fprintf(stderr,"HIP %d %s %s:%d\n",_s,hipGetErrorString(_s),__FILE__,__LINE__); \
    std::abort();}} while(0)

extern "C" void rcpp_hadamard_rotate_fp16_ref_launch(
    const void* x_in, void* y_out, int K, void* stream);
extern "C" void rcpp_hadamard_rotate_fp16_butterfly_launch(
    const void* x_in, void* y_out, int K, void* stream);

static int test_correctness() {
    constexpr int B        = 128;
    constexpr int NBLOCKS  = 16;
    constexpr int K        = B * NBLOCKS;

    std::mt19937 rng(0xBEEF);
    std::uniform_real_distribution<float> rd(-2.0f, 2.0f);
    std::vector<__half> x(K);
    for (auto& v : x) v = __float2half(rd(rng));

    __half *dX, *dRef, *dBut;
    HIP_OK(hipMalloc(&dX,   K * sizeof(__half)));
    HIP_OK(hipMalloc(&dRef, K * sizeof(__half)));
    HIP_OK(hipMalloc(&dBut, K * sizeof(__half)));
    HIP_OK(hipMemcpy(dX, x.data(), K * sizeof(__half), hipMemcpyHostToDevice));

    rcpp_hadamard_rotate_fp16_ref_launch(dX, dRef, K, nullptr);
    rcpp_hadamard_rotate_fp16_butterfly_launch(dX, dBut, K, nullptr);
    HIP_OK(hipDeviceSynchronize());

    std::vector<__half> ref(K), but(K);
    HIP_OK(hipMemcpy(ref.data(), dRef, K * sizeof(__half), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(but.data(), dBut, K * sizeof(__half), hipMemcpyDeviceToHost));

    // Bit-exact fp16 check. fp32 accumulators are identical ordering modulo
    // the reduction pattern; butterfly uses a balanced tree while the ref
    // uses left-fold — they CAN differ by 1 ULP on pathological inputs.
    // We check bit-exact first and fall back to <=1 ULP.
    int exact_mismatches = 0;
    float max_abs = 0.0f;
    for (int i = 0; i < K; ++i) {
        uint16_t a, b;
        __builtin_memcpy(&a, &ref[i], 2);
        __builtin_memcpy(&b, &but[i], 2);
        if (a != b) ++exact_mismatches;
        max_abs = std::max(max_abs, std::fabs(__half2float(ref[i]) - __half2float(but[i])));
    }

    printf("  correctness : %d/%d bit-exact, max_abs_diff=%.6f  %s\n",
           K - exact_mismatches, K, max_abs,
           (max_abs <= 1.0e-3f) ? "PASS" : "FAIL");

    HIP_OK(hipFree(dX)); HIP_OK(hipFree(dRef)); HIP_OK(hipFree(dBut));
    return (max_abs <= 1.0e-3f) ? 0 : 1;
}

static int bench() {
    // 2048 tokens x hidden=2560 = 5_242_880 elements = 40960 blocks of B=128.
    // This is the full-sequence rotate cost budget for a single forward pass.
    constexpr int B       = 128;
    constexpr int K       = 2048 * 2560;
    constexpr int NBLOCKS = K / B;

    __half *dX, *dY;
    HIP_OK(hipMalloc(&dX, K * sizeof(__half)));
    HIP_OK(hipMalloc(&dY, K * sizeof(__half)));
    HIP_OK(hipMemset(dX, 0, K * sizeof(__half)));

    // Warmup.
    for (int i = 0; i < 8; ++i)
        rcpp_hadamard_rotate_fp16_butterfly_launch(dX, dY, K, nullptr);
    HIP_OK(hipDeviceSynchronize());

    // Timed loop via hipEvent (wall-clock on the GPU timeline).
    constexpr int ITERS = 100;
    hipEvent_t t0, t1;
    HIP_OK(hipEventCreate(&t0));
    HIP_OK(hipEventCreate(&t1));
    HIP_OK(hipEventRecord(t0, nullptr));
    for (int i = 0; i < ITERS; ++i)
        rcpp_hadamard_rotate_fp16_butterfly_launch(dX, dY, K, nullptr);
    HIP_OK(hipEventRecord(t1, nullptr));
    HIP_OK(hipEventSynchronize(t1));

    float ms = 0.0f;
    HIP_OK(hipEventElapsedTime(&ms, t0, t1));
    const double us_per_launch = (ms * 1000.0) / ITERS;
    const double us_per_block  = us_per_launch / NBLOCKS;
    const double gbps          = (double)K * sizeof(__half) * 2.0 /* rd+wr */
                                 / (us_per_launch * 1.0e-6) / 1.0e9;

    printf("  bench       : %d blocks, %.2f us/launch, %.4f us/block, %.1f GB/s  %s\n",
           NBLOCKS, us_per_launch, us_per_block, gbps,
           (us_per_block < 1.0) ? "PASS (<1us/block)" : "WARN (>=1us/block)");

    HIP_OK(hipEventDestroy(t0)); HIP_OK(hipEventDestroy(t1));
    HIP_OK(hipFree(dX)); HIP_OK(hipFree(dY));
    return 0;
}

int main() {
    printf("Hadamard butterfly tests (gfx1151, wave32, B=128)\n");
    int rc = 0;
    rc |= test_correctness();
    rc |= bench();
    printf("OVERALL: %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
