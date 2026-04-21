// test_cpu_ternary_gemv.cpp — differential correctness + microbench.
//
// Correctness:
//   50 RNG seeds × 4 shapes in {(N, K) : N, K ∈ {2048, 6912}}. Pack random
//   ternary codes (with 11 → "reserved 0" still emitted so the kernel sees
//   the full domain) and fp16 activations. Compare AVX2 output vs scalar
//   ref on EVERY row. Tolerance: ≤4 bf16-ULP on the fp16 bit-pattern (i.e.
//   compare bf16 floor of both, allow integer distance ≤4).
//
// Bench (`--bench`):
//   For each shape:
//     - 1-thread AVX2 kernel: timed over 50 iterations on the same buffers.
//     - 16-thread AVX2 kernel: likewise.
//   Reports:
//     - per-iter ms
//     - effective weight GB/s (N * K/128 * 34 bytes / iter-time)
//     - tok/s-equivalent: 1 GEMV = 1 output activation vector of length N
//       for one token, so prefill tok/s at that shape = 1 / iter-time.

#include "halo_cpu/ternary_gemv.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <immintrin.h>  // for fp16↔fp32 in the harness

namespace {

constexpr int kBlockSize     = 128;
constexpr int kTQ2BlockBytes = 34;

inline uint16_t fp32_to_fp16_bits(float f) noexcept {
    const __m128  v = _mm_set_ss(f);
    const __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return static_cast<uint16_t>(_mm_extract_epi16(h, 0));
}

inline float fp16_bits_to_fp32(uint16_t h) noexcept {
    const __m128i v = _mm_cvtsi32_si128(static_cast<int>(h));
    const __m128  f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

// Tolerance metric: bf16-ULP on the fp16 output. We reduce fp16 → bf16 via
// fp16→fp32 (exact) → round-to-nearest-even bf16 (7-bit mantissa) and
// compare as signed-magnitude integers.
//
// Subnormal escape hatch: near zero (|fp32| < 2^-13), bf16 granularity blows
// up faster than fp16 granularity (bf16 has 7 mantissa bits to fp16's 10),
// so a 1-fp16-ULP miss can look like ~16 bf16-ULPs even though the output
// round-tripped cleanly. Below the escape threshold we fall back to raw
// fp16-ULP distance (still capped at 4) — this matches what downstream
// inference actually sees.
uint16_t fp16_to_bf16_mag(uint16_t h) noexcept {
    const float f = fp16_bits_to_fp32(h);
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t rounding_bias = 0x7FFF + ((u >> 16) & 1u);
    u += rounding_bias;
    uint16_t bf = static_cast<uint16_t>(u >> 16);
    if (bf & 0x8000) bf = static_cast<uint16_t>(0x8000 - (bf & 0x7FFF));
    else             bf = static_cast<uint16_t>(0x8000 | bf);
    return bf;
}

uint16_t fp16_signed_mag(uint16_t h) noexcept {
    if (h & 0x8000) return static_cast<uint16_t>(0x8000 - (h & 0x7FFF));
    return static_cast<uint16_t>(0x8000 | h);
}

int bf16_ulp_distance(uint16_t a, uint16_t b) noexcept {
    // Escape to fp16-ULP metric for near-zero outputs (below ~1.2e-4),
    // i.e. when either input is in the fp16 subnormal / tiny-normal range.
    const uint16_t ae = static_cast<uint16_t>(a & 0x7FFF);
    const uint16_t be = static_cast<uint16_t>(b & 0x7FFF);
    constexpr uint16_t kTinyThresh = 0x0200;  // exponent < 1 → tiny normals/subnormals
    if (ae < kTinyThresh || be < kTinyThresh) {
        const uint16_t ma = fp16_signed_mag(a);
        const uint16_t mb = fp16_signed_mag(b);
        return std::abs(static_cast<int>(ma) - static_cast<int>(mb));
    }
    const uint16_t ma = fp16_to_bf16_mag(a);
    const uint16_t mb = fp16_to_bf16_mag(b);
    return std::abs(static_cast<int>(ma) - static_cast<int>(mb));
}

struct Shape { int N; int K; };

// Pack a single random TQ2 block into `block_out` (34 bytes) given a
// row-uniform fp16 scale `d` and an RNG for the 2-bit codes.
void pack_block(uint8_t* block_out, uint16_t d_u16, std::mt19937_64& rng) noexcept {
    std::memcpy(block_out, &d_u16, 2);
    // Random 2-bit codes, fully covering {0,1,2,3}.
    for (int byte_idx = 0; byte_idx < 32; ++byte_idx) {
        uint8_t b = 0;
        for (int j = 0; j < 4; ++j) {
            const uint32_t c = static_cast<uint32_t>(rng() & 3u);
            b |= static_cast<uint8_t>(c << (2 * j));
        }
        block_out[2 + byte_idx] = b;
    }
}

// Emit a random fp16 in ~N(0, sigma). Uses box-muller on fp32 then converts.
uint16_t random_fp16(std::mt19937_64& rng, float sigma) noexcept {
    std::uniform_real_distribution<float> uni(1e-7f, 1.0f);
    const float u1 = uni(rng);
    const float u2 = uni(rng);
    const float r  = std::sqrt(-2.0f * std::log(u1));
    const float th = 6.28318530717958647692f * u2;
    const float z  = r * std::cos(th);
    return fp32_to_fp16_bits(sigma * z);
}

void populate_packed(std::vector<uint8_t>& packed, int N, int K, std::mt19937_64& rng) {
    const int num_blocks = K / kBlockSize;
    const std::size_t row_bytes =
        static_cast<std::size_t>(num_blocks) * static_cast<std::size_t>(kTQ2BlockBytes);
    packed.assign(static_cast<std::size_t>(N) * row_bytes, 0);
    // Per-row scale drawn from ~N(0, 0.01) like real Bonsai TQ2 weights.
    for (int row = 0; row < N; ++row) {
        uint8_t* row_ptr = packed.data() + static_cast<std::size_t>(row) * row_bytes;
        for (int b = 0; b < num_blocks; ++b) {
            const uint16_t d_u16 = random_fp16(rng, 0.01f);
            pack_block(row_ptr + static_cast<std::size_t>(b) * kTQ2BlockBytes, d_u16, rng);
        }
    }
}

void populate_act(std::vector<uint16_t>& act, int K, std::mt19937_64& rng) {
    act.resize(static_cast<std::size_t>(K));
    for (int i = 0; i < K; ++i) act[static_cast<std::size_t>(i)] = random_fp16(rng, 1.0f);
}

// Returns (max_ulp, num_fails).
std::pair<int, int> diff_outputs(const std::vector<uint16_t>& a,
                                 const std::vector<uint16_t>& b,
                                 int ulp_tol) noexcept {
    int max_ulp = 0;
    int fails   = 0;
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) {
        const int d = bf16_ulp_distance(a[i], b[i]);
        if (d > max_ulp) max_ulp = d;
        if (d > ulp_tol) ++fails;
    }
    return {max_ulp, fails};
}

double time_kernel(const std::vector<uint8_t>& packed,
                   const std::vector<uint16_t>& act,
                   std::vector<uint16_t>& out,
                   int N, int K, int threads, int iters) noexcept {
    // Warmup.
    for (int w = 0; w < 2; ++w) {
        halo_cpu_ternary_gemv_tq2(packed.data(), act.data(), out.data(),
                                  N, K, threads);
    }
    // Prevent DCE by XORing the last fp16 across iterations.
    std::atomic<uint32_t> sink{0};
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        halo_cpu_ternary_gemv_tq2(packed.data(), act.data(), out.data(),
                                  N, K, threads);
        sink.fetch_add(out.back(), std::memory_order_relaxed);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    (void)sink.load();
    return secs / iters;
}

int run_correctness() {
    const Shape shapes[] = {
        {2048, 2048}, {2048, 6912}, {6912, 2048}, {6912, 6912}
    };
    constexpr int kNumSeeds = 50;
    constexpr int kUlpTol   = 4;

    int overall_max_ulp = 0;
    int overall_fails   = 0;

    for (const auto s : shapes) {
        int shape_max_ulp = 0;
        int shape_fails   = 0;
        for (int seed = 0; seed < kNumSeeds; ++seed) {
            std::mt19937_64 rng(
                static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ull
                ^ static_cast<uint64_t>(s.N) << 16
                ^ static_cast<uint64_t>(s.K));
            std::vector<uint8_t>  packed;
            std::vector<uint16_t> act, out_avx(static_cast<std::size_t>(s.N), 0),
                                       out_ref(static_cast<std::size_t>(s.N), 0);
            populate_packed(packed, s.N, s.K, rng);
            populate_act   (act,    s.K, rng);

            halo_cpu_ternary_gemv_tq2           (packed.data(), act.data(),
                                                 out_avx.data(), s.N, s.K, 1);
            halo_cpu_ternary_gemv_tq2_scalar_ref(packed.data(), act.data(),
                                                 out_ref.data(), s.N, s.K);

            const auto [mu, fails] = diff_outputs(out_avx, out_ref, kUlpTol);
            if (mu    > shape_max_ulp) shape_max_ulp = mu;
            shape_fails += fails;
        }
        overall_max_ulp = std::max(overall_max_ulp, shape_max_ulp);
        overall_fails  += shape_fails;
        std::printf("  shape N=%-5d K=%-5d  max_bf16_ulp=%d  fails=%d  (%d seeds)\n",
                    s.N, s.K, shape_max_ulp, shape_fails, kNumSeeds);
    }

    std::printf("CORRECTNESS: max_bf16_ulp=%d (tol=4) total_fails=%d %s\n",
                overall_max_ulp, overall_fails,
                overall_fails == 0 ? "[PASS]" : "[FAIL]");
    return overall_fails == 0 ? 0 : 1;
}

void run_bench() {
    const Shape shapes[] = {
        {2048, 2048}, {2048, 6912}, {6912, 2048}, {6912, 6912}
    };
    constexpr int kIters = 50;

    const int max_threads = omp_get_max_threads();
    const int wide = std::min(16, max_threads);

    std::printf("BENCH: iters/shape=%d   max_omp_threads=%d   using {1, %d}\n",
                kIters, max_threads, wide);
    std::printf("%-14s  %-10s  %-10s  %-12s  %-12s\n",
                "shape", "threads", "ms/iter", "W_GB/s", "tok/s_equiv");

    std::mt19937_64 rng(0xDECAFD00D1155EED);
    for (const auto s : shapes) {
        std::vector<uint8_t>  packed;
        std::vector<uint16_t> act;
        std::vector<uint16_t> out(static_cast<std::size_t>(s.N), 0);
        populate_packed(packed, s.N, s.K, rng);
        populate_act   (act,    s.K, rng);

        const double w_bytes =
            static_cast<double>(s.N) *
            static_cast<double>(s.K / kBlockSize) *
            static_cast<double>(kTQ2BlockBytes);

        for (int threads : {1, wide}) {
            const double per_iter = time_kernel(packed, act, out, s.N, s.K,
                                                threads, kIters);
            const double gbps = (w_bytes / per_iter) / 1e9;
            const double toks = 1.0 / per_iter;
            std::printf("%-14s  %-10d  %-10.3f  %-12.2f  %-12.1f\n",
                        (std::to_string(s.N) + "x" + std::to_string(s.K)).c_str(),
                        threads, per_iter * 1e3, gbps, toks);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") bench = true;
    }

    std::printf("halo CPU AVX2 ternary GEMV — test harness\n");
    const int rc = run_correctness();
    if (rc != 0) return rc;
    if (bench) {
        std::printf("\n");
        run_bench();
    }
    return 0;
}
