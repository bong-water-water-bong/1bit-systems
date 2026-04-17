# How We Built a 1-bit GEMM Kernel That Beats CK on gfx1151

A chronological recipe: blank repo → a C library that ships a drop-in 1-bit BitNet / ternary compute path on AMD Strix Halo (Radeon 8060S, gfx1151) at **101.7% of Composable Kernel's performance**, with **zero CK headers in the winning kernel's translation unit**.

TL;DR at the bottom. All numbers are measurable from this repo — no black boxes, no fudge.

## The ceiling we're measuring against

Before any kernel-level optimization, measure the actual ceiling. `src/wmma_peak_probe.hip` runs a wave that pins A/B fragments in registers and loops 100,000 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` calls with zero memory traffic on the hot path. Result on gfx1151:

```
gfx1151 WMMA peak (f32 accum, f16 in, no-memory loop) = 55.36 TFlops
```

That's the wall. **Anything WMMA-based on this hardware converges around 55% of that on realistic shapes** because of the fundamental cost of real memory traffic. CK does. We do.

```
BitNet FFN up shape (2560 x 6912 x 2560)
──────────────────────────────────────────────
CK reference DeviceGemm_Wmma_CShuffleV3    30.20 TFlops   54.6% of peak
Our Phase 4h standalone                     30.75 TFlops   55.6% of peak
```

## Prerequisites

```bash
# CachyOS / Arch Linux — same toolchain we ship
sudo pacman -S --needed base-devel cmake ninja git patchelf gcc-fortran
pip install --break-system-packages pyyaml joblib packaging tqdm CppHeaderParser

# TheRock ROCm 7.13 built from source for gfx1151 — the compiler + rocBLAS reference
git clone https://github.com/ROCm/TheRock.git ~/therock
cd ~/therock && git submodule update --init --recursive
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTHEROCK_AMDGPU_TARGETS=gfx1151 \
    -DTHEROCK_DIST_AMDGPU_FAMILIES=gfx115X-all \
    -DTHEROCK_ENABLE_BLAS=ON
cmake --build build --parallel $(nproc)
# -> ~/therock/build/dist/rocm/
```

Runtime env used for every measurement below:

```bash
export THEROCK=$HOME/therock/build/dist/rocm
export LD_LIBRARY_PATH=$THEROCK/lib:/opt/rocm/lib:$PWD/build
export ROCBLAS_TENSILE_LIBPATH=$THEROCK/lib/rocblas/library
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_ENABLE_SDMA=0
export ROCBLAS_USE_HIPBLASLT=1
export HIP_VISIBLE_DEVICES=0
```

## Layout of the kernel

Every variant computes `C = A * B` on gfx1151 where:

| Operand | Storage layout |
|---|---|
| A | FP16 row-major `[M, K]`, stride K |
| B | pk_i4 bytes, WMMA-permuted, `[K*N/2]` |
| C | FP16 row-major `[M, N]`, stride N |

The weight packer `rcpp_ternary_pack_pk_i4` runs once at model load:

```
ternary int8 {-1, 0, +1}  ->  pk_i4 nibble {0x7, 0x8, 0x9}
                              (CK applies a '-8' bias in its FP16 decode,
                               so we pre-offset the nibble values)
```

Then a two-stage permute:
1. Block reshape K -> `[K0, N, K1]` with `K1 = KPerBlock = 32` (byte-level memcpy at 16-byte granularity).
2. Within-8 nibble permute `01234567 -> 20643175` (matches CK's upstream `gemm_wmma_fp16_pk_i4_v3.cpp` lines 168–215).

**High-nibble-first byte ordering** because `CK_USE_PK4_LAYOUT_SHUFFLE = 1` is defined in `ck/ck.hpp`. Getting this wrong produces a consistent ~6× magnitude error on output — silent, not obviously broken.

## The progression

Each phase ships correctness against CK's scalar host reference (`check_err` at FP16 tolerance) before optimizing. If a phase regresses in correctness we roll it back. Numbers below are on BitNet FFN up, `2560 x 6912 x 2560`.

### Phase 0 — drop libutility.a (commit `6b694cb`)

Before: `librocm_cpp.so` linked `composable_kernel::utility`, pulling `ck::HostTensorDescriptor`, `ck::Tensor`, and `ck::utils::conv::*` into the shared object.

Fix: rewrite `rcpp_ternary_pack_pk_i4` in pure C++ with one `std::vector<uint8_t>`. Explicit byte-level memcpy for the block permute; in-place 4-byte rewrites for the within-8 nibble permute.

Size: 425 KiB → 390 KiB. Zero `ck::utils::conv` symbols remaining.

### Phase 1 — standalone naive kernel (`599b78e`)

First kernel in `src/prefill_standalone.hip` with **zero `ck/` includes**. One thread per output element. Each thread loops K internally, decoding pk_i4 on the fly and reversing the within-8 permute in register math:

```cpp
const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};
const int inv_perm[8]     = {1, 5, 0, 4, 3, 7, 2, 6};
```

Output bit-matches CK across 5 BitNet-realistic shapes. Perf: **1.16 TFlops** (3.8% of CK). This isn't a win — it's a **correctness anchor** that future phases can't break.

### Phase 2 — LDS tiling (`621b489`)

Same kernel logic, 16×16 output tile per block, K_TILE=32, A and B tiles staged through LDS. Within-8 nibble permute moved to LDS-load time so the accumulator loop reads K-contiguous values directly.

Bit-perfect. Perf: **1.73 TFlops** (5.7% of CK). LDS alone doesn't help much with a single-wave block — you need multi-wave consumers to amortize the load work. Kept anyway as a stepping stone.

### Phase 3 — WMMA (`93cf7c3`)

Now we use the hardware. One wave per block, 16×16 output, FP32 accumulator.

```cpp
using half16_t = __attribute__((ext_vector_type(16))) __fp16;
using float8_t = __attribute__((ext_vector_type(8))) float;

acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A_frag, B_frag, acc);
```

**The hazard:** RDNA 3 Wave32 WMMA's C fragment layout is **interleaved rows**, not blocked. Lane `i`, slot `r` in 0..7 holds `C[2*r + (i >= 16), i % 16]`. Lanes 0..15 own even rows of their column, lanes 16..31 own odd rows.

The wrong (obvious) guess — `C[r + 8*(i >= 16), i % 16]` (blocked) — produces `max_abs ~15` garbage that still runs and reports TFlops like it's fine. Cost us one iteration. **Always diff against a scalar reference before trusting a WMMA result.**

Bit-perfect once fixed. Perf: **14.46 TFlops** (48% of CK) — one step, 8× jump.

### Phase 4b — 1 wave, 16×64 output (`32c113c`)

Each wave now does 4 WMMA tiles along N (16×64 total output per wave). A fragment loaded once per K step, used across 4 B fragments. Arithmetic intensity: 4 WMMAs per 5 fragment loads (1.33 / 5).

Bit-perfect. Perf: **14.47 TFlops** (48%) — marginal. A reuse helps, but B-side decode overhead is now the bottleneck.

### Phase 4c — 1 wave, 32×64 output (`e02c60d`)

**2 A fragments × 4 B fragments = 8 accumulators.** Cross product: each A fragment feeds 4 B fragments, each B fragment feeds 2 A fragments. Arithmetic intensity 8/6 = 1.33 WMMAs/load. Wait — that's the same? Look at total work: 8 WMMAs per step × 2 steps per K1 block = 16 WMMAs per K1 block vs Phase 4b's 8. Double.

Bit-perfect. Perf: **22.35 TFlops** (74%) — 50% jump.

### Phase 4d / 4f — the honest regressions (`6cfc205`)

Tried two LDS-caching experiments that **failed**:

- **4d** (1 wave × 32×64 + B decoded to LDS): 15.29 TFlops. LDS decode costs nothing to read but the single-wave block has no one to share LDS with.
- **4f** (4 waves × 64×64, A+B in LDS): 10.46 TFlops. 128-thread blocks cut CU occupancy on gfx1151 (40 CUs × 2 SIMDs × 2 waves = 160 wave slots; 128-thread blocks drop block count from 8,640 to 4,320). On Strix Halo, **wide-thin (1 wave, many blocks) beats deep-wide**. CK's 256-thread / 128×128 tile design is optimal on MI300 with many more WGPs — it inverts on gfx11.

We keep these in-tree with the `/* ... */` reasoning preserved. Silent failures teach nothing; documented failures are cheap lessons.

### Phase 4g — 1 wave, 64×64 output (`344a89a`)

**4 A × 4 B = 16 accumulators.** Arithmetic intensity 16/8 = **2.0 WMMAs/load**. Block count for 2560×6912: 40 × 108 = **4,320 blocks / 160 wave slots = 27× occupancy**, plenty.

Register pressure estimate:
- `acc[4][4]` float8_t = 128 VGPRs/lane
- `A_frags[4]` half16_t = 32 VGPRs/lane
- `B_frag` half16_t = 8 VGPRs/lane
- Loop state + temporaries ≈ 40 VGPRs
- **Total ≈ 208 / 256** — fits gfx11's per-thread VGPR budget with headroom.

Bit-perfect. Perf: **28.28 TFlops** (94% of CK). One structural decision, 20 percentage points.

### Phase 4h — A/B cached across K steps (`675510d`)

Phase 4g loaded A and B on **every** WMMA step (2 per K1 block). Phase 4h loads both fragments **once per K1 block**, caching them in registers, then runs the same 16 WMMAs from registers.

```cpp
half16_t A_cache[4][2];   // 4 m_sub x 2 K steps = 128 halves/lane = 64 VGPRs
half16_t B_cache[2];      //         2 K steps  =  32 halves/lane = 16 VGPRs
// ... + 128 VGPRs for acc -> ~215 VGPRs/lane, fits 256 budget
```

Per-block memory traffic halves. Per-block WMMAs unchanged at 32. Bit-perfect.

**Perf: 30.75 TFlops. 101.7% of CK.** Public API `rcpp_standalone_gemm` now routes here.

### Phase 4i — vectorized A loads (`9bd111c`)

Swapped 16 scalar `__half` loads for 4 `half4_t` (64-bit) loads per fragment half. Neutral result: **30.64 TFlops** — within measurement noise. hipcc was already emitting 128-bit `buffer_load` ops for the unrolled scalar loads on gfx11. Kept in-tree as a reference; the scalar-load Phase 4h stays the default.

## Why we stopped

Peak probe (`0d22b1f`) measured the WMMA ceiling at **55.36 TFlops**. At 30.75 we're **55.6% of peak** on a shape where CK gets 54.6%. The gap isn't tuning headroom — it's memory + pk_i4 decode overhead that **any** FP16×FP16 WMMA kernel hits on this shape.

**Past this wall requires a different compute model**, not a different tile shape:
- Phase 5 — BitNet select-and-add: weights are {-1, 0, +1}, so `A * B` is just sign-conditional add. No FP16 multiplication instruction, no WMMA. The theoretical path past CK specifically for ternary, per the [BitNet 1.58 arxiv paper](https://arxiv.org/abs/2402.17764).
- INT8 WMMA — quantize A to INT8 (lossy but acceptable for BitNet inference), use `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`. Same throughput as FP16 WMMA, smaller A bandwidth.

Both are different kernels, not tuned variants of this one. They are Phase 5 work for a future session.

## How to consume this

Every phase's binary lives in the tree. Link against `librocm_cpp.so`, include one header:

```c
#include <rocm_cpp/ck_gemm.h>

// Once at model load — host side, no GPU:
rcpp_ternary_pack_pk_i4(ternary_KN, packed_KN_div2, K, N);

// Hot path (zero-CK standalone backend, Phase 4h):
rcpp_standalone_gemm(A_fp16_dev, B_packed_dev, C_fp16_dev, M, N, K, stream);

// Or explicitly the CK-backed path:
rcpp_ck_gemm_handle_t* h;
rcpp_ck_gemm_create(M, N, K, &h);
rcpp_ck_gemm_run(h, A_fp16_dev, B_packed_dev, C_fp16_dev, stream);
```

## Build recipe

```bash
cd rocm-cpp
cmake -B build -G Ninja
ninja -C build
# Outputs:
#   build/librocm_cpp.so      (the C library)
#   build/test_ck_gemm        (C API end-to-end correctness + perf)
#   build/test_standalone     (3-way diff: CK / Phase 1-4g / Phase 4h all vs CK)
#   build/ck-prefill/*        (research binaries)
```

## Tools to measure what we claim

```bash
# End-to-end correctness: CPU scalar GEMM vs GPU through the C API
./build/test_ck_gemm 512 512 2560
# -> PASS (max abs 0.008 vs CPU reference, threshold 0.5)

# Head-to-head: CK vs all standalone variants on the same packed weights
./build/test_standalone 2560 6912 2560
# -> diff 0.000000 across Phase 1/2/3/4g, perf progression printed

# Hardware ceiling probe (no memory traffic)
# (builds from src/wmma_peak_probe.hip; see docs for the driver)
# -> ~55 TFlops

# Full 7-model 1-bit llama-bench burn (unrelated to our library —
# exercises PrismML's prism llama.cpp kernel)
~/prism-eng-llamacpp/build-rocm/bin/llama-bench -m <model.gguf> -ngl 99 -p 512 -n 128 -r 3
```

## TL;DR

- Hardware ceiling: **55.36 TFlops** (measured, gfx1151 WMMA pure compute)
- CK `DeviceGemm_Wmma_CShuffleV3` on BitNet FFN up: **30.20 TFlops** (54.6% of peak)
- `librocm_cpp` Phase 4h on same shape: **30.75 TFlops (55.6% of peak, 101.7% of CK)**
- Correctness: bit-perfect vs CK's scalar host reference (max abs `0.000000`) across 5 shapes including BitNet FFN
- Zero `ck/` headers in the Phase 4h kernel TU
- Build-time CK dep removed from the packer (`libutility.a` dropped)
- Five honest regressions preserved in-tree with root cause
- Public C API: 4 functions, one header, consumable without pulling CK templates into downstream TUs

*End of line.*
