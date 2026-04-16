# rocm-cpp

Native ROCm C++ for Strix Halo (gfx1151). Built from scratch. No Python at runtime.

## What This Is

A pure C++ inference and compute stack targeting AMD Strix Halo APUs. Custom Wave32 HIP kernels, native Tensile GEMM from source, fused ternary inference — all C++, all on RDNA 3.5 silicon.

## Results (April 16, 2026)

### Full 1-Bit Burn — 7 Models

```
Model                     Quant    Size       pp512 t/s    ±std     tg128 t/s    ±std
──────────────────────────────────────────────────────────────────────────────────────
Bonsai-1.7B               Q1_0     231 MB     4,172.2     ±16.8      232.4      ±0.8
BitNet-2B-4T              Q1_0     538 MB     3,030.4      ±3.1      110.5      ±0.3
Bonsai-4B                 Q1_0     540 MB     2,014.1      ±4.7      125.3      ±1.0
Bonsai-8B                 Q1_0     1.07 GB    1,278.1      ±3.5       94.1      ±0.1
Qwen3-Coder-Next 80B      IQ1_S    17.6 GB      642.6      ±9.0       50.5      ±0.0
Llama-4-Scout 108B         IQ1_S    27.2 GB      323.3      ±2.3       21.2      ±0.0
BitNet-2B-4T              TQ1_0    1.02 GB      272.1      ±0.5       50.0      ±0.0

PrismML prism branch + TheRock gfx1151, llama-bench 3 rounds, ngl=99
```

80B MoE at 51 tok/s. 108B at 21 tok/s. 8B in 1 GB at 94 tok/s. All on one APU.

### ROCm vs Vulkan (Q1_0, same hardware)

```
Model           ROCm pp    Vulkan pp    Delta     ROCm tg    Vulkan tg    Delta
─────────────────────────────────────────────────────────────────────────────────
Bonsai-1.7B     4,172      3,121        +34%      232        137          +69%
Bonsai-4B       2,014      1,401        +44%      125         85          +47%
Bonsai-8B       1,278        831        +54%       94         64          +47%

ROCm beats Vulkan on both prompt AND generation. First Q1_0 GPU kernel for HIP.
```

### Benchmark Progression — Same Chip, Better Code

```
Engine                          Model              tok/s
────────────────────────────────────────────────────────
vLLM ROCm                       Qwen3-0.6B FP16    116.7
MLX ROCm C++ engine              Qwen3-0.6B-4bit    151.2    +30%
MLX C++ + TheRock Tensile        Qwen3-0.6B-4bit    153.3    native gfx1151
Vulkan (llama.cpp)               Qwen3-Coder GGUF    47.4    sustained
```

### Native Tensile GEMM — System vs TheRock (FP16 TFLOPS)

```
Shape (MxNxK)         System    TheRock    Change
──────────────────────────────────────────────────
1024x1024x1024        37.22     37.34      ~same
2560x6912x2560        25.04     32.97      +32%  ← BitNet FFN
4096x11008x4096       27.74     28.16      ~same
GEMV 1x2560x2560       0.06      0.07      +15%
GEMV 1x4096x4096       0.04      0.05      +18%
```

### Fused Ternary GEMV — First on gfx1151

```
Shape (MxK)          Time (μs)   Est tok/s   Correct
──────────────────────────────────────────────────────
2560x2560 (Q/K/V/O)    37.5       ~148        ✓
6912x2560 (FFN up)     109.3       ~51        ✓
2560x6912 (FFN down)   104.0       ~53        ✓
128256x2560 (LM head)  2653.3      ~2         ✓ (bottleneck)
```

### Q1_0 vs TQ1_0 (BitNet-2B-4T)

```
Format      Size      pp512 t/s    tg128 t/s    Speedup
──────────────────────────────────────────────────────────
Q1_0        538 MB    3,030         110          DP4A kernel
TQ1_0       1.02 GB     272          50          generic path
                        11.1x        2.2x
```

## The Problem

- No optimized Tensile/rocBLAS GEMM kernels exist for gfx1151 in any shipped package
- No ternary-aware kernel path exists on ROCm — anywhere
- Everyone falls back to generic dequantize-then-matmul (the slowest path)
- Missing compiler flags cause 69% regression that nobody documents
- hipBLASLt is "unsupported" on gfx1151 but works

## What We Built

- **TheRock from source** — ROCm 7.13 with 55 native Tensile GEMM kernels for gfx1151
- **Fused Wave32 ternary GEMV** — first HIP kernel for 1-bit inference on RDNA 3.5
- **Q1_0 HIP kernel** — added to llama.cpp, 24-33x faster prompt processing
- **GEMM micro-benchmarks** — proof that native Tensile beats system by 32% on LLM shapes
- **Full documentation** — every flag, every env var, every bug fix to replicate

## Quick Start

### Prerequisites

```bash
# CachyOS / Arch Linux
sudo pacman -S --needed base-devel cmake ninja git rocm-hip-sdk patchelf gcc-fortran

# Python deps for Tensile kernel generation
pip install --break-system-packages pyyaml joblib packaging tqdm CppHeaderParser
```

### Environment

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_ENABLE_SDMA=0
export ROCBLAS_USE_HIPBLASLT=1
export HIP_VISIBLE_DEVICES=0
```

### Build the Fused Ternary Kernel

```bash
cd rocm-cpp
hipcc -O3 --offload-arch=gfx1151 -ffast-math -munsafe-fp-atomics \
    -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64 \
    kernels/ternary_gemv.hip tools/bench_ternary.cpp \
    -o tools/bench_ternary
./tools/bench_ternary
```

### Build the GEMM Benchmark

```bash
hipcc -O3 --offload-arch=gfx1151 \
    -I/opt/rocm/include -L/opt/rocm/lib -lrocblas -lamdhip64 \
    tools/bench_gemm.cpp -o tools/bench_gemm
./tools/bench_gemm
```

### Build TheRock (ROCm from Source)

See [docs/02-therock-build.md](docs/02-therock-build.md) for the full guide.

```bash
git clone https://github.com/ROCm/TheRock.git therock
cd therock && git submodule update --init --recursive
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTHEROCK_AMDGPU_TARGETS=gfx1151 \
    -DTHEROCK_DIST_AMDGPU_FAMILIES=gfx115X-all \
    -DTHEROCK_ENABLE_BLAS=ON
cmake --build build --parallel $(nproc)
```

### Run 1-Bit Models (PrismML prism branch)

```bash
git clone https://github.com/PrismML-Eng/llama.cpp.git && cd llama.cpp
git checkout prism
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
    -DCMAKE_HIP_COMPILER=$HOME/therock/build/compiler/amd-llvm/dist/lib/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=$HOME/therock/build/compiler/amd-llvm/dist/lib/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=$HOME/therock/build/compiler/amd-llvm/dist/lib/llvm/bin/clang++
cmake --build build-rocm --parallel $(nproc)

export LD_LIBRARY_PATH=$HOME/therock/build/math-libs/BLAS/rocBLAS/dist/lib:/opt/rocm/lib
export ROCBLAS_TENSILE_LIBPATH=$HOME/therock/build/math-libs/BLAS/rocBLAS/dist/lib/rocblas/library
./build-rocm/bin/llama-bench -m Bonsai-8B.gguf -ngl 99 -p 512 -n 128 -r 3
```

## Structure

```
kernels/
  ternary_gemv.hip     — Fused Wave32 ternary GEMV (the core kernel)
tools/
  bench_gemm.cpp       — rocBLAS GEMM benchmark (system vs TheRock)
  bench_ternary.cpp    — Fused ternary kernel benchmark + correctness
  run_bench.sh         — Automated comparison script
  test_triton_fork.py  — Triton subprocess test (TheRock #4552)
results/
  bonsai-q1_0-rocm-20260416.md    — Q1_0 kernel results
  full-1bit-burn-20260416.md      — Full 7-model burn
docs/
  00-hardware.md       — Strix Halo specs, unified memory, BIOS
  01-environment.md    — Runtime vars, shell setup, verification
  02-therock-build.md  — Building ROCm from source step by step
  03-compiler-flags.md — The 69% flag and all HIP AOT flags
  04-wave32-kernels.md — RDNA 3.5 kernel design guide
  05-ternary-inference.md — 1-bit theory, packing, kernel design
  06-benchmarking.md   — All numbers and comparison tables
  07-forks-landscape.md — Every relevant fork and what they did
  08-known-issues.md   — Every gfx1151 bug with workarounds
```

## Hardware

```
AMD Ryzen AI Max+ 395 (Strix Halo)
Radeon 8060S (gfx1151, RDNA 3.5, Wave32, 20 CUs)
128GB unified LPDDR5X
CachyOS kernel 7.0
```

## TheRock Build Patches (GCC 15)

If building on GCC 15 / bleeding edge, these patches are needed:

1. **elfutils** — add `-Wno-error=discarded-qualifiers` to CPPFLAGS
2. **rocprofiler-sdk elfio** — add `#include <cstdint>` to `elf_types.hpp`
3. **rocprofiler-sdk yaml-cpp** — add `#include <cstdint>` to `emitterutils.cpp`
4. **aqlprofile test** — skip integration test (wrong compiler for HIP)
5. **Missing packages** — `xxd` (gvim), `pyyaml`, `CppHeaderParser`, `joblib`, `packaging`, `tqdm`, `gcc-fortran`

See [docs/02-therock-build.md](docs/02-therock-build.md) for details.

## Related Repos

- [bleeding-edge](https://github.com/stampby/bleeding-edge) — Wiki with full build log and known issues
- [lemon-mlx-engine](https://github.com/stampby/lemon-mlx-engine) — C++ MLX engine hitting 153 t/s
- [halo-1bit](https://github.com/stampby/halo-1bit) — 1-bit inference engine (Phase 1 + Phase 2)
- [PrismML llama.cpp](https://github.com/PrismML-Eng/llama.cpp) — Prism branch with Q1_0 DP4A kernels

## License

If it can be done in C++, we do it in C++.

Fork it. Improve it. Push it back.
