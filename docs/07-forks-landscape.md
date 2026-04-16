# Fork Landscape — Who's Doing What

## The Players

### NripeshN/mlx — Original ROCm Backend
- The first MLX ROCm port
- Author doesn't have an AMD GPU — built everything in Docker
- PR #2300 open for 10 months, upstream won't merge
- Base code for all other forks
- Useful as: starting point, reference implementation

### goniz/mlx — APU Optimizations
- Has Strix Halo hardware
- Flash attention implementation for ROCm
- Allocator redesign for unified memory (APU-aware)
- QMV vectorization for integrated GPU
- 52 tok/s Python on Strix Halo
- Useful as: APU-specific fixes, allocator patterns

### lemonade-sdk Patches — The Speed Patches
- Warp-cooperative GEMV kernel (the big one for 4-bit)
- hipBLASLt integration (2.6x prompt speedup)
- Slab allocator for GPU memory
- iGPU spin-wait optimization
- Custom env var MLX_ROCM_HIPBLASLT to enable hipBLASLt on "unsupported" targets
- Useful as: the performance patches, especially for 4-bit models

### PrismML-Eng/mlx — 1-bit Quantization
- 1-bit affine quantization branch
- Quantization framework for MLX
- Useful as: quantization approach reference

### carlosfundora/llama.cpp-1-bit-turbo — Ternary HIP Kernels
- **Most relevant to our work**
- Custom Wave32 HIP kernels for RDNA
- Q1_0_G128 ternary quantization format
- Native HIP dequant + dot-product fused kernel
- 128-thread blocks optimized for Wave32
- 209 tok/s on RX 6700 XT for Bonsai 1.7B
- Useful as: the reference for building ternary kernels

### Geramy/mlx — AMD Employee
- CMake build fixes
- FP8 experiments
- Part of the Lemonade team at AMD
- "2x slower than llama.cpp" — their own assessment of MLX ROCm
- Useful as: minor fixes only

## Upstream MLX Status

Apple's MLX project has:
- NO ROCm roadmap
- All-in on CUDA for non-Metal backends (246 merged PRs)
- 5 missing primitives that could be ported from CUDA: Hadamard, FFT, BlockMaskedMM, SegmentedMM, QQMatmul
- Linear algebra ops (LU/QR/SVD) are CPU-only in ALL backends
- Won't merge ROCm PR — too much maintenance burden for Apple

## What This Means for Us

Nobody is going to do this upstream. The ROCm C++ path for gfx1151 is ours to build.

The pieces exist across these forks but nobody has assembled them:
- Wave32 ternary kernels (carlosfundora) + APU allocator (goniz) + hipBLASLt integration (lemonade) + native Tensile (TheRock) = the complete stack

We're building that stack. In C++. From scratch.
