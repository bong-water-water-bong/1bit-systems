# rocm-cpp

Native ROCm C++ for Strix Halo (gfx1151). Built from scratch.

## What This Is

A pure C++ inference and compute stack targeting AMD Strix Halo APUs. No Python. No wrappers. Direct HIP kernels on RDNA 3.5 silicon.

## The Problem

- No optimized Tensile/rocBLAS GEMM kernels exist for gfx1151
- No ternary-aware kernel path exists on ROCm anywhere
- Everyone falls back to generic dequantize-then-matmul (the slowest path)
- Missing compiler flags cause 69% regression that nobody documents
- hipBLASLt is "unsupported" on gfx1151 but works — undocumented

## The Goal

Own the entire inference path in C++ for gfx1151:
- Custom Wave32 HIP kernels for ternary (1-bit) models
- Native GEMM kernels tuned for RDNA 3.5
- OpenAI-compatible API server
- Zero Python dependencies at runtime

## What We Know

### Compiler Flags That Matter

- `--amdgpu-unroll-threshold-local=600` — without this, 69% prompt regression
- `-O3 -ffast-math -munsafe-fp-atomics` — standard HIP AOT flags
- Wave32, NOT Wave64 — RDNA 3.5 native warp size

### Runtime Environment

- `HSA_OVERRIDE_GFX_VERSION=11.5.1`
- `HSA_ENABLE_SDMA=0` — required for Strix Halo
- `ROCBLAS_USE_HIPBLASLT=1` — 2.6x prompt speedup on 4-bit (not relevant for ternary)

### Benchmarks (Vulkan baseline)

Qwen3-Coder-Next-GGUF on CachyOS kernel 7.0:
- Prompt: 146-279 t/s (scales up with context length)
- Generation: 47.4 t/s sustained — flat line, no degradation at 2K tokens

### Reference Implementations

- `carlosfundora/llama.cpp-1-bit-turbo` — Wave32 HIP ternary kernels, Q1_0_G128, 209 t/s on RX 6700 XT
- `lemonade-sdk` patches — warp-cooperative GEMV, hipBLASLt, slab allocator
- `goniz/mlx` — flash attention, APU allocator fixes
- `PrismML-Eng/mlx` — 1-bit affine quantization branch

### Architecture (BitNet b1.58-2B-4T)

- hidden_size: 2560, layers: 30, attention_heads: 20
- kv_heads: 5, intermediate: 6912, vocab: 128256
- rope_theta: 500000.0, activation: relu2 (NOT SiLU)
- Llama 3 tokenizer

## Structure

```
kernels/       — Custom HIP kernels (Wave32 ternary, GEMM)
inference/     — Model loading, forward pass, KV cache
server/        — HTTP API server (OpenAI-compatible)
tokenizer/     — BPE tokenizer (C++)
tools/         — Benchmarks, export utilities
```

## Building

Requires:
- ROCm (TheRock build from source for gfx1151)
- CMake + Ninja
- gfx1151 hardware (AMD Strix Halo)

## Status

Setting up. TheRock (ROCm 7.13) building from source for gfx1151 with full BLAS stack.
