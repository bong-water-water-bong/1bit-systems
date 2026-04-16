# Bonsai Q1_0 — ROCm HIP with Wave32 Kernel (gfx1151)

Date: 2026-04-16
Hardware: AMD Ryzen AI Max+ 395 (Strix Halo), 128GB unified, Radeon 8060S
GPU Target: gfx1151 (RDNA 3.5), Wave Size: 32, CUs: 20
OS: CachyOS, kernel 7.0.0-1-cachyos
Compiler: TheRock LLVM (ROCm 7.13 from source)
Backend: llama.cpp HIP with custom Q1_0 vec_dot + dequantize kernels

## Results

```
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3 1.7B Q1_0                | 231.13 MiB |     1.72 B | ROCm       |  99 |           pp512 |      3638.24 ± 11.51 |
| qwen3 1.7B Q1_0                | 231.13 MiB |     1.72 B | ROCm       |  99 |           tg128 |         59.85 ± 0.32 |
| qwen3 4B Q1_0                  | 540.09 MiB |     4.02 B | ROCm       |  99 |           pp512 |      1934.32 ± 10.26 |
| qwen3 4B Q1_0                  | 540.09 MiB |     4.02 B | ROCm       |  99 |           tg128 |         28.58 ± 0.00 |
| qwen3 8B Q1_0                  |   1.07 GiB |     8.19 B | ROCm       |  99 |           pp512 |       1058.17 ± 2.28 |
| qwen3 8B Q1_0                  |   1.07 GiB |     8.19 B | ROCm       |  99 |           tg128 |         21.80 ± 0.00 |
```

## Comparison — Before vs After Q1_0 Kernel

```
Model          Before pp512   After pp512   Speedup    Before tg128   After tg128   Speedup
──────────────────────────────────────────────────────────────────────────────────────────────
Bonsai-1.7B      149.1        3,638.2       24.4x        49.3           59.9         +21%
Bonsai-4B         59.1        1,934.3       32.7x        29.0           28.6         ~same
Bonsai-8B         32.4        1,058.2       32.7x        16.6           21.8         +31%
```

## Comparison — ROCm vs Vulkan

```
Model          ROCm pp512    Vulkan pp512   ROCm tg128    Vulkan tg128
──────────────────────────────────────────────────────────────────────
Bonsai-1.7B    3,638          3,121           59.9          136.8
Bonsai-4B      1,934          1,401           28.6           85.0
Bonsai-8B      1,058            831           21.8           63.8

ROCm prompt processing beats Vulkan: +17% on 1.7B, +38% on 4B, +27% on 8B
Vulkan decode still faster (optimized compute shaders for generation path)
```

## What Changed

Added Q1_0 GPU support to llama.cpp HIP backend:
- `vecdotq_q1_0.cuh` — Wave32 vec_dot kernel for Q1_0 ternary × Q8_1 activation
- `convert.cu` — Q1_0 dequantize kernel (bit=1 → +scale, bit=0 → -scale)
- `mmvq.cu` — Q1_0 dispatch in all switch cases
- `common.cuh` — Q1_0 type traits (qk=128, qr=1, qi=4)
- `ggml-cuda.cu` — Q1_0 registered as supported GPU type

## Method

llama-bench, 3 rounds, pp512 + tg128, all layers on GPU (ngl=99)
TheRock native Tensile GEMM kernels loaded via ROCBLAS_TENSILE_LIBPATH
