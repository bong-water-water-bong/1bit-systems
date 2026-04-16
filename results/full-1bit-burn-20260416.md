# Full 1-Bit Model Burn — ROCm HIP gfx1151 + TheRock Native Tensile

Date: 2026-04-16
Hardware: AMD Ryzen AI Max+ 395 (Strix Halo), 128GB unified, Radeon 8060S (gfx1151)
Stack: PrismML prism branch (e2d67422c, 8796) + TheRock ROCm 7.13 from source
Method: llama-bench, 3 rounds, pp512 + tg128, ngl=99

## Results

```
| model                                              |       size |     params | backend | ngl |    test |              t/s |
| -------------------------------------------------- | ---------: | ---------: | ------- | --: | ------: | ----------------: |
| Bonsai-1.7B (Q1_0)                                 | 231.13 MiB |     1.72 B | ROCm    |  99 |   pp512 |  4172.17 ± 16.76 |
| Bonsai-1.7B (Q1_0)                                 | 231.13 MiB |     1.72 B | ROCm    |  99 |  tg128  |    232.43 ± 0.84 |
| Bonsai-4B (Q1_0)                                   | 540.09 MiB |     4.02 B | ROCm    |  99 |   pp512 |   2014.14 ± 4.65 |
| Bonsai-4B (Q1_0)                                   | 540.09 MiB |     4.02 B | ROCm    |  99 |  tg128  |    125.34 ± 0.96 |
| Bonsai-8B (Q1_0)                                   |   1.07 GiB |     8.19 B | ROCm    |  99 |   pp512 |   1278.13 ± 3.51 |
| Bonsai-8B (Q1_0)                                   |   1.07 GiB |     8.19 B | ROCm    |  99 |  tg128  |     94.05 ± 0.08 |
| BitNet-2B-4T (Q1_0)                                | 538.03 MiB |     2.41 B | ROCm    |  99 |   pp512 |   3030.38 ± 3.11 |
| BitNet-2B-4T (Q1_0)                                | 538.03 MiB |     2.41 B | ROCm    |  99 |  tg128  |    110.46 ± 0.30 |
| Qwen3-Coder-Next 80B-A3B (IQ1_S, 1.56bpw)         |  17.64 GiB |    79.67 B | ROCm    |  99 |   pp512 |    642.59 ± 8.95 |
| Qwen3-Coder-Next 80B-A3B (IQ1_S, 1.56bpw)         |  17.64 GiB |    79.67 B | ROCm    |  99 |  tg128  |     50.52 ± 0.03 |
| Llama-4-Scout 17Bx16E (IQ1_S, 1.56bpw)            |  27.24 GiB |   107.77 B | ROCm    |  99 |   pp512 |    323.26 ± 2.29 |
| Llama-4-Scout 17Bx16E (IQ1_S, 1.56bpw)            |  27.24 GiB |   107.77 B | ROCm    |  99 |  tg128  |     21.24 ± 0.00 |
| BitNet-2B-4T (TQ1_0, 1.69bpw)                     |   1.02 GiB |     2.41 B | ROCm    |  99 |   pp512 |    272.11 ± 0.49 |
| BitNet-2B-4T (TQ1_0, 1.69bpw)                     |   1.02 GiB |     2.41 B | ROCm    |  99 |  tg128  |     49.95 ± 0.01 |
```

## Summary Table

```
Model                     Quant    Size       pp512 t/s    tg128 t/s
─────────────────────────────────────────────────────────────────────
Bonsai-1.7B               Q1_0     231 MB     4,172         232
BitNet-2B-4T              Q1_0     538 MB     3,030         110
Bonsai-4B                 Q1_0     540 MB     2,014         125
Bonsai-8B                 Q1_0     1.07 GB    1,278          94
Qwen3-Coder-Next 80B      IQ1_S    17.6 GB      643          51
Llama-4-Scout 108B         IQ1_S    27.2 GB      323          21
BitNet-2B-4T              TQ1_0    1.02 GB      272          50
```

## Q1_0 vs TQ1_0 (BitNet-2B-4T)

```
Format      Size      pp512 t/s    tg128 t/s    Speedup
──────────────────────────────────────────────────────────
Q1_0        538 MB    3,030.4       110.5        ← DP4A kernel
TQ1_0       1.02 GB     272.1        50.0        ← generic path
                        11.1x        2.2x
```

## How to Replicate

```bash
# 1. Build TheRock from source
git clone https://github.com/ROCm/TheRock.git && cd TheRock
git submodule update --init --recursive
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DTHEROCK_AMDGPU_TARGETS=gfx1151 -DTHEROCK_ENABLE_BLAS=ON
cmake --build build --parallel $(nproc)

# 2. Build PrismML llama.cpp (prism branch)
git clone https://github.com/PrismML-Eng/llama.cpp.git && cd llama.cpp
git checkout prism
cmake -B build-rocm -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
    -DCMAKE_HIP_COMPILER=$HOME/therock/build/compiler/amd-llvm/dist/lib/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=$HOME/therock/build/compiler/amd-llvm/dist/lib/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=$HOME/therock/build/compiler/amd-llvm/dist/lib/llvm/bin/clang++
cmake --build build-rocm --parallel $(nproc)

# 3. Environment
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_ENABLE_SDMA=0
export LD_LIBRARY_PATH=$HOME/therock/build/math-libs/BLAS/rocBLAS/dist/lib:/opt/rocm/lib
export ROCBLAS_TENSILE_LIBPATH=$HOME/therock/build/math-libs/BLAS/rocBLAS/dist/lib/rocblas/library

# 4. Run
./build-rocm/bin/llama-bench -m Bonsai-8B.gguf -ngl 99 -p 512 -n 128 -r 3
```

GCC 15 patches: https://github.com/stampby/rocm-cpp/docs/02-therock-build.md
