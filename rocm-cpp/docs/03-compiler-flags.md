# HIP Compiler Flags for gfx1151

## The Critical Flag Nobody Documents

```
--amdgpu-unroll-threshold-local=600
```

Without this flag, prompt processing drops from 4,932 to 1,520 tok/s — a **69% regression**. This was discovered in ROCm 7.2. Nobody in the MLX ecosystem uses it. Nobody documents it.

This flag controls how aggressively the AMDGPU backend unrolls loops that access local (LDS) memory. The default threshold is too conservative for RDNA 3.5's LDS architecture. Setting it to 600 lets the compiler unroll inner loops that touch shared memory, which eliminates branch overhead and enables better occupancy.

## Full HIP AOT Compilation Flags

When compiling HIP kernels ahead-of-time for gfx1151:

```bash
hipcc \
    --offload-arch=gfx1151 \
    -O3 \
    -ffast-math \
    -munsafe-fp-atomics \
    --amdgpu-unroll-threshold-local=600 \
    -o output \
    source.cpp
```

### Flag Breakdown

**`--offload-arch=gfx1151`**
Target architecture. Must match your GPU exactly. gfx1151 = RDNA 3.5 (Strix Halo Radeon 8060S).

**`-O3`**
Maximum optimization. Essential for kernel performance. Without it, the compiler doesn't vectorize or schedule instructions aggressively enough.

**`-ffast-math`**
Allows the compiler to reorder floating-point operations, use reciprocal approximations, and assume no NaN/Inf. Gives 5-15% speedup on math-heavy kernels. Safe for inference (not for training where numerical stability matters).

**`-munsafe-fp-atomics`**
Allows the compiler to use hardware floating-point atomics even when they might give slightly different results than software atomics. On RDNA 3.5, the hardware atomics are fast and accurate enough for inference workloads.

**`--amdgpu-unroll-threshold-local=600`**
The big one. See above. 69% regression without it.

## Wave32 vs Wave64

RDNA 3.5 (gfx1151) natively supports Wave32. This is the optimal warp size for this architecture.

To force Wave32 in your kernels:

```cpp
__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void my_kernel(...) {
    // 128 threads = 4 Wave32 warps
    // NOT 2 Wave64 warps
}
```

### Why Wave32 Matters

- Wave32 has half the latency of Wave64 for the same instruction
- Better occupancy on RDNA 3.5 — more waves fit in the SIMD
- carlosfundora proved this: Wave32 ternary kernels hit 209 t/s vs generic Wave64

### Setting Wave Size in CMake

```cmake
set(HIP_WAVE_SIZE 32)
target_compile_options(my_target PRIVATE
    --offload-arch=gfx1151
    -mwavefrontsize64=0  # Force Wave32
)
```

## CMake Integration

```cmake
# Find HIP
find_package(hip REQUIRED)

# Set global flags for gfx1151
set(CMAKE_HIP_ARCHITECTURES gfx1151)
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -ffast-math -munsafe-fp-atomics --amdgpu-unroll-threshold-local=600")

# Add a HIP kernel library
add_library(my_kernels
    kernels/ternary_gemv.hip
    kernels/dequant.hip
)
target_link_libraries(my_kernels hip::device)
```
