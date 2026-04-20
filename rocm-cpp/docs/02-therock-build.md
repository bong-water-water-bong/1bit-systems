# Building TheRock (ROCm from Source)

## Why Build from Source

The system ROCm packages ship generic kernels. For gfx1151, that means:
- No optimized Tensile GEMM kernels for RDNA 3.5 ISA
- rocBLAS falls back to generic kernels (3.5x slower than tuned ones)
- hipBLASLt may not have gfx1151 entries at all

Building TheRock from source with `gfx1151` target generates native Tensile kernels tuned for your exact silicon.

## What TheRock Contains

TheRock is AMD's unified ROCm build system. One repo, one build, everything:

- **LLVM/Clang** — the compiler (biggest component, ~8300 files)
- **HIP** — the runtime API
- **rocBLAS** — BLAS library with Tensile-generated GEMM kernels
- **hipBLASLt** — lightweight GEMM library (2.6x prompt speedup)
- **rocRoller** — kernel code generator for gfx1151
- **rocPRIM** — GPU primitives
- **rocFFT** — FFT library
- **MIOpen** — deep learning primitives
- **composable_kernel** — performance-portable GPU kernels
- **rocSPARSE/hipSPARSE** — sparse matrix libraries
- **rocSOLVER/hipSOLVER** — linear algebra solvers

## Prerequisites

```bash
sudo pacman -S --needed \
    base-devel cmake ninja git python \
    patchelf libdrm numactl \
    rocm-hip-sdk
```

## Clone

```bash
cd ~
git clone https://github.com/ROCm/TheRock.git therock
cd therock
git submodule update --init --recursive
```

Note: This pulls a LOT of submodules. Give it time.

## Configure

```bash
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTHEROCK_AMDGPU_TARGETS=gfx1151 \
    -DTHEROCK_DIST_AMDGPU_FAMILIES=gfx115X-all \
    -DTHEROCK_ENABLE_BLAS=ON
```

### What the flags mean

- `THEROCK_AMDGPU_TARGETS=gfx1151` — compile kernels for this GPU only
- `THEROCK_DIST_AMDGPU_FAMILIES=gfx115X-all` — distribution target family
- `THEROCK_ENABLE_BLAS=ON` — build rocBLAS, hipBLASLt, rocRoller (the performance-critical libraries)

### What gets excluded

- `hipSPARSELt` — manually excluded for gfx1151 (not supported yet, non-critical warning)
- This is expected and harmless

## Build

```bash
# Full parallel build — uses all cores
cmake --build build --parallel $(nproc)

# Or run in background with logging
nohup cmake --build build --parallel $(nproc) > ~/therock-build.log 2>&1 &
```

### Build Times

On Strix Halo (Ryzen AI Max+ 395):
- LLVM is the long pole: ~8300 files, 2-3 hours
- Total build: 3-4 hours
- Disk space needed: ~50GB for build directory

### Monitoring Progress

```bash
# Watch the log
tail -f ~/therock-build.log

# Check which component is building
tail -5 ~/therock-build.log

# LLVM progress (the big one)
grep -c "Building CXX" ~/therock-build.log
```

## After Build

The built libraries land in `~/therock/build/`. Set your environment to use them:

```bash
export THEROCK_PATH=$HOME/therock/build
export LD_LIBRARY_PATH=$THEROCK_PATH/lib:$LD_LIBRARY_PATH
```

## Known Issues

1. **Cache path mismatch** — if you see "CMakeCache.txt directory is different", delete `build/` and reconfigure. Happens when the repo was previously configured in a container or different path.

2. **patchelf required** — TheRock needs `patchelf` for RPATH fixups. Install it before building.

3. **hipSPARSELt exclusion warning** — expected for gfx1151, harmless.

4. **Disk space** — the build tree is large. Ensure 50GB+ free.
