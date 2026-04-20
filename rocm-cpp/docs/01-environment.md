# Environment Setup

## Operating System

CachyOS (Arch-based) — native packages, no Flatpak, no containers.

```bash
# System packages needed
sudo pacman -S --needed \
    base-devel cmake ninja git \
    rocm-hip-sdk rocm-opencl-sdk \
    python python-pip
```

Note: System ROCm packages provide the base. TheRock build from source provides optimized libraries for gfx1151.

## Runtime Environment Variables

These MUST be set for gfx1151. Without them, performance drops or things crash silently.

```bash
# Required for gfx1151
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_ENABLE_SDMA=0

# Required for hipBLASLt (2.6x prompt speedup on 4-bit models)
export ROCBLAS_USE_HIPBLASLT=1

# Optional but recommended
export HIP_VISIBLE_DEVICES=0
export GPU_MAX_HW_QUEUES=8
```

### Why Each Variable Matters

**HSA_OVERRIDE_GFX_VERSION=11.5.1**
ROCm's runtime looks up pre-compiled kernels by GPU target ID. gfx1151 is new and many libraries don't ship kernels for it. This override tells the runtime to use gfx1151 kernels when available, falling back to compatible ones.

**HSA_ENABLE_SDMA=0**
The System DMA engine on Strix Halo's unified memory architecture causes data corruption or hangs. Disabling it forces the GPU to do its own copies, which is actually fine on unified memory since there's no PCIe to cross anyway.

**ROCBLAS_USE_HIPBLASLT=1**
hipBLASLt provides optimized GEMM kernels that are 2.6x faster for prompt processing. Officially "unsupported" on gfx1151 (ROCm issue #5643) but works. Only helps bf16/fp16 GEMM — irrelevant for ternary/1-bit inference.

## Shell Configuration

Add to `~/.zshrc` or `~/.bashrc`:

```bash
# ROCm C++ environment
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_ENABLE_SDMA=0
export ROCBLAS_USE_HIPBLASLT=1
export HIP_VISIBLE_DEVICES=0

# ROCm paths (system install)
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip
export PATH=$ROCM_PATH/bin:$PATH

# TheRock paths (from-source build)
export THEROCK_PATH=$HOME/therock/build
export LD_LIBRARY_PATH=$THEROCK_PATH/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

## Verifying the Environment

```bash
# GPU is visible
rocminfo | grep gfx

# HIP compiler works
hipcc --version

# GPU responds
rocm-smi --showuse

# Quick sanity test — compile and run a trivial HIP program
cat > /tmp/hip_test.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
    int count;
    hipGetDeviceCount(&count);
    printf("GPU devices: %d\n", count);
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    printf("Device: %s\n", props.name);
    printf("Compute Units: %d\n", props.multiProcessorCount);
    printf("GCN Arch: %s\n", props.gcnArchName);
    return 0;
}
EOF
hipcc /tmp/hip_test.cpp -o /tmp/hip_test && /tmp/hip_test
```
