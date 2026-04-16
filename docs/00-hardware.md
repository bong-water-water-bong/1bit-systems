# Hardware — Strix Halo (gfx1151)

## The Chip

AMD Ryzen AI Max+ 395 (Strix Halo)
- GPU: Radeon 8060S (RDNA 3.5, gfx1151)
- Compute Units: 40
- Wave Size: 32 (Wave32) — NOT 64. This is critical for kernel design.
- VRAM: Shared unified memory with CPU (up to 128GB configurable in BIOS)
- Memory Bandwidth: ~256 GB/s (LPDDR5X)
- ROCm Target: gfx1151

## Why This Matters

Strix Halo is an APU — CPU and GPU share the same memory. There is no PCIe bottleneck for data transfer. The GPU can directly access CPU memory and vice versa. This changes everything about memory allocation strategy.

Most ROCm software assumes discrete GPUs with dedicated VRAM. Strix Halo breaks those assumptions:
- GTT (system memory) and VRAM are the same physical memory
- `HSA_ENABLE_SDMA=0` is required — the SDMA engine doesn't work right on unified memory
- `hsa_amd_svm_attributes_set()` spams non-fatal errors — ignore them
- 1GB dedicated VRAM allocation actually gives BETTER inference than 4GB (ROCm issue #3128)

## BIOS Configuration

- Set VRAM allocation in BIOS (UMA Frame Buffer Size)
- 1GB dedicated performs better than 4GB for inference workloads
- Rest of memory is shared and accessible by both CPU and GPU

## Kernel

CachyOS kernel 7.0.0-1-cachyos (mainline)
- Arch-based, optimized for AMD
- Ships with working amdgpu driver for gfx1151

## Identifying Your Hardware

```bash
# Check GPU target
rocminfo | grep gfx

# Check compute units
rocminfo | grep "Compute Unit"

# Check memory
rocm-smi --showmeminfo vram

# Check GPU utilization
rocm-smi --showuse
```
