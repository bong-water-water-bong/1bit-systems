# Known Issues — ROCm on gfx1151

## Critical

### No Optimized Tensile GEMM for gfx1151 (ROCm #2882)
- rocBLAS ships generic kernels, not RDNA 3.5 tuned ones
- Overriding to gfx1100 (RDNA3) gives 3.5x speedup — proves the issue is missing kernels, not hardware
- Fix: build TheRock from source with `-DTHEROCK_AMDGPU_TARGETS=gfx1151` to generate native Tensile kernels

### 69% Prompt Regression Without Unroll Flag
- Missing `--amdgpu-unroll-threshold-local=600` causes prompt processing to drop from 4,932 to 1,520 tok/s
- Discovered in ROCm 7.2, nobody documents it
- Fix: add the flag to all HIP AOT compilation

### GTT Memory Overflow (ROCm #519)
- PyTorch uses GTT (system memory region) instead of VRAM on Strix Halo
- Causes OOM even when GPU memory is available
- Related to unified memory architecture confusion in the allocator
- Fix: set `HIP_VISIBLE_DEVICES=0` and use explicit memory placement

## Important

### hipBLASLt "Unsupported" (ROCm #5643)
- hipBLASLt gives 2.6x prompt speedup but is officially "unsupported" on gfx1151
- Works anyway with `ROCBLAS_USE_HIPBLASLT=1`
- lemonade-sdk uses `MLX_ROCM_HIPBLASLT` env var to toggle
- Only relevant for fp16/bf16 GEMM (not ternary)

### 1GB VRAM Better Than 4GB (ROCm #3128)
- Setting 1GB dedicated VRAM in BIOS gives better inference than 4GB
- Memory allocation anomaly on unified memory
- Likely related to how the allocator partitions dedicated vs shared regions

### SDMA Engine Broken
- `HSA_ENABLE_SDMA=0` is mandatory on Strix Halo
- SDMA causes data corruption or hangs on unified memory
- GPU doing its own copies is fine — no PCIe to cross

### hsa_amd_svm_attributes_set() Spam
- Non-fatal error messages spam stdout/stderr
- Ignore them — unified memory attribute setting that doesn't apply to APU
- Harmless but annoying

### PyTorch ROCm Version Mismatch
- PyTorch 2.9.1+rocm6.3 segfaults on ROCm 7.2.1
- Must use PyTorch 2.11.0+rocm7.2 to match system ROCm
- Always match PyTorch ROCm version to system ROCm version

### hipBLASLt Deadlock on SIGTERM
- hipBLASLt library doesn't respond to SIGTERM during computation
- Must use SIGKILL to terminate processes using hipBLASLt
- Affects server shutdown and process management

## Minor

### hipSPARSELt Not Available
- gfx1151 is manually excluded in TheRock's target filtering
- Non-critical — sparse matrix support not needed for inference
- Warning during cmake configure is expected

### Wave64 Default
- All ROCm kernel libraries default to Wave64 (GCN legacy)
- Must explicitly set Wave32 for RDNA 3.5 optimal performance
- Not a bug per se, but a performance trap
