# Wave32 Kernel Design for RDNA 3.5

## The Core Insight

Every existing ROCm kernel library defaults to Wave64 (GCN legacy). RDNA 3.5 is Wave32 native. Running Wave64 on RDNA 3.5 works but wastes half the potential — each instruction takes twice the cycles it needs to.

## RDNA 3.5 SIMD Layout

- 40 Compute Units on Strix Halo
- Each CU has 2 SIMD units
- Each SIMD processes Wave32 natively
- Wave64 is emulated by issuing 2x Wave32 — double the latency
- LDS (Local Data Share): 64KB per CU
- Register file: 256 VGPRs per SIMD

## Thread Block Design

For Wave32 on gfx1151, optimal thread blocks are multiples of 32:

```cpp
// 128 threads = 4 waves of 32
__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void kernel(...) { }

// 256 threads = 8 waves of 32 (higher occupancy, more register pressure)
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel(...) { }
```

128 threads (4 waves) is the sweet spot for most inference kernels — enough parallelism without exhausting registers.

## Wave32 Intrinsics

RDNA 3.5 provides Wave32-specific cross-lane operations:

```cpp
// Warp-level reduction (Wave32)
float sum = value;
sum += __shfl_xor(sum, 16);  // Reduce across 32 lanes
sum += __shfl_xor(sum, 8);
sum += __shfl_xor(sum, 4);
sum += __shfl_xor(sum, 2);
sum += __shfl_xor(sum, 1);

// Warp ballot (32-bit mask, not 64-bit)
unsigned int mask = __ballot(predicate);

// Warp-level broadcast
float val = __shfl(source, lane_id);
```

## LDS (Shared Memory) Usage

64KB LDS per CU. On RDNA 3.5, LDS bandwidth is high but latency-sensitive — this is why `--amdgpu-unroll-threshold-local=600` matters so much.

```cpp
__shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

// Load into LDS
tile[threadIdx.y][threadIdx.x] = input[global_idx];
__syncthreads();

// Compute from LDS (fast, broadcast-capable)
float result = tile[threadIdx.x][threadIdx.y];
```

### Bank Conflict Avoidance

LDS has 32 banks (matching Wave32). Accessing consecutive elements from consecutive lanes = no conflict. Stride-32 access = all lanes hit the same bank = serialized.

Pad shared memory arrays by 1 to break stride patterns:
```cpp
__shared__ float tile[32][33]; // 33 not 32
```

## Memory Coalescing

Global memory access must be coalesced — consecutive threads read consecutive addresses:

```cpp
// GOOD: coalesced — thread i reads element i
float val = input[blockIdx.x * blockDim.x + threadIdx.x];

// BAD: strided — thread i reads element i * stride
float val = input[threadIdx.x * stride]; // Serialized memory access
```

On Strix Halo's unified memory, the penalty for uncoalesced access is less severe than discrete GPUs (no PCIe), but still significant — scattered reads waste memory bandwidth.

## Occupancy

Target 4-8 waves per SIMD for good latency hiding:
- 4 waves = 128 threads per SIMD, uses 64 VGPRs per thread max
- 8 waves = 256 threads per SIMD, uses 32 VGPRs per thread max

Check register usage:
```bash
hipcc --offload-arch=gfx1151 -Rpass-analysis=kernel-resource-usage kernel.hip
```

## Reference: carlosfundora's Approach

From `llama.cpp-1-bit-turbo` — the only working Wave32 ternary kernel on RDNA:

- 128 threads per block (4 Wave32 warps)
- Q1_0_G128 quantization format (128-element groups)
- Each thread processes one group
- HIP-native dequantize + dot product fused in single kernel
- No intermediate buffer — ternary values decoded and multiplied in-register
- Result: 209 tok/s on RX 6700 XT for 1.7B ternary model
