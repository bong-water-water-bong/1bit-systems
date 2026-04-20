# Kernel v2.0 Design — RDNA 3.5 Optimized

Reference: RDNA 3.5 ISA, Seb-V GEMM blog (49 TFLOPS), Geramy's guidance

## Architecture Constants (gfx1151)

```
Compute Units:     40
SIMDs per CU:      2 (80 total)
FPUs per SIMD:     2 (dual-issue FMA)
L1 (LDS):         32 KB per CU (64 banks, 4 bytes wide)
L2:                2 MB
L3 (Infinity):    32 MB
Cacheline:        128 bytes
Wave Size:         32 (native)
Max Waves/CU:     32
Max Clock:        2900 MHz
WMMA:             16x16x16, FP16/BF16/IU8/IU4
```

## Two Kernel Paths

### Path 1: Decode GEMV (batch=1, generation)

This is the latency-sensitive path. One token at a time.

**v1.0 (current):** Simple LDS tiling, no dual-issue
**v2.0 targets:**
- Dual-issue `v_dual_fmac_f32` — 2 FP ops per cycle
- VGPR bank management — A in banks 2-3, B in banks 0-1
- 16 precomputed SGPR base addresses
- LDS padding (+4 per row) for bank conflict elimination
- XOR preshuffle for zero-overhead conflict avoidance
- 8x loop unroll with interleaved GMEM loads
- Target: 2x improvement over v1.0

For ternary specifically:
- Pack ternary as INT8 (+1/-1/0)
- Use `v_dot4_i32_iu8` — hardware 4-element dot product
- Each lane processes 4 ternary values per cycle
- 32 lanes × 4 values = 128 ternary values per wave per cycle

### Path 2: Prefill GEMM (batch>1, prompt processing)

This is the throughput path. Many tokens at once.

**v2.0 targets:**
- WMMA `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`
- Pack ternary weights as INT8, activations as INT8 (quantized)
- 512 INT8 ops/clock/CU × 40 CUs × 2.9 GHz = 59.4 TOPS theoretical
- Tiling: 128x128 block, 64x16 wave, 8x8 thread
- LDS double-buffering for GMEM latency hiding
- Target: approach 49 TFLOPS (Seb-V's RDNA3 result)

## LDS Bank Conflict Avoidance

```
32 banks, 4 bytes wide
bank = (address_bytes / 4) % 32

Padding: float tile[BK][BM + 4]  — wastes 12.5% LDS
XOR preshuffle: col' = (row % stride) ^ col  — zero overhead, preferred
```

## VGPR Bank Management (from Seb-V)

```
4 VGPR banks: bank = vgpr_index % 4

For dual-issue v_dual_fmac_f32:
  Operand 1 must be from different bank than operand 2
  
Strategy:
  Matrix A registers → banks 2, 3
  Matrix B registers → banks 0, 1
  Accumulator → any bank (read after write hazard already managed)
```

## Occupancy

```
Target: 4 waves per SIMD = 128 threads per SIMD
Max VGPRs per thread: 64 (at 4 waves)

Register budget per thread:
  8 VGPRs for ternary weight tile
  8 VGPRs for activation tile
  8 VGPRs for accumulator
  8 VGPRs for addresses/temps
  = 32 VGPRs → 8 waves per SIMD (high occupancy)
```

## Implementation Plan

1. Read RDNA 3.5 ISA reference (dual-issue encoding rules)
2. Write v2.0 decode GEMV with v_dot4_i32_iu8
3. Write v2.0 prefill GEMM with WMMA
4. Profile with Radeon GPU Profiler
5. Iterate on VGPR banks and LDS patterns
6. Benchmark against v1.0 and PrismML DP4A
