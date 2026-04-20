# Kernel v2 GEMM — Ternary Prefill via WMMA

## Why a separate kernel

v2 GEMV was the wrong target. Batch=1 decode cannot amortize the float→int8→float round-trip that INT8 dot-product requires, and on RDNA 3.5 a fused-float MADD chain (`v1`) wins the decode path by 15–25%. Shelving v2 GEMV permanently.

v2 GEMM is the right target. In **prefill** (pp512, pp2048) the batch is the full prompt, often 128–512 tokens. One activation quantization pass serves all output rows, making int8 WMMA a net win on paper:

- WMMA `v_wmma_i32_16x16x16_iu8_w32` on gfx1151: one instruction = 16×16×16 = 4096 INT8 MACs per wave per cycle (on paper). Theoretical 2× the throughput of float FMA on the same silicon.
- rocBLAS FP16 GEMM is the incumbent path. Beating it on BitNet FFN shapes (2560×6912, 6912×2560) is the success criterion.

## Target shapes

BitNet-2B FFN at prefill batch:

| M (rows) | N (batch) | K (inner) | What |
|---:|---:|---:|---|
| 2560 | 128 | 2560 | Q/K/V/O projection, pp512 slice |
| 6912 | 128 | 2560 | FFN up (gate, up) |
| 2560 | 6912 | 128 | FFN down |
| 2560 | 512  | 2560 | pp2048 slice |

## Skeleton status (2026-04-16)

- `kernels/ternary_gemm_v2.hip` compiles and runs.
- Grid: `(M/16, B/16)` blocks, block = 32 lanes = 1 wave.
- Each wave walks K in WMMA_K=16 steps.
- **Does not yet issue the real WMMA intrinsic** — uses a scalar `v_dot4_i32_iu8` fallback that only fills column 0 of each 16×16 output tile. This is intentional: it lets us validate the launch config, packing, and scale arithmetic before wiring up the register-layout-sensitive WMMA builtin.
- `tools/test_gemm_v2` measures v2 skeleton vs `rocblas_hgemm` on the shapes above.

Skeleton baseline (gfx1151, Radeon 8060S, TheRock 7.13):

```
M=2560 K=2560 B=128 : v2 119 us, rocBLAS FP16  82 us  (0.69x)
M=6912 K=2560 B=128 : v2 482 us, rocBLAS FP16 137 us  (0.28x)
M=2560 K=6912 B=128 : v2 295 us, rocBLAS FP16 236 us  (0.80x)
M=2560 K=2560 B=512 : v2 944 us, rocBLAS FP16 202 us  (0.21x)
```

These numbers are a floor, not a target. The WMMA pass should give a 10–50× step just by populating all 256 cells per tile instead of 16.

## v2.1 — first real WMMA

Wire the actual intrinsic:

```cpp
using int32x8 = __attribute__((__vector_size__(32))) int;
int32x8 c_acc = {0};
c_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
    /*sign_a=*/ true, a_int32, /*sign_b=*/ true, b_int32, c_acc, /*clamp=*/ false);
```

Key subtleties:

- `a_int32` and `b_int32` are one int32 per lane (4 INT8 values packed), not per-row/per-col. The WMMA instruction expects a specific lane → (row, col) mapping that differs between Wave32 and Wave64.
- `c_acc` is per-lane 8× int32, so 256 total ints across 32 lanes = 16×16 C tile. Each lane writes back 8 output cells.
- On gfx1151 the WMMA result layout is "AccumulatorLayoutVOPC16" — four groups of 4 consecutive lanes each hold a row stripe.

## v2.2 — LDS staging

After WMMA is correct, move from per-iter operand loads to LDS-staged tiles:

- Block: 4 waves = 128 threads, producing a 64×64 output tile per block.
- LDS holds one 64×16 slab of A-ints and one 16×64 slab of B-ints.
- Double-buffer so loads of the next K slab overlap with WMMA on the current one.

## v2.3 — quantization amortization

In the skeleton, each WMMA step computes a new activation scale for the 16 K-values in that slice. In v2.3, compute a single per-batch-per-tile scale (block-cooperative reduce once at block entry) and reuse it across the whole K loop. This is the real cost of WMMA; doing it once is what makes int8 cheaper than float for prefill.

## What success looks like

v2.3 should hit at least 1.5× rocBLAS FP16 on BitNet FFN shapes at B=128, and 2× at B=512. Anything less means we're memory-bound on the activation tensor, and the DP4A densification doesn't matter; we'd be optimizing the wrong thing.

If v2.3 under-performs on both B=128 and B=512, the right call is to shelve v2 GEMM too and document that ternary inference on gfx1151 is a memory-bandwidth problem solved by v1 already.

## Not in scope

- Dual-issue `v_dual_fmac_f32` tuning (v1 research, not WMMA)
- VGPR bank management (v1 research)
- XOR preshuffle (v1 research)
- Prefill batching beyond B=512 (need >32 MB of activation LDS budget)
- Grouped MoE GEMM (Qwen3-Coder-Next 80B path — different kernel entirely)
