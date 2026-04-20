# NPU-Kernel-Design

Design notes for the straight-C++/Peano reimplementation of IRON's INT8 GEMM,
targeting the Strix Halo XDNA2 NPU for BitNet prefill. Source reference:
amd/IRON @ `devel` (cloned 2026-04-20 to `/tmp/IRON-read`).

## AIE2P hardware layout (Strix Halo)

| Property              | Value                       | Source                                                                  |
|-----------------------|-----------------------------|-------------------------------------------------------------------------|
| Rows per column       | 4 compute tiles             | `iron/operators/gemm/design.py:151` (`n_aie_rows = 4`)                  |
| Columns on Strix Halo | 8                           | `design.py:209-212` ("NPU2 (Strix/Strix Halo/Krackan) has 8 columns"), IRON #55 |
| Shim row              | row 0                       | `design.py:325` (tiles indexed `[col][row]`, shim placements use `row=0`)|
| Mem-tile row          | row 1                       | `design.py:384,408,433` (all `Tile(col, 1)` placements)                 |
| Core rows             | rows 2-5                    | `design.py:326` (`core_tiles = tiles[2:]`)                              |
| Vector store width    | 512 bit                     | `aie_kernels/aie2p/zero.cc:22` (`r = 512 / (sizeof(T)*8)`)              |
| INT8 MAC shape        | 8 x 8 x 8 (r,s,t)           | `aie_kernels/aie2p/mm.cc:366-378`, confirmed IRON #93 body              |
| Peak INT8             | ~50 TOPS array              | IRON #55 xrt-smi validate + andrej comment                              |
| bf16 : int8 ratio     | ~1:8 (not 1:2 like AIE-ML)  | IRON #55 comment 3706297989 (andrej)                                    |

The 50 TOPS figure comes straight from `xrt-smi validate` in IRON #55. AngryLoki
in that thread notes 1.8 GHz x 512 INT8 MACs/cycle/core x 32 cores ≈ 58.8 TOPS
theoretical; IRON's 64x64x64 design hits ~8% of that.

## INT8 per-core tile layout

The kernel is a single template: `matmul_vectorized_2x2_mmul` at
`aie_kernels/aie2p/mm.cc:83-208`. It uses `aie::mmul<r,s,t,T_in,T_in,accauto>`
and unrolls 2x in M and 2x in N, giving **4 accumulator registers** (C00, C01,
C10, C11 at `mm.cc:147-150`) that stay resident across the K reduction.

For INT8 the wrapper is `matmul_vectorized_8x8x8_i8_i32` at `mm.cc:396-410` —
r=s=t=8, so each outer 2x2 step lands a **16x16** block of C from two 8-wide A
rows and two 8-wide B columns. The K-reduction loop is `for i in colA` at
`mm.cc:152-177`; each iteration issues 4 `mac` ops (`mm.cc:173-176`).

Per-core L1 buffers (from `design.py:277-279`, default 64x64x64, `i8` in /
`i32` out):
- A: `m*k = 64*64 = 4096 B`, ping-pong x2 = 8 KiB
- B: `k*n = 64*64 = 4096 B`, ping-pong x2 = 8 KiB
- C: `m*n*4 = 16384 B` (i32), ping-pong x2 = 32 KiB. Total ~48 KiB of 64 KiB.

PR #94 bumps INT8 min tile to (16,8,16) (`op.py` diff, line ~66) because the
2x r and 2x t unroll requires `m % 16 == 0`, `n % 16 == 0` — static_assert at
`mm.cc:372-374`.

## DMA descriptor pattern

One sentence: **A broadcasts across columns and distributes 4-ways across rows
from 4 shim DMAs; B broadcasts across rows and distributes 8-ways across
columns from 8 shim DMAs; C is joined 4-to-1 per column and drained through
8 shim DMAs, all on a 3-level L3 -> L2 -> L1 ObjectFifo pipeline with
`fifo_depth=2` ping-pong.**

L3->L2 fifos (`design.py:360,395`): one per shim-mem pair, carry a
`mem_tile_m_A x k` = 256x64 slab of A and a `k x n` = 64x64 slab of B.

L2->L1 split/forward (`design.py:376-388, 400-409`): `.split()` on A with
`dims_to_stream = [(m/r, r*k), (k/s, s), (r, k), (s, 1)]` performs the
**r-by-s tile reordering** on the mem-tile DMA — this is the 4D strided
descriptor that Peano HW BD regs need to replicate. B uses
`[(k/s, s*n), (n/t, t), (s, n), (t, 1)]` (row-major, `design.py:399`) or
the `b_col_maj` variant at `design.py:397`.

C drain (`design.py:413-421`): mem-tile runs the **inverse reorder**
`[(m/r, r*n), (r, t), (n/t, r*t), (t, 1)]`, joining 4 rows of 16x16 C
fragments into one 64x64 tile that streams back to shim.

Runtime-side `npu_dma_memcpy_nd` descriptors for the full M,K,N walk are built
by `TensorTiler2D.group_tiler` / `step_tiler` at `design.py:517-545` and
dispatched in a double-buffered ping-pong at `design.py:569-577`.

## Stream-switch configuration

Each ObjectFifo lowers to a pair of stream-switch circuits:
- shim DMA <-> mem tile: 4 A channels + 8 B channels + 8 C channels =
  20 shim->mem circuit-routed streams (`design.py:360, 395, 416`).
- mem tile <-> compute tile: A uses `.split()` -> 4 streams/column-group
  (`design.py:379`); B uses `.forward()` -> 1 stream/column (`design.py:403`);
  C uses `.join()` -> 4 streams/column (`design.py:428`).

No tile-to-tile (cascade/LDM) streams; all compute cores pull from mem tiles.
This means stream-switch is **bipartite shim<->mem<->core** only — simpler than
a cascade design, which is good news for hand-rolling the MLIR.

RTPs at `design.py:339-350` are 2x int32 per core (`rtp_K_div_k`,
`rtp_n_tiles_per_core`), written by runtime via `use_write_rtp=True`.

## Ternary (int2) adaptation

Weights land in L1 as packed int2 (2 bits/elem, 4 elems/byte). Required
on-tile work before the MAC pipeline:

1. **Decompress int2 -> int8** into a scratch B tile. AIE2P has a 512-bit
   vector shuffle; `aie::unpack` / `aie::shift_bytes` on a 128-elem int2
   word give 128-elem int8 in ~4-8 cycles.
2. `k*n = 64*64 = 4096` i8 outputs per B tile -> **32 vector unpacks** per K
   tile.
3. The shuffle unit runs on the VPERM slot; MAC runs on MUL/MAC slot. On
   AIE2P these are separate issue slots, so the decompress **should overlap**
   with the `mac` chain at `mm.cc:173-176` as long as we double-buffer the
   decompressed B.
4. Cost model: 32 unpacks/tile vs 8 mac/iter x 8 K-iters = 64 macs. Unpack is
   ~50% of MAC count — fits in the VPERM slot without stealing MAC cycles
   IF we decompress into a second B' scratch while the previous B' drives the
   MAC.
5. Storage hit: B' scratch is another 4096 B (1 KiB) -> push per-core L1 from
   ~48 KiB to ~49 KiB, well inside 64 KiB.

Alternative: keep weights as int2 and build a LUT-mpGeMM kernel
(vlut.cpp-style). Defer — one-shot unpack-to-int8 reuses IRON's MAC pipe
unchanged.

## Reimplementation checklist (>=8 items)

1. **Port `matmul_vectorized_2x2_mmul`** (`mm.cc:83-208`) byte-for-byte; it is
   the hot loop and Peano's AIE-API headers match.
2. **Port `zero_vectorized`** (`aie_kernels/aie2p/zero.cc:20-31`) using
   `aie::zeros<int32,16>()` for the i32 accumulator tile.
3. **Replace the ObjectFifo python with hand-written MLIR** for 4 A, 8 B,
   8 C circuits mirroring `design.py:358-437`.
4. **Encode the L2->L1 A reorder** 4D BD pattern from `design.py:368-375`
   (`(m/r,r*k),(k/s,s),(r,k),(s,1)`) directly into shim+mem BD registers.
5. **Encode the L2->L1 B reorder** from `design.py:397-399` — both
   `b_col_maj` and row-major variants.
6. **Encode the C join inverse reorder** from `design.py:413-415`
   (`(m/r,r*n),(r,t),(n/t,r*t),(t,1)`).
7. **Implement the TensorTiler2D walk** from `design.py:517-545` as a
   runtime BD list; ping-pong tb_max_n_rows=4 per `design.py:514`.
8. **Ship an int2->int8 unpack prologue** ahead of the MAC loop at
   `mm.cc:152`; keep ping-pong scratch so VPERM overlaps MAC.
9. **Mirror the kernel flag macros** `-Di8_i32_ONLY`, `-DDIM_M/K/N` from
   `op.py:115-132` (PR #94 diff) as compile-time constants in Peano build.
10. **Replicate the 2x int32 RTP slot** (`design.py:339-350`) so runtime can
    pass `K_div_k` and `n_tiles_per_core` to cores.

## Source file map

| IRON path                                   | Reimplementation target            |
|---------------------------------------------|------------------------------------|
| `aie_kernels/aie2p/mm.cc:83-208`            | `kernels/aie2p/mm_core.cc`         |
| `aie_kernels/aie2p/mm.cc:396-410`           | `kernels/aie2p/mm_i8_i32.inc`      |
| `aie_kernels/aie2p/zero.cc:20-31`           | `kernels/aie2p/zero.inc`           |
| `iron/operators/gemm/design.py:358-437`     | `host/gemm_fifos.mlir`             |
| `iron/operators/gemm/design.py:517-761`     | `host/gemm_runtime.cc`             |
| `iron/operators/gemm/op.py:115-150` (PR#94) | `host/gemm_build.cmake`            |

Sources:
- IRON repo @ `devel`: https://github.com/amd/IRON
- Issue #55 (column count, TOPS): https://github.com/amd/IRON/issues/55
- Issue #93 (INT8 request): https://github.com/amd/IRON/issues/93
- PR #94 (INT8 wiring): https://github.com/amd/IRON/pull/94
