# De-CK Plan

Goal: `librocm_cpp.so` ships without any `ck/` include surface or `libutility.a`
link. Consumers link one C header and one .so; no TheRock build required at
consumer build-time.

## Phase 0 — libutility.a drop [DONE, commit 6b694cb]

- Rewrote `rcpp_ternary_pack_pk_i4` in pure C++ using a single `std::vector<uint8_t>`.
- Removed `find_package(composable_kernel ... COMPONENTS utility)` from the link.
- Size: 425 KiB → 390 KiB.
- `nm librocm_cpp.so | grep -c 'ck5utils4conv'` = 0 (was several).
- End-to-end test still passes (max abs 0.008 vs CPU reference).

## Phase 1 — standalone kernel, correctness [DONE, commit 599b78e]

- `src/prefill_standalone.hip` — naive one-thread-per-output kernel with
  explicit WMMA-permute reversal and pk_i4 → FP16 decode. **Zero `ck/` includes.**
- `tests/test_standalone.cpp` diffs CK output vs standalone on the same packed weights.
- Verified on BitNet shapes 256×256×512 through 2560×6912×2560 — all PASS at FP16 tolerance.
- Perf: 6–30× slower than CK. Expected for naive.

## Phase 2 — LDS tiling [DONE, commit 621b489]

- Added `standalone_ternary_gemm_lds` with 16×16 output tile, K_TILE=32,
  A/B staged in LDS each K step. Within-8 nibble permute unwound on load.
- Correctness: max abs 0.0005 vs CK (better than Phase 1 because FP32
  accumulator stays hot in LDS rather than rematerializing).
- Perf: ~1.2–1.4× over naive. Tile is still launch-overhead-bound at
  16×16 output — real lift needs larger tiles + WMMA.

## Phase 3 — WMMA [NEXT]

**Scope:** replace the scalar FMA inner loop with
`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`. Use one wave per 16×16 output
tile initially, then expand.

**Hazards documented in `~/therock/build/dist/rocm/include/ck/utility/amd_wmma.hpp`:**

- RDNA 3 Wave32 WMMA fragment registers are `half16_t` per lane but only hold
  8 unique values — the upper 8 slots are duplicates. Direct load as if
  they were 16 distinct values produces silent wrong results.
- C accumulator fragment (`float8_t` per lane for f32 accum, `half16_t` for
  f16 accum) uses `OpSel` to pick which half of the register holds the
  result. For the f16-accum variant, `Opsel=false` writes to D0.[0:15],
  `Opsel=true` to D0.[16:31]. Wrong Opsel = output ends up in the "wrong"
  lanes and the store is garbage.
- Lane-to-fragment mapping: lanes 0..15 and lanes 16..31 hold **duplicate**
  A/B data but produce different halves of C. Read this out of CK's
  `BlockwiseGemmWMMA` before coding — don't guess.

**Plan:**

1. Write `standalone_ternary_gemm_wmma_v1`: one wave = one 16×16 output tile.
   K tile = 16 per WMMA step. Use FP32 accumulator (simpler to reason about)
   then cast to FP16 for output.
2. Load A fragment (half16_t per lane) respecting the duplication: only the
   first 8 slots need unique data per lane.
3. Load B fragment similarly (decoded from pk_i4 on load).
4. Iterate K in WMMA steps, accumulating into `float8_t`.
5. Write C: each lane owns 8 scattered output positions; compute their
   coordinates from lane id and store.
6. Diff against CK. If diff clean, move to Phase 4.

**Expected perf:** ~1.5–2× slower than CK. 16×16 per wave is still too
small; Phase 4 expands to 64×64 or 128×128 with multi-wave output tiles.

## Phase 4 — multi-wave tile + pipeline [LATER]

- Expand output tile to 128×128, 8 waves per block, 4×2 WMMA mapping
  (matches CK's winning instance).
- Interwave v1 pipeline: ping-pong LDS A/B buffers, double-buffered global
  loads.
- Match `KPack=16`, vectorized 128-bit LDS reads.
- Expected perf: ±10% of CK.

## Phase 5 — BitNet compute simplification [POST-PARITY]

- Once Phase 4 matches CK, replace the FP16 WMMA math path with the
  arxiv-documented ternary simplification: weights are {-1, 0, +1}, so
  the compute is select-and-add, not multiply-and-accumulate.
- No WMMA mul instructions on the hot path — just sign-aware
  conditional adds from A values into the accumulator.
- Target: beat CK specifically on BitNet shapes (the reference `-8` bias
  decode + FP16 multiply in CK does unnecessary arithmetic).
- This is the "we don't need CK, template is just a reference" win
  condition.

## Checkpoint / rollback

After each phase, `librocm_cpp.so` must pass `test_ck_gemm` at the
0.5 max-abs threshold. `test_standalone` must pass at 0.25 threshold
against the current CK path. If not, roll back before starting the
next phase.

## CMake / distribution story

Phases 0–2 done: CK is **header-only** at compile time. No ck/ symbols
ship in librocm_cpp.so. Phase 3+ removes the CK headers entirely, then
`BUILD_WITH_CK=OFF` becomes the default and TheRock is no longer a
hard requirement for consumers — only an alternate backend for perf
A/B during development.
