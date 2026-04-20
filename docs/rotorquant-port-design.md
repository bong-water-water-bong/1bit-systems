# RotorQuant / PlanarQuant / IsoQuant port feasibility vs rocm-cpp FD attention

Status: design-only, 2026-04-19. No HIP code written this sprint.
Reviewer: halo-ai core (gfx1151).
Upstream artifacts consulted:
- `scrya-com/rotorquant` README, CLAUDE.md, paper (`paper/rotorquant.pdf`), and `turboquant/csrc/*.cu` (8 files, 2050 LOC).
- Our own `rocm-cpp/src/kv_cache_attn_fd.hip` (293 LOC), `kv_cache_attn_i8.hip` (352 LOC).

## tl;dr

**Tier (a): one-week drop-in for PlanarQuant-3 K+V; the full RotorQuant Clifford variant is not worth porting.** The real production algorithm in the repo is not the Clifford rotor from the paper title — it is PlanarQuant (2D Givens) or IsoQuant (4D quaternion) in a separate llama.cpp fork. Both are block-diagonal, parameter-light (128 params for d=128 head dim), and their forward/inverse rotations are 4 FMAs/pair and 16 FMAs/quad respectively — trivial to implement in HIP and trivial to fuse inside our Flash-Decoding inner loop. Our existing `kv_cache_attn_i8.hip` already shows the dequant-in-inner-loop pattern. The unknown is the PPL delta on BitNet-2B, which their numbers do not bound. Gate the port on a small Python-free PPL A/B using the existing `bitnet_decode --ppl` harness after the kernel lands.

## Algorithm summary (what actually ships, not what the paper markets)

The paper describes Clifford-rotor quantization (`Cl(3,0)`, ~56 FMAs/group-of-3, 372 params for d=128). The **production** algorithm acknowledged in both README and CLAUDE.md is simpler and was in fact contributed by an external collaborator (`@ParaMind2025`):

1. **Split each head-dim-128 vector into blocks** of 2 (PlanarQuant) or 4 (IsoQuant). For d=128 that is 64 pairs or 32 quads per vector.
2. **Per-block fixed orthogonal rotation.** PlanarQuant uses a 2D Givens rotation `R(θ)` = `[[c,-s],[s,c]]` with one angle per pair (stored as `(cos, sin)` pre-baked at model-load, 128 f32 for d=128 — shared across all layers/heads). IsoQuant uses a 4D quaternion sandwich `q_L v q_L^*`, 128 f32 (4×32 quats).
3. **Lloyd-Max scalar quantization** of each rotated coordinate to 2^bits centroids (`n_levels=8` for 3-bit, `n_levels=16` for 4-bit). Centroids precomputed for the Gaussian-approximated `N(0, 1/d)` distribution induced by the rotation; shared across all layers.
4. **On read (attention-inner-loop):** index → centroid f32, then apply the **inverse** rotation `R(-θ)` / `q_L^* v q_L`. The inverse step is what pushed PPL from 15,369 → 7.05 on Llama-3.1-8B (CLAUDE.md, commit `6e5a4aa`) — any port must get this right.
5. **Deferred K path.** During prefill, K stays FP16; only the post-prefill decode tokens are inserted quantized. The paper/README claim this removes ~3× of the roundtrip-quantization PPL penalty; our FD kernel already has a separate prefill path and this fits.

Rotations are **fixed (pre-baked random orthogonal), not learned**, and **not Hadamard** — that is the entire point of the paper vs TurboQuant. They are loaded once as `__constant__` memory and reused for every layer/head. This is why the param count is 128, not 128 × layers × heads.

## Kernel-level cost

Integration point is `kv_cache_attn_fd.hip`. The current Pass-1 inner loop (lines 100–134) loads FP16 K/V rows directly:

```
const __half* K_row = K + ((size_t)t * num_kv_heads + kv_head) * head_dim;
partial += (float)Q_shared[d] * (float)K_row[d];
...
o_local[ei] += beta * (float)V_row[d];
```

The port turns that into: load 3-bit index byte → look up centroid in `__shared__` (n_levels=8, trivially fits in LDS) → apply inverse 2D Givens on every even/odd pair → use in Q·K dot and β·V accumulator.

**Concrete patch shape:**
- New file: `src/kv_cache_attn_rq.hip` (~450 LOC) — clone of `kv_cache_attn_fd.hip` with two inner-loop dequant helpers and a `__constant__` rotation/centroid table. Keep FP16 FD intact as the reference path.
- New file: `src/rotor_requantize.hip` (~150 LOC) — one-shot FP16→PlanarQ3 requantizer used at prefill end to convert the K cache in place. Mirrors the "double-buffer deferred" pattern described in CLAUDE.md's `src/llama-kv-cache.cpp` notes.
- Model-load path in `h1b_loader.*` grows ~40 LOC to mmap the centroids + `cos/sin` blob. Prefer shipping these as an extra `.h1b` section rather than hard-coding (lets us regenerate codebooks per-calibration without a binary rebuild).
- CLI/flag: `--kv-rotorquant` alongside the existing `--kv-int8`. Mutually exclusive.

**Total LOC delta: ~700–800 new, 0 modified** in the FD/FP16 hot path. FP16 default stays bit-exact.

Register/LDS budget on gfx1151, head_dim=128, BLOCK=128:
- Centroid LUT: 8 × f32 = 32 B (4-bit: 16 × f32 = 64 B). Broadcast-friendly.
- Rotation table: 128 × (f32 c, f32 s) = 1 KB. Put in `__constant__` (8 KB available), not LDS.
- Per-thread extra: 2 floats for the pair's rotated partner. No spill risk at current 4 reg/thread headroom.

## Memory win

Our dimensions: 30 layers × 20 q-heads × GQA 4:1 → 5 kv-heads × head_dim 128 = 3840 KV elements / token / layer × 30 layers = **115,200 KV elements / token total**.

| format | bytes/token | bytes/token @ 4k ctx × 30 layers |
|---|---:|---:|
| FP16 K+V (current default) | 230,400 | **460.8 MB** @ N=2048 → **921.6 MB** @ N=4096 |
| INT8 K+V (`--kv-int8`) | 115,200 | 230.4 MB @ N=2048 → 460.8 MB @ N=4096 |
| PlanarQ3 K+V (proposed) | ~46,200 (see below) | ~92 MB @ N=2048 → ~185 MB @ N=4096 |

PlanarQ3 storage per element: 3 bits indices + amortized scale/norm. Paper Table 7 for Qwen2.5-3B: 289 MB FP16 → 57.6 MB at 3-bit = **5.02× compression** (not 10.3× — the README marketing number assumes bare indices and ignores norms and codebook/calibration). We scale the paper's 5.02× to our KV: 460 MB → ~92 MB at N=2048, a net **~368 MB saved per live context** on top of FP16, or **~138 MB** on top of INT8. At N=4096 this grows to ~736 MB / ~276 MB respectively.

This is real: Strix Halo shares 128 GB LPDDR5 with GPU. Every MB off the KV is a MB for weights, compile cache, or Chromium.

## Accuracy risk

Paper numbers, **all from their benchmarks, not ours**:
- Llama-3.1-8B-Instruct Q4_K_M, WikiText-2, ctx=2048: FP16 KV PPL **6.63**, iso3 **6.91** (+4.2%), planar3 **7.05** (+6.3%), turbo3 **7.07** (+6.6%) — README line 14–18.
- Qwen2.5-3B + Triton backend, WikiText-2, roundtrip: FP16 **7.59**, PlanarQuant-4 **9.56** (+26%), IsoQuant-4 **9.03** (+19%) — README line 173–175.

Our baseline: **PPL 9.1607 on wikitext-1024** (`CUTOVER.md:13`). The uncertainty bounds:
- Rotorquant was evaluated on Llama-3.1-8B (dense FP16 weights) and Qwen2.5-3B (FP16 weights). BitNet-2B has ternary weights and substantially higher baseline PPL — its activations are already coarser.
- Lower-bound expectation: if KV error composes **linearly** with weight error, BitNet is more tolerant of noisy KV than fp16-weight models (the weights are the loss-dominant term). PPL delta could be smaller than Llama's +4–6%.
- Upper-bound expectation: if KV decorrelation assumes near-Gaussian post-RMSNorm activations and BitNet-2B's activations deviate (they have RMSNorm but post-ternary-matmul stats are different), Lloyd-Max centroids tuned on `N(0, 1/d)` are mis-matched and PPL could hit +10–15%.
- Realistic guess: planar3 K+V on BitNet-2B lands PPL **9.5–10.2** (= +4–11% over 9.16). That is acceptable for K-only (**planar3/f16**, ~0% PPL loss on Llama-3.1) but needs measurement for symmetric. **Do not ship symmetric without running `bitnet_decode --ppl`.**

If calibration is off, the fix is running our own Lloyd-Max solve against BitNet-2B KV captures. The `lloyd_max.py` solver is Python-only (SciPy `integrate.quad`), but the output is a 2^bits-element f32 table; we generate offline, write into the `.h1b` section, load at startup. No runtime Python.

## Integration with Flash-Decoding (the make-or-break question)

**It fits cleanly inside the FD inner loop.** Specifically:

- Pass-1 (`kv_cache_attn_decode_split_kernel`) inner `for (int t = t_begin; t < t_end; ++t)` reads K_row[d] and V_row[d] for every t in the tile. Swap the FP16 load for a 3-bit-index load + LUT + inverse-rotation. The quantized K_row occupies `head_dim × 3/8 = 48` bytes instead of 256 — we **more than halve DRAM bytes per FD step**. Our main long-context bottleneck (tok/s 83→68 from N=64→1024) is KV read bandwidth, so this is the right axis.
- The Givens inverse rotation is **pair-local** (2 coords at a time). Each thread in our BLOCK=128 processes 1 coord of head_dim=128; we'd need to either (a) exchange via `__shfl_xor` within a warp to grab the partner coord, or (b) change the load pattern so thread `tid` handles the pair `(2*tid, 2*tid+1)`. (b) is simpler: halve active threads to 64, each handles a full 2D block including its inverse rotation, LUT, and contribution to Q·K. Wave32 stays natural because the pair-local math is fully intra-lane.
- **No materialize-full-KV intermediate is forced.** This is the critical observation: since the rotation is block-diagonal (2D or 4D), dequant-then-use-then-forget works register-local, exactly as INT8 dequant already does. Contrast this with TurboQuant WHT where the d=128 butterfly network **cannot** be applied per-coord — that would force a full-head materialization and kill FD's streaming benefit. That is also why TurboQuant's prefill is 5.3× slower than PlanarQuant in their table.
- V dequant in the β accumulator: identical pattern, inverse Givens per pair, then `o_local[ei] += beta * dequant_V[d]`. Commit `6e5a4aa` in their tree is the reference for the inverse-rotation step that matters for V; miss it and PPL explodes to 15k.

Pass-2 reduce kernel is untouched — still operates on f32 partials.

## Composition with ternary weights (BitNet-specific)

RotorQuant is KV-only; it does not touch weights. The cross-interaction concerns are:

1. **Calibration distribution.** Their Lloyd-Max codebook assumes post-rotation coordinates are `~N(0, 1/d)`. That assumption holds when K/V come from a roughly iid softmax-normalized attention. BitNet applies subln (custom RMSNorm variant) before QKV projection, and its ternary KV-projection weights produce activation statistics that differ from FP16 Llama. Verify empirically: capture 10k BitNet-2B KV vectors at ctx=2048, check per-coord variance after rotation, re-solve Lloyd-Max if `σ²` is far from `1/d = 1/128 ≈ 0.0078`.
2. **Prefill compute.** Our prefill already uses ternary-GEMV for QKV projection. Rotation + quant happens after QKV projection, so the ternary path is untouched. Deferred K (FP16 during prefill) means no rotation overhead in the bandwidth-critical prefill stage.
3. **No double-quantization.** Weights are 2-bit-per-element (packed 4/byte), KV would be 3-bit; these are independent. No shared codebook, no shared scale.
4. **RoPE.** RoPE is applied to Q and K **before** they go to the cache. Rotation for PlanarQuant is applied on top — it's a second, different 2D rotation per pair. Compose as `K_cached = Quant(PlanarRot(RoPE(K)))`, inverse on read. Worth a sanity check that the two rotations do not accidentally undo each other (they operate on different pair-groupings and different angles; mathematically independent, but runtime test with a fixed seed before declaring victory).

## Recommendation

**Tier (a): one-week drop-in for PlanarQuant-3 K+V, scoped as follows.**

Do: PlanarQuant (2D Givens), 3-bit K + 3-bit V, deferred K prefill. Skip Clifford rotors entirely (paper's own Table 5 shows PPL RMSE 0.048 vs TQ's 0.037 at 3-bit — worse, and 28 FMAs/group vs 4). Skip IsoQuant first pass (quaternion sandwich is 4× the FMA count of Givens for +0.14 PPL gain on Llama; revisit if PlanarQuant is insufficient on BitNet). The port is bounded: ~800 LOC of HIP across two new files, zero changes to FP16 FD path, one `.h1b` section for the centroid+rotation blob, one Lloyd-Max offline solver (offline Python is fine; runtime stays C++). The memory win is ~4-5× over FP16 KV and roughly 2.5× over INT8 KV, and the paper's PlanarQ3 matches or beats TurboQuant on every axis they benchmarked. The single gate is a PPL A/B on our existing `bitnet_decode --ppl` harness; if planar3/planar3 clears +10% PPL on wikitext-1024 we ship it behind `--kv-rotorquant`. If it misses, fall back to **planar3 K-only** (their tables show ~0% PPL regression at 5.1× K compression) which is still a real win.

### Proposed extern "C" signature (tier-a sketch)

```c
// Drop-in replacement for rcpp_kv_cache_attn_decode_fd when KV is PlanarQ3.
// K is packed 3-bit indices: (seq_len, num_kv_heads, head_dim*3/8) bytes.
// V same layout. Rotation + centroid tables loaded at startup into __constant__.
extern "C" rcpp_status_t
rcpp_kv_cache_attn_decode_fd_pq3(
    const void* Q_dev,       // __half, (num_q_heads, head_dim) — unchanged
    const void* K_idx_dev,   // uint8_t, packed 3-bit indices
    const void* V_idx_dev,   // uint8_t, packed 3-bit indices
    void*       out_dev,     // __half, (num_q_heads, head_dim)
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float scale, void* stream);

// One-shot FP16 → PlanarQ3 requantize, called at end of prefill.
extern "C" rcpp_status_t
rcpp_kv_requantize_pq3(
    const void* K_fp16_dev,  // __half, (seq_len, num_kv_heads, head_dim)
    void*       K_idx_dev,   // uint8_t output
    int seq_len, int num_kv_heads, int head_dim, void* stream);
```

## Constraints recheck

- **hipBLAS banned.** Not triggered. Every kernel in `turboquant/csrc/*.cu` is a hand-written CUDA kernel; no cuBLAS calls. The only "BLAS matmul" reference in the paper is a description of TurboQuant (the thing they replaced). Rotation is per-pair FMAs, LUT is `__shared__`, no GEMM primitive is touched. Clean.
- **wave32 / gfx1151.** Their `#define WARP_SIZE 32` in `planar2_fused_kernel.cu:18` is already wave32-shaped. No warp-shuffle reductions on assumed lane64 — the rotation is pair-local and does not cross lanes. One-to-one transliteration to HIP.
- **No WMMA.** Intentionally: the rotations are 2×2 / 4×4, far below WMMA's 16×16×16 tile. We don't want WMMA here — dequant is strictly a load+scalar-op fusion to the existing FD kernel, and WMMA would force a materialize intermediate which defeats the point.
- **BitNet ternary weights.** Unaffected by KV quant; see "composition" section.

## Numbers cited (with source)

1. **PPL 6.91 iso3 vs 7.07 turbo3 @ 10.3× compression** — `README.md` line 15, 17 (same benchmark, "Llama 3.1 8B Instruct Q4_K_M — Symmetric 3-bit K+V Compression (RTX 5090)" table).
2. **44× fewer parameters (128 vs 16,384 at d=128)** — `paper/rotorquant.pdf` §3.3 Table 1, corroborated by README line 25.
3. **289 MB FP16 → 57.6 MB at 3-bit = 5.02× compression** — `paper/rotorquant.pdf` §5.7 Table 7 (Qwen2.5-3B-Instruct 8K ctx, 36 layers). Note this conflicts with the README's 10.3× marketing figure; the 5× number is what actually lands with norms + codebooks included.
4. **Inverse-rotation fix `6e5a4aa`: PPL 15,369 → 7.05** — `CLAUDE.md` lines 25–27. This is a hard warning: if the V dequant forgets the inverse Givens / inverse quaternion step, PlanarQuant/IsoQuant produces unusable output.
5. **4 FMAs/pair for PlanarQuant inverse rotation** — `turboquant/csrc/planar2_fused_kernel.cu` lines 37–45 (`rot2_apply` + `rot2_inverse` are 4 multiplies + 2 adds each; trivial on gfx1151 V_ALU).

## Verify (how we'd prove the port works)

- Unit: bit-exact `rcpp_kv_requantize_pq3` → dequant roundtrip on a random FP16 KV buffer has MSE ≤ 1.1 × paper's Table 4 distortion for d=128, 3 bits (~0.081).
- Integration: `bitnet_decode --ppl --kv-rotorquant` on wikitext-1024 produces PPL ≤ 10.2 (+11% over 9.1607 baseline).
- Perf: `rocprofv3 --stats bitnet_decode --ppl` shows `MemUnitBusy` drops below FP16 FD's current ~92% during attention phase (because KV bytes per step drop 5×), and `tok/s` at N=1024 rises from 68 → ≥ 95 (5× KV bandwidth for K and V combined means ~3× attention-phase wall time reduction; assume 60% of decode wall time is attention at N=1024, so end-to-end ≥ 1.3×, i.e. 68 → ~88 bounded below, realistic ~95).
- Memory: `halo bench --n 4096 --kv-rotorquant` RSS reports ≥ 250 MB lower than `--kv-int8` at same seq_len.
