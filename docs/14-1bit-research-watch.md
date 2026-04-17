# 1-bit LLM Research Watch

Running log of papers that matter for our kernels. Append new entries on top.
Each entry: paper, date, one-line takeaway, concrete relevance to rocm-cpp.

## Live entries (2026)

### Sparse-BitNet (arxiv:2603.05168, March 2026)
**Takeaway:** 1.58-bit BitNet weights naturally are ~42% zeros — semi-structured
sparsity comes for free.
**Relevance:** Our Phase 5 kernel already branchlessly skips zeros on compute
(the `(bits==1) - (bits==2)` expression yields 0). But we still issue the A
LDS read for zero-weight positions. A sparse-aware B encoding that packs only
non-zero positions could cut LDS reads further. gfx11 lacks 2:4 sparse WMMA,
so the win has to come from CSR-style iteration, not hardware sparsity.
**Status:** NOT INTEGRATED. Future Phase 5.2 candidate.

### BitNet b1.58 2B4T Technical Report (arxiv:2504.12285, March 2026)
**Takeaway:** The exact model we benchmark against. Confirms 2B params, 4T
tokens, ternary weights, 8-bit activations.
**Relevance:** Our Phase 5 matches their activation precision (INT8). Sim
says we beat their reference decode by 2.20x on gfx1151.
**Status:** Confirmed arch match. No change needed to our kernels.

### BitNet a4.8 — 4-bit Activations for 1-bit LLMs (Microsoft Research)
**Takeaway:** Move from INT8 to INT4 activations with per-group scales.
Claims further compute savings.
**Relevance:** Tested naïve INT4-A (Phase 5.1, kernels/ternary_gemv_phase5_i4a.hip)
with per-tensor scale. Result: 0.96-1.15x (wash) on our shapes because we're
compute-bound, not bandwidth-bound. Correctness collapses with per-tensor
scale — a4.8's actual value is the per-group scale scheme, NOT raw INT4.
**Status:** NEGATIVE-RESULT variant shipped in-tree for reference. Proper
per-group INT4 is scope for Phase 5.2.

### Tequila — Deadzone-free Ternary Quant (ICLR 2026 OpenReview)
**Takeaway:** 3.0x inference speedup on their stack via avoiding the "dead
zone" around zero.
**Relevance:** Our Phase 5 dot4 hits 3.4-5.5x on gfx1151 via INT8 A + dot4
builtin, so we already meet/exceed this perf target. Their training-time
contribution (dead-zone avoidance in quant) is orthogonal to kernel work.
**Status:** No kernel action. Flag for our QAT training side if that becomes
relevant.

### Bitnet.cpp — Efficient Edge Inference for Ternary LLMs (arxiv:2502.11880, Feb 2025)
**Takeaway:** Microsoft's CPU inference paper. TL1/TL2 lookup-table schemes,
6.25x over FP baseline, 2.32x over low-bit baseline on CPU.
**Relevance:** CPU-only. We ship GPU kernels. Our Phase 5 on gfx1151 does
10.8 us on 2560x2560; their AMD EPYC 7543 result on 2048x2048 is 93.276 ms
(from Microsoft's BitNet src/README). Two to three orders of magnitude gap
favoring GPU.
**Status:** Reference for CPU comparison narrative in our docs.

### OneBit (NeurIPS 2024, arxiv:2402.11295)
**Takeaway:** Quant-training framework with matrix-decomp initialization.
Achieves 81% of non-quantized LLaMA performance with pure 1-bit weights.
**Relevance:** Training-time method, not kernel work. Would feed weights into
our Phase 5 inference path.
**Status:** No kernel action.

## How to add an entry

1. Find paper via arxiv / OpenReview / NeurIPS / ICLR search.
2. Append a new `###` section on TOP of Live entries. Date in title.
3. Fill out: Takeaway (1 line), Relevance (how it applies to our kernels),
   Status (INTEGRATED / NEGATIVE / FLAGGED / NO ACTION).
4. Commit as `docs: 1-bit research watch — <paper title>`.

Never delete entries. Even stale papers are useful history.
