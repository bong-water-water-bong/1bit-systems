# 1-bit LLM Research Watch

Running log of papers that matter for our kernels. Append new entries on top.
Each entry: paper, date, one-line takeaway, concrete relevance to rocm-cpp.

## Live entries (2026)

### BitDecoding GitHub — open-source implementation (github.com/OpenBitSys/BitDecoding, github.com/DD-DuDa/BitDecoding)
**Takeaway:** The BitDecoding paper above has working CUDA code open-sourced.
Tensor-Cores-Centric BitFusion, warp-efficient parallel decoder, async pipeline.
**Relevance:** **Direct port reference for our KV cache work.** Their CUDA
kernels use Tensor Core primitives; gfx11 has WMMA which is the analog.
Porting their 4-bit K/V decode kernel to gfx11 WMMA is Phase 6 concrete scope.
Perf targets: 7.5x RTX 4090, 4.8x A100, 8.9x H100 — our gfx1151 target is
the gap-to-FA3-like FP16 decode on long context.
**Status:** FLAGGED — Phase 6 primary reference.

### carlosfundora/llama.cpp-1-bit-turbo (github.com/carlosfundora/llama.cpp-1-bit-turbo)
**Takeaway:** HIP/ROCm llama.cpp fork optimized for RDNA2 (gfx1030) with:
  - PrismML Q1_0_G128 support (their 1-bit format)
  - **RotorQuant** (ICLR 2026) — geometric-rotation KV cache quant
  - **TurboQuant** — extreme KV cache quant
  - EAGLE3 + P-EAGLE speculative decoding
  - Wave32 kernel optimizations
**Relevance:** **Direct-adjacent stack.** They target RDNA2/gfx1030; we
target RDNA3.5/gfx1151. Same architectural family, same 1-bit weight format
interest. Their RotorQuant integration in GGML is proven to work on HIP.
Study their PR history and Wave32 kernel patterns as prior art for our
RDNA3.5 work. Not a fork we adopt — reference-only.
**Status:** FLAGGED — prior-art reference for KV cache + Wave32.

### RotorQuant — Geometric rotation KV cache quant (ICLR 2026)
**Takeaway:** Pre-rotates K/V via geometric transforms before quantization,
making them quantization-friendly. Already integrated into llama.cpp GGML
with CPU + CUDA/HIP + Flash Attention support.
**Relevance:** The quantization-side complement to BitDecoding's kernel-side.
If we add a KV cache kernel, adopting RotorQuant pre-rotations at cache-write
time (trivial HIP kernel) preserves accuracy under aggressive K/V quant.
**Status:** FLAGGED — pair with BitDecoding port.

### TurboQuant — Extreme KV cache quant (llama.cpp discussion #20969)
**Takeaway:** Extreme low-bit KV cache quant in active llama.cpp development.
**Relevance:** Downstream of our kernel work. Their llama.cpp integration
could consume our KV cache kernels once we ship them.
**Status:** FLAGGED — integration target.

### Qwen3 Quantization Empirical Study (arxiv:2505.02214)
**Takeaway:** Systematic PTQ sweep on Qwen3 across 1-8 bit. Qwen3 holds up
at moderate bits, degrades under ultra-low precision on linguistic tasks.
**Relevance:** Not our weights today (we track Bonsai / BitNet), but if the
community quants Qwen3 to 1-bit, our Phase 5 GEMV would be the natural
inference kernel.
**Status:** NO ACTION. Market watch.

### PT-BitNet (ScienceDirect S089360802500735X)
**Takeaway:** Post-training BitNet quantization — scales existing FP LLMs
to 1-bit without retraining.
**Relevance:** Weights-side. Would feed into our Phase 5 kernel. If it
lands with Llama-3-70B or similar, our stack has a much bigger model catalog.
**Status:** WATCH.

### BitDecoding — Low-bit KV cache with Tensor Cores (arxiv:2503.18773)
**Takeaway:** CUDA+Tensor Core pipeline for low-bit KV cache GEMV in long-context
decode. Dequant kernel + GEMM overlap + extra metadata movement. Claims up to
**8.6× over FP16 FlashDecoding-v2** on long-context workloads.
**Relevance:** **WE DO NOT HAVE A KV CACHE KERNEL.** The library currently
ships a prefill GEMM + decode weight-GEMV path but doesn't touch KV cache
attention compute. Adding a low-bit KV cache variant is the next real
feature-level gap for long-context usage. Porting their software-pipeline
approach to gfx11 WMMA would be 1–2 sessions of kernel work.
**Status:** FLAGGED — next feature. Phase 6 candidate.

### Arbitrary Precision Acceleration for LLMs on GPU Tensor Cores (arxiv:2409.17870)
**Takeaway:** Bit-level matrix decomposition recovers arbitrary-precision
matmul on tensor cores. Includes W1A2 (1-bit weights, 2-bit activations)
schemes. Flexible precision on fixed-width hardware.
**Relevance:** Our Phase 5 is W1.58A8. Their W1A2 scheme would cut both
weight and activation bandwidth further on gfx11 WMMA. Requires custom
bit-serial accumulation logic — non-trivial port. Read for technique,
not direct adoption.
**Status:** FLAGGED — future research lever.

### ButterflyQuant — Ultra-low-bit via butterfly transforms (arxiv:2509.09679, Feb 2026)
**Takeaway:** Replaces Hadamard rotation in quant pipelines with learnable
butterfly transforms. O(n log n), smooth optimization, orthogonality preserved.
Targets 2-bit regime where activation outliers kill naive quant.
**Relevance:** Training-side. Could help if we ever do our own QAT training
of a BitNet-variant. Not a kernel thing.
**Status:** NO ACTION. Flag for training side.

### LittleBit — 0.1 BPW via latent factorization (arxiv:2506.13771, Feb 2026)
**Takeaway:** Sub-1-bit quantization down to 0.1 BPW (10× below BitNet 1.58).
31× memory vs FP16. Llama2-13B fits in 0.9 GB. Factorization approach, not
direct ternary.
**Relevance:** Weights format is completely different from BitNet ternary —
factored low-rank approximation, not ternary codes. Our Phase 5 GEMV kernel
wouldn't apply. Watch if it matures; would need an entirely new kernel class.
**Status:** NO ACTION. Watch maturity.

### BTC-LLM — Sub-1-bit via adaptive transform + binary codebook (arxiv:2506.12040, Nov 2025)
**Takeaway:** Sub-1-bit LLM quant. Claims **surpasses FP16 baseline on
LLaMA-1-30B** under aggressive compression. Binary codebook with
learnable transforms.
**Relevance:** Different encoding (binary codebook, not ternary). Doesn't
feed into our Phase 5 path but competitive positioning matters — their
story is "sub-1-bit and still beats FP16." If adoption picks up, the
community's reference 1-bit stack shifts.
**Status:** WATCH. Not kernel-actionable today.

### ParetoQ — Unified training+quant for sub-4-bit (arxiv:2502.02631, Oct 2025)
**Takeaway:** First unified approach for sub-4-bit training and quant.
Pareto-optimal scaling laws.
**Relevance:** Training-side. Defines "what tradeoff to train for" in
quant regime we care about.
**Status:** NO ACTION. Training reference.

### BitNet Distillation (arxiv:2510.13998)
**Takeaway:** Distill full-precision teacher LLMs down to BitNet b1.58 students.
**Relevance:** Training pipeline for our QAT work. Not kernel.
**Status:** NO ACTION. Training reference.

### Continual QAT Pre-Training for BitNet (arxiv:2502.11895)
**Takeaway:** Train FP16 first, then transition to 1.58-bit QAT. Better than
1.58-bit-from-scratch. Open question: when to transition.
**Relevance:** Training-side. The "transition moment" question is their
open problem. Not kernel.
**Status:** NO ACTION. Training reference.

### MAGNET — Autonomous expert generation + BitNet training (arxiv:2603.25813)
**Takeaway:** 618M BitNet b1.58 model + decentralized autoresearch framework
for expert generation.
**Relevance:** Shows BitNet adoption beyond Microsoft's own models. Good
signal for ecosystem momentum.
**Status:** NO ACTION. Ecosystem watch.

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
