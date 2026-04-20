# Why no NPU yet?

**One-line answer**: evaluated four stacks (ONNX-RT + Vitis EP, FastFlowLM, IREE-AIE, direct xrt). **Deferred** — no path runs BitNet-b1.58 on Strix Halo's XDNA 2 in Linux today, and the realistic decode ceiling is below our current iGPU. Update posture: quarterly passive monitoring, not active work.

## What's on the box

- **AMD XDNA 2** — 50 TOPS int8 matrix accelerator in the Strix Halo APU package.
- **`amdxdna` kernel driver** — shipped with kernel 7.x on CachyOS (one of the two reasons the landing page insists on CachyOS).
- **HSA/XRT userspace** — present via ROCm-7, ready for a dispatch layer to sit on top.

## Why we haven't used it yet

1. **No Rust-native BitNet NPU runtime exists publicly.** The mature options are Python (AMD Ryzen AI SDK), and those target generic ONNX models — **BitNet's ternary matmul is a custom op** that standard ONNX runtimes don't know. Someone has to write it; so far nobody has, on this hardware, in public.

2. **iGPU is already bandwidth-bound, not compute-bound.** Our ternary GEMV hits 92% of LPDDR5 peak on the weight-read path. Adding NPU compute doesn't create new bandwidth. The NPU would share the same LPDDR5; it wouldn't make decode faster, only change which silicon the math lands on.

3. **NPU's real win is prefill + speculative**, not decode. At prefill (M > 1 matmul instead of M = 1 GEMV) the NPU's int8 matrix units would be fed more efficiently. Same for Medusa-style speculative decoder heads — multiple candidate tokens in parallel is exactly what the NPU is designed for.

## The two stacks we're evaluating

### Stack A — ONNX Runtime + Vitis/AMD NPU execution provider

**Shape**: Export BitNet to ONNX with a custom ternary-matmul op. Use ONNX Runtime's AMD EP to route that op to XDNA 2.

**Pros**:
- ONNX is a known format; most of the ecosystem speaks it.
- `ort` Rust crate gives us a Rule-A-clean bindings surface.
- If someone else writes the custom ternary op we can consume their work.

**Cons**:
- Custom ops in ONNX Runtime are a pain. You write them in C++ and rebuild the runtime.
- AMD's NPU EP shipped with Ryzen AI SDK 1.x — Linux support is incomplete; Windows-first.
- Calibration step (model → NPU-ready blob) requires AMD's Python toolchain at build-time.

### Stack B — FastFlowFM / Ryzen AI Flow

**Shape**: AMD's newer model-format + compiler, supposedly targets GPU+NPU dispatch with a unified IR.

**Pros**:
- Designed for the Ryzen AI hardware family — no EP plug-in step.
- Claims GPU+NPU co-dispatch, which matches our bandwidth story (iGPU pulls weights, NPU does matmul).

**Cons**:
- Unclear maturity. Documentation is sparse. Might be SDK-only, pre-release, or Windows-locked.
- Unclear if it handles ternary weights at all.

### Stack C — IREE AMD AIE

**Shape**: IREE compiler pipeline with the `amd-aie` backend.

**Pros**: open source end-to-end, MLIR-based, Rust bindings via `iree-compiler` crate.

**Cons**: bleeding edge; BitNet model support unknown; perf not characterized on Strix Halo.

## Final verdict (2026-04-20)

Research concluded — see `project_npu_path_analysis.md` memory. **Defer until one of:**

1. Ryzen AI SDK ≥ 1.8 adds **STX-H (Strix Halo)** to its Linux-supported SKU list. Today's 1.7.1 (April 2026) lists only STX + KRK.
2. `microsoft/BitNet` ships an XDNA backend. [Issue #408](https://github.com/microsoft/BitNet/issues/408) — "Intel & AMD NPU support?" — has been open since Feb 2026 with zero Microsoft replies.
3. **FastFlowLM** open-sources its NPU kernels (currently closed-source, non-redistributable under their EULA) or adds a 1.58-bit model family.
4. Our iGPU path saturates the 212 GB/s LPDDR5 ceiling. Today we run at ~15% utilization — plenty of Sherry / activation-sparsity / KV-compression runway.
5. A third party publishes a working BitNet → AIE kernel in public.

## The four stacks, scored

### Stack A — ONNX Runtime + Vitis AI EP

- **Blocker**: Vitis EP has no ternary op; would need a custom AIE kernel.
- **Harder blocker**: Ryzen AI 1.7.1 Linux doesn't list Strix Halo at all. STX + KRK only.
- **Status**: multi-quarter dependency chain before it could even start.

### Stack B — FastFlowLM (note: "FastFlowFM" was a transcription artefact; real name is FastFlowLM)

- **Only Linux path** that actually touches XDNA 2 on Strix Halo today.
- **Format**: `Q4NX` (GGUF Q4_0 / Q4_1 derivatives). **Not ternary.** BitNet not in their 22-model catalog.
- **Kernels**: proprietary, non-redistributable under their EULA ($10M ARR commercial cap).
- **Rule A**: acceptable (Lemonade is the runtime, our systemd unit calls it).
- **Status**: wrong model format, closed kernels. No path for halo-ai today.

### Stack C — IREE AMD-AIE (`nod-ai/iree-amd-aie`)

- **Cleanest technical path** for a custom ternary MLIR kernel — open source, MLIR-based.
- **Blocker**: still "early-phase". Requires Peano/LLVM-AIE and a Chess/Vitis license for full perf.
- **Effort**: multi-engineer-quarter build from scratch. We're one operator.
- **Status**: valid long-term bet, not actionable this quarter.

### Stack D — `xrt` direct

- Driver + `libxrt` are fine; no public BitNet kernel exists for it.
- Same MLIR-kernel-author problem as Stack C without the compiler help.

## The decode-tok/s math

**XDNA 2 bandwidth**: ~120 GB/s (dual-channel LPDDR5).
**iGPU bandwidth**: ~212 GB/s of the 256 GB/s pool (our measured ceiling).

BitNet-2B decode is memory-bandwidth-bound on the weight-read path. Linear scaling on the T-MAN paper's 50 tok/s result on Qualcomm Hexagon (~77 GB/s) gives XDNA 2 a **~77 tok/s ceiling** — below our measured **83 tok/s** on the iGPU today.

Decode-on-NPU is **negative ROI**. The compute is there (50 TOPS int8), the bandwidth isn't. This is the same memory-bound story the ternary choice already exploits — see [`Why-Ternary.md`](./Why-Ternary.md).

## What could still pay

**Prefill-on-NPU (tier a) only.** At prefill (many tokens at once), the matmul is dense enough that compute matters more than bandwidth. NPU's 50 TOPS int8 could roughly double prefill throughput on 128-token prompts.

That's a nice-to-have, not a cutover gate. And it still requires the AMD Linux SKU gap to close first.

## Status

- **Today**: iGPU-only, 83 tok/s decode, production-green.
- **NPU research**: concluded. Deferred.
- **Monitoring**: quarterly — check Ryzen AI SDK release notes + microsoft/BitNet#408 + FastFlowLM model catalog.

Memory pointer: `project_npu_path_analysis.md` has the full comparison table, citations, and defer-until conditions.
