# Why no NPU yet?

**One-line answer**: the XDNA 2 NPU (50 TOPS int8) is on the box and visible to the kernel, but no mature Rust/C++ path to dispatch BitNet-b1.58 to it exists yet. We're evaluating ONNX Runtime vs AMD's FastFlowFM vs IREE-AIE. The iGPU path already delivers 80+ tok/s and is production-green, so NPU is an optimization, not a blocker.

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

## Current posture

We have a deep-research agent scanning the three stacks for (a) BitNet readiness today, (b) Rust binding quality, (c) realistic decode-tok/s expectation on XDNA 2, (d) AMD's published Linux status, and (e) prior art from anyone else who has run BitNet on XDNA 2 in public. Result will land in `project_npu_path_analysis.md` and update this page with a concrete recommendation + "defer until" condition.

## "What would the payoff look like?"

If we route prefill + speculative through XDNA 2:
- Prefill: today ~50 tok/s on 128-token prompts (iGPU). NPU could double to ~100 tok/s if the int8 pipe can be fed.
- Speculative (MedusaBitNet heads): 4 candidate tokens per backbone step. 2–3× decode throughput is the expected win.
- Decode on NPU alone: unlikely to beat iGPU — same memory, same bandwidth ceiling.

So the NPU story is **faster prefill + speculative decoding**, not "make decode faster". That framing matters for choosing which stack to port.

## Status

- **Today**: iGPU-only, 80+ tok/s decode, shipping.
- **Research agent**: in flight, comparing ONNX / FastFlowFM / IREE.
- **Decision window**: after the agent lands + after the user's mics arrive (the recording-ready cycle takes priority).

Memory pointer: `project_npu_path_analysis.md` will have the comparison table + recommendation when the agent finishes.
