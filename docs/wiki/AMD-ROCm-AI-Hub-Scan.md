# AMD ROCm AI Dev Hub — Scan for 1-bit BitNet on Strix Halo

**Date:** 2026-04-20
**Source root:** <https://www.amd.com/en/developer/resources/rocm-hub/dev-ai.html>
**Fetch method:** WebSearch (WebFetch permission denied). Link anchors verified via search snippets, not DOM walk.
**Scope:** ROCm AI landing + top 10 sub-resources relevant to low-bit inference.

## Root page inventory

The dev-ai hub surfaces five buckets: **Tutorials** (Jupyter), **Performance Results**, **ROCm Docs**, **Infinity Hub containers**, and **AMD Developer Cloud / AI Dev Program**. No direct BitNet / ternary references appear on the landing page; everything is Instinct-first marketing with Ryzen AI as a secondary tier.

## Resources crawled

| Name | URL | Purpose | License | Python-only? | gfx1151? |
|---|---|---|---|---|---|
| AI Developer Hub root | [amd.com/.../dev-ai.html](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai.html) | Landing page, links to all below | marketing | n/a | no |
| Performance Results | [.../dev-ai/performance-results.html](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html) | MI300X/MI325X/MI355 benchmarks | marketing | n/a | no (Instinct only) |
| Tutorials for AI developers | [rocm.docs.amd.com/.../ai-developer-hub/latest](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/) | Jupyter notebooks hub | MIT (repo) | **yes** (ipynb) | no |
| ROCm/gpuaidev (notebook source) | [github.com/ROCm/gpuaidev](https://github.com/ROCm/gpuaidev) | Git source for above | MIT | **yes** | no |
| AMD Quark quantizer | [quark.docs.amd.com/latest](https://quark.docs.amd.com/latest/intro.html) | Quantization toolkit (FP8/MXFP4/INT4/INT3) | proprietary redistributable | **yes** (PyTorch/ONNX) | no |
| Quark MXFP4 for vLLM tutorial | [.../mxfp4_quantization_quark_vllm.html](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm.html) | Llama3.3-70B MXFP4 recipe | MIT | yes | no |
| ROCm-LLMExt | [github.com/ROCm/ROCm-LLMExt](https://github.com/ROCm/ROCm-LLMExt) + [docs](https://rocm.docs.amd.com/projects/rocm-llmext/en/latest/index.html) | LLM reference stack (train→infer→orch) | MIT | yes | not called out |
| AITER (AI Tensor Engine) | [github.com/ROCm/aiter](https://github.com/ROCm/aiter) | Centralized high-perf op library; MLA/all-gather/etc. | MIT | Python bindings, kernels C++/Triton | no (MI300/MI350 focus) |
| Composable Kernel (CK) | [github.com/ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) (now in rocm-libraries) | Templated C++ GEMM/reduction device library; pk_int4_t | MIT | **C++** | gfx1153 listed, gfx1151 not |
| Strix Halo system optimization | [rocm.docs.amd.com/.../system-optimization/strixhalo.html](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html) | TTM/GTT tuning, amd-ttm, kernel requirements | docs | n/a | **yes** |
| llama.cpp on ROCm (official) | [rocm.docs.amd.com/projects/llama-cpp/en/docs-26.02](https://rocm.docs.amd.com/projects/llama-cpp/en/docs-26.02/index.html) | AMD-hosted llama.cpp branch + prebuilt binaries for gfx1151 | MIT | **C++** | **yes** |
| AI Developer Program / Cloud | [amd.com/.../ai-dev-program.html](https://www.amd.com/en/developer/ai-dev-program.html) | $100 free cloud credits (MI-class) | TOS | n/a | no (cloud = Instinct) |

## Rule-A (no-Python-runtime) verdict

Service-side-safe: **CK**, **llama.cpp on ROCm**, **Strix Halo optimization doc**, **AITER core kernels** (bindings are Python but the kernels themselves compile down; vLLM/SGLang integration is still Python-glue).
Service-side-blocked: Quark, gpuaidev notebooks, ROCm-LLMExt, all Jupyter tutorials. Useful for caller-side reference/design only.

## Ternary / sub-byte tooling findings

- **Quark floor is INT3** (plus INT4/UINT4, MXFP4, FP6, FP8). No INT2, no ternary, no 1-bit. Confirms our earlier reject.
- **CK Tile GEMM** added `pk_int4_t` packed-int4 preshuffle. Block/row/col/tensor scaling present. Still no sub-4-bit.
- **hipBLASLt** has MXFP8/MXFP4 GEMM tuning. Useless for ternary.
- **MIOpen** moved to the monorepo `ROCm/rocm-libraries`. No ternary primitives advertised; still tensor/conv focused.
- **ZenDNN** not surfaced from dev-ai hub (CPU-path, BLIS-style). No entry point from the AI hub — irrelevant here.

## gfx1151-specific mentions

Two real ones:

1. **Strix Halo system optimization page** — TTM/GTT knobs, `amd-ttm` tool, VRAM-vs-shared guidance (keep BIOS VRAM at 0.5 GB, push TTM limit to ~100 GB). We should sanity-check our current TTM against this.
2. **AMD-hosted llama.cpp prebuilt binaries** for Ubuntu 24.04 on gfx1150/gfx1151. Build flags documented: `-DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DAMDGPU_TARGETS=gfx1151`. We already have our own HIP kernels but their Flash-Attn rocWMMA flag is worth diffing against our split-KV FD kernel.

Everything else in the dev-ai hub is MI300/MI325/MI355 or generic RDNA3.5 (gfx1150/51/52 lumped).

## New-to-us list

Things not previously documented in memory:

1. **ROCm-LLMExt** (`ROCm/ROCm-LLMExt`) — AMD's own reference LLM stack. Python/vLLM-heavy → caller-side reference only, but worth reading for recipe parity.
2. **AITER** (`ROCm/aiter`) — centralized AMD op library; kernels land in vLLM/SGLang upstream. No gfx1151 target today but the MLA decode pattern is relevant once we scale context.
3. **`amd-ttm` tool + Strix Halo optimization doc** — official knob for TTM/GTT page limit. Action item: verify our `/etc/modprobe.d/ttm.conf` matches AMD's guidance.
4. **AMD-hosted llama.cpp branch** at `rocm.docs.amd.com/projects/llama-cpp` with gfx1151 prebuilt binaries — diff their HIP build flags vs ours.
5. **AMD Developer Cloud $100 free credits** via AI Dev Program — MI-class only, useful for distillation runs if Battlemage slips.
6. **CK `pk_int4_t` preshuffle GEMM** — confirms 4-bit is the floor AMD upstream cares about; our ternary kernel has no competition in the official tree.

## Honesty notes

- WebFetch was blocked by sandbox; all link verification is from WebSearch result snippets, not direct DOM. Every URL in the table is one that search returned as a first-class result, not synthesized.
- The dev-ai landing page is heavily Instinct-oriented. "AI Developer Hub" in AMD's marketing means MI300-class; Strix Halo content is reached via the ROCm docs tree, not the hub landing.
- No BitNet, ternary, 1-bit, or sub-INT3 artifact was found anywhere in the AMD-owned tree. We remain the only gfx1151 ternary-kernel public player.
