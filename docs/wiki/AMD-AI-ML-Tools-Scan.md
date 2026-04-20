# AMD AI/ML SDK Tools Scan

Scope: catalog AMD-published AI/ML/inference/quantization/model-serving SDKs from
`amd.com/en/developer/browse-by-resource-type/software-tools.html` and adjacent AMD
pages. FPGA-only (Vitis/Versal) excluded — another agent owns that lane. Filter
applied: Rule A (no Python at runtime), Rule B (C++ for kernels, Rust above).

AMD's top-level "Software Tools" page itself is mostly a tile-grid marketing hub
(`WebFetch` was denied; content reconstructed from cited AMD sub-pages and GitHub).
The real signal lives on the product sub-pages. Catalogue below is deduplicated to
**distinct tools** (libraries that are just sub-libraries of ROCm rolled up).

## Catalogue

| name | purpose | license | Rule A safe? | Strix Halo today? | worth evaluating? |
|---|---|---|---|---|---|
| ROCm 7 (umbrella) | GPU compute stack (HIP, LLVM, clang, runtime) | MIT-ish / open | Yes | Yes (gfx1151) | Already in use |
| AITER | ROCm "AI Tensor Engine" — fused FA/MLA/MoE kernels | MIT (ROCm/aiter) | C++/HIP core, Py dispatch | Instinct-tuned; gfx1151 untested | **Steal (kernels only)** |
| Composable Kernel (CK) | Header-only C++ GEMM/conv kernel templates | MIT | Yes (pure C++) | gfx1151 builds (~1h) | **Steal** — fused ternary GEMV candidate |
| MIOpen | cuDNN-analog (conv/rnn/batchnorm) | MIT | Yes | Yes | Skip — we bypass it |
| MIGraphX | Graph-mode inference runtime (ONNX/TF in) | MIT | Yes (C++ core) | Yes | Skip — no ternary op |
| hipBLAS / hipBLASLt | BLAS façade, MXFP8/MXFP4 path | MIT | Yes | Yes | **Banned** (CLAUDE.md Rule C) |
| rocBLAS / Tensile | GEMM kernels under hipBLAS | MIT | Yes | Yes | Skip (dragged in by ban) |
| rocFFT / rocSPARSE / rocRAND / rocSOLVER / rocPRIM / rocThrust / rocWMMA / hipCUB / RCCL | Math/parallel primitives | MIT | Yes | Yes | Available if needed; not core |
| AMD Quark | Model quantization toolkit (PTQ/QAT, ONNX/PyTorch) | MIT | **No — Python** | N/A (dev-box only) | **Watch** — offline compile OK |
| ZenDNN / ZenDNNL | Zen CPU DNN kernels (BF16, INT8, exp INT4) | Apache-2.0 (fork of oneDNN) | Yes (C++ lib) | CPU lane on Strix Halo Zen5 | **Steal** — CPU fallback + llama.cpp backend |
| AOCL (BLIS, libFLAME, FFTW, Sparse, libm) | CPU math | Mixed: BLIS/libFLAME Apache-ish, some closed EULA | Mostly yes | Zen5 optimized | Evaluate for non-AI host math only |
| AOCL-DLP | CPU deep-learning primitives (low-prec GEMM) | AMD EULA (not open) | C lib, Rule-A safe at runtime | Zen5 | **Watch** — closed licence is a negative |
| ROCm Triton | Python DSL → GPU kernels | MIT | **No — Python compile** | gfx1151 (patchy) | Skip (Rule A + GEAK is Py too) |
| GEAK-OptimAgent / GEAK-OpenEvolve | LLM-agent Triton kernel auto-tuner | Research | Python | MI-class only | Skip |
| ROCprofiler-SDK + rocpd | SQLite-backed profiler replacing rocprof | MIT | Yes (C API) | gfx1151 fork by woct0rdho | **Steal** — upgrade path from rocprof |
| AMD SMI | GPU telemetry CLI + C lib | MIT | Yes | Yes | Available |
| AMD uProf | CPU/GPU sampling profiler (IBS, PMC) | Freeware (proprietary) | Yes (binary) | Zen5 + Instinct only | Evaluate — no gfx1151 GPU path |
| Ryzen AI Software 1.7.1 | NPU SDK (ONNX + Vitis AI EP) | Proprietary | No — Python wrappers | **STX-H unsupported** | Already rejected |
| MIVisionX / RPP | CV graph + preprocessing | MIT | Yes | Yes | Not AI/LLM — skip |
| Infinity Hub | Catalog of Docker containers | N/A | Containers violate bare-metal rule | Instinct-oriented | Skip |
| AMD AI Developer Portal | Website / learning hub | N/A | N/A | N/A | Reference only |
| Adrenalin AI Bundle | Desktop installer (Ollama + PyTorch + ComfyUI) | Proprietary shell | No | Radeon-dGPU | Skip |

Count: **22 distinct tools / SDK families** catalogued after dedup.

## Verdicts

### Steal (adopt now, fills a real gap)

1. **Composable Kernel (CK)** — `ROCm/composable_kernel`. Header-only C++ template
   kernels; gfx1151 target already compiles. Candidate backbone for a
   **fused ternary-GEMV + Sherry 3:4 unpack** kernel, staying inside Rule A/B.
2. **AITER** — `ROCm/aiter`. Reference implementations of Flash-Decoding, MLA,
   MoE dispatch. Even though the Python dispatcher violates Rule A, the **HIP
   kernel sources are MIT** and directly portable into our Rust-driven loader.
3. **ZenDNN 5.2** — ships a **llama.cpp backend** and INT4 experimental path. Drop-in
   CPU lane for Strix Halo Zen5 cores when iGPU is saturated or during fallback.
   Apache-2.0 fork of oneDNN; pure C++; zero Python at runtime.

### Watch (not ready but roadmap-relevant)

1. **AMD Quark** — only AMD quantizer with real **LLM INT4 / UINT4 + AutoSearch**.
   Python-only, so we'd treat it as an offline compile tool, never runtime. Watch
   for a ternary / 1.58-bit codepath (currently none). MIT licensed.
2. **ROCprofiler-SDK + rocpd** — successor to the rocprof we already use. SQLite
   output is a big win for kernel-level regression tracking on bitnet_decode.
   Community fork has the gfx1151 PC-sampling patch (woct0rdho/rocm-systems).

### Skip (closed, Python-only runtime, or already owned)

- Ryzen AI SDK 1.7.1 (no STX-H Linux, already rejected).
- hipBLAS/hipBLASLt (Rule C ban).
- MIGraphX / MIOpen (no ternary op, we bypass with native kernels).
- Triton + GEAK (Python compile-time; GEAK targets MI only).
- Infinity Hub (containers violate bare-metal lock-in).
- Adrenalin AI Bundle (consumer-desktop, Windows-first).
- AOCL-DLP (EULA is not redistributable — BLIS is fine, DLP is not).

## Notes on source quality

AMD's `software-tools.html` returned as a permission-blocked WebFetch and the AI
Developer Portal (March 2026 launch) is mostly a tile-grid. The **real**
catalogue lives at `rocm.docs.amd.com/.../api-libraries.html` and in the
individual GitHub orgs (`ROCm/*`, `amd/Quark`, `amd/ZenDNN`). Nothing surprising
surfaced that isn't already in ROCm or a named Zen/Ryzen SDK — AMD is not hiding
a secret low-precision library.
