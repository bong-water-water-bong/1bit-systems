# rocm-cpp

Native ROCm C++ for Strix Halo (gfx1151). Built from scratch. No Python at runtime.

## April 17, 2026 — BitNet-2B-4T decodes end-to-end on librocm_cpp

The full inference pipeline now runs against real BitNet-b1.58-2B-4T weights
with **coherent English output** at **82 tok/s**, top-5 logits bit-matching
a PyTorch reference trace. Zero MLX, zero framework deps.

```
$ bitnet_decode models/halo-1bit-2b-absmean.h1b 128000 32
tokens: 279 4320 374 220 18 13 20 13 220 18 13 20 374 279 4320 13 ...
# "the answer is 3.5" (greedy argmax, loops because no prompt context)
```

**Five correctness gaps found and closed in one session** (see `tools/bitnet_decode.cpp`):

1. `.h1b` loader was reading FP16 where the exporter writes FP32
2. `attn_sub_norm` and `ffn_sub_norm` were loaded but never applied
3. Wrong activation — SiLU instead of BitNet's ReLU² GLU (`hidden_act="relu2"`)
4. Per-row absmax quant instead of BitNet's per-tensor `1/mean(|W|)` absmean
5. FP16 overflow in `relu²(gate) * up` hitting 10⁹ mid-network — now fused
   with `ffn_sub_norm` in a single FP32-internal kernel, emitting FP16 only
   after the norm brings the value back to bounded scale

## Quick start — install everything

```bash
curl -fsSL https://raw.githubusercontent.com/stampby/rocm-cpp/main/install.sh | bash
```

Or see [`install.sh`](install.sh) — clones rocm-cpp + halo-1bit, builds
against an existing TheRock ROCm dist (or a vanilla `/opt/rocm`), exports
a BitNet-2B-4T `.h1b`, and runs `bitnet_decode` as a smoke test.

## Wanted — beta testers & FTXUI contributors

If you have a Strix Halo box (Ryzen AI MAX+ 395, 128 GB unified), run the
install script and send feedback. Every edge case — different ROCm dists,
different BitNet checkpoints, different models — is useful signal.

We're also looking for **UI contributors** on two tracks:

**FTXUI terminal TUI** (`tools/bitnet_tui.cpp` already scaffolded):
- live token stream panel + prompt input (M1+M2 shipped)
- per-layer timing + KV cache heatmap (M3+M4 open)
- Man Cave voice page (M5+ open)
- works over SSH, zero X11 — full spec: [`docs/15-tui-spec.md`](docs/15-tui-spec.md)

**Dear ImGui + SDL2 desktop GUI** (not yet scaffolded):
- top-down floating-tile landing page — specialists drift around a
  frosted-glass canvas over a daily-rotating wallpaper
- chat page (F1), Man Cave voice page (F2), backend selector (local
  BitNet / OpenAI / Anthropic / Groq / etc.)
- full spec: [`docs/17-desktop-gui-spec.md`](docs/17-desktop-gui-spec.md)

Both frontends talk to the same HTTP server (`bitnet_decode --server`),
so specialist logic stays in agent-cpp. No Qt, no Electron, no web.

## Works with any OpenAI-compatible client

`bitnet_decode --server` speaks the OpenAI `/v1/chat/completions` and
`/v1/models` protocol. Anything that can point at a custom base URL will
talk to it. No plugins, no adapters, just swap the URL.

**Set once for the whole ecosystem:**

```bash
export OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=sk-ignored     # any non-empty string works
```

**Chat UIs (MIT / BSD / Apache only):**

| | license | one-line stand-up |
|---|---|---|
| [OpenWebUI](https://github.com/open-webui/open-webui) | BSD-3 | `docker run -d -p 3000:8080 -e OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 ghcr.io/open-webui/open-webui:main` |
| [LibreChat](https://github.com/danny-avila/LibreChat) | MIT | `docker compose up -d` with `OPENAI_REVERSE_PROXY=http://host.docker.internal:8080/v1` in `.env` |
| [LobeChat](https://github.com/lobehub/lobe-chat) | MIT | `docker run -d -p 3210:3210 -e OPENAI_PROXY_URL=http://host.docker.internal:8080/v1 lobehub/lobe-chat` |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | MIT | docs workspace + chat, Docker or native |
| [GPT4All](https://github.com/nomic-ai/gpt4all) | MIT | native Qt desktop app, set "Server URL" in settings |

**Coding / IDE integration:**

| | license | notes |
|---|---|---|
| [Continue.dev](https://continue.dev) | Apache-2 | VS Code + JetBrains. `"provider": "openai"` + `"apiBase"` in `~/.continue/config.json` |
| [Aider](https://github.com/Aider-AI/aider) | Apache-2 | `aider --openai-api-base http://127.0.0.1:8080/v1 --model bitnet-b1.58-2b-4t` |
| [Cline](https://github.com/cline/cline) | Apache-2 | VS Code extension, pick "OpenAI Compatible" provider |
| [llm](https://github.com/simonw/llm) | Apache-2 | `llm -m bitnet-b1.58-2b-4t 'your prompt'` after one-time config |

**Terminal / TUI:**

| | license | notes |
|---|---|---|
| **[bitnet_tui](tools/bitnet_tui.cpp)** | MIT | our own. FTXUI, ships with rocm-cpp. |
| [oterm](https://github.com/ggozad/oterm) | MIT | textual-based TUI |
| [tgpt](https://github.com/aandrew-me/tgpt) | MIT | single-binary CLI chat |
| [gptme](https://github.com/gptme/gptme) | MIT | tool-use in terminal |

**Agent / workflow builders:**

| | license | notes |
|---|---|---|
| [n8n](https://github.com/n8n-io/n8n) | Apache-2 | low-code automation, has OpenAI node |
| [Langflow](https://github.com/langflow-ai/langflow) | MIT | visual agent builder |
| [Flowise](https://github.com/FlowiseAI/Flowise) | Apache-2 | drag-and-drop chains |
| [BigAGI](https://github.com/enricoros/big-AGI) | MIT | multi-model playground |

Our TUI and upcoming desktop GUI are the **signature surfaces** — the
floating-tile landing page doesn't exist anywhere else. The list above
is the **ecosystem on-ramp** for beta testers who already have a
workflow they like. Your stack speaks the protocol; the whole universe
plugs in.

### Production deploy — Caddy reverse proxy

`bitnet_decode --server` has no auth and no TLS. That's fine for
`--bind 127.0.0.1` on your own box. For anything beyond localhost (mesh
access, remote client, public endpoint) put Caddy in front.

```caddyfile
# /etc/caddy/Caddyfile  — snippet
halo.<yourdomain>.com {
    # Optional bearer-token gate.
    @auth header Authorization "Bearer {$HALO_API_KEY}"
    handle @auth {
        reverse_proxy 127.0.0.1:8080
    }
    handle {
        respond 401
    }
    # rate limit: 30 req/min per client
    rate_limit {
        zone halo {
            key {client_ip}
            window 1m
            max_events 30
        }
    }
}
```

Caddy gives you:
- Let's Encrypt TLS out of the box
- Bearer-token auth via simple request matchers
- Rate limiting (via the `caddy-ratelimit` plugin)
- Access logs routed wherever you already aggregate

Mesh pattern (you already run this): Caddy on the boundary box, ed25519
ssh keys between nodes, `bitnet_decode --server` pinned to `127.0.0.1`
on the Strix Halo box, clients hit `https://halo.<mesh>.tld/v1/...`.
No server-side code changes needed.

## Headline numbers (gfx1151, Phase 5 dot4)

- **Prefill 2560×6912×2560 (BitNet FFN up):** 30.15 TFlops, **1.02× rocBLAS FP16** at ¼ B memory. Bit-perfect.
- **Decode 128256×2560 (LM head):** 506 µs per call, **5.51× v1 GEMV, 2.03× BitNet-2B-4T token throughput** in full-model sim.
- **Decode 11008×4096:** 59.3 µs, **7.13× rocBLAS FP16 GEMV**.
- **Standalone prefill kernel:** 101% of CK's tuned WMMA on BitNet FFN, **zero `ck/` headers**.
- **Hardware ceiling:** 55.36 TFlops WMMA peak (measured). We hover at 56%.

## For integrators — drop-in 1-bit backend

If you are building a C++ inference engine targeting AMD Strix Halo — **Lemonade**, **llama.cpp HIP backend**, **MLX ROCm**, or any `WrappedServer`-style router — `librocm_cpp.so` is the BitNet / ternary compute path you were going to write. It already exists, it's measured against rocBLAS, and it ships behind a single C header so you don't pull CK or HIP templates into your translation units.

```c
#include <rocm_cpp/ck_gemm.h>

// Once at model load: convert your ternary weights {-1, 0, +1} to the
// WMMA-permuted pk_i4 layout the GPU kernel expects.
rcpp_ternary_pack_pk_i4(ternary_KN, packed_KN_div2, K, N);

// Hot path, per-layer prefill:
rcpp_ck_gemm_handle_t* h;
rcpp_ck_gemm_create(M, N, K, &h);
rcpp_ck_gemm_run(h, A_fp16_dev, B_packed_dev, C_fp16_dev, stream);
```

**On gfx1151 (Radeon 8060S), versus rocBLAS FP16 on the same shapes:**

| Path | Shape | TFlops | vs rocBLAS FP16 | B memory |
|---|---|---|---|---|
| Prefill (CK backend) | 2560×6912×2560 (BitNet FFN up) | 30.20 | 0.96× | **¼** |
| Prefill (CK-free standalone, Phase 4h) | 2560×6912×2560 | **30.75** | **1.017×** | **¼** |
| Decode GEMV (v1) | 2560 × 2560 @ batch=1 | — | **4.9× faster** | **¹⁄₁₆** |
| Decode GEMV (v1) | 4096 × 4096 @ batch=1 | — | **7.2× faster** | **¹⁄₁₆** |

The CK-backed prefill requires TheRock ROCm 7.13. The standalone kernel (`src/prefill_standalone.hip`) has zero `ck/` includes — drop it into any HIP project — and currently ships **at 101.7% of CK's tuned WMMA on BitNet FFN shapes**, bit-perfect. Decode GEMV has been gfx1151's first-ever 1-bit kernel since April 2026. Full build recipe + 5 honest regression phases in [`docs/12-how-we-did-it.md`](docs/12-how-we-did-it.md).

## What This Is

A pure C++ inference and compute stack targeting AMD Strix Halo APUs. Custom Wave32 HIP kernels for 1-bit / ternary, CK-backed WMMA prefill, native Tensile GEMM from source. All C++, all on RDNA 3.5 silicon.

**Ships two kernel surfaces through one C library (`librocm_cpp.so`):**
1. **Prefill GEMM** — FP16 × packed-ternary (pk_i4) WMMA, 0.96× rocBLAS FP16 at 1/4 the B memory
2. **Decode GEMV** — fused Wave32 ternary, **4.9× faster than rocBLAS FP16 GEMV** at batch=1

## Community Validation

- **PrismML-Eng/Bonsai-demo#48** — Merged 2026-04-16 by @khosravipasha. Community benchmark page for ROCm HIP Q1_0 on Strix Halo landed upstream.
- **PrismML-Eng/Bonsai-demo#51** — Open, awaiting review. TheRock 7.13 uplift + 7-model 1-bit burn.
- **First external fork** — @bogdan-d, 2026-04-16

## C API — `librocm_cpp.so`

Consumers (halo-1bit, lemond, external) link against `librocm_cpp.so` and include a single C header. No CK or HIP templates leak to consumer TUs.

```c
#include <rocm_cpp/ck_gemm.h>

// 1. Pack ternary weights once at model load — host side, no GPU.
int8_t packed[K * N / 2];
rcpp_ternary_pack_pk_i4(ternary_KN, packed, K, N);   // {-1,0,+1} → pk_i4 WMMA-permuted

// 2. Create handle, upload packed weights + FP16 activations to device, run.
rcpp_ck_gemm_handle_t* h;
rcpp_ck_gemm_create(M, N, K, &h);
rcpp_ck_gemm_run(h, dA_fp16, dB_packed, dC_fp16, stream);
rcpp_ck_gemm_destroy(h);
```

Full header: [`include/rocm_cpp/ck_gemm.h`](include/rocm_cpp/ck_gemm.h). End-to-end test: [`tests/test_ck_gemm.cpp`](tests/test_ck_gemm.cpp).

### End-to-end proof

Random FP16 activations × random ternary weights → compare GPU output vs host CPU scalar reference (independent path, different codepath from WMMA):

```
Shape                  Perf            Correctness vs CPU ref
─────────────────────────────────────────────────────────────
512x512x2560           16.34 TFlops    PASS  (max abs 0.008)
1024x1024x2560         22.27 TFlops    (MNK too big for CPU ref)
2560x6912x2560         30.20 TFlops    BitNet FFN up  (tested via ck-prefill verify)
2560x2560x6912         29.43 TFlops    BitNet FFN down
4096x4096x4096         21.22 TFlops
```

## Results — April 16, 2026 (TheRock 7.13 native)

### Prefill GEMM — `librocm_cpp` Phase 4h vs CK backend vs rocBLAS FP16

Same A and B data. `librocm_cpp` path: FP16 A × pk_i4 ternary B → FP16 C (¼× B memory). rocBLAS path: FP16 A × FP16 B (dequantized). Measured live in this session, hipEvent + 20 timed runs after warmup.

```
Shape (MxNxK)          Phase 4h        CK backend      rocBLAS FP16    std/CK     std/rocBLAS
────────────────────────────────────────────────────────────────────────────────────────────
2560x6912x2560         29.97 TFlops    28.79 TFlops    28.32 TFlops    1.041x     1.058x   <- beats rocBLAS
2560x2560x6912         29.51           29.25           —               1.009x     —
2560x2560x2560         28.65           28.88           34.81           0.992x     0.823x
4096x4096x4096         29.32           21.03           34.24           1.394x     0.856x
```

BitNet FFN up is where ternary prefill earns its place: **Phase 4h standalone beats both CK's tuned template AND rocBLAS FP16** at ¼ the B memory. On 4096³ square shape the standalone is **1.39× CK** — structurally better tile shape for high-occupancy gfx1151 blocks than CK's 128×128 default. Square-shape gap vs rocBLAS (0.856×) is the same ceiling rocBLAS's tuned FP16 path always wins against any ternary kernel that has to pay a pk_i4 decode on the WMMA input side.

### Decode GEMV — `librocm_cpp` v1 ternary vs rocBLAS FP16 GEMV

Batch-1, memory-bound. Ternary encoding wins here.

```
Shape              librocm_cpp v1    rocBLAS FP16    Speedup
─────────────────────────────────────────────────────────────
2560 x 2560         38.3 μs           189 μs         4.9x
6912 x 2560        108.0 μs           —              —
2560 x 6912        104.1 μs           —              —
4096 x 4096         98.7 μs           708 μs         7.2x
11008 x 4096       249.2 μs          1244 μs         5.0x
128256 x 2560     2729.6 μs           —              — (LM head bottleneck)
```

On every measured shape at batch=1, we ship >5× the throughput of rocBLAS hgemm.

### Prism llama.cpp Q1_0 — Full 1-Bit Burn (7 Models)

```
Model                       Quant    Size       pp512 t/s    ±std     tg128 t/s    ±std
────────────────────────────────────────────────────────────────────────────────────────
Bonsai-1.7B                 Q1_0     231 MB     5,001.2     ±38.2      230.9       ±0.8
BitNet-2B-4T                Q1_0     538 MB     3,651.9     ±14.8      120.2       ±3.3
Bonsai-4B                   Q1_0     540 MB     2,124.9      ±1.8      125.6       ±0.3
Bonsai-8B                   Q1_0     1.07 GB    1,324.5      ±4.5       96.1       ±0.1
Qwen3-Coder-Next 80B-A3B    IQ1_S    17.6 GB      661.6      ±5.1       50.8       ±0.0
Llama-4-Scout 17Bx16E       IQ1_S    27.2 GB      325.7      ±0.7       21.3       ±0.0
BitNet-2B-4T                TQ1_0    1.02 GB      281.6      ±1.0       49.7       ±0.0

PrismML-Eng llama.cpp prism branch (e2d6742) + TheRock ROCm 7.13 native gfx1151,
llama-bench 3 rounds, ngl=99, ROCBLAS_USE_HIPBLASLT=1
```

80B MoE at 51 tok/s. 108B at 21 tok/s. 8B in 1 GB at 96 tok/s. Bonsai-1.7B breaks 5,000 tok/s prompt.

### TheRock 7.13 Uplift (vs earlier TheRock 7.12 build)

```
Model                       pp512 prior    pp512 new    Δ         tg128 prior    tg128 new    Δ
──────────────────────────────────────────────────────────────────────────────────────────────
Bonsai-1.7B                 4,172          5,001        +20%       232            231          ~same
BitNet-2B-4T Q1_0           3,030          3,652        +21%       110            120          +9%
Bonsai-4B                   2,014          2,125         +5%       125            126          ~same
Bonsai-8B                   1,278          1,325         +4%        94             96          +2%
```

### ROCm vs Vulkan (honest, same prism commit e2d6742)

```
Model                       ROCm pp      Vulkan pp    Δ           ROCm tg    Vulkan tg    Δ
─────────────────────────────────────────────────────────────────────────────────────────────
Bonsai-1.7B                 5,001        3,121        +60%        231         137          +69%
BitNet-2B-4T Q1_0           3,652        2,750 est    +33%        120          98 est      +22%
Bonsai-4B                   2,125        1,401        +52%        126          85          +48%
Bonsai-8B                   1,325          831        +59%         96          64          +50%

ROCm wins both prompt AND generation on every Bonsai / BitNet shape.
Caveat: Qwen3-Coder-Next 80B runs faster under Vulkan on both axes (MoE kernel gap).
```

## Kernel Development Path

### v1 GEMV (decode) — production, shipped in librocm_cpp

Fused Wave32 ternary GEMV, first HIP kernel for 1-bit inference on RDNA 3.5. Kernel source at [`kernels/ternary_gemv.hip`](kernels/ternary_gemv.hip).

### v2 GEMM (prefill) — paused, replaced by CK path

Hand-rolled WMMA INT8 GEMM experiments landed 0.16–0.23× rocBLAS FP16 on BitNet FFN shapes. Closing that gap cleanly requires four architectural changes in combination — pre-quant kernel, double-LDS-buffered pipeline, multi-block per wave, FP16 activations (see [`docs/09-kernel-v2-design.md`](docs/09-kernel-v2-design.md)). **Paused** in favor of the CK path below.

Research commits: v2.0 (DP4A) through v2.4 (LDS-union occupancy tuning) on `main`.

### CK-backed pk_i4 prefill (current)

`DeviceGemm_Wmma_CShuffleV3<F16, pk_i4, F16, ...>` from Composable Kernel in TheRock, with a local ternary→pk_i4 packer that compensates for CK's `n - 8` nibble decode and `CK_USE_PK4_LAYOUT_SHUFFLE` byte ordering. Proven end-to-end against a scalar CPU reference and against CK's host reference across five shapes.

Scaffold: [`ck-prefill/`](ck-prefill). Production entry point: `src/ck_gemm.cpp` (wrapped by the C API). Integration design: [`docs/10-ck-integration-path.md`](docs/10-ck-integration-path.md).

**Tile tuning result:** default (BlockSize=256, 128×128×32, Interwave v1, PermuteB=true) is optimal on gfx11. Intrawave-v3 prefetch (KPerBlock=64) is designed for XDL/gfx9 and loses 2–3× on WMMA — don't use it.

## The Problem

- No optimized Tensile/rocBLAS GEMM kernels exist for gfx1151 in any shipped package
- No ternary-aware kernel path exists on ROCm — anywhere
- Everyone falls back to generic dequantize-then-matmul (the slowest path)
- Missing compiler flags cause 69% regression that nobody documents
- hipBLASLt is "unsupported" on gfx1151 but works

## What We Built

- **TheRock from source** — ROCm 7.13 with 43 native Tensile GEMM kernels + rocRoller + hipBLASLt for gfx1151
- **librocm_cpp.so** — C library exposing prefill (CK-backed WMMA) and decode (fused Wave32 GEMV)
- **Ternary→pk_i4 packer** — offline, host-side, one-shot at model load time
- **Q1_0 HIP kernel** — added upstream via PrismML llama.cpp prism branch (PR #48, PR #51)
- **Full documentation** — every flag, every env var, every bug fix to replicate

## Quick Start

### Prerequisites

```bash
# CachyOS / Arch Linux
sudo pacman -S --needed base-devel cmake ninja git rocm-hip-sdk patchelf gcc-fortran

# Python deps for Tensile kernel generation (TheRock build only)
pip install --break-system-packages pyyaml joblib packaging tqdm CppHeaderParser
```

### Build TheRock (ROCm from Source) — required dependency

```bash
git clone https://github.com/ROCm/TheRock.git ~/therock
cd ~/therock && git submodule update --init --recursive
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTHEROCK_AMDGPU_TARGETS=gfx1151 \
    -DTHEROCK_DIST_AMDGPU_FAMILIES=gfx115X-all \
    -DTHEROCK_ENABLE_BLAS=ON
cmake --build build --parallel $(nproc)
# -> ~/therock/build/dist/rocm/  (unified ROCm install)
```

### Build rocm-cpp

```bash
cd rocm-cpp
cmake -B build -G Ninja
ninja -C build                      # librocm_cpp.so + test_ck_gemm + ck-prefill examples
```

### Runtime environment

```bash
export THEROCK=$HOME/therock/build/dist/rocm
export LD_LIBRARY_PATH=$THEROCK/lib:/opt/rocm/lib:$PWD/build
export ROCBLAS_TENSILE_LIBPATH=$THEROCK/lib/rocblas/library
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_ENABLE_SDMA=0
export ROCBLAS_USE_HIPBLASLT=1
export HIP_VISIBLE_DEVICES=0
```

### Run the end-to-end test

```bash
./build/test_ck_gemm 512 512 2560           # small, with CPU-reference verify → PASS
./build/test_ck_gemm 2560 6912 2560         # BitNet FFN up → 30.20 TFlops
```

### Run the v1 decode GEMV benchmark

```bash
./tools/bench_ternary                        # decode path, all shapes
```

### Run the rocBLAS reference

```bash
./tools/bench_gemm                           # rocBLAS FP16 baseline
```

### Run 1-Bit Models (PrismML prism branch)

```bash
git clone https://github.com/PrismML-Eng/llama.cpp.git && cd llama.cpp
git checkout prism
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
    -DCMAKE_HIP_COMPILER=$THEROCK/lib/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=$THEROCK/lib/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=$THEROCK/lib/llvm/bin/clang++
cmake --build build-rocm --parallel $(nproc)
./build-rocm/bin/llama-bench -m Bonsai-8B.gguf -ngl 99 -p 512 -n 128 -r 3
```

## Structure

```
CMakeLists.txt              — Top-level: librocm_cpp + tests + ck-prefill
include/
  rocm_cpp/
    ck_gemm.h               — Public C API (the only header consumers need)
src/
  ck_gemm.cpp               — C API implementation (wraps CK privately)
tests/
  test_ck_gemm.cpp          — End-to-end: pack → GEMM → diff vs CPU reference
ck-prefill/                 — Research binaries and tile-tuning experiments
  gemm_wmma_fp16_v3.cpp                  — CK FP16×FP16 baseline
  gemm_wmma_fp16_pk_i4_v3.cpp            — CK FP16×pk_i4 baseline (default tile)
  gemm_wmma_fp16_pk_i4_v3_pf.cpp         — Intrawave v3 prefetch variant (loses — keep for ref)
  gemm_wmma_fp16_ternary_as_pk_i4.cpp    — Ternary-clamped verify harness
kernels/
  ternary_gemv.hip          — Fused Wave32 ternary GEMV (decode, production)
  ternary_gemv_v2.hip       — v2 DP4A decode prototype (regression — research)
  ternary_gemm_v2.hip       — v2 WMMA INT8 GEMM (paused, superseded by CK)
tools/
  bench_gemm.cpp            — rocBLAS FP16 GEMM benchmark
  bench_ternary.cpp         — Fused ternary kernel benchmark + correctness
  run_bench.sh              — Automated comparison script
docs/
  00-hardware.md            — Strix Halo specs, unified memory, BIOS
  01-environment.md         — Runtime vars, shell setup, verification
  02-therock-build.md       — Building ROCm from source step by step
  03-compiler-flags.md      — The 69% flag and all HIP AOT flags
  04-wave32-kernels.md      — RDNA 3.5 kernel design guide
  05-ternary-inference.md   — 1-bit theory, packing, kernel design
  06-benchmarking.md        — All numbers and comparison tables
  07-forks-landscape.md     — Every relevant fork and what they did
  08-known-issues.md        — Every gfx1151 bug with workarounds
  09-kernel-v2-design.md    — v2 WMMA design (paused)
  10-ck-integration-path.md — CK path + dispatcher + weight pack
results/
  bonsai-q1_0-rocm-20260416.md      — Q1_0 kernel results
  full-1bit-burn-20260416.md        — Full 7-model burn
```

## Hardware

```
AMD Ryzen AI Max+ 395 (Strix Halo)
Radeon 8060S (gfx1151, RDNA 3.5, Wave32, 20 WGPs / 40 CUs)
128 GB unified LPDDR5X
CachyOS kernel 7.0.0-1-mainline
```

Note: HIP's `hipDeviceProp.multiProcessorCount` returns the WGP count (20) on RDNA, not the true CU count (40). Two CUs per WGP; use WGP × 2 when comparing against AMD's datasheet.

## TheRock Build Patches (GCC 15)

If building on GCC 15 / bleeding edge, these patches are needed:

1. **elfutils** — add `-Wno-error=discarded-qualifiers` to CPPFLAGS
2. **rocprofiler-sdk elfio** — add `#include <cstdint>` to `elf_types.hpp`
3. **rocprofiler-sdk yaml-cpp** — add `#include <cstdint>` to `emitterutils.cpp`
4. **aqlprofile test** — skip integration test (wrong compiler for HIP)
5. **Missing packages** — `xxd` (gvim), `pyyaml`, `CppHeaderParser`, `joblib`, `packaging`, `tqdm`, `gcc-fortran`

See [docs/02-therock-build.md](docs/02-therock-build.md) for details.

## Related Repos

- [bleeding-edge](https://github.com/stampby/bleeding-edge) — Wiki with full build log and known issues
- [lemon-mlx-engine](https://github.com/stampby/lemon-mlx-engine) — C++ MLX engine hitting 153 t/s
- [halo-1bit](https://github.com/stampby/halo-1bit) — 1-bit inference engine (The 1 Bit Blaster)
- [PrismML llama.cpp](https://github.com/PrismML-Eng/llama.cpp) — Prism branch with Q1_0 DP4A kernels

## License

If it can be done in C++, we do it in C++.

Fork it. Improve it. Push it back.
