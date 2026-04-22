# Eight Models Roadmap — matching AMD's Strix Halo demo

**Target.** Run 8 distinct inference workloads concurrently on one
Strix Halo 128 GB box, matching the many-model narrative AMD pushed in
the "bun fight" VideoCardz piece, with native HIP kernels instead of
generic llama.cpp. Halo's ternary weights make 8 models cheaper on
memory than AMD's fp16 demo.

## Gate

**OPTC CRTC hang must stay silent under sustained 8-model load.**
`halo-gpu-perf.service` pins SCLK=high (Tier-1). If the hang still fires
under the soak test, escalate to `/etc/modprobe.d/amdgpu-halo.conf`
(`ppfeaturemask=0xffffbfff runpm=0`) which requires reboot. Without
this gate, 8-way concurrency is crash roulette.

## Current inventory (as of 2026-04-21)

| Slot | Service          | Weight size | Backend              | Live ? |
|------|------------------|-------------|----------------------|--------|
| 1    | halo-bitnet      | 0.9 GB      | rocm-cpp native HIP  | yes    |
| 2    | strix-server     | 0.9 GB      | rocm-cpp (gen-2)     | yes    |
| 3    | halo-sd          | 7 GB (SDXL) | sd.cpp native HIP    | yes    |
| 4    | halo-whisper     | 142 MB      | whisper.cpp CPU      | yes    |
| 5    | halo-kokoro      | 310 MB      | onnxruntime CPU      | yes    |
| 6    | **halo-wan**     | *5 B DiT*   | sd.cpp master path   | **missing** |
| 7    | **halo-embed**   | *100 MB*    | onnxruntime CPU      | **missing** |
| 8    | **halo-medusa**  | *+ heads*   | rocm-cpp (tree-attn) | **missing** |

Live weights in use today: ~9.2 GB. Headroom before hitting the
COMPOSITOR_RESERVE floor: ~55 GB (per `halo budget` snapshot 2026-04-21).

## Slot 6 — halo-wan (video generation)

**Pick.** Wan 2.2 TI2V-5B. Apache 2.0, 5 B parameters, DiT architecture,
already supported in sd.cpp master via the `--model-type ti2v` flag.
See `project_video_gen_pick.md`.

**Weight budget.** ~10 GB fp16 / ~3 GB q4. Activation cache is the
real cost: worst case ~45 GB during a 720p generation pass, so slot 6
cannot run concurrently with a full-res slot 3 burst. Gate it with the
sommelier router: ("sd active" XOR "wan active").

**LOC estimate.** 200 LOC in `1bit-lemonade` for the gateway route +
300 LOC in `1bit-server` for pipeline wiring + systemd unit. No new
kernel family — sd.cpp upstream handles it. 2-4 weeks end-to-end.

**Blockers.** OPTC gate. sd.cpp PR review on the TI2V path. ffmpeg
dep for mp4 mux (ffmpeg-sys-next, build-time feature).

## Slot 7 — halo-embed (embeddings server)

**Pick.** BAAI/bge-small-en-v1.5 ONNX. ~33 MB fp16, 384-dim, MIT,
already one of the fastest CPU embedders with strong MTEB numbers.
Alternative: nomic-ai/nomic-embed-text-v1.5 (~140 MB, longer context
2048, competitive quality).

**Weight budget.** 100 MB peak. Runs on CPU via onnxruntime — does not
compete for gfx1151. Can stay live permanently.

**LOC estimate.** 150 LOC in a new `1bit-embed` crate (tokenizer +
onnxrt call + axum `/v1/embeddings` that matches the OpenAI shape). ~3
days. systemd unit on :8084.

**Why it matters.** Cartograph is memory-index-only today. A real
embedder lets halo-agent retrieval actually converge on relevant
passages; without it, the "recall" MCP tool is keyword-only.

## Slot 8 — halo-medusa (speculative-decoding LLM)

**Pick.** parrishcorcoran/MedusaBitNet-2B-4T (per
`project_medusa_plan.md`). Medusa heads bolted onto BitNet-2B; tree
attention + small-M ternary GEMM kernels required.

**Weight budget.** Shared backbone with slot 1 (halo-bitnet can serve
as the base); heads add ~30 % → ~1.2 GB. Cheapest slot of the three.

**LOC estimate.** 800 LOC between a new `tree_attn.hip` kernel, a
small-M ternary GEMM variant, FFI glue, and a 1bit-server route. 3-4
weeks if Medusa heads train cleanly, longer if we have to fine-tune.

**Blockers.** Sherry requantizer bug (`project_sherry_root_cause.md`)
— do not touch Medusa until Sherry retrain lands; otherwise we'd be
chasing two kernel-correctness ghosts at once.

## Order of execution

1. **OPTC soak test** — burn the existing 5-model workload + shadow
   traffic for ≥ 24 h under the SCLK pin. No OPTC signatures = ready
   to grow. `journalctl -b | grep -c optc35_disable_crtc` must return 0.
2. **Slot 7 first** — cheapest, CPU-only, unlocks cartograph's real
   recall. Ships in a week.
3. **Slot 6 second** — biggest user-visible win (video). Multi-week
   port but low kernel risk.
4. **Slot 8 last** — gated on Sherry correctness. Biggest perf upside
   but deepest kernel work.

## Demo surface

When all 8 live, `halo budget` will print:

```
GTT   ~23 GB / 67 GB  (44 GB free)
RAM   ~95 GB / 128 GB available
next model budget  ≈ 40 GB
---- halo services ----
  bitnet_decode            ~0.9 GB
  halo-server-real         ~0.9 GB
  sd-server                 ~7 GB
  wan-server                ~10 GB
  embed-server              ~0.1 GB
  medusa-head               ~1.2 GB
  whisper-server            ~0.2 GB
  kokoro                    ~0.4 GB
```

Eight rows, ~22 GB weights. AMD's demo: 8 rows, ~48 GB weights. Same
model count, less than half the memory — that's the halo headline.
