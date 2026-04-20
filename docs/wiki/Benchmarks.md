# Live tok/s + PPL

What we measure and what each number means. All numbers taken on the production box (`strixhalo`, 100.64.0.1) unless noted.

## The table

| metric | value | source | what it means |
|---|---:|---|---|
| Decode throughput @ L=64 | **83 tok/s** | `bench.sh` | Short-context user-chat speed. What you feel typing a prompt. |
| Decode throughput @ L=1024 | **33 tok/s** | `bench.sh` | Long-context agent speed. KV-cache bandwidth dominates here. |
| PPL, wikitext-103 (gen-2 Rust) | **9.1805** | `benchmarks/ppl-gen2.sh` | Distribution-level quality. Lower is better. |
| PPL, wikitext-103 (gen-1 baseline) | **9.1607** | historical | Reference point; gen-2 is +0.02, inside ±0.05 tolerance. |
| Shadow-burnin byte-exact | **95.55%** | `halo burnin stats` | Argmax-level parity, gen-1 vs gen-2. 14,344 rounds. |
| Ternary GEMV roofline | **92% of LPDDR5 peak** | `rocprof` | Kernel is bandwidth-bound, not compute-bound. Bytes-read reduction (Sherry) is rank-1. |
| Split-KV Flash-Decoding attn | **6.78× vs prior** @ L=2048 | `benchmarks/attn_fd.sh` | Bit-exact speedup over single-block attention. Default since 2026-04-19. |
| Voice mouth-to-ear first audio | **1.23 s** | `benchmarks/voice.sh` | End-to-end: STT + LLM + TTS first chunk. 3-5× faster than naive serial loop. |
| Tests across 13 crates | **201 passing, 0 failing** | `cargo test --workspace --release` | Workspace-wide green. CI gate. |
| `halo-server` binary, stripped | **2.4 MB** | `size target/release/halo-server` | Static-friendly Rust binary; ships without a runtime. |
| Landing live tok/s | **pulled from `/metrics` via `/_live/stats` SSE** | `crates/halo-landing/src/telemetry.rs` | The number you see in the hero on `https://strixhalo.local/` is no longer a static guess — it's the same `tokps_recent` the Prom scraper sees, pushed over SSE every 1.5 s. |

## One-liners per number

- **83 @ 64 / 33 @ 1024** — same kernel; the drop is pure KV-cache bandwidth. A 70B FP16 model on the same box does ~18 tok/s. See [Why ternary?](./Why-Ternary.md).
- **PPL 9.1805** — we re-pack Microsoft's weights (no retraining). The +0.02 delta vs gen-1 is FP16-reordering noise, not quality loss.
- **95.55% byte-exact** — **74.9% of the remaining 4.45% is one prompt** (idx=7, "chemical symbol for gold"). Fix that sampler delta and parity jumps to ~98.9%. See [Why parity gates?](./Why-Parity-Gates.md).
- **92% of peak** — roofline math from `rocprof` confirms the ternary GEMV is memory-bound. Compute-side optimization has near-zero headroom; Sherry 1.25-bit packing (bytes-read reduction) is the next 15-25% lever.
- **6.78× attn** — one block per head → split across blocks with an online-softmax reduction. Bit-exact means PPL didn't move when we flipped the default.
- **1.23 s voice** — whisper.cpp partials + sentence-boundary TTS streaming. Naive serial (wait-full-STT → wait-full-LLM → wait-full-TTS) is ~4-6 s.
- **201 tests** — includes property tests on tokenizer round-trips and parity assertions on `.h1b` load paths.
- **2.4 MB** — the whole serving daemon. For comparison, gen-1 Python install was ~1.2 GB.

## Reproducibility notes

- **Box**: Ryzen AI 9 HX 370 (Strix Halo / gfx1151), 128 GB LPDDR5 @ 256 GB/s, CachyOS, kernel 7.0.
- **Ambient**: 22-24 °C room temperature, closet airflow, headless. Thermal throttle shows up above ~28 °C — numbers above are the 22 °C set.
- **Power**: `halo power balanced` profile, 150 W sustained TDP, voltage curve at default. See [Why `halo power`?](./Why-halo-power.md) for the under-volt recipe that adds ~4% throughput without stability loss.
- **Model**: `microsoft/bitnet-b1.58-2B-4T` → `.h1b` v3, 1.8 GB on disk.
- **Temperature**: 0 (greedy argmax) for parity + PPL; 0.7 for voice UX only.

Re-run locally with:

```bash
cd /home/bcloud/repos/halo-workspace
./benchmarks/bench.sh           # decode throughput
./benchmarks/ppl-gen2.sh        # PPL
./benchmarks/shadow-burnin.sh   # parity (long-running)
halo burnin stats               # live summary
```

Numbers regenerate into `~/claude output/`.

## Links

- [Why shadow-burnin?](./Why-Shadow-Burnin.md) — how the parity number is produced
- [Why parity gates?](./Why-Parity-Gates.md) — how we gate cutover on it
- [Why this way + how?](./Why-This-Way-How.md) — long-form walkthrough of the full stack
