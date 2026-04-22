# Observability

Three levels: service logs, kernel-level profiling, and model-quality benchmarking. Everything local; no telemetry leaves the host.

## Logs

```bash
# service logs
journalctl -u 1bit-halo-bitnet -f
journalctl -u 1bit-halo-server -f --since "1h ago"

# user-scope services
journalctl --user -u 1bit-halo-memory-sync.timer
```

Structured JSON logs with Loki + Grafana on the same host are planned; for now, `journalctl -o json` is the paved road.

## Kernel profiling

```bash
# bandwidth-bound sanity check
rocprof --stats --timestamp on \
  ./build/bitnet_decode --model 1bit-halo-v2.h1b --port 8080 --bench 64

# expect: ternary GEMV at ~92% LPDDR5 peak
# if lower, the tile or packed layout regressed
```

## Model quality — PPL harness

```bash
# wikitext-103 perplexity · post-RoPE-fix reference numbers:
# 1bit-halo-v2: PPL ~12 on wikitext-103, ~1.04 on repetition
./build/bitnet_decode \
  --model ~/1bit-halo/models/1bit-halo-v2.h1b \
  --ppl ~/datasets/wikitext-103/wiki.test.tokens \
  --context 2048
```

## Live benchmark

```bash
# clean-burn reference numbers (2026-04-18):
# 64-token context:   66 tok/s
# 1024-token context: 33 tok/s
./build/bitnet_decode --bench 64 --bench 256 --bench 1024
```

## Output conventions

On the reference host, benchmark JSON lands in `/home/bcloud/claude output/`. Other hosts pick their own path; the convention matters for the project's internal tracking only.
