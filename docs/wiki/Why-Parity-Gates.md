# Why shadow-traffic parity gates?

**One-line answer**: unit tests verify behaviors; parity gates verify **indistinguishability from the previous version**. A PR that adds a feature and keeps tests green can still subtly shift logits. Parity says: prove it's the same.

## Tests vs parity

| dimension | unit tests | parity gates |
|---|---|---|
| what | "does this function return the right thing" | "does gen-2 give the same bytes as gen-1" |
| shape | fixed inputs, asserted outputs | live prompt stream, byte-compare |
| coverage | one code path | emergent end-to-end behavior |
| breaking cost | cheap fix | **cannot ship until green** |

Unit tests run on every PR. Parity runs before every cutover.

## The two harnesses

### Shadow-burnin (argmax byte-compare)

`benchmarks/shadow-burnin.sh` fires the same prompt at `/v1` (gen-1 C++ bitnet_decode on :8080) and `/v2` (gen-2 Rust 1bit-server on :8180), diffs the replies byte-for-byte. Pace: ~1 round / 2-3 s.

- **Current**: **95.55% byte-exact** over **14,344 rounds** (2026-04-20)
- **Drift analysis**: **74.9% of all misses trace to ONE prompt** — `idx=7`, "chemical symbol for gold". That's a single sampler delta, not a systemic divergence. Fix it and parity jumps to **~98.9%**.

More detail on what the not-byte-exact rounds mean in [Why shadow-burnin?](./Why-Shadow-Burnin.md) — short version: sub-ULP FP16 noise propagated through 30 layers flips argmax at logit ties. Not a bug.

### PPL (distribution-level)

`benchmarks/ppl-gen2.sh` runs wikitext-103 perplexity on both servers.

- **gen-1 baseline**: 9.1607
- **gen-2 current**: 9.1805
- **delta**: +0.02, inside the ±0.05 tolerance → **PASS**

PPL measures the **distribution** both models see. Shadow-burnin measures the **argmax pick**. Different failure modes; we gate on both.

## Why we gate the gates

From [`CLAUDE.md`](../../CLAUDE.md) §Testing:

> Parity vs gen-1 is the ultimate cutover gate.

A PR that passes `cargo test --workspace --release` but drops parity below threshold **cannot ship**. The human running cutover runs:

```bash
halo burnin stats       # overall byte-exact rate
halo burnin drift       # what's missing and why
halo burnin recent      # last N rounds
halo burnin since 2026-04-19T00:00:00Z
```

If any of those trend down in the 48 hours before cutover, we don't flip. Receipts matter more than vibes.

## The operator interface

`halo burnin {stats,drift,recent,since}` reads the JSONL log at `~/claude output/shadow-burnin.jsonl` (note the space in the path, quote it). State persists at `~/.local/share/1bit systems/shadow-burnin.state`.

```bash
halo burnin stats
# bytes_exact:   14344 / 15012  (95.55%)
# top_miss:      idx=7 "chemical symbol..." (75.0% of misses)
# ppl_gen1:      9.1607
# ppl_gen2:      9.1805  (Δ +0.02)  PASS
```

The `top_miss` field is the actionable output. Drift on one prompt is a sampler bug. Drift on many prompts is a model bug.

## Why this rigor

This is a single-box project serving requests to a single user. There's no A/B infra, no gradual rollout, no blue-green. The only way to flip gen-1 → gen-2 safely is to prove the behaviors match **before** flipping the route in Caddy.

We've been burned twice:
1. The KV-cache reset bug (2026-04-19 SEGV) — PPL passed, only burnin caught it at round ~200.
2. The RoPE convention bug (2026-04-19) — PPL reported 524 on long context. Caught before any cutover attempt.

Both would have shipped if the gate was "tests pass." Parity gates are the seatbelt.

## Pointers

- Shadow-burnin harness: `benchmarks/shadow-burnin.sh`
- PPL harness: `benchmarks/ppl-gen2.sh`
- Data: `~/claude output/shadow-burnin.jsonl`
- Deep dive: [Why shadow-burnin?](./Why-Shadow-Burnin.md)
- Formal rule: [`CLAUDE.md`](../../CLAUDE.md) §Testing
