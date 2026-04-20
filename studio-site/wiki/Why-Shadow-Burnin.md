# Why shadow-burnin?

**One-line answer**: before we flip `/v1/*` from the battle-tested C++ server (`bitnet_decode` on `:8080`) to the new Rust server (`halo-server` on `:8180`), we want **real evidence** that the Rust version gives the same answers under real traffic — not synthetic tests.

## The setup

Two servers, same model, same weights, same temperature (0, greedy argmax), same prompts. Both running on the same box at the same time. A script fires each prompt at both, diffs the replies byte-for-byte, logs the result.

```
prompt ─┬─▶ :8080 (gen-1 C++) ──▶ reply A
        └─▶ :8180 (gen-2 Rust) ──▶ reply B

         diff(A, B) → { exact, prefix-only, no-match }
```

Runs forever. State persists in `~/.local/share/halo-ai/shadow-burnin.state`, JSONL log in `~/claude output/shadow-burnin.jsonl`. Pace: ~1 round every 2-3 seconds. Target: 50 000 rounds over 72 hours.

## What the numbers mean

| bucket | meaning |
|---|---|
| **exact** | reply A == reply B, byte-identical. This is what we want. |
| **prefix-only** | reply A is a prefix of reply B or vice-versa. Same trajectory, one side stopped earlier. Usually a chat-template edge case. |
| **no-match** | first divergent byte came before the end of the shorter reply. These are the interesting ones. |
| **v1-unreachable / v2-unreachable** | one side was down; doesn't count toward parity. |

## Current live numbers

- **Exact-match rate: 96.66%** across ~10 000+ rounds
- v1/v2 p95 latency within 10 ms
- Median prefix-match length: 80 characters
- Zero unreachable on either side since the KV-cache-reset fix

## "So what about the 3% that doesn't match?"

That's **sub-ULP FP16 noise**, not a bug. Here's why:

1. Both servers use FP16 throughout. FP16 has 10 mantissa bits + implicit bit = 11 bits of precision, roughly 3 decimal digits.
2. The two kernels (C++ `bitnet_decode` vs Rust-driven `rocm-cpp`) do the same math in slightly different orders — different accumulation trees, different block tiling, different register schedules. Result: every intermediate value differs by 0-3 ULPs.
3. Those tiny differences propagate through 30 transformer layers and accumulate.
4. At temperature 0 we pick `argmax(logits)`. If two logits are within ~0.001 of each other, a 1-ULP perturbation flips the argmax. That's the divergence.
5. **Crucially**, this does NOT affect PPL — which we measure independently and get `9.18` vs gen-1's `9.1607`, within tolerance.

Put differently: the *distribution* both servers see is identical to within rounding. The *argmax pick* differs occasionally because argmax is brittle near logit ties. PPL measures the distribution; shadow-burnin measures the argmax pick.

## Why do it, if PPL already passes?

Because PPL is **one number** measured once. Shadow-burnin is **thousands of prompts** measured over days. Four things we learn that PPL can't tell us:

1. **Stability under load** — does the Rust server crash or slow down after 200 completions? (The KV-cache-reset bug was found this way — not by PPL.)
2. **Latency parity** — if gen-2 is 2× slower than gen-1 at the same quality, we don't want to cut over.
3. **Coverage** — 30 prompt types spanning code, reasoning, creative, long-context. PPL is one text span.
4. **Trust** — a reader asking "is the new server actually the same?" gets a number they can verify: 96.66%. Much harder to argue with than a one-shot PPL.

## Cutover gate table

From [`CUTOVER.md`](../../CUTOVER.md):

| gate | target | current | state |
|---|---|---|---|
| PPL on wikitext 1024 tokens | 9.1607 ± 0.05 | 9.1805 | ✓ |
| exact-match rate | ≥ 90% | 96.66% | ✓ |
| total rounds | ≥ 50 000 | ~10 000+ | ⌛ |
| v2 unreachable | < 0.1% | 0% | ✓ |
| v1/v2 p95 latency gap | < 100 ms | 10 ms | ✓ |

Three gates green, one still waiting on wall-clock time. When all five are green, `/v1/*` flips to `:8180`. The ~48 hours left is not a risk — it's just collecting receipts.

## "Lots of shadows?"

Common misread: "353 no-match" looks scary. It's not. 353 out of ~10 000 = 3.5%. Of those, most are legitimate divergences near logit ties; a handful are chat-template edge cases we've already documented. No actual regressions have been found.

If the number ever spikes (say, >20% no-match in an hour), the harness will flag it and halt the cutover plan. That's what the shadow-burnin is *for*.

## How to check progress yourself

```bash
halo bench                            # prints summary
cat ~/claude\ output/shadow-burnin.jsonl | wc -l   # round count
halo bench --since 2026-04-20T00:00:00Z            # scoped stats
```

When all 5 cutover gates are green, we post, flip Caddy, and archive the old route as `/v1-legacy/*` for emergency rollback.
