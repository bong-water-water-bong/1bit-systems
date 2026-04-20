---
phase: verified
owner: anvil
---

# Crate: 1bit-lemonade

## Problem

Lemonade-SDK is the blessed AMD client surface for Ryzen AI + Strix Halo. Many third-party tools (vLLM bridges, Lemonade Discord community, GPT4All, etc.) already speak `http://127.0.0.1:8000/v1/*`. 1bit-lemonade is our parallel OpenAI-compat gateway on `:8200` that adds bearer-auth + per-token metrics + request logging so we can share the 1bit-server inference path with Lemonade-native clients without exposing the raw `:8180` endpoint.

Name: it's the gateway that **bridges** 1bit-server to the Lemonade ecosystem. Not a Lemonade binary clone — we don't re-run Lemonade; we present the same wire shape.

## Invariants

1. **No auth duplication.** 1bit-lemonade trusts Caddy's bearer gate. It accepts requests already authenticated upstream; it does NOT reimplement bearer validation. Defense in depth means Caddy is ONE layer + 1bit-lemonade is the API layer — not two bearer checkers.
2. **OpenAI v1 wire compat.** `/v1/chat/completions`, `/v1/models`, `/v1/embeddings` (if we add it later) match the canonical OpenAI shape exactly. SSE `data: {...}\n\n` framing on streaming responses.
3. **Upstream-agnostic.** 1bit-server is the only upstream today, but the dispatch layer is a trait so we can flip to lemond or flm without rewriting 1bit-lemonade.
4. **Request metrics are recorded, payloads are not.** Per-request: latency, token count, backend used, return status. Per-request body is NEVER logged — bearer tokens may appear in headers and prompts may contain PII.
5. **Fails open to 1bit-server on routing errors.** If our internal routing can't decide (dispatch trait can't pick an upstream), we fall through to 1bit-server on 127.0.0.1:8180. No 502 churn for the caller.

## Non-goals

- Not a model host. Doesn't load weights. Doesn't invoke HIP kernels. Pure HTTP gateway.
- Not a rate limiter. Quota logic lives in 1bit-server (future) or Caddy (today: bearer presence gate only).
- Not a Lemonade drop-in replacement. We speak its wire shape, not its internal config + model-management API.
- Not a Python-accessible surface at runtime. It's a Rust axum binary. External Python clients can still hit `:8200` over HTTP.

## Interface

```
GET  /v1/models               → list canonical 1bit systems model id(s)
POST /v1/chat/completions     → dispatch to 1bit-server /v1/chat/completions
POST /v1/completions          → dispatch to 1bit-server /v1/completions
GET  /healthz                 → plain 200 if 1bit-server is reachable
GET  /metrics                 → Prometheus text format (per-route counters + histograms)
```

Config:

```rust
pub struct GatewayConfig {
    pub bind: SocketAddr,              // default 127.0.0.1:8200
    pub upstream: url::Url,             // default http://127.0.0.1:8180
    pub upstream_timeout: Duration,     // default 10min (long SSE streams)
}
```

## Test matrix

| Invariant | Test |
|---|---|
| 2 (wire compat) | `v1_chat_completions_streams_like_openai`, `v1_models_shape_matches` |
| 3 (upstream-agnostic) | `dispatch_trait_is_object_safe`, mock upstream test |
| 4 (no body logging) | `tracing_subscriber_filter_strips_prompt_field` |
| 5 (fail-open) | `routing_error_falls_through_to_onebit_server` |

## TODO

- [x] axum routes live (`:8200` active via `strix-lemonade.service`)
- [x] `/v1/models` + `/v1/chat/completions` + `/healthz`
- [x] `/metrics` endpoint — Prometheus text format, counter + histogram + upstream-up gauge wired through the dispatch layer (2026-04-20)
- [ ] `/v1/embeddings` — not shipped (1bit-server doesn't expose embeddings yet)
- [x] Dispatch-trait abstraction — `Upstream` trait + `HaloServer` impl live (2026-04-20); swap to lemond/flm by constructing a different `Arc<dyn Upstream>` at boot, no route changes

## Spec cross-ref

| Spec section | Code file |
|---|---|
| Interface | `crates/1bit-lemonade/src/main.rs` |
| Invariant 1 (no auth dup) | Caddy handles bearer; no auth code here |
| Invariant 4 (no body log) | `tracing` filter in main — verify doesn't capture `body` field |

## Phase: verified

Promoted 2026-04-20. Checklist:
- [x] `/metrics` endpoint is live (Prometheus text, `# TYPE` markers for counter + histogram + gauge)
- [x] Dispatch trait extracted from hardcoded 1bit-server (`Upstream` trait in `src/dispatch.rs`, `HaloServer` is the today-impl)
- [x] Smoke: 23 in-crate tests green under `cargo test -p 1bit-lemonade --release`, including object-safety + swappable-backends + /metrics format assertions
- [ ] Prometheus scrape from 1bit-landing confirms counters update in prod — pending landing-side wiring (tracked separately; doesn't block gateway phase)

## Cross-refs

- `docs/wiki/Hermes-Integration.md` — Hermes hits us at `:8200` as a canonical Lemonade peer
- `docs/wiki/SDD-Workflow.md` — phase gate framework
- `crates/1bit-lemonade/` — current code
- `strixhalo/systemd/strix-lemonade.service` — user unit
