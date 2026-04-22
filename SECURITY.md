# Security Policy

## Supported versions

`1bit-systems` is pre-1.0. Only the tip of `main` is supported. Tagged
releases (if any) are snapshots, not long-term support branches.

| branch / tag         | supported                |
| -------------------- | ------------------------ |
| `main`               | yes                      |
| any tagged release   | best-effort, no backports |

## Reporting a vulnerability

**Do not open a public GitHub issue** for anything with a security
impact. Instead, use GitHub's private vulnerability reporting:

**→ [Report a vulnerability](https://github.com/bong-water-water-bong/1bit-systems/security/advisories/new)**

Or email the maintainer directly. If the private-report button is
unavailable, a minimal disclosure (problem, versions affected, proof of
concept if you have one) is enough to start the conversation; we'll
move the rest to a private channel.

## What to expect

- **Acknowledgement** within 72 hours of the report.
- **Triage assessment** within 7 days (severity + reproducibility).
- **Fix or public advisory** within 30 days for anything confirmed
  high-severity; shorter for critical. If we need longer, we'll say so
  with a reason.
- **Credit** in the advisory if you want it, anonymous if you don't.

## Scope

In scope: everything under this repository — the Rust workspace, the
`rocm-cpp/` HIP kernels subtree, `strixhalo/` dotfiles, `install.sh`,
and the Caddy / systemd integration examples.

Out of scope (report upstream):

- `bun`, `lemonade-sdk`, `stable-diffusion.cpp`, `whisper.cpp`,
  `kokoro.cpp` — all upstream projects with their own security policies.
- Third-party clients (DSPy, Open WebUI, LibreChat) — upstream.
- Hardware / firmware issues (AMD ROCm stack, kernel modules, XDNA2
  driver) — report to AMD.

## Known-sensitive areas

If you're looking for something to pick apart, these are worth a harder
look than average:

- **`/etc/caddy/Caddyfile` bearer-token check** — reverse-proxy surface
  for `/v2/*`. Placeholder in `strixhalo/caddy/Caddyfile` is
  `sk-halo-REPLACE_ME`; real tokens live only in the root-owned file.
- **GGUF / `.h1b` loaders** — we `mmap` attacker-controlled files. Any
  out-of-bounds read or integer-overflow in the parser is in scope.
- **`1bit-mcp`** — stdio JSON-RPC bridge. Input comes from whatever is
  calling the MCP server.
- **`install.sh`** — curl-pipe-bash entrypoint. Served from CF Pages at
  `1bit.systems/install.sh`. Any way to execute unintended code from it
  is critical.

## What we don't treat as a vulnerability

- **Out-of-memory on over-sized prompts.** `1bit-server` is a
  single-tenant local service, not a public API. Rate-limiting is the
  operator's job (Caddy layer).
- **Model output.** Hallucinations, unsafe completions, prompt injection
  in the generated text are model-behavior problems, not security bugs
  — file them as regular issues.
- **Tokens in your own dev logs.** `1bit-server` logs request metadata,
  not bearers. If you're logging your own bearer to stdout, fix your
  Caddyfile.
