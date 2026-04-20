# Hermes Agent integration

[NousResearch/hermes-agent](https://github.com/nousresearch/hermes-agent) is a self-improving agent with a skill-creation loop, FTS5 session search, Honcho dialectic user modeling, messaging-gateway (Telegram/Discord/Slack/WhatsApp/Signal), cron scheduler, MCP integration, and subagent spawn. MIT, Python.

Rule A (no Python in runtime) forbids Hermes **on strixhalo as a service**. But nothing stops a user from running Hermes *on their laptop* talking to strixhalo's OpenAI-compat endpoint. That is the sanctioned path.

## External client — 5 minute setup

On the user's machine (Linux / macOS / WSL2 / Android Termux):

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc
hermes setup         # wizard
hermes model         # pick "custom endpoint"
```

When prompted for the endpoint, use the halo-server URL from the mesh:

```
base_url:    http://100.64.0.1:8180/v1          # Headscale mesh, strixhalo
model:       halo-1bit-2b
api_key:     sk-halo-local                       # ignored by halo-server
```

That's it. `hermes` TUI now drives halo-server. Every request lands on our native HIP kernels, no detour through NousPortal/OpenRouter.

## Expose halo-mcp to Hermes

halo-mcp is the Rust stdio JSON-RPC bridge that exposes every halo-agents specialist as an MCP tool. Point Hermes at it:

```bash
hermes config set mcp.servers.halo.command /path/to/halo-mcp
hermes config set mcp.servers.halo.args '[]'
```

Then inside a Hermes conversation:

```
/mcp list                   # see halo-agents specialists as tools
```

Hermes' 40+ built-in tools + our 17 halo-agents specialists → 57-tool agent on commodity Python, talking to native-HIP inference. Rule A untouched: Python lives on the laptop, kernels stay on strixhalo.

## Features worth porting into halo-agents

Hermes has genuine research content, not just a wrapper. Port candidates, ranked:

| Feature | Why port | Effort | Notes |
|---|---|---|---|
| Skill self-improvement loop | The actual novelty. Skills rewrite themselves during use. | 3-4 weeks | Would live in `halo-agents` as a new `SelfImproving` trait. Needs training-set of skill-revision traces — capture from Hermes first, then train. |
| FTS5 session search | Cheap, useful. Rust `rusqlite` has FTS5. | 3 days | Add to halo-agents memory layer. Replaces agent's current linear-scan recall. |
| Honcho dialectic user modeling | [plastic-labs/honcho](https://github.com/plastic-labs/honcho). Builds a growing model of who you are. | 1-2 weeks | Would bolt onto halo-agents profile layer. Honcho is Python — we'd reimplement the dialectic surface in Rust, not FFI. |
| Cron scheduling with platform delivery | We already have systemd timers. Hermes' model is nicer because natural-language scheduling. | 1 week | Thin wrapper around `halo-cli schedule` + a parser. |
| Subagent spawn with zero-context-cost turns | We do this today via `halo-agents`. | Already done | Keep. |
| Messaging gateway | Telegram/Discord/Slack/WhatsApp/Signal. | 2 weeks/platform | Do NOT port. Wrap Hermes instead — Python gateway on a user VPS is fine, strixhalo stays clean. |
| Agentic skill creation after complex tasks | Autonomous skill authoring. | 4-6 weeks | Depends on skill-improvement loop landing first. |

Skip: OpenClaw migration, Termux path, serverless (Modal/Daytona/Singularity) — none of this applies to strixhalo.

## Integration order

1. Wire Hermes → halo-server this week (above, 5 min on operator side).
2. Expose halo-mcp to Hermes (above, 2 min).
3. FTS5 session search into halo-agents memory (3 days, concrete win).
4. Everything else behind Medusa + Sherry + BitNet v2. Hermes' skill-loop is interesting, not urgent.

## Honest comparison

What Hermes has that halo-agents doesn't:
- Skill self-improvement loop (genuinely novel)
- Multi-platform messaging bridge
- 104k users, battle-tested

What halo-agents has that Hermes doesn't:
- Native HIP inference (Hermes is model-agnostic — the model is whatever endpoint you point it at)
- Rule A compliance (no Python in the hot path)
- Shadow-burnin parity harness
- Live on Strix Halo as a tiny-box mini-PC

Complementary, not competitive. Hermes is the cockpit; halo-ai-rs is the engine.

## References

- [Hermes README](https://github.com/NousResearch/hermes-agent)
- [Hermes docs](https://hermes-agent.nousresearch.com/docs/)
- [Honcho upstream](https://github.com/plastic-labs/honcho)
- [agentskills.io](https://agentskills.io) — open standard for skill sharing
- Our halo-mcp crate: `crates/halo-mcp/`
