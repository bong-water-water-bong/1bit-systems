# 1bit-systems wiki

Plain-English explanations of the architectural calls we made — one page per decision, citations where they exist.

## Decisions

- [Why 1.58-bit ternary?](./Why-Ternary.md) — the weight format
- [Why Rust above + C++ below?](./Why-Rust.md) — the language split
- [Why Strix Halo (gfx1151)?](./Why-Strix-Halo.md) — the hardware target
- [Why shadow-burnin?](./Why-Shadow-Burnin.md) — the cutover discipline
- [Why our own `.h1b` format?](./Why-H1b-Format.md) — vs GGUF
- [Why Caddy + systemd?](./Why-Caddy-Systemd.md) — the ops layer
- [Why 1bit-agents?](./Why-Halo-Agents.md) — the self-maintaining mesh
- [Why no Python at runtime?](./Why-No-Python.md) — Rule A
- [Why shadow-traffic parity gates?](./Why-Parity-Gates.md) — cutover criteria
- [Why no NPU yet?](./Why-No-NPU-Yet.md) — XDNA 2 status + ONNX/FastFlowLM/IREE evaluation
- [Why this way + how?](./Why-This-Way-How.md) — long-form walkthrough of all the decisions + end-to-end request path

## Integrations

- [Hermes Agent integration](./Hermes-Integration.md) — Nous Research's agent as external client on 1bit-server; feature-port list for 1bit-agents

## Benchmarks + proof

- [Live tok/s + PPL](./Benchmarks.md) — what we measure, what it means
- [FAQ](./FAQ.md) — short answers to common questions

## Pointers

- Architectural data-flow: [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)
- Cutover runbook: [`../../CUTOVER.md`](../../CUTOVER.md)
- Demo script: [`../../DEMO.md`](../../DEMO.md)
- Contributing: [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md)
- Repo conventions for AI agents: [`../../CLAUDE.md`](../../CLAUDE.md)
