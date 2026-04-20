# halo-ai.studio — static landing site

Static HTML/CSS that will deploy at **halo-ai.studio**. No JavaScript, no build step, no tracking. Dark theme + cyan accent, per the architect-website ethos (clean, simple, everything visible, no hamburger menus).

## Layout

```
studio-site/
├── index.html        # single page, 155 lines
├── assets/
│   ├── logo.svg      # same halo ring glyph as crates/halo-landing
│   └── style.css     # dark theme + metrics grid + card grid + responsive
└── README.md
```

## Sections

- **Hero** — tagline + 6-cell live-metric grid (tok/s, PPL, test count, parity %).
- **The stack** — 6 pillar cards (Rust, HIP, MLX, OpenAI API, Lemonade shim, MCP).
- **Benchmarks** — the 4-row table from project_bitnet_live_bench (83 / 73.5 / 71.1 / 68.6 tok/s).
- **Works with** — Open WebUI, LibreChat, DSPy, Claude Code, halo-gaia, BYO.
- **The studios** — the architect / the halos / halo-ai core / halo-agents umbrella.
- **Install** — `install-strixhalo.sh` one-liner + CachyOS note.

## Preview locally

```bash
cd studio-site
python3 -m http.server 8300
# → http://127.0.0.1:8300/
```

## Deploy options

1. **Caddy on strixhalo**: add a `handle_path /studio/*` block pointing at this dir, LAN-only. Already have the pattern — see `strixhalo/caddy/Caddyfile`.
2. **Pi public-facing**: rsync to `/var/www/halo-ai.studio/` on the Pi (100.64.0.4), add Caddy `halo-ai.studio { file_server }` block, route via DNS once the domain lands.
3. **Cloudflare Pages**: `wrangler pages deploy studio-site/`. Free, global CDN, HTTPS out of the box.

Default path: deploy to Pi with Caddy. The Pi is already the canonical archive and has public bandwidth. Domain (halo-ai.studio) needs an A record at the Pi's WAN IP or a Cloudflare tunnel.

## Design decisions

- **No JS** — works on every reader/browser, zero attack surface, no tracking.
- **No build** — HTML/CSS is shipped as-is. Future-proof for 20 years.
- **Static metrics** — the big numbers are snapshot at publish time, not live. For live tok/s, `crates/halo-landing` on `:8190` is the interactive sibling.
- **Dark only** — matches the Man Cave + architect-website aesthetic. No light mode this run.
- **Single page** — every section visible from the top. No hamburger, no scroll hunting.

## Content maintenance

When the numbers drift (tok/s / PPL / test count), update `index.html` in place — grep for the number in the Hero `.metrics` block and the Benchmarks `<table>`. Consider wiring a `halo site-refresh` subcommand that rewrites those numbers from `halo bench` + `halo ppl` output, but that's a future sprint.
