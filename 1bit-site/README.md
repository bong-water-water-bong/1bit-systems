# 1bit-site

Static site for **1bit.systems** — landing page for the 1-bit monster
(BitNet-b1.58 on AMD Strix Halo).

## Layout

```
1bit-site/
  index.html          main landing
  install.sh          curl | bash bootstrap (Strix-Halo-gated)
  assets/             style.css + logo.svg (cyan #00d4ff on #0d1117)
  docs/               rendered wiki (build.rs + build.sh)
  voice/              1bit.voice PWA (mic → whisper → chat → kokoro)
  audio/              kokoro TTS sampler
  join/               private beta signup
  mobile/onboard/     phone onboarding pane
  _headers _redirects wrangler.toml   — Cloudflare Pages config
```

## Preview locally

```bash
cd 1bit-site
python3 -m http.server 8765
# → open http://localhost:8765/
```

## Render docs

Source of truth is `../docs/wiki/*.md` (do not edit under `1bit-site/docs/`).
To regenerate the HTML pages:

```bash
bash 1bit-site/docs/build.sh
```

`build.sh` compiles the rustc-only renderer in `docs/build.rs` on first
run, then emits one lowercase `.html` per `.md` into `1bit-site/docs/`.
No crates, no markdown library, no network.

## Deploy (Cloudflare Pages)

1. DNS: point `1bit.systems` NS records at Cloudflare (Namecheap → CF nameservers).
2. In the CF dashboard: Pages → Create → Connect Git → repo `halo-ai-rs` → production branch `main`.
3. Build command: `bash 1bit-site/docs/build.sh`
4. Build output directory: `1bit-site`
5. Add custom domain: `1bit.systems` + `www.1bit.systems`.

`wrangler.toml` is present for the CLI path (`wrangler pages deploy .`)
once you install `wrangler` locally; it's optional.

Domain: **1bit.systems** (Namecheap registration, Cloudflare nameservers + Pages hosting).
