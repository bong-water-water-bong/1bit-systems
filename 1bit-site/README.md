# 1bit-site

Static landing page for **1bit.systems** — single self-contained `index.html`
with inline CSS + vanilla ES modules. Zero CDN, zero analytics, no framework.

## Deploy (Cloudflare Pages)

1. Push this repo to GitHub.
2. Cloudflare dashboard → Pages → Create → Connect to Git → pick the repo.
3. Build command: **(none)**.
4. Build output directory: **`/`** (or `1bit-site` if deploying from monorepo root).
5. Save and deploy. Add custom domain `1bit.systems` in Pages → Custom domains.
6. `_headers` carries CSP + cache rules automatically.

Local preview: `python3 -m http.server 8765` inside this dir, then hit `localhost:8765`.
