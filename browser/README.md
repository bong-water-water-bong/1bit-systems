# halo-browser

Bun + puppeteer-core scripts that **attach** to a long-running Chrome
instance with its own isolated user-data profile. Your daily Chrome
(default profile) is not touched — halo-browser runs in parallel and
shares nothing with it.

## Why attach instead of spawn?

- Log in once (Reddit / Discord / GitHub / etc.) in the halo-browser
  window; cookies persist across script runs.
- Real browser = no headless-detection weirdness on sites that sniff.
- `disconnect()` at script end keeps Chrome running for the next script.

## Layout

```
browser/
├── launch-chrome.sh    opens the halo-browser Chrome on :9222
├── package.json        bun + puppeteer-core
├── src/
│   ├── attach.ts       sanity: list open tabs
│   ├── screenshot.ts   save a PNG of any URL
│   └── fetch.ts        authed fetch via the attached session
└── README.md
```

## First run

```bash
cd browser
bun install                          # pulls puppeteer-core
./launch-chrome.sh &                  # opens halo-browser Chrome
# (log into whatever sites you need — once. Profile persists.)
bun run src/attach.ts                 # sanity check
bun run src/screenshot.ts https://strixhalo.local/  /tmp/landing.png
```

## Env overrides

- `HALO_BROWSER_PROFILE` — user-data-dir (default `~/.local/share/halo-browser/profile`)
- `HALO_BROWSER_PORT` — CDP port (default `9222`)
- `HALO_BROWSER_BIN` — browser binary (default `google-chrome-stable`)

## Safety notes

- `browser.disconnect()` at the end of every script — **never** `browser.close()`.
  `close()` would kill the user's Chrome window and profile session.
- Scripts open/close their own `newPage()` rather than stealing existing tabs.
- The `--remote-debugging-port` is bound to 127.0.0.1 only. External LAN
  clients cannot drive the browser.
