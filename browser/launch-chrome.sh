#!/usr/bin/env bash
# launch-chrome.sh — start an isolated Chrome profile for halo-browser scripts.
#
# Daily Chrome (default profile) is untouched. This opens a SECOND Chrome
# with its own user-data-dir and a remote debugging port so Bun/puppeteer
# scripts can attach via CDP. Log in once here (Reddit / Discord / GH),
# cookies persist across runs.

set -euo pipefail

PROFILE_DIR="${HALO_BROWSER_PROFILE:-$HOME/.local/share/halo-browser/profile}"
DEBUG_PORT="${HALO_BROWSER_PORT:-9222}"
CHROME="${HALO_BROWSER_BIN:-google-chrome-stable}"

mkdir -p "$PROFILE_DIR"

# Don't relaunch if one is already listening on the debug port.
if curl -fsS --max-time 1 "http://127.0.0.1:$DEBUG_PORT/json/version" >/dev/null 2>&1; then
    echo "halo-browser already up on :$DEBUG_PORT"
    curl -sS "http://127.0.0.1:$DEBUG_PORT/json/version" | head -c 300
    echo
    exit 0
fi

echo "launching $CHROME on :$DEBUG_PORT, profile=$PROFILE_DIR"
exec "$CHROME" \
    --user-data-dir="$PROFILE_DIR" \
    --remote-debugging-port="$DEBUG_PORT" \
    --remote-allow-origins=http://127.0.0.1:$DEBUG_PORT \
    --no-first-run \
    --no-default-browser-check \
    --disable-background-networking \
    --disable-background-timer-throttling \
    --disable-breakpad \
    --disable-client-side-phishing-detection \
    --disable-default-apps \
    --disable-sync \
    "$@"
