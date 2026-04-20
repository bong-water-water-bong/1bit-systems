#!/usr/bin/env bash
# halo-mesh-revoke.sh — expire a Headscale pre-auth key + drop the user's
# bearer token from Caddy's bearer file.
#
# Usage: strixhalo/bin/halo-mesh-revoke.sh <handle>

set -euo pipefail

BEARER_FILE=/etc/caddy/bearers.txt

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <handle>" >&2
    exit 2
fi
handle="$1"

# 1. Expire all pre-auth keys for this user.
sudo headscale preauthkeys --user "$handle" expire 2>&1 | head -5 || true

# 2. Drop the user's bearer from /etc/caddy/bearers.txt.
if [[ -f "$BEARER_FILE" ]]; then
    tmp=$(mktemp)
    sudo grep -v "# ${handle}[[:space:]]" "$BEARER_FILE" > "$tmp" || true
    sudo install -o caddy -g caddy -m 0640 "$tmp" "$BEARER_FILE"
    rm -f "$tmp"
fi

# 3. Remove any registered nodes belonging to this user (optional — they
#    can rejoin with a new key tomorrow).
sudo headscale users destroy --name "$handle" 2>&1 | head -3 || true

# 4. Reload Caddy.
sudo systemctl reload caddy.service

echo "[mesh] $handle revoked — Headscale + bearer cleared."
