#!/usr/bin/env bash
# 1bit-mesh-revoke.sh — expire a Headscale pre-auth key + drop the user's
# bearer token from Caddy's bearer file.
#
# Usage: strixhalo/bin/1bit-mesh-revoke.sh <handle>

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

# 4. Rebuild the shared Caddy bearer matcher + reload. Revoking one user
#    must leave the other nine accepted — the helper regenerates
#    /etc/caddy/bearers.conf from the current bearers.txt.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/1bit-caddy-bearers.sh" --reload

echo "[mesh] $handle revoked — Headscale + bearer cleared."
