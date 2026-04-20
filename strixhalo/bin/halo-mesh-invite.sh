#!/usr/bin/env bash
# halo-mesh-invite.sh — generate Headscale pre-auth key + per-user bearer
# token for the halo-ai 10-seat private beta.
#
# Usage:
#   strixhalo/bin/halo-mesh-invite.sh <handle>
#
# where <handle> is the invitee's nickname / GitHub-handle. We print a
# ready-to-send message containing:
#   - the --authkey they paste into `tailscale up`
#   - the bearer token they set on /v2/* requests
#   - the LAN URL they hit once joined
#
# State:
#   - Headscale users + keys managed via the `headscale` CLI.
#   - Per-user bearer tokens stored one-per-line in
#     /etc/caddy/bearers.txt (0640 caddy:caddy), comment prefix "#".
#     We append a new line: "<token>  # <handle>  # <date>"
#
# Reload Caddy after editing bearers.txt:
#   sudo systemctl reload caddy.service
#
# Revoke:
#   strixhalo/bin/halo-mesh-revoke.sh <handle>      # expires both
#
# Beta cap: 10 users. Script refuses the 11th invite; bump MAX_USERS when
# we're ready to scale past the parity-burnin gate.

set -euo pipefail

MAX_USERS=10
BEARER_FILE=/etc/caddy/bearers.txt

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <handle>" >&2
    exit 2
fi
handle="$1"

# Sanity: handle is alphanumeric + dashes, 3-32 chars.
if ! [[ "$handle" =~ ^[a-zA-Z0-9-]{3,32}$ ]]; then
    echo "error: handle must be 3-32 chars, [a-zA-Z0-9-] only" >&2
    exit 2
fi

# Enforce the 10-user cap on the bearer file (one non-comment line per user).
if [[ -f "$BEARER_FILE" ]]; then
    active=$(sudo grep -cv '^\s*#\|^\s*$' "$BEARER_FILE" || true)
    if (( active >= MAX_USERS )); then
        echo "error: bearer file at cap ($active/$MAX_USERS). Raise MAX_USERS in $(basename "$0") once parity burnin passes." >&2
        exit 3
    fi
fi

# Create the Headscale user if missing.
if ! sudo headscale users list | grep -qw "$handle"; then
    echo "[mesh] creating Headscale user '$handle'"
    sudo headscale users create "$handle" >/dev/null
fi

# Generate a 24-hour, single-use pre-auth key.
authkey=$(sudo headscale preauthkeys create --user "$handle" --expiration 24h --reusable=false 2>/dev/null \
          | grep -oE '[a-f0-9]{48,}' | head -1)
if [[ -z "$authkey" ]]; then
    echo "error: failed to generate Headscale pre-auth key" >&2
    exit 4
fi

# Mint a per-user bearer token. sk-halo-<32 random bytes hex>.
bearer="sk-halo-$(openssl rand -hex 16)"

# Append to the Caddy bearer file, root-owned + caddy-readable.
# We tag each line with both `issued` and `expires`. The 10-day TTL is
# enforced by strixhalo/bin/halo-beta-expire.sh (timer-driven).
# Keep both fields: `issued` for audit, `expires` for the sweeper.
# Policy doc: docs/wiki/Beta-10-Day-TTL.md
tmp=$(mktemp)
if [[ -f "$BEARER_FILE" ]]; then
    sudo cp "$BEARER_FILE" "$tmp"
fi
issued_at="$(date -u +%Y-%m-%dT%H:%MZ)"
expires_at="$(date -u -d '+10 days' +%Y-%m-%dT%H:%MZ)"
printf "%s  # %s  # issued %s  # expires %s\n" \
    "$bearer" "$handle" "$issued_at" "$expires_at" >> "$tmp"
sudo install -o caddy -g caddy -m 0640 "$tmp" "$BEARER_FILE"
rm -f "$tmp"

# Rebuild the shared Caddy bearer matcher + reload. The helper is the
# single source of truth for /etc/caddy/bearers.conf — every route
# imports the (bearer_matcher) snippet so all bearers live/revoke
# together without editing the Caddyfile itself.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/halo-caddy-bearers.sh" --reload

# Print ready-to-send invite message.
cat <<EOF

╔══════════════════════════════════════════════════════════════╗
║ halo-ai mesh invite — paste below to @$handle                 ║
╚══════════════════════════════════════════════════════════════╝

Welcome to the halo-ai 10-seat private beta.

One-time steps (Linux / macOS):

  # 1. Install Tailscale
  #    Arch:   sudo pacman -S tailscale && sudo systemctl enable --now tailscaled
  #    Other:  https://tailscale.com/download

  # 2. Join the halo-ai mesh
  sudo tailscale up \\
       --login-server https://headscale.halo-ai.studio \\
       --authkey $authkey

  # 3. Test completions
  curl https://strixhalo.local/v2/v1/chat/completions \\
       -H "Authorization: Bearer $bearer" \\
       -H "content-type: application/json" \\
       -d '{"model":"halo-1bit-2b","messages":[{"role":"user","content":"hi"}]}'

Mobile (Tailscale iOS / Android): in the app, add a custom
coordination server https://headscale.halo-ai.studio and paste the
same --authkey.

Browser: once on the mesh, install our root CA cert once —
  https://strixhalo.local/ca/root.crt
then https://strixhalo.local/studio/ loads natively.

This bearer expires on $expires_at (10-day beta TTL). To extend,
ask bcloud to re-run halo-mesh-invite.sh for a fresh 10-day token.

Revoke on our side: strixhalo/bin/halo-mesh-revoke.sh $handle

EOF
