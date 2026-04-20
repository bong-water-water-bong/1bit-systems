#!/usr/bin/env bash
# halo-anvil — clone/pull → build → bench → post to Discord as 🔨 anvil.
# Polls rocm-cpp for new commits. When HEAD advances, rebuild bitnet_decode
# and re-run the live tok/s sweep. Posts a compact summary via discord-mcp
# (webhook relay) to #changelog as specialist "anvil".

set -euo pipefail

REPO="$HOME/repos/rocm-cpp"
BENCH_LOG_DIR="$HOME/claude output"
STATE_FILE="$HOME/.local/share/1bit systems/anvil.state"
CHANNEL_ID="1488836467039010936"   # #changelog
MCP_WRAPPER="$HOME/repos/1bit systems-core/discord-mcp/bin/halo-discord.sh"

mkdir -p "$(dirname "$STATE_FILE")"

post_anvil() {
    # $1 = plain-text body (will be JSON-escaped with jq)
    local content=$1
    local escaped
    escaped=$(jq -Rs . <<<"$content")
    local rpc
    rpc=$(cat <<JSON
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"discord_post_as_specialist","arguments":{"channel_id":"$CHANNEL_ID","specialist":"anvil","content":$escaped}}}
JSON
)
    timeout 10 bash "$MCP_WRAPPER" <<<"$rpc" >/dev/null 2>&1 || {
        echo "[anvil] discord post failed" >&2
    }
}

cd "$REPO"
git fetch origin main --quiet

remote_head=$(git rev-parse origin/main)
local_head=$(git rev-parse HEAD)
last_benched=$(cat "$STATE_FILE" 2>/dev/null || echo 'none')

if [[ "$remote_head" == "$last_benched" ]]; then
    exit 0   # nothing new; silent no-op
fi

# Fast-forward local checkout (never force — abort if user has local divergent work)
if ! git merge --ff-only "$remote_head" >/dev/null 2>&1; then
    echo "[anvil] local diverged from origin/main; skipping build" >&2
    exit 1
fi

# Build
if ! (cd build && cmake --build . --target bitnet_decode -j "$(nproc)" >/tmp/anvil-build.log 2>&1); then
    tail -40 /tmp/anvil-build.log >&2
    tail_excerpt=$(tail -20 /tmp/anvil-build.log)
    post_anvil "🚨 build failed at \`${remote_head:0:7}\`

$tail_excerpt"
    exit 1
fi

# Bench — quick sweep across 4 context sizes
MODEL="$HOME/1bit systems/models/halo-1bit-2b.h1b"
TMPFILE="$(mktemp)"
trap 'rm -f "$TMPFILE"' EXIT

for N in 64 256 512 1024; do
    RAW=$(mktemp)
    if ! "$REPO/build/bitnet_decode" "$MODEL" --text "Explain nuclear fusion in detail." "$N" >"$RAW" 2>&1; then
        tail_excerpt=$(tail -20 "$RAW")
        rm -f "$RAW"
        post_anvil "🚨 bitnet_decode crashed during bench (N=$N) at \`${remote_head:0:7}\`

$tail_excerpt"
        exit 1
    fi
    out=$(grep 'decode.*tok/s' "$RAW" || echo '(no tok/s line matched)')
    rm -f "$RAW"
    printf 'N=%5d  %s\n' "$N" "$out" >> "$TMPFILE"
done

commit_msg=$(git log -1 --format='%s' "$remote_head")
short_sha=${remote_head:0:7}

body="**✅ built + benched \`$short_sha\`** — $commit_msg

$(cat "$TMPFILE")"

post_anvil "$body"

# Save bench text for archival
cp "$TMPFILE" "$BENCH_LOG_DIR/anvil-bench-${short_sha}-$(date -u +%Y%m%dT%H%M%SZ).txt"

# Record new state
echo "$remote_head" > "$STATE_FILE"
