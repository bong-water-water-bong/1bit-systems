#!/usr/bin/env bash
# halo-librarian — CHANGELOG appender.
# Walks git log across halo-ai-core + rocm-cpp + agent-cpp + halo-1bit,
# extracts conventional-commit-formatted lines since last run, appends
# to halo-ai-core/CHANGELOG.md. Posts summary to Discord as 📚 librarian.

set -euo pipefail

STATE_FILE="$HOME/.local/share/halo-ai/librarian.state"
CHANGELOG="$HOME/halo-ai-core/CHANGELOG.md"
CHANNEL_ID="1488836467039010936"   # #changelog
MCP_WRAPPER="$HOME/repos/halo-ai-core/discord-mcp/bin/halo-discord.sh"

REPOS=(
    "$HOME/halo-ai-core:halo-ai-core"
    "$HOME/repos/rocm-cpp:rocm-cpp"
)
# agent-cpp + halo-1bit added when their trees exist locally
for candidate in "$HOME/repos/agent-cpp:agent-cpp" "$HOME/repos/halo-1bit:halo-1bit"; do
    [[ -d "${candidate%%:*}/.git" ]] && REPOS+=("$candidate")
done

mkdir -p "$(dirname "$STATE_FILE")"
touch "$STATE_FILE"

post_librarian() {
    local content=$1
    local escaped
    escaped=$(jq -Rs . <<<"$content")
    local rpc
    rpc=$(cat <<JSON
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"discord_post_as_specialist","arguments":{"channel_id":"$CHANNEL_ID","specialist":"librarian","content":$escaped}}}
JSON
)
    timeout 10 bash "$MCP_WRAPPER" <<<"$rpc" >/dev/null 2>&1 || {
        echo "[librarian] discord post failed" >&2
    }
}

added_lines=()

for entry in "${REPOS[@]}"; do
    path="${entry%%:*}"
    name="${entry##*:}"
    state_key="librarian_${name//[-\/]/_}"
    last=$(grep "^${state_key}=" "$STATE_FILE" | head -1 | cut -d= -f2 || echo '')

    head=$(git -C "$path" rev-parse HEAD)
    if [[ "$head" == "$last" ]]; then
        continue
    fi

    range="${last}..${head}"
    [[ -z "$last" ]] && range="HEAD~20..${head}"   # first run: last 20 commits

    # Conventional-commit prefixes we care about.
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        added_lines+=("- **${name}**: $line")
    done < <(git -C "$path" log "$range" --format='%s' 2>/dev/null \
              | grep -E '^(feat|fix|perf|docs|refactor|build|ci|chore)(\([^)]+\))?: ' || true)

    # update state — fail loudly if we can't write the tmp file. grep -v
    # exit 1 is fine (means no prior entry), but an I/O error must not be
    # swallowed, or the mv clobbers state with an empty file.
    { grep -v "^${state_key}=" "$STATE_FILE" || true; } > "${STATE_FILE}.tmp"
    echo "${state_key}=${head}" >> "${STATE_FILE}.tmp"
    if ! [[ -s "${STATE_FILE}.tmp" ]]; then
        echo "[librarian] state tmp write produced empty file, aborting" >&2
        rm -f "${STATE_FILE}.tmp"
        exit 1
    fi
    mv "${STATE_FILE}.tmp" "$STATE_FILE"
done

if [[ ${#added_lines[@]} -eq 0 ]]; then
    exit 0
fi

# Prepend today's bucket to CHANGELOG.md
today=$(date -u +%Y-%m-%d)
bucket=$(printf '%s\n' "${added_lines[@]}")

if [[ ! -f "$CHANGELOG" ]]; then
    cat > "$CHANGELOG" <<EOF
# CHANGELOG — halo-ai

Auto-appended by halo-librarian. Walks commit history of halo-ai-core,
rocm-cpp, agent-cpp, halo-1bit for conventional-commit prefixes
(feat / fix / perf / docs / refactor / build / ci / chore).

---

EOF
fi

# Insert new bucket after the "---" separator.
tmp=$(mktemp)
awk -v today="$today" -v bucket="$bucket" '
    /^---$/ && !inserted {
        print
        print ""
        print "## " today
        print ""
        print bucket
        print ""
        inserted=1
        next
    }
    { print }
' "$CHANGELOG" > "$tmp"
mv "$tmp" "$CHANGELOG"

summary_body="updated \`CHANGELOG.md\` with ${#added_lines[@]} new entries.

$bucket"

post_librarian "$summary_body"
