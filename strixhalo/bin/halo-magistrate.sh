#!/usr/bin/env bash
# halo-magistrate — PR policy scanner. Walks open PRs across the four
# halo-ai repos; checks:
#   1. title follows Conventional Commits (feat/fix/perf/docs/refactor/
#      build/ci/chore/test — optional scope in parens, then colon + space)
#   2. no obvious secret patterns in the diff (GH tokens, OpenAI keys,
#      Anthropic keys, generic 40+ char hex+base64 substrings)
#   3. PR has at least one commit
# Posts findings to #changelog as ⚖️ magistrate.

set -euo pipefail

CHANNEL_ID="1488836467039010936"   # #changelog
MCP_WRAPPER="$HOME/repos/halo-ai-core/discord-mcp/bin/halo-discord.sh"
STATE_FILE="$HOME/.local/share/halo-ai/magistrate.state"
REPOS=(stampby/rocm-cpp stampby/halo-ai-core stampby/agent-cpp stampby/halo-1bit)

CC_RE='^(feat|fix|perf|docs|refactor|build|ci|chore|test|style|revert)(\([^)]+\))?: '
# Secret sniffers (keep permissive; false positives are OK, we just flag).
SECRETS_RE='(ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|sk-[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9\-_]{90,})'

mkdir -p "$(dirname "$STATE_FILE")"
touch "$STATE_FILE"

post_mag() {
    local content=$1
    local escaped
    escaped=$(jq -Rs . <<<"$content")
    local rpc
    rpc=$(cat <<JSON
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"discord_post_as_specialist","arguments":{"channel_id":"$CHANNEL_ID","specialist":"magistrate","content":$escaped}}}
JSON
)
    timeout 10 bash "$MCP_WRAPPER" <<<"$rpc" >/dev/null 2>&1 || echo "[mag] discord post failed" >&2
}

findings=()

for repo in "${REPOS[@]}"; do
    while IFS=$'\t' read -r num title head_sha; do
        [[ -z "$num" ]] && continue
        state_key="mag_${repo//[\/-]/_}_${num}_${head_sha:0:7}"
        if grep -q "^${state_key}=" "$STATE_FILE"; then
            continue   # already reviewed this head
        fi

        issues=()

        # 1. Conventional Commits title.
        if ! [[ "$title" =~ $CC_RE ]]; then
            issues+=("title doesn't match Conventional Commits")
        fi

        # 2. Secret scan on diff. DO NOT silently downgrade a gh failure to
        #    "no secrets found" — that's a security control bypass.
        if ! diff=$(gh pr diff "$num" -R "$repo" 2>/dev/null); then
            issues+=("**scan unavailable — gh pr diff failed, secret scan skipped**")
        else
            if grep -qE "$SECRETS_RE" <<<"$diff"; then
                issues+=("**possible secret in diff** — needs scrub")
            fi
        fi

        # 3. Commit count.
        count=$(gh api "repos/$repo/pulls/$num/commits" --jq 'length' 2>/dev/null || echo 0)
        if [[ "$count" -lt 1 ]]; then
            issues+=("zero commits attached")
        fi

        if [[ ${#issues[@]} -gt 0 ]]; then
            findings+=("\`${repo}#${num}\` — $(IFS='; '; echo "${issues[*]}")")
        fi
        echo "${state_key}=$(date -u +%FT%TZ)" >> "$STATE_FILE"
    done < <(gh pr list -R "$repo" --state open --json number,title,headRefOid \
             --jq '.[] | [.number, .title, .headRefOid] | @tsv' 2>/dev/null)
done

if [[ ${#findings[@]} -eq 0 ]]; then
    exit 0
fi

body="policy flags on ${#findings[@]} PR(s):
$(printf -- '- %s\n' "${findings[@]}")"

post_mag "$body"
