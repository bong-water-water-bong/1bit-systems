#!/usr/bin/env bash
# halo-quartermaster — GitHub issue triage. Walks open issues across the four
# 1bit systems repos; for any issue with zero labels, applies "needs-triage".
# Posts a summary to #changelog as 📦 quartermaster when new triage happens.

set -euo pipefail

CHANNEL_ID="1488836467039010936"   # #changelog
MCP_WRAPPER="$HOME/repos/1bit systems-core/discord-mcp/bin/halo-discord.sh"
STATE_FILE="$HOME/.local/share/1bit systems/quartermaster.state"
REPOS=(bong-water-water-bong/rocm-cpp bong-water-water-bong/1bit systems-core bong-water-water-bong/agent-cpp bong-water-water-bong/halo-1bit)

mkdir -p "$(dirname "$STATE_FILE")"
touch "$STATE_FILE"

post_qm() {
    local content=$1
    local escaped
    escaped=$(jq -Rs . <<<"$content")
    local rpc
    rpc=$(cat <<JSON
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"discord_post_as_specialist","arguments":{"channel_id":"$CHANNEL_ID","specialist":"quartermaster","content":$escaped}}}
JSON
)
    timeout 10 bash "$MCP_WRAPPER" <<<"$rpc" >/dev/null 2>&1 || echo "[qm] discord post failed" >&2
}

triaged=()

for repo in "${REPOS[@]}"; do
    # gh api returns [] for empty. Filter issues only (not PRs — PRs have pull_request key).
    while IFS=$'\t' read -r num title labels_count; do
        [[ -z "$num" ]] && continue
        if [[ "$labels_count" == "0" ]]; then
            state_key="qm_${repo//[\/-]/_}_${num}"
            if grep -q "^${state_key}=" "$STATE_FILE"; then
                continue
            fi
            if ! gh issue edit "$num" -R "$repo" --add-label "needs-triage" >/dev/null 2>&1; then
                # Label may not exist; create + retry. If the retry also
                # fails we MUST NOT record state, or the issue silently
                # skips triage forever.
                gh label create needs-triage -R "$repo" --color FBCA04 \
                    --description "awaiting quartermaster review" >/dev/null 2>&1 || true
                if ! gh issue edit "$num" -R "$repo" --add-label "needs-triage" >/dev/null 2>&1; then
                    echo "[qm] failed to label ${repo}#${num}, leaving for retry" >&2
                    continue
                fi
            fi
            triaged+=("- \`${repo}#${num}\` — ${title}")
            echo "${state_key}=$(date -u +%FT%TZ)" >> "$STATE_FILE"
        fi
    done < <(gh api "repos/$repo/issues?state=open" \
             --jq '.[] | select(.pull_request == null) | [.number, .title, (.labels | length)] | @tsv' 2>/dev/null)
done

if [[ ${#triaged[@]} -eq 0 ]]; then
    exit 0
fi

body="labelled \`needs-triage\` on ${#triaged[@]} issue(s):
$(printf '%s\n' "${triaged[@]}")"

post_qm "$body"
