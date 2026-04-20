#!/usr/bin/env bash
# 1bit-stampby-nuke.sh — tear down the stampby GitHub account's repos.
#
# Reads the CSV produced on 2026-04-20 and calls `gh repo delete stampby/<name>`
# for every row tagged action=delete. DRY-RUN by default; pass --apply to
# actually delete. Pass --force to skip the per-repo y/N prompt.
#
# Usage:
#   1bit-stampby-nuke.sh [--csv PATH] [--apply] [--force] [--log PATH]
#
# Safety:
#   - Requires `gh auth status` to be logged in as stampby with `delete_repo`
#     scope. Run `gh auth refresh -h github.com -s delete_repo` first.
#   - Dry-run prints every action without touching GitHub.
#   - Only operates on repos under the `stampby/` owner — the owner is
#     hardcoded in the gh command line.
set -euo pipefail

CSV_DEFAULT="/home/bcloud/claude output/stampby-repos-2026-04-20.csv"
LOG_DEFAULT="/home/bcloud/claude output/stampby-nuke-$(date -u +%Y-%m-%dT%H%M%SZ).log"

CSV="$CSV_DEFAULT"
LOG="$LOG_DEFAULT"
APPLY=0
FORCE=0
OWNER="stampby"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv)   CSV="$2"; shift 2 ;;
    --log)   LOG="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$CSV" ]]; then
  echo "CSV not found: $CSV" >&2
  exit 1
fi

if ! command -v gh >/dev/null; then
  echo "gh CLI not found on PATH" >&2
  exit 1
fi

# Verify we're logged in as the expected owner.
if [[ "$APPLY" -eq 1 ]]; then
  ACTIVE_USER=$(gh api user --jq .login 2>/dev/null || true)
  if [[ "$ACTIVE_USER" != "$OWNER" ]]; then
    echo "refusing to --apply: gh is authenticated as '$ACTIVE_USER', not '$OWNER'" >&2
    echo "fix with: gh auth switch -u $OWNER" >&2
    exit 3
  fi
fi

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

log() {
  local line="$*"
  echo "$line"
  echo "$line" >> "$LOG"
}

log "halo-stampby-nuke starting $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "csv=$CSV  apply=$APPLY  force=$FORCE  log=$LOG"
log ""

# Skip header row with tail -n +2. Parse naive CSV (no embedded commas in our data).
total=0
deleted=0
skipped=0
errored=0

# Read rows manually so we can prompt on stdin even when awk is between us.
# shellcheck disable=SC2034
while IFS=',' read -r c_name c_isFork c_parent c_isPrivate c_isArchived c_pushedAt c_action; do
  # Strip CSV quotes from each field.
  unquote() { printf '%s' "${1%\"}" | sed 's/^"//'; }
  name=$(unquote "$c_name")
  parent=$(unquote "$c_parent")
  pushed=$(unquote "$c_pushedAt")
  action=$(unquote "$c_action")

  # Header / empty guard.
  [[ -z "$name" || "$name" == "name" ]] && continue
  total=$((total + 1))

  if [[ "$action" != "delete" ]]; then
    log "SKIP  stampby/$name  action=$action"
    skipped=$((skipped + 1))
    continue
  fi

  url="https://github.com/$OWNER/$name"
  log "---"
  log "TARGET  $url"
  log "        parent=$parent  pushedAt=$pushed"

  if [[ "$APPLY" -eq 0 ]]; then
    log "DRY-RUN would run: gh repo delete $OWNER/$name --yes"
    continue
  fi

  if [[ "$FORCE" -eq 0 ]]; then
    # Prompt on /dev/tty so it works even when stdin is piped.
    printf "delete %s? [y/N] " "$url" > /dev/tty
    read -r reply < /dev/tty || reply=""
    case "$reply" in
      y|Y|yes|YES) ;;
      *) log "USER SKIP  $url"; skipped=$((skipped + 1)); continue ;;
    esac
  fi

  if gh repo delete "$OWNER/$name" --yes >>"$LOG" 2>&1; then
    log "DELETED  $url"
    deleted=$((deleted + 1))
  else
    log "ERROR    $url  (see log above)"
    errored=$((errored + 1))
  fi
done < <(tail -n +2 "$CSV")

log ""
log "summary: total=$total  deleted=$deleted  skipped=$skipped  errored=$errored  apply=$APPLY"
log "log file: $LOG"

if [[ "$APPLY" -eq 0 ]]; then
  log "DRY-RUN complete. Re-run with --apply to execute."
fi
