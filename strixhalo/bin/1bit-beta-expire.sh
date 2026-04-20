#!/usr/bin/env bash
# 1bit-beta-expire.sh — 10-day TTL enforcer for the 1bit systems private beta.
#
# Policy (user directive 2026-04-20): the demo runs for 10 days, then the
# beta artifacts purge themselves. That means:
#
#   1. Every bearer in /etc/caddy/bearers.txt with an `expires <ISO>` field
#      (or, fallback, `issued <ISO>` + 10 days) that has passed gets
#      revoked via 1bit-mesh-revoke.sh <handle>.
#   2. Purged bearer lines are archived to
#      /var/log/halo-beta/expired-<YYYY-MM-DD>.log (root:root, 0600)
#      before they leave bearers.txt, so we keep an auditable trail.
#   3. Claude memory files under
#      ~/.claude/projects/-home-bcloud/memory/ that are *explicitly* marked
#      as beta-era get purged too — either by an `expires: <ISO-date>`
#      YAML frontmatter field in the past, OR by a "[beta-ttl]" body tag
#      AND mtime > 10 days. A short list of load-bearing project notes is
#      hard-allow-listed and NEVER auto-purges.
#
# Default is DRY RUN. Pass --apply to actually revoke/delete.
#
# Exit code: 0 on success, 1 on partial failure (anything printed to stderr
# that looks like a real error).
#
# Idempotent: re-running is safe. Already-revoked handles are a no-op; the
# memory-file scan only flags files that still exist and still match.
#
# Cross-refs:
#   - docs/wiki/Beta-10-Day-TTL.md  (policy rationale)
#   - docs/wiki/VPN-Only-API.md     (why the mesh + bearer fences exist)
#   - strixhalo/bin/1bit-mesh-revoke.sh (reused as-is)
#   - strixhalo/bin/1bit-mesh-invite.sh (issues the `expires` field)
#   - strixhalo/systemd/halo-beta-expire.{service,timer}

set -uo pipefail

# -------- config ---------------------------------------------------------

BEARER_FILE="${HALO_BETA_BEARER_FILE:-/etc/caddy/bearers.txt}"
AUDIT_DIR="${HALO_BETA_AUDIT_DIR:-/var/log/halo-beta}"
MEMORY_DIR="${HALO_BETA_MEMORY_DIR:-/home/bcloud/.claude/projects/-home-bcloud/memory}"
REVOKE_CMD="${HALO_BETA_REVOKE_CMD:-$(dirname "$(readlink -f "$0")")/1bit-mesh-revoke.sh}"
TTL_DAYS="${HALO_BETA_TTL_DAYS:-10}"

# Allow-list: load-bearing project notes that must NEVER auto-purge, even
# if someone accidentally drops a `[beta-ttl]` tag into them. Filename
# basenames (without path, without extension).
#
# Keep in sync with the wiki doc (Beta-10-Day-TTL.md) when you add/remove.
readonly -a ALLOWLIST=(
    project_reddit_relaunch
    project_hermes_integration
    project_lemonade_10_2_pivot
    project_strix_halo_hardware
    project_voice_latency_sharding
    project_apu_thesis
)

APPLY=0
VERBOSE=0

# -------- args -----------------------------------------------------------

for arg in "$@"; do
    case "$arg" in
        --apply) APPLY=1 ;;
        -v|--verbose) VERBOSE=1 ;;
        -h|--help)
            sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown arg: $arg" >&2
            echo "usage: $0 [--apply] [-v]" >&2
            exit 2
            ;;
    esac
done

# -------- helpers --------------------------------------------------------

now_epoch() { date -u +%s; }

# Convert an ISO8601 timestamp to epoch seconds. Accepts both "Z" and
# "+00:00" suffixes; accepts both "YYYY-MM-DDTHH:MMZ" and "YYYY-MM-DD".
# Prints the epoch to stdout, or nothing on parse failure.
iso_to_epoch() {
    local iso="${1:-}"
    [[ -z "$iso" ]] && return 0
    # date -d is GNU-date; on strixhalo (CachyOS) that's fine.
    date -u -d "$iso" +%s 2>/dev/null || true
}

# parse_bearer_expiry <line> → prints epoch seconds of expiry, or empty.
#
# Expected line format:
#   sk-halo-XXXX  # handle  # issued 2026-04-20T14:00Z  # expires 2026-04-30T14:00Z
#
# Preference: `expires <ISO>` wins. If absent, fall back to
# `issued <ISO>` + TTL_DAYS days. If neither is present, prints empty
# (caller decides; we treat empty as "unknown, do not touch").
parse_bearer_expiry() {
    local line="$1"
    local expires issued
    expires=$(printf '%s' "$line" | grep -oE 'expires[[:space:]]+[0-9TZ:\-]+' | awk '{print $2}' | head -1)
    if [[ -n "$expires" ]]; then
        iso_to_epoch "$expires"
        return 0
    fi
    issued=$(printf '%s' "$line" | grep -oE 'issued[[:space:]]+[0-9TZ:\-]+' | awk '{print $2}' | head -1)
    if [[ -n "$issued" ]]; then
        local issued_ep
        issued_ep=$(iso_to_epoch "$issued")
        [[ -n "$issued_ep" ]] && printf '%s' $(( issued_ep + TTL_DAYS * 86400 ))
    fi
}

# parse_bearer_handle <line> → prints the first `#`-comment word (the handle).
parse_bearer_handle() {
    local line="$1"
    # Layout: token  # handle  # issued ...  # expires ...
    # Grab the substring between the first '#' and the second '#'.
    printf '%s' "$line" | awk -F'#' '{gsub(/^[[:space:]]+|[[:space:]]+$/,"",$2); print $2}'
}

# in_allowlist <basename-without-ext>
in_allowlist() {
    local needle="$1"
    local e
    for e in "${ALLOWLIST[@]}"; do
        [[ "$e" == "$needle" ]] && return 0
    done
    return 1
}

log() { printf '[halo-beta-expire] %s\n' "$*"; }
vlog() { (( VERBOSE )) && printf '[halo-beta-expire] %s\n' "$*" || true; }
err() { printf '[halo-beta-expire] ERROR: %s\n' "$*" >&2; }

# -------- bearer sweep ---------------------------------------------------

EXIT_CODE=0

sweep_bearers() {
    if [[ ! -f "$BEARER_FILE" ]]; then
        log "bearer file $BEARER_FILE not present — skipping bearer sweep."
        return 0
    fi

    local now
    now=$(now_epoch)
    local -a expired_handles=()
    local -a expired_lines=()

    # Read root-owned file; we may need sudo in production but in tests
    # the file is user-readable.
    local reader="cat"
    if [[ ! -r "$BEARER_FILE" ]]; then
        reader="sudo cat"
    fi

    # Iterate non-comment, non-blank lines.
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue

        local expiry_ep handle
        expiry_ep=$(parse_bearer_expiry "$line")
        handle=$(parse_bearer_handle "$line")

        if [[ -z "$expiry_ep" ]]; then
            vlog "no expiry + no issued on line (handle='$handle'); leaving alone"
            continue
        fi
        if (( expiry_ep > now )); then
            vlog "handle='$handle' still valid for $(( (expiry_ep - now) / 3600 ))h"
            continue
        fi

        expired_handles+=("$handle")
        expired_lines+=("$line")
    done < <($reader "$BEARER_FILE")

    if (( ${#expired_handles[@]} == 0 )); then
        log "no bearers past TTL."
        return 0
    fi

    log "found ${#expired_handles[@]} expired bearer(s):"
    local i
    for (( i=0; i<${#expired_handles[@]}; i++ )); do
        printf '  - %s\n' "${expired_handles[$i]}"
    done

    if (( ! APPLY )); then
        log "dry-run: would revoke the above handles and archive their lines."
        return 0
    fi

    # Archive the lines we're about to drop. Root-owned, 0600, audit only.
    local audit_file="$AUDIT_DIR/expired-$(date -u +%Y-%m-%d).log"
    if command -v sudo >/dev/null 2>&1; then
        sudo install -d -o root -g root -m 0700 "$AUDIT_DIR" 2>/dev/null || \
            mkdir -p "$AUDIT_DIR"
    else
        mkdir -p "$AUDIT_DIR"
    fi

    {
        printf '# halo-beta-expire audit — %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        for line in "${expired_lines[@]}"; do
            printf '%s\n' "$line"
        done
    } | {
        if command -v sudo >/dev/null 2>&1 && [[ ! -w "$AUDIT_DIR" ]]; then
            sudo tee -a "$audit_file" >/dev/null
        else
            tee -a "$audit_file" >/dev/null
        fi
    }
    if command -v sudo >/dev/null 2>&1; then
        sudo chown root:root "$audit_file" 2>/dev/null || true
        sudo chmod 0600 "$audit_file" 2>/dev/null || true
    else
        chmod 0600 "$audit_file" 2>/dev/null || true
    fi
    log "archived $(( ${#expired_lines[@]} )) line(s) to $audit_file"

    # Revoke each handle. 1bit-mesh-revoke.sh drops the bearer + expires
    # the authkey + reloads Caddy; idempotent.
    if [[ ! -x "$REVOKE_CMD" ]]; then
        err "revoke command not executable: $REVOKE_CMD"
        EXIT_CODE=1
        return 0
    fi
    for handle in "${expired_handles[@]}"; do
        if [[ -z "$handle" ]]; then
            err "blank handle on an expired line; skipping"
            EXIT_CODE=1
            continue
        fi
        log "revoking $handle..."
        if ! "$REVOKE_CMD" "$handle"; then
            err "revoke failed for '$handle' (will retry on next run)"
            EXIT_CODE=1
        fi
    done
}

# -------- memory-file sweep ---------------------------------------------

sweep_memory() {
    if [[ ! -d "$MEMORY_DIR" ]]; then
        log "memory dir $MEMORY_DIR not present — skipping memory sweep."
        return 0
    fi

    local now
    now=$(now_epoch)
    local cutoff=$(( now - TTL_DAYS * 86400 ))
    local -a to_purge=()

    # Walk .md files one level deep (ignore subdirs like _from_ryzen).
    local f
    while IFS= read -r -d '' f; do
        local base basename_noext
        base=$(basename "$f")
        basename_noext="${base%.md}"

        # Skip the top-level MEMORY.md index outright.
        [[ "$base" == "MEMORY.md" ]] && continue

        # Read only the frontmatter block (between first two `---` lines)
        # plus keep the body for the [beta-ttl] tag scan. Cheap; files are
        # small (a few KB).
        local frontmatter type_field expires_field
        frontmatter=$(awk 'NR==1 && /^---/ {p=1; next} p && /^---/ {exit} p' "$f" 2>/dev/null)
        type_field=$(printf '%s\n' "$frontmatter" | awk -F': *' 'tolower($1)=="type"{print $2; exit}')
        expires_field=$(printf '%s\n' "$frontmatter" | awk -F': *' 'tolower($1)=="expires"{print $2; exit}')

        # Only `type: project` files are candidates.
        [[ "$type_field" != "project" ]] && continue

        # Allow-list short-circuits everything.
        if in_allowlist "$basename_noext"; then
            vlog "allow-listed (load-bearing): $base"
            continue
        fi

        local expired=0

        # Path 1 — explicit expires frontmatter field in the past.
        if [[ -n "$expires_field" ]]; then
            local exp_ep
            exp_ep=$(iso_to_epoch "$expires_field")
            if [[ -n "$exp_ep" ]] && (( exp_ep <= now )); then
                expired=1
            fi
        fi

        # Path 2 — body has [beta-ttl] tag AND file mtime older than TTL.
        if (( ! expired )); then
            if grep -qF '[beta-ttl]' "$f" 2>/dev/null; then
                local mt
                mt=$(stat -c %Y "$f" 2>/dev/null || echo 0)
                if (( mt > 0 && mt <= cutoff )); then
                    expired=1
                fi
            fi
        fi

        (( expired )) && to_purge+=("$f")
    done < <(find "$MEMORY_DIR" -maxdepth 1 -type f -name '*.md' -print0)

    if (( ${#to_purge[@]} == 0 )); then
        log "no memory files past TTL."
        return 0
    fi

    log "found ${#to_purge[@]} memory file(s) past TTL:"
    local p
    for p in "${to_purge[@]}"; do
        printf '  - %s\n' "$p"
    done

    if (( ! APPLY )); then
        log "dry-run: would delete the above memory file(s). Re-run with --apply."
        return 0
    fi

    for p in "${to_purge[@]}"; do
        if rm -f "$p"; then
            log "purged $p"
        else
            err "failed to delete $p"
            EXIT_CODE=1
        fi
    done
}

# -------- main -----------------------------------------------------------

main() {
    if (( APPLY )); then
        log "mode: APPLY (will revoke + delete)"
    else
        log "mode: DRY RUN (no changes — pass --apply to commit)"
    fi
    log "ttl_days=$TTL_DAYS bearer_file=$BEARER_FILE memory_dir=$MEMORY_DIR"
    sweep_bearers
    sweep_memory
    exit "$EXIT_CODE"
}

# Skip `main` when sourced (test harness sources this file to call the
# helper functions directly).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
