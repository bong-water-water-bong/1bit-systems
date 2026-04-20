#!/usr/bin/env bash
# halo-beta-expire.test.sh — plain-bash unit tests for 1bit-beta-expire.sh.
#
# `bats` is not installed on strixhalo as of 2026-04-20, so we roll a
# minimal assert-style runner instead of adding a dependency. One .sh
# file, no new packages, self-contained. If bats lands later we can
# port; the test structure deliberately mirrors bats' @test blocks.
#
# Run: bash strixhalo/bin/tests/halo-beta-expire.test.sh
#
# Exit 0 = all passed. Exit 1 = one or more failed.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/../1bit-beta-expire.sh"

if [[ ! -f "$SCRIPT" ]]; then
    echo "FATAL: $SCRIPT not found"
    exit 2
fi

# ---- runner state -------------------------------------------------------

TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_NAMES=()

_expect() {
    # _expect <description> <actual> <expected-substring-or-equals>
    local desc="$1"; local actual="$2"; local expected="$3"
    if [[ "$actual" == *"$expected"* ]]; then
        printf '  ok   — %s\n' "$desc"
        TESTS_PASSED=$(( TESTS_PASSED + 1 ))
    else
        printf '  FAIL — %s\n    expected (substring): %s\n    actual: %s\n' \
            "$desc" "$expected" "$actual"
        TESTS_FAILED=$(( TESTS_FAILED + 1 ))
        FAILED_NAMES+=("$desc")
    fi
}

_expect_eq() {
    local desc="$1"; local actual="$2"; local expected="$3"
    if [[ "$actual" == "$expected" ]]; then
        printf '  ok   — %s\n' "$desc"
        TESTS_PASSED=$(( TESTS_PASSED + 1 ))
    else
        printf '  FAIL — %s\n    expected: %s\n    actual:   %s\n' \
            "$desc" "$expected" "$actual"
        TESTS_FAILED=$(( TESTS_FAILED + 1 ))
        FAILED_NAMES+=("$desc")
    fi
}

_test() {
    TESTS_RUN=$(( TESTS_RUN + 1 ))
    printf '\n[test] %s\n' "$1"
}

# ---- fixtures -----------------------------------------------------------

# Per-test temp sandbox. Clean up via trap.
make_sandbox() {
    local d
    d=$(mktemp -d)
    mkdir -p "$d/caddy" "$d/audit" "$d/memory" "$d/bin"

    # Mock 1bit-mesh-revoke.sh — a no-op that records who it was called for.
    cat > "$d/bin/1bit-mesh-revoke.sh" <<'EOF'
#!/usr/bin/env bash
# test mock — just log the handle to $MOCK_REVOKE_LOG
printf '%s\n' "$1" >> "$MOCK_REVOKE_LOG"
exit 0
EOF
    chmod 0755 "$d/bin/1bit-mesh-revoke.sh"

    printf '%s' "$d"
}

# ---- helpers --------------------------------------------------------

iso_past()    { date -u -d "$1 days ago"     +%Y-%m-%dT%H:%MZ; }
iso_future()  { date -u -d "$1 days"         +%Y-%m-%dT%H:%MZ; }
iso_daysago() { date -u -d "@$(( $(date -u +%s) - $1 * 86400 ))" +%Y-%m-%dT%H:%MZ; }

# ---- tests --------------------------------------------------------------

test_parse_bearer_expiry_reads_expires_field() {
    _test "parse_bearer_expiry reads the expires field"
    # shellcheck disable=SC1090
    source "$SCRIPT"
    local exp_iso exp_ep expected
    exp_iso=$(iso_future 5)
    expected=$(date -u -d "$exp_iso" +%s)
    exp_ep=$(parse_bearer_expiry "sk-halo-abc  # alice  # issued 2026-04-20T14:00Z  # expires $exp_iso")
    _expect_eq "expires field parsed as epoch" "$exp_ep" "$expected"
}

test_parse_bearer_expiry_falls_back_to_issued_plus_ttl() {
    _test "parse_bearer_expiry falls back to issued + 10d when expires missing"
    # shellcheck disable=SC1090
    source "$SCRIPT"
    local issued_iso issued_ep expected_ep exp_ep
    issued_iso=$(iso_daysago 3)
    issued_ep=$(date -u -d "$issued_iso" +%s)
    expected_ep=$(( issued_ep + 10 * 86400 ))
    exp_ep=$(parse_bearer_expiry "sk-halo-xyz  # bob  # issued $issued_iso")
    _expect_eq "issued+10d fallback" "$exp_ep" "$expected_ep"
}

test_parse_bearer_expiry_empty_when_no_fields() {
    _test "parse_bearer_expiry prints empty when neither field present"
    # shellcheck disable=SC1090
    source "$SCRIPT"
    local exp_ep
    exp_ep=$(parse_bearer_expiry "sk-halo-none  # ghost")
    _expect_eq "no fields → empty" "$exp_ep" ""
}

test_parse_bearer_handle() {
    _test "parse_bearer_handle extracts handle between first two # separators"
    # shellcheck disable=SC1090
    source "$SCRIPT"
    local h
    h=$(parse_bearer_handle "sk-halo-abc  # alice  # issued 2026-04-20T14:00Z  # expires 2026-04-30T14:00Z")
    _expect_eq "handle extracted" "$h" "alice"
}

test_dry_run_reports_expired_without_touching() {
    _test "dry-run reports pending expirations and touches nothing"
    local sandbox; sandbox=$(make_sandbox)
    local bearers="$sandbox/caddy/bearers.txt"

    local expired_iso valid_iso
    expired_iso=$(iso_past 3)    # 3 days ago
    valid_iso=$(iso_future 5)    # 5 days out

    {
        printf '# top-of-file comment, should be ignored\n'
        printf 'sk-halo-111  # alice  # issued 2026-04-01T00:00Z  # expires %s\n' "$expired_iso"
        printf 'sk-halo-222  # bob    # issued 2026-04-18T00:00Z  # expires %s\n' "$valid_iso"
    } > "$bearers"

    local mock_log="$sandbox/revoke.log"
    : > "$mock_log"

    # Copy the script into the sandbox bin dir so REVOKE_CMD resolves to
    # our mock (revoke is discovered by `dirname $0`).
    cp "$SCRIPT" "$sandbox/bin/1bit-beta-expire.sh"
    chmod 0755 "$sandbox/bin/1bit-beta-expire.sh"

    local out
    out=$(HALO_BETA_BEARER_FILE="$bearers" \
          HALO_BETA_AUDIT_DIR="$sandbox/audit" \
          HALO_BETA_MEMORY_DIR="$sandbox/memory" \
          MOCK_REVOKE_LOG="$mock_log" \
          bash "$sandbox/bin/1bit-beta-expire.sh" 2>&1)

    _expect "dry-run mentions alice as expired" "$out" "alice"
    _expect "dry-run announces DRY RUN mode" "$out" "DRY RUN"
    _expect "dry-run does NOT mention bob" "$out" \
        "$(printf 'found 1 expired bearer')"

    # bearers.txt must be untouched.
    local bobline
    bobline=$(grep -c '^sk-halo-222' "$bearers" 2>/dev/null || printf 0)
    _expect_eq "bearers.txt bob line still present" "$bobline" "1"

    local aliceline
    aliceline=$(grep -c '^sk-halo-111' "$bearers" 2>/dev/null || printf 0)
    _expect_eq "bearers.txt alice line still present (dry-run)" "$aliceline" "1"

    # Mock must not have been called.
    local revoked
    revoked=$(wc -l < "$mock_log" 2>/dev/null || printf 0)
    _expect_eq "mock revoke was NOT called in dry-run" "$revoked" "0"

    rm -rf "$sandbox"
}

test_apply_calls_mock_revoke() {
    _test "--apply calls 1bit-mesh-revoke.sh (mocked) for expired handles"
    local sandbox; sandbox=$(make_sandbox)
    local bearers="$sandbox/caddy/bearers.txt"

    local expired_iso valid_iso
    expired_iso=$(iso_past 2)
    valid_iso=$(iso_future 7)

    {
        printf 'sk-halo-aaa  # alice    # issued 2026-04-01T00:00Z  # expires %s\n' "$expired_iso"
        printf 'sk-halo-bbb  # bob      # issued 2026-04-18T00:00Z  # expires %s\n' "$valid_iso"
        printf 'sk-halo-ccc  # charlie  # issued 2026-04-01T00:00Z  # expires %s\n' "$expired_iso"
    } > "$bearers"

    local mock_log="$sandbox/revoke.log"
    : > "$mock_log"

    cp "$SCRIPT" "$sandbox/bin/1bit-beta-expire.sh"
    chmod 0755 "$sandbox/bin/1bit-beta-expire.sh"

    local out
    out=$(HALO_BETA_BEARER_FILE="$bearers" \
          HALO_BETA_AUDIT_DIR="$sandbox/audit" \
          HALO_BETA_MEMORY_DIR="$sandbox/memory" \
          MOCK_REVOKE_LOG="$mock_log" \
          bash "$sandbox/bin/1bit-beta-expire.sh" --apply 2>&1) || true

    _expect "apply announces APPLY mode" "$out" "APPLY"

    # Mock called for alice + charlie, NOT bob.
    local alice_hits bob_hits charlie_hits
    alice_hits=$(grep -cx 'alice'   "$mock_log" 2>/dev/null || printf 0)
    bob_hits=$(grep     -cx 'bob'     "$mock_log" 2>/dev/null || printf 0)
    charlie_hits=$(grep -cx 'charlie' "$mock_log" 2>/dev/null || printf 0)

    _expect_eq "mock revoke called for alice"   "$alice_hits"   "1"
    _expect_eq "mock revoke NOT called for bob" "$bob_hits"     "0"
    _expect_eq "mock revoke called for charlie" "$charlie_hits" "1"

    rm -rf "$sandbox"
}

test_apply_writes_audit_log() {
    _test "--apply writes the expired lines to the audit log"
    local sandbox; sandbox=$(make_sandbox)
    local bearers="$sandbox/caddy/bearers.txt"
    local expired_iso
    expired_iso=$(iso_past 4)

    printf 'sk-halo-ddd  # dave  # issued 2026-04-01T00:00Z  # expires %s\n' "$expired_iso" > "$bearers"

    local mock_log="$sandbox/revoke.log"
    : > "$mock_log"
    cp "$SCRIPT" "$sandbox/bin/1bit-beta-expire.sh"
    chmod 0755 "$sandbox/bin/1bit-beta-expire.sh"

    HALO_BETA_BEARER_FILE="$bearers" \
        HALO_BETA_AUDIT_DIR="$sandbox/audit" \
        HALO_BETA_MEMORY_DIR="$sandbox/memory" \
        MOCK_REVOKE_LOG="$mock_log" \
        bash "$sandbox/bin/1bit-beta-expire.sh" --apply >/dev/null 2>&1 || true

    local audit_file
    audit_file=$(find "$sandbox/audit" -name 'expired-*.log' -type f | head -1)
    if [[ -z "$audit_file" ]]; then
        _expect_eq "audit file exists" "missing" "present"
    else
        local contents
        contents=$(cat "$audit_file")
        _expect "audit log contains dave's bearer line" "$contents" "sk-halo-ddd"
    fi

    rm -rf "$sandbox"
}

test_memory_allowlist_protects_load_bearing_files() {
    _test "allow-listed memory files are never purged even with [beta-ttl] tag"
    local sandbox; sandbox=$(make_sandbox)
    local mem="$sandbox/memory"

    # Allow-listed file with a beta-ttl tag AND an ancient mtime — must stay.
    cat > "$mem/project_reddit_relaunch.md" <<EOF
---
name: Reddit relaunch
type: project
---

Body with [beta-ttl] tag.
EOF
    touch -d "30 days ago" "$mem/project_reddit_relaunch.md"

    # Non-allow-listed file with a beta-ttl tag and ancient mtime — must purge (in --apply).
    cat > "$mem/project_beta_note_xyz.md" <<EOF
---
name: Beta note xyz
type: project
---

Body with [beta-ttl] tag.
EOF
    touch -d "30 days ago" "$mem/project_beta_note_xyz.md"

    local mock_log="$sandbox/revoke.log"
    : > "$mock_log"
    cp "$SCRIPT" "$sandbox/bin/1bit-beta-expire.sh"
    chmod 0755 "$sandbox/bin/1bit-beta-expire.sh"

    local out
    out=$(HALO_BETA_BEARER_FILE="/dev/null" \
          HALO_BETA_AUDIT_DIR="$sandbox/audit" \
          HALO_BETA_MEMORY_DIR="$mem" \
          MOCK_REVOKE_LOG="$mock_log" \
          bash "$sandbox/bin/1bit-beta-expire.sh" --apply 2>&1)

    if [[ -f "$mem/project_reddit_relaunch.md" ]]; then
        _expect_eq "allow-listed file survived --apply" "present" "present"
    else
        _expect_eq "allow-listed file survived --apply" "DELETED" "present"
    fi

    if [[ ! -f "$mem/project_beta_note_xyz.md" ]]; then
        _expect_eq "tagged non-allow-listed file was purged" "purged" "purged"
    else
        _expect_eq "tagged non-allow-listed file was purged" "present" "purged"
    fi

    rm -rf "$sandbox"
}

test_memory_expires_frontmatter_field() {
    _test "memory files with frontmatter expires: in the past purge on --apply"
    local sandbox; sandbox=$(make_sandbox)
    local mem="$sandbox/memory"
    local past_date
    past_date=$(date -u -d '2 days ago' +%Y-%m-%d)

    cat > "$mem/project_beta_ephemeral.md" <<EOF
---
name: Ephemeral beta note
type: project
expires: $past_date
---

Body.
EOF

    local mock_log="$sandbox/revoke.log"
    : > "$mock_log"
    cp "$SCRIPT" "$sandbox/bin/1bit-beta-expire.sh"
    chmod 0755 "$sandbox/bin/1bit-beta-expire.sh"

    # Dry run first — no deletion.
    HALO_BETA_BEARER_FILE="/dev/null" \
        HALO_BETA_AUDIT_DIR="$sandbox/audit" \
        HALO_BETA_MEMORY_DIR="$mem" \
        MOCK_REVOKE_LOG="$mock_log" \
        bash "$sandbox/bin/1bit-beta-expire.sh" >/dev/null 2>&1

    if [[ -f "$mem/project_beta_ephemeral.md" ]]; then
        _expect_eq "dry-run kept the file" "present" "present"
    else
        _expect_eq "dry-run kept the file" "DELETED" "present"
    fi

    # Apply — now it goes.
    HALO_BETA_BEARER_FILE="/dev/null" \
        HALO_BETA_AUDIT_DIR="$sandbox/audit" \
        HALO_BETA_MEMORY_DIR="$mem" \
        MOCK_REVOKE_LOG="$mock_log" \
        bash "$sandbox/bin/1bit-beta-expire.sh" --apply >/dev/null 2>&1

    if [[ ! -f "$mem/project_beta_ephemeral.md" ]]; then
        _expect_eq "--apply purged expires: file" "purged" "purged"
    else
        _expect_eq "--apply purged expires: file" "present" "purged"
    fi

    rm -rf "$sandbox"
}

test_idempotent_no_bearers_file() {
    _test "missing bearer file is a non-fatal no-op"
    local sandbox; sandbox=$(make_sandbox)

    cp "$SCRIPT" "$sandbox/bin/1bit-beta-expire.sh"
    chmod 0755 "$sandbox/bin/1bit-beta-expire.sh"

    local out rc
    out=$(HALO_BETA_BEARER_FILE="$sandbox/caddy/does-not-exist.txt" \
          HALO_BETA_AUDIT_DIR="$sandbox/audit" \
          HALO_BETA_MEMORY_DIR="$sandbox/memory" \
          MOCK_REVOKE_LOG="$sandbox/revoke.log" \
          bash "$sandbox/bin/1bit-beta-expire.sh" --apply 2>&1)
    rc=$?
    _expect_eq "exit code 0 when bearer file missing" "$rc" "0"
    _expect "message mentions skipping bearer sweep" "$out" "skipping bearer sweep"

    rm -rf "$sandbox"
}

# ---- run ---------------------------------------------------------------

printf '1bit-beta-expire.sh test suite\n==============================\n'
test_parse_bearer_expiry_reads_expires_field
test_parse_bearer_expiry_falls_back_to_issued_plus_ttl
test_parse_bearer_expiry_empty_when_no_fields
test_parse_bearer_handle
test_dry_run_reports_expired_without_touching
test_apply_calls_mock_revoke
test_apply_writes_audit_log
test_memory_allowlist_protects_load_bearing_files
test_memory_expires_frontmatter_field
test_idempotent_no_bearers_file

printf '\n------------------------------\n'
printf 'tests: %d, passed: %d, failed: %d\n' \
    "$TESTS_RUN" "$TESTS_PASSED" "$TESTS_FAILED"
if (( TESTS_FAILED > 0 )); then
    printf 'failed cases:\n'
    for n in "${FAILED_NAMES[@]}"; do
        printf '  - %s\n' "$n"
    done
    exit 1
fi
exit 0
