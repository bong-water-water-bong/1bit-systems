#!/usr/bin/env bash
# halo-caddy-bearers.test.sh — unit tests for 1bit-caddy-bearers.sh.
#
# Runs entirely in a temp directory; never touches /etc/caddy. We
# override BEARER_FILE + CONF_FILE via environment and skip the
# `--reload` path (no systemd in CI).
#
# Run: bash strixhalo/bin/tests/halo-caddy-bearers.test.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/../1bit-caddy-bearers.sh"

if [[ ! -f "$SCRIPT" ]]; then
    echo "FATAL: $SCRIPT not found"
    exit 2
fi

TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_NAMES=()

_expect() {
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

_test() {
    TESTS_RUN=$(( TESTS_RUN + 1 ))
    printf '\n[test] %s\n' "$1"
}

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

run_helper() {
    # Run the script without --reload and without sudo. We set both
    # paths into the tmpdir so the test never touches /etc/caddy.
    BEARER_FILE="$tmpdir/bearers.txt" \
    CONF_FILE="$tmpdir/bearers.conf" \
        bash "$SCRIPT" 2>&1
}

# -------- tests ----------------------------------------------------------

_test "missing bearers.txt → ^\$ pattern (rejects everything)"
rm -f "$tmpdir/bearers.txt" "$tmpdir/bearers.conf"
out=$(run_helper)
_expect "wrote snippet" "$out" "wrote"
conf=$(cat "$tmpdir/bearers.conf")
_expect "empty → ^\$ regex" "$conf" 'Authorization ^$'
_expect "bearer count 0" "$conf" 'Bearer count: 0'

_test "single bearer → single-alternative regex"
cat > "$tmpdir/bearers.txt" <<'EOF'
sk-halo-aaaa  # alice  # issued 2026-04-20T00:00Z  # expires 2026-04-30T00:00Z
EOF
out=$(run_helper)
conf=$(cat "$tmpdir/bearers.conf")
_expect "one token present" "$conf" '(sk-halo-aaaa)'
_expect "bearer count 1" "$conf" 'Bearer count: 1'

_test "three bearers → pipe-joined alternation"
cat > "$tmpdir/bearers.txt" <<'EOF'
sk-halo-aaaa  # alice  # issued 2026-04-20T00:00Z  # expires 2026-04-30T00:00Z
sk-halo-bbbb  # bob  # issued 2026-04-20T00:00Z  # expires 2026-04-30T00:00Z
sk-halo-cccc  # carol  # issued 2026-04-20T00:00Z  # expires 2026-04-30T00:00Z
EOF
run_helper >/dev/null
conf=$(cat "$tmpdir/bearers.conf")
_expect "alice present" "$conf" 'sk-halo-aaaa'
_expect "bob present" "$conf" 'sk-halo-bbbb'
_expect "carol present" "$conf" 'sk-halo-cccc'
_expect "alternation uses |" "$conf" 'sk-halo-aaaa|sk-halo-bbbb|sk-halo-cccc'
_expect "count is 3" "$conf" 'Bearer count: 3'

_test "comments + blank lines + leading/trailing whitespace ignored"
cat > "$tmpdir/bearers.txt" <<'EOF'
# standalone comment
   # indented comment

sk-halo-onlyme  # dave  # issued x  # expires y

# trailing comment
EOF
run_helper >/dev/null
conf=$(cat "$tmpdir/bearers.conf")
_expect "one real token" "$conf" 'Bearer count: 1'
_expect "dave's token present" "$conf" 'sk-halo-onlyme'

_test "malformed token (wildcard in bearer line) is skipped"
cat > "$tmpdir/bearers.txt" <<'EOF'
sk-halo-good  # alice  # expires later
.*  # attacker widens regex
sk-halo-alsogood  # bob  # expires later
EOF
out=$(run_helper 2>&1)
conf=$(cat "$tmpdir/bearers.conf")
_expect "warns about malformed line" "$out" "malformed"
_expect "good tokens kept" "$conf" 'sk-halo-good|sk-halo-alsogood'
_expect "wildcard rejected — no .* in regex body" "$(echo "$conf" | grep -c '\.\*' || true)" "0"

_test "idempotent — second run with same input reports no change"
# Using the last-known-good bearers.txt from previous test (cleaned).
cat > "$tmpdir/bearers.txt" <<'EOF'
sk-halo-aaaa  # alice  # issued x  # expires y
EOF
run_helper >/dev/null
out2=$(run_helper)
_expect "idempotent no-op" "$out2" "already current"

_test "content change triggers fresh write"
cat > "$tmpdir/bearers.txt" <<'EOF'
sk-halo-aaaa  # alice  # issued x  # expires y
sk-halo-zzzz  # zoe  # issued x  # expires y
EOF
out3=$(run_helper)
_expect "rewrite announced" "$out3" "wrote"
conf=$(cat "$tmpdir/bearers.conf")
_expect "zoe added" "$conf" 'sk-halo-zzzz'

# -------- summary --------------------------------------------------------

printf '\n================================\n'
printf '  ran: %d  passed: %d  failed: %d\n' "$TESTS_RUN" "$TESTS_PASSED" "$TESTS_FAILED"
if (( TESTS_FAILED > 0 )); then
    printf '\nFailed tests:\n'
    for name in "${FAILED_NAMES[@]}"; do
        printf '  - %s\n' "$name"
    done
    exit 1
fi
printf '  all passed\n'
exit 0
