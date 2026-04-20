#!/usr/bin/env bash
# strix-ai-rs / halo-ai bootstrap. Clones + builds the gen-2 Rust stack and
# its native-HIP kernel dependency (rocm-cpp), then hands off to the
# in-tree `halo install` package manager for per-component wiring.
#
# Idempotent: re-runs only fetch-and-rebuild what changed.
#
# Supported hosts: CachyOS / Arch (pacman) on gfx1151. Other distros can
# crib the deps-check section manually; the build path itself is generic.

set -euo pipefail

# ── palette + feedback helpers ───────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m';  DIM='\033[2m';       BOLD='\033[1m'; NC='\033[0m'

TOTAL_STEPS=6
STEP_IDX=0
STEP_START=0

banner() {
    printf '\n'
    printf '%b╔══════════════════════════════════════════════════════════╗%b\n' "$CYAN" "$NC"
    printf '%b║  %b halo-ai-rs · strix-halo bootstrap %b                      %b║%b\n' "$CYAN" "$BOLD" "$NC" "$CYAN" "$NC"
    printf '%b║  %bgfx1151 · ternary BitNet · Rust orchestration%b             %b║%b\n' "$CYAN" "$DIM" "$NC" "$CYAN" "$NC"
    printf '%b╚══════════════════════════════════════════════════════════╝%b\n' "$CYAN" "$NC"
}

step() {
    STEP_IDX=$((STEP_IDX + 1))
    STEP_START=$(date +%s)
    printf '\n%b▸ [%d/%d] %s%b\n' "$BOLD" "$STEP_IDX" "$TOTAL_STEPS" "$*" "$NC"
    printf '%b  ─────────────────────────────────────────────────────%b\n' "$DIM" "$NC"
}

ok() {
    local dt=$(( $(date +%s) - STEP_START ))
    printf '%b  ✓ %s%b %b(%ds)%b\n' "$GREEN" "$*" "$NC" "$DIM" "$dt" "$NC"
}

info() { printf '%b  · %s%b\n' "$DIM" "$*" "$NC"; }
warn() { printf '%b  ⚠ %s%b\n' "$YELLOW" "$*" "$NC" >&2; }
die()  { printf '%b  ✗ %s%b\n' "$RED" "$*" "$NC" >&2; exit 1; }

# Progress dots for a long-running command. Reads stdin, emits a heartbeat
# every ~2 seconds, buffers output for later retrieval. Returns the
# command's exit code. Usage: `some_long_cmd 2>&1 | progress_pipe "label"`.
progress_pipe() {
    local label=$1
    local n=0
    local line=""
    while IFS= read -r line; do
        n=$((n + 1))
        if (( n % 20 == 0 )); then
            printf '%b    %s · %d lines%b\r' "$DIM" "$label" "$n" "$NC" >&2
        fi
    done
    printf '%b    %s · %d lines total%b\n' "$DIM" "$label" "$n" "$NC" >&2
}

# ── configuration ────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-$HOME/repos}"
ROCM_CPP_URL="${ROCM_CPP_URL:-https://github.com/stampby/rocm-cpp.git}"
ROCM_CPP_DIR="${ROCM_CPP_DIR:-$REPO_ROOT/rocm-cpp}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"
CI_MODE="${CI:-${GITHUB_ACTIONS:-0}}"

banner
info "workspace: $WORKSPACE_DIR"
info "rocm-cpp:  $ROCM_CPP_DIR"
info "rocm root: $ROCM_ROOT"
[[ "$CI_MODE" != "0" ]] && info "mode:      CI (syntax-only)"

# ── step 1: host check ───────────────────────────────────────
step "host + dependency check"
if [[ "$CI_MODE" == "0" ]]; then
    [[ -x "$ROCM_ROOT/bin/rocminfo" ]] || die "no rocminfo at $ROCM_ROOT/bin (install rocm-hip-sdk)"
    if "$ROCM_ROOT/bin/rocminfo" | grep -q gfx1151; then
        info "gfx1151 detected"
    else
        warn "no gfx1151 detected — gen-2 kernels target this specifically"
    fi
else
    info "CI mode — skipping gfx1151 gate"
fi
for t in git cargo cmake; do
    command -v "$t" >/dev/null 2>&1 || die "missing: $t (install first)"
    info "$t $(${t} --version 2>&1 | head -1)"
done
ok "host looks good"

# ── step 2: rocm-cpp sync ────────────────────────────────────
step "fetching rocm-cpp (HIP kernels)"
mkdir -p "$REPO_ROOT"
if [[ -d "$ROCM_CPP_DIR/.git" ]]; then
    info "existing checkout — pulling origin/main"
    git -C "$ROCM_CPP_DIR" pull --ff-only 2>&1 | progress_pipe "git pull"
else
    info "fresh clone — $ROCM_CPP_URL"
    git clone "$ROCM_CPP_URL" "$ROCM_CPP_DIR" 2>&1 | progress_pipe "git clone"
fi
ok "rocm-cpp HEAD $(git -C "$ROCM_CPP_DIR" rev-parse --short HEAD)"

# ── step 3: rocm-cpp build ───────────────────────────────────
step "building rocm-cpp (bitnet_decode + librocm_cpp.so)"
if [[ "$CI_MODE" != "0" ]]; then
    info "CI mode — skipping (no ROCm on runners)"
    ok "skipped"
else
    mkdir -p "$ROCM_CPP_DIR/build"
    pushd "$ROCM_CPP_DIR/build" >/dev/null
    info "cmake configure"
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | progress_pipe "cmake"
    info "cmake build (this takes ~2 min on first run)"
    cmake --build . -j "$(nproc)" --target bitnet_decode rocm_cpp 2>&1 | progress_pipe "hipcc"
    popd >/dev/null
    ok "bitnet_decode built"
fi

# ── step 4: halo-workspace build ─────────────────────────────
step "building halo-workspace (Rust, release)"
if [[ "$CI_MODE" != "0" ]]; then
    info "cargo check (CI mode — no link)"
    ( cd "$WORKSPACE_DIR" && cargo check --workspace --release 2>&1 | progress_pipe "cargo" )
    ok "check clean"
else
    info "cargo build --release --workspace (this takes ~1 min on first run)"
    ( cd "$WORKSPACE_DIR" && cargo build --release --workspace 2>&1 | progress_pipe "cargo" )
    ok "build clean"
fi

# ── step 5: halo-cli install ─────────────────────────────────
step "installing halo-cli → \$HOME/.cargo/bin/halo"
( cd "$WORKSPACE_DIR" && cargo install --path crates/halo-cli --force --quiet 2>&1 | progress_pipe "install" )
if [[ -x "$HOME/.cargo/bin/halo" ]]; then
    ok "$($HOME/.cargo/bin/halo --version 2>&1 | head -1)"
else
    warn "halo binary not at ~/.cargo/bin — is \$PATH set?"
fi

# ── step 6: packages.toml handoff ────────────────────────────
step "running halo install core (gen-2 Rust server + HIP backend)"
if [[ "$CI_MODE" != "0" ]]; then
    info "CI mode — skipping"
    ok "skipped"
else
    export ROCM_CPP_LIB_DIR="$ROCM_CPP_DIR/build"
    if "$HOME/.cargo/bin/halo" install core 2>&1 | progress_pipe "halo install"; then
        ok "core component up"
    else
        warn "halo install core failed; inspect with 'halo doctor'"
    fi
fi

# ── banner ────────────────────────────────────────────────────
printf '\n%b╔══════════════════════════════════════════════════════════╗%b\n' "$GREEN" "$NC"
printf     '%b║  ✓ halo-ai-rs bootstrap complete                         ║%b\n' "$GREEN" "$NC"
printf     '%b╚══════════════════════════════════════════════════════════╝%b\n\n' "$GREEN" "$NC"

cat <<EOF
Next steps:
  halo status             # 7-service snapshot
  halo doctor             # full health check (GPU, kernel, ports, endpoints)
  halo install --list     # optional components (agents, voice, sd, gaia)
  halo chat               # interactive REPL against :8180
  halo say "hello"        # TTS via halo-kokoro :8083
  halo bench              # shadow-burnin parity summary
  halo ppl                # perplexity vs gen-1 baseline 9.1607

Landing page:  https://strixhalo.local/
Docs:          README.md, ARCHITECTURE.md, DEMO.md

EOF
