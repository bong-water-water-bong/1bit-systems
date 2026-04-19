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

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { printf "${CYAN}[halo-install]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[halo-install]${NC} %s\n" "$*" >&2; }
die()  { printf "${RED}[halo-install]${NC} %s\n" "$*" >&2; exit 1; }

REPO_ROOT="${REPO_ROOT:-$HOME/repos}"
ROCM_CPP_URL="${ROCM_CPP_URL:-https://github.com/stampby/rocm-cpp.git}"
ROCM_CPP_DIR="${ROCM_CPP_DIR:-$REPO_ROOT/rocm-cpp}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"
CI_MODE="${CI:-${GITHUB_ACTIONS:-0}}"

step() { log "$*"; }

# ── dependency check ─────────────────────────────────────────
need() {
    command -v "$1" >/dev/null 2>&1 || die "missing: $1 (install first)"
}

step "checking host"
if [[ "$CI_MODE" == "0" ]]; then
    [[ -x "$ROCM_ROOT/bin/rocminfo" ]] || die "no rocminfo at $ROCM_ROOT/bin (install rocm-hip-sdk)"
    "$ROCM_ROOT/bin/rocminfo" | grep -q gfx1151 || warn "no gfx1151 detected — gen-2 kernels target this specifically"
else
    log "CI mode ($CI_MODE) — skipping gfx1151 gate"
fi
for t in git cargo cmake; do need "$t"; done

# ── rocm-cpp (HIP kernels) ───────────────────────────────────
mkdir -p "$REPO_ROOT"
if [[ -d "$ROCM_CPP_DIR/.git" ]]; then
    step "pulling rocm-cpp (existing checkout at $ROCM_CPP_DIR)"
    git -C "$ROCM_CPP_DIR" pull --ff-only
else
    step "cloning rocm-cpp → $ROCM_CPP_DIR"
    git clone "$ROCM_CPP_URL" "$ROCM_CPP_DIR"
fi

step "building rocm-cpp (bitnet_decode + librocm_cpp.so)"
mkdir -p "$ROCM_CPP_DIR/build"
pushd "$ROCM_CPP_DIR/build" >/dev/null
if [[ "$CI_MODE" != "0" ]]; then
    log "CI mode — cmake configure only (no compile)"
    cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null
else
    cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build . -j "$(nproc)" --target bitnet_decode rocm_cpp
fi
popd >/dev/null

# ── halo-workspace (Rust) ────────────────────────────────────
step "building halo-workspace (Rust, release)"
if [[ "$CI_MODE" != "0" ]]; then
    ( cd "$WORKSPACE_DIR" && cargo check --workspace --release )
else
    ( cd "$WORKSPACE_DIR" && cargo build --release --workspace )
fi

step "installing halo-cli → ~/.cargo/bin/halo"
( cd "$WORKSPACE_DIR" && cargo install --path crates/halo-cli --force --quiet )

# ── packages.toml handoff ────────────────────────────────────
if [[ "$CI_MODE" == "0" ]]; then
    step "running 'halo install core' (gen-2 Rust server + HIP backend)"
    export ROCM_CPP_LIB_DIR="$ROCM_CPP_DIR/build"
    "$HOME/.cargo/bin/halo" install core || warn "halo install core failed; inspect with 'halo doctor'"
else
    log "CI mode — skipping halo install core"
fi

# ── finish ────────────────────────────────────────────────────
printf '\n%b╔════════════════════════════════════════════════╗%b\n' "$GREEN" "$NC"
printf   '%b║  halo-ai-rs bootstrap complete                 ║%b\n' "$GREEN" "$NC"
printf   '%b╚════════════════════════════════════════════════╝%b\n\n' "$GREEN" "$NC"

cat <<EOF
Next:
  halo status           # service snapshot
  halo doctor           # full health check
  halo install --list   # see optional components
  halo install agents   # 17 specialist dispatcher
  halo install voice    # whisper + kokoro
  halo install sd       # stable diffusion

Reference: README.md, ARCHITECTURE.md.
EOF
