#!/usr/bin/env bash
# strixhalo/deploy.sh — install the tracked dotfiles onto this box.
# Substitutes __HOME__ placeholder with the invoking user's real home
# directory so systemd unit paths resolve correctly per-user.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REAL_HOME="${REAL_HOME:-$HOME}"
USER_SYSTEMD_DIR="$REAL_HOME/.config/systemd/user"
LOCAL_BIN_DIR="$REAL_HOME/bin"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log() { printf "${CYAN}[deploy]${NC} %s\n" "$*"; }
ok()  { printf "${GREEN}✓${NC} %s\n" "$*"; }

log "deploying with HOME=$REAL_HOME"

# ── systemd user units ──────────────────────────────────────
mkdir -p "$USER_SYSTEMD_DIR"
for unit in "$THIS_DIR/systemd"/*.service "$THIS_DIR/systemd"/*.timer; do
    [[ -f "$unit" ]] || continue
    name=$(basename "$unit")
    sed "s|__HOME__|$REAL_HOME|g" "$unit" > "$USER_SYSTEMD_DIR/$name"
    ok "$name"
done
systemctl --user daemon-reload
log "units installed to $USER_SYSTEMD_DIR"

# ── bash helpers ────────────────────────────────────────────
mkdir -p "$LOCAL_BIN_DIR"
for src in "$THIS_DIR/bin"/*.sh; do
    [[ -f "$src" ]] || continue
    install -m755 "$src" "$LOCAL_BIN_DIR/"
    ok "$(basename "$src")"
done
log "helpers installed to $LOCAL_BIN_DIR"

# ── fish config (append-only, skip if already present) ──────
if [[ -f "$THIS_DIR/fish/config.fish" ]]; then
    mkdir -p "$REAL_HOME/.config/fish"
    if [[ ! -f "$REAL_HOME/.config/fish/config.fish" ]]; then
        install -m644 "$THIS_DIR/fish/config.fish" "$REAL_HOME/.config/fish/config.fish"
        ok "fish/config.fish (new)"
    else
        log "fish config already present — leaving alone"
    fi
fi

# ── Caddy (optional, requires sudo) ─────────────────────────
if [[ -f "$THIS_DIR/caddy/Caddyfile" ]]; then
    log "Caddy: cp $THIS_DIR/caddy/Caddyfile /etc/caddy/Caddyfile (requires sudo + token replacement)"
    log "      sudo sed -i \"s/sk-halo-REPLACE_ME/\$YOUR_TOKEN/g\" /etc/caddy/Caddyfile"
fi

cat <<EOF

${GREEN}deploy complete.${NC}

Enable + start relevant units:
  systemctl --user enable --now strix-server.service strix-landing.service strix-lemonade.service
  systemctl --user enable --now halo-anvil.timer halo-memory-sync.timer

Caddy config is optional — edit /etc/caddy/Caddyfile to install if wanted.
EOF
