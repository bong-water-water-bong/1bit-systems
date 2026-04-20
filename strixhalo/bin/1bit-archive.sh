#!/usr/bin/env bash
# halo-archive — one-way additive sync strixhalo → pi ZFSPool archive.
#
# Rule: NEVER `--delete`. Files that disappear from strixhalo stay on the Pi
# forever. This is a safety archive, not a mirror.
#
# Destination: pi:/ZFSPool/archive/strixhalo/  (ZFS on the Pi's 5×1TB SATA HAT)
# Transport: ssh over the Headscale tailnet (bcloud@100.64.0.4)

set -u

# Override via HALO_ARCHIVE_PEER / HALO_ARCHIVE_DEST / $HOME to match the
# operator's box. Defaults match the strixhalo reference machine.
PEER="${HALO_ARCHIVE_PEER:-$USER@100.64.0.4}"
DEST_BASE="${HALO_ARCHIVE_DEST:-/ZFSPool/archive/strixhalo}"

# Each entry: "<local_path> <remote_subdir>". Paths resolve under $HOME.
SETS=(
    "$HOME/1bit systems                    1bit systems"
    "$HOME/repos                      repos"
    "$HOME/models                     models"
    "$HOME/.claude/projects           claude-memory"
    "$HOME/claude output              claude-output"
)

RSYNC_OPTS=(
    -a                          # archive mode (perms, times, recursive, symlinks)
    -H                          # preserve hard links
    --info=stats2,progress2
    --human-readable
    # Exclude noise — regenerable or provider-specific junk.
    --exclude='.git/objects/tmp*'
    --exclude='build/'
    --exclude='build-*/'
    --exclude='.cache/'
    --exclude='__pycache__/'
    --exclude='node_modules/'
    --exclude='target/'
    --exclude='.ninja_log'
    --exclude='.ninja_deps'
    --exclude='*.o'
    --exclude='*.obj'
    # NO --delete (explicit, safety-first — Pi retains what strixhalo discards)
)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    RSYNC_OPTS+=(--dry-run)
fi

total_start=$SECONDS
for entry in "${SETS[@]}"; do
    read -r src subdir <<<"$entry"
    dst="$PEER:$DEST_BASE/$subdir/"
    if [[ ! -e "$src" ]]; then
        echo "[halo-archive] SKIP: $src missing"
        continue
    fi
    echo ""
    echo "[halo-archive] === $src → $dst ==="
    # ensure remote dir exists
    ssh "$PEER" "mkdir -p '$DEST_BASE/$subdir'" || {
        echo "[halo-archive] FAIL: remote mkdir"; exit 1;
    }
    # note trailing slash on source — copies contents, not wrapping dir
    rsync "${RSYNC_OPTS[@]}" "${src}/" "$dst" || {
        echo "[halo-archive] FAIL on $src (rsync exit $?)"; exit 1;
    }
done

echo ""
echo "[halo-archive] done in $((SECONDS - total_start))s $([[ $DRY_RUN -eq 1 ]] && echo '(dry-run)' || echo '(committed)')"
