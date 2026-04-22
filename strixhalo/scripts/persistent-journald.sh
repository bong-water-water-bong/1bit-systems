#!/usr/bin/env bash
# persistent-journald.sh — install the halo journald drop-in so crash logs
# survive reboots (OPTC/SMU/VCN hangs on LTS kernel leave nothing on disk
# when Storage=volatile is in effect).
#
# Hostname-agnostic: works on strixhalo, sliger, ryzen, pi — any host with
# systemd-journald. Idempotent: safe to re-run.
#
# Rollout:
#   ssh 100.64.0.1 'bash -s' < strixhalo/scripts/persistent-journald.sh
#   ssh 100.64.0.2 'bash -s' < strixhalo/scripts/persistent-journald.sh
#   # or local:
#   bash strixhalo/scripts/persistent-journald.sh
#
# What it does:
#   1. copies 10-halo-persistent.conf into /etc/systemd/journald.conf.d/
#   2. ensures /var/log/journal/ exists (journald creates on restart too)
#   3. restarts systemd-journald
#   4. prints verification (journalctl --disk-usage, storage mode, retention)
#
# Exit 0 on success, non-zero on any install/restart failure.

set -euo pipefail

STEP() { echo -e "\n=== $* ===" >&2; }

DROPIN_NAME="10-halo-persistent.conf"
DROPIN_DST="/etc/systemd/journald.conf.d/${DROPIN_NAME}"

# Locate the drop-in source next to this script (repo layout:
# strixhalo/scripts/<this>  +  strixhalo/systemd/journald.conf.d/<dropin>).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DROPIN_SRC_DEFAULT="${SCRIPT_DIR}/../systemd/journald.conf.d/${DROPIN_NAME}"
DROPIN_SRC="${DROPIN_SRC:-$DROPIN_SRC_DEFAULT}"

if [ ! -f "$DROPIN_SRC" ]; then
    echo "ERROR: drop-in source not found at $DROPIN_SRC" >&2
    echo "       pass DROPIN_SRC=/path or run from a full repo checkout" >&2
    exit 1
fi

# -- 1. install drop-in -------------------------------------------------------
STEP "1/4 install $DROPIN_NAME into /etc/systemd/journald.conf.d/"
sudo install -d -m0755 /etc/systemd/journald.conf.d
sudo install -m0644 "$DROPIN_SRC" "$DROPIN_DST"

# -- 2. ensure journal dir ----------------------------------------------------
STEP "2/4 ensure /var/log/journal exists"
sudo install -d -m2755 -g systemd-journal /var/log/journal 2>/dev/null \
    || sudo install -d -m2755 /var/log/journal

# -- 3. restart journald ------------------------------------------------------
STEP "3/4 restart systemd-journald"
sudo systemctl restart systemd-journald
sleep 1

# -- 4. verify ----------------------------------------------------------------
STEP "4/4 verify"
echo "-- effective config --"
sudo systemd-analyze cat-config systemd/journald.conf 2>/dev/null \
    | grep -E '^(Storage|MaxRetentionSec|SystemMaxUse)' \
    || grep -E '^(Storage|MaxRetentionSec|SystemMaxUse)' "$DROPIN_DST"
echo
echo "-- journal disk usage --"
sudo journalctl --disk-usage
echo
echo "-- persistence check --"
if [ -d /var/log/journal ] && [ -n "$(sudo ls -A /var/log/journal 2>/dev/null)" ]; then
    echo "persistent journal live at /var/log/journal"
else
    echo "WARN: /var/log/journal is empty — journald may not have rotated yet"
fi

STEP "done"
echo "crash logs will now survive reboot on $(hostname)"
