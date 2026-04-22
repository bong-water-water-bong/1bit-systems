#!/usr/bin/env bash
# sliger-b580-install.sh — Intel Arc B580 (Xe2) audio-pipeline bootstrap.
#
# Run on the sliger host after the B580 is physically swapped in for the
# 1080 Ti. Installs Intel GPU stack, builds whisper.cpp with Vulkan backend
# into /opt/whisper-vulkan, fetches large-v3 weights, and primes the
# systemd unit.
#
# Usage:  ssh 100.64.0.2 'bash -s' < this-script
#
# Prereqs on sliger:
#   - CachyOS or Arch, kernel ≥ 6.19 (built-in xe driver for Battlemage)
#   - B580 physically installed, PCIe link up
#   - Passwordless sudo (already true per memory `user_environment.md`)
#
# Idempotent: re-running skips already-satisfied steps.

set -euo pipefail

STEP() { echo -e "\n=== $* ===" >&2; }

# -- 1. Intel GPU userspace ---------------------------------------------------
STEP "1/6 pacman: Intel GPU + Vulkan + compute-runtime"
sudo pacman -Syu --needed --noconfirm \
    mesa \
    vulkan-intel \
    intel-media-driver \
    intel-compute-runtime \
    level-zero-loader \
    vulkan-tools \
    vulkan-headers \
    vulkan-validation-layers \
    clinfo \
    base-devel \
    cmake \
    git

# -- 2. Driver binding --------------------------------------------------------
STEP "2/6 confirm xe driver binds B580"
if ! lspci -nnk | grep -A3 -iE 'Arc B580|0x3E96|0xE20B|Battlemage' | grep -q 'Kernel driver in use: xe'; then
    echo "WARN: xe driver not bound. If i915 is loaded for Battlemage, add to kernel cmdline:"
    echo "      module_blacklist=i915 xe.force_probe=!0x3E96,!0xE20B,*"
    echo "      Or: force_probe=xe  in /etc/modprobe.d/xe.conf"
    echo "      Then reboot and re-run this script."
fi

# -- 3. Vulkan sanity ---------------------------------------------------------
STEP "3/6 vulkan smoke"
vulkaninfo --summary 2>/dev/null | head -40 || {
    echo "vulkaninfo failed — check that /dev/dri/renderD128 exists + user is in 'render' group"
    groups | grep -q render || sudo usermod -aG render,video "$USER"
    exit 1
}

# -- 4. whisper.cpp build (Vulkan backend) ------------------------------------
STEP "4/6 whisper.cpp with -DGGML_VULKAN=ON into /opt/whisper-vulkan"
WHISPER_SRC="$HOME/whisper.cpp"
WHISPER_PREFIX="/opt/whisper-vulkan"
if [ ! -d "$WHISPER_SRC/.git" ]; then
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp "$WHISPER_SRC"
fi
cd "$WHISPER_SRC"
git pull --ff-only || true
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DWHISPER_BUILD_SERVER=ON
cmake --build build -j"$(nproc)"
sudo mkdir -p "$WHISPER_PREFIX/bin" "$WHISPER_PREFIX/lib" "$WHISPER_PREFIX/include" "$WHISPER_PREFIX/models"
sudo cp build/bin/whisper-cli       "$WHISPER_PREFIX/bin/"
sudo cp build/bin/whisper-server    "$WHISPER_PREFIX/bin/"
# Static-library output layout varies across whisper.cpp revisions — glob both shapes.
sudo cp build/src/libwhisper.so*     "$WHISPER_PREFIX/lib/" 2>/dev/null || true
sudo cp build/libwhisper.so*         "$WHISPER_PREFIX/lib/" 2>/dev/null || true
sudo cp include/whisper.h           "$WHISPER_PREFIX/include/"
sudo ldconfig

# -- 5. weights ---------------------------------------------------------------
STEP "5/6 fetch large-v3 weights (~3.1 GB)"
MODEL="$WHISPER_PREFIX/models/ggml-large-v3.bin"
if [ ! -s "$MODEL" ]; then
    sudo bash -c "cd '$WHISPER_PREFIX/models' && \
        curl -L -o ggml-large-v3.bin \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
fi
ls -lh "$MODEL"

# -- 6. systemd unit ----------------------------------------------------------
STEP "6/6 install + start 1bit-halo-whisper-vulkan.service"
UNIT_SRC="${UNIT_SRC:-$HOME/1bit-halo-workspace/strixhalo/systemd/1bit-halo-whisper-vulkan.service}"
if [ -f "$UNIT_SRC" ]; then
    sudo install -m0644 "$UNIT_SRC" /etc/systemd/system/1bit-halo-whisper-vulkan.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now 1bit-halo-whisper-vulkan.service
    sleep 2
    sudo systemctl status 1bit-halo-whisper-vulkan.service --no-pager | head -12
    echo
    echo "smoke: curl -s http://127.0.0.1:8190/health || echo \"port not reachable\""
    curl -s http://127.0.0.1:8190/health || true
else
    echo "unit file not found at $UNIT_SRC — skipping systemd install"
    echo "rsync the workspace repo to sliger first or pass UNIT_SRC=/path"
fi

STEP "done"
echo "audio pipeline live on sliger at http://100.64.0.2:8190"
echo "next: point 1bit-voice / 1bit-echo clients at this host"
