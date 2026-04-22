# Troubleshooting

Known failure modes and their fixes. Ordered by frequency, not severity.

## amdgpu OPTC CRTC hang — full Wayland freeze

**Symptom:** compositor freezes hard under concurrent model servers. Requires power-cycle. Kernel log:

```
amdgpu: [drm:dc_dmub_srv_wait_idle [amdgpu]] *ERROR*
  REG_WAIT timeout 1us * 100000 tries - optc35_disable_crtc
```

**Cause:** gfx1151 bug on kernel 7.x.

**Fix:** rollback to LTS via snapper snapshot #6 ("7.00 with claude"):

```bash
sudo snapper -c root rollback 6
sudo limine-mkconfig
sudo reboot
# at limine menu, pick 6.18.22-lts
```

## SMU / VCN / PSP hang on LTS

**Symptom:** `journalctl -b` on boot:

```
amdgpu: SMU: Failed to send message 0x... rv -110  (-ETIME)
amdgpu: [PSP] Failed to load IP FW — LOAD_IP_FW failed
amdgpu: VPE / VCN powergate transition failed
```

**Cause:** `/etc/modprobe.d/halo.conf` Tier-3b parameters tuned for 7.0 misfire on LTS.

**Fix:**

```bash
sudo mv /etc/modprobe.d/halo.conf /etc/modprobe.d/halo.conf.disabled
sudo mkinitcpio -P
sudo reboot
```

## Long-context PPL explodes

**Symptom:** PPL rises monotonically with context, repetition PPL > 4.

**Cause:** RoPE convention drift (interleaved vs HF split-half). Fixed 2026-04-19.

**Fix:** confirm flag and commit:

```bash
bitnet_decode --rope-mode hf-split-half   # not `interleaved`
git -C ~/1bit-halo-core log --oneline | grep -i rope
# must include the 2026-04-19 fix commit
```

## Service won't start — mlock failed

**Symptom:** systemd unit exits immediately, journal shows:

```
bitnet_decode: mlock failed: Operation not permitted
```

**Fix:** add to the unit:

```ini
LimitMEMLOCK=infinity
```

The NPU probe path (`xrt-smi`) needs the same, for the day the gate opens.

## ROCm build fails — gfx1151 not supported

**Symptom:** CMake reports target not supported, or linker bails on unknown arch.

**Fix:** pass the target explicitly everywhere:

```bash
cmake -B build \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DGPU_TARGETS=gfx1151
```

If the distro ROCm drops the arch entirely, build from source. The `llamacpp-rocm` fork's install script is the paved road.

## Latency spikes under load

**Symptom:** tok/s drops 30 – 60% after the first minute of sustained generation.

**Cause:** SCLK falls out of high-perf state.

**Fix:** `1bit-halo-gpu-perf.service` pins SCLK high. Verify:

```bash
systemctl status 1bit-halo-gpu-perf
cat /sys/class/drm/card0/device/power_dpm_force_performance_level
# expect: high
cat /sys/class/drm/card0/device/pp_dpm_sclk
```

## 1bit-halo-memory-sync failing every 15 min

**Symptom:** user timer logs show:

```
1bit-halo-memory-sync: push failed (credentials?)
```

**Cause:** GitHub PAT expired, corrupted `GH_TOKEN` fish universal variable, or missing `admin:public_key` scope.

**Fix:**

```bash
# 1. clear corrupted universal env
set -e --universal GH_TOKEN

# 2. refresh auth with correct scopes
gh auth login --scopes "admin:public_key,repo,workflow"

# 3. re-run the timer
systemctl --user restart 1bit-halo-memory-sync.timer
journalctl --user -u 1bit-halo-memory-sync -f
```
