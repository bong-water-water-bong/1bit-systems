# AMD Platform Driver Scan — Strix Halo (Ryzen AI MAX+ 395)

Scanned 2026-04-20. Box: Bosgame M5 / BeyondMax, Sixunited AXB35 board, BIOS 1.07, NPU fw 1.1.2.65, CachyOS 7.0.0.

## The originally linked URL is wrong

`drivers.html/chipsets/am5/b650.html` is the **AM5 desktop chipset** page. Strix Halo is a BGA APU (FP8/FP11 socket-class, soldered); it has **no discrete chipset driver SKU**. The B650 page does not apply.

Correct AMD support roots:

- Ryzen AI MAX+ 395 consumer: `https://www.amd.com/en/support/downloads/drivers.html/processors/ryzen/ryzen-ai-max-series/amd-ryzen-ai-max-plus-395.html`
- Ryzen AI MAX+ PRO 395: `.../ryzen-pro/ryzen-ai-max-pro-300-series/amd-ryzen-ai-max-plus-pro-395.html`

## What AMD officially ships for this SKU on Linux

| Component | AMD package | Linux? | Notes |
|---|---|---|---|
| GPU (gfx1151) | `amdgpu` (mainline) + ROCm 7.x | Yes | Already installed. ROCm Strix Halo guide exists. |
| NPU (XDNA2) | `amdxdna` (mainline kernel 6.10+) + `xrt` + `xrt-plugin-amdxdna` + Ryzen AI SW 1.7.1 | Yes | Installed. SW 1.7.1 (2026-04-08) lists **STX + KRK only**; Strix Halo (STX-H) not in supported matrix — works in practice, not guaranteed. |
| Chipset driver | 8.02.18.557 (2026-03-09) | **Windows only** | Bundles PPM/PMF profiles, SMBus, GPIO. No Linux package. Equivalents live in `amd_pmf`, `amd_pstate_epp`, `k10temp`, `amd_hsmp` kernel drivers — already in CachyOS kernel. |
| Sensor Fusion Hub | `amd-sfh-hid` (mainline ≥ 5.11) | Yes | Laptop sensors only (accel/gyro/ALS). Strix Halo mini-PC has none wired — driver will bind to nothing, that's fine. |
| Infinity Fabric monitoring | **No AMD userland tool** | — | Use `amd_hsmp` + `rocm-smi` + `amd_energy` + `turbostat`. No official package missing. |
| EC / thermals | Vendor BIOS (Bosgame/Sixunited) | — | Not AMD's deliverable. `amd_pmf` slider covers the AMDI0102 ACPI path. |

## What we might be missing — honest list

**Nothing from amd.com is installable on Linux that we don't already have.** The Windows "Chipset Driver 8.02" maps 1:1 to in-tree kernel drivers we already run. Ryzen AI Software 1.7.1 for Linux is the only other AMD-shipped Linux artefact, and it's redundant with our `xrt` + `amdxdna` stack (it bundles the same kernel module + xrt-smi we already have from CachyOS repos).

## BIOS / NPU firmware update path

- AMD does **not** ship BIOS. Strix Halo BIOS comes from the board OEM (Sixunited AXB35 reference → rebadged by Bosgame).
- Canonical update source: **strixhalo.wiki Sixunited AXB35 Firmware page** (`https://strixhalo.wiki/Hardware/Boards/Sixunited_AXB35/Firmware`). All AXB35-based devices share a compatible BIOS family; OEM cross-flashing is community-confirmed but unofficial.
- Bosgame's own page: `https://www.bosgamepc.com/pages/bios-download-center`.
- Linux-native flash path: EFI Shell via `capetron/minisforum-ms-s1-max-bios` recipe (no Windows needed).
- NPU firmware (`1.1.2.65`) is shipped inside `linux-firmware` (`amdnpu/*.sbin`), not the BIOS. Bumps arrive via distro `linux-firmware` updates, not AMD.com.

## Recommendation

Do not install anything from amd.com. Our existing CachyOS-repo stack (`amdgpu`, `amdxdna`, `xrt`, `xrt-plugin-amdxdna`, `linux-firmware`, kernel `amd_pmf`/`amd_hsmp`/`amd-sfh-hid`) is a superset of what AMD publishes for this SKU on Linux. BIOS updates go through the OEM/Sixunited channel on the Strix Halo wiki.
