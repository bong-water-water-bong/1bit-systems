# AMD Manageability Tools Scan

Source: <https://www.amd.com/en/support/downloads/manageability-tools.html>
(WebFetch timed out three times against the JS-rendered page; reconstructed from
AMD sub-pages and search results — cited inline.)
Date: 2026-04-20.
Target: Ryzen AI MAX+ 395 (Strix Halo, gfx1151, XDNA2) on Linux.

## Verdict

Page is **enterprise-only**. Nothing on it reads LPDDR5 bandwidth,
Infinity Fabric utilization, per-core Zen5 telemetry, iGPU memory clock,
or NPU tile occupancy on a consumer Strix Halo APU under Linux.
Our existing stack (`amd_hsmp`, `rocm-smi`, `xrt-smi`, `ryzenadj` via
`halo power`) already covers the gaps these tools *don't* fill.

## Catalogue

| Tool | URL | Platform | Linux | Relevant |
|---|---|---|---|---|
| AMD Management Console (AMC) | <https://www.amd.com/en/technologies/manageability-tools> | Ryzen PRO desktop/mobile (DASH) | No — Windows only | No — OOB corporate remote-power/asset, not telemetry |
| DASH CLI (v2.2) | <https://developer.amd.com/tools-for-dmtf-dash/> | Ryzen PRO (DASH client) | Yes — open-source SDK builds on Linux | No — OOB power/inventory; no perf counters |
| AMD Management Plugin for SCCM (AMPS) | same | Ryzen PRO | No — Windows/SCCM | No |
| EPYC System Management Software (E-SMS / E-SMI in-band + APML OOB) | <https://www.amd.com/en/developer/e-sms.html>, <https://github.com/amd/esmi_ib_library> | **EPYC server only** | Yes | No — EPYC SoC, not Strix Halo client APU |
| amd\_smi\_exporter | <https://github.com/amd/esmi_ib_library> | EPYC + Instinct → Prometheus | Yes | **Partial** — pattern is reusable, but the exporter itself binds to EPYC/Instinct, not client APU |
| Rasdaemon (AMD EPYC MCE decoding) | linked under E-SMS | EPYC | Yes | No |
| Radeon Pro / Instinct mgmt (ROCm SMI, AMD SMI) | already installed | Instinct + Radeon | Yes | Already in use — not new |

Everything else on the landing page is EPYC BMC / DASH / firmware-update fodder.
One-line dismissal of enterprise-only entries: E-SMS, APML, SB-RMI/SB-TSI,
Radeon Pro manageability, Instinct BMC — not applicable to a consumer BGA APU.

## Anything for `halo power` or `halo doctor`?

**No net-new integrations.** Three notes for future reference:

1. `amd_smi_exporter` (EPYC/Instinct) is a decent **design template** for a
   future `halo_smi_exporter` that unifies `amd_hsmp` + `rocm-smi` +
   `xrt-smi` into one Prometheus endpoint. Code is not reusable; pattern is.
2. DASH CLI is irrelevant for a headless mini-PC we already `ssh` into over
   Headscale.
3. If we ever ship a `halo pro` SKU with AMD PRO Manageability-enabled
   firmware, revisit AMC. Not the case today.

## Honest denials

- WebFetch against the landing page timed out (3x, 60s each). Content
  reconstructed from `manageability-tools` (technologies), `e-sms`, and
  `tools-for-dmtf-dash` sub-pages plus search snippets.
- If AMD later adds a Strix-Halo-specific tool to the page, re-scan.
