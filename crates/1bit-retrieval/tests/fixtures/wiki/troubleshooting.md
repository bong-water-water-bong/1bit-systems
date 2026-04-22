# Troubleshooting

Common failure modes and how to get out of them.

## amdgpu OPTC hang

Kernel 7.0 on gfx1151 hits an OPTC CRTC hang signature
`REG_WAIT timeout 1us * 100000 tries - optc35_disable_crtc`.
The amdgpu driver freezes Wayland and needs a power-cycle.
Mitigation: stay on 6.18.22-lts, or set `dcdebugmask=0x410`.

## SMU mailbox timeout

VPE/VCN powergate fails with ETIME. Roll back GFXOFF tunables.

## Caddy bearer mismatch

If `/v1/chat/completions` returns 401, check that the Caddyfile bearer token matches
the client's Authorization header.
