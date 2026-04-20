# Cloudflare Tunnel Setup — api.halo-ai.studio

**Status 2026-04-20:** `halo-ai.studio` apex is **already live** as a Hugo site on Cloudflare Pages (HTTP/2, CF-Ray header, 172.67.134.39). That is the marketing site, served from an external repo, unrelated to this tunnel.

**This tunnel exposes a _different_ subdomain: `api.halo-ai.studio` → halo-server `:8180`**. Purpose: public OpenAI-compat endpoint so Hermes / OpenWebUI / third-party clients can point at us over the open internet without needing Tailscale.

## State on strixhalo

- `cloudflared 2026.3.0` installed (`pacman -S cloudflared`, already in `cachyos-extra-znver4` repo — no AUR needed).
- `~/.cloudflared/` exists, empty except `config.yml.template` (staged, ready to rename after tunnel create).
- `~/.config/systemd/user/strix-cloudflared.service` staged.
- `strixhalo/cloudflared/config.yml.template` + `strixhalo/systemd/strix-cloudflared.service` tracked in the repo for fresh-box installs.

Nothing is running or reachable yet — auth + UUID fill-in still needed.

## Why Cloudflare Tunnel (not port-forward)

- **No public IP exposure.** Outbound-only WebSocket from strixhalo to CF edge. Home router stays closed.
- **No router config.** No UPnP, no static WAN IP, no DDNS. Residential NAT is fine.
- **TLS at the edge.** CF terminates HTTPS with a real publicly-trusted cert. We forward plaintext HTTP to `127.0.0.1:8180` on loopback. No cert management on our side, no Let's Encrypt rate-limit risk.
- **DNS is automatic.** `cloudflared tunnel route dns` writes the CNAME; CF is already authoritative for the zone.
- **Rule A clean.** cloudflared is a single Go binary. No Python, no interpreter.

## Three interactive steps (operator runs)

```bash
# 1. Browser-authenticate with your Cloudflare account
cloudflared tunnel login
#    → opens browser, pick the halo-ai.studio zone, writes
#      ~/.cloudflared/cert.pem

# 2. Create the tunnel (returns a UUID, writes ~/.cloudflared/<UUID>.json)
cloudflared tunnel create api-halo-ai-studio
#    → note the UUID; call it TUUID

# 3. Bind the subdomain to the tunnel (creates a proxied CNAME in CF DNS)
cloudflared tunnel route dns api-halo-ai-studio api.halo-ai.studio
```

## Four mechanical steps (fill in UUID + enable)

```bash
# 4. Copy the staged template to the live config, swap in your UUID
TUUID=<paste from step 2>
cp ~/.cloudflared/config.yml.template ~/.cloudflared/config.yml
sed -i "s/PLACEHOLDER_UUID/$TUUID/g" ~/.cloudflared/config.yml

# 5. Quick smoke test (foreground, kill with Ctrl-C)
cloudflared tunnel --config ~/.cloudflared/config.yml run
#    → from another machine:
#      curl -sS https://api.halo-ai.studio/v1/models
#      should return halo-server's OpenAI-compat /v1/models JSON

# 6. Enable persistent user service
systemctl --user daemon-reload
systemctl --user enable --now strix-cloudflared.service

# 7. Let it survive logout (once per machine)
sudo loginctl enable-linger bcloud

# 8. Verify after reboot:
systemctl --user status strix-cloudflared.service
journalctl --user -u strix-cloudflared.service -n 20
```

## Config reference — `~/.cloudflared/config.yml`

```yaml
tunnel: <UUID>
credentials-file: /home/bcloud/.cloudflared/<UUID>.json

ingress:
  - hostname: api.halo-ai.studio
    service: http://127.0.0.1:8180
    originRequest:
      connectTimeout: 30s
      noTLSVerify: true
  - service: http_status:404
```

`http://127.0.0.1:8180` is the halo-server bind (from `strix-server.service`). halo-server will see requests as coming from 127.0.0.1; the real client IP lands in the `CF-Connecting-IP` header. If rate-limiting or logging per-user is needed later, read that header in halo-server routes.

## Coexistence with Caddy

| Listener | Source | Untouched by tunnel |
|---|---|---|
| `strixhalo.local:443` (Caddy, tls internal) | LAN mesh | ✓ |
| `strixhalo.local:8443` | LAN mesh | ✓ |
| `10.0.0.10:8099` (Caddy bootstrap HTTP) | LAN | ✓ |
| `127.0.0.1:8180` (halo-server) | loopback | **now also routed via CF tunnel → api.halo-ai.studio** |
| `127.0.0.1:8190` (halo-landing) | loopback | keep LAN-only for now |

No Caddyfile change required.

## What NOT to expose via tunnel

- **Admin / metrics endpoints.** `/metrics` on halo-landing leaks per-specialist stats. Keep LAN-only.
- **halo-mcp stdio JSON-RPC.** Not HTTP; wouldn't work anyway.
- **halo-gaia desktop client.** No reason to publicly expose.
- **strix-burnin** / **shadow-burnin** logs. Private.

If we need per-path gating later, CF Access can front the tunnel with email-link auth. That's a follow-up, not required for launch.

## Rollback

```bash
systemctl --user disable --now strix-cloudflared.service
cloudflared tunnel route dns --overwrite-dns api-halo-ai-studio  # removes CNAME
cloudflared tunnel delete api-halo-ai-studio
```

## References

- [`strixhalo/systemd/strix-cloudflared.service`](../../strixhalo/systemd/strix-cloudflared.service) — tracked copy of the user unit
- [`strixhalo/cloudflared/config.yml.template`](../../strixhalo/cloudflared/config.yml.template) — tracked template with PLACEHOLDER_UUID
- Cloudflare Tunnel docs: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/
- halo-server port: `crates/halo-server/src/lib.rs` (bind `127.0.0.1:8180`)
