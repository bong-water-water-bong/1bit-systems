# Cloudflare Tunnel Setup — halo-ai.studio

Status: plan, 2026-04-20. No live changes made. strixhalo has **no**
`cloudflared` binary and **no** `~/.cloudflared/` or `/etc/cloudflared/`
directory today. This doc is the runbook for the operator (human with
the CF account credentials) to stand the tunnel up.

Companion notes:
- `/etc/caddy/Caddyfile` — existing LAN-only reverse proxy (keeps working)
- `crates/halo-landing` — the marketing page the tunnel will expose
- memory: `project_halo_vision.md` — why halo-ai.studio matters

## 1. Why Cloudflare Tunnel (not port-forward)

- **No public IP exposure.** The tunnel is an outbound-only WebSocket
  from strixhalo to Cloudflare's edge. Home router stays closed.
- **No router config.** No UPnP, no static WAN IP required, no DDNS.
  strixhalo is behind a residential NAT with an unknown (and possibly
  CGNAT'd) public IP — port-forward is not an option anyway.
- **TLS at the edge.** Cloudflare terminates HTTPS with a real
  publicly-trusted cert for `halo-ai.studio`. We forward plaintext
  HTTP to `127.0.0.1:8190` on the loopback — no cert management on
  our side, no Let's Encrypt rate-limit risk.
- **DNS is automatic.** `cloudflared tunnel route dns` writes the
  CNAME for us (chip.ns.cloudflare.com / daisy.ns.cloudflare.com are
  already authoritative; confirmed via `dig NS halo-ai.studio @1.1.1.1`).
- **Rule-A clean.** `cloudflared` is a single Go binary — no Python,
  no interpreter, runs as a user systemd service.

## 2. Install path on CachyOS

`cloudflared` is in the **AUR**, not the core/extra repos. CachyOS
ships `paru` by default; Arch package name is `cloudflared`. Install:

```fish
paru -S cloudflared
```

Verify:

```fish
cloudflared --version
```

Do NOT use the Cloudflare-hosted `.deb`/`.rpm` or the `install.sh` off
cloudflare.com — paru pulls upstream releases and keeps them tracked
by pacman so `halo update` sees them.

## 3. Operator step sequence

All commands run as user `bcloud` on strixhalo. The tunnel lives in
the user's home, not `/etc/`, so it can be managed by `systemctl
--user` and doesn't need root after install.

```fish
# 1. Install (one-time).
paru -S cloudflared

# 2. Browser login — opens a CF dashboard page, operator picks the
#    zone halo-ai.studio, cert.pem lands in ~/.cloudflared/.
cloudflared tunnel login

# 3. Create the tunnel. Writes ~/.cloudflared/<UUID>.json (credentials).
cloudflared tunnel create halo-ai-studio

# 4. Bind DNS. Creates a CNAME halo-ai.studio → <UUID>.cfargotunnel.com
#    in the Cloudflare zone, proxied (orange cloud on).
cloudflared tunnel route dns halo-ai-studio halo-ai.studio

# 5. Drop config.yml (see §4).
#    Note the UUID from step 3 — it's the filename of the JSON in
#    ~/.cloudflared/ and the value of `tunnel:` in config.yml.

# 6. Manual smoke test before wiring systemd.
cloudflared tunnel run halo-ai-studio
#    curl -I https://halo-ai.studio/ should return 200 from halo-landing.
#    Ctrl-C once happy.

# 7. Install + enable user unit (see §5).
systemctl --user daemon-reload
systemctl --user enable --now strix-cloudflared.service
systemctl --user status strix-cloudflared.service

# 8. Enable user-lingering so the unit runs without an active login.
sudo loginctl enable-linger bcloud
```

After step 4 the public DNS is live; after step 7 the tunnel is
persistent. Total: 8 operator steps (install + 7 configure/run).

## 4. ~/.cloudflared/config.yml

```yaml
# ~/.cloudflared/config.yml
# One tunnel, one ingress rule, plus the mandatory catch-all 404.
# Replace <UUID> with the value of `cloudflared tunnel create` output.

tunnel: <UUID>
credentials-file: /home/bcloud/.cloudflared/<UUID>.json

# Keep metrics on loopback for curl-scraping.
metrics: 127.0.0.1:3333

# Allow protocol fallback — default "quic" can get NAT-dropped on
# some ISPs; http2 is the safe-harbour.
protocol: auto

ingress:
  - hostname: halo-ai.studio
    service: http://127.0.0.1:8190
    originRequest:
      # halo-landing is a Rust/axum app on loopback — no cert verify
      # needed, no SNI needed.
      httpHostHeader: halo-ai.studio
      connectTimeout: 10s
      keepAliveConnections: 32

  # Mandatory: last rule must be a catch-all with no hostname.
  - service: http_status:404
```

If we later want `api.halo-ai.studio` to hit `127.0.0.1:8080` with the
bearer gate intact, add another `hostname:` block above the 404 and
run `cloudflared tunnel route dns halo-ai-studio api.halo-ai.studio`.

## 5. ~/.config/systemd/user/strix-cloudflared.service

```ini
[Unit]
Description=Cloudflare Tunnel for halo-ai.studio
Documentation=https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# --no-autoupdate: pacman owns the binary, don't let cloudflared
# reach out and rewrite itself.
ExecStart=/usr/bin/cloudflared --no-autoupdate tunnel run halo-ai-studio
Restart=on-failure
RestartSec=5s
# Hardening — cloudflared only needs loopback + outbound WAN.
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=true

[Install]
WantedBy=default.target
```

Filename matches the `strix-*` convention used by other user units
(`strix-server.service`, etc.). User unit, not system unit, because
credentials live under `/home/bcloud/.cloudflared/` and the service
doesn't need CAP_NET_BIND_SERVICE (it only reaches out, never binds <1024).

## 6. Interaction with existing Caddy

**Zero conflict.** Two independent paths:

| Path | Listener | Who sees it |
|------|----------|-------------|
| LAN: `https://strixhalo.local/` (and `https://10.0.0.10/`) | Caddy on :443 with `tls internal` | Home network + Tailnet peers |
| WAN: `https://halo-ai.studio/` | cloudflared → `127.0.0.1:8190` | Public Internet via Cloudflare |

Caddy keeps serving `strixhalo.local` / `10.0.0.10` / `10.0.0.10:8443`
exactly as today — it doesn't know the tunnel exists, and the tunnel
doesn't touch port 443. halo-landing on `127.0.0.1:8190` is the
shared origin, reached two different ways.

The `/v1/*`, `/v2/*`, `/studio/*`, `/mancave/*`, `/lemon/*`, `/sd/*`
endpoints stay **LAN-only** — they're still gated by Caddy and never
exposed to the public tunnel. Only `halo-ai.studio` → landing.

## 7. halo-landing client-IP gotcha

When the request arrives via Cloudflare Tunnel, halo-landing (axum on
`127.0.0.1:8190`) will see `peer_addr = 127.0.0.1`. The real client
IP is in the `CF-Connecting-IP` header. If we ever want access logs
with real IPs or per-IP rate-limiting, read that header, not
`ConnectInfo<SocketAddr>`. For today's landing page (static status
probe), no code change needed.

## 8. Rollback

If something breaks and we need to shut the public surface fast:

```fish
systemctl --user stop strix-cloudflared.service
systemctl --user disable strix-cloudflared.service
```

Or nuclear: go to dash.cloudflare.com → Zero Trust → Networks →
Tunnels → delete `halo-ai-studio`. DNS record auto-cleans. Re-running
steps 3–7 rebuilds it.
