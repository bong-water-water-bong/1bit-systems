# Network topology — the private mesh

A four-node WireGuard mesh, self-hosted coordinator, zero third-party
control plane. Everything 1bit systems runs — inference, model
sync, archival, Discord/Telegram bots, the pi's ZFS backup — rides on
this mesh. It is the LAN layer the rest of the docs assume.

## The nodes

| ID | hostname          | mesh IP     | LAN IP(s)              | role                                                                                |
| -- | ----------------- | ----------- | ---------------------- | ----------------------------------------------------------------------------------- |
| 1  | `strixhalo-box`   | `100.64.0.1` | `10.0.0.10`           | Headscale coordinator; flagship inference box (gfx1151); Caddy TLS front door       |
| 2  | `sliger`          | `100.64.0.2` | `10.0.0.20`           | Secondary workstation                                                               |
| 3  | `ryzen`           | `100.64.0.3` | `10.0.0.25`, `10.0.0.207` | Dev / training host (dual-homed: ethernet + WiFi)                                   |
| 4  | `pi`              | `100.64.0.4` | `10.0.0.40`           | Archive (ZFS pool, nightly rsync target, long-term weights/bench/model storage)     |

All mesh IPs are in the `100.64.0.0/10` CGNAT range; Tailscale's
conventional space. IPv6 counterparts live under
`fd7a:115c:a1e0::/64`. All four nodes share a single user,
`bcloud`, and are registered to the same Headscale user.

## Physical layout

- Underlying LAN: `10.0.0.0/24` on an ASUS ZenWiFi ET12 mesh (house
  "Garage" + "Shop" points) plus a YuLinca 2.5 Gb switch at `10.0.0.50`.
- Internet at the shop enters via ATM on the IoT SSID; the Garage
  side is the primary internet-backed path.
- The mesh is "private" — no Tailscale DERP relays, no
  `tailscale.com` control plane. DERP can be fallen back to in theory
  (the server config includes STUN on `:3478`) but every pair in the
  current topology is direct LAN and shows `in 0s–1ms` in `tailscale
  ping`.

## The coordinator (Headscale)

`strixhalo-box` runs the Headscale server directly on the box, under
systemd:

```
unit           : headscale.service
listen (plain) : 127.0.0.1:8380
listen (gRPC)  : 127.0.0.1:50443
server_url     : https://headscale.strixhalo.local
stun           : 0.0.0.0:3478
metrics        : 127.0.0.1:9090
```

TLS for `headscale.strixhalo.local` is **terminated by Caddy** at
`:443`, which reverse-proxies to `127.0.0.1:8380`. That means:

1. Headscale itself never speaks TLS directly.
2. Anything that needs to talk to the coordinator must resolve
   `headscale.strixhalo.local` — either via mDNS, via an
   `/etc/hosts` entry, or via a LAN DNS server that maps the name to
   the host.
3. On the coordinator box itself, the name is resolved with an
   `/etc/hosts` pin (`127.0.0.1 headscale.strixhalo.local`).

## Adding a node

```bash
# 1. On strixhalo-box (the coordinator), mint a preauth key.
sudo headscale preauthkeys create --user 1 --expiration 1h
# → hskey-auth-...

# 2. On the joining node, make sure the coordinator is resolvable.
echo "<strixhalo-box-LAN-ip> headscale.strixhalo.local" \
  | sudo tee -a /etc/hosts

# 3. Bring up the tailscale client pointed at Headscale.
sudo tailscale up \
  --login-server=https://headscale.strixhalo.local \
  --authkey=hskey-auth-... \
  --accept-dns=false \
  --force-reauth

# 4. Verify from the coordinator.
sudo headscale nodes list
```

Preauth keys are **single-use by default**. Pass `--reusable=true` if
a bring-up is going to fail and retry a few times.

## Trust model

- All four nodes trust the same user (`bcloud`). No ACL
  segmentation is in place today; any node can reach any service on
  any other node it has an open port on.
- Node-to-node SSH authenticates via the user's SSH key
  (`~/.ssh/id_rsa_asus`); passwordless `sudo` is configured on
  strixhalo-box and sliger, matching `project_halo_network.md` memory.
- The coordinator's Caddy front door gates HTTP services by bearer
  tokens (`sk-halo-*`), not by tailnet membership — mesh reachability
  is necessary but not sufficient to hit a service.

## DNS

`--accept-dns=false` on the tailscale client means the tailnet does
**not** push DNS config to nodes. Name resolution for
`100.64.0.x` hosts happens via:

- **Headscale "magic" hostnames** (`ryzen`, `pi`, etc.) when
  `MagicDNS` is wired — presently broken on strixhalo-box per
  `systemd-resolved` / `NetworkManager` integration (cosmetic warning
  from `tailscale status`).
- **Direct use of the `100.64.0.x` IP** — always works.
- **`/etc/hosts`** on each node for the services it speaks to by name
  (especially `headscale.strixhalo.local`).

Treat MagicDNS as a convenience; the IPs are the source of truth.

## Verifying reachability

Run from any node:

```bash
for p in 100.64.0.1 100.64.0.2 100.64.0.3 100.64.0.4; do
  tailscale ping -c 1 --timeout=3s "$p"
done
```

A healthy matrix looks like the one captured on 2026-04-22:

```
             →  strixhalo   sliger     ryzen       pi
strixhalo    →     —         1ms       0ms         1ms
sliger       →    0ms          —       0ms         1ms
ryzen        →    1ms        0ms         —         0ms
pi           →    0ms        1ms       0ms           —
```

All entries are `pong via 10.0.0.X:41641` — direct LAN, no DERP.

## Operational gotchas

- **Post-reboot DNS on the coordinator.** `/etc/hosts` has to pin
  `127.0.0.1 headscale.strixhalo.local`. If it isn't there — whether
  because the hosts file was regenerated or because the coordinator
  was just rebuilt — `tailscale up` will fail with `failed to resolve
  "headscale.strixhalo.local"`. Re-add the line, retry.
- **"Offline" nodes that just rebooted.** `tailscaled` logs out on
  long uptime gaps and on explicit `tailscale logout`. Re-login
  requires a fresh preauth key, not the old one.
- **Caddy is a single point of failure for the coordinator.** If
  `caddy.service` is down, no client can reach Headscale even though
  the daemon is running — the `:443` front door is the only TLS
  listener. `systemctl status caddy` should always be `active
  (running)` on strixhalo-box.
- **`10.0.0.10:8443`** is a backup Caddy host definition for the
  Headscale vhost — useful when LAN-side clients can't resolve names
  but can still reach IPs. Present in the Caddyfile as a fallback.

## Cross-refs

- `strixhalo/` dotfiles in this repo — Caddyfile, systemd units, the
  `99-npu-memlock.conf` and friends that are installed as part of
  `1bit install core`.
- [Why Caddy + systemd?](./Why-Caddy-Systemd.md) — decision doc for
  the ops layer.
- [VPN-only API](./VPN-Only-API.md) — design decision to keep public
  HTTPS off and serve the `/v2/*` surface only over the mesh.
- `project_halo_network.md` (operator memory) — service/token
  locations, passwordless-sudo scope, nightly rsync target on pi.
