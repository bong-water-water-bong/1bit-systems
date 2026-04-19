# strixhalo/

Canonical dotfiles for the **strixhalo** reference box
(AMD Ryzen AI MAX+ 395 / Radeon 8060S gfx1151 / CachyOS).
Versioned here so the machine can be rebuilt from a clean install and
so other Strix Halo operators can crib the config.

## Layout

```
strixhalo/
├── systemd/  user systemd units for every halo-* + strix-* service + timer
├── caddy/    Caddyfile with route blocks (BEARER TOKEN REDACTED)
├── bin/      bash helpers that cron-tick into Discord (anvil / librarian / …)
└── fish/     fish shell config (PATH + aliases, no secrets)
```

## What's **not** here

- `/etc/halo-ai/*` — bearer tokens, Reddit session cookies, Discord webhook
  secrets. Never commit.
- `~/.ssh/*` — private keys.
- Model files `~/halo-ai/models/*.h1b` — multi-GB binaries; fetched separately.
- `~/.cache/*`, `~/.cargo/*`, `target/` — build artifacts and caches.

## Deploy on a fresh box

After `install.sh` has bootstrapped the Rust workspace:

```bash
# systemd --user units
install -Dm644 strixhalo/systemd/*.service ~/.config/systemd/user/
install -Dm644 strixhalo/systemd/*.timer  ~/.config/systemd/user/ 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable --now strix-server.service strix-landing.service

# bash helpers
install -Dm755 strixhalo/bin/*.sh ~/bin/

# Caddy (replace the REPLACE_ME bearer token first!)
sudo cp strixhalo/caddy/Caddyfile /etc/caddy/Caddyfile
sudo sed -i "s/sk-halo-REPLACE_ME/$(openssl rand -base64 33 | tr -d '=/+' | head -c 40)/" /etc/caddy/Caddyfile
sudo systemctl restart caddy

# fish
install -Dm644 strixhalo/fish/config.fish ~/.config/fish/config.fish
```

## Update flow

When a unit or helper script changes on the running box:

```bash
# In halo-workspace/
./strixhalo/sync-in.sh    # TODO: pulls live config back into this tree
git diff strixhalo/
git commit -m "strixhalo: update <what>"
git push
```

## Token rotation

The Caddyfile's `sk-halo-…` bearer should be rotated quarterly. Run
`openssl rand -base64 33 | tr -d '=/+' | head -c 40` to generate a new
one, replace in `/etc/caddy/Caddyfile`, distribute to clients, restart
caddy. Do NOT commit the real token back to git.
