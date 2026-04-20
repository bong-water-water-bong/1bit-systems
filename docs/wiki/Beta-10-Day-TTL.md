# Beta 10-Day TTL

**Policy (2026-04-20):** the halo-ai private beta demo window is ten
days. After ten days every bearer-token + Headscale user issued during
the window auto-revokes, and any memory files explicitly tagged as beta
artifacts purge themselves. Re-invitation issues a fresh ten-day token.

See also: [VPN-Only-API](VPN-Only-API.md).

## Why a TTL at all

1. **Controlled exposure.** We cap the beta at ten seats behind
   per-user bearers (see [VPN-Only-API](VPN-Only-API.md)). Without a
   TTL, inactive tokens pile up and the cap stops meaning anything.
2. **Demo window, not lifetime access.** The first ten days are a
   hands-on evaluation. Staying on past that is a deliberate act —
   the invitee asks for a renewal, we re-run `halo-mesh-invite.sh`,
   the clock resets. Nobody drifts into permanent access by inertia.
3. **Artifact hygiene.** Memory notes like "testing with @alice on
   2026-04-20" are legitimate during the demo and dead weight a month
   later. Explicit `expires:` metadata lets us write them down without
   them becoming permanent noise.
4. **Blast-radius cap.** If a laptop walks off with a bearer in
   `~/.config/`, we have at most ten days of exposure, not a year.

## What gets purged

| Artifact | Where | Trigger |
|---|---|---|
| Bearer token line | `/etc/caddy/bearers.txt` | `expires <ISO>` in the past (or, fallback, `issued <ISO>` + 10 days) |
| Headscale user + registered nodes | Headscale DB | `halo-mesh-revoke.sh <handle>` (chained from the sweep) |
| Caddy auth | reloaded via `systemctl reload caddy.service` | chained from revoke |
| Bearer line audit trail | `/var/log/halo-beta/expired-<YYYY-MM-DD>.log`, root:root 0600 | **archived** before the line leaves `bearers.txt` |
| Claude memory file with `expires: <ISO-date>` in frontmatter, past | `~/.claude/projects/-home-bcloud/memory/*.md` | explicit date hit |
| Claude memory file with `[beta-ttl]` body tag AND mtime > 10 days | same | tag + stale |

## What does NOT get purged

None of the following is touched by `halo-beta-expire.sh`, ever:

- **Load-bearing project memory files** — hard allow-list in the
  script. Current list:
  - `project_reddit_relaunch`
  - `project_hermes_integration`
  - `project_lemonade_10_2_pivot`
  - `project_strix_halo_hardware`
  - `project_voice_latency_sharding`
  - `project_apu_thesis`

  When you add a long-horizon project note that should outlive any
  individual beta window, add it to `ALLOWLIST=(...)` in
  `strixhalo/bin/halo-beta-expire.sh` and to the list above.
- **`MEMORY.md`** — the index is explicitly skipped.
- **Git history** of this repo (or any other repo). The sweep deletes
  working-copy memory files; it does not rewrite git.
- **Public documentation** (`docs/wiki/*.md`, `docs/*`, `README*`) — not
  in the scan path.
- **Any file without `type: project` in its YAML frontmatter.** Feedback
  notes, fallback records, reference pages — untouched.
- **Any memory file without EITHER an `expires:` frontmatter field OR a
  `[beta-ttl]` body tag.** The default is "keep." A project note stays
  forever unless it was explicitly tagged as beta-era when authored.

## How to extend a user's access

Just re-invite. `halo-mesh-invite.sh` issues a fresh ten-day token and
appends a new line to `bearers.txt` with a new `expires <ISO>` stamp:

```bash
strixhalo/bin/halo-mesh-invite.sh <handle>
```

There is no "renew in place" operation — we always mint a new token,
so an old compromised token is already in the revocation archive.

## Bearer line format

After 2026-04-20, every line written by `halo-mesh-invite.sh` looks
like:

```
sk-halo-XXXX  # handle  # issued 2026-04-20T14:00Z  # expires 2026-04-30T14:00Z
```

The sweeper prefers the `expires` field. If (for a pre-policy line) the
`expires` field is missing, it falls back to `issued + 10 days`. If
neither is present, the line is left alone — we'd rather miss a sweep
than revoke something we don't understand.

## Cron schedule

Systemd --user timer on strixhalo:

- **`halo-beta-expire.timer`** — `OnCalendar=*-*-* 03:00:00`,
  `OnBootSec=5min`, `Persistent=true`,
  `RandomizedDelaySec=5min`.
- **`halo-beta-expire.service`** — runs
  `strixhalo/bin/halo-beta-expire.sh --apply`.

Install + enable:

```bash
systemctl --user daemon-reload
systemctl --user enable --now halo-beta-expire.timer
systemctl --user list-timers | grep halo-beta
```

### Ad-hoc dry run

No arguments = dry run. The script prints what it *would* do and
exits 0 without touching anything:

```bash
strixhalo/bin/halo-beta-expire.sh          # dry run, default
strixhalo/bin/halo-beta-expire.sh -v       # verbose dry run
strixhalo/bin/halo-beta-expire.sh --apply  # actually revoke + purge
```

### Manual trigger under the timer

```bash
systemctl --user start halo-beta-expire.service
journalctl --user -u halo-beta-expire.service -n 50 --no-pager
```

## Audit trail

Every `--apply` run that revokes at least one bearer writes a
timestamped audit line set to
`/var/log/halo-beta/expired-<YYYY-MM-DD>.log` (root:root, 0600). The
file contains the original bearer lines verbatim — so we have an offline
record of what was valid when, even after the line disappears from
`bearers.txt`. Ship those logs to the pi archive with the rest of
`/var/log` if/when we wire that.

## Cross-refs

- [VPN-Only-API](VPN-Only-API.md) — mesh + bearer fences the TTL protects
- `strixhalo/bin/halo-beta-expire.sh` — the sweep script
- `strixhalo/bin/halo-mesh-invite.sh` — issuer (stamps `expires`)
- `strixhalo/bin/halo-mesh-revoke.sh` — revoker (called per-handle)
- `strixhalo/systemd/halo-beta-expire.{service,timer}` — schedule
- `strixhalo/bin/tests/halo-beta-expire.test.sh` — unit tests (plain bash)
