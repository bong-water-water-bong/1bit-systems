# halo-agents watch binaries

Two opt-in binaries that keep the 17-specialist registry aware of activity
outside the local machine:

- `halo-watch-discord` — Discord gateway client.
- `halo-watch-github`  — GitHub issue/PR poller (read-only).

Both live in the `halo-agents` crate so they share the registry type and
don't need an IPC boundary to dispatch to a specialist.

## What the Discord bot does

The bot is a **lurker**. It:

1. Connects to Discord using `DISCORD_BOT_TOKEN`.
2. Observes messages in a whitelist of channel IDs
   (`HALO_DISCORD_CHANNELS`, comma-separated).
3. Classifies each message with a lightweight keyword heuristic.
4. Routes the classification + message to a specialist via
   `Registry::dispatch`.
5. Posts a reply **only** when directly `@mentioned` AND the command is
   one of a tiny whitelist (currently: `status`).

It does not moderate, does not DM users, does not follow links, does not
forward bearer tokens, does not read message reactions.

## Classification → specialist routing

| Classification    | Keyword signals                                    | Specialist    |
| ----------------- | -------------------------------------------------- | ------------- |
| `bug_report`      | `bug`, `panic`, `traceback`, `crash`, `broken`…    | `sentinel`    |
| `feature_request` | `can we add`, `would be nice`, `feat:`, `rfc:`…    | `magistrate`  |
| `question`        | trailing `?`, leading `how/what/why/when/where`…   | `herald`      |
| `chat`            | everything else                                    | `herald`      |

Rationale:

- **sentinel** already tails logs + metrics; bug reports feed straight into
  the thing that's watching for regressions.
- **magistrate** is the reviewer; feature proposals are design-review items
  before they touch a branch.
- **herald** is comms; both questions and general chat are herald's lane,
  and herald decides whether to escalate.

Priority when signals overlap: `bug > feature > question > chat`. A
message like "why is this panicking?" is a bug, not a question.

## Environment

| Var                     | Required | Default                     | Purpose                                                   |
| ----------------------- | -------- | --------------------------- | --------------------------------------------------------- |
| `DISCORD_BOT_TOKEN`     | yes      | —                           | Raw Discord bot token (no `Bot ` prefix).                 |
| `HALO_DISCORD_CHANNELS` | yes      | —                           | Comma-separated channel IDs (u64) to observe.             |
| `HALO_SERVER_URL`       | no       | `http://127.0.0.1:8180`     | halo-server base URL for `status` reply (`/v1/models`).   |
| `HALO_LANDING_URL`      | no       | `http://127.0.0.1:8190`     | halo-landing base URL for `status` reply (`/metrics`).    |
| `RUST_LOG`              | no       | `halo_watch_discord=info`   | Standard tracing filter.                                  |

If `DISCORD_BOT_TOKEN` is unset or empty, the binary prints the help
banner and exits 0. This is the fresh-box default — `systemctl --user
status strix-watch-discord` will show `inactive`.

## Plumbing a token

The systemd unit ships at `strixhalo/systemd/strix-watch-discord.service`
with **no token**. To activate:

```sh
mkdir -p ~/.config/systemd/user/strix-watch-discord.service.d
cat >~/.config/systemd/user/strix-watch-discord.service.d/token.conf <<'EOF'
[Service]
Environment="DISCORD_BOT_TOKEN=<your bot token>"
Environment="HALO_DISCORD_CHANNELS=123456789012345678,234567890123456789"
EOF
chmod 600 ~/.config/systemd/user/strix-watch-discord.service.d/token.conf
systemctl --user daemon-reload
systemctl --user enable --now strix-watch-discord
```

The bot needs the **MESSAGE CONTENT INTENT** enabled in the Discord
developer portal; without it classification falls back to `chat` for
every message.

## What the bot MUST NOT do

Hard rules, enforced by absence of code paths. If you find yourself
reaching for any of these, stop.

- **No auto-moderation.** No kicks, bans, deletes, mutes, role changes.
  The bot has no privileged intents for any of that and should not be
  granted them.
- **No channel spam.** Replies only happen on direct `@mention` AND a
  recognised command. There is no scheduled reminder, no "hello new
  member" greeting, no daily summary.
- **No DMs.** The bot does not initiate direct messages. If a user DMs
  the bot, the message is classified + dispatched silently; no reply.
- **No bearer-token handling.** Do not add code that reads
  `Authorization:` headers out of message content, and do not paste
  halo-server tokens into any channel the bot watches.
- **No outbound network except halo-server / halo-landing.** The
  `status` command probes those two only; don't bolt on arbitrary
  web requests.
- **No writes to disk.** The bot has no on-disk state. If you need
  persistence, route through a specialist — do not open a log file in
  this binary.

## Wire format to specialists

Every dispatch sends this JSON to `Registry::dispatch`:

```json
{
  "source": "discord",
  "channel_id": 123456789012345678,
  "author": "username",
  "classification": "bug_report",
  "content": "raw message text"
}
```

Specialists must tolerate unknown fields — we'll add more context
(thread ID, message ID, attachments) in later iterations.

## Testing

Classifier + env-parser tests live in `halo_agents::watch::tests` and run
under `cargo test -p halo-agents`. The startup-without-token contract is
exercised by the integration test at `tests/watch_discord_startup.rs`,
which spawns the compiled binary with the token env cleared and asserts
exit 0 + help banner on stdout.

---

## What the GitHub watcher does

`halo-watch-github` is a **read-only oneshot poller**. It:

1. Reads `HALO_GH_REPOS` (comma-separated `owner/repo`, defaulted below).
2. On each invocation, asks GitHub for issues + PRs updated in the last
   `HALO_GH_POLL_SECONDS × 2` seconds (the 2× lookback buffers for late
   timer fires).
3. Classifies each event from labels + title keywords.
4. Dispatches a compact typed payload to the target specialist via
   `Registry::dispatch`.
5. Exits. Cadence is handled by the systemd timer; the binary does not
   hold its own `loop { sleep }`.

It never writes to GitHub — no issue comments, no labels, no reactions,
no starred, nothing. The PAT scope should be read-only (public_repo is
sufficient for our repos).

## GitHub label → specialist routing

| Signal                                             | Specialist    |
| -------------------------------------------------- | ------------- |
| Pull request (any)                                 | `magistrate`  |
| Label `bug` OR title contains `error`/`crash`/`fail` | `sentinel`  |
| Label `enhancement` OR `feature`                   | `planner`     |
| Label `documentation`                              | `scribe`      |
| Fallback (unlabeled, non-PR, no fault keyword)     | `sentinel`    |

Priority when signals overlap: PR > bug > feature > docs > fallback.

Rationale:

- **magistrate** is the code reviewer; every PR warrants a review ping
  regardless of labels.
- **sentinel** already watches for incidents — anything that smells like
  a failure funnels there. Unrouted issues land on sentinel as well so
  nothing falls through the cracks.
- **planner** owns the roadmap; enhancement / feature issues belong in
  its queue before touching a branch.
- **scribe** handles docs-only edits without burning reviewer time.

## GitHub environment

| Var                     | Required | Default                                                                             | Purpose                                                             |
| ----------------------- | -------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `GITHUB_TOKEN`          | no       | —                                                                                   | PAT (read-only `public_repo`) or installation token. Unset → skip.  |
| `HALO_GH_REPOS`         | no       | `bong-water-water-bong/halo-ai-rs,bong-water-water-bong/bitnet-mlx.rs,strix-ai-rs/halo-workspace` | Comma-separated `owner/repo` list.                                  |
| `HALO_GH_POLL_SECONDS`  | no       | `300`                                                                               | Poll interval in seconds. Lookback is `2×` this.                    |
| `RUST_LOG`              | no       | `info`                                                                              | Standard tracing filter.                                            |

If `GITHUB_TOKEN` is unset, the binary logs `no GITHUB_TOKEN set,
skipping poll` and exits 0. The anonymous 60 req/hr limit would
otherwise get burnt by a 5-minute timer inside the first hour.

## Plumbing a GitHub token

```sh
mkdir -p ~/.config/systemd/user/strix-watch-github.service.d
cat >~/.config/systemd/user/strix-watch-github.service.d/token.conf <<'EOF'
[Service]
Environment="GITHUB_TOKEN=ghp_..."
EOF
chmod 600 ~/.config/systemd/user/strix-watch-github.service.d/token.conf
systemctl --user daemon-reload
systemctl --user enable --now strix-watch-github.timer
```

## What the GitHub watcher MUST NOT do

- **No writes to any GitHub repo.** No comments, labels, reactions,
  reviews, or commits. The PAT is provisioned read-only; keep the code
  aligned with the scope.
- **No long-running loop.** The binary is a oneshot. Timing is the
  timer's job; a resident process would complicate the rate-limit
  story and deviate from the rest of the `strix-*.timer` set.
- **No network outside `api.github.com`.** If you need to hit another
  host, route through a specialist — don't bolt a second transport onto
  this binary.
- **No secrets in tracked files.** `strix-watch-github.service` has no
  `Environment=GITHUB_TOKEN=…` line; the token lives in a drop-in at
  `~/.config/systemd/user/strix-watch-github.service.d/token.conf`.

## Wire format to specialists (GitHub)

Every dispatch sends this JSON to `Registry::dispatch`:

```json
{
  "title": "server crash on startup",
  "body":  "full issue / PR body",
  "labels": ["bug"],
  "author": "githubuser",
  "url":    "https://github.com/owner/repo/issues/123",
  "repo":   "owner/repo",
  "kind":   "issue"
}
```

`kind` is `"issue"` or `"pr"`. Specialists must tolerate unknown fields;
we'll fold in commit SHA, review comments, check-run status in later
iterations.

## GitHub testing

Classifier + env-parser tests live in `halo_agents::watch::github::tests`
and run under `cargo test -p halo-agents --release`. The token-absent
short-circuit is not covered by a binary integration test because
`Registry::default_stubs()` is exercised elsewhere and the branch is
trivially `info!(); return Ok(())`.
