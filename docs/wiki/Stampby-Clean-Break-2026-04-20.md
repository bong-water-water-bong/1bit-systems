# Stampby Clean Break — 2026-04-20

Operator runbook for tearing down the `stampby` GitHub account. One-time
action driven by user directive: **"I need a complete break from stampby."**

## Why

`stampby` is the legacy identity. Everything that still matters has been (or
will be) mirrored to `bong-water-water-bong` or pulled from true upstream.
The account carries three kinds of liability worth retiring:

1. **Fork-of-fork sprawl** — we accidentally forked from Tetramatrix's
   mirrors instead of the true upstreams (see below), so stars and issues
   accrue to the wrong source.
2. **Dozens of empty placeholder repos** from the March–April "halo"
   scaffolding sweeps (`shield`, `net`, `vault`, `mirror`, `fang`, …). Noise.
3. **Identity hygiene** — user wants the name gone from the network graph.

## What gets deleted

All **259** repos owned by `stampby` are tagged `action=delete` in
`/home/bcloud/claude output/stampby-repos-2026-04-20.csv`. Breakdown:

| bucket       | count |
| ------------ | ----- |
| originals    | 98    |
| forks        | 161   |
| private      | 25    |
| public       | 234   |
| **total**    | **259** |

Notable originals being nuked (all already archived; Halo code lives elsewhere):

- `rocm-cpp`, `halo-ai-core`, `halo-mcp`, `agent-cpp`, `halo-brain`,
  `halo-browser`, `halo-1bit`, `halo-ai-site`, `halo-discord-agents`,
  `halo-finetuning`, `halo-arcade`, `bleeding-edge`, `claude-hybrid-proxy`,
  `claude-memory`, `lemonade-finetune`, `comfyui-voice`, `voice-digest`,
  `dotfiles` / `dotfiles-strixhalo`, `echo` / `echo-reddit`, `forge`,
  `sentinel`, `dealer`, `halo`, and the ~30 empty placeholder repos
  (`shield`, `fang`, `shadow`, `ghost`, `meek`, `net`, `vault`, `mirror`,
  `gate`, `pulse`, `keys`, `shelf`, `clock`, `vigil`, `conductor`,
  `bones`, `bottom`, `rhythm`, `axe`, `piper`, `mechanic`, `quartermaster`,
  `bounty`, `crypto`, `arcade`, `mixer`, `muse`, `amp`, `benchmarks`,
  `voxel-extraction`, `voxel-foxhole`, `vox-recorder`, `man-cave`,
  `kokoro-voice`, `sar-bi-ai`, `bonsai-servers`, `ssh-tunnel-llama`,
  `undercroft-reddit`).

Notable forks being nuked (all mirrored / replaceable from true upstream):

- `halo-kokoro` (forked from olokobayusuf/kokoro.cpp — see Halo-Kokoro memory)
- `llamacpp-rocm`, `lemonade-nexus`, `lemon-mlx-engine`, `vllm-rocm`,
  `interviewer`, `stable-diffusion.cpp`, `infinity-arcade` (lemonade-sdk forks)
- `amd-strix-halo-*` (kyuz0 forks)
- `TheRock` (ROCm), `gaia` (amd), `xdna-driver` (amd), `mlx` (ml-explore)
- The three fork-of-fork offenders below.

## What does NOT get deleted

Nothing. Every row in the CSV is tagged `action=delete`. There are no
`keep-review` or `transfer` rows. Rationale: the user asked for a complete
break; anything we still need has been (or will be) re-created cleanly under
`bong-water-water-bong` from true upstream.

Transfer to `bong-water-water-bong` was ruled out up front — the stampby
token doesn't have admin on the destination org, so `gh repo transfer` can't
complete cleanly. Faster to delete and re-fork.

## Fork-of-fork offenders and the true upstreams to re-fork from

If any of these are re-created later under `bong-water-water-bong`, fork
from the **true upstream**, not from Tetramatrix / eddierichter-amd:

| stampby repo          | current parent (proxy fork)      | TRUE upstream to re-fork from                                    |
| --------------------- | -------------------------------- | ---------------------------------------------------------------- |
| `stampby/lemonade`    | `eddierichter-amd/lemonade`      | <https://github.com/lemonade-sdk/lemonade>                       |
| `stampby/mempalace`   | `Tetramatrix/mempalace`          | <https://github.com/MemPalace/mempalace>                         |
| `stampby/caveman`     | `Tetramatrix/caveman`            | <https://github.com/JuliusBrussee/caveman>                       |

Corroborated against the `project_tetramatrix_ecosystem` and
`project_upstream_watch` memory entries — we've already flagged Tetramatrix
as a mirror layer, not the canonical source.

## How the operator runs the nuke script

1. **Refresh gh token with `delete_repo` scope** (current token has `repo`,
   `read:org`, `gist` — not enough to delete):

   ```sh
   gh auth refresh -h github.com -s delete_repo
   ```

   Verify: `gh auth status` should now list `delete_repo` in scopes.

2. **Dry-run first (default, safe)**:

   ```sh
   /home/bcloud/repos/halo-workspace/strixhalo/bin/halo-stampby-nuke.sh
   ```

   Every row prints as `DRY-RUN would run: gh repo delete stampby/<name> --yes`.
   Nothing is touched. Log written to
   `/home/bcloud/claude output/stampby-nuke-<UTCstamp>.log`.

3. **When ready, apply**. Two modes:

   - **Interactive** (recommended for the first handful, sanity check):

     ```sh
     /home/bcloud/repos/halo-workspace/strixhalo/bin/halo-stampby-nuke.sh --apply
     ```

     Prompts `delete https://github.com/stampby/<name>? [y/N]` before each one.

   - **Unattended** (after you've sanity-checked a few):

     ```sh
     /home/bcloud/repos/halo-workspace/strixhalo/bin/halo-stampby-nuke.sh --apply --force
     ```

     No prompts. Rips through all 259 rows.

4. Script refuses to `--apply` unless `gh api user` reports the active user
   is `stampby`. Flip accounts with `gh auth switch -u stampby` first.

## Post-nuke checklist

- [ ] `gh auth logout -u stampby` — remove the stampby token from the
      keyring so we can't accidentally act as them again.
- [ ] `gh auth login` as a `bong-water-water-bong` identity (or whichever
      org account you designate as primary).
- [ ] `rg -n '\bstampby/' /home/bcloud/repos` — no hits; every remaining
      reference should point to the new canonical fork/org. Fix any
      lingering URLs before they 404.
- [ ] `git remote -v` in any active repo (rocm-cpp, halo-workspace, etc.)
      — re-point `origin` / `upstream` off of `stampby/*`.
- [ ] Check `.git/config` and `CMakeLists.txt` / `Cargo.toml` for
      `github.com/stampby/...` URLs. Same fix.
- [ ] Retain the CSV and final `stampby-nuke-*.log` under
      `/home/bcloud/claude output/` (and rsync to Pi archive via
      `halo-archive.sh` — they're the audit trail).
- [ ] **Scorched-earth option**: if the user wants the `stampby` GitHub
      profile itself deleted (not just the repos), that's a manual step
      GitHub only exposes via the web UI:
      <https://github.com/settings/admin> → "Delete account". The gh CLI
      can't do it. Do this AFTER the nuke script finishes, not before
      — deleting the profile first would kill our ability to run the
      script as stampby.

## Files of record

- JSON snapshot: `/home/bcloud/claude output/stampby-repos-2026-04-20.json`
- CSV action list: `/home/bcloud/claude output/stampby-repos-2026-04-20.csv`
- Delete script: `/home/bcloud/repos/halo-workspace/strixhalo/bin/halo-stampby-nuke.sh`
- Run log (produced by script): `/home/bcloud/claude output/stampby-nuke-<UTCstamp>.log`
