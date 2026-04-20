# Why `halo power`?

**One-line answer**: Strix Halo shares one LPDDR5 memory controller between CPU and GPU, so package-power spent on idle x86 cores directly costs decode tok/s. `halo power` exposes three named envelopes that you can flip between without memorizing `ryzenadj` flags.

## The three profiles

| Profile     | stapm | fast | slow | tctl  | When to switch to it                                  |
| ----------- | ----- | ---- | ---- | ----- | ----------------------------------------------------- |
| `inference` |  65 W | 80 W | 75 W | 95 °C | Long bitnet decode, batch PPL runs, burn-in sweeps.   |
| `chat`      |  45 W | 65 W | 55 W | 90 °C | Default — interactive `halo chat`, halo-gaia traffic. |
| `idle`      |  20 W | 35 W | 25 W | 80 °C | Closet-quiet after >60 s of zero traffic.             |

Values come from `docs/halo-power-design.md`; they're a starting point pending a decode-bench sweep.

## Why these three and not more?

We drive profile by *traffic pattern*, not by per-process rules (the RyzenZPilot model). Our process set is small and our HTTP surface is one server — three envelopes cover every realistic halo-ai workload and every extra knob is just another thing to forget about.

## How it works under the hood

- `halo power <profile>` → `sudo ryzenadj --tctl-temp=N --slow-limit=N --fast-limit=N --stapm-limit=N`.
- Flag order is deliberate: thermal ceiling up first, power limits up second. Ctrl-C mid-apply leaves you in a safe state.
- `halo power` (no args) reads back current state via `ryzenadj --info`.
- `halo power --dry-run inference` prints the argv without executing.

## Why not link a Rust ryzenadj crate?

Because `FlyGoat/ryzenadj` tracks AMD Family 19h stepping quirks upstream. When a new Strix Halo BIOS lands, we `pacman -Syu` and inherit the fix. A vendored Rust port would make us the second ryzenadj maintainer — an 800–1200-LOC commitment for ~zero upside. See `docs/halo-power-design.md` option (c).

## Install

Not bundled. `ryzenadj` is an AUR/pacman package on CachyOS:

```
sudo pacman -S ryzenadj
```

If it's missing, `halo power` warns cleanly and no-ops — the CLI stays usable on boxes where thermal tuning isn't relevant (e.g. dev laptops).

## Not in scope

CPU governor (`cpupower`), amdgpu sysfs (`power_dpm_force_performance_level`), and fan curves (BIOS/EC) are all owned elsewhere. See `docs/halo-power-design.md` safety notes.
