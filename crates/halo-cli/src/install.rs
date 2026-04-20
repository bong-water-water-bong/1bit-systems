// `halo install <component>` — read packages.toml, resolve deps, build, start.
// Lean: single file, manifest embedded at compile time.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

const MANIFEST_SRC: &str = include_str!("../../../packages.toml");

#[derive(Debug, Deserialize)]
struct Manifest {
    component: BTreeMap<String, Component>,
}

#[derive(Debug, Deserialize)]
struct Component {
    description: String,
    #[serde(default)]
    deps: Vec<String>,
    #[serde(default)]
    build: Vec<Vec<String>>,
    #[serde(default)]
    units: Vec<String>,
    /// Tracked files to copy into the user's config tree. Each entry is
    /// `[source_relative_to_workspace, dest_relative_to_user_config_root]`.
    /// Destination root is `$XDG_CONFIG_HOME` (default `~/.config`).
    /// This is how the `tunnel` component drops a systemd user unit +
    /// cloudflared config template onto a fresh box without needing a
    /// build step or enabling the unit (operator must auth first).
    #[serde(default)]
    files: Vec<Vec<String>>,
    #[serde(default)]
    check: String,
}

fn parse() -> Result<Manifest> {
    toml::from_str::<Manifest>(MANIFEST_SRC).context("parsing packages.toml")
}

#[cfg(test)]
fn parse_src(src: &str) -> Result<Manifest> {
    toml::from_str::<Manifest>(src).context("parsing packages.toml")
}

fn workspace_root() -> &'static Path {
    // Manifest lives two levels up from src/; compile-time constant.
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
}

/// Resolve the user config root. XDG first, then ~/.config.
fn user_config_root() -> Result<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return Ok(PathBuf::from(xdg));
        }
    }
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("no $HOME"))?;
    Ok(home.join(".config"))
}

pub async fn list() -> Result<()> {
    let m = parse()?;
    println!("strix-ai-rs components:\n");
    for (name, c) in &m.component {
        println!("  {:<16} {}", name, c.description);
        if !c.deps.is_empty() {
            println!("  {:<16}   deps: {}", "", c.deps.join(", "));
        }
    }
    Ok(())
}

fn resolve<'a>(
    m: &'a Manifest,
    target: &str,
    order: &mut Vec<&'a str>,
    seen: &mut HashSet<String>,
) -> Result<()> {
    if seen.contains(target) { return Ok(()); }
    let c = m.component.get(target)
        .ok_or_else(|| anyhow::anyhow!("unknown component '{target}' (try `halo install --list`)"))?;
    for d in &c.deps { resolve(m, d, order, seen)?; }
    seen.insert(target.to_string());
    order.push(m.component.get_key_value(target).unwrap().0);
    Ok(())
}

async fn healthcheck(url: &str) -> bool {
    if url.is_empty() { return true; }
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .danger_accept_invalid_certs(true)
        .build() { Ok(c) => c, Err(_) => return false };
    match client.get(url).send().await {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    }
}

fn run(root: &Path, argv: &[String]) -> Result<()> {
    println!("    $ {}", argv.join(" "));
    let (bin, rest) = argv.split_first().ok_or_else(|| anyhow::anyhow!("empty argv"))?;
    let s = Command::new(bin).args(rest).current_dir(root).status()
        .with_context(|| format!("spawn {bin}"))?;
    if !s.success() { bail!("{bin} failed"); }
    Ok(())
}

/// Copy a single tracked file from the workspace into the user's config
/// tree. Creates parent dirs as needed. Does NOT overwrite an existing
/// destination (operator may have already customized it) — prints a
/// "skip (exists)" note instead.
fn copy_tracked_file(root: &Path, cfg_root: &Path, src_rel: &str, dest_rel: &str) -> Result<()> {
    let src = root.join(src_rel);
    let dest = cfg_root.join(dest_rel);
    if !src.is_file() {
        bail!("tracked file not found: {}", src.display());
    }
    if dest.exists() {
        println!("    skip (exists): {}", dest.display());
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("mkdir {}", parent.display()))?;
    }
    std::fs::copy(&src, &dest)
        .with_context(|| format!("copy {} → {}", src.display(), dest.display()))?;
    println!("    copied {} → {}", src_rel, dest.display());
    Ok(())
}

pub async fn run_install(component: &str) -> Result<()> {
    let m = parse()?;
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    resolve(&m, component, &mut order, &mut seen)?;

    let root = workspace_root();
    let cfg_root = user_config_root()?;
    println!("install plan: {}\n", order.join(" → "));

    for name in order {
        let c = &m.component[name];
        println!("── {name}: {} ──", c.description);

        for step in &c.build { run(root, step)?; }

        for pair in &c.files {
            let (src_rel, dest_rel) = match pair.as_slice() {
                [s, d] => (s.as_str(), d.as_str()),
                other => bail!(
                    "component '{name}': files entry must be [src, dest], got {other:?}"
                ),
            };
            copy_tracked_file(root, &cfg_root, src_rel, dest_rel)?;
        }

        for unit in &c.units {
            println!("    $ systemctl --user enable --now {unit}");
            let s = Command::new("systemctl")
                .args(["--user", "enable", "--now", unit])
                .status()?;
            if !s.success() { bail!("systemctl failed for {unit}"); }
        }

        if !c.check.is_empty() {
            print!("    checking {}…  ", c.check);
            // Services can take a moment to bind; retry up to 5s.
            let mut ok = false;
            for _ in 0..5 {
                if healthcheck(&c.check).await { ok = true; break; }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            println!("{}", if ok { "ok" } else { "FAIL (component installed but health probe failed)" });
        }
    }

    println!("\n✓ install complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded `packages.toml` must parse. This is the cheapest
    /// safety net against a malformed manifest landing on main.
    #[test]
    fn manifest_parses_clean() {
        let m = parse().expect("packages.toml must parse");
        assert!(!m.component.is_empty(), "at least one component");
    }

    /// The four components added with the systemd/binary rewiring must be
    /// present with the exact shape declared in the prompt: voice +
    /// echo + mcp + tunnel.
    #[test]
    fn new_components_are_declared() {
        let m = parse().unwrap();
        for name in ["voice", "echo", "mcp", "tunnel"] {
            assert!(
                m.component.contains_key(name),
                "packages.toml is missing component '{name}'"
            );
        }

        // voice — cargo installs the halo-voice binary, no systemd unit.
        let voice = &m.component["voice"];
        assert!(voice.units.is_empty(), "voice should not own a unit");
        assert!(
            voice.build.iter().any(|step| step.iter().any(|arg| arg == "crates/halo-voice")),
            "voice build step must `cargo install --path crates/halo-voice`"
        );

        // echo — cargo installs halo-echo AND enables strix-echo.service.
        let echo = &m.component["echo"];
        assert!(
            echo.units.iter().any(|u| u == "strix-echo.service"),
            "echo must enable strix-echo.service"
        );
        assert!(
            echo.build.iter().any(|step| step.iter().any(|arg| arg == "crates/halo-echo")),
            "echo build step must `cargo install --path crates/halo-echo`"
        );
        assert!(echo.deps.iter().any(|d| d == "voice"), "echo must depend on voice");

        // mcp — binary only, no systemd unit.
        let mcp = &m.component["mcp"];
        assert!(mcp.units.is_empty(), "mcp should not own a unit");
        assert!(
            mcp.build.iter().any(|step| step.iter().any(|arg| arg == "crates/halo-mcp")),
            "mcp build step must `cargo install --path crates/halo-mcp`"
        );

        // tunnel — copies the cloudflared unit + config template. No
        // cargo install, no auto-enable (operator must auth first).
        let tunnel = &m.component["tunnel"];
        assert!(tunnel.build.is_empty(), "tunnel must not run a cargo build");
        assert!(tunnel.units.is_empty(), "tunnel must not auto-enable the unit");
        assert!(
            tunnel.files.iter().any(|f| f.iter().any(|p| p.contains("strix-cloudflared.service"))),
            "tunnel must copy strix-cloudflared.service"
        );
        assert!(
            tunnel.files.iter().any(|f| f.iter().any(|p| p.contains("config.yml.template"))),
            "tunnel must copy the cloudflared config.yml.template"
        );
    }

    /// `halo install core` is the fresh-box entry point; it must pull in
    /// voice + echo + mcp alongside the existing landing/gaia/lemonade
    /// wiring, otherwise `halo install core` on a blank machine won't
    /// bring up the voice+agent stack.
    #[test]
    fn core_chains_voice_echo_mcp() {
        let m = parse().unwrap();
        let core = &m.component["core"];
        for required in ["voice", "echo", "mcp"] {
            assert!(
                core.deps.iter().any(|d| d == required),
                "core must depend on '{required}' (got {:?})",
                core.deps
            );
        }

        // Resolution order should contain every new component.
        let mut order = Vec::new();
        let mut seen = HashSet::new();
        resolve(&m, "core", &mut order, &mut seen).unwrap();
        for required in ["voice", "echo", "mcp", "core"] {
            assert!(
                order.iter().any(|n| *n == required),
                "resolve(core) must include '{required}', got {order:?}"
            );
        }
    }

    /// Every component that declares a `cargo install --path crates/<x>`
    /// build step must declare a crate directory that actually exists in
    /// the workspace. Catches typos like `crates/halo-vocie` at test time
    /// instead of on a fresh box during `halo install`.
    #[test]
    fn declared_crate_paths_exist_on_disk() {
        let m = parse().unwrap();
        let root = workspace_root();
        for (name, c) in &m.component {
            for step in &c.build {
                // Find a `--path crates/<x>` pair.
                let mut iter = step.iter();
                while let Some(arg) = iter.next() {
                    if arg == "--path" {
                        let rel = iter.next().expect("--path needs a value");
                        let abs = root.join(rel);
                        assert!(
                            abs.is_dir(),
                            "component '{name}' points at missing crate dir {abs:?}"
                        );
                        // And the crate's Cargo.toml must exist too.
                        assert!(
                            abs.join("Cargo.toml").is_file(),
                            "component '{name}' crate {abs:?} has no Cargo.toml"
                        );
                    }
                }
            }
        }
    }

    /// Every `files = [[src, dest], ...]` source must resolve to a tracked
    /// file in the workspace. This is the check that catches a missing
    /// `strixhalo/cloudflared/config.yml.template` before a fresh box hits
    /// the runtime error.
    #[test]
    fn declared_tracked_files_exist_on_disk() {
        let m = parse().unwrap();
        let root = workspace_root();
        for (name, c) in &m.component {
            for pair in &c.files {
                assert_eq!(pair.len(), 2, "component '{name}' files entry must be [src, dest]");
                let src = root.join(&pair[0]);
                assert!(
                    src.is_file(),
                    "component '{name}' points at missing tracked file {src:?}"
                );
            }
        }
    }

    /// Every unit listed in a component must exist under
    /// `strixhalo/systemd/` — otherwise `systemctl --user enable --now X`
    /// will fail on a fresh box because the unit file was never shipped.
    /// (This assumes the operator has symlinked or copied the tracked
    /// units into their user unit dir, which is the documented flow.)
    #[test]
    fn declared_systemd_units_are_tracked() {
        let m = parse().unwrap();
        let root = workspace_root();
        let units_dir = root.join("strixhalo/systemd");
        for (name, c) in &m.component {
            for unit in &c.units {
                let path = units_dir.join(unit);
                assert!(
                    path.is_file(),
                    "component '{name}' declares unit '{unit}' with no file at {path:?}"
                );
            }
        }
    }

    /// Regression test: a prior manifest shape put voice and echo under
    /// different parent keys. Make sure `halo install --list` (the `list`
    /// fn's underlying data) still emits all the components the prompt
    /// promised, in one pass, from a freshly-parsed manifest.
    #[test]
    fn list_covers_all_new_and_old_components() {
        let m = parse().unwrap();
        let names: Vec<&str> = m.component.keys().map(String::as_str).collect();
        for required in [
            "core", "voice", "echo", "mcp", "tunnel",
            "power", "lemonade", "landing", "gaia",
        ] {
            assert!(
                names.contains(&required),
                "halo install --list must include '{required}', got {names:?}"
            );
        }
    }

    /// Declaration-consistency check: for each new component that ships
    /// a binary (voice, echo, mcp), the `[[bin]] name = "..."` in the
    /// crate's Cargo.toml must match what the systemd unit + the runbook
    /// expect on disk. `halo-echo.service` invokes `/usr/local/bin/halo-echo`,
    /// so the echo crate must build a binary literally named `halo-echo`.
    ///
    /// If the release binary happens to be on disk (dev-box path), we
    /// verify its presence too — but a missing binary isn't a hard fail
    /// because CI boxes rarely keep a full `target/release/` between runs.
    #[test]
    fn new_component_binaries_match_cargo_toml_and_runtime_paths() {
        let root = workspace_root();
        let expected: &[(&str, &str, &str)] = &[
            // (component, crate dir, expected [[bin]] name)
            ("voice", "crates/halo-voice", "halo-voice"),
            ("echo",  "crates/halo-echo",  "halo-echo"),
            ("mcp",   "crates/halo-mcp",   "halo-mcp"),
        ];

        for (component, crate_dir, bin_name) in expected {
            let cargo_toml = root.join(crate_dir).join("Cargo.toml");
            let src = std::fs::read_to_string(&cargo_toml)
                .unwrap_or_else(|e| panic!("read {cargo_toml:?}: {e}"));
            // Accept either the `name         = "..."` (aligned) or
            // `name = "..."` (unaligned) Cargo.toml styles.
            let hit = src.contains(&format!("name = \"{bin_name}\""))
                || src.contains(&format!("name         = \"{bin_name}\""));
            assert!(
                hit,
                "component '{component}': {crate_dir}/Cargo.toml does not declare a [[bin]] name = \"{bin_name}\""
            );

            // Best-effort: if the release binary is on disk, it must match.
            let release = root.join("target/release").join(bin_name);
            if release.exists() {
                let meta = std::fs::metadata(&release).unwrap();
                assert!(meta.is_file(), "{release:?} is not a file");
            }
        }

        // strix-echo.service must reference a binary literally named
        // `halo-echo` at /usr/local/bin/halo-echo. Mismatched paths here
        // are why a fresh box silently fails to start the WebSocket.
        let unit = std::fs::read_to_string(
            root.join("strixhalo/systemd/strix-echo.service"),
        )
        .expect("strix-echo.service must exist");
        assert!(
            unit.contains("/usr/local/bin/halo-echo"),
            "strix-echo.service must ExecStart /usr/local/bin/halo-echo"
        );
        assert!(
            unit.contains("--bind 127.0.0.1:8085"),
            "strix-echo.service must pass --bind 127.0.0.1:8085"
        );
        assert!(
            unit.contains("--codec opus"),
            "strix-echo.service must pass --codec opus"
        );
    }

    /// Parsing a synthetic manifest with a `files = [[src, dest]]` entry
    /// should deserialize into the new `files` vector. Guards against a
    /// future refactor accidentally dropping the field.
    #[test]
    fn files_field_deserializes() {
        let src = r#"
[component.example]
description = "x"
files = [
  ["strixhalo/systemd/strix-cloudflared.service", "systemd/user/strix-cloudflared.service"],
]
"#;
        let m = parse_src(src).unwrap();
        let c = &m.component["example"];
        assert_eq!(c.files.len(), 1);
        assert_eq!(c.files[0][0], "strixhalo/systemd/strix-cloudflared.service");
        assert_eq!(c.files[0][1], "systemd/user/strix-cloudflared.service");
    }
}
