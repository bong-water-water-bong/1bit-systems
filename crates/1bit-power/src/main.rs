// 1bit-power / halo-power — Linux power + thermal control for Strix Halo.
//
// Replaces the Windows-only, closed-source RyzenZPilot tray. Rule-A
// compliant: pure Rust in this crate, shells out to the upstream
// FlyGoat/RyzenAdj binary (C, open source) for the actual MSR writes.
//
// Subcommands:
//   halo-power status                   — print current profile + knobs
//   halo-power profile <name>           — apply a named profile from
//                                         /etc/halo-power/profiles.toml
//   halo-power set <key> <value>        — set a single knob ad-hoc
//                                         (stapm-limit, fast-limit, …)
//   halo-power log                      — emit one JSON metric line and exit
//                                         (systemd timer calls this every 30s)

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::{info, warn};

mod metrics;
mod profiles;
mod ryzen;

use profiles::Profiles;
use ryzen::{PowerBackend, ShelloutBackend};

/// Default on-disk location for the system-wide profile table.
const PROFILES_PATH: &str = "/etc/halo-power/profiles.toml";

#[derive(Parser, Debug)]
#[command(
    name = "halo-power",
    version,
    about = "Strix Halo power/thermal control (RyzenAdj wrapper)"
)]
struct Cli {
    /// Alternate path to profiles.toml (defaults to /etc/halo-power/profiles.toml).
    #[arg(long, global = true)]
    profiles: Option<String>,

    /// Dry-run: print the RyzenAdj invocation, but do not execute.
    #[arg(long, global = true)]
    dry_run: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Print current profile + last-applied knobs as JSON.
    Status,
    /// Apply a named profile from profiles.toml.
    Profile {
        /// Profile name: quiet | balanced | boost | max (per default table).
        name: String,
    },
    /// Override a single RyzenAdj knob without switching profiles.
    Set {
        /// One of: stapm-limit, fast-limit, slow-limit, tctl-temp, vrm-current,
        /// vrmmax-current, vrmsoc-current, vrmsocmax-current.
        key: String,
        /// Integer value. Power limits are mW, currents are mA, temps are °C.
        value: u32,
    },
    /// Emit one line of JSON metrics on stdout and exit.
    /// Intended for a systemd .timer at 30 s cadence.
    Log,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .compact()
        .init();

    let cli = Cli::parse();
    let path = cli.profiles.as_deref().unwrap_or(PROFILES_PATH);
    let backend = ShelloutBackend::new(cli.dry_run);

    match cli.cmd {
        Cmd::Status => {
            let profiles = Profiles::load(path).unwrap_or_else(|e| {
                warn!(error = %e, "could not load profiles.toml; showing defaults");
                Profiles::default()
            });
            let snap = serde_json::json!({
                "profiles_path": path,
                "known_profiles": profiles.names(),
                "backend": backend.name(),
                "dry_run": cli.dry_run,
            });
            println!("{}", serde_json::to_string_pretty(&snap)?);
        }
        Cmd::Profile { name } => {
            let profiles = Profiles::load(path)
                .with_context(|| format!("loading profiles from {path}"))?;
            let prof = profiles
                .get(&name)
                .with_context(|| format!("profile `{name}` not in {path}"))?;
            info!(profile = %name, "applying");
            backend.apply_profile(prof)?;
        }
        Cmd::Set { key, value } => {
            info!(key = %key, value, "one-shot set");
            backend.set_one(&key, value)?;
        }
        Cmd::Log => {
            let line = metrics::sample().context("collecting metrics")?;
            println!("{line}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_parses() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_profile_subcommand() {
        let cli = Cli::try_parse_from(["halo-power", "profile", "balanced"]).unwrap();
        match cli.cmd {
            Cmd::Profile { name } => assert_eq!(name, "balanced"),
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn parses_set_subcommand() {
        let cli =
            Cli::try_parse_from(["halo-power", "set", "stapm-limit", "55000"]).unwrap();
        match cli.cmd {
            Cmd::Set { key, value } => {
                assert_eq!(key, "stapm-limit");
                assert_eq!(value, 55000);
            }
            _ => panic!("wrong subcommand"),
        }
    }
}
