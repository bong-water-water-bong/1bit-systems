// `1bit update` — pull + rebuild + restart. Lean: three phases, fail-fast.

use anyhow::{Context, Result, bail};
use std::path::Path;
use std::process::Command;

// Paths resolved at runtime via $HOME so the CLI isn't tied to any one
// operator's username. Override via HALO_WORKSPACE / HALO_ROCM_CPP /
// HALO_ANVIL env if the layout diverges from the default.
fn home() -> std::path::PathBuf {
    dirs::home_dir().unwrap_or_else(|| ".".into())
}
fn workspace_dir() -> std::path::PathBuf {
    std::env::var_os("HALO_WORKSPACE")
        .map(Into::into)
        .unwrap_or_else(|| home().join("repos/1bit-halo-workspace"))
}
fn rocm_cpp_dir() -> std::path::PathBuf {
    std::env::var_os("HALO_ROCM_CPP")
        .map(Into::into)
        .unwrap_or_else(|| home().join("repos/rocm-cpp"))
}
fn anvil_path() -> std::path::PathBuf {
    std::env::var_os("HALO_ANVIL")
        .map(Into::into)
        .unwrap_or_else(|| home().join("bin/1bit-anvil.sh"))
}

fn step(title: &str) {
    println!("\n── {title} ──");
}

fn run_in(dir: &Path, bin: &str, args: &[&str]) -> Result<()> {
    println!("  $ {bin} {}", args.join(" "));
    let s = Command::new(bin)
        .args(args)
        .current_dir(dir)
        .status()
        .with_context(|| format!("spawn {bin}"))?;
    if !s.success() {
        bail!("{bin} {} failed in {}", args.join(" "), dir.display());
    }
    Ok(())
}

pub async fn run(no_build: bool, no_restart: bool) -> Result<()> {
    let ws = workspace_dir();
    let rc = rocm_cpp_dir();
    let anvil = anvil_path();

    step("pull");
    run_in(&ws, "git", &["pull", "--ff-only"])?;
    if rc.exists() {
        run_in(&rc, "git", &["pull", "--ff-only"]).ok(); // rocm-cpp pull is best-effort
    }

    if !no_build {
        step("build rust workspace");
        run_in(&ws, "cargo", &["build", "--release", "--workspace"])?;
        run_in(
            &ws,
            "cargo",
            &["install", "--path", "crates/1bit-cli", "--force", "--quiet"],
        )?;
        run_in(
            &ws,
            "cargo",
            &[
                "install",
                "--path",
                "crates/1bit-server",
                "--force",
                "--quiet",
                "--features",
                "real-backend",
            ],
        )?;

        if anvil.exists() {
            step("delegate rocm-cpp rebuild to anvil");
            run_in(&ws, anvil.to_str().unwrap_or("1bit-anvil.sh"), &[])?;
        } else {
            println!(
                "  (anvil missing at {}, skip kernel rebuild)",
                anvil.display()
            );
        }
    }

    if !no_restart {
        step("restart gen-2 server");
        run_in(
            &ws,
            "systemctl",
            &["--user", "restart", "strix-server.service"],
        )?;
    }

    println!("\n✓ update complete");
    Ok(())
}
