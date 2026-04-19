// `halo update` — pull + rebuild + restart. Lean: three phases, fail-fast.

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::Command;

const WORKSPACE: &str = "/home/bcloud/repos/halo-workspace";
const ROCM_CPP:  &str = "/home/bcloud/repos/rocm-cpp";
const ANVIL:     &str = "/home/bcloud/bin/halo-anvil.sh";

fn step(title: &str) { println!("\n── {title} ──"); }

fn run_in(dir: &Path, bin: &str, args: &[&str]) -> Result<()> {
    println!("  $ {bin} {}", args.join(" "));
    let s = Command::new(bin).args(args).current_dir(dir).status()
        .with_context(|| format!("spawn {bin}"))?;
    if !s.success() { bail!("{bin} {} failed in {}", args.join(" "), dir.display()); }
    Ok(())
}

pub async fn run(no_build: bool, no_restart: bool) -> Result<()> {
    let ws = Path::new(WORKSPACE);
    let rc = Path::new(ROCM_CPP);

    step("pull");
    run_in(ws, "git", &["pull", "--ff-only"])?;
    if rc.exists() {
        run_in(rc, "git", &["pull", "--ff-only"]).ok(); // rocm-cpp pull is best-effort
    }

    if !no_build {
        step("build rust workspace");
        run_in(ws, "cargo", &["build", "--release", "--workspace"])?;
        run_in(ws, "cargo", &["install", "--path", "crates/halo-cli",    "--force", "--quiet"])?;
        run_in(ws, "cargo", &["install", "--path", "crates/halo-server", "--force", "--quiet",
                              "--features", "real-backend"])?;

        if Path::new(ANVIL).exists() {
            step("delegate rocm-cpp rebuild to anvil");
            run_in(ws, ANVIL, &[])?;
        } else {
            println!("  (anvil missing at {ANVIL}, skip kernel rebuild)");
        }
    }

    if !no_restart {
        step("restart gen-2 server");
        run_in(ws, "systemctl", &["--user", "restart", "strix-server.service"])?;
    }

    println!("\n✓ update complete");
    Ok(())
}
