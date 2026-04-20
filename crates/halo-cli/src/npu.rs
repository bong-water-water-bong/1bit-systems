//! `halo npu` — XDNA 2 NPU diagnostics on strixhalo. Wraps `xrt-smi` and
//! a few sysfs probes so the operator can see hardware status without
//! remembering the XRT CLI incantations.

use anyhow::{Context, Result};
use clap::Subcommand;
use std::path::Path;
use std::process::Command;

#[derive(Subcommand, Debug)]
pub enum NpuCmd {
    /// Print device name, firmware, XRT + amdxdna versions. Fast.
    Status,
    /// Full `xrt-smi examine` dump.
    Examine,
    /// Run `xrt-smi validate`. CachyOS package doesn't ship test xclbins;
    /// expect "No archive found" until we build our own via Peano.
    Validate,
    /// List firmware blobs under /usr/lib/firmware/amdnpu/.
    Firmware,
    /// Print the path to our NPU boot probe snapshot.
    Snapshot,
}

pub fn run(cmd: NpuCmd) -> Result<()> {
    match cmd {
        NpuCmd::Status     => status(),
        NpuCmd::Examine    => passthrough("xrt-smi", &["examine"]),
        NpuCmd::Validate   => passthrough("xrt-smi", &["validate"]),
        NpuCmd::Firmware   => firmware(),
        NpuCmd::Snapshot   => snapshot(),
    }
}

fn status() -> Result<()> {
    let accel_present = Path::new("/dev/accel/accel0").exists();
    println!("device node   : /dev/accel/accel0 {}",
             if accel_present { "✓" } else { "✗ MISSING — amdxdna module not loaded?" });

    let memlock = Command::new("sh").arg("-c").arg("ulimit -l")
        .output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "?".into());
    let memlock_ok = memlock == "unlimited" || memlock.parse::<u64>().map(|n| n >= 100_000).unwrap_or(false);
    println!("memlock ulimit: {} {}", memlock,
             if memlock_ok { "✓" } else { "✗ raise to unlimited; see /etc/security/limits.d/99-npu-memlock.conf" });

    let examine = Command::new("xrt-smi").arg("examine")
        .output().context("spawn xrt-smi — is it installed? `sudo pacman -S xrt xrt-plugin-amdxdna`")?;
    if !examine.status.success() {
        println!("xrt-smi       : ✗ exit {}", examine.status);
        println!("{}", String::from_utf8_lossy(&examine.stderr));
        anyhow::bail!("xrt-smi examine failed");
    }
    let out = String::from_utf8_lossy(&examine.stdout);
    for line in out.lines() {
        let l = line.trim();
        if l.starts_with("NPU Firmware Version") || l.starts_with("amdxdna Version")
            || l.starts_with("Version") || l.starts_with("Processor")
            || l.starts_with("|[") {
            println!("  {}", l);
        }
    }
    Ok(())
}

fn passthrough(bin: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(bin).args(args).status()
        .with_context(|| format!("spawn {bin}"))?;
    if !status.success() {
        anyhow::bail!("{bin} {:?} exited with {status}", args);
    }
    Ok(())
}

fn firmware() -> Result<()> {
    let dir = Path::new("/usr/lib/firmware/amdnpu");
    if !dir.exists() {
        println!("/usr/lib/firmware/amdnpu/ missing — xrt-plugin-amdxdna not installed?");
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let e = entry?;
        println!("  {}", e.file_name().to_string_lossy());
    }
    println!();
    println!("codes: 17f0_10 = STX, 17f0_11 = STX-H (Strix Halo), 1502_00 = older");
    Ok(())
}

fn snapshot() -> Result<()> {
    let snapshot_dir = "/home/bcloud/claude output/npu-boot-2026-04-20";
    println!("{snapshot_dir}");
    if Path::new(snapshot_dir).exists() {
        for entry in std::fs::read_dir(snapshot_dir)? {
            let e = entry?;
            let name = e.file_name();
            let path = e.path();
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            println!("  {:6} {}", size, name.to_string_lossy());
        }
    } else {
        println!("  (empty — run `halo npu examine > {snapshot_dir}/examine.txt` to populate)");
    }
    Ok(())
}
