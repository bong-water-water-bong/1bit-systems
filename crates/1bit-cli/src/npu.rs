//! `1bit npu` — XDNA 2 NPU diagnostics on strixhalo. Wraps `xrt-smi` and
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
    /// List firmware blobs under /usr/lib/firmware/amdnpu/. With
    /// `--check-remote`, diffs local against the canonical linux-firmware
    /// repo at git.kernel.org and reports any newer NPU fw available.
    Firmware {
        #[arg(long)]
        check_remote: bool,
    },
    /// Print the path to our NPU boot probe snapshot.
    Snapshot,
}

pub fn run(cmd: NpuCmd) -> Result<()> {
    match cmd {
        NpuCmd::Status => status(),
        NpuCmd::Examine => passthrough("xrt-smi", &["examine"]),
        NpuCmd::Validate => passthrough("xrt-smi", &["validate"]),
        NpuCmd::Firmware { check_remote } => firmware(check_remote),
        NpuCmd::Snapshot => snapshot(),
    }
}

fn status() -> Result<()> {
    let accel_present = Path::new("/dev/accel/accel0").exists();
    println!(
        "device node   : /dev/accel/accel0 {}",
        if accel_present {
            "✓"
        } else {
            "✗ MISSING — amdxdna module not loaded?"
        }
    );

    let memlock = Command::new("sh")
        .arg("-c")
        .arg("ulimit -l")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "?".into());
    let memlock_ok = memlock == "unlimited"
        || memlock
            .parse::<u64>()
            .map(|n| n >= 100_000)
            .unwrap_or(false);
    println!(
        "memlock ulimit: {} {}",
        memlock,
        if memlock_ok {
            "✓"
        } else {
            "✗ raise to unlimited; see /etc/security/limits.d/99-npu-memlock.conf"
        }
    );

    let examine = Command::new("xrt-smi")
        .arg("examine")
        .output()
        .context("spawn xrt-smi — is it installed? `sudo pacman -S xrt xrt-plugin-amdxdna`")?;
    if !examine.status.success() {
        println!("xrt-smi       : ✗ exit {}", examine.status);
        println!("{}", String::from_utf8_lossy(&examine.stderr));
        anyhow::bail!("xrt-smi examine failed");
    }
    let out = String::from_utf8_lossy(&examine.stdout);
    for line in out.lines() {
        let l = line.trim();
        if l.starts_with("NPU Firmware Version")
            || l.starts_with("amdxdna Version")
            || l.starts_with("Version")
            || l.starts_with("Processor")
            || l.starts_with("|[")
        {
            println!("  {}", l);
        }
    }
    Ok(())
}

fn passthrough(bin: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(bin)
        .args(args)
        .status()
        .with_context(|| format!("spawn {bin}"))?;
    if !status.success() {
        anyhow::bail!("{bin} {:?} exited with {status}", args);
    }
    Ok(())
}

fn firmware(check_remote: bool) -> Result<()> {
    let dir = Path::new("/usr/lib/firmware/amdnpu");
    if !dir.exists() {
        println!("/usr/lib/firmware/amdnpu/ missing — xrt-plugin-amdxdna not installed?");
        return Ok(());
    }

    let mut local_stx_h: Vec<String> = Vec::new();
    println!("local /usr/lib/firmware/amdnpu/:");
    for entry in std::fs::read_dir(dir)? {
        let e = entry?;
        println!("  {}", e.file_name().to_string_lossy());
    }

    let stx_h = dir.join("17f0_11");
    if stx_h.exists() {
        println!();
        println!("STX-H (17f0_11) blobs:");
        for entry in std::fs::read_dir(&stx_h)? {
            let e = entry?;
            let name = e.file_name().to_string_lossy().to_string();
            println!("  {}", name);
            if let Some(v) = extract_version(&name) {
                local_stx_h.push(v);
            }
        }
    }

    println!();
    println!("codes: 17f0_10 = STX, 17f0_11 = STX-H (Strix Halo), 1502_00 = older");

    if check_remote {
        println!();
        println!("polling canonical linux-firmware.git…");
        match poll_remote() {
            Ok(remote) => {
                println!("remote STX-H (17f0_11) fw versions:");
                for v in &remote {
                    println!("  {v}");
                }
                let newer: Vec<&String> = remote
                    .iter()
                    .filter(|r| !local_stx_h.iter().any(|l| l == *r))
                    .collect();
                println!();
                if newer.is_empty() {
                    println!("current — no newer NPU fw upstream.");
                } else {
                    println!("NEWER upstream:");
                    for v in newer {
                        println!("  {v}  ← bump linux-firmware package");
                    }
                }
            }
            Err(e) => println!("remote poll failed: {e}"),
        }
    }
    Ok(())
}

/// Parse a filename like `npu.sbin.1.1.2.65.zst` or `npu.sbin.1.1.2.65`
/// into `1.1.2.65`. Returns `None` for symlinks / other shapes.
fn extract_version(name: &str) -> Option<String> {
    let stem = name.strip_suffix(".zst").unwrap_or(name);
    let v = stem.strip_prefix("npu.sbin.")?;
    if v.chars().all(|c| c.is_ascii_digit() || c == '.') {
        Some(v.to_string())
    } else {
        None
    }
}

fn poll_remote() -> Result<Vec<String>> {
    let url = "https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/plain/amdnpu/17f0_11/";
    let out = Command::new("curl")
        .args(["-sSL", "--max-time", "10", url])
        .output()
        .context("curl fetch remote amdnpu listing")?;
    if !out.status.success() {
        anyhow::bail!("curl exited with {}", out.status);
    }
    let body = String::from_utf8_lossy(&out.stdout);
    let mut versions = Vec::new();
    for line in body.lines() {
        if let Some(idx) = line.find("npu.sbin.") {
            let rest = &line[idx + "npu.sbin.".len()..];
            let v: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let v = v.trim_end_matches('.');
            if !v.is_empty() && !versions.contains(&v.to_string()) {
                versions.push(v.to_string());
            }
        }
    }
    Ok(versions)
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
        println!("  (empty — run `1bit npu examine > {snapshot_dir}/examine.txt` to populate)");
    }
    Ok(())
}
