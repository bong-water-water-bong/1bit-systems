// `halo bench` — summary view for the shadow-burnin harness.

use anyhow::{Result, bail};
use std::path::PathBuf;
use std::process::Command;

fn script_path() -> PathBuf {
    std::env::var_os("HALO_BURNIN")
        .map(Into::into)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(format!("{home}/1bit systems-core/benchmarks/shadow-burnin.sh"))
        })
}

pub async fn run(max_rounds: Option<u32>, since: Option<String>) -> Result<()> {
    let path = script_path();
    if !path.exists() {
        bail!(
            "shadow-burnin.sh missing at {} (set HALO_BURNIN=/path/to/shadow-burnin.sh)",
            path.display()
        );
    }
    let mut cmd = Command::new("bash");
    cmd.arg(&path);
    if let Some(r) = max_rounds {
        cmd.arg("--max-rounds").arg(r.to_string());
    }
    if let Some(s) = since {
        cmd.arg("--since").arg(s);
    }
    if max_rounds.is_none() {
        cmd.arg("--summary");
    }
    let s = cmd.status()?;
    if !s.success() {
        bail!("shadow-burnin.sh exited {s}");
    }
    Ok(())
}
