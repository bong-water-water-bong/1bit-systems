use anyhow::{Result, bail};
use std::process::Command;

use crate::status::SERVICES;

fn resolve(short: &str) -> Option<&'static str> {
    SERVICES
        .iter()
        .find(|(s, _, _)| *s == short)
        .map(|(_, u, _)| *u)
}

pub async fn run(service: &str) -> Result<()> {
    let unit = resolve(service).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown service '{service}' (known: {})",
            SERVICES
                .iter()
                .map(|(s, _, _)| *s)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;
    let s = Command::new("systemctl")
        .args(["--user", "restart", unit])
        .status()?;
    if !s.success() {
        bail!("restart failed for {unit}");
    }
    println!("✓ {unit} restarted");
    Ok(())
}
