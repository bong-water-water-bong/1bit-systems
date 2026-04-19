use anyhow::{bail, Result};
use std::process::Command;
pub async fn run(service: &str) -> Result<()> {
    let unit = format!("halo-{service}.service");
    let s = Command::new("systemctl").args(["--user", "restart", &unit]).status()?;
    if !s.success() { bail!("restart failed for {unit}"); }
    println!("✓ {unit} restarted");
    Ok(())
}
