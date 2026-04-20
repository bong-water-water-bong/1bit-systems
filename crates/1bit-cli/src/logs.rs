use anyhow::Result;
use std::process::Command;

use crate::status::SERVICES;

pub async fn run(service: &str, follow: bool, lines: u32) -> Result<()> {
    let unit = SERVICES
        .iter()
        .find(|(s, _, _)| *s == service)
        .map(|(_, u, _)| *u)
        .ok_or_else(|| anyhow::anyhow!("unknown service '{service}'"))?;
    let mut c = Command::new("journalctl");
    c.args(["--user", "-u", unit, "-n", &lines.to_string()]);
    if follow {
        c.arg("-f");
    }
    c.status()?;
    Ok(())
}
