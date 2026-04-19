use anyhow::Result;
use std::process::Command;
pub async fn run(service: &str, follow: bool, lines: u32) -> Result<()> {
    let unit = format!("halo-{service}.service");
    let mut c = Command::new("journalctl");
    c.args(["--user", "-u", &unit, "-n", &lines.to_string()]);
    if follow { c.arg("-f"); }
    c.status()?;
    Ok(())
}
