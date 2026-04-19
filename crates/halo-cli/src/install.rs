// `halo install <component>` — read packages.toml, resolve deps, build, start.
// Lean: single file, manifest embedded at compile time.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::process::Command;
use std::time::Duration;

const MANIFEST_SRC: &str = include_str!("../../../packages.toml");

#[derive(Debug, Deserialize)]
struct Manifest {
    component: BTreeMap<String, Component>,
}

#[derive(Debug, Deserialize)]
struct Component {
    description: String,
    #[serde(default)]
    deps: Vec<String>,
    #[serde(default)]
    build: Vec<Vec<String>>,
    #[serde(default)]
    units: Vec<String>,
    #[serde(default)]
    check: String,
}

fn parse() -> Result<Manifest> {
    toml::from_str::<Manifest>(MANIFEST_SRC).context("parsing packages.toml")
}

fn workspace_root() -> &'static Path {
    // Manifest lives two levels up from src/; compile-time constant.
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
}

pub async fn list() -> Result<()> {
    let m = parse()?;
    println!("strix-ai-rs components:\n");
    for (name, c) in &m.component {
        println!("  {:<14} {}", name, c.description);
        if !c.deps.is_empty() {
            println!("  {:<14}   deps: {}", "", c.deps.join(", "));
        }
    }
    Ok(())
}

fn resolve<'a>(
    m: &'a Manifest,
    target: &str,
    order: &mut Vec<&'a str>,
    seen: &mut HashSet<String>,
) -> Result<()> {
    if seen.contains(target) { return Ok(()); }
    let c = m.component.get(target)
        .ok_or_else(|| anyhow::anyhow!("unknown component '{target}' (try `halo install --list`)"))?;
    for d in &c.deps { resolve(m, d, order, seen)?; }
    seen.insert(target.to_string());
    order.push(m.component.get_key_value(target).unwrap().0);
    Ok(())
}

async fn healthcheck(url: &str) -> bool {
    if url.is_empty() { return true; }
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .danger_accept_invalid_certs(true)
        .build() { Ok(c) => c, Err(_) => return false };
    match client.get(url).send().await {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    }
}

fn run(root: &Path, argv: &[String]) -> Result<()> {
    println!("    $ {}", argv.join(" "));
    let (bin, rest) = argv.split_first().ok_or_else(|| anyhow::anyhow!("empty argv"))?;
    let s = Command::new(bin).args(rest).current_dir(root).status()
        .with_context(|| format!("spawn {bin}"))?;
    if !s.success() { bail!("{bin} failed"); }
    Ok(())
}

pub async fn run_install(component: &str) -> Result<()> {
    let m = parse()?;
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    resolve(&m, component, &mut order, &mut seen)?;

    let root = workspace_root();
    println!("install plan: {}\n", order.join(" → "));

    for name in order {
        let c = &m.component[name];
        println!("── {name}: {} ──", c.description);

        for step in &c.build { run(root, step)?; }

        for unit in &c.units {
            println!("    $ systemctl --user enable --now {unit}");
            let s = Command::new("systemctl")
                .args(["--user", "enable", "--now", unit])
                .status()?;
            if !s.success() { bail!("systemctl failed for {unit}"); }
        }

        if !c.check.is_empty() {
            print!("    checking {}…  ", c.check);
            // Services can take a moment to bind; retry up to 5s.
            let mut ok = false;
            for _ in 0..5 {
                if healthcheck(&c.check).await { ok = true; break; }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            println!("{}", if ok { "ok" } else { "FAIL (component installed but health probe failed)" });
        }
    }

    println!("\n✓ install complete");
    Ok(())
}
