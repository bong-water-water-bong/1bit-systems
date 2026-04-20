// `halo ppl` — run wikitext PPL against halo-server :8180 /ppl endpoint.

use anyhow::{Context, Result, bail};
use serde_json::json;
use std::time::Duration;

const DEFAULT_URL: &str = "http://127.0.0.1:8180/ppl";
const GEN1_BASELINE: f64 = 9.1607;

pub async fn run(url: Option<String>, stride: u32, max_tokens: u32, bytes: usize) -> Result<()> {
    let url =
        url.unwrap_or_else(|| std::env::var("HALO_PPL_URL").unwrap_or_else(|_| DEFAULT_URL.into()));
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let wikitext_path = std::env::var("HALO_WIKITEXT")
        .unwrap_or_else(|_| format!("{home}/halo-ai/datasets/wikitext-103-test.txt"));

    let text =
        std::fs::read_to_string(&wikitext_path).with_context(|| format!("read {wikitext_path}"))?;
    let text: String = text.chars().take(bytes).collect();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;
    let body = json!({ "text": text, "stride": stride, "max_tokens": max_tokens });

    println!(
        "POST {url}  (stride={stride} max_tokens={max_tokens} {} bytes)",
        text.len()
    );
    let res = client.post(&url).json(&body).send().await?;
    if !res.status().is_success() {
        bail!(
            "ppl endpoint returned {}: {}",
            res.status(),
            res.text().await.unwrap_or_default()
        );
    }
    let v: serde_json::Value = res.json().await?;
    let ppl = v["perplexity"].as_f64().unwrap_or(f64::NAN);
    let mean_nll = v["mean_nll"].as_f64().unwrap_or(f64::NAN);
    let elapsed_ms = v["elapsed_ms"].as_f64().unwrap_or(0.0);
    let tokens = v["tokens"].as_u64().unwrap_or(0);

    println!();
    println!("  mean_nll   = {mean_nll:.6}");
    println!("  perplexity = {ppl:.6}  (gen-1 baseline {GEN1_BASELINE})");
    println!("  tokens     = {tokens}");
    println!("  elapsed_ms = {elapsed_ms:.1}");
    println!("  delta      = {:+.4}", ppl - GEN1_BASELINE);

    let tol = 0.05;
    if (ppl - GEN1_BASELINE).abs() <= tol {
        println!("\n✓ PASS — within ±{tol} of gen-1 baseline");
    } else {
        println!("\n✗ FAIL — outside ±{tol} of gen-1 baseline");
        std::process::exit(1);
    }
    Ok(())
}
