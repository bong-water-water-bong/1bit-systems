use anyhow::Result;
pub async fn run() -> Result<()> {
    println!(
        "halo {} — strix-ai-rs gen 2 (Rust)",
        env!("CARGO_PKG_VERSION")
    );
    Ok(())
}
