//! 1bit-helm — egui/eframe desktop client against 1bit-server.
//!
//! Renamed from halo-gaia on 2026-04-20 to avoid a naming collision with
//! AMD GAIA (amd-gaia.ai). See `docs/wiki/AMD-GAIA-Integration.md`.
//!
//! Config is via env (same keys the TUI scaffold used — callers already rely
//! on these):
//!   HALO_HELM_URL    — server URL (default http://127.0.0.1:8180)
//!   HALO_HELM_MODEL  — model id (default halo-1bit-2b)
//!   HALO_HELM_TOKEN  — optional bearer token
//!
//! Legacy HALO_GAIA_* env vars are honored as fallbacks so existing shell
//! profiles keep working without a scripted rewrite.

use anyhow::Result;
use onebit_helm::{HelmApp, SessionConfig};

fn load_config() -> SessionConfig {
    let url = env_any(&["HALO_HELM_URL", "HALO_GAIA_URL"])
        .unwrap_or_else(|| "http://127.0.0.1:8180".to_string());
    let model = env_any(&["HALO_HELM_MODEL", "HALO_GAIA_MODEL"])
        .unwrap_or_else(|| "1bit-monster-2b".to_string());
    let mut cfg = SessionConfig::new(url, model);
    cfg.bearer = env_any(&["HALO_HELM_TOKEN", "HALO_GAIA_TOKEN"]);
    cfg
}

fn env_any(keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Ok(v) = std::env::var(k) {
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}

fn main() -> Result<()> {
    // eframe spins its own event loop; it does NOT want to live inside a
    // tokio worker. Network calls from the UI will spawn onto a multi-thread
    // runtime handed to the app in a follow-up — not here, to keep the
    // window-open path short.
    let cfg = load_config();

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("1bit-helm")
            .with_inner_size([1024.0, 720.0])
            .with_min_inner_size([640.0, 480.0]),
        ..Default::default()
    };

    eframe::run_native(
        "1bit-helm",
        native_options,
        Box::new(move |cc| Ok(Box::new(HelmApp::from_cc(cc, cfg)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))?;
    Ok(())
}
