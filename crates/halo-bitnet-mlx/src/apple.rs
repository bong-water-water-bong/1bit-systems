//! Real MLX backend. Wraps `bitnet-inference` from strix-ai-rs/bitnet-mlx-rs.
//! Stub body for now — fleshed in the next session (needs Apple box to test).

use crate::{BackendStatus, TernaryBackend};

#[derive(Debug, Default)]
pub struct MlxBackend {
    loaded: bool,
}

impl TernaryBackend for MlxBackend {
    fn name(&self) -> &'static str {
        "mlx-apple (native, via bitnet-mlx-rs)"
    }
    fn supports_shape(&self, _: usize, _: usize) -> bool {
        true
    }
    fn load_h1b(&mut self, _path: &str) -> anyhow::Result<()> {
        // TODO: bitnet_inference::load(path)? + convert .h1b ternary packing
        // to whatever mlx-rs expects. bitnet-mlx-rs's bitnet-quant has the
        // weight loaders — we just wire .h1b → their types here.
        self.loaded = true;
        Ok(())
    }
    fn generate(&mut self, _prompt: &[u32], _max_new: usize) -> anyhow::Result<Vec<u32>> {
        if !self.loaded {
            anyhow::bail!("call load_h1b() first");
        }
        // TODO: bitnet_inference::generate(...)
        anyhow::bail!("mlx-apple generate: not wired yet")
    }
}

pub fn status() -> BackendStatus {
    BackendStatus::Ok
}
