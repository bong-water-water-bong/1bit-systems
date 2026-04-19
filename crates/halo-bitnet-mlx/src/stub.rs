//! No-op MLX backend. Compiles everywhere, returns NotCompiled on every call.
//! Primary purpose: keep `cargo check --workspace` green on AMD hosts where
//! mlx-rs can't build (Apple-only toolchain).

use crate::{BackendStatus, TernaryBackend};

#[derive(Debug, Default)]
pub struct MlxBackend;

impl TernaryBackend for MlxBackend {
    fn name(&self) -> &'static str { "mlx-apple (stub; rebuild with --features mlx-apple on Apple Silicon)" }
    fn supports_shape(&self, _: usize, _: usize) -> bool { false }
    fn load_h1b(&mut self, _path: &str) -> anyhow::Result<()> {
        anyhow::bail!("halo-bitnet-mlx built without 'mlx-apple' feature; MLX backend is a stub here")
    }
    fn generate(&mut self, _prompt: &[u32], _max_new: usize) -> anyhow::Result<Vec<u32>> {
        anyhow::bail!("halo-bitnet-mlx stub — build with --features mlx-apple on macOS to enable")
    }
}

pub fn status() -> BackendStatus { BackendStatus::NotCompiled }
