//! 1bit-mlx — Apple Silicon ternary backend for 1bit systems.
//!
//! Sibling of `1bit-hip` (AMD gfx1151). Both expose the same `Backend`
//! trait that `1bit-router` consumes; one wraps MLX-via-mlx-rs, the other wraps
//! librocm_cpp.so. Cargo features pick at build time.
//!
//! Build on Apple Silicon:  `cargo build -p 1bit-mlx --features mlx-apple`
//! Build anywhere else:     `cargo build -p 1bit-mlx`  (stub, no engine)

#[cfg(feature = "mlx-apple")]
pub mod apple;

#[cfg(not(feature = "mlx-apple"))]
pub mod stub;

/// Status returned by every halo-ternary backend call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendStatus {
    Ok,
    NotCompiled,
    InvalidArg,
    RuntimeFailed,
}

/// The shape every halo ternary backend must expose. Both -hip and -mlx
/// implement this; 1bit-router picks at dispatch time.
pub trait TernaryBackend {
    fn name(&self) -> &'static str;
    fn supports_shape(&self, n_heads: usize, head_dim: usize) -> bool;
    fn load_h1b(&mut self, path: &str) -> anyhow::Result<()>;
    fn generate(&mut self, prompt_tokens: &[u32], max_new: usize) -> anyhow::Result<Vec<u32>>;
}
