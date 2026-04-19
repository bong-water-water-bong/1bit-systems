//! halo-core — CPU-side primitives for gen-2 halo-ai.
//!
//! This crate replaces the glue that used to live in `lemonade-sdk/lemonade`
//! (Python) and the non-kernel half of `rocm-cpp/tools/bitnet_decode.cpp`
//! (C++). It is **CPU only**: kernels / GEMV live in `halo-bitnet-hip`.
//!
//! Scope
//! -----
//! * [`h1b`]    — `.h1b` ternary-BitNet model format (parser + writer).
//! * [`htok`]   — `.htok` tokenizer file format (parser + writer).
//! * [`sampler`] — host-side logits sampler (temperature, top-k, top-p,
//!                 repetition penalty). Ported from `bitnet_decode.cpp`'s
//!                 `sample_host` lambda.
//! * [`error`]  — unified [`HaloError`] (via `thiserror`).
//! * [`types`]  — shared small types (ids, dtype tags, helpers).
//!
//! The loaders use [`memmap2`] so weight blobs are never fully copied into
//! heap; downstream crates get zero-copy byte slices they can hand to HIP.
//!
//! Safety
//! ------
//! The only `unsafe` in this crate is the narrow memory-mapping wrapper in
//! [`h1b::Mapped`]; every other public API is safe Rust.

pub mod error;
pub mod h1b;
pub mod htok;
pub mod sampler;
pub mod types;

pub use error::HaloError;
pub use h1b::{H1bConfig, H1bFile, H1bLayerOffsets, H1bWeightFormat, H1B_MAGIC};
pub use htok::{HtokFile, Merge, HTOK_MAGIC};
pub use sampler::{Sampler, SamplerConfig};
pub use types::{TokenId, MAX_SUPPORTED_VERSION, MIN_SUPPORTED_VERSION};
