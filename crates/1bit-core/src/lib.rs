//! 1bit-core — CPU-side primitives for gen-2 1bit systems.
//!
//! This crate replaces the glue that used to live in `lemonade-sdk/lemonade`
//! (Python) and the non-kernel half of `rocm-cpp/tools/bitnet_decode.cpp`
//! (C++). It is **CPU only**: kernels / GEMV live in `1bit-hip`.
//!
//! Scope
//! -----
//! * [`h1b`]    — `.h1b` ternary-BitNet model format (parser + writer).
//! * [`gguf`]   — GGUF v3 parser (mmap + metadata KVs + tensor directory).
//!   Drop-in compat with public BitNet GGUFs (e.g.
//!   `microsoft/bitnet-b1.58-2B-4t-gguf`). The top-level
//!   [`gguf::GgufFile`] is parse-only; bit-unpacking from
//!   llama.cpp IQ2_S (the BitNet-compatible 2-bit format)
//!   into halo's 2-bit packed ternary + per-block fp16 scale
//!   lives in [`gguf::unpack`].
//! * [`htok`]   — `.htok` tokenizer file format (parser + writer).
//! * [`sampler`] — host-side logits sampler (temperature, top-k, top-p,
//!   repetition penalty). Ported from `bitnet_decode.cpp`'s
//!   `sample_host` lambda.
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
pub mod gguf;
pub mod h1b;
pub mod htok;
pub mod sampler;
pub mod types;

pub use error::HaloError;
pub use gguf::{
    BitnetHeader, GGUF_MAGIC, GGUF_MIN_VERSION, GgufArray, GgufFile, GgufTensorInfo,
    GgufTensorType, GgufValue, GgufValueType,
};
pub use h1b::{
    H1B_FLAG_HADAMARD_ROTATED, H1B_MAGIC, H1bConfig, H1bFile, H1bLayerOffsets, H1bWeightFormat,
};
pub use htok::{HTOK_MAGIC, HtokFile, Merge};
pub use sampler::{Sampler, SamplerConfig};
pub use types::{MAX_SUPPORTED_VERSION, MIN_SUPPORTED_VERSION, TokenId};
