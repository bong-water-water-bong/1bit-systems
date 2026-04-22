//! Concrete specialist implementations.
//!
//! This module hosts real (non-stub) specialists. Today that's the four
//! LLM-backed ones — [`llm_backed::LlmSpecialist`] — used by Discord
//! triage (`Herald` / `Sentinel` / `Magistrate`) and GitHub event triage
//! (`Quartermaster`). The other 13 names in [`crate::Name`] stay on the
//! `Stub` fallback registered by [`crate::Registry::default_stubs`].
//!
//! Keeping implementations in their own module avoids puffing out
//! `lib.rs` as more specialists come online. Factories here are
//! consumed by [`crate::Registry::default_live`].

pub mod llm_backed;

pub use llm_backed::{LlmSpecialist, default_base_url, default_model_id};
