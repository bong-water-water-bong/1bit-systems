//! halo-pkg — plugin package manager for Helm.
//!
//! Implements the client side of the plugin system spec'd in
//! `halo-ai-core/docs/wiki/Helm-Plugin-API.md` (v0.1).
//!
//! The crate is split into three modules:
//!
//! - [`manifest`] — `plugin.toml` schema + parser.
//! - [`registry`] — trait + local-file stub for plugin discovery.
//! - [`store`]    — on-disk layout under
//!   `~/.local/share/halo-pkg/plugins/<name>/`.
//!
//! The `halo-pkg` binary ([`crate::main`](../main/index.html)) is a thin
//! clap dispatcher over these modules. Most concrete implementations are
//! still `todo!()` — this crate is a compile-ready scaffold, not a
//! working package manager yet.

pub mod manifest;
pub mod registry;
pub mod store;
