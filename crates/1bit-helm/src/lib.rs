//! 1bit-helm — native desktop client for 1bit-server.
//!
//! Renamed from `halo-gaia` on 2026-04-20 to avoid clashing with AMD GAIA
//! (amd-gaia.ai) — see `docs/wiki/AMD-GAIA-Integration.md`. Stack choice is
//! egui + eframe (glow backend). Pure Rust, single ELF, no Electron, no
//! web-view.
//!
//! Layout:
//! * transport modules (`client`, `conversation`, `session`, `stream`) —
//!   reused from the scaffold; OpenAI-compat chat client with SSE stream
//!   parser. No UI dependency.
//! * `app` — eframe `App` impl. Five panes (Status, Chat, Skills, Memory,
//!   Models) behind a top-bar switcher + left nav panel.

pub mod app;
pub mod client;
pub mod conversation;
pub mod session;
pub mod stream;

pub use app::{HelmApp, Pane};
pub use client::HelmClient;
pub use conversation::{ChatTurn, Conversation, Role};
pub use session::SessionConfig;
pub use stream::{SseEvent, parse_sse_line};
