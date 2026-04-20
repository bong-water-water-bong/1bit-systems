//! watch — shared helpers for the halo-watch-* binaries.
//!
//! Two binaries live next to this module:
//!
//! * `halo-watch-github` — polls configured repos for new issues + PRs and
//!   routes each event to a specialist via the [`Registry`](crate::Registry).
//! * `halo-watch-discord` — gateway-resident Discord lurker (owned by a
//!   separate module, added by a sibling patch).
//!
//! Shared concerns (classification, env parsing, Registry dispatch) live
//! here so both binaries stay thin. Each binary owns its transport.

pub mod github;
</content>
</invoke>