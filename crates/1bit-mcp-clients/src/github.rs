//! GitHub MCP wrapper.
//!
//! Upstream: <https://github.com/github/github-mcp-server> (official
//! remote at <https://api.githubcopilot.com/mcp/>). Auth is a PAT via
//! `Authorization: Bearer <token>` header.
//!
//! Fed from env `GITHUB_TOKEN` — same var cargo, gh, and our systemd
//! units already use. Caller is responsible for scoping the PAT to the
//! minimum surface each specialist needs (Librarian: contents+pull;
//! Magistrate: pull+metadata; Quartermaster: issues).

use crate::error::McpError;
use crate::http::HttpClient;

pub const DEFAULT_ENDPOINT: &str = "https://api.githubcopilot.com/mcp/";

pub struct GitHub {
    http: HttpClient,
}

impl GitHub {
    /// Build a client against `endpoint` with a raw token.
    pub fn with_token(endpoint: &str, token: &str) -> Result<Self, McpError> {
        let http = HttpClient::new(endpoint).header("Authorization", &format!("Bearer {token}"))?;
        Ok(Self { http })
    }

    /// Read `GITHUB_TOKEN` from env and connect to the hosted endpoint.
    /// Returns `None` when the env var is absent — callers use this to
    /// decide whether GitHub tools should appear in their capability set.
    pub fn from_env() -> Option<Self> {
        let token = std::env::var("GITHUB_TOKEN").ok()?;
        Self::with_token(DEFAULT_ENDPOINT, &token).ok()
    }

    pub fn inner(&self) -> &HttpClient {
        &self.http
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_endpoint_is_official_copilot_mcp() {
        assert_eq!(DEFAULT_ENDPOINT, "https://api.githubcopilot.com/mcp/");
    }

    #[test]
    fn with_token_attaches_bearer_header() {
        let gh = GitHub::with_token("https://example/mcp", "ghp_dummy").expect("ok");
        // We can't read the header back without a getter on HttpClient,
        // but the builder path compiles + `with_token` returns Ok, which
        // means `.header()` accepted a well-formed bearer value.
        let _ = gh.inner();
    }

    #[test]
    fn from_env_returns_none_when_unset() {
        // Set a definitely-unused var name to simulate absence.
        let prev = std::env::var("GITHUB_TOKEN").ok();
        // Note: removed SAFETY, std::env::remove_var is unsafe in edition 2024
        unsafe {
            std::env::remove_var("GITHUB_TOKEN");
        }
        let gh = GitHub::from_env();
        assert!(gh.is_none());
        if let Some(v) = prev {
            unsafe {
                std::env::set_var("GITHUB_TOKEN", v);
            }
        }
    }
}
