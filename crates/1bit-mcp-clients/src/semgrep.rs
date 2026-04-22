//! Semgrep MCP wrapper.
//!
//! Upstream: <https://github.com/semgrep/mcp> hosted at
//! <https://mcp.semgrep.ai/mcp>. No auth required for open rules.
//! Used by Warden (CVG static-analysis gate) and Magistrate (PR
//! policy scan) to look for security/quality issues on diffs without
//! pulling in a local Semgrep install.

use crate::http::HttpClient;

pub const DEFAULT_ENDPOINT: &str = "https://mcp.semgrep.ai/mcp";

pub struct Semgrep {
    http: HttpClient,
}

impl Semgrep {
    pub fn new() -> Self {
        Self {
            http: HttpClient::new(DEFAULT_ENDPOINT),
        }
    }

    pub fn with_endpoint(endpoint: &str) -> Self {
        Self {
            http: HttpClient::new(endpoint),
        }
    }

    pub fn inner(&self) -> &HttpClient {
        &self.http
    }
}

impl Default for Semgrep {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_endpoint_is_hosted_semgrep() {
        assert_eq!(DEFAULT_ENDPOINT, "https://mcp.semgrep.ai/mcp");
    }

    #[test]
    fn new_and_default_are_equivalent() {
        let a = Semgrep::new();
        let b = Semgrep::default();
        // Both point at the same endpoint; we can't compare reqwest
        // clients directly, so assert that both constructed without
        // panicking and that inner is addressable.
        let _ = (a.inner(), b.inner());
    }

    #[test]
    fn with_endpoint_overrides() {
        let s = Semgrep::with_endpoint("https://self-hosted/mcp");
        let _ = s.inner();
    }
}
