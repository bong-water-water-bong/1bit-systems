//! Dispatch decision: run locally on a halo-router backend, or proxy upstream.
//!
//! `Local` holds a module-path `String` for now to avoid a hard dep on
//! halo-router. The router crate binds these strings to concrete backend
//! handles once the trait stabilises.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Dispatch {
    Local { router: String },
    Upstream { url: String },
}

impl Dispatch {
    pub fn is_local(&self) -> bool {
        matches!(self, Dispatch::Local { .. })
    }
    pub fn is_upstream(&self) -> bool {
        matches!(self, Dispatch::Upstream { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_variants_match() {
        let l = Dispatch::Local {
            router: "halo_router::bitnet".into(),
        };
        let u = Dispatch::Upstream {
            url: "https://api.openai.com".into(),
        };
        assert!(l.is_local() && !l.is_upstream());
        assert!(u.is_upstream() && !u.is_local());
        match &l {
            Dispatch::Local { router } => assert_eq!(router, "halo_router::bitnet"),
            Dispatch::Upstream { .. } => unreachable!(),
        }
        match &u {
            Dispatch::Upstream { url } => assert!(url.starts_with("https://")),
            Dispatch::Local { .. } => unreachable!(),
        }
    }
}
