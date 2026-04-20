//! Per-specialist MCP metadata (just descriptions, for now).
//!
//! The canonical list of specialists lives in `onebit_agents::Name::ALL` —
//! this module only provides short human-readable descriptions that get
//! surfaced to MCP clients via `tools/list`. Keeping the list in
//! `1bit-agents` guarantees the MCP bridge and the dispatch registry
//! stay in lock-step: any specialist added to `Name` shows up here
//! automatically with a sensible fallback blurb.

use onebit_agents::Name;

/// One-line description per specialist. Shown in `tools/list.description`.
///
/// Unknown `Name` variants fall back to a generic "halo specialist" line —
/// this keeps us building if `1bit-agents` grows the enum before we update
/// this table.
pub fn description_for(n: Name) -> &'static str {
    match n {
        Name::Anvil => "anvil — clone, build, and benchmark a repo end-to-end.",
        Name::Carpenter => "carpenter — install-help regex + LLM fallback in Discord.",
        Name::Cartograph => "cartograph — semantic memory index (keyword + usearch v2).",
        Name::EchoEar => "echo_ear — whisper-server STT bridge.",
        Name::EchoMouth => "echo_mouth — kokoro TTS bridge.",
        Name::Forge => "forge — tool dispatcher, runs approved tools after CVG.",
        Name::Gateway => "gateway — external-surface entry point for requests.",
        Name::Herald => "herald — Discord poster (write side, REST).",
        Name::Librarian => "librarian — CHANGELOG appender + docs-gap issue filer.",
        Name::Magistrate => "magistrate — GitHub PR policy scanner.",
        Name::Muse => "muse — LLM chat via sommelier.",
        Name::Planner => "planner — ReAct-style goal-to-plan reasoner.",
        Name::Quartermaster => "quartermaster — GitHub issue triage and labelling.",
        Name::Scribe => "scribe — hash-chained JSONL session audit log.",
        Name::Sentinel => "sentinel — Discord channel watcher (read side, poll).",
        Name::Sommelier => "sommelier — LLM backend router (local rocm-cpp + paid APIs).",
        Name::Warden => "warden — CVG 4-check gate on every tool call.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_has_description() {
        for n in Name::ALL {
            let d = description_for(*n);
            assert!(!d.is_empty(), "empty description for {:?}", n);
            // Descriptions should lead with the specialist's own name so
            // MCP clients can grep the tool list for a human label.
            assert!(
                d.starts_with(n.as_str()),
                "description for {:?} should start with '{}': got {:?}",
                n,
                n.as_str(),
                d
            );
        }
    }

    #[test]
    fn descriptions_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for n in Name::ALL {
            let d = description_for(*n);
            assert!(seen.insert(d), "duplicate description for {:?}: {}", n, d);
        }
    }
}
