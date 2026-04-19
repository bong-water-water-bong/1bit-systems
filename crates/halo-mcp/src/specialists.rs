//! The 17 halo-agents specialists exposed as MCP tools.
//!
//! Authoritative source: `/home/bcloud/repos/halo-mcp/src/tool_registry.cpp`
//! (C++ Phase 0 implementation). Keep this list in lock-step with the C++
//! side until the Rust binary fully replaces the C++ one.
//!
//! Each entry becomes ONE MCP tool named `<specialist>_call` in Phase 0.
//! Phase 1 will fan these out into multiple per-capability tools once the
//! bus bridge into `halo-agents` is wired up.
//!
//! Ordering matches `halo-ai-core/docs/wiki/Agents.md`.
//!
//! `is_write` marks specialists that mutate state (Discord posts, GitHub
//! edits, disk writes). CVG / warden consumes this flag in Phase 1 to gate
//! calls behind approval. Phase 0 just echoes it in the tool description
//! metadata for now — the stub handler returns "not implemented"
//! regardless.

/// A Phase-0 specialist descriptor.
#[derive(Debug, Clone, Copy)]
pub struct Specialist {
    /// Specialist name. The MCP tool name is `{name}_call`.
    pub name: &'static str,
    /// One-line job summary — becomes the tool description prefix.
    pub one_liner: &'static str,
    /// True if the specialist mutates external state.
    pub is_write: bool,
}

/// Canonical specialist list — 17 entries.
///
/// Order matches the C++ `kSpecialists` array verbatim. Do not alphabetise;
/// downstream tests (and humans reading `tools/list`) expect this order.
pub const KNOWN: &[Specialist] = &[
    Specialist { name: "muse",          one_liner: "LLM chat via sommelier.",                                                is_write: false },
    Specialist { name: "planner",       one_liner: "ReAct-style goal-to-plan reasoner.",                                     is_write: false },
    Specialist { name: "forge",         one_liner: "Tool dispatcher — runs approved tools after warden CVG clearance.",      is_write: true  },
    Specialist { name: "warden",        one_liner: "CVG 4-check gate (policy/intent/consent/bounds) on every tool call.",    is_write: true  },
    Specialist { name: "cartograph",    one_liner: "Semantic memory — keyword + usearch v2 index.",                          is_write: true  },
    Specialist { name: "scribe",        one_liner: "Hash-chained JSONL session audit log.",                                  is_write: true  },
    Specialist { name: "sommelier",     one_liner: "LLM backend router — local rocm-cpp + 5 paid APIs.",                     is_write: false },
    Specialist { name: "stdout_sink",   one_liner: "Terminal relay for debug output.",                                       is_write: false },
    Specialist { name: "herald",        one_liner: "Discord poster (write side, REST).",                                     is_write: true  },
    Specialist { name: "sentinel",      one_liner: "Discord channel watcher (read side, poll).",                             is_write: false },
    Specialist { name: "carpenter",     one_liner: "Install-help via regex + LLM fallback in a dedicated Discord channel.",  is_write: true  },
    Specialist { name: "quartermaster", one_liner: "GitHub issue triage and labelling.",                                     is_write: true  },
    Specialist { name: "magistrate",    one_liner: "GitHub PR policy scanner.",                                              is_write: false },
    Specialist { name: "librarian",     one_liner: "CHANGELOG appender + docs-gap issue filer.",                             is_write: true  },
    Specialist { name: "echo_ear",      one_liner: "whisper-server STT bridge.",                                             is_write: false },
    Specialist { name: "echo_mouth",    one_liner: "kokoro TTS bridge.",                                                     is_write: true  },
    Specialist { name: "anvil",         one_liner: "Clone, build, and benchmark a repo end-to-end.",                         is_write: true  },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_seventeen_specialists() {
        assert_eq!(KNOWN.len(), 17);
    }

    #[test]
    fn names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for s in KNOWN {
            assert!(seen.insert(s.name), "duplicate specialist: {}", s.name);
        }
    }

    #[test]
    fn names_match_cpp_order() {
        // Mirror check against the C++ kSpecialists order.
        let expected = [
            "muse", "planner", "forge", "warden", "cartograph", "scribe",
            "sommelier", "stdout_sink", "herald", "sentinel", "carpenter",
            "quartermaster", "magistrate", "librarian", "echo_ear",
            "echo_mouth", "anvil",
        ];
        let actual: Vec<&str> = KNOWN.iter().map(|s| s.name).collect();
        assert_eq!(actual, expected);
    }
}
