//! SKILL.md file format: YAML frontmatter + markdown body.
//!
//! The schema mirrors Hermes' `SKILL.md` 1:1 so skills authored on either
//! platform round-trip unchanged. Only difference: the `metadata.*` key is
//! `halo` rather than `hermes`. That single key rename is enough for
//! agentskills.io tooling to treat us as a peer.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

/// Halo-specific metadata nested under `metadata.halo` in the frontmatter.
///
/// Fields are all `Vec<String>` or `String` to keep serialization lossless:
/// deserialize → reserialize is a byte-stable roundtrip modulo YAML
/// whitespace. Missing lists default to empty.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Metadata {
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub fallback_for_toolsets: Vec<String>,
    #[serde(default)]
    pub requires_toolsets: Vec<String>,
}

/// Serde shim for the `metadata:` block so we can write `metadata.halo.*`
/// without inventing a custom (de)serializer.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct MetadataWrapper {
    #[serde(default)]
    halo: Metadata,
}

/// Frontmatter block of a SKILL.md. Internal only — callers interact with
/// [`Skill`], which hoists these fields to the top level.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Frontmatter {
    pub name: String,
    pub description: String,
    pub version: String,
    #[serde(default)]
    pub platforms: Vec<String>,
    #[serde(default)]
    pub metadata: MetadataWrapper,
}

/// A parsed SKILL.md: frontmatter fields hoisted to the top level plus the
/// markdown body as an opaque string.
///
/// # Round-trip
///
/// [`Skill::parse`] + [`Skill::render`] are not byte-exact (YAML key order
/// and whitespace may normalize). They **are** semantically lossless: a
/// re-parse of `render()` yields `PartialEq`-equal `Skill`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub version: String,
    pub platforms: Vec<String>,
    pub metadata_halo: Metadata,
    /// Markdown body (everything after the second `---` line). Leading
    /// whitespace is preserved verbatim.
    pub body: String,
}

impl Skill {
    /// Build a skill from component fields with an empty body / default
    /// metadata. Convenience for tests + the `SkillAction::Create` path.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            version: "0.1.0".to_string(),
            platforms: vec!["linux".to_string()],
            metadata_halo: Metadata::default(),
            body: String::new(),
        }
    }

    /// Parse a SKILL.md string: expects a YAML frontmatter block delimited
    /// by `---` on its own line, then arbitrary markdown.
    ///
    /// Accepts (and preserves) leading blank lines before the opening `---`
    /// because some editors add them.
    pub fn parse(src: &str) -> Result<Self> {
        let trimmed = src.trim_start_matches(|c: char| c == '\u{feff}' || c.is_whitespace());
        let after_open = trimmed
            .strip_prefix("---\n")
            .or_else(|| trimmed.strip_prefix("---\r\n"))
            .ok_or_else(|| anyhow!("SKILL.md missing opening '---' frontmatter fence"))?;

        // Find the closing `---` on its own line.
        let close_idx = after_open
            .split_inclusive('\n')
            .scan(0usize, |acc, line| {
                let start = *acc;
                *acc += line.len();
                Some((start, line))
            })
            .find(|(_, line)| {
                let l = line.trim_end_matches(['\r', '\n']);
                l == "---"
            })
            .map(|(start, _)| start)
            .ok_or_else(|| anyhow!("SKILL.md missing closing '---' frontmatter fence"))?;

        let yaml = &after_open[..close_idx];
        // Advance past the closing fence line itself.
        let rest = &after_open[close_idx..];
        let body = rest.split_once('\n').map(|(_, after)| after).unwrap_or("");

        let fm: Frontmatter =
            serde_yaml_ng::from_str(yaml).context("failed to parse SKILL.md YAML frontmatter")?;

        if fm.name.is_empty() {
            bail!("SKILL.md frontmatter missing required field 'name'");
        }

        Ok(Skill {
            name: fm.name,
            description: fm.description,
            version: fm.version,
            platforms: fm.platforms,
            metadata_halo: fm.metadata.halo,
            body: body.to_string(),
        })
    }

    /// Render a skill back to a SKILL.md string.
    pub fn render(&self) -> Result<String> {
        let fm = Frontmatter {
            name: self.name.clone(),
            description: self.description.clone(),
            version: self.version.clone(),
            platforms: self.platforms.clone(),
            metadata: MetadataWrapper {
                halo: self.metadata_halo.clone(),
            },
        };
        let yaml =
            serde_yaml_ng::to_string(&fm).context("failed to serialize SKILL.md frontmatter")?;
        let mut out = String::with_capacity(yaml.len() + self.body.len() + 16);
        out.push_str("---\n");
        out.push_str(&yaml);
        if !yaml.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("---\n");
        out.push_str(&self.body);
        Ok(out)
    }

    /// The on-disk category this skill lives under.
    pub fn category(&self) -> &str {
        if self.metadata_halo.category.is_empty() {
            "uncategorized"
        } else {
            &self.metadata_halo.category
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Skill {
        Skill {
            name: "hello".into(),
            description: "greet the user".into(),
            version: "1.0.0".into(),
            platforms: vec!["linux".into(), "macos".into()],
            metadata_halo: Metadata {
                tags: vec!["demo".into(), "starter".into()],
                category: "examples".into(),
                fallback_for_toolsets: vec!["web".into()],
                requires_toolsets: vec!["terminal".into()],
            },
            body: "\n# Hello\n\nBody text.\n".into(),
        }
    }

    #[test]
    fn frontmatter_roundtrip_preserves_all_fields() {
        let s = sample();
        let rendered = s.render().unwrap();
        // Sanity: rendered must include the fence + a recognizable key.
        assert!(rendered.starts_with("---\n"));
        assert!(rendered.contains("name: hello"));
        assert!(rendered.contains("halo:"));
        let parsed = Skill::parse(&rendered).unwrap();
        assert_eq!(parsed, s);
    }

    #[test]
    fn parse_accepts_minimal_frontmatter() {
        let src = "---\nname: x\ndescription: y\nversion: 0.1.0\n---\nbody\n";
        let s = Skill::parse(src).unwrap();
        assert_eq!(s.name, "x");
        assert_eq!(s.description, "y");
        assert_eq!(s.version, "0.1.0");
        assert!(s.platforms.is_empty());
        assert_eq!(s.body, "body\n");
    }

    #[test]
    fn parse_rejects_missing_open_fence() {
        let err = Skill::parse("name: x\n").unwrap_err();
        assert!(err.to_string().contains("opening"));
    }

    #[test]
    fn parse_rejects_missing_close_fence() {
        let err = Skill::parse("---\nname: x\n").unwrap_err();
        assert!(err.to_string().contains("closing"));
    }

    #[test]
    fn category_falls_back_to_uncategorized() {
        let mut s = sample();
        s.metadata_halo.category.clear();
        assert_eq!(s.category(), "uncategorized");
    }
}
