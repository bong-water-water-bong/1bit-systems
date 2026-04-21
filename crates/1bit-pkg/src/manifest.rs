//! `plugin.toml` schema.
//!
//! Matches the manifest format documented in
//! `halo-ai-core/docs/wiki/Helm-Plugin-API.md` § 2. Kept deliberately
//! permissive — unknown fields are accepted and ignored so older halo-pkg
//! binaries don't choke on newer manifests.
//!
//! Validation beyond the structural level (regex on `name` / `mount`,
//! existence of `entry`, SPDX sanity check on `license`) lives in
//! `todo!()` stubs below — those will light up once we have real
//! fixtures to test against.

use serde::{Deserialize, Serialize};

/// Root of the `plugin.toml` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub plugin: PluginMeta,

    /// Optional environment overrides, passed into the plugin process.
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, String>,

    /// Tools the plugin advertises. Helm reconciles against `tools/list`
    /// at runtime; this block is a hint for UI + install-time sanity.
    #[serde(default, rename = "tools")]
    pub tools: Vec<ToolDecl>,

    /// Skill files bundled with the plugin (markdown + YAML frontmatter).
    #[serde(default, rename = "skills")]
    pub skills: Vec<SkillDecl>,

    /// Model weights bundled with the plugin.
    #[serde(default, rename = "weights")]
    pub weights: Vec<WeightsDecl>,

    /// Native widget stub. RESERVED — parsed but unused in v0.3.
    #[serde(default)]
    pub widget: Option<WidgetDecl>,
}

/// The mandatory `[plugin]` header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMeta {
    /// Unique plugin identifier. Matches `^[a-z0-9][a-z0-9-]{0,62}$`.
    pub name: String,

    /// Semver 2.0 version string.
    pub version: String,

    /// Plugin kind: one of the variants in [`Kind`].
    pub kind: Kind,

    /// How to launch the plugin. Parsed as an argv-like string; see
    /// § 2.3 of the spec. Optional when `kind = "skill" | "weights"`.
    #[serde(default)]
    pub entry: Option<String>,

    /// Tool-namespace prefix. Matches `^[a-z0-9][a-z0-9-]{0,31}$`.
    pub mount: String,

    /// Helm plugin API version this manifest targets. Must parse as
    /// `<major>.<minor>`.
    pub api: String,

    /// SPDX licence identifier. REQUIRED.
    pub license: String,

    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub homepage: Option<String>,
    #[serde(default)]
    pub authors: Vec<String>,

    /// Capability tokens (see § 2.5). Advisory in v0.3.
    #[serde(default)]
    pub caps: Vec<String>,
}

/// Plugin kinds recognised by the v0.1 spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    McpStdio,
    McpHttp,
    Skill,
    Weights,
    Widget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDecl {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDecl {
    /// Path relative to the plugin root.
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightsDecl {
    pub name: String,
    /// Weight format: `"h1b" | "gguf" | "safetensors"`.
    pub format: String,
    /// Path relative to the plugin root.
    pub path: String,
    /// 64 hex chars. Used by halo-pkg to verify the download.
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetDecl {
    /// Path relative to the plugin root. Reserved for v0.4+.
    pub path: String,
}

impl Manifest {
    /// Parse a `plugin.toml` string.
    pub fn from_toml(src: &str) -> anyhow::Result<Self> {
        let m: Manifest = toml::from_str(src)?;
        Ok(m)
    }

    /// Structural + semantic validation. Regex checks on `name` / `mount`,
    /// kind/entry consistency, SPDX sanity.
    pub fn validate(&self) -> anyhow::Result<()> {
        // TODO: ^[a-z0-9][a-z0-9-]{0,62}$ on name.
        // TODO: ^[a-z0-9][a-z0-9-]{0,31}$ on mount.
        // TODO: entry required when kind ∈ {McpStdio, McpHttp}.
        // TODO: api parses as <major>.<minor> and is within our supported range.
        // TODO: SPDX identifier lookup against the bundled licence list.
        todo!("implement manifest validation")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_mcp_stdio_manifest() {
        let src = r#"
            [plugin]
            name    = "claude-context"
            version = "0.3.1"
            kind    = "mcp-stdio"
            entry   = "node dist/index.js"
            mount   = "context"
            api     = "0.1"
            license = "MIT"
        "#;
        let m = Manifest::from_toml(src).expect("parse");
        assert_eq!(m.plugin.name, "claude-context");
        assert_eq!(m.plugin.kind, Kind::McpStdio);
    }
}
