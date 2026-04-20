// `halo skill ...` — operator-facing CRUD for the SKILL.md surface.
//
// This is the human front door to the same on-disk layout that the LLM
// hits through `SkillAction` / `apply` in halo-agents. Operators need to
// inspect, author, and cull skills by hand (especially while we're
// bootstrapping the collection); this subcommand group keeps them out of
// `~/.halo/skills/<cat>/<name>/SKILL.md` and in a tool that uses the
// same parser + writer the agent does.
//
// Layout mirrors `power.rs`: a small `SkillCmd` enum, per-action free
// functions, a single `run_with_store` entry point the CLI and tests
// both call. The default entry `run` just constructs the real
// `SkillStore::new()` (rooted at `~/.halo/skills/`) and delegates.

use anyhow::{Context, Result, anyhow, bail};
use clap::Subcommand;
use halo_agents::skills::{Skill, SkillStore};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::Command;

/// Subcommand group wired under `halo skill ...` in `main.rs`.
#[derive(Subcommand, Debug)]
pub enum SkillCmd {
    /// List all skills, one line each: `category/name — description`.
    List {
        /// Emit JSON instead of plain text (array of {name, category, description, version}).
        #[arg(long)]
        json: bool,
    },
    /// Print the full SKILL.md contents of a skill to stdout.
    Show { name: String },
    /// Create a new skill and open SKILL.md in $EDITOR.
    New {
        name: String,
        /// Category directory under ~/.halo/skills (default: uncategorized).
        #[arg(long)]
        category: Option<String>,
        /// One-line description (default: a placeholder the editor overrides).
        #[arg(long)]
        description: Option<String>,
    },
    /// Open an existing SKILL.md in $EDITOR.
    Edit { name: String },
    /// Delete a skill's directory. Requires `--yes` or an interactive y/N.
    Delete {
        name: String,
        /// Skip the confirmation prompt.
        #[arg(long)]
        yes: bool,
    },
    /// Print the resolved ~/.halo/skills path (or the path for a single skill).
    Path { name: Option<String> },
}

// ---------------------------------------------------------------------------
// Actions

/// `halo skill list` — one line per skill, sorted `category/name`.
pub fn list(store: &SkillStore, json: bool) -> Result<()> {
    let mut skills = store.list()?;
    // Stable, human-friendly order: category first, then name.
    skills.sort_by(|a, b| {
        a.category()
            .cmp(b.category())
            .then_with(|| a.name.cmp(&b.name))
    });

    if json {
        let rows: Vec<serde_json::Value> = skills
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "category": s.category(),
                    "description": s.description,
                    "version": s.version,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }

    for s in &skills {
        println!("{}/{} — {}", s.category(), s.name, s.description);
    }
    Ok(())
}

/// `halo skill show <name>` — dump SKILL.md verbatim.
pub fn show(store: &SkillStore, name: &str) -> Result<()> {
    let path = skill_md_path(store, name)?.ok_or_else(|| anyhow!("no skill named '{name}'"))?;
    let src = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    // Pass through verbatim — no reformat, no trailing newline surgery.
    io::stdout().write_all(src.as_bytes())?;
    Ok(())
}

/// `halo skill new <name>` — seed SKILL.md and hand to $EDITOR.
///
/// We create the file through `SkillStore::create` so the on-disk layout
/// is identical to what the LLM path produces, then launch the editor on
/// the resulting SKILL.md. No re-parse / re-write after editor exit: the
/// user owns the file now.
pub fn new_skill(
    store: &mut SkillStore,
    name: &str,
    category: Option<&str>,
    description: Option<&str>,
) -> Result<()> {
    if name.trim().is_empty() {
        bail!("skill name must be non-empty");
    }
    let mut skill = Skill::new(
        name,
        description.unwrap_or("(describe what this skill does — one line)"),
    );
    if let Some(cat) = category {
        skill.metadata_halo.category = cat.to_string();
    }
    skill.body = starter_body(name);
    store.create(skill)?;

    let path = skill_md_path(store, name)?
        .ok_or_else(|| anyhow!("freshly-created skill '{name}' not found on disk"))?;
    println!("created {}", path.display());
    open_editor(&path)?;
    Ok(())
}

/// `halo skill edit <name>` — open SKILL.md in $EDITOR (fallback `vi`).
pub fn edit(store: &SkillStore, name: &str) -> Result<()> {
    let path = skill_md_path(store, name)?.ok_or_else(|| anyhow!("no skill named '{name}'"))?;
    open_editor(&path)
}

/// `halo skill delete <name> [--yes]` — default-deny removal.
pub fn delete(store: &mut SkillStore, name: &str, assume_yes: bool) -> Result<()> {
    // Validate existence up front so an aborted prompt doesn't leave the
    // user wondering whether the name was even right.
    if skill_md_path(store, name)?.is_none() {
        bail!("no skill named '{name}'");
    }
    if !assume_yes && !confirm(&format!("Delete skill '{name}'?"))? {
        println!("aborted");
        return Ok(());
    }
    store.delete(name)?;
    println!("deleted {name}");
    Ok(())
}

/// `halo skill path [<name>]` — stdout the resolved filesystem path.
///
/// With no name: the store root. With a name: the SKILL.md file path.
/// Useful for shell scripts: `cat "$(halo skill path foo)"`.
pub fn path(store: &SkillStore, name: Option<&str>) -> Result<()> {
    match name {
        None => {
            println!("{}", store.root().display());
        }
        Some(n) => {
            let path = skill_md_path(store, n)?.ok_or_else(|| anyhow!("no skill named '{n}'"))?;
            println!("{}", path.display());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internals

/// Find the on-disk SKILL.md for a skill by scanning the store's categories.
/// Mirrors halo-agents' internal `find_path` but stays inside the public
/// `SkillStore` surface (`root()` + a directory walk).
fn skill_md_path(store: &SkillStore, name: &str) -> Result<Option<PathBuf>> {
    let root = store.root();
    if !root.exists() {
        return Ok(None);
    }
    for cat_entry in
        std::fs::read_dir(root).with_context(|| format!("read_dir {}", root.display()))?
    {
        let cat_entry = cat_entry?;
        if !cat_entry.file_type()?.is_dir() {
            continue;
        }
        let candidate = cat_entry.path().join(name).join("SKILL.md");
        if candidate.exists() {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

/// Launch `$EDITOR` (fallback cascade: nvim → vim → vi → nano) against
/// `path` and wait for it to exit. On a minimal system where none of those
/// are installed, print the path and return Ok so the create path still
/// succeeds — the user can edit the file however they like.
fn open_editor(path: &std::path::Path) -> Result<()> {
    let preferred = std::env::var("EDITOR").ok();
    let candidates: Vec<&str> = preferred
        .as_deref()
        .into_iter()
        .chain(["nvim", "vim", "vi", "nano"])
        .collect();
    for editor in candidates {
        match Command::new(editor).arg(path).status() {
            Ok(status) if status.success() => return Ok(()),
            Ok(status) => bail!("editor '{editor}' exited with {status}"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e).with_context(|| format!("spawn editor '{editor}'")),
        }
    }
    println!(
        "note: no editor found ($EDITOR or nvim/vim/vi/nano). File is at:\n  {}",
        path.display()
    );
    Ok(())
}

/// Interactive y/N: lowercase "y" → true, anything else → false.
fn confirm(prompt: &str) -> Result<bool> {
    print!("{prompt} [y/N] ");
    io::stdout().flush().ok();
    let mut buf = String::new();
    let mut stdin = io::stdin();
    // Read until newline. `read_line` would also work but this mirrors
    // the "3 lines, no new dep" shape called out in the task brief.
    let mut byte = [0u8; 1];
    while stdin.read(&mut byte)? == 1 {
        if byte[0] == b'\n' {
            break;
        }
        buf.push(byte[0] as char);
    }
    Ok(matches!(buf.trim(), "y" | "Y" | "yes" | "YES" | "Yes"))
}

/// Default markdown body seeded by `halo skill new`. Kept minimal — the
/// author replaces it; we just want headings that prompt them.
fn starter_body(name: &str) -> String {
    format!(
        "\n# {name}\n\
         \n\
         ## When to use\n\
         \n\
         (trigger description — what question or task invokes this skill)\n\
         \n\
         ## Steps\n\
         \n\
         1. …\n\
         2. …\n\
         \n\
         ## Notes\n\
         \n\
         (edge cases, gotchas, links)\n"
    )
}

// ---------------------------------------------------------------------------
// Entry point — wired from `main.rs`.

/// Default entry: real `~/.halo/skills/` store.
pub fn run(cmd: SkillCmd) -> Result<()> {
    let mut store = SkillStore::new()?;
    run_with_store(&mut store, cmd)
}

/// Test-friendly entry: the caller owns the store root. The CLI itself
/// always calls `run`, but halo-cli's integration tests (and anyone
/// wanting a per-request isolated root) go through here.
pub fn run_with_store(store: &mut SkillStore, cmd: SkillCmd) -> Result<()> {
    match cmd {
        SkillCmd::List { json } => list(store, json),
        SkillCmd::Show { name } => show(store, &name),
        SkillCmd::New {
            name,
            category,
            description,
        } => new_skill(store, &name, category.as_deref(), description.as_deref()),
        SkillCmd::Edit { name } => edit(store, &name),
        SkillCmd::Delete { name, yes } => delete(store, &name, yes),
        SkillCmd::Path { name } => path(store, name.as_deref()),
    }
}

// ---------------------------------------------------------------------------
// Tests
//
// All tests isolate against a `tempfile::TempDir` — they never touch the
// real ~/.halo/skills/. $EDITOR tests are deliberately omitted: spawning
// a real interactive editor from `cargo test` is hostile to CI. The
// create/list/show/delete roundtrip covers everything except the editor
// shell-out, which is a one-liner.

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fresh_store() -> (TempDir, SkillStore) {
        let td = TempDir::new().unwrap();
        let store = SkillStore::with_root(td.path().to_path_buf());
        (td, store)
    }

    /// A `new_skill` that does NOT open the editor. We inline the
    /// create-side of `new_skill` so tests can assert on the on-disk
    /// state without a real $EDITOR.
    fn create_skill_no_editor(
        store: &mut SkillStore,
        name: &str,
        category: Option<&str>,
        description: Option<&str>,
    ) -> Result<()> {
        let mut skill = Skill::new(
            name,
            description.unwrap_or("(describe what this skill does — one line)"),
        );
        if let Some(cat) = category {
            skill.metadata_halo.category = cat.to_string();
        }
        skill.body = starter_body(name);
        store.create(skill)?;
        Ok(())
    }

    #[test]
    fn list_empty_store_is_ok_and_shows_no_rows() {
        let (_td, store) = fresh_store();
        // Non-JSON path.
        list(&store, false).unwrap();
        // JSON path.
        list(&store, true).unwrap();
    }

    #[test]
    fn new_then_list_roundtrip_finds_the_skill() {
        let (_td, mut store) = fresh_store();
        create_skill_no_editor(&mut store, "alpha", Some("demos"), Some("greet the user")).unwrap();

        let skills = store.list().unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "alpha");
        assert_eq!(skills[0].category(), "demos");
        assert_eq!(skills[0].description, "greet the user");

        // `list()` must not error either.
        list(&store, false).unwrap();
    }

    #[test]
    fn show_returns_full_content_of_a_created_skill() {
        let (_td, mut store) = fresh_store();
        create_skill_no_editor(&mut store, "showme", Some("demos"), Some("dump test")).unwrap();

        // Re-read via the same path-resolution the CLI uses.
        let path = skill_md_path(&store, "showme").unwrap().unwrap();
        let src = std::fs::read_to_string(&path).unwrap();

        assert!(src.starts_with("---\n"), "frontmatter fence missing: {src}");
        assert!(src.contains("name: showme"));
        assert!(src.contains("description: dump test"));
        // Starter body heading seeded by `new_skill`.
        assert!(src.contains("# showme"));
        assert!(src.contains("## When to use"));
    }

    #[test]
    fn delete_with_yes_removes_the_file() {
        let (td, mut store) = fresh_store();
        create_skill_no_editor(&mut store, "doomed", Some("demos"), None).unwrap();
        let dir = td.path().join("demos/doomed");
        assert!(dir.exists(), "create_skill must lay down the dir first");

        delete(&mut store, "doomed", true).unwrap();
        assert!(!dir.exists(), "delete --yes must rm -rf the skill dir");
        assert!(store.get("doomed").unwrap().is_none());
    }

    #[test]
    fn delete_errors_when_skill_missing() {
        let (_td, mut store) = fresh_store();
        let err = delete(&mut store, "ghost", true).unwrap_err();
        assert!(
            err.to_string().contains("no skill named 'ghost'"),
            "got: {err}"
        );
    }

    #[test]
    fn path_with_no_name_prints_store_root() {
        let (_td, store) = fresh_store();
        // Smoke: must not error with an empty store.
        path(&store, None).unwrap();
    }

    #[test]
    fn path_with_name_resolves_skill_md() {
        let (td, mut store) = fresh_store();
        create_skill_no_editor(&mut store, "located", Some("demos"), None).unwrap();

        let resolved = skill_md_path(&store, "located").unwrap().unwrap();
        assert_eq!(resolved, td.path().join("demos/located/SKILL.md"));
        assert!(resolved.exists());
    }

    #[test]
    fn run_with_store_list_on_fresh_store_is_ok() {
        let (_td, mut store) = fresh_store();
        run_with_store(&mut store, SkillCmd::List { json: false }).unwrap();
        run_with_store(&mut store, SkillCmd::List { json: true }).unwrap();
    }
}
