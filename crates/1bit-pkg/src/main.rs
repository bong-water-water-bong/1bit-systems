// halo-pkg — plugin package manager CLI.
//
// Spec: halo-ai-core/docs/wiki/Helm-Plugin-API.md (v0.1).
// Implementation status: scaffold only. Every non-trivial path is todo!().

use anyhow::Result;
use clap::{Parser, Subcommand};

/// halo-pkg — install, list, update, remove Helm plugins.
#[derive(Parser, Debug)]
#[command(name = "halo-pkg", about, version, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Install a plugin by name (optionally pinned to a version).
    Install {
        /// Plugin name, optionally suffixed with `@<version>`.
        name: String,
    },
    /// Remove an installed plugin. Must be disabled first.
    Remove { name: String },
    /// List installed plugins, one per line.
    List,
    /// Search the active registry for plugins matching `query`.
    Search { query: String },
    /// Refresh the local registry snapshot, or update a specific plugin.
    Update {
        /// Optional plugin name; omit to refresh the registry only.
        name: Option<String>,
    },
    /// Print the manifest + state for an installed plugin.
    Info { name: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Install { name } => {
            println!("halo-pkg: install {name} — not implemented yet");
            todo!("resolve via registry, fetch tarball, verify sha256, extract into store")
        }
        Cmd::Remove { name } => {
            println!("halo-pkg: remove {name} — not implemented yet");
            todo!("ensure disabled, rm -rf plugin dir, drop state entry")
        }
        Cmd::List => {
            println!("halo-pkg: list — not implemented yet");
            todo!("enumerate store::installed(), print `<name> <version> <kind> <state>`")
        }
        Cmd::Search { query } => {
            println!("halo-pkg: search {query} — not implemented yet");
            todo!("Registry::search(&query) + pretty-print top N")
        }
        Cmd::Update { name } => match name {
            Some(n) => {
                println!("halo-pkg: update {n} — not implemented yet");
                todo!("resolve latest compatible, side-by-side install, flip current symlink")
            }
            None => {
                println!("halo-pkg: update registry — not implemented yet");
                todo!("Registry::refresh()")
            }
        },
        Cmd::Info { name } => {
            println!("halo-pkg: info {name} — not implemented yet");
            todo!("load manifest + state, dump as human-readable TOML")
        }
    }
}
