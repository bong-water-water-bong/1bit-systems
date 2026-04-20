//! `halo memory {list,show,add,replace,remove}` — operator UX for
//! MEMORY.md / USER.md at `~/.halo/memories/`. Thin shell over
//! [`onebit_agents::MemoryStore`].

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};
use onebit_agents::{MemoryKind, MemoryStore};
use std::path::PathBuf;

#[derive(Subcommand, Debug)]
pub enum MemoryCmd {
    /// Print every entry in the target file, one per line.
    List(KindFlag),
    /// Dump the raw file body (entries joined by § delimiter).
    Show(KindFlag),
    /// Append an entry.
    Add {
        #[command(flatten)]
        kind: KindFlag,
        /// Entry text (quote if it contains spaces).
        entry: String,
    },
    /// Replace the first entry matching `needle` with `entry`.
    Replace {
        #[command(flatten)]
        kind: KindFlag,
        /// Substring to locate.
        needle: String,
        /// Replacement entry (full text).
        entry: String,
    },
    /// Remove the first entry matching `needle`.
    Remove {
        #[command(flatten)]
        kind: KindFlag,
        /// Substring to locate.
        needle: String,
    },
    /// Print resolved file path for the target kind.
    Path(KindFlag),
}

#[derive(Args, Debug, Clone, Copy)]
pub struct KindFlag {
    /// Which file to target (default memory).
    #[arg(long, value_enum, default_value_t = KindArg::Memory)]
    pub kind: KindArg,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum KindArg {
    Memory,
    User,
}

impl From<KindArg> for MemoryKind {
    fn from(v: KindArg) -> Self {
        match v {
            KindArg::Memory => MemoryKind::Memory,
            KindArg::User => MemoryKind::User,
        }
    }
}

pub fn run(cmd: MemoryCmd) -> Result<()> {
    let store = MemoryStore::new()?;
    run_with_store(&store, cmd)
}

pub fn run_with_store(store: &MemoryStore, cmd: MemoryCmd) -> Result<()> {
    match cmd {
        MemoryCmd::List(k) => {
            for e in store.list(k.kind.into())? {
                println!("{e}");
            }
            Ok(())
        }
        MemoryCmd::Show(k) => {
            print!("{}", store.snapshot(k.kind.into())?);
            Ok(())
        }
        MemoryCmd::Add { kind, entry } => {
            store.add(kind.kind.into(), &entry)?;
            println!("added to {}", MemoryKind::from(kind.kind).filename());
            Ok(())
        }
        MemoryCmd::Replace {
            kind,
            needle,
            entry,
        } => {
            store.replace(kind.kind.into(), &needle, &entry)?;
            println!("replaced in {}", MemoryKind::from(kind.kind).filename());
            Ok(())
        }
        MemoryCmd::Remove { kind, needle } => {
            store.remove(kind.kind.into(), &needle)?;
            println!("removed from {}", MemoryKind::from(kind.kind).filename());
            Ok(())
        }
        MemoryCmd::Path(k) => {
            let filename = MemoryKind::from(k.kind).filename();
            let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("no home dir"))?;
            let path: PathBuf = home.join(".halo").join("memories").join(filename);
            println!("{}", path.display());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn fresh_store() -> (tempfile::TempDir, MemoryStore) {
        let d = tempdir().unwrap();
        let s = MemoryStore::with_root(d.path()).unwrap();
        (d, s)
    }

    #[test]
    fn add_then_list_roundtrip() {
        let (_d, s) = fresh_store();
        run_with_store(
            &s,
            MemoryCmd::Add {
                kind: KindFlag {
                    kind: KindArg::Memory,
                },
                entry: "gpu gfx1151".into(),
            },
        )
        .unwrap();
        assert_eq!(s.list(MemoryKind::Memory).unwrap(), vec!["gpu gfx1151"]);
    }

    #[test]
    fn user_kind_goes_to_user_md() {
        let (d, s) = fresh_store();
        run_with_store(
            &s,
            MemoryCmd::Add {
                kind: KindFlag {
                    kind: KindArg::User,
                },
                entry: "bcloud".into(),
            },
        )
        .unwrap();
        assert!(d.path().join("USER.md").exists());
        assert!(!d.path().join("MEMORY.md").exists());
    }

    #[test]
    fn replace_then_remove() {
        let (_d, s) = fresh_store();
        let km = KindFlag {
            kind: KindArg::Memory,
        };
        run_with_store(
            &s,
            MemoryCmd::Add {
                kind: km,
                entry: "old fact".into(),
            },
        )
        .unwrap();
        run_with_store(
            &s,
            MemoryCmd::Replace {
                kind: km,
                needle: "old".into(),
                entry: "new fact".into(),
            },
        )
        .unwrap();
        assert_eq!(s.list(MemoryKind::Memory).unwrap(), vec!["new fact"]);
        run_with_store(
            &s,
            MemoryCmd::Remove {
                kind: km,
                needle: "new".into(),
            },
        )
        .unwrap();
        assert!(s.list(MemoryKind::Memory).unwrap().is_empty());
    }

    #[test]
    fn list_on_empty_store_is_ok() {
        let (_d, s) = fresh_store();
        run_with_store(
            &s,
            MemoryCmd::List(KindFlag {
                kind: KindArg::Memory,
            }),
        )
        .unwrap();
    }
}
