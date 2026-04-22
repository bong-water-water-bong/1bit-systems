//! Model registry — discovers `.h1b` files on disk and exposes them to the
//! server for both `/v1/models` enumeration and chat-completion `model`
//! field validation.
//!
//! The registry is **validation + discovery only** right now: the server
//! still loads a single backend at startup and serves one model at a time.
//! The registry lets callers (a) see every `.h1b` the box could serve, and
//! (b) get a clear 400 (with the list of known ids) instead of a silent
//! success when they ask for a model id that doesn't exist on disk.
//!
//! Multi-model concurrent serving (lazy-load on first request, LRU evict
//! second-loaded, etc.) is deliberately deferred. The dispatch site in
//! [`crate::routes::chat_completions`] carries a `TODO: lazy-load on
//! request` marker so picking that up later is a two-line change.
//!
//! ## Discovery rules
//!
//! * The registry scans one directory (the CLI `--models-dir` flag).
//! * It does not recurse — `.h1b` files directly inside the dir are picked
//!   up. Nested subdirectories are ignored on purpose so Hugging Face-style
//!   extracted repos (each in their own folder) can't accidentally surface
//!   half-populated converted drafts.
//! * Model id = file basename without the `.h1b` extension (so
//!   `halo-1bit-2b.h1b` registers as `halo-1bit-2b`).
//! * Optional sidecar: if `<basename>.json` exists next to the `.h1b`, we
//!   parse `{"description": "...", "friendly_name": "..."}` and store the
//!   description. Unknown sidecar fields are ignored; a malformed sidecar
//!   logs a warning and the model still registers with an empty
//!   description.
//! * Missing directory, permission-denied, and empty-but-present directory
//!   are **not** errors — the registry reports an empty list and logs a
//!   warning. This matches the "server must still start without a model"
//!   pattern that already lives in `main.rs::build_backend`.

use std::fs;
use std::path::{Path, PathBuf};

use tracing::{debug, warn};

use crate::api::{ModelCard, ModelList};

/// One discovered `.h1b` model, with an optional human description pulled
/// from a sidecar JSON file.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Stable id used on the wire. Basename of the `.h1b` file without the
    /// extension (e.g. `halo-1bit-2b`).
    pub id: String,
    /// Absolute path to the `.h1b` file.
    pub path: PathBuf,
    /// Optional sidecar `description` — populated if a matching `.json`
    /// sibling exists and parses cleanly. Empty otherwise.
    pub description: String,
    /// Optional sidecar `friendly_name` — nothing on the wire cares about
    /// this today, but the field is surfaced on the internal API so a
    /// future `/v1/models` shape extension can pick it up without another
    /// disk scan.
    pub friendly_name: Option<String>,
}

impl ModelEntry {
    /// Project to the OpenAI `/v1/models` card shape.
    ///
    /// `ModelCard` is deliberately narrow today (id / object / owned_by),
    /// so the description/friendly_name fields we carry internally don't
    /// leak yet. The schema stays byte-identical to the pre-registry
    /// single-card output.
    pub fn to_card(&self) -> ModelCard {
        ModelCard::halo(self.id.clone())
    }
}

/// Registry of `.h1b` models discovered on disk at startup.
///
/// Cheap to clone (it's a `Vec<ModelEntry>` under an `Arc` at the call
/// site). Lookups are linear — the expected directory has at most a dozen
/// files, so a `HashMap` would be overkill and would burn a heap
/// allocation per id lookup.
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    entries: Vec<ModelEntry>,
    /// The directory we scanned. Held for diagnostics (the empty-list
    /// warning, doctor probes later on).
    source_dir: Option<PathBuf>,
}

/// Shape of the optional `<basename>.json` sidecar next to an `.h1b` file.
#[derive(Debug, serde::Deserialize, Default)]
struct ModelSidecar {
    #[serde(default)]
    description: String,
    #[serde(default)]
    friendly_name: Option<String>,
}

impl ModelRegistry {
    /// Empty registry — useful for tests and for the `--models-dir`-not-set
    /// fallback path.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Scan `dir` for `.h1b` files. Never returns an error for the three
    /// "environment is wrong" cases — missing directory, permission-denied,
    /// and empty directory all produce an empty registry and a warning log.
    /// Propagating those as fatal at server boot would prevent the
    /// EchoBackend smoke path (`1bit-server` with no model) from working.
    ///
    /// Individual malformed sidecars are also non-fatal: the model is
    /// registered with an empty description and a warning is logged.
    pub fn from_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        let mut out = Self {
            entries: Vec::new(),
            source_dir: Some(dir.to_path_buf()),
        };

        let read = match fs::read_dir(dir) {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    dir = %dir.display(),
                    error = %e,
                    "model registry: cannot read dir — serving with empty model list"
                );
                return out;
            }
        };

        for entry in read {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    warn!(error = %e, "model registry: skipping unreadable dir entry");
                    continue;
                }
            };
            let path = entry.path();
            // Only top-level `.h1b` files. Directories (HF-style converted
            // repos with a `.h1b` inside) are ignored — that would pull in
            // half-converted drafts.
            if !path.is_file() {
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("h1b") {
                continue;
            }
            let id = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) if !s.is_empty() => s.to_string(),
                _ => {
                    warn!(path = %path.display(), "model registry: skipping .h1b with empty basename");
                    continue;
                }
            };

            let sidecar_path = path.with_extension("json");
            let (description, friendly_name) = if sidecar_path.is_file() {
                match fs::read_to_string(&sidecar_path) {
                    Ok(text) => match serde_json::from_str::<ModelSidecar>(&text) {
                        Ok(s) => (s.description, s.friendly_name),
                        Err(e) => {
                            warn!(
                                sidecar = %sidecar_path.display(),
                                error = %e,
                                "model registry: malformed sidecar JSON, using empty description"
                            );
                            (String::new(), None)
                        }
                    },
                    Err(e) => {
                        warn!(
                            sidecar = %sidecar_path.display(),
                            error = %e,
                            "model registry: sidecar exists but unreadable"
                        );
                        (String::new(), None)
                    }
                }
            } else {
                (String::new(), None)
            };

            debug!(id = %id, path = %path.display(), has_sidecar = sidecar_path.is_file(), "model registry: registered");
            out.entries.push(ModelEntry {
                id,
                path,
                description,
                friendly_name,
            });
        }

        // Stable order — the OpenAI clients that drive selection UIs expect
        // `/v1/models` to be deterministic across restarts.
        out.entries.sort_by(|a, b| a.id.cmp(&b.id));

        if out.entries.is_empty() {
            warn!(
                dir = %dir.display(),
                "model registry: no .h1b files discovered — clients will see an empty /v1/models list"
            );
        }
        out
    }

    /// Ensure that a well-known `id` (today: whatever model the single
    /// backend reports via `InferenceBackend::list_models()`) is in the
    /// registry. No-op if already present.
    ///
    /// This guards the "real-backend loaded a model outside `--models-dir`"
    /// case so `/v1/models` and chat validation stay consistent with what
    /// the backend actually answers.
    pub fn ensure_id(&mut self, id: impl Into<String>) {
        let id = id.into();
        if self.entries.iter().any(|e| e.id == id) {
            return;
        }
        self.entries.push(ModelEntry {
            id,
            path: PathBuf::new(),
            description: String::new(),
            friendly_name: None,
        });
        self.entries.sort_by(|a, b| a.id.cmp(&b.id));
    }

    /// True iff `id` names a discovered model.
    pub fn contains(&self, id: &str) -> bool {
        self.entries.iter().any(|e| e.id == id)
    }

    /// True iff no models are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All discovered model ids (stable, alphabetical).
    pub fn ids(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.id.clone()).collect()
    }

    /// Project to an OpenAI-shaped `/v1/models` envelope.
    pub fn to_list(&self) -> ModelList {
        ModelList {
            object: "list",
            data: self.entries.iter().map(ModelEntry::to_card).collect(),
        }
    }

    /// Full internal view — used by diagnostics and (in the future) the
    /// lazy-load dispatcher.
    pub fn entries(&self) -> &[ModelEntry] {
        &self.entries
    }

    /// Directory we scanned at construction, if any. `None` for
    /// [`ModelRegistry::empty`].
    pub fn source_dir(&self) -> Option<&Path> {
        self.source_dir.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Manual tempdir — a couple of crates in the workspace already use
    /// `tempfile`, but `onebit-server` doesn't pull it in today and the
    /// task scope is explicit about not growing unrelated deps. This
    /// helper gives us a unique, auto-cleaning dir under
    /// `$CARGO_TARGET_TMPDIR` (falling back to `/tmp`).
    struct TmpDir(PathBuf);
    impl TmpDir {
        fn new(tag: &str) -> Self {
            let root = std::env::var_os("CARGO_TARGET_TMPDIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| std::env::temp_dir());
            let pid = std::process::id();
            let nonce: u32 = fastrand::u32(..);
            let path = root.join(format!("onebit-server-registry-{tag}-{pid}-{nonce:08x}"));
            fs::create_dir_all(&path).expect("create tempdir");
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn touch(p: &Path) {
        let mut f = fs::File::create(p).expect("create file");
        f.write_all(b"0").expect("write byte");
    }

    #[test]
    fn registry_discovers_h1b_files() {
        let dir = TmpDir::new("discover");
        touch(&dir.path().join("alpha.h1b"));
        touch(&dir.path().join("beta.h1b"));
        touch(&dir.path().join("gamma.h1b"));
        // Non-h1b files must be ignored.
        touch(&dir.path().join("notes.txt"));
        touch(&dir.path().join("beta.htok"));

        let reg = ModelRegistry::from_dir(dir.path());
        let ids = reg.ids();
        assert_eq!(ids, vec!["alpha", "beta", "gamma"], "got {ids:?}");
        assert!(reg.contains("beta"));
        assert!(!reg.contains("delta"));
    }

    #[test]
    fn registry_empty_dir_returns_empty_list() {
        let dir = TmpDir::new("empty");
        let reg = ModelRegistry::from_dir(dir.path());
        assert!(reg.is_empty());
        assert_eq!(reg.ids(), Vec::<String>::new());
        // Envelope must still be a well-formed OpenAI list — empty data.
        let list = reg.to_list();
        assert_eq!(list.object, "list");
        assert!(list.data.is_empty());
    }

    #[test]
    fn registry_missing_dir_returns_empty_list() {
        // Path that definitely does not exist — we assert only that the
        // call does not panic and yields an empty registry.
        let bogus = PathBuf::from(format!(
            "/tmp/onebit-server-registry-does-not-exist-{}",
            std::process::id()
        ));
        let reg = ModelRegistry::from_dir(&bogus);
        assert!(reg.is_empty());
        assert_eq!(reg.source_dir(), Some(bogus.as_path()));
    }

    #[test]
    fn registry_sidecar_json_populates_description() {
        let dir = TmpDir::new("sidecar");
        touch(&dir.path().join("halo-1bit-2b.h1b"));
        fs::write(
            dir.path().join("halo-1bit-2b.json"),
            r#"{"description": "1.58-bit 2B base model", "friendly_name": "Halo 2B"}"#,
        )
        .expect("write sidecar");

        let reg = ModelRegistry::from_dir(dir.path());
        let entry = reg
            .entries()
            .iter()
            .find(|e| e.id == "halo-1bit-2b")
            .expect("entry present");
        assert_eq!(entry.description, "1.58-bit 2B base model");
        assert_eq!(entry.friendly_name.as_deref(), Some("Halo 2B"));
    }

    #[test]
    fn registry_sidecar_malformed_is_non_fatal() {
        let dir = TmpDir::new("sidecar-bad");
        touch(&dir.path().join("m.h1b"));
        fs::write(dir.path().join("m.json"), "{ not: valid json").expect("write bad sidecar");
        let reg = ModelRegistry::from_dir(dir.path());
        assert_eq!(reg.ids(), vec!["m"]);
        assert_eq!(reg.entries()[0].description, "");
    }

    #[test]
    fn registry_ignores_subdirs() {
        let dir = TmpDir::new("subdirs");
        touch(&dir.path().join("root.h1b"));
        let inner = dir.path().join("nested");
        fs::create_dir(&inner).expect("mkdir nested");
        touch(&inner.join("hidden.h1b"));
        let reg = ModelRegistry::from_dir(dir.path());
        assert_eq!(reg.ids(), vec!["root"]);
    }

    #[test]
    fn ensure_id_noops_when_present_and_appends_when_missing() {
        let dir = TmpDir::new("ensure");
        touch(&dir.path().join("a.h1b"));
        let mut reg = ModelRegistry::from_dir(dir.path());
        reg.ensure_id("a"); // no dupe
        reg.ensure_id("b"); // append
        assert_eq!(reg.ids(), vec!["a", "b"]);
    }
}
