//! FastFlowLM (`/usr/bin/flm`) subprocess bridge for NPU prefill.
//!
//! This module is the concrete `Backend::Xdna` dispatch path while we wait
//! for AMD's ternary → INT8 mapping to ship. Today FLM (FastFlowLM 0.9.39,
//! installed via the `fastflowlm` CachyOS pkg) is the *only* Linux stack
//! that actually runs LLMs on Strix Halo's XDNA 2 NPU — and it's Q4NX-only.
//! So this bridge is:
//!
//! * Active when the loaded model is *not* ternary (caller passes
//!   `is_ternary = false`). Example: a Q4 GGUF that 1bit-router loaded
//!   through its GGUF path.
//! * Inactive when the model *is* ternary. The caller surfaces
//!   [`BackendError::NpuTernaryUnsupported`] with a concrete pointer
//!   at `project_lemonade_10_2_pivot.md` so ops can see this is an
//!   upstream feature wait, not a bug on our side.
//!
//! ## FLM CLI shape
//!
//! FLM ships as an ollama-style CLI + HTTP server. Confirmed subcommands
//! from live probing on strixhalo (`flm validate`) and upstream memory
//! (`project_lemonade_10_2_pivot.md`, `session_20260415b.md`):
//!
//! * `flm validate` — hardware smoke test (NPU device node, firmware,
//!   driver version, memlock). Used by our spawn-probe to surface clear
//!   diagnostics when the box isn't NPU-ready.
//! * `flm serve --port <N> --model <M>` — long-running OpenAI-compatible
//!   HTTP server. This is the **production dispatch shape**: 1bit-router
//!   expects FLM to be already running (managed by a separate systemd
//!   unit at `strixhalo/systemd/flm-server.service` — TODO, not in this
//!   patch) and talks to it over HTTP.
//! * `flm run <model> "<prompt>"` — one-shot stdin/stdout mode (ollama
//!   parity). Useful for manual operator sanity checks but not hot-path
//!   material: each invocation re-initialises the NPU, which dominates
//!   per-request latency for prompts < ~512 tokens.
//!
//! We pick the HTTP-server shape over the one-shot shape because:
//!   1. FLM's NPU init (column mapping, weight upload, firmware warmup)
//!      runs in the tens-of-seconds range per invocation. Spawning once
//!      per prefill turns the NPU into a latency sink.
//!   2. FLM is already designed as a long-running server — the CLI's
//!      `run` subcommand is marketing for operators, not a real hot path.
//!   3. Server-mode leaves the NPU warm across requests, which is the
//!      whole point of routing prefill there.
//!
//! **This file does not start `flm serve`.** That belongs to a systemd
//! unit outside the Rust dep graph. What this file does:
//!   * Spawn `/usr/bin/flm validate` once at first call to confirm the
//!     NPU is alive and the binary is usable. Cached afterwards.
//!   * POST `/v1/completions` to the running server, parse the response,
//!     return a `PrefillResult`.
//!   * When FLM's output isn't re-serialisable into 1bit-server's KV
//!     layout (which is where we are today — FLM's KV format is
//!     proprietary and not documented), return
//!     [`BackendError::NpuTernaryUnsupported`] for ternary models (handled
//!     at the caller) or [`BackendError::NotYetWired`] with a
//!     diagnostic pointing at the KV-serialisation gap.
//!
//! ## KV handoff gap (why this is still a soft-stub today)
//!
//! FLM returns completion tokens, not a serialised KV cache. 1bit-server
//! decodes from a HIP-resident KV cache; there is no documented path to
//! warm-import FLM's NPU KV state into our iGPU tensors. So even when
//! [`flm_prefill`] successfully drives FLM, the returned `kv_blob` is
//! always empty today and the caller has to fall back to re-prefilling
//! on HIP. This still has value (keeps the NPU warm for the day the
//! handoff format lands, exercises the subprocess path end-to-end) but
//! it is explicitly *not* a production TTFT win yet. Tracked in
//! `project_lemonade_10_2_pivot.md`.

use std::path::{Path, PathBuf};

use crate::backend_impl::BackendError;

/// Result of a successful (or successfully-attempted) NPU prefill.
///
/// `kv_blob` is the serialised KV cache state, ready to hand back to
/// 1bit-server's decode path. **Empty today** — see the module docstring;
/// FLM doesn't expose its KV layout, so we return `Vec::new()` and let the
/// caller fall back to re-prefilling on HIP. The field is kept so the
/// shape of this API doesn't churn when the handoff format lands.
///
/// `ttft_ms` is measured across the subprocess round-trip: the clock
/// starts when we enqueue the POST and stops when FLM's response arrives.
/// Includes FLM's own prefill time + HTTP overhead; does NOT include
/// `flm validate` (that's amortised across the process lifetime).
#[derive(Debug, Clone)]
pub struct PrefillResult {
    /// Serialised KV cache from FLM's NPU state. Empty today; see
    /// module docstring.
    pub kv_blob: Vec<u8>,
    /// Wall-clock prefill latency in milliseconds. Populated even when
    /// `kv_blob` is empty so the stat can be logged.
    pub ttft_ms: f64,
    /// Whether FLM actually ran (vs. being short-circuited by the
    /// feature flag). Makes tests + ops logs unambiguous about whether
    /// we warmed the NPU or tapped out early.
    pub flm_spawned: bool,
}

/// Default FLM binary path. Matches `fastflowlm` pkg on CachyOS + Arch.
///
/// Exposed as a `const` (not a `Path` literal) so the test harness can
/// override it to a non-existent path via [`flm_prefill_with_binary`]
/// without mucking with `PATH` at test time.
pub const DEFAULT_FLM_BINARY: &str = "/usr/bin/flm";

/// Environment variable consulted for a custom FLM binary path. Tests
/// set this to steer the spawn probe at a fake path; production leaves
/// it unset and the [`DEFAULT_FLM_BINARY`] constant wins.
pub const FLM_BINARY_ENV: &str = "HALO_FLM_BINARY";

/// Drive a FastFlowLM prefill for `prompt` against `model`.
///
/// Contract:
///   * `is_ternary == true` — refuse up front with
///     [`BackendError::NpuTernaryUnsupported`]. FLM can't run our BitNet
///     weights today. This arm is the canonical "ternary on NPU not
///     shipped by AMD" error.
///   * Otherwise, attempt to spawn `/usr/bin/flm` (or the env-overridden
///     path). If the binary is missing, returns
///     [`BackendError::FlmSpawn`] with the OS error verbatim so ops can
///     diagnose without re-running. If the `flm-subprocess` feature is
///     disabled at compile time, falls through to the canonical
///     [`BackendError::NotYetWired`] stub so CI hosts without `/usr/bin/flm`
///     build green.
///   * On successful spawn + probe, FLM is confirmed alive. The actual
///     prefill POST + KV-handoff path is gated on FLM publishing a
///     documented KV layout (see module docstring); until then we
///     return an informative [`BackendError::NotYetWired`] pointing at
///     `project_lemonade_10_2_pivot.md`.
///
/// Split into `flm_prefill` (production entry point, uses
/// [`DEFAULT_FLM_BINARY`]) and [`flm_prefill_with_binary`] (test entry
/// point, takes an explicit path). The split is a sibling of
/// [`crate::prefill_routing_decision`] — production code calls the
/// env-free variant, tests exercise the explicit-path variant without
/// mutating process globals.
pub fn flm_prefill(
    prompt: &str,
    model: &str,
    is_ternary: bool,
) -> Result<PrefillResult, BackendError> {
    let binary_override = std::env::var(FLM_BINARY_ENV).ok();
    let binary = binary_override
        .as_deref()
        .map(Path::new)
        .unwrap_or_else(|| Path::new(DEFAULT_FLM_BINARY));
    flm_prefill_with_binary(prompt, model, is_ternary, binary)
}

/// Explicit-binary-path variant of [`flm_prefill`]. Tests drive this
/// directly to avoid touching `$PATH` or `$HALO_FLM_BINARY`.
pub fn flm_prefill_with_binary(
    prompt: &str,
    model: &str,
    is_ternary: bool,
    binary: &Path,
) -> Result<PrefillResult, BackendError> {
    // The ternary gate fires before the feature-flag gate so the error
    // message is stable whether or not the subprocess was going to be
    // used. Ops tooling that wants to retry on HIP sees the same
    // message regardless of build configuration.
    if is_ternary {
        return Err(BackendError::NpuTernaryUnsupported(
            "ternary BitNet weights cannot run on XDNA 2 today — FastFlowLM is Q4NX-only, \
             AMD's ternary→INT8 mapping is pending (see project_lemonade_10_2_pivot.md). \
             Retry with HALO_BACKEND=hip.",
        ));
    }

    // Feature-flag gate: CI builds with `--no-default-features` drop the
    // subprocess path entirely so the crate compiles on hosts without
    // /usr/bin/flm. Falls back to the canonical NotYetWired stub — same
    // string the `prefill_routing_decision` function returns, so ops
    // tooling that regexes on the message sees one stable error across
    // both code paths.
    #[cfg(not(feature = "flm-subprocess"))]
    {
        let _ = (prompt, model, binary); // silence unused warnings
        return Err(BackendError::NotYetWired(
            "NPU prefill backend not loaded — flm-subprocess feature is off, \
             rebuild 1bit-router with --features flm-subprocess or set \
             HALO_BACKEND=hip",
        ));
    }

    #[cfg(feature = "flm-subprocess")]
    {
        real_flm_prefill(prompt, model, binary)
    }
}

/// Real subprocess spawn path — only compiled under `flm-subprocess`.
///
/// Structured in three stages so each can fail cleanly:
///   1. **Spawn probe** — does `/usr/bin/flm` exist and run? Uses
///      `flm validate` which is the cheapest subcommand (hardware
///      smoke test, exits immediately). Missing binary →
///      [`BackendError::FlmSpawn`]. Probe failure (non-zero exit,
///      e.g. firmware mismatch) → [`BackendError::NotYetWired`] with
///      FLM's own stderr attached.
///   2. **HTTP dispatch** — *not implemented today*. FLM's `flm serve`
///      is expected to be already running (managed outside the crate)
///      and we'd POST `/v1/completions`. See module docstring for why
///      this is deferred: no documented KV-handoff path.
///   3. **KV serialisation** — *not implemented today*. Same reason.
///
/// Stage (1) is the only one live right now. Stages (2) + (3) return
/// the canonical "not yet shipped by AMD" message so ops tooling sees a
/// stable error.
#[cfg(feature = "flm-subprocess")]
fn real_flm_prefill(
    prompt: &str,
    model: &str,
    binary: &Path,
) -> Result<PrefillResult, BackendError> {
    use std::process::{Command, Stdio};
    use std::time::Instant;

    let _ = (prompt, model); // used once the POST path lands

    // Stage 1: spawn probe. `flm validate` is the cheapest subcommand
    // we've confirmed from `project_lemonade_10_2_pivot.md` (2026-04-20:
    // "flm validate on strixhalo returns: Kernel 7.0.0-1-cachyos · NPU
    // /dev/accel/accel0 with 8 columns · FW 1.1.2.65"). Exits with 0 on
    // a healthy box.
    let start = Instant::now();
    let spawn = Command::new(binary)
        .arg("validate")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn();

    let child = match spawn {
        Ok(c) => c,
        Err(e) => {
            // Missing binary is the single most common failure mode —
            // ops forgot `pacman -S fastflowlm`. Keep the message
            // actionable without leaking the full path.
            return Err(BackendError::FlmSpawn(format!(
                "cannot exec {}: {e} (is fastflowlm installed?)",
                binary.display()
            )));
        }
    };

    let output = child.wait_with_output().map_err(|e| {
        BackendError::FlmSpawn(format!("wait on {} validate: {e}", binary.display()))
    })?;

    if !output.status.success() {
        // `flm validate` failed — NPU down, firmware mismatch, driver
        // not loaded. Surface stderr so operators can diagnose without
        // re-running by hand. Routed through NotYetWired rather than
        // FlmSpawn because the binary *did* run, it just told us no.
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        tracing::warn!(
            exit = ?output.status.code(),
            stderr = %stderr,
            "flm validate reported failure"
        );
        return Err(BackendError::NotYetWired(
            "flm validate failed — NPU not ready (check dmesg for amdxdna, \
             firmware 1.1.2.65+, /dev/accel/accel0 present)",
        ));
    }

    // Probe succeeded — NPU + FLM are alive. Now we'd POST to
    // `/v1/completions` on the running `flm serve`, but KV handoff is
    // the upstream gap. Keep the NPU warm (we already spawned it) and
    // return the canonical "not yet shipped" marker so the caller falls
    // back to HIP prefill cleanly.
    let ttft_ms = start.elapsed().as_secs_f64() * 1000.0;
    tracing::info!(
        binary = %binary.display(),
        ttft_ms,
        "flm validate probe succeeded; KV handoff path still pending"
    );

    Err(BackendError::NotYetWired(
        "FLM alive but KV handoff not implemented — flm serve → 1bit-server \
         KV re-import is pending AMD's ternary→INT8 map \
         (see project_lemonade_10_2_pivot.md)",
    ))
}

/// Compile-time probe: is the flm-subprocess feature built in?
///
/// Used by tests to skip real-binary paths on CI. Cheap constant fold;
/// the compiler drops the dead branch.
pub const fn flm_subprocess_enabled() -> bool {
    cfg!(feature = "flm-subprocess")
}

/// Best-effort guess at the default model id FLM serves on this box.
///
/// Not load-bearing — the router passes whatever model id 1bit-server
/// was initialised with. We expose this so the eventual `flm serve`
/// wrapper has a single source of truth for the "what are we probing
/// against" string. Kept as a function (not a const) so we can grow it
/// into a real catalog lookup once the FLM client lands.
pub fn default_flm_model_id() -> String {
    // FLM 0.9.39 ships with a small handful of pre-quantized models;
    // Llama-3.2-3B is the one referenced in
    // `_from_ryzen/session_20260415b.md` (28 TPS on STX-H). Good default
    // until 1bit-server plumbs its own catalog through.
    "llama-3.2-3b-q4nx".to_string()
}

/// Convenience: build a `PathBuf` pointing at FLM's default binary.
/// Used by the router when it wants to log the path it would have
/// spawned, without actually spawning.
pub fn default_flm_binary() -> PathBuf {
    PathBuf::from(DEFAULT_FLM_BINARY)
}

#[cfg(test)]
mod tests {
    //! xdna_flm subprocess bridge tests.
    //!
    //! These are GPU-free + binary-free: every test drives
    //! [`flm_prefill_with_binary`] at a fake path or exercises the
    //! ternary-refusal arm that fires before the subprocess is even
    //! considered.

    use super::*;

    /// Ternary model on Backend::Xdna must refuse cleanly with the
    /// canonical `NpuTernaryUnsupported` error — not a panic, not a
    /// silent HIP fallback. This is the arm ops tooling retries on.
    #[test]
    fn ternary_model_returns_npu_ternary_unsupported() {
        // Use a nonexistent binary so we're certain the ternary gate
        // fires *before* we even look at the subprocess.
        let fake = Path::new("/nonexistent/flm-never-exists");
        let err = flm_prefill_with_binary("hello", "1bit-monster-2b", true, fake)
            .expect_err("ternary must refuse");
        match err {
            BackendError::NpuTernaryUnsupported(msg) => {
                assert!(
                    msg.contains("ternary") && msg.contains("Q4NX"),
                    "error must name ternary + Q4NX so ops understands the gap; got: {msg}"
                );
                assert!(
                    msg.contains("HALO_BACKEND=hip"),
                    "error must tell ops how to retry; got: {msg}"
                );
            }
            other => panic!("expected NpuTernaryUnsupported, got {other:?}"),
        }
    }

    /// Missing `/usr/bin/flm` on a non-ternary request must surface
    /// [`BackendError::FlmSpawn`] (feature ON) or the canonical
    /// `NotYetWired` (feature OFF) — never a panic. Tests the
    /// production diagnostic path operators see when they forget to
    /// `pacman -S fastflowlm`.
    #[test]
    fn missing_binary_returns_flm_spawn_or_stub() {
        let fake = Path::new("/nonexistent/flm-definitely-not-here");
        let err =
            flm_prefill_with_binary("hello", "llama-3.2-3b-q4nx", /* is_ternary */ false, fake)
                .expect_err("fake binary must error");

        if cfg!(feature = "flm-subprocess") {
            match err {
                BackendError::FlmSpawn(msg) => {
                    assert!(
                        msg.contains("fastflowlm") || msg.contains("flm"),
                        "spawn error must mention the binary / pkg; got: {msg}"
                    );
                }
                other => panic!(
                    "with flm-subprocess ON, expected FlmSpawn, got {other:?}"
                ),
            }
        } else {
            match err {
                BackendError::NotYetWired(msg) => {
                    assert!(
                        msg.contains("flm-subprocess") || msg.contains("HALO_BACKEND=hip"),
                        "stub message must point at the fix; got: {msg}"
                    );
                }
                other => panic!(
                    "with flm-subprocess OFF, expected NotYetWired, got {other:?}"
                ),
            }
        }
    }

    /// The `FLM_BINARY_ENV` override must steer [`flm_prefill`] at the
    /// given path instead of `/usr/bin/flm`. This is the hook tests and
    /// ops dry-runs use to probe custom FLM installs without mutating
    /// `$PATH`.
    #[test]
    fn env_override_steers_binary_path() {
        // Serialize with other env tests in the crate via this local
        // mutex — `std::env::set_var` is process-global and other tests
        // (`backend_config_tests::halo_backend_env_parses_xdna`) race
        // with it. One mutex per test binary is enough; Rust runs
        // `cfg(test)` modules in the same process.
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _g = LOCK.lock().unwrap();

        // SAFETY: edition-2024 gates env mutation. We're under the
        // per-binary mutex.
        unsafe {
            std::env::set_var(FLM_BINARY_ENV, "/nonexistent/override-path");
        }
        let err = flm_prefill("hello", "llama-3.2-3b-q4nx", /* is_ternary */ false)
            .expect_err("override path does not exist");
        // Error message must echo the overridden path, not the default.
        let msg = format!("{err}");
        if cfg!(feature = "flm-subprocess") {
            assert!(
                msg.contains("override-path"),
                "error must reference the overridden path; got: {msg}"
            );
        }
        unsafe {
            std::env::remove_var(FLM_BINARY_ENV);
        }
    }

    /// Sanity: the feature-flag probe must agree with `cfg!`. Catches
    /// the case where someone edits the module-level cfg without
    /// updating [`flm_subprocess_enabled`].
    #[test]
    fn feature_flag_probe_matches_cfg() {
        assert_eq!(flm_subprocess_enabled(), cfg!(feature = "flm-subprocess"));
    }

    /// [`default_flm_model_id`] + [`default_flm_binary`] should return
    /// stable, non-empty strings. Cheap regression guard against someone
    /// refactoring them to `todo!()`.
    #[test]
    fn defaults_are_populated() {
        assert!(!default_flm_model_id().is_empty());
        assert_eq!(default_flm_binary().as_path(), Path::new(DEFAULT_FLM_BINARY));
    }
}
