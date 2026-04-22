// oobe_error.rs — user-facing error shape for OOBE (anchor #4).
//
// OOBE hard-bar element 4 says every error must have: what broke, likely
// cause / expected state, an exact repro or fix command, and a link to the
// wiki `#troubleshooting` section. We encode that shape as a struct so
// callers cannot forget a field and so the Display block is consistent
// across every gate and step.
//
// Kept intentionally tiny: no `thiserror`, no `anyhow` wrapping — the
// whole point is the four-field contract, and mixing error-chain noise
// into the user face defeats the purpose. `install::print_oobe_error`
// renders this into the terminal with the same shape every time.

use std::fmt;

/// A diagnostic, user-facing error for the OOBE path. Every field is
/// required; the whole point of the struct is that you cannot construct
/// one without the four contract pieces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OobeError {
    /// One short sentence describing what failed. Not a stack trace.
    /// Example: "Kernel 7.0.x is not supported on Strix Halo yet."
    pub what: &'static str,

    /// What the installer expected / what the operator needs to have.
    /// Example: "Linux 6.18-lts (amdgpu OPTC hang fixed on LTS)."
    pub expected: String,

    /// Exact command or concrete action the operator should run.
    /// Example: "sudo limine-snapshot rollback 6 && reboot"
    pub repro: String,

    /// Deep-link into the GitHub wiki `#troubleshooting` section so the
    /// user can copy-paste and land on the right anchor.
    pub wiki_link: &'static str,
}

impl OobeError {
    /// Kernel is too new (7.x); the amdgpu OPTC CRTC hang freezes Wayland
    /// on Strix Halo on anything newer than the 6.18-lts baseline.
    pub fn kernel_too_new(current: &str) -> Self {
        Self {
            what: "Kernel is too new for Strix Halo OOBE.",
            expected: format!(
                "Linux 6.18-lts is the recommended baseline. Detected: {current}."
            ),
            repro: "Boot snapper snapshot #6 or install linux-lts and reboot.".to_string(),
            wiki_link: "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#kernel-too-new",
        }
    }

    /// ROCm 7.x userspace missing. We don't attempt to install it from
    /// Rust — that's install.sh's job — but we tell the operator exactly
    /// how to.
    pub fn rocm_missing() -> Self {
        Self {
            what: "ROCm 7.x userspace not detected.",
            expected: "rocminfo reachable on $PATH with a gfx1151 or gfx1201 agent.".to_string(),
            repro: "sudo pacman -S rocm-hip-runtime rocminfo".to_string(),
            wiki_link: "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#rocm-missing",
        }
    }

    /// Disk free is below the 10 GB OOBE floor (binaries + weights).
    pub fn disk_too_small(free_gb: u64) -> Self {
        Self {
            what: "Free disk is below the 10 GB OOBE floor.",
            expected: format!("≥ 10 GB free on the install target. Detected: {free_gb} GB."),
            repro: "df -h / ; remove old caches under ~/.cargo/registry and ~/.halo/logs".to_string(),
            wiki_link: "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#disk-too-small",
        }
    }

    /// RAM is below the 64 GB OOBE floor. Warn below 128 GB is handled
    /// separately in `preflight` as a `Skip` outcome, not an error.
    pub fn ram_too_small(have_gb: u64, floor_gb: u64) -> Self {
        Self {
            what: "RAM is below the OOBE minimum.",
            expected: format!(
                "≥ {floor_gb} GB RAM for halo-v2 at Q4_K_M. Detected: {have_gb} GB."
            ),
            repro: "Close other tenants or choose a smaller default model (Qwen3-4B Q4).".to_string(),
            wiki_link: "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#ram-too-small",
        }
    }
}

impl fmt::Display for OobeError {
    /// Fixed four-line block. The preflight renderer relies on this shape
    /// so a human eye can scan `what → expected → fix → wiki` in any
    /// error the OOBE emits.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  what     : {}", self.what)?;
        writeln!(f, "  expected : {}", self.expected)?;
        writeln!(f, "  fix      : {}", self.repro)?;
        write!(f, "  wiki     : {}", self.wiki_link)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every constructor must populate all four fields — the whole
    /// contract is that the OOBE error has `what / expected / fix /
    /// wiki` and the Display block therefore never loses a line.
    #[test]
    fn constructors_fill_all_four_fields() {
        let errs = vec![
            OobeError::kernel_too_new("7.1.0"),
            OobeError::rocm_missing(),
            OobeError::disk_too_small(3),
            OobeError::ram_too_small(32, 64),
        ];
        for e in &errs {
            assert!(!e.what.is_empty(), "what must be populated");
            assert!(!e.expected.is_empty(), "expected must be populated");
            assert!(!e.repro.is_empty(), "repro must be populated");
            assert!(
                e.wiki_link.starts_with("https://"),
                "wiki_link must be an https URL, got {:?}",
                e.wiki_link
            );
            assert!(
                e.wiki_link.contains("#"),
                "wiki_link must deep-link to a #section anchor, got {:?}",
                e.wiki_link
            );
        }
    }

    /// The Display block is the user-facing surface; it must emit all
    /// four fields in the documented order. Any future refactor that
    /// drops a line or reorders them breaks the OOBE contract.
    #[test]
    fn display_block_emits_four_lines_in_order() {
        let e = OobeError::kernel_too_new("7.1.0");
        let out = format!("{e}");
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 4, "must be exactly 4 lines, got: {out:?}");
        assert!(lines[0].trim_start().starts_with("what"), "line 0 = what");
        assert!(
            lines[1].trim_start().starts_with("expected"),
            "line 1 = expected"
        );
        assert!(lines[2].trim_start().starts_with("fix"), "line 2 = fix");
        assert!(lines[3].trim_start().starts_with("wiki"), "line 3 = wiki");
    }
}
