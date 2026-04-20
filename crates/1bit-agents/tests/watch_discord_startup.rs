//! Integration test: `halo-watch-discord` gracefully degrades when no
//! `DISCORD_BOT_TOKEN` is set.
//!
//! We use Cargo's `CARGO_BIN_EXE_halo-watch-discord` env var (populated
//! during `cargo test`) to get the compiled binary path, then run it with
//! a deliberately cleared token env. The contract is:
//!
//!   * exit status 0
//!   * stdout contains the help banner ("DISCORD_BOT_TOKEN")
//!
//! The bot must NOT attempt to connect to Discord in this path — that
//! would hang CI and surface noise in journal.

use std::process::Command;
use std::time::Duration;

#[test]
fn startup_without_token_prints_help_and_exits_zero() {
    let bin = env!("CARGO_BIN_EXE_halo-watch-discord");

    // Clear any inherited token + channel list. Pass an empty token so
    // the `if !t.trim().is_empty()` branch in main() routes us to the
    // help path regardless of the outer shell env.
    let mut cmd = Command::new(bin);
    cmd.env_remove("DISCORD_BOT_TOKEN");
    cmd.env_remove("HALO_DISCORD_CHANNELS");
    cmd.env("RUST_LOG", "off");
    cmd.stdin(std::process::Stdio::null());

    // Short timeout via a spawn+wait_with_output; if main() tried to
    // actually connect we'd block on the gateway. Use a kill-on-timeout
    // pattern without pulling in extra deps.
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn halo-watch-discord");

    // Give the binary a few seconds to print + exit. In practice the
    // help path is ~immediate; the timeout is just a safety net.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => {
                let output = child.wait_with_output().expect("wait_with_output");
                assert!(
                    status.success(),
                    "expected exit 0 on missing token, got {status:?}. stderr={}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let stdout = String::from_utf8_lossy(&output.stdout);
                assert!(
                    stdout.contains("DISCORD_BOT_TOKEN"),
                    "help text missing from stdout. got: {stdout}"
                );
                assert!(
                    stdout.contains("HALO_DISCORD_CHANNELS"),
                    "help text missing channel env. got: {stdout}"
                );
                return;
            }
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                panic!("halo-watch-discord did not exit within 5s — it tried to connect?");
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}
