// `halo say <text>` — speak text through halo-kokoro :8083, play via
// the first available ALSA/PulseAudio CLI player. Voice-loop mic-check
// path for recording sessions.

use anyhow::{Context, Result, anyhow, bail};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

const DEFAULT_URL: &str = "http://127.0.0.1:8083/tts";
const DEFAULT_VOICE: &str = "af_sky";

pub async fn run(text: &str, voice: &str, speed: f32) -> Result<()> {
    if text.trim().is_empty() {
        bail!("halo say: empty text");
    }
    let url = std::env::var("HALO_KOKORO_URL").unwrap_or_else(|_| DEFAULT_URL.into());

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;

    // kokoro_tts's cxxopts-based speed parser throws std::bad_cast on
    // floats ("1.0"). Omit the field when the user didn't change the
    // default — the server then skips passing --speed to the CLI.
    let mut body = serde_json::json!({ "text": text, "voice": voice });
    if (speed - 1.0).abs() > f32::EPSILON {
        body["speed"] = serde_json::json!(speed);
    }
    let res = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;
    let status = res.status();
    if !status.is_success() {
        let msg = res.text().await.unwrap_or_default();
        bail!("halo-kokoro returned {status}: {msg}");
    }
    let wav = res.bytes().await?;
    play_wav(&wav)?;
    Ok(())
}

fn play_wav(wav: &[u8]) -> Result<()> {
    // Pick the first player on PATH — pulse first (desktop), alsa second,
    // ffplay as last-ditch. All accept wav on stdin.
    let (bin, args): (&str, &[&str]) = if which("paplay") {
        ("paplay", &[])
    } else if which("aplay") {
        ("aplay", &["-q"])
    } else if which("ffplay") {
        (
            "ffplay",
            &["-nodisp", "-autoexit", "-loglevel", "error", "-"],
        )
    } else {
        return Err(anyhow!(
            "no audio player found — install pulseaudio, alsa-utils, or ffmpeg"
        ));
    };

    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawn {bin}"))?;

    child
        .stdin
        .as_mut()
        .ok_or_else(|| anyhow!("no stdin"))?
        .write_all(wav)?;
    let out = child.wait()?;
    if !out.success() {
        bail!("{bin} exited {out}");
    }
    Ok(())
}

fn which(bin: &str) -> bool {
    Command::new("which")
        .arg(bin)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn default_voice() -> &'static str {
    DEFAULT_VOICE
}
