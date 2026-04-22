//! 1bit-kokoro CLI — synthesize a line of text with a voice + speed, write
//! mono 22 050 Hz s16le PCM to stdout.
//!
//! Under the default `stub` feature this prints an error and exits non-
//! zero; the real path activates with
//! `--features real-kokoro --no-default-features`.
//!
//! Usage:
//!
//! ```text
//! 1bit-kokoro <model.onnx> <voice> <speed> <text...>
//! 1bit-kokoro kokoro-v1.onnx af_bella 1.0 "hello world" > out.pcm
//! ```

use onebit_kokoro::{KokoroEngine, KokoroError};
use std::env;
use std::io::{self, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: 1bit-kokoro <model.onnx> <voice> <speed> <text...>");
        return ExitCode::from(2);
    }
    let model_path = &args[1];
    let voice = &args[2];
    let speed: f32 = match args[3].parse() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("1bit-kokoro: could not parse speed {:?}: {e}", args[3]);
            return ExitCode::from(2);
        }
    };
    let text = args[4..].join(" ");

    let mut engine = match KokoroEngine::new(model_path) {
        Ok(e) => e,
        Err(KokoroError::UnsupportedStub) => {
            eprintln!(
                "1bit-kokoro: built with `stub` feature; rebuild with \
                 --features real-kokoro --no-default-features"
            );
            return ExitCode::from(3);
        }
        Err(e) => {
            eprintln!("1bit-kokoro: model load failed: {e}");
            return ExitCode::from(4);
        }
    };

    let pcm = match engine.synthesize(&text, voice, speed) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("1bit-kokoro: synth failed: {e}");
            return ExitCode::from(5);
        }
    };

    // Write s16le PCM to stdout.
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    if let Err(e) = stdout.write_all(&bytes) {
        eprintln!("1bit-kokoro: stdout write failed: {e}");
        return ExitCode::from(6);
    }
    ExitCode::SUCCESS
}
