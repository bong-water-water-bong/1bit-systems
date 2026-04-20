//! 1bit-whisper CLI — read 16 kHz s16le mono PCM from stdin, stream
//! partial transcripts to stdout line-by-line.
//!
//! Under the default `stub` feature this prints an error and exits non-
//! zero; the real path activates with
//! `--features real-whisper --no-default-features`.
//!
//! Usage:
//!
//! ```text
//! arecord -f S16_LE -r 16000 -c 1 -t raw | 1bit-whisper path/to/ggml-base.en.bin
//! ```

use onebit_whisper::{WhisperEngine, WhisperError};
use std::env;
use std::io::{self, Read, Write};
use std::process::ExitCode;

const CHUNK_BYTES: usize = 4096; // 2048 samples @ 16-bit

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let model_path = match args.get(1) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: 1bit-whisper <model.ggml>");
            return ExitCode::from(2);
        }
    };

    let mut engine = match WhisperEngine::new(&model_path) {
        Ok(e) => e,
        Err(WhisperError::UnsupportedStub) => {
            eprintln!(
                "1bit-whisper: built with `stub` feature; rebuild with \
                 --features real-whisper --no-default-features"
            );
            return ExitCode::from(3);
        }
        Err(e) => {
            eprintln!("1bit-whisper: model load failed: {e}");
            return ExitCode::from(4);
        }
    };

    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    let mut byte_buf = vec![0u8; CHUNK_BYTES];

    loop {
        match stdin.read(&mut byte_buf) {
            Ok(0) => break, // EOF
            Ok(n) if n % 2 == 0 => {
                // Reinterpret as i16 little-endian.
                let samples: Vec<i16> = byte_buf[..n]
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();
                if let Err(e) = engine.feed(&samples) {
                    eprintln!("1bit-whisper: feed failed: {e}");
                    return ExitCode::from(5);
                }
                match engine.drain_partials() {
                    Ok(parts) => {
                        for p in parts {
                            let _ = writeln!(stdout, "[{}..{}ms] {}", p.start_ms, p.end_ms, p.text);
                            let _ = stdout.flush();
                        }
                    }
                    Err(e) => {
                        eprintln!("1bit-whisper: drain failed: {e}");
                        return ExitCode::from(6);
                    }
                }
            }
            Ok(_n) => {
                // Odd number of bytes — stdin feed lost a byte. Bail.
                eprintln!("1bit-whisper: stdin delivered odd byte count, exiting");
                return ExitCode::from(7);
            }
            Err(e) => {
                eprintln!("1bit-whisper: stdin read failed: {e}");
                return ExitCode::from(8);
            }
        }
    }

    ExitCode::SUCCESS
}
