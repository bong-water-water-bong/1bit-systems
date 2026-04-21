//! `gguf-to-h1b` — frame a PrismML Bonsai GGUF into a `.h1b` with the
//! Bonsai flag bit set.
//!
//! ```text
//!   gguf-to-h1b --in /models/Ternary-Bonsai-1.7B-Q2_0.gguf \
//!               --out /models/bonsai-1.7b-tq2.h1b
//! ```
//!
//! The tool is framing-only: it does NOT re-quantize, does NOT rewrite
//! block layouts, does NOT touch tensor payloads. It reads the Bonsai
//! tensor directory, validates every ternary weight matches a Bonsai
//! dtype (41 = Q1_0_g128, 42 = TQ2_0_g128), and streams those payloads
//! verbatim into `.h1b` slots with the corresponding flag bit set in the
//! header's reserved word.
//!
//! Runtime wiring (teaching `bitnet_decode` to consume the resulting
//! `.h1b`) is an explicitly-separate pass — see
//! `docs/wiki/Bonsai-Kernel-Spec.md`.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;

use gguf_to_h1b::{ConvertStats, convert_file};

#[derive(Parser, Debug)]
#[command(name = "gguf-to-h1b", version, about, long_about = None)]
struct Args {
    /// Input `.gguf` file — must be a PrismML Bonsai GGUF (arch=qwen3,
    /// ternary tensors of dtype 41 or 42).
    #[arg(long = "in", value_name = "FILE")]
    input: PathBuf,

    /// Output `.h1b` file. Overwritten if it exists.
    #[arg(long = "out", value_name = "FILE")]
    output: PathBuf,

    /// Print per-tensor progress instead of just the summary.
    #[arg(long)]
    verbose: bool,
}

fn run(args: Args) -> Result<ConvertStats> {
    if args.verbose {
        eprintln!("gguf-to-h1b: {:?} -> {:?}", args.input, args.output);
    }
    let stats = convert_file(&args.input, &args.output)?;
    Ok(stats)
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(s) => {
            println!(
                "[gguf-to-h1b] dtype={:?} layers={} hidden={} ff={} heads={} kv_heads={} \
                 head_dim={} vocab={} ctx={} rope_theta={} eps={:.2e} \
                 ternary_bytes={} output_bytes={} reserved_flags=0x{:x} path={}",
                s.dtype,
                s.num_layers,
                s.hidden_size,
                s.intermediate_size,
                s.num_heads,
                s.num_kv_heads,
                s.head_dim,
                s.vocab_size,
                s.context_length,
                s.rope_theta,
                s.rms_norm_eps,
                s.ternary_bytes_carried,
                s.output_bytes,
                s.h1b_reserved_flags,
                s.output_path.display(),
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("gguf-to-h1b: error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
