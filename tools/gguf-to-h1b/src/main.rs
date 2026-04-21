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

use gguf_to_h1b::{ConvertStats, HtokStats, convert_file, export_htok_sidecar};

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

    /// Skip sidecar `.htok` emission. Default is to write
    /// `<out>.htok` (with `.h1b` replaced) so `bitnet_decode` auto-
    /// finds the right tokenizer for Bonsai models whose Qwen3 vocab
    /// doesn't match halo-1bit-2b's LLaMA-3 vocab.
    #[arg(long)]
    no_htok: bool,

    /// Skip the `.h1b` framing pass and only emit the `.htok`. Useful
    /// when the `.h1b` is already on disk from a previous run and
    /// you just want to refresh the tokenizer sidecar — avoids
    /// rewriting a 1+ GB framed file.
    #[arg(long)]
    htok_only: bool,

    /// Print per-tensor progress instead of just the summary.
    #[arg(long)]
    verbose: bool,
}

fn run(args: Args) -> Result<(Option<ConvertStats>, Option<HtokStats>)> {
    if args.verbose {
        eprintln!("gguf-to-h1b: {:?} -> {:?}", args.input, args.output);
    }
    let stats = if args.htok_only {
        None
    } else {
        Some(convert_file(&args.input, &args.output)?)
    };
    let htok = if args.no_htok {
        None
    } else {
        // Derive <output stem>.htok. Works for foo.h1b → foo.htok,
        // and for outputs without an extension it appends .htok.
        let mut htok_path = args.output.clone();
        if htok_path.extension().and_then(|e| e.to_str()) == Some("h1b") {
            htok_path.set_extension("htok");
        } else {
            let mut s = htok_path.into_os_string();
            s.push(".htok");
            htok_path = PathBuf::from(s);
        }
        Some(export_htok_sidecar(&args.input, &htok_path)?)
    };
    Ok((stats, htok))
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok((s_opt, htok)) => {
            if let Some(s) = s_opt {
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
            }
            if let Some(h) = htok {
                println!(
                    "[gguf-to-h1b] htok vocab={} merges={} bos={} eos={} bytes={} dropped={} path={}",
                    h.vocab_size,
                    h.num_merges,
                    h.bos_id,
                    h.eos_id,
                    h.output_bytes,
                    h.dropped_merges,
                    h.output_path.display(),
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("gguf-to-h1b: error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
