//! `bitnet-to-tq2` — CLI entry point. See `lib.rs` for the full doc comment.

use std::process::ExitCode;

use anyhow::Result;
use bitnet_to_tq2::{ConvertStats, cli::Args, convert};
use clap::Parser;

fn run(args: Args) -> Result<ConvertStats> {
    let stats = convert(&args.input, &args.output)?;
    Ok(stats)
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(s) => {
            println!(
                "[bitnet-to-tq2] layers={} hidden={} ff={} heads={} kv_heads={} vocab={} ctx={} \
                 rope_theta={} eps={:.2e} packed_ternary_bytes={} output_bytes={} \
                 reserved_flags=0x{:x} path={}",
                s.config.num_hidden_layers,
                s.config.hidden_size,
                s.config.intermediate_size,
                s.config.num_attention_heads,
                s.config.num_key_value_heads,
                s.config.vocab_size,
                s.config.max_position_embeddings,
                s.config.rope_theta,
                s.config.rms_norm_eps,
                s.packed_ternary_bytes,
                s.output_bytes,
                s.h1b_reserved_flags,
                s.output_path.display(),
            );
            if !s.unmatched_tensors.is_empty() {
                eprintln!(
                    "[bitnet-to-tq2] note: {} unmatched HF tensors (diagnostic only):",
                    s.unmatched_tensors.len()
                );
                for n in &s.unmatched_tensors {
                    eprintln!("    {n}");
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("bitnet-to-tq2: error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
