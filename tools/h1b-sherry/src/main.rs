//! `h1b-sherry` — offline requantizer CLI.
//!
//! ```text
//!   h1b-sherry --in /models/halo-1bit-2b.h1b --out /models/halo-1bit-2b-sherry.h1b
//! ```
//!
//! Reads a TQ1 v4 `.h1b`, repacks every ternary tensor in Sherry 1.25-bit
//! (3:4-sparse), and writes a v3 `.h1b` with the `H1B_FLAG_SHERRY_FP16` bit
//! set so the runtime dispatcher routes weights through
//! `sherry_ternary_gemv_launch`. The Hadamard rotation flag is preserved.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;

use h1b_sherry::convert_file;

/// Offline requantizer: halo-1bit TQ1 v4 -> Sherry 1.25-bit v3 (fp16 flag).
#[derive(Parser, Debug)]
#[command(name = "h1b-sherry", version, about, long_about = None)]
struct Args {
    /// Input .h1b file (must be version 4 / TQ1 packing).
    #[arg(long = "in", value_name = "FILE")]
    input: PathBuf,

    /// Output .h1b file (will be v3 with H1B_FLAG_SHERRY_FP16 set).
    #[arg(long = "out", value_name = "FILE")]
    output: PathBuf,

    /// Print detailed per-layer stats instead of just the summary.
    #[arg(long)]
    verbose: bool,
}

fn run(args: Args) -> Result<()> {
    let stats = convert_file(&args.input, &args.output)?;
    let pct = stats.flip_fraction() * 100.0;
    eprintln!(
        "h1b-sherry: wrote {} (layers={}, rows={}, groups={}, forced_zero_flips={} = {:.3}%, hadamard_preserved={})",
        args.output.display(),
        stats.layers_processed,
        stats.rows_total,
        stats.groups_total,
        stats.forced_zero_flips,
        pct,
        stats.hadamard_preserved,
    );
    if args.verbose {
        eprintln!(
            "h1b-sherry: sign-change upper bound is 25% per group; observed {:.3}% avg",
            pct,
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("h1b-sherry: error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
