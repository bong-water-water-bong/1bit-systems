//! Microbenchmark: measure the sampler's wall-clock cost on a realistic
//! 128,256-element logit vector (BitNet's vocab size).
//!
//! ## Purpose
//!
//! Decides whether moving the sampler off the iGPU critical path is worth
//! the plumbing. The pipelined gain ("iGPU starts next forward while CPU
//! finishes current sampler") is bounded above by the sampler's share of
//! one decode step. If the sampler takes <5% of a forward pass, the gain
//! is noise and we punt.
//!
//! Reference numbers to compare against:
//!
//! * Live bench (`project_bitnet_live_bench.md`): 66 tok/s @64-tok, 33
//!   tok/s @1024-tok on gfx1151 → 15 ms–30 ms per forward step.
//! * A sampler cost >1.5 ms would be ≥5% of the 30 ms step and worth
//!   pipelining. <750 µs on the 15 ms step → noise, punt.
//!
//! ## How to run
//!
//! ```sh
//! cargo bench -p onebit-server --bench sampler
//! # or
//! cargo run -p onebit-server --release --bench sampler
//! ```
//!
//! The bench is a plain `main()` with `harness = false` so it works on
//! stable Rust without `criterion` / the unstable `test::Bencher` harness.
//! Output goes to stdout as a short summary + the raw measured medians.

use std::time::Instant;

use onebit_core::sampler::{Sampler, SamplerConfig};
use onebit_router::cpu_lane::CpuLane;

const VOCAB: usize = 128_256;
/// How many samples per configuration. 500 is enough to let the median
/// smooth over scheduler jitter but short enough to keep the whole run
/// under a second on a warm box.
const ITERS: usize = 500;
/// How many warm-up runs before timing starts. JIT-style code-cache warm
/// plus allocator steady-state.
const WARMUP: usize = 50;

fn make_logits(seed: u64) -> Vec<f32> {
    // Deterministic, non-degenerate logits. fastrand gives us a cheap RNG
    // already in the tree (workspace dep), same one the production sampler
    // uses, so we don't drift distributions between bench and prod.
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..VOCAB).map(|_| rng.f32() * 10.0 - 5.0).collect()
}

fn median_ns(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn bench<F: FnMut() -> u32>(name: &str, mut f: F) -> u128 {
    // Warm up — don't measure.
    for _ in 0..WARMUP {
        let _ = std::hint::black_box(f());
    }

    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let out = f();
        let dt = start.elapsed().as_nanos();
        std::hint::black_box(out);
        samples.push(dt);
    }
    let med = median_ns(samples);
    // Micro precision for the eyeball; the nanosecond field is kept for
    // machine parsing (grep / awk in benchmark scripts).
    println!(
        "  {name:<40} median = {:>8.3} µs  ({med} ns)",
        med as f64 / 1_000.0,
    );
    med
}

fn main() {
    println!("sampler microbench  (vocab={VOCAB}, iters={ITERS}, warmup={WARMUP})");
    println!(
        "build = {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    println!();

    // ---- Inline (single-threaded) path ------------------------------------
    // Exercise the exact same function the router calls today in
    // `generate_blocking`: `Sampler::sample(logits, history)` for temp>0,
    // `Sampler::greedy(logits)` for temp=0. Each call mutates its logits
    // in place so we rebuild the vector between samples.

    println!("inline Sampler (onebit-core):");

    // temp=0 → greedy argmax (this is what we hit today because the GPU
    // also returns the argmax, so the inline sampler call is skipped when
    // temp<=0 — but we still measure it so we know the lower bound).
    let logits = make_logits(0xA1);
    let greedy_inline = bench("greedy (temp=0.0)", || {
        Sampler::greedy(std::hint::black_box(&logits)).unwrap() as u32
    });

    // temp=0.7, top_k=50, top_p=0.95 — a common "creative" setting.
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.95,
        rep_penalty: 1.1,
        rep_last_n: 64,
        seed: 0xC0FFEE,
    });
    let base = make_logits(0xA2);
    let history: Vec<i32> = (0..128).collect();
    let full_inline = bench("full path (T=0.7, k=50, p=0.95)", || {
        let mut logits = base.clone();
        sampler.sample(&mut logits, &history).unwrap() as u32
    });

    // temp=1.0 no-top-k no-top-p path — the other interesting upper
    // bound (softmax over the whole vocab, no nth_element prune).
    let mut sampler_full = Sampler::new(SamplerConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        rep_penalty: 1.0,
        rep_last_n: 0,
        seed: 0x1234,
    });
    let multinomial_inline = bench("multinomial only (T=1.0, k=0, p=1.0)", || {
        let mut logits = base.clone();
        sampler_full.sample(&mut logits, &history).unwrap() as u32
    });

    // ---- Parallel lane path ----------------------------------------------
    // CpuLane::parallel_sample is the scaffold we're deciding whether to
    // wire. Today it only implements argmax + top-k (no top-p, no rep
    // penalty) — so comparing against the inline "greedy" number is the
    // fair apples-to-apples check.

    println!();
    println!("CpuLane::parallel_sample (onebit-router):");

    for n in [1usize, 4, 8, 14] {
        let lane = CpuLane::with_threads(n).expect("build lane");
        let key = format!("parallel argmax (threads={n})");
        bench(&key, || lane.parallel_sample(&logits, 1, 1.0, 0.0));
    }

    // ---- Decision summary ------------------------------------------------
    //
    // Compare sampler cost vs forward-pass cost. Forward-pass floor is
    // ~15 ms (fast path, 64-tok context). If the full sampler costs less
    // than 5% of that (750 µs) the pipelined gain is noise.

    println!();
    println!("decision heuristics:");
    let greedy_us = greedy_inline as f64 / 1_000.0;
    let full_us = full_inline as f64 / 1_000.0;
    let multi_us = multinomial_inline as f64 / 1_000.0;
    let fwd_fast_ms = 15.0;
    let fwd_slow_ms = 30.0;

    println!(
        "  greedy:       {:.3} µs = {:.2}% of 15 ms forward, {:.2}% of 30 ms",
        greedy_us,
        greedy_us / (fwd_fast_ms * 1_000.0) * 100.0,
        greedy_us / (fwd_slow_ms * 1_000.0) * 100.0,
    );
    println!(
        "  full path:    {:.3} µs = {:.2}% of 15 ms forward, {:.2}% of 30 ms",
        full_us,
        full_us / (fwd_fast_ms * 1_000.0) * 100.0,
        full_us / (fwd_slow_ms * 1_000.0) * 100.0,
    );
    println!(
        "  multinomial:  {:.3} µs = {:.2}% of 15 ms forward, {:.2}% of 30 ms",
        multi_us,
        multi_us / (fwd_fast_ms * 1_000.0) * 100.0,
        multi_us / (fwd_slow_ms * 1_000.0) * 100.0,
    );
    println!();
    let worth_pipelining = full_us > 750.0 || multi_us > 750.0;
    if worth_pipelining {
        println!("  VERDICT: sampler ≥5% of the 15 ms forward path — pipelining is on the table.");
    } else {
        println!("  VERDICT: sampler <5% of the 15 ms forward path — pipelining gain is noise.");
        println!("  Recommendation: land the env-gated switch so the lane is measurable,");
        println!("  but don't claim a speedup we can't measure. Real CPU-lane win lives");
        println!("  elsewhere (tokenizer offload, stop-string match).");
    }
}
