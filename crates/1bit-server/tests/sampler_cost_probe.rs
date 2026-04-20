//! Sampler wall-clock probe, wrapped as a `#[test] #[ignore]` so we can
//! run it through `cargo test` in sandboxed environments where direct
//! bench-binary execution is not permitted.
//!
//! Run with:
//!
//! ```sh
//! cargo test --release -p onebit-server --test sampler_cost_probe -- \
//!     --ignored --nocapture
//! ```
//!
//! Same measurements as `benches/sampler.rs` — kept the two in sync so
//! either entry point yields identical numbers. The "real" harness lives
//! in `benches/`; this shim just exists so CI / sandbox runs can still
//! capture the measurement.

use std::time::Instant;

use onebit_core::sampler::{Sampler, SamplerConfig};
use onebit_router::cpu_lane::CpuLane;

const VOCAB: usize = 128_256;
const ITERS: usize = 500;
const WARMUP: usize = 50;

fn make_logits(seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..VOCAB).map(|_| rng.f32() * 10.0 - 5.0).collect()
}

fn median_ns(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn probe<F: FnMut() -> u32>(name: &str, mut f: F) -> u128 {
    for _ in 0..WARMUP {
        std::hint::black_box(f());
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
    println!(
        "  {name:<40} median = {:>8.3} µs  ({med} ns)",
        med as f64 / 1_000.0,
    );
    med
}

#[test]
#[ignore]
fn sampler_cost_probe() {
    println!("\nsampler cost probe  (vocab={VOCAB}, iters={ITERS}, warmup={WARMUP})");
    println!(
        "build = {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    );
    println!();

    println!("inline Sampler (onebit-core):");
    let logits = make_logits(0xA1);
    let greedy_inline = probe("greedy (temp=0.0)", || {
        Sampler::greedy(std::hint::black_box(&logits)).unwrap() as u32
    });

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
    let full_inline = probe("full path (T=0.7, k=50, p=0.95)", || {
        let mut logits = base.clone();
        sampler.sample(&mut logits, &history).unwrap() as u32
    });

    let mut sampler_multi = Sampler::new(SamplerConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        rep_penalty: 1.0,
        rep_last_n: 0,
        seed: 0x1234,
    });
    let multinomial_inline = probe("multinomial only (T=1.0, k=0, p=1.0)", || {
        let mut logits = base.clone();
        sampler_multi.sample(&mut logits, &history).unwrap() as u32
    });

    println!();
    println!("CpuLane::parallel_sample (onebit-router):");
    for n in [1usize, 4, 8, 14] {
        let lane = CpuLane::with_threads(n).expect("build lane");
        let key = format!("parallel argmax (threads={n})");
        probe(&key, || lane.parallel_sample(&logits, 1, 1.0, 0.0));
    }

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
}
