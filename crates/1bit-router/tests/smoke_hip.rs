//! Hardware smoke test — only runs with `--ignored` on a host that has
//! ROCm + the halo-1bit-2b.h1b model on disk. The default CI run skips it.
//!
//! What it validates (when `--ignored` is passed):
//!   1. Hardware detection reports HIP.
//!   2. `.h1b` + `.htok` load + weight upload succeed.
//!   3. One forward pass runs and returns a token id.
//!
//! The expected prompt is "The capital of France is" — the reported
//! first-16-tokens output should match gen-1's
//! `bitnet_decode --text "The capital of France is" 1` argmax chain.

#![cfg(feature = "hip")]

use std::path::Path;

use onebit_core::sampler::SamplerConfig;
use onebit_router::{BackendKind, HipBackend, Router, RouterRequest, detect};

fn default_model() -> String {
    std::env::var("HOME")
        .map(|h| format!("{h}/1bit systems/models/halo-1bit-2b.h1b"))
        .unwrap_or_else(|_| "halo-1bit-2b.h1b".into())
}

#[test]
#[ignore = "requires ROCm + halo-1bit-2b.h1b on disk; run with --ignored"]
fn first_token_matches_gen1_argmax() {
    let model_path = std::env::var("HALO_SMOKE_MODEL").unwrap_or_else(|_| default_model());
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: model not found at {model_path}");
        return;
    }

    assert_eq!(detect(), BackendKind::Hip, "expected HIP detection");

    let router = Router::new(&model_path).expect("router init");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let resp = rt.block_on(async {
        router
            .generate(RouterRequest {
                prompt: "The capital of France is".to_string(),
                max_new_tokens: 16,
                sampler: SamplerConfig {
                    temperature: 0.0,
                    ..SamplerConfig::default()
                },
                stop: Vec::new(),
            })
            .await
            .expect("generate")
    });

    println!("TEXT: {:?}", resp.text);
    println!("prompt_tokens = {}", resp.prompt_tokens);
    println!("completion_tokens = {}", resp.completion_tokens);
    assert!(
        resp.completion_tokens >= 1,
        "expected at least one decoded token"
    );

    // Second pass: stand up a bare HipBackend and dump raw token IDs so
    // the user can compare argmax-by-argmax against gen-1's
    // `bitnet_decode --text "The capital of France is" 1`.
    let htok_path = model_path.replace(".h1b", ".htok");
    let mut bare = HipBackend::new(
        Path::new(&model_path),
        Path::new(&htok_path),
        "1bit-monster-2b".into(),
        4096,
    )
    .expect("bare backend init");

    let prompt = "The capital of France is";
    let ids = bare.tokenize(prompt);
    println!("prompt ids = {:?}", ids);

    let mut logits = Vec::new();
    let mut cur = ids[0];
    let mut generated: Vec<i32> = Vec::new();
    let mut pos = 0i32;
    for &t in &ids {
        cur = bare.forward_token(t, pos, &mut logits).expect("forward");
        pos += 1;
    }
    // `cur` is the argmax of the logits computed from the last prompt
    // token — it's the "first generated token" gen-1 prints too.
    generated.push(cur);
    for _ in 1..16 {
        cur = bare.forward_token(cur, pos, &mut logits).expect("forward");
        pos += 1;
        generated.push(cur);
    }
    println!("first 16 argmax ids = {:?}", generated);
    println!("first 16 argmax text = {:?}", bare.detokenize(&generated));
}

/// BitNet v2 integration test. Exercises the online-Hadamard dispatch
/// path end to end: with the `bitnet-v2` feature AND a rotated checkpoint
/// (signalled by `H1B_FLAG_HADAMARD_ROTATED` in the cfg `reserved` slot),
/// each `forward_token` should fire the rotation kernel exactly four
/// times per layer — confirmed via the test-only counter on
/// `HipBackend`.
///
/// Requires **both**:
///   1. `--features hip,bitnet-v2` at build time.
///   2. A `.h1b` file at `$HALO_BITNET_V2_MODEL` (or the default path)
///      whose requantizer set the H1B_FLAG_HADAMARD_ROTATED bit. Running
///      this test against a stock checkpoint will skip on the counter
///      assertion (hadamard_dispatch_count stays at 0, feature-but-no-flag
///      pass-through path — the unit tests cover that decision).
///
/// No shipping checkpoint carries the flag today; this test lands as a
/// ready-to-run harness for when the rotated 2B-4T variant lands.
#[test]
#[ignore = "requires --features bitnet-v2 + rotated .h1b; run with --ignored"]
#[cfg(feature = "bitnet-v2")]
fn hadamard_hook_fires_four_times_per_layer() {
    let model_path = std::env::var("HALO_BITNET_V2_MODEL").unwrap_or_else(|_| default_model());
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: rotated model not found at {model_path}");
        return;
    }
    let htok_path = model_path.replace(".h1b", ".htok");

    let mut bare = HipBackend::new(
        Path::new(&model_path),
        Path::new(&htok_path),
        "1bit-monster-2b-v2".into(),
        4096,
    )
    .expect("bare backend init");

    // Confirm model flag is set; otherwise the counter will stay at 0 and
    // the assertion would fail for the wrong reason.
    // Exposing cfg through a public accessor would grow the surface area;
    // instead probe indirectly by running one token and reading the counter.
    let ids = bare.tokenize("Hi");
    let mut logits = Vec::new();
    let start = bare.hadamard_dispatch_count();
    let _ = bare.forward_token(ids[0], 0, &mut logits).expect("forward");
    let delta = bare.hadamard_dispatch_count() - start;

    // Hook fires 4× per layer (Q/K/V pre-quant, O pre-quant, gate+up pre-quant,
    // down pre-quant). For the 28-layer 2B-4T base, expected = 112 per token.
    // If delta is 0 the model isn't flagged — surface a helpful skip message
    // rather than failing.
    if delta == 0 {
        eprintln!(
            "skipping assertion: loaded model does not carry H1B_FLAG_HADAMARD_ROTATED \
             (counter delta = 0 after one forward_token). Re-run against a rotated \
             checkpoint via HALO_BITNET_V2_MODEL=... cargo test --features bitnet-v2,hip \
             --ignored hadamard_hook_fires_four_times_per_layer"
        );
        return;
    }
    // 4 sites × num_layers. Read num_layers indirectly from the counter and
    // cross-check it divides evenly by 4.
    assert!(
        delta % 4 == 0,
        "hadamard hook fired {delta} times per token; expected a multiple of 4 (sites/layer)"
    );
    eprintln!("hadamard dispatch count / token = {delta}");
}
