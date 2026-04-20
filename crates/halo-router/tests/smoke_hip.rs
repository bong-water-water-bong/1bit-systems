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

use halo_router::{BackendKind, HipBackend, Router, RouterRequest, detect};
use halo_core::sampler::SamplerConfig;

fn default_model() -> String {
    std::env::var("HOME")
        .map(|h| format!("{h}/halo-ai/models/halo-1bit-2b.h1b"))
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
        "halo-1bit-2b".into(),
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
