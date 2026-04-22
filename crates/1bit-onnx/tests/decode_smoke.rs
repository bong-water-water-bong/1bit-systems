//! End-to-end smoke test: load the TriLM 3.9B int4 artifact, run 8 greedy
//! tokens of prefill + decode, print tok/s.
//!
//! Marked `#[ignore]` — CI boxes don't carry the 2.9 GB artifact and won't
//! have a ready libonnxruntime.so of the right version on LD_LIBRARY_PATH.
//! Run on the strix-halo box with:
//!
//! ```
//! ORT_DYLIB_PATH=/home/bcloud/1bit-halo-core/tools/trilm-export/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4 \
//! cargo test -p onebit-onnx --test decode_smoke -- --ignored --nocapture
//! ```
//!
//! Intent: confirm that our inputs/outputs plumbing matches the exported
//! graph. NOT a perf bench — CPU decode on 3.9B at int4 is slow by design.
//! The real perf story is the HIP lane (BitNet 2B on gfx1151, 83 tok/s).
//! This test exists so that when AMD ships the Linux STX-H VitisAI EP we
//! already know our Rust-side decode loop works.

use onebit_onnx::{GenerateRequest, OnnxSession};

const ARTIFACT: &str = "/home/bcloud/1bit-halo-core/artifacts/trilm-3.9b-int4-n4";

#[test]
#[ignore = "needs 2.9 GB artifact + libonnxruntime.so.1.24.x on ORT_DYLIB_PATH"]
fn decode_eight_tokens_greedy() {
    // Sanity: bail early with a clear message if the artifact isn't here,
    // instead of chasing a confusing ORT error.
    assert!(
        std::path::Path::new(ARTIFACT).join("model.onnx").exists(),
        "artifact not found at {ARTIFACT}"
    );

    let mut session = OnnxSession::load(ARTIFACT).expect("load session");
    eprintln!("lane = {}", session.lane.label());

    let req = GenerateRequest::greedy("The capital of France is", /* max_new_tokens */ 8);
    let resp = session.generate(&req).expect("generate");

    eprintln!("prompt_tokens  = {}", resp.prompt_tokens);
    eprintln!("completion_tok = {}", resp.completion_tokens);
    eprintln!("wall_ms        = {}", resp.wall_ms);
    eprintln!("tok/s          = {:.2}", resp.tokens_per_second);
    eprintln!("output         = {:?}", resp.text);

    assert!(resp.completion_tokens > 0, "no tokens generated");
    assert!(resp.prompt_tokens > 0, "no prompt tokens encoded");
}
