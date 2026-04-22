//! One-shot parse smoke test against the actual converted Medusa file.
//!
//! Skipped by default — the path is hard-coded to the dev-box location
//! where `tools/medusa-convert/convert.py` writes its output. Run with:
//!
//! ```bash
//! cargo test -p onebit-router --release --test medusa_real_file -- --ignored
//! ```
//!
//! CPU-only. No GPU touched. Validates:
//!   1. the file opens cleanly,
//!   2. the header reports v2 + 4 heads + hd=2560 + 1 residual layer + fp16,
//!   3. every head view has the right `w_in` / `w_out` lengths,
//!   4. file size matches the header's declared total.

use onebit_router::medusa::heads::{MEDUSA_HIDDEN_DIM, NUM_MEDUSA_HEADS};
use onebit_router::medusa::loader::{
    MEDUSA_DTYPE_FP16, MEDUSA_FORMAT_VERSION, MEDUSA_RESIDUAL_LAYERS, MedusaHeadsFile,
};
use std::path::Path;

const REAL_MEDUSA_PATH: &str =
    "/home/bcloud/1bit-halo-models/models/medusa/halo-medusa-heads.h1b-medusa";

#[test]
#[ignore = "requires converted .h1b-medusa file on dev box; run with --ignored"]
fn opens_real_converted_file() {
    let path = Path::new(REAL_MEDUSA_PATH);
    assert!(
        path.exists(),
        "converted file missing at {} — run tools/medusa-convert/convert.py first",
        REAL_MEDUSA_PATH,
    );

    let file = MedusaHeadsFile::open(path).expect("must open the converted file cleanly");
    let hdr = file.header();
    assert_eq!(hdr.version, MEDUSA_FORMAT_VERSION);
    assert_eq!(hdr.num_heads as usize, NUM_MEDUSA_HEADS);
    assert_eq!(hdr.hidden_dim as usize, MEDUSA_HIDDEN_DIM);
    assert_eq!(hdr.residual_layers, MEDUSA_RESIDUAL_LAYERS);
    assert_eq!(hdr.dtype, MEDUSA_DTYPE_FP16);

    // Canonical shape → header + 4 × 25 MiB = 104_857_668 bytes.
    let expected_per_head = 2 * MEDUSA_HIDDEN_DIM * MEDUSA_HIDDEN_DIM * 2;
    assert_eq!(hdr.per_head_bytes().unwrap(), expected_per_head);
    let expected_total = 68 + NUM_MEDUSA_HEADS * expected_per_head;
    assert_eq!(file.mmap_len(), expected_total);

    // Walk each head and confirm the views have the right length.
    let hd = MEDUSA_HIDDEN_DIM;
    let expected_w_len = hd * hd;
    for i in 0..NUM_MEDUSA_HEADS {
        let view = file.head(i).expect("head within bounds");
        assert_eq!(view.w_in.len(), expected_w_len, "head {i} w_in length");
        assert_eq!(view.w_out.len(), expected_w_len, "head {i} w_out length");
    }
}
