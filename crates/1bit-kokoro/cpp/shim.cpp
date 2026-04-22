// cpp/shim.cpp — C-linkage stubs for the onebit_kokoro_* surface.
//
// This is the *skeleton* implementation: every entry point returns
// early with a placeholder result so the `real-kokoro` feature builds
// cleanly once it's enabled. Real kokoro.cpp wiring (onnxruntime
// session, voice pack load, phoneme pipeline, ONNX forward pass, i16
// conversion) lands in a follow-up — tracked in project_halo_kokoro.md.
//
// Keeping the shim as a zero-behaviour skeleton means:
//   * Rust FFI declarations stay honest (no dangling symbols),
//   * build.rs compiles something under `--features real-kokoro`,
//   * integration tests that enable the feature fail loudly with a
//     shim error code instead of linking against an absent symbol.

#include "shim.h"

#include <cstdint>
#include <cstring>

extern "C" {

struct KokoroCtx* onebit_kokoro_init(const char* model_path) {
    // Skeleton: no onnxruntime session built yet; refuse to hand back a
    // context that can't actually synthesize. Returning NULL makes the
    // Rust side bubble a ModelLoadFailed error, which is the correct
    // behaviour until the real wiring lands.
    (void)model_path;
    return nullptr;
}

void onebit_kokoro_free(struct KokoroCtx* ctx) {
    // Nothing to free yet. Tolerate NULL per the header contract.
    (void)ctx;
}

int64_t onebit_kokoro_synthesize(struct KokoroCtx* ctx,
                                 const char*       text,
                                 const char*       voice,
                                 float             speed,
                                 int16_t*          out_pcm,
                                 size_t            out_cap) {
    (void)ctx;
    (void)text;
    (void)voice;
    (void)speed;
    (void)out_pcm;
    (void)out_cap;
    // Return a negative status so any accidental call in skeleton mode
    // surfaces as a ShimError on the Rust side rather than silent
    // success. -5 matches the "onnxruntime error" slot in shim.h.
    return -5;
}

} // extern "C"
