// cpp/shim.h — C-linkage surface the Rust side FFIs to.
//
// The Rust crate cannot call kokoro.hpp's C++-ish API directly (the
// header pulls in onnxruntime_cxx_api.h, which is templated C++). This
// shim reduces the surface to a handful of plain C functions over
// opaque `KokoroCtx*` pointers.
//
// First-pass skeleton: implementations return -1 until the kokoro.cpp
// upstream wiring lands. See project_halo_kokoro.md for the integration
// plan.

#ifndef ONEBIT_KOKORO_SHIM_H
#define ONEBIT_KOKORO_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle. Owned by the shim; Rust only ever holds a pointer.
struct KokoroCtx;

// Load a kokoro ONNX model from disk. Returns NULL on failure (bad
// path, OOM, corrupt file, onnxruntime init failure). Caller must
// eventually pass the returned pointer to `onebit_kokoro_free` exactly
// once.
struct KokoroCtx* onebit_kokoro_init(const char* model_path);

// Free a context previously returned by `onebit_kokoro_init`. Safe to
// pass NULL.
void onebit_kokoro_free(struct KokoroCtx* ctx);

// Synthesize `text` with `voice` at `speed`. On success, writes at most
// `out_cap` s16le samples (mono, 22 050 Hz) to `out_pcm` and returns
// the number of samples written. On failure returns negative:
//
//   * -1 — null pointer arg / ctx
//   * -2 — unknown voice id
//   * -3 — speed outside (0, 4]
//   * -4 — buffer too small (the shim refuses to truncate audio)
//   * -5 — onnxruntime error during forward pass
//
// `speed` is multiplied into the phoneme-duration prediction; upstream
// kokoro.cpp clamps to (0, 4] but we replicate the check on the Rust
// side too so bad inputs never cross the FFI boundary.
int64_t onebit_kokoro_synthesize(struct KokoroCtx* ctx,
                                 const char*       text,
                                 const char*       voice,
                                 float             speed,
                                 int16_t*          out_pcm,
                                 size_t            out_cap);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ONEBIT_KOKORO_SHIM_H
