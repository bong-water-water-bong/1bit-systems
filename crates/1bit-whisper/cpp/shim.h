// cpp/shim.h — C-linkage surface the Rust side FFIs to.
//
// The Rust crate cannot call whisper.h's C++-ish API directly (the header
// relies on struct forward-declarations that are easier to wrap once).
// This shim reduces the surface to four plain C functions over opaque
// `WhisperCtx*` pointers.

#ifndef ONEBIT_WHISPER_SHIM_H
#define ONEBIT_WHISPER_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle. Owned by the shim; Rust only ever holds a pointer.
struct WhisperCtx;

// Load a ggml whisper model from disk. Returns NULL on failure (bad path,
// OOM, corrupt file). Caller must eventually pass the returned pointer to
// `onebit_whisper_free` exactly once.
struct WhisperCtx* onebit_whisper_init(const char* model_path);

// Free a context previously returned by `onebit_whisper_init`. Safe to
// pass NULL.
void onebit_whisper_free(struct WhisperCtx* ctx);

// Append `n_samples` of mono 16 kHz s16le PCM to the context's internal
// ring buffer. Every 500 ms worth of accumulated audio, the shim fires a
// `whisper_full` over the current sliding window so subsequent `drain`
// calls can harvest fresh segments.
//
// Returns 0 on success, negative on error (null ctx, allocation failure).
int32_t onebit_whisper_feed(struct WhisperCtx* ctx,
                            const int16_t* pcm,
                            size_t n_samples);

// Copy the UTF-8 text of the most recently decoded segments into `out_buf`
// (size `out_len`), concatenated with '\n' separators. Always
// NUL-terminated. Returns the number of bytes written (excluding NUL), or
// negative on error.
int32_t onebit_whisper_drain(struct WhisperCtx* ctx,
                             char* out_buf,
                             size_t out_len);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ONEBIT_WHISPER_SHIM_H
