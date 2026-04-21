// cpp/shim.h — C-linkage surface the Rust side FFIs to.
//
// The Rust crate cannot call whisper.h's C++-ish API directly (the header
// relies on struct forward-declarations that are easier to wrap once).
// This shim reduces the surface to a handful of plain C functions over
// opaque `WhisperCtx*` pointers.
//
// Two parallel output channels are exposed:
//
//   * `onebit_whisper_drain`           — legacy best-effort text blob
//                                         (concatenation of the current
//                                         sliding window's segments).
//   * `onebit_whisper_drain_segment`   — pop one segment record from a
//                                         monotonically-growing ring of
//                                         committed segments, with t0/t1
//                                         and an `is_final` flag.
//
// Both surfaces can coexist: `_drain` still snapshots the latest full
// rolling-window text on each tick, while `_drain_segment` delivers the
// per-segment firehose the Rust `WhisperStream` consumes. Callers that
// want one or the other pick; nothing stops them from using both.

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

// Pop one segment worth of text + timestamps from the committed-segment
// ring. Segments are pushed onto the ring in FIFO order by `_feed`'s
// sliding-window tick (draft segments) and by `_commit` (final segments);
// this call pops the oldest un-drained one.
//
//   * Returns 1 if a segment was written to the caller-owned buffers.
//   * Returns 0 if the ring is empty.
//   * Returns <0 on error: -1 for null pointer args, -2 if `out_text_cap`
//     is too small to hold the segment's text + NUL terminator (the
//     segment is *kept* on the ring so the caller can retry with a bigger
//     buffer — no silent truncation).
//
// `out_is_final` is set to 1 for segments produced by `_commit` (VAD tail
// / end-of-utterance commit), 0 for sliding-window drafts. Timestamps are
// reported in milliseconds relative to the start of the most recent
// sliding window the segment was decoded from — they are *local* to that
// window and the Rust side is responsible for translating them to stream
// time using its own sample counter.
int onebit_whisper_drain_segment(
    struct WhisperCtx* ctx,
    int64_t*           out_t0_ms,
    int64_t*           out_t1_ms,
    int*               out_is_final,
    char*              out_text,
    size_t             out_text_cap);

// Signal end-of-utterance (e.g. VAD tail fired). Runs one last
// `whisper_full` over the current sliding-window audio with
// `no_context=true, single_segment=false`, and pushes every resulting
// segment onto the ring stamped with `is_final=1`. The caller should
// follow up with `_drain_segment` calls until it returns 0 to harvest the
// committed text.
//
// Returns 0 on success, negative on error (null ctx, whisper_full
// failure).
int onebit_whisper_commit(struct WhisperCtx* ctx);

// Monotonic counter incremented once per segment pushed onto the ring.
// Intended for the Rust side's overlap-dedup heuristics (sliding windows
// re-decode the trailing audio, so the same segment may appear twice).
// Reading this is lock-free from the caller's perspective but internally
// takes the ring mutex for a consistent snapshot.
uint64_t onebit_whisper_seg_seq(const struct WhisperCtx* ctx);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ONEBIT_WHISPER_SHIM_H
