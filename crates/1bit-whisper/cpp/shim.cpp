// cpp/shim.cpp — implementation of the onebit_whisper_* C-linkage surface.
//
// Model: one-ctx-one-stream. The shim holds a `whisper_context*`, a f32
// ring buffer of recent audio, and a bounded ring of committed
// transcript segments (see segment_ring.hpp). Every 500 ms worth of feed
// we fire a `whisper_full` over the current sliding window and:
//
//   * rebuild `latest_text` (legacy `_drain` surface — the rolling-
//     window text blob), and
//   * push any *new* segments onto the segment ring with `is_final=0`
//     (sliding-window drafts).
//
// `_commit` flushes the audio tail with `no_context=true,
// single_segment=false` and pushes every resulting segment onto the ring
// with `is_final=1`. The Rust side dedups overlaps using `_seg_seq`.
//
// Streaming design doc: docs/wiki/Halo-Whisper-Streaming-Plan.md.

#include "shim.h"
#include "segment_ring.hpp"

#include <whisper.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

// 16 kHz mono in, 500 ms window.
constexpr int     kSampleRate    = 16000;
constexpr size_t  kStepSamples   = kSampleRate / 2; // 500 ms -> 8000
// Maximum sliding window we will hand whisper_full. 5 s keeps the encoder
// cost bounded while retaining word-boundary context.
constexpr size_t  kWindowSamples = kSampleRate * 5;

// Ring capacity. 32 was chosen because a 500 ms tick emits 1–3 segments
// on typical English speech, giving the Rust side ~5–15 s of slack
// before overflow — well beyond any realistic drain latency.
constexpr size_t  kRingCapacity  = 32;

using onebit_whisper::SegmentRecord;
using onebit_whisper::SegmentRing;

struct CtxImpl {
    whisper_context*          wctx     = nullptr;
    std::mutex                mu;       // guards audio + latest_text
    std::vector<float>        pcm_f32;
    size_t                    unfired_since_tick = 0;
    std::string               latest_text;        // legacy `_drain`

    // Segment ring has its own internal mutex; no locking order issue
    // because we never hold `mu` and the ring's mutex simultaneously
    // on a code path that could ping-pong.
    SegmentRing<kRingCapacity> ring;

    // Last t1 (in ms, local to the window) we pushed onto the ring from
    // a sliding-window tick. New tick segments must advance past this
    // to be considered "new". Reset to 0 on `_commit` so the committed
    // pass can re-push everything as finals.
    //
    // NOTE: this is a whisper.cpp quirk we work around — each
    // `whisper_full` call resets the segment indexing to 0..N-1 for
    // *that call's* output, and timestamps are local to the window we
    // handed it. We cannot use a global segment index as the dedup key;
    // we use the window-local t1 as a "high water mark" and assume
    // segments come out in monotonic order within a single decode,
    // which whisper.cpp guarantees.
    int64_t                   last_window_t1_ms = 0;
};

// Convert a PCM accumulator pointer into the ring'd f32 buffer.
// Not a method because it's trivially free.
void PushPcm(CtxImpl* impl, const int16_t* pcm, size_t n_samples) {
    impl->pcm_f32.reserve(impl->pcm_f32.size() + n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        impl->pcm_f32.push_back(static_cast<float>(pcm[i]) / 32768.0f);
    }
}

// Trim `pcm_f32` down to at most `kWindowSamples`, dropping the oldest.
void TrimToWindow(CtxImpl* impl) {
    if (impl->pcm_f32.size() > kWindowSamples) {
        size_t drop = impl->pcm_f32.size() - kWindowSamples;
        impl->pcm_f32.erase(impl->pcm_f32.begin(),
                            impl->pcm_f32.begin() + drop);
    }
}

// Harvest segments from the most recent whisper_full and push the ones
// newer than `last_window_t1_ms` onto the ring. Also rebuilds
// `latest_text`. Caller must hold `impl->mu`.
void HarvestLocked(CtxImpl* impl, bool is_final) {
    std::string rebuilt;
    const int n_seg = whisper_full_n_segments(impl->wctx);
    int64_t highest_t1 = impl->last_window_t1_ms;

    for (int i = 0; i < n_seg; ++i) {
        const char* t = whisper_full_get_segment_text(impl->wctx, i);
        if (t == nullptr) {
            continue;
        }
        if (!rebuilt.empty()) {
            rebuilt.push_back('\n');
        }
        rebuilt.append(t);

        // Timestamps are in 10 ms units per whisper.cpp convention (see
        // WHISPER_API whisper_full_get_segment_t0 docs).
        const int64_t t0_cs = whisper_full_get_segment_t0(impl->wctx, i);
        const int64_t t1_cs = whisper_full_get_segment_t1(impl->wctx, i);
        const int64_t t0_ms = t0_cs * 10;
        const int64_t t1_ms = t1_cs * 10;

        // Dedup: for drafts, skip if t1 hasn't advanced past our high
        // water mark (same segment re-decoded by the sliding window).
        // For commits, we already reset last_window_t1_ms=0 so
        // everything passes through.
        if (!is_final && t1_ms <= impl->last_window_t1_ms) {
            continue;
        }

        SegmentRecord rec;
        rec.t0_ms    = t0_ms;
        rec.t1_ms    = t1_ms;
        rec.is_final = is_final;
        rec.text.assign(t);
        impl->ring.push(std::move(rec));

        if (t1_ms > highest_t1) {
            highest_t1 = t1_ms;
        }
    }
    impl->latest_text     = std::move(rebuilt);
    impl->last_window_t1_ms = highest_t1;
}

} // namespace

extern "C" {

struct WhisperCtx* onebit_whisper_init(const char* model_path) {
    if (model_path == nullptr) {
        return nullptr;
    }
    auto cparams = whisper_context_default_params();
    whisper_context* wctx =
        whisper_init_from_file_with_params(model_path, cparams);
    if (wctx == nullptr) {
        return nullptr;
    }
    auto* impl = new (std::nothrow) CtxImpl();
    if (impl == nullptr) {
        whisper_free(wctx);
        return nullptr;
    }
    impl->wctx = wctx;
    impl->pcm_f32.reserve(kWindowSamples * 2);
    return reinterpret_cast<WhisperCtx*>(impl);
}

void onebit_whisper_free(struct WhisperCtx* ctx) {
    if (ctx == nullptr) {
        return;
    }
    auto* impl = reinterpret_cast<CtxImpl*>(ctx);
    if (impl->wctx != nullptr) {
        whisper_free(impl->wctx);
        impl->wctx = nullptr;
    }
    delete impl;
}

int32_t onebit_whisper_feed(struct WhisperCtx* ctx,
                            const int16_t* pcm,
                            size_t n_samples) {
    if (ctx == nullptr || (pcm == nullptr && n_samples > 0)) {
        return -1;
    }
    auto* impl = reinterpret_cast<CtxImpl*>(ctx);
    std::lock_guard<std::mutex> lock(impl->mu);

    PushPcm(impl, pcm, n_samples);
    impl->unfired_since_tick += n_samples;

    // Fire every 500 ms of accumulated audio.
    if (impl->unfired_since_tick >= kStepSamples) {
        impl->unfired_since_tick = 0;
        TrimToWindow(impl);

        whisper_full_params wparams =
            whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_realtime   = false;
        wparams.print_progress   = false;
        wparams.print_timestamps = false;
        wparams.print_special    = false;
        wparams.translate        = false;
        wparams.single_segment   = true;
        wparams.no_context       = false;
        wparams.language         = "en";

        const int rc = whisper_full(impl->wctx,
                                    wparams,
                                    impl->pcm_f32.data(),
                                    static_cast<int>(impl->pcm_f32.size()));
        if (rc != 0) {
            // Leave latest_text / ring untouched so previously-pushed
            // segments and prior best hypothesis remain drainable.
            return 0;
        }
        HarvestLocked(impl, /*is_final=*/false);
    }
    return 0;
}

int32_t onebit_whisper_drain(struct WhisperCtx* ctx,
                             char* out_buf,
                             size_t out_len) {
    if (ctx == nullptr || out_buf == nullptr || out_len == 0) {
        return -1;
    }
    auto* impl = reinterpret_cast<CtxImpl*>(ctx);
    std::lock_guard<std::mutex> lock(impl->mu);

    const size_t n = impl->latest_text.size();
    const size_t copy = (n + 1 <= out_len) ? n : (out_len - 1);
    std::memcpy(out_buf, impl->latest_text.data(), copy);
    out_buf[copy] = '\0';
    return static_cast<int32_t>(copy);
}

int onebit_whisper_drain_segment(
    struct WhisperCtx* ctx,
    int64_t*           out_t0_ms,
    int64_t*           out_t1_ms,
    int*               out_is_final,
    char*              out_text,
    size_t             out_text_cap) {
    if (ctx == nullptr || out_t0_ms == nullptr || out_t1_ms == nullptr ||
        out_is_final == nullptr || out_text == nullptr || out_text_cap == 0) {
        return -1;
    }
    auto* impl = reinterpret_cast<CtxImpl*>(ctx);

    // Bounds-check before popping so we never lose a segment to an
    // undersized caller buffer. `peek_text_size` returns SIZE_MAX if
    // the ring is empty.
    const size_t peeked = impl->ring.peek_text_size();
    if (peeked == static_cast<size_t>(-1)) {
        return 0;
    }
    if (peeked + 1 > out_text_cap) {
        return -2;
    }

    SegmentRecord rec;
    if (!impl->ring.pop(rec)) {
        // Race: another thread popped between peek and pop. Treat as
        // "no segment available" rather than error.
        return 0;
    }

    *out_t0_ms    = rec.t0_ms;
    *out_t1_ms    = rec.t1_ms;
    *out_is_final = rec.is_final ? 1 : 0;

    std::memcpy(out_text, rec.text.data(), rec.text.size());
    out_text[rec.text.size()] = '\0';
    return 1;
}

int onebit_whisper_commit(struct WhisperCtx* ctx) {
    if (ctx == nullptr) {
        return -1;
    }
    auto* impl = reinterpret_cast<CtxImpl*>(ctx);
    std::lock_guard<std::mutex> lock(impl->mu);

    if (impl->pcm_f32.empty()) {
        // Nothing buffered — still "succeed", callers that invoke
        // commit on an empty stream are not in an error condition.
        return 0;
    }

    TrimToWindow(impl);

    whisper_full_params wparams =
        whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = false;
    wparams.print_special    = false;
    wparams.translate        = false;
    wparams.single_segment   = false; // allow natural sentence splits
    wparams.no_context       = true;  // fresh decode, no carryover bias
    wparams.language         = "en";

    const int rc = whisper_full(impl->wctx,
                                wparams,
                                impl->pcm_f32.data(),
                                static_cast<int>(impl->pcm_f32.size()));
    if (rc != 0) {
        return -3;
    }

    // Reset the high-water mark so *all* segments from this pass are
    // considered fresh and get pushed with is_final=1.
    impl->last_window_t1_ms = 0;
    HarvestLocked(impl, /*is_final=*/true);
    // After a commit the draft dedup state is no longer meaningful for
    // subsequent sliding-window ticks — a new utterance starts fresh.
    // Clear the pcm buffer and unfired counter so the next `_feed`
    // begins cleanly.
    impl->pcm_f32.clear();
    impl->unfired_since_tick = 0;
    impl->last_window_t1_ms  = 0;
    return 0;
}

uint64_t onebit_whisper_seg_seq(const struct WhisperCtx* ctx) {
    if (ctx == nullptr) {
        return 0;
    }
    const auto* impl = reinterpret_cast<const CtxImpl*>(ctx);
    return impl->ring.seq();
}

} // extern "C"
