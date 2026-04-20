// cpp/shim.cpp — implementation of the onebit_whisper_* C-linkage surface.
//
// Model: one-ctx-one-stream. The shim holds a `whisper_context*` plus a
// simple float32 ring buffer of recent audio. Every 500 ms worth of feed
// (measured in samples at 16 kHz = 8000 samples) we fire a `whisper_full`
// over the current sliding window and stash the resulting segments for
// subsequent `drain` calls to harvest.
//
// This is intentionally a minimal starting point — the full streaming
// design (VAD tail commit, per-stream `whisper_state`, dedup on overlap)
// lives in docs/wiki/Halo-Whisper-Streaming-Plan.md and will layer on
// top. The v1 goal is: load a model, feed PCM, drain a text blob.

#include "shim.h"

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

struct CtxImpl {
    whisper_context*      wctx     = nullptr;
    std::mutex            mu;
    // Accumulated PCM since last fire, plus the trailing kWindowSamples of
    // history we hand to whisper_full each tick.
    std::vector<float>    pcm_f32;
    size_t                unfired_since_tick = 0;
    // Latest segment text, rebuilt on each fire. Drained by Rust.
    std::string           latest_text;
};

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

    // Append incoming samples as f32 in [-1, 1].
    impl->pcm_f32.reserve(impl->pcm_f32.size() + n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        impl->pcm_f32.push_back(static_cast<float>(pcm[i]) / 32768.0f);
    }
    impl->unfired_since_tick += n_samples;

    // Fire every 500 ms of accumulated audio.
    if (impl->unfired_since_tick >= kStepSamples) {
        impl->unfired_since_tick = 0;

        // Keep only the trailing window.
        if (impl->pcm_f32.size() > kWindowSamples) {
            size_t drop = impl->pcm_f32.size() - kWindowSamples;
            impl->pcm_f32.erase(impl->pcm_f32.begin(),
                                impl->pcm_f32.begin() + drop);
        }

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
            // Leave latest_text untouched so drain still returns the prior
            // best hypothesis.
            return 0;
        }

        std::string rebuilt;
        const int n_seg = whisper_full_n_segments(impl->wctx);
        for (int i = 0; i < n_seg; ++i) {
            const char* t = whisper_full_get_segment_text(impl->wctx, i);
            if (t == nullptr) {
                continue;
            }
            if (!rebuilt.empty()) {
                rebuilt.push_back('\n');
            }
            rebuilt.append(t);
        }
        impl->latest_text = std::move(rebuilt);
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

} // extern "C"
