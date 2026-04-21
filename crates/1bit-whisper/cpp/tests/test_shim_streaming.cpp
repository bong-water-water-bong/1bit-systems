// test_shim_streaming.cpp — unit coverage for the segment ring that backs
// `onebit_whisper_drain_segment` / `_commit` / `_seg_seq`.
//
// This test targets `segment_ring.hpp` directly rather than the full shim:
// the shim requires libwhisper + a real ggml model + 16 kHz PCM to
// exercise end-to-end, none of which belongs in a unit-test harness.
// What the shim adds on top of the ring is:
//
//   * A dedup high-water mark (`last_window_t1_ms`) that skips drafts
//     whose t1 <= previous high; committed pushes reset this to 0.
//   * `is_final` stamping (drafts = false, commits = true).
//
// Both of those are simple bool/int64 comparisons — they carry no code
// paths this test could meaningfully assert beyond the ring's FIFO and
// sequence semantics covered here.
//
// Build:
//
//     cmake -S . -B build -DONEBIT_WHISPER_BUILD_SHIM_TESTS=ON
//     cmake --build build
//     ./build/test_shim_streaming
//
// The CMake gate keeps this out of the default crate build — cargo
// never invokes cmake; this is a developer-side sanity harness.

#include "../segment_ring.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using onebit_whisper::SegmentRecord;
using onebit_whisper::SegmentRing;

namespace {

int g_failed = 0;

#define EXPECT(cond)                                                        \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d  %s\n",                        \
                         __FILE__, __LINE__, #cond);                        \
            ++g_failed;                                                     \
        }                                                                   \
    } while (0)

SegmentRecord mk(std::int64_t t0, std::int64_t t1, bool is_final, std::string text) {
    SegmentRecord r;
    r.t0_ms    = t0;
    r.t1_ms    = t1;
    r.is_final = is_final;
    r.text     = std::move(text);
    return r;
}

// Test 1: the canonical spec path — feed synthetic "draft" segments,
// drain two, commit, drain the final. Verifies:
//   - Drain pops in FIFO order
//   - seq counter advances monotonically on push
//   - is_final flag travels correctly
//   - Drain past the end returns false (the "no segment available" case
//     that maps to `_drain_segment` returning 0)
void test_feed_drain_commit_drain_final() {
    SegmentRing<32> ring;

    // Shim-side "sliding-window tick": push 4 draft segments.
    ring.push(mk(0,    480,  false, "hello"));
    ring.push(mk(480,  920,  false, "world"));
    ring.push(mk(920,  1400, false, "how"));
    ring.push(mk(1400, 1880, false, "are you"));

    EXPECT(ring.size() == 4);
    EXPECT(ring.seq()  == 4);

    // Drain twice (caller consumes first two drafts).
    SegmentRecord out;
    EXPECT(ring.pop(out));
    EXPECT(out.text == "hello");
    EXPECT(out.is_final == false);
    EXPECT(out.t0_ms == 0 && out.t1_ms == 480);

    EXPECT(ring.pop(out));
    EXPECT(out.text == "world");
    EXPECT(out.is_final == false);
    EXPECT(ring.size() == 2);
    // seq does NOT decrement on pop.
    EXPECT(ring.seq()  == 4);

    // Shim-side "commit": push committed segments (is_final=true).
    ring.push(mk(0,    1880, true, "hello world how are you"));
    EXPECT(ring.size() == 3);
    EXPECT(ring.seq()  == 5);

    // Drain the remaining two drafts first (FIFO order).
    EXPECT(ring.pop(out));
    EXPECT(out.text == "how");
    EXPECT(out.is_final == false);

    EXPECT(ring.pop(out));
    EXPECT(out.text == "are you");
    EXPECT(out.is_final == false);

    // Now the committed segment.
    EXPECT(ring.pop(out));
    EXPECT(out.text == "hello world how are you");
    EXPECT(out.is_final == true);

    // Empty ring: pop returns false (maps to `_drain_segment` == 0).
    EXPECT(!ring.pop(out));
    EXPECT(ring.size() == 0);
    EXPECT(ring.seq()  == 5); // still 5 — seq is lifetime-monotonic
    EXPECT(ring.dropped() == 0);
}

// Test 2: peek_text_size surfaces the "buffer too small" case before
// pop. This is exactly the dance `_drain_segment` does to return -2
// without losing the segment.
void test_peek_text_size_for_bounds_check() {
    SegmentRing<8> ring;

    // Empty ring -> SIZE_MAX sentinel.
    EXPECT(ring.peek_text_size() == static_cast<std::size_t>(-1));

    ring.push(mk(0, 500, false, "abcdefghij")); // 10 bytes
    EXPECT(ring.peek_text_size() == 10);

    // Peek is non-destructive.
    EXPECT(ring.size() == 1);
    EXPECT(ring.peek_text_size() == 10);
    EXPECT(ring.size() == 1);

    // Pop consumes; next peek sees the empty sentinel.
    SegmentRecord out;
    EXPECT(ring.pop(out));
    EXPECT(ring.peek_text_size() == static_cast<std::size_t>(-1));
}

// Test 3: overflow behaviour. With capacity 4, pushing 6 should
// overwrite the two oldest entries and bump `dropped()` twice.
void test_overflow_overwrites_oldest() {
    SegmentRing<4> ring;
    for (int i = 0; i < 6; ++i) {
        ring.push(mk(i * 100, (i + 1) * 100, false,
                     std::string("seg") + char('0' + i)));
    }
    EXPECT(ring.size()    == 4);
    EXPECT(ring.seq()     == 6);
    EXPECT(ring.dropped() == 2);

    // The two oldest (seg0, seg1) were overwritten; first pop should
    // now yield seg2.
    SegmentRecord out;
    EXPECT(ring.pop(out));
    EXPECT(out.text == "seg2");
    EXPECT(ring.pop(out));
    EXPECT(out.text == "seg3");
    EXPECT(ring.pop(out));
    EXPECT(out.text == "seg4");
    EXPECT(ring.pop(out));
    EXPECT(out.text == "seg5");
    EXPECT(!ring.pop(out));
}

// Test 4: contended push/pop across two threads. A brittle timing test
// is not the goal; the goal is to prove that the mutex-guarded ring
// doesn't corrupt state under concurrent access — ASAN/TSAN in CI
// would catch races if they existed.
void test_threaded_push_pop_is_safe() {
    SegmentRing<64> ring;
    constexpr int N = 1000;

    std::thread producer([&] {
        for (int i = 0; i < N; ++i) {
            ring.push(mk(i, i + 1, (i & 1) != 0,
                         std::string("t") + std::to_string(i)));
        }
    });

    std::thread consumer([&] {
        int popped = 0;
        SegmentRecord out;
        while (popped < N) {
            if (ring.pop(out)) {
                ++popped;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT(ring.size() == 0);
    EXPECT(ring.seq()  == static_cast<std::uint64_t>(N));
}

} // namespace

int main() {
    test_feed_drain_commit_drain_final();
    test_peek_text_size_for_bounds_check();
    test_overflow_overwrites_oldest();
    test_threaded_push_pop_is_safe();

    if (g_failed == 0) {
        std::printf("ok: all segment-ring tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "FAIL: %d assertion(s) failed\n", g_failed);
    return 1;
}
