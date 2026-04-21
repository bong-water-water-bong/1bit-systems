// cpp/segment_ring.hpp — fixed-capacity FIFO ring of committed transcript
// segments, used by the streaming shim.
//
// Pulled out of shim.cpp so it can be unit-tested without a dependency
// on libwhisper. The shim itself holds one instance; tests instantiate
// their own and drive it synthetically.
//
// Threading: every public method acquires the internal `std::mutex`.
// Per architectural direction the structure does NOT use lock-free
// atomics for its bookkeeping — a single mutex is the whole story.

#ifndef ONEBIT_WHISPER_SEGMENT_RING_HPP
#define ONEBIT_WHISPER_SEGMENT_RING_HPP

#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

namespace onebit_whisper {

// One entry in the ring. `text` is owned std::string so we can push
// arbitrarily sized segments without a secondary allocation policy.
struct SegmentRecord {
    int64_t     t0_ms    = 0;
    int64_t     t1_ms    = 0;
    bool        is_final = false;
    std::string text;
};

// FIFO ring, capacity fixed at compile time. Older entries are
// overwritten when the ring is full — drafts are ephemeral and the
// caller is expected to drain at 500 ms cadence. Final (commit) segments
// can also be overwritten if the caller is severely behind; that is
// considered the caller's problem.
template <std::size_t Cap>
class SegmentRing {
    static_assert(Cap > 0, "SegmentRing capacity must be positive");

public:
    static constexpr std::size_t capacity = Cap;

    // Push a new segment. The internal sequence counter is incremented
    // exactly once per call. Returns the post-increment sequence number
    // (i.e. the 1-based index of this push). Never throws (std::string
    // copy may throw std::bad_alloc — but we let that propagate; a
    // broken allocator is not something the shim is going to paper
    // over).
    std::uint64_t push(SegmentRecord rec) {
        std::lock_guard<std::mutex> lock(mu_);
        slots_[head_] = std::move(rec);
        head_ = (head_ + 1) % Cap;
        if (size_ < Cap) {
            ++size_;
        } else {
            // Ring full — overwrite oldest by advancing tail.
            tail_ = (tail_ + 1) % Cap;
            ++dropped_;
        }
        ++seq_;
        return seq_;
    }

    // Pop the oldest entry into `out`. Returns true on success, false
    // if the ring is empty.
    bool pop(SegmentRecord& out) {
        std::lock_guard<std::mutex> lock(mu_);
        if (size_ == 0) {
            return false;
        }
        out = std::move(slots_[tail_]);
        slots_[tail_] = SegmentRecord{};
        tail_ = (tail_ + 1) % Cap;
        --size_;
        return true;
    }

    // Peek at the oldest entry's text size without popping, so a
    // bounds-check-first caller can refuse to pop if their buffer is
    // too small. Returns SIZE_MAX if the ring is empty (distinguishable
    // from a legitimate 0-byte text).
    std::size_t peek_text_size() const {
        std::lock_guard<std::mutex> lock(mu_);
        if (size_ == 0) {
            return static_cast<std::size_t>(-1);
        }
        return slots_[tail_].text.size();
    }

    // Current number of entries.
    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return size_;
    }

    // Monotonic sequence counter — one tick per push(), never
    // decreases, never resets on pop(). Intended for Rust-side overlap
    // dedup.
    std::uint64_t seq() const {
        std::lock_guard<std::mutex> lock(mu_);
        return seq_;
    }

    // Total number of pushes that overwrote a not-yet-popped entry.
    // Useful as a health metric; tests assert this stays zero.
    std::uint64_t dropped() const {
        std::lock_guard<std::mutex> lock(mu_);
        return dropped_;
    }

    // Clear all pending entries. Does NOT reset the sequence counter —
    // consumers rely on it being monotonic across the whole lifetime.
    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& s : slots_) {
            s = SegmentRecord{};
        }
        head_ = 0;
        tail_ = 0;
        size_ = 0;
    }

private:
    mutable std::mutex           mu_;
    std::array<SegmentRecord, Cap> slots_{};
    std::size_t                  head_    = 0;
    std::size_t                  tail_    = 0;
    std::size_t                  size_    = 0;
    std::uint64_t                seq_     = 0;
    std::uint64_t                dropped_ = 0;
};

} // namespace onebit_whisper

#endif // ONEBIT_WHISPER_SEGMENT_RING_HPP
