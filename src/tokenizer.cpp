// BitNet tokenizer — pure C++ / zero Python at runtime.
//
// Reads the .htok binary emitted by halo-1bit/scripts/export_tokenizer.py
// (build-time Python is allowed per project rules; runtime is not).
//
// Scope:
//   * Byte-level BPE identical to tiktoken / LLaMA-3 at the merge level.
//   * GPT-2 byte<->unicode mapping (256-entry LUT) applied at the boundary.
//   * Regex pre-tokenizer is NOT implemented — we encode the whole input
//     as a single chunk. This matches HF reference tokenization on clean
//     ASCII text and diverges on some mixed-script / punctuation-heavy
//     inputs. Pre-tokenizer port is a follow-up.

#include "rocm_cpp/tokenizer.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// GPT-2 byte-to-unicode mapping (same table as tiktoken / LLaMA-3).
// Printable ASCII bytes map to themselves; the rest are relocated into
// U+0100..U+017F so every byte becomes a valid printable char.
struct ByteMap {
    std::array<uint32_t, 256> byte_to_cp;   // byte -> unicode codepoint
    std::array<int, 0x180>    cp_to_byte;   // codepoint (in used range) -> byte, or -1

    ByteMap() {
        cp_to_byte.fill(-1);
        std::vector<uint8_t> bs;
        for (int b = int('!'); b <= int('~'); ++b) bs.push_back((uint8_t)b);
        for (int b = 0xA1; b <= 0xAC; ++b)         bs.push_back((uint8_t)b);
        for (int b = 0xAE; b <= 0xFF; ++b)         bs.push_back((uint8_t)b);
        std::vector<uint32_t> cps;
        for (uint8_t b : bs) cps.push_back(b);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), (uint8_t)b) == bs.end()) {
                bs.push_back((uint8_t)b);
                cps.push_back(0x100u + (uint32_t)n);
                ++n;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            byte_to_cp[bs[i]] = cps[i];
            if (cps[i] < cp_to_byte.size()) cp_to_byte[cps[i]] = bs[i];
        }
    }
};

const ByteMap& byte_map() { static ByteMap bm; return bm; }

// Encode one codepoint to UTF-8. Returns bytes written (1..4).
int utf8_encode(uint32_t cp, char out[4]) {
    if (cp < 0x80)    { out[0] = (char)cp; return 1; }
    if (cp < 0x800)   { out[0] = (char)(0xC0 | (cp >> 6));       out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12));      out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6)  & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

// Decode one UTF-8 codepoint. Writes bytes_read, returns codepoint or 0xFFFD.
uint32_t utf8_decode(const uint8_t* s, size_t len, size_t& bytes_read) {
    if (len == 0) { bytes_read = 0; return 0; }
    uint8_t b0 = s[0];
    if (b0 < 0x80) { bytes_read = 1; return b0; }
    if ((b0 & 0xE0) == 0xC0 && len >= 2) {
        bytes_read = 2;
        return ((uint32_t)(b0 & 0x1F) << 6) | (s[1] & 0x3F);
    }
    if ((b0 & 0xF0) == 0xE0 && len >= 3) {
        bytes_read = 3;
        return ((uint32_t)(b0 & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
    }
    if ((b0 & 0xF8) == 0xF0 && len >= 4) {
        bytes_read = 4;
        return ((uint32_t)(b0 & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12)
             | ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    }
    bytes_read = 1;
    return 0xFFFD;
}

struct MergeKey {
    int32_t a, b;
    bool operator==(const MergeKey& o) const { return a == o.a && b == o.b; }
};
struct MergeKeyHash {
    size_t operator()(const MergeKey& k) const noexcept {
        return (size_t)k.a * 0x9E3779B97F4A7C15ull ^ (size_t)k.b;
    }
};

}  // namespace

struct rcpp_tokenizer {
    std::vector<std::string> id_to_bytes;               // token id -> raw bytes (GPT-2 mapped)
    std::unordered_map<std::string, int32_t> bytes_to_id;
    std::unordered_map<MergeKey, std::pair<int32_t, int32_t>, MergeKeyHash> merges;
    int32_t bos_id = 128000;
    int32_t eos_id = 128001;
};

extern "C" rcpp_status_t
rcpp_tokenizer_load(const char* path, rcpp_tokenizer_t** out)
{
    if (!path || !out) return RCPP_INVALID_ARG;
    std::ifstream f(path, std::ios::binary);
    if (!f) return RCPP_INVALID_ARG;

    char magic[4];
    f.read(magic, 4);
    if (std::strncmp(magic, "HTOK", 4) != 0) { fprintf(stderr, "bad .htok magic\n"); return RCPP_INVALID_ARG; }

    uint32_t vocab_size = 0, num_merges = 0, bos = 0, eos = 0;
    f.read(reinterpret_cast<char*>(&vocab_size), 4);
    f.read(reinterpret_cast<char*>(&num_merges), 4);
    f.read(reinterpret_cast<char*>(&bos), 4);
    f.read(reinterpret_cast<char*>(&eos), 4);

    auto t = new rcpp_tokenizer();
    t->bos_id = (int32_t)bos;
    t->eos_id = (int32_t)eos;
    t->id_to_bytes.resize(vocab_size);
    t->bytes_to_id.reserve(vocab_size);

    for (uint32_t i = 0; i < vocab_size; ++i) {
        uint16_t len = 0;
        f.read(reinterpret_cast<char*>(&len), 2);
        std::string s(len, '\0');
        if (len) f.read(s.data(), len);
        t->id_to_bytes[i] = s;
        if (!s.empty()) t->bytes_to_id[s] = (int32_t)i;
    }

    t->merges.reserve(num_merges);
    // Rank = insertion order; earlier merges have priority.
    for (uint32_t i = 0; i < num_merges; ++i) {
        uint32_t a = 0, b = 0, merged = 0;
        f.read(reinterpret_cast<char*>(&a), 4);
        f.read(reinterpret_cast<char*>(&b), 4);
        f.read(reinterpret_cast<char*>(&merged), 4);
        t->merges.emplace(MergeKey{(int32_t)a, (int32_t)b},
                          std::make_pair((int32_t)merged, (int32_t)i));
    }

    *out = t;
    return RCPP_OK;
}

extern "C" void rcpp_tokenizer_free(rcpp_tokenizer_t* t) { delete t; }

extern "C" int rcpp_tokenizer_bos_id(const rcpp_tokenizer_t* t) { return t ? t->bos_id : -1; }
extern "C" int rcpp_tokenizer_eos_id(const rcpp_tokenizer_t* t) { return t ? t->eos_id : -1; }

// Convert raw UTF-8 text into a vector of single-char (GPT-2 mapped)
// byte-strings, one entry per input byte. These are the starting
// "pieces" the BPE merge loop works on.
static std::vector<std::string> byte_level_split(const std::string& text) {
    const auto& bm = byte_map();
    std::vector<std::string> pieces;
    pieces.reserve(text.size());
    char buf[4];
    for (uint8_t b : text) {
        uint32_t cp = bm.byte_to_cp[b];
        int n = utf8_encode(cp, buf);
        pieces.emplace_back(buf, buf + n);
    }
    return pieces;
}

extern "C" rcpp_status_t
rcpp_tokenizer_encode(const rcpp_tokenizer_t* t,
                      const char* text, size_t text_len,
                      int add_bos,
                      int* ids_out, size_t max_out, size_t* out_count)
{
    if (!t || !text || !out_count) return RCPP_INVALID_ARG;
    *out_count = 0;

    std::string s(text, text_len);
    auto pieces = byte_level_split(s);

    // Convert each piece to its vocab id. Unknown byte-strings are a
    // programmer error (GPT-2 byte mapping always produces in-vocab chars).
    std::vector<int32_t> ids;
    ids.reserve(pieces.size() + 1);
    if (add_bos) ids.push_back(t->bos_id);
    for (auto& p : pieces) {
        auto it = t->bytes_to_id.find(p);
        if (it == t->bytes_to_id.end()) {
            fprintf(stderr, "tokenizer: unknown byte piece '%s' (len %zu)\n", p.c_str(), p.size());
            return RCPP_INTERNAL;
        }
        ids.push_back(it->second);
    }

    // Greedy BPE: repeatedly merge the adjacent pair with the lowest
    // rank. Simple O(n * merges-applied) algorithm — fine for prompt-
    // sized inputs. A priority-queue / linked-list variant can come
    // later if the bench shows it matters.
    const int bos_offset = add_bos ? 1 : 0;
    while (true) {
        int best_rank = INT32_MAX;
        int best_pos  = -1;
        int32_t best_new = 0;
        for (int i = bos_offset; i < (int)ids.size() - 1; ++i) {
            auto mit = t->merges.find(MergeKey{ids[i], ids[i+1]});
            if (mit != t->merges.end() && mit->second.second < best_rank) {
                best_rank = mit->second.second;
                best_pos  = i;
                best_new  = mit->second.first;
            }
        }
        if (best_pos < 0) break;
        ids[best_pos] = best_new;
        ids.erase(ids.begin() + best_pos + 1);
    }

    *out_count = ids.size();
    size_t n = std::min(ids.size(), max_out);
    if (ids_out) for (size_t i = 0; i < n; ++i) ids_out[i] = ids[i];
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_tokenizer_decode(const rcpp_tokenizer_t* t,
                      const int* ids, size_t n_ids,
                      char* out, size_t max_bytes, size_t* out_len)
{
    if (!t || !ids || !out_len) return RCPP_INVALID_ARG;

    // First reconstruct the GPT-2-mapped UTF-8 string, then undo the
    // byte mapping to produce raw UTF-8 output.
    std::string mapped;
    for (size_t i = 0; i < n_ids; ++i) {
        int id = ids[i];
        if (id < 0 || id >= (int)t->id_to_bytes.size()) continue;
        mapped += t->id_to_bytes[id];
    }

    const auto& bm = byte_map();
    std::string raw;
    raw.reserve(mapped.size());
    const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.data());
    size_t remain = mapped.size();
    while (remain) {
        size_t used = 0;
        uint32_t cp = utf8_decode(p, remain, used);
        if (cp < bm.cp_to_byte.size() && bm.cp_to_byte[cp] >= 0) {
            raw.push_back((char)bm.cp_to_byte[cp]);
        }
        // unknown codepoints silently dropped — special tokens have
        // empty decoded text by design.
        p += used; remain -= used;
    }

    *out_len = raw.size();
    size_t n = std::min(raw.size(), max_bytes);
    if (out) std::memcpy(out, raw.data(), n);
    if (out && max_bytes > n) out[n] = '\0';
    return RCPP_OK;
}
