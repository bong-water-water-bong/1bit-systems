//! Byte-level BPE encoder / decoder.
//!
//! Rust port of the minimal encoder in
//! `rocm-cpp/src/tokenizer.cpp`. Behaviour matches byte-for-byte on ASCII
//! text; the pre-tokenizer is only the "split digit runs into 3-digit
//! chunks" rule (the rest of LLaMA-3's regex is not implemented — same
//! as the C++ reference).

use std::collections::HashMap;

use halo_core::htok::HtokFile;

/// GPT-2 byte <-> Unicode mapping. Printable ASCII maps to itself; the
/// rest are shunted into U+0100..U+017F so every byte is a printable
/// codepoint. Same table tiktoken / LLaMA-3 use.
struct ByteMap {
    byte_to_cp: [u32; 256],
    cp_to_byte: HashMap<u32, u8>,
}

impl ByteMap {
    fn new() -> Self {
        let mut bs: Vec<u8> = Vec::new();
        for b in b'!'..=b'~' {
            bs.push(b);
        }
        for b in 0xA1u8..=0xACu8 {
            bs.push(b);
        }
        for b in 0xAEu8..=0xFFu8 {
            bs.push(b);
        }
        let mut cps: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
        let mut n: u32 = 0;
        for b in 0u16..=255u16 {
            let b = b as u8;
            if !bs.contains(&b) {
                bs.push(b);
                cps.push(0x100 + n);
                n += 1;
            }
        }
        let mut byte_to_cp = [0u32; 256];
        let mut cp_to_byte = HashMap::new();
        for (&b, &cp) in bs.iter().zip(cps.iter()) {
            byte_to_cp[b as usize] = cp;
            cp_to_byte.insert(cp, b);
        }
        Self {
            byte_to_cp,
            cp_to_byte,
        }
    }
}

/// UTF-8-encode a codepoint. Returns the byte slice length (1..=4).
fn utf8_encode(cp: u32, out: &mut [u8; 4]) -> usize {
    if cp < 0x80 {
        out[0] = cp as u8;
        1
    } else if cp < 0x800 {
        out[0] = 0xC0 | ((cp >> 6) as u8);
        out[1] = 0x80 | ((cp & 0x3F) as u8);
        2
    } else if cp < 0x10000 {
        out[0] = 0xE0 | ((cp >> 12) as u8);
        out[1] = 0x80 | (((cp >> 6) & 0x3F) as u8);
        out[2] = 0x80 | ((cp & 0x3F) as u8);
        3
    } else {
        out[0] = 0xF0 | ((cp >> 18) as u8);
        out[1] = 0x80 | (((cp >> 12) & 0x3F) as u8);
        out[2] = 0x80 | (((cp >> 6) & 0x3F) as u8);
        out[3] = 0x80 | ((cp & 0x3F) as u8);
        4
    }
}

/// UTF-8-decode one codepoint. Returns (codepoint, bytes_read); replacement
/// character `0xFFFD` on malformed input.
fn utf8_decode(s: &[u8]) -> (u32, usize) {
    if s.is_empty() {
        return (0, 0);
    }
    let b0 = s[0];
    if b0 < 0x80 {
        return (b0 as u32, 1);
    }
    if (b0 & 0xE0) == 0xC0 && s.len() >= 2 {
        return (
            ((b0 as u32 & 0x1F) << 6) | (s[1] as u32 & 0x3F),
            2,
        );
    }
    if (b0 & 0xF0) == 0xE0 && s.len() >= 3 {
        return (
            ((b0 as u32 & 0x0F) << 12) | ((s[1] as u32 & 0x3F) << 6) | (s[2] as u32 & 0x3F),
            3,
        );
    }
    if (b0 & 0xF8) == 0xF0 && s.len() >= 4 {
        return (
            ((b0 as u32 & 0x07) << 18)
                | ((s[1] as u32 & 0x3F) << 12)
                | ((s[2] as u32 & 0x3F) << 6)
                | (s[3] as u32 & 0x3F),
            4,
        );
    }
    (0xFFFD, 1)
}

/// Byte-level BPE tokenizer — encodes / decodes text using the vocabulary
/// and merges baked into the `.htok` file.
pub struct ByteLevelBpe {
    id_to_bytes: Vec<Vec<u8>>,
    bytes_to_id: HashMap<Vec<u8>, i32>,
    /// (a, b) → (merged_id, rank). Lower rank = higher priority.
    merges: HashMap<(i32, i32), (i32, u32)>,
    bos_id: i32,
    #[allow(dead_code)]
    eos_id: i32,
    byte_map: ByteMap,
}

impl ByteLevelBpe {
    /// Construct from a parsed `.htok` file.
    pub fn from_htok(tok: HtokFile) -> Self {
        let mut bytes_to_id = HashMap::with_capacity(tok.id_to_bytes.len());
        for (i, b) in tok.id_to_bytes.iter().enumerate() {
            if !b.is_empty() {
                bytes_to_id.insert(b.clone(), i as i32);
            }
        }
        let mut merges = HashMap::with_capacity(tok.merges.len());
        for m in &tok.merges {
            merges.insert((m.a, m.b), (m.merged, m.rank));
        }
        Self {
            id_to_bytes: tok.id_to_bytes,
            bytes_to_id,
            merges,
            bos_id: tok.bos_id,
            eos_id: tok.eos_id,
            byte_map: ByteMap::new(),
        }
    }

    /// Encode text into BitNet token ids. If `add_bos`, prepends the BOS
    /// token configured in the `.htok` header.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<i32> {
        let chunks = pre_tokenize(text);
        let mut all_ids: Vec<i32> = Vec::new();
        if add_bos {
            all_ids.push(self.bos_id);
        }
        for chunk in chunks {
            let pieces = self.byte_level_split(&chunk);
            let ids = self.bpe_encode(&pieces);
            all_ids.extend_from_slice(&ids);
        }
        all_ids
    }

    /// Decode token ids back to UTF-8 bytes (lossy on unknown codepoints,
    /// matching the C++ reference).
    pub fn decode(&self, ids: &[i32]) -> String {
        let mut mapped: Vec<u8> = Vec::new();
        for &id in ids {
            if id < 0 {
                continue;
            }
            if let Some(bytes) = self.id_to_bytes.get(id as usize) {
                mapped.extend_from_slice(bytes);
            }
        }
        let mut out: Vec<u8> = Vec::with_capacity(mapped.len());
        let mut i = 0;
        while i < mapped.len() {
            let (cp, used) = utf8_decode(&mapped[i..]);
            if let Some(&b) = self.byte_map.cp_to_byte.get(&cp) {
                out.push(b);
            }
            // Unknown codepoints silently dropped — matches C++ behaviour
            // for special tokens with empty decoded text.
            i += used.max(1);
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    fn byte_level_split(&self, text: &str) -> Vec<Vec<u8>> {
        let mut buf = [0u8; 4];
        let mut out: Vec<Vec<u8>> = Vec::with_capacity(text.len());
        for &b in text.as_bytes() {
            let cp = self.byte_map.byte_to_cp[b as usize];
            let n = utf8_encode(cp, &mut buf);
            out.push(buf[..n].to_vec());
        }
        out
    }

    fn bpe_encode(&self, pieces: &[Vec<u8>]) -> Vec<i32> {
        let mut ids: Vec<i32> = Vec::with_capacity(pieces.len());
        for p in pieces {
            match self.bytes_to_id.get(p) {
                Some(&i) => ids.push(i),
                None => {
                    tracing::warn!(
                        "tokenizer: unknown byte piece (len {}), skipping",
                        p.len()
                    );
                    return ids;
                }
            }
        }
        loop {
            let mut best_rank: u32 = u32::MAX;
            let mut best_pos: isize = -1;
            let mut best_new: i32 = 0;
            for i in 0..ids.len().saturating_sub(1) {
                if let Some(&(merged, rank)) = self.merges.get(&(ids[i], ids[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = i as isize;
                        best_new = merged;
                    }
                }
            }
            if best_pos < 0 {
                break;
            }
            let p = best_pos as usize;
            ids[p] = best_new;
            ids.remove(p + 1);
        }
        ids
    }
}

/// Minimal pre-tokenizer: split digit runs into 3-digit chunks, keep
/// everything else as-is. Mirrors the C++ reference.
fn pre_tokenize(text: &str) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        let c = bytes[i];
        let is_digit = |b: u8| (b'0'..=b'9').contains(&b);
        if is_digit(c) {
            let mut j = i;
            while j < n && is_digit(bytes[j]) {
                j += 1;
            }
            let mut pos = i;
            while pos < j {
                let take = core::cmp::min(3, j - pos);
                chunks.push(String::from_utf8_lossy(&bytes[pos..pos + take]).into_owned());
                pos += take;
            }
            i = j;
        } else {
            let mut j = i;
            while j < n && !is_digit(bytes[j]) {
                j += 1;
            }
            chunks.push(String::from_utf8_lossy(&bytes[i..j]).into_owned());
            i = j;
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_map_round_trip() {
        let bm = ByteMap::new();
        for b in 0u16..=255u16 {
            let cp = bm.byte_to_cp[b as usize];
            assert!(cp != 0);
            assert_eq!(bm.cp_to_byte.get(&cp), Some(&(b as u8)));
        }
    }

    #[test]
    fn pre_tokenize_digits() {
        assert_eq!(pre_tokenize("abc"), vec!["abc".to_string()]);
        assert_eq!(pre_tokenize("1234"), vec!["123".to_string(), "4".to_string()]);
        assert_eq!(
            pre_tokenize("foo42bar"),
            vec!["foo".to_string(), "42".to_string(), "bar".to_string()]
        );
    }

    #[test]
    fn utf8_round_trip() {
        let mut buf = [0u8; 4];
        for cp in [0x41u32, 0xE9, 0x4E2D, 0x1F600] {
            let n = utf8_encode(cp, &mut buf);
            let (dec, used) = utf8_decode(&buf[..n]);
            assert_eq!(dec, cp);
            assert_eq!(used, n);
        }
    }
}
