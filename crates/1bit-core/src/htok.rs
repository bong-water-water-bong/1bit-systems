//! `.htok` — halo tokenizer binary format.
//!
//! Rust port of `/home/bcloud/repos/rocm-cpp/src/tokenizer.cpp`. The on-disk
//! layout (all integers little-endian) is:
//!
//! ```text
//!   offset  size         field
//!   0x00    4            magic       = b"HTOK"
//!   0x04    4            vocab_size  (u32)
//!   0x08    4            num_merges  (u32)
//!   0x0C    4            bos_id      (u32)
//!   0x10    4            eos_id      (u32)
//!   0x14    ...          vocab       : vocab_size × { u16 len, u8[len] bytes }
//!   ....    ...          merges      : num_merges × { u32 a, u32 b, u32 merged }
//! ```
//!
//! The vocab byte strings are already byte-level-BPE (GPT-2 mapped) — that
//! is, each entry is a sequence of UTF-8 bytes in the 0x21..0x7E / 0x100..
//! range. We do not re-expand them; encode/decode callers live in a future
//! `halo-tokenizer` crate. `1bit-core`'s job is purely the format.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian};

use crate::error::HaloError;

pub const HTOK_MAGIC: [u8; 4] = *b"HTOK";

/// A single BPE merge, rank is the insertion order (lower = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Merge {
    pub a: i32,
    pub b: i32,
    pub merged: i32,
    pub rank: u32,
}

/// Fully-parsed tokenizer file. All fields are owned so the file handle can
/// drop immediately after parsing — the file is ~2 MB for LLaMA-3-class
/// vocab, which is dwarfed by model weights.
#[derive(Debug, Clone)]
pub struct HtokFile {
    pub bos_id: i32,
    pub eos_id: i32,
    /// `id_to_bytes[i]` is the raw (GPT-2-mapped) byte string for token `i`.
    pub id_to_bytes: Vec<Vec<u8>>,
    pub merges: Vec<Merge>,
    path: PathBuf,
}

impl HtokFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, HaloError> {
        let p = path.as_ref();
        let mut f = File::open(p).map_err(|e| HaloError::io_at(p, e))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)
            .map_err(|e| HaloError::io_at(p, e))?;
        Self::parse(p.to_path_buf(), &buf)
    }

    pub fn parse_bytes(path: impl Into<PathBuf>, bytes: &[u8]) -> Result<Self, HaloError> {
        Self::parse(path.into(), bytes)
    }

    fn parse(path: PathBuf, buf: &[u8]) -> Result<Self, HaloError> {
        const HEADER: usize = 4 + 4 + 4 + 4 + 4;
        if buf.len() < HEADER {
            return Err(HaloError::Truncated {
                offset: 0,
                needed: HEADER,
                have: buf.len(),
            });
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);
        if magic != HTOK_MAGIC {
            return Err(HaloError::BadMagic {
                expected: HTOK_MAGIC,
                got: magic,
            });
        }

        let vocab_size = LittleEndian::read_u32(&buf[4..8]) as usize;
        let num_merges = LittleEndian::read_u32(&buf[8..12]) as usize;
        let bos_id = LittleEndian::read_u32(&buf[12..16]) as i32;
        let eos_id = LittleEndian::read_u32(&buf[16..20]) as i32;

        let mut cursor = HEADER;
        let mut id_to_bytes = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            if cursor + 2 > buf.len() {
                return Err(HaloError::Truncated {
                    offset: cursor,
                    needed: 2,
                    have: buf.len().saturating_sub(cursor),
                });
            }
            let len = LittleEndian::read_u16(&buf[cursor..cursor + 2]) as usize;
            cursor += 2;
            if cursor + len > buf.len() {
                return Err(HaloError::Truncated {
                    offset: cursor,
                    needed: len,
                    have: buf.len().saturating_sub(cursor),
                });
            }
            id_to_bytes.push(buf[cursor..cursor + len].to_vec());
            cursor += len;
            let _ = i;
        }

        let mut merges = Vec::with_capacity(num_merges);
        for rank in 0..num_merges {
            if cursor + 12 > buf.len() {
                return Err(HaloError::Truncated {
                    offset: cursor,
                    needed: 12,
                    have: buf.len().saturating_sub(cursor),
                });
            }
            let a = LittleEndian::read_u32(&buf[cursor..cursor + 4]) as i32;
            let b = LittleEndian::read_u32(&buf[cursor + 4..cursor + 8]) as i32;
            let merged = LittleEndian::read_u32(&buf[cursor + 8..cursor + 12]) as i32;
            cursor += 12;
            merges.push(Merge {
                a,
                b,
                merged,
                rank: rank as u32,
            });
        }

        Ok(Self {
            bos_id,
            eos_id,
            id_to_bytes,
            merges,
            path,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    /// Serialize back to on-disk form. Round-trip identity with [`parse_bytes`].
    pub fn to_bytes(&self) -> Result<Vec<u8>, HaloError> {
        let mut out = Vec::with_capacity(20 + self.id_to_bytes.len() * 8 + self.merges.len() * 12);
        out.extend_from_slice(&HTOK_MAGIC);
        out.extend_from_slice(&(self.id_to_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.merges.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.bos_id as u32).to_le_bytes());
        out.extend_from_slice(&(self.eos_id as u32).to_le_bytes());

        for bytes in &self.id_to_bytes {
            if bytes.len() > u16::MAX as usize {
                return Err(HaloError::InvalidConfig(
                    "tokenizer vocab entry exceeds u16::MAX bytes",
                ));
            }
            out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            out.extend_from_slice(bytes);
        }
        for m in &self.merges {
            out.extend_from_slice(&(m.a as u32).to_le_bytes());
            out.extend_from_slice(&(m.b as u32).to_le_bytes());
            out.extend_from_slice(&(m.merged as u32).to_le_bytes());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini() -> HtokFile {
        HtokFile {
            bos_id: 128_000,
            eos_id: 128_001,
            id_to_bytes: vec![
                b"<|bos|>".to_vec(),
                b"<|eos|>".to_vec(),
                b"a".to_vec(),
                b"b".to_vec(),
                b"ab".to_vec(),
            ],
            merges: vec![Merge {
                a: 2,
                b: 3,
                merged: 4,
                rank: 0,
            }],
            path: PathBuf::from("<test>"),
        }
    }

    #[test]
    fn round_trip() {
        let t = mini();
        let bytes = t.to_bytes().unwrap();
        let parsed = HtokFile::parse_bytes("t", &bytes).unwrap();
        let bytes2 = parsed.to_bytes().unwrap();
        assert_eq!(bytes, bytes2);
        assert_eq!(parsed.bos_id, 128_000);
        assert_eq!(parsed.eos_id, 128_001);
        assert_eq!(parsed.vocab_size(), 5);
        assert_eq!(parsed.merges.len(), 1);
        assert_eq!(parsed.merges[0], t.merges[0]);
        assert_eq!(parsed.id_to_bytes, t.id_to_bytes);
    }

    #[test]
    fn bad_magic() {
        let mut bytes = vec![0u8; 20];
        bytes[0..4].copy_from_slice(b"XXXX");
        let err = HtokFile::parse_bytes("t", &bytes).unwrap_err();
        matches!(err, HaloError::BadMagic { .. });
    }

    #[test]
    fn truncated_vocab_detected() {
        // Header claims vocab_size=1, but no entry bytes follow.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&HTOK_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes()); // vocab_size = 1
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let err = HtokFile::parse_bytes("t", &bytes).unwrap_err();
        matches!(err, HaloError::Truncated { .. });
    }
}
