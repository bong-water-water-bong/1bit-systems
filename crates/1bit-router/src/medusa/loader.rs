//! `.h1b-medusa` on-disk format + zero-copy mmap loader.
//!
//! This module is the **parse-only** half of the Medusa weight-loading
//! pass. It opens the file, validates magic + shape, and exposes four
//! zero-copy per-head tensor views over the mmap. The GPU upload path
//! (device buffers, `HipBackend::upload`, and the small-M ternary GEMM
//! dispatch) lives in a follow-up pass behind the same env gate — this
//! module only speaks bytes.
//!
//! # File layout (from `docs/wiki/Medusa-Integration-Plan.md` and the
//! `parrishcorcoran/MedusaBitNet-2B-4T` artifact)
//!
//! All integers are little-endian. Magic is an 8-byte ASCII tag carrying
//! an explicit nul so it round-trips as a `u64` read without losing the
//! zero byte — mirrors the main `.h1b` convention.
//!
//! ```text
//! offset  size   field
//! ------  ----   -----
//!   0      8     MAGIC            "MEDUSA1\0"
//!   8      4     version          u32
//!  12      4     num_heads        u32 (must == NUM_MEDUSA_HEADS == 4)
//!  16      4     hidden_dim       u32 (must == MEDUSA_HIDDEN_DIM == 2560)
//!  20      4     vocab_size       u32 (128256 for the upstream artifact)
//!  24      4     residual_layers  u32 (2 per head)
//!  28     40     _reserved        [u32; 10]
//!  68    ...     per-head payload (see below)
//! ```
//!
//! Header is exactly 68 bytes. Per-head payload, repeated `num_heads`
//! times in head-index order:
//!
//! ```text
//!   weight_0   fp16[hidden_dim * hidden_dim]   residual block 0
//!   bias_0     fp16[hidden_dim]
//!   weight_1   fp16[hidden_dim * hidden_dim]   residual block 1
//!   bias_1     fp16[hidden_dim]
//!   proj       fp16[hidden_dim * vocab_size]
//! ```
//!
//! Per-head byte size at the Microsoft backbone's shape (hd=2560,
//! vocab=128256):
//!
//!   2 × (2560 × 2560) × 2 + 2 × 2560 × 2 + (2560 × 128256) × 2
//!   = 26_214_400 + 10_240 + 656_670_720
//!   = 682_895_360 bytes ≈ 651 MiB
//!
//! Four heads ≈ 2.54 GiB on disk. Mmap is the only sane read strategy.
//! The 13 MB figure in `heads.rs` / the upstream artifact card refers to
//! the *residual-block-only* bytes, where `proj` is borrowed from the
//! backbone's `lm_head` rather than owned by each head. Our on-disk
//! format owns the proj per head so the loader is self-contained; a
//! later shared-lm_head variant can be added by flipping a version bit.
//!
//! # Zero-copy
//!
//! [`MedusaHeadsFile::open`] maps the file read-only and hands back
//! borrowed `&[u16]` views into the mmap. fp16 is carried as `u16` at
//! the parser boundary; the forward-pass crate (`1bit-hip`) takes the
//! `&[u16]` bytes and `memcpy`s them into a device buffer — no
//! allocation in this module, no host-side decode.
//!
//! # What this module does NOT do
//!
//! * No device upload. [`MedusaHeadsFile`] is a pure CPU-side handle.
//! * No weight download. A follow-up lane fetches from Hugging Face
//!   and converts to `.h1b-medusa`. See `project_medusa_plan.md`.
//! * No activation of the Medusa path. The env gate + wiring in
//!   [`super::MedusaState::from_config`] owns that decision.

use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use super::MedusaError;
use super::heads::{MEDUSA_HIDDEN_DIM, NUM_MEDUSA_HEADS};

/// Magic bytes at offset 0. ASCII `MEDUSA1` with an explicit nul so the
/// tag reads back byte-identically regardless of whether it was written
/// as `u64`-LE or `[u8; 8]`.
pub const MEDUSA_MAGIC: [u8; 8] = *b"MEDUSA1\0";

/// Current on-disk format version. Bump when the per-head layout changes
/// (e.g. a shared-lm_head variant, or a packed/rotated weight encoding).
pub const MEDUSA_FORMAT_VERSION: u32 = 1;

/// Exact size of the fixed-width header prefix. 8 (magic) + 4 × 5 (version,
/// num_heads, hidden_dim, vocab_size, residual_layers) + 4 × 10 (reserved).
pub const MEDUSA_HEADER_BYTES: usize = 68;

/// Per-head residual-block count. Pinned to the upstream artifact today;
/// kept as a named const so the loader has a single source of truth.
pub const MEDUSA_RESIDUAL_LAYERS: u32 = 2;

/// Parsed, validated, little-endian header. Cheap to clone (all u32s).
#[derive(Debug, Clone, Copy)]
pub struct MedusaHeader {
    /// File-format version. Validated against [`MEDUSA_FORMAT_VERSION`]
    /// at open time — older/newer versions get a structured error, not
    /// a best-effort parse.
    pub version: u32,
    /// Number of heads declared by the file. Must equal
    /// [`NUM_MEDUSA_HEADS`]; anything else is an ABI break.
    pub num_heads: u32,
    /// Per-head hidden dim. Must equal [`MEDUSA_HIDDEN_DIM`] today —
    /// matches the backbone. Carried in the header anyway so future
    /// versions can relax this without breaking the format.
    pub hidden_dim: u32,
    /// Vocab width of the per-head projection. Not pinned in the
    /// scaffold — the 2B-4T backbone ships 128256, but a retrain may
    /// want a different vocab. The loader trusts the header here and
    /// uses the value to size the proj tensor.
    pub vocab_size: u32,
    /// Number of residual blocks per head. Currently 2.
    pub residual_layers: u32,
}

impl MedusaHeader {
    /// Byte size of one head's payload at this header's shape.
    ///
    /// Layout: `residual_layers × (W + b) + proj`, all fp16 (2 bytes each).
    ///   = residual_layers × (hidden_dim × hidden_dim + hidden_dim) × 2
    ///   + hidden_dim × vocab_size × 2
    ///
    /// Returns `None` on integer overflow — practically unreachable at
    /// the scaffold shape, but cheap protection against a corrupt
    /// header claiming absurd dims.
    pub fn per_head_bytes(&self) -> Option<usize> {
        let hd = self.hidden_dim as usize;
        let vocab = self.vocab_size as usize;
        let layers = self.residual_layers as usize;

        let residual_pair = hd.checked_mul(hd)?.checked_add(hd)?;
        let residual_all = residual_pair.checked_mul(layers)?;
        let proj = hd.checked_mul(vocab)?;
        let total_fp16 = residual_all.checked_add(proj)?;
        total_fp16.checked_mul(2) // fp16 → bytes
    }

    /// Total on-disk size this header describes: header + all heads.
    pub fn expected_file_bytes(&self) -> Option<usize> {
        let per = self.per_head_bytes()?;
        per.checked_mul(self.num_heads as usize)?
            .checked_add(MEDUSA_HEADER_BYTES)
    }
}

/// Per-head borrowed tensor views over the mmap.
///
/// All views are fp16 carried as `u16` — no host-side decode. The
/// forward-pass crate (`1bit-hip`) takes the raw bytes and uploads them
/// to device memory; nothing in the router decodes fp16 on the host.
#[derive(Debug, Clone, Copy)]
pub struct MedusaHeadView<'a> {
    /// Residual-block weight tensors. One per `residual_layer`.
    /// Each is `hidden_dim × hidden_dim` row-major fp16.
    pub residual_weights: [&'a [u16]; MEDUSA_RESIDUAL_LAYERS as usize],
    /// Residual-block biases. One per `residual_layer`. Each is
    /// `hidden_dim` fp16.
    pub residual_biases: [&'a [u16]; MEDUSA_RESIDUAL_LAYERS as usize],
    /// Vocab-projection tensor. `hidden_dim × vocab_size` row-major fp16.
    pub proj: &'a [u16],
}

/// Owning mmap handle + parsed header + cached head offsets.
///
/// The mmap lives for the lifetime of the struct; [`Self::head`] returns
/// views borrowed from it. Move this into the [`super::MedusaHeads`]
/// scaffold struct to give it real tensor data.
///
/// NOT `Clone` — the mmap is ref-counted inside `memmap2::Mmap`, but we
/// want exactly one owner of the parsed-offsets cache per open file.
/// Wrap in `Arc` if multiple owners are needed later.
#[derive(Debug)]
pub struct MedusaHeadsFile {
    /// The read-only mmap. Kept here so views can borrow from it.
    mmap: Mmap,
    /// Parsed + validated file header.
    header: MedusaHeader,
    /// Path the file was opened from. Kept for error messages and
    /// `/metrics` reporting (same contract as
    /// [`super::heads::MedusaHeads::source_path`]).
    path: PathBuf,
    /// Per-head base offset in the mmap (byte index of the head's first
    /// weight tensor). Cached so [`Self::head`] is O(1) without
    /// recomputing the `residual_layers`-dependent stride.
    head_offsets: Vec<usize>,
}

impl MedusaHeadsFile {
    /// Open `path`, validate against [`NUM_MEDUSA_HEADS`] /
    /// [`MEDUSA_HIDDEN_DIM`] and the header's own self-consistency, and
    /// return the parsed handle.
    ///
    /// Errors:
    ///
    /// * [`MedusaError::WeightsNotFound`] — file missing.
    /// * [`MedusaError::LoaderError`] — anything else: short file, bad
    ///   magic, version mismatch, shape mismatch, file-size mismatch,
    ///   alignment issue on the fp16 cast. The message carries enough
    ///   context for an operator to act without attaching a debugger.
    pub fn open(path: &Path) -> Result<Self, MedusaError> {
        let file = File::open(path).map_err(|e| {
            // `NotFound` → structured error; anything else carries the
            // OS error string verbatim through `LoaderError` so the
            // operator can grep dmesg.
            if e.kind() == std::io::ErrorKind::NotFound {
                MedusaError::WeightsNotFound {
                    path: path.to_path_buf(),
                }
            } else {
                MedusaError::LoaderError(format!("open {}: {e}", path.display()))
            }
        })?;

        // SAFETY: read-only map of an immutable model file. Same contract
        // as `1bit-core::h1b::Mapped::from_file` — the kernel could change
        // bytes under us, but model files are treated as immutable by
        // deployment policy. If ops swaps the file live, the resulting
        // torn reads are on them.
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                MedusaError::LoaderError(format!("mmap {}: {e}", path.display()))
            })?
        };

        let bytes = &mmap[..];
        if bytes.len() < MEDUSA_HEADER_BYTES {
            return Err(MedusaError::LoaderError(format!(
                "short file: {} bytes, need at least {} for header",
                bytes.len(),
                MEDUSA_HEADER_BYTES
            )));
        }

        // Magic check first — gives the best error message for the most
        // common operator mistake (pointing at the wrong file).
        if bytes[..8] != MEDUSA_MAGIC {
            return Err(MedusaError::LoaderError(format!(
                "bad magic: expected {:?}, got {:?} (is this really a .h1b-medusa?)",
                MEDUSA_MAGIC,
                &bytes[..8],
            )));
        }

        let header = MedusaHeader {
            version: u32_le(&bytes[8..12]),
            num_heads: u32_le(&bytes[12..16]),
            hidden_dim: u32_le(&bytes[16..20]),
            vocab_size: u32_le(&bytes[20..24]),
            residual_layers: u32_le(&bytes[24..28]),
            // bytes[28..68] is 10 × u32 reserved — read-through for
            // round-trip but not surfaced today.
        };

        if header.version != MEDUSA_FORMAT_VERSION {
            return Err(MedusaError::LoaderError(format!(
                "version mismatch: file says {}, loader expects {}",
                header.version, MEDUSA_FORMAT_VERSION
            )));
        }

        if header.num_heads as usize != NUM_MEDUSA_HEADS {
            return Err(MedusaError::LoaderError(format!(
                "shape mismatch: num_heads={}, expected {} (ABI break — did the \
                 retrained artifact change head count?)",
                header.num_heads, NUM_MEDUSA_HEADS
            )));
        }

        if header.hidden_dim as usize != MEDUSA_HIDDEN_DIM {
            return Err(MedusaError::LoaderError(format!(
                "shape mismatch: hidden_dim={}, expected {} (must match backbone)",
                header.hidden_dim, MEDUSA_HIDDEN_DIM
            )));
        }

        if header.residual_layers != MEDUSA_RESIDUAL_LAYERS {
            return Err(MedusaError::LoaderError(format!(
                "shape mismatch: residual_layers={}, expected {} — the scaffold's \
                 per-head view is fixed-size, bump it if the format grows",
                header.residual_layers, MEDUSA_RESIDUAL_LAYERS
            )));
        }

        if header.vocab_size == 0 {
            return Err(MedusaError::LoaderError(
                "shape mismatch: vocab_size == 0 is not a legal proj shape".to_string(),
            ));
        }

        let expected_total = header.expected_file_bytes().ok_or_else(|| {
            MedusaError::LoaderError(format!(
                "shape overflow: header dims hd={} vocab={} layers={} \
                 overflow usize when computing per-head bytes",
                header.hidden_dim, header.vocab_size, header.residual_layers
            ))
        })?;

        if bytes.len() != expected_total {
            return Err(MedusaError::LoaderError(format!(
                "file-size mismatch: got {} bytes, header implies {} \
                 ({} × {} heads + {} header)",
                bytes.len(),
                expected_total,
                header
                    .per_head_bytes()
                    .expect("already checked above via expected_file_bytes"),
                header.num_heads,
                MEDUSA_HEADER_BYTES,
            )));
        }

        // Per-head base offset cache. Filled after validation so the
        // offsets are guaranteed to fall inside the mapped range.
        let per_head = header
            .per_head_bytes()
            .expect("already checked above via expected_file_bytes");
        let head_offsets: Vec<usize> = (0..header.num_heads as usize)
            .map(|i| MEDUSA_HEADER_BYTES + i * per_head)
            .collect();

        Ok(Self {
            mmap,
            header,
            path: path.to_path_buf(),
            head_offsets,
        })
    }

    /// Parsed header — dims, version, counts.
    pub fn header(&self) -> &MedusaHeader {
        &self.header
    }

    /// Path the file was opened from. Useful for error messages and
    /// `/metrics`.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Byte length of the backing mmap. Lets callers sanity-check the
    /// mapped region without walking through the `Mmap` handle.
    pub fn mmap_len(&self) -> usize {
        self.mmap.len()
    }

    /// Borrowed view of head `i` (0..NUM_MEDUSA_HEADS).
    ///
    /// Returns `MedusaError::BadInput` if `i` is out of range — callers
    /// must iterate over `0..NUM_MEDUSA_HEADS`, not trust a raw usize.
    pub fn head(&self, i: usize) -> Result<MedusaHeadView<'_>, MedusaError> {
        if i >= NUM_MEDUSA_HEADS {
            return Err(MedusaError::BadInput(
                "medusa loader: head index out of range (must be 0..NUM_MEDUSA_HEADS)",
            ));
        }

        let hd = self.header.hidden_dim as usize;
        let vocab = self.header.vocab_size as usize;
        let bytes = &self.mmap[..];

        let base = self.head_offsets[i];

        // Advance `cursor` through the per-head payload, carving out
        // each fp16 tensor as a `&[u16]` view. Offsets inside the head
        // are deterministic by layout; the mmap length was verified at
        // open time so none of the slices can overrun.
        let mut cursor = base;

        // Residual blocks: weight + bias, in order.
        let mut residual_weights: [&[u16]; MEDUSA_RESIDUAL_LAYERS as usize] =
            [&[]; MEDUSA_RESIDUAL_LAYERS as usize];
        let mut residual_biases: [&[u16]; MEDUSA_RESIDUAL_LAYERS as usize] =
            [&[]; MEDUSA_RESIDUAL_LAYERS as usize];

        for layer in 0..MEDUSA_RESIDUAL_LAYERS as usize {
            let w_elems = hd * hd;
            let w_bytes = w_elems * 2;
            residual_weights[layer] =
                fp16_slice(&bytes[cursor..cursor + w_bytes], w_elems)?;
            cursor += w_bytes;

            let b_elems = hd;
            let b_bytes = b_elems * 2;
            residual_biases[layer] = fp16_slice(&bytes[cursor..cursor + b_bytes], b_elems)?;
            cursor += b_bytes;
        }

        // Vocab projection — `hidden_dim × vocab_size` row-major fp16.
        let proj_elems = hd * vocab;
        let proj_bytes = proj_elems * 2;
        let proj = fp16_slice(&bytes[cursor..cursor + proj_bytes], proj_elems)?;

        Ok(MedusaHeadView {
            residual_weights,
            residual_biases,
            proj,
        })
    }
}

/// Read a little-endian `u32` from a 4-byte slice. Panics only if the
/// caller passes a non-4-byte slice — all call sites above are bounded
/// to exactly 4 bytes, so this is infallible in practice.
fn u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes(b.try_into().expect("u32_le requires a 4-byte slice"))
}

/// Cast a `&[u8]` span of `nbytes = elems * 2` into a `&[u16]` view
/// over `elems` fp16 values. Uses `bytemuck::try_cast_slice` for the
/// alignment + length check — returns a `LoaderError` if either fails.
///
/// In practice alignment is not an issue: mmap regions are page-aligned
/// (4 KiB) and the header is 68 bytes so every per-head tensor starts
/// at an even offset inside the map. The check stays to catch a future
/// layout change that accidentally introduces a 1-byte field.
fn fp16_slice(bytes: &[u8], elems: usize) -> Result<&[u16], MedusaError> {
    if bytes.len() != elems * 2 {
        return Err(MedusaError::LoaderError(format!(
            "internal: fp16 slice length mismatch, got {} bytes for {} elems",
            bytes.len(),
            elems
        )));
    }
    bytemuck::try_cast_slice::<u8, u16>(bytes).map_err(|e| {
        MedusaError::LoaderError(format!(
            "fp16 alignment/length: {e} (len={}, elems={})",
            bytes.len(),
            elems
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a valid-shape header into `buf` with the given field values
    /// — lets individual tests mutate one field to exercise a failure
    /// mode without reproducing the header layout in every test.
    fn write_header(
        buf: &mut Vec<u8>,
        magic: &[u8; 8],
        version: u32,
        num_heads: u32,
        hidden_dim: u32,
        vocab_size: u32,
        residual_layers: u32,
    ) {
        buf.extend_from_slice(magic);
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&num_heads.to_le_bytes());
        buf.extend_from_slice(&hidden_dim.to_le_bytes());
        buf.extend_from_slice(&vocab_size.to_le_bytes());
        buf.extend_from_slice(&residual_layers.to_le_bytes());
        // reserved: 10 × u32 of zero.
        buf.extend_from_slice(&[0u8; 40]);
        debug_assert_eq!(buf.len(), MEDUSA_HEADER_BYTES);
    }

    /// Create a sparse file of the exact expected length for the given
    /// header, write the header bytes into the prefix, and return the
    /// temp-file handle + path. Sparse because the tensor payload is
    /// 2.5 GiB of fp16; `set_len` on Linux creates a sparse extent that
    /// reads back as zeros without consuming blocks on disk. On tmpfs
    /// (`/tmp` on CachyOS) a sparse file reads zeros without allocating
    /// pages either, so the test peaks at a few KiB of real memory.
    fn make_medusa_file(
        num_heads: u32,
        hidden_dim: u32,
        vocab_size: u32,
    ) -> (tempfile::NamedTempFile, PathBuf) {
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let mut header_bytes = Vec::with_capacity(MEDUSA_HEADER_BYTES);
        write_header(
            &mut header_bytes,
            &MEDUSA_MAGIC,
            MEDUSA_FORMAT_VERSION,
            num_heads,
            hidden_dim,
            vocab_size,
            MEDUSA_RESIDUAL_LAYERS,
        );

        // Compute the exact expected total (same math as
        // `MedusaHeader::expected_file_bytes`) and extend the file to
        // that length as a sparse allocation.
        let hd = hidden_dim as usize;
        let vocab = vocab_size as usize;
        let layers = MEDUSA_RESIDUAL_LAYERS as usize;
        let per_head = (layers * (hd * hd + hd) + hd * vocab) * 2;
        let total = MEDUSA_HEADER_BYTES + per_head * num_heads as usize;

        tmp.write_all(&header_bytes).expect("write header");
        tmp.as_file_mut()
            .set_len(total as u64)
            .expect("sparse extend to total");
        let path = tmp.path().to_path_buf();
        (tmp, path)
    }

    /// Reject a file whose magic bytes don't match `MEDUSA1\0`.
    ///
    /// The loader must return `LoaderError` with a message that mentions
    /// both the expected and observed magic so an operator can tell at a
    /// glance that they pointed at the wrong file.
    #[test]
    fn rejects_bad_magic() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let mut header = Vec::new();
        write_header(
            &mut header,
            b"NOTMEDSA", // 8 bytes, wrong tag
            MEDUSA_FORMAT_VERSION,
            NUM_MEDUSA_HEADS as u32,
            MEDUSA_HIDDEN_DIM as u32,
            128256,
            MEDUSA_RESIDUAL_LAYERS,
        );
        tmp.write_all(&header).unwrap();
        // Extend to a plausible size so the error reason is definitely
        // "bad magic", not "short file".
        tmp.as_file_mut()
            .set_len(MEDUSA_HEADER_BYTES as u64 + 4096)
            .unwrap();

        let err = MedusaHeadsFile::open(tmp.path()).expect_err("bad magic must be rejected");
        match err {
            MedusaError::LoaderError(msg) => {
                assert!(msg.contains("magic"), "message must mention 'magic': {msg}");
            }
            other => panic!("expected LoaderError, got {other:?}"),
        }
    }

    /// Reject a file whose declared shape doesn't match the scaffold's
    /// pinned constants.
    ///
    /// Two variants exercised in one test:
    ///   * `num_heads` != NUM_MEDUSA_HEADS (ABI break)
    ///   * `hidden_dim` != MEDUSA_HIDDEN_DIM (backbone mismatch)
    /// Both must surface as `LoaderError` so the operator sees which
    /// invariant failed.
    #[test]
    fn rejects_shape_mismatch() {
        // Variant A: num_heads=5 (scaffold expects 4). Use a tiny
        // hidden_dim/vocab so the file stays minimal — the shape check
        // fires on `num_heads` before file-size math runs.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let mut header = Vec::new();
        write_header(
            &mut header,
            &MEDUSA_MAGIC,
            MEDUSA_FORMAT_VERSION,
            5, // bad: not NUM_MEDUSA_HEADS
            MEDUSA_HIDDEN_DIM as u32,
            128256,
            MEDUSA_RESIDUAL_LAYERS,
        );
        tmp.write_all(&header).unwrap();
        tmp.as_file_mut()
            .set_len(MEDUSA_HEADER_BYTES as u64 + 64)
            .unwrap();
        let err = MedusaHeadsFile::open(tmp.path()).expect_err("bad num_heads must be rejected");
        match err {
            MedusaError::LoaderError(msg) => {
                assert!(
                    msg.contains("num_heads"),
                    "message must mention num_heads: {msg}"
                );
            }
            other => panic!("expected LoaderError, got {other:?}"),
        }

        // Variant B: hidden_dim mismatch. num_heads is right so we
        // reach the second shape check.
        let mut tmp2 = tempfile::NamedTempFile::new().unwrap();
        let mut header2 = Vec::new();
        write_header(
            &mut header2,
            &MEDUSA_MAGIC,
            MEDUSA_FORMAT_VERSION,
            NUM_MEDUSA_HEADS as u32,
            1024, // bad: not MEDUSA_HIDDEN_DIM (2560)
            128256,
            MEDUSA_RESIDUAL_LAYERS,
        );
        tmp2.write_all(&header2).unwrap();
        tmp2.as_file_mut()
            .set_len(MEDUSA_HEADER_BYTES as u64 + 64)
            .unwrap();
        let err = MedusaHeadsFile::open(tmp2.path())
            .expect_err("bad hidden_dim must be rejected");
        match err {
            MedusaError::LoaderError(msg) => {
                assert!(
                    msg.contains("hidden_dim"),
                    "message must mention hidden_dim: {msg}"
                );
            }
            other => panic!("expected LoaderError, got {other:?}"),
        }
    }

    /// Synthesize a well-formed `.h1b-medusa` file at the canonical
    /// shape, reopen it, and walk every per-head view to confirm the
    /// lengths match spec. This is the successful-path smoke test — it
    /// does not validate tensor contents (payload is sparse zeros).
    #[test]
    fn parses_synthetic_file() {
        // Canonical shape: NUM_MEDUSA_HEADS × MEDUSA_HIDDEN_DIM × 128256.
        // The file is ~2.54 GiB on disk but sparse, so actual block
        // allocation on tmpfs/ext4 is a few KiB.
        let num_heads = NUM_MEDUSA_HEADS as u32;
        let hidden_dim = MEDUSA_HIDDEN_DIM as u32;
        let vocab_size = 128_256u32;
        let (_tmp, path) = make_medusa_file(num_heads, hidden_dim, vocab_size);

        let file = MedusaHeadsFile::open(&path).expect("successful open");

        // Header round-trip.
        let hdr = file.header();
        assert_eq!(hdr.version, MEDUSA_FORMAT_VERSION);
        assert_eq!(hdr.num_heads, num_heads);
        assert_eq!(hdr.hidden_dim, hidden_dim);
        assert_eq!(hdr.vocab_size, vocab_size);
        assert_eq!(hdr.residual_layers, MEDUSA_RESIDUAL_LAYERS);

        // File-size math matches header.
        assert_eq!(
            file.mmap_len(),
            hdr.expected_file_bytes().expect("no overflow at canonical shape"),
        );

        // Every head view is the right length.
        let hd = hidden_dim as usize;
        let vocab = vocab_size as usize;
        let expected_w_len = hd * hd;
        let expected_b_len = hd;
        let expected_proj_len = hd * vocab;

        for i in 0..NUM_MEDUSA_HEADS {
            let view = file.head(i).expect("head within bounds");
            for (layer, w) in view.residual_weights.iter().enumerate() {
                assert_eq!(
                    w.len(),
                    expected_w_len,
                    "head {i} residual weight layer {layer} length"
                );
            }
            for (layer, b) in view.residual_biases.iter().enumerate() {
                assert_eq!(
                    b.len(),
                    expected_b_len,
                    "head {i} residual bias layer {layer} length"
                );
            }
            assert_eq!(
                view.proj.len(),
                expected_proj_len,
                "head {i} proj length"
            );
        }

        // Out-of-range head index must be a structured error, not a panic.
        let err = file
            .head(NUM_MEDUSA_HEADS)
            .expect_err("out-of-range head must error");
        assert!(matches!(err, MedusaError::BadInput(_)));
    }
}
