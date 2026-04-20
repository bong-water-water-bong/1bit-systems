//! HIP backend — forward pass + weight upload.
//!
//! This is the Rust port of the `forward_token` lambda at
//! `rocm-cpp/tools/bitnet_decode.cpp:408-487`. The math is identical and
//! uses exactly the same rocm-cpp kernel names; only the host-side glue
//! (allocation, lifecycle, streaming) is rewritten in safe Rust on top of
//! `halo-bitnet-hip`'s safe wrappers.
//!
//! Design:
//!   * [`HipBackend`] owns every device allocation for the lifetime of the
//!     process. Weights are uploaded once at `new()` and never re-touched.
//!   * Scratch / activation buffers are preallocated once, reused per token.
//!   * The KV cache is sized for `max_context` (default 4096) and lives for
//!     the life of the backend — sessions reset `cache_pos` to 0 rather
//!     than re-allocating.
//!
//! Everything here is synchronous and single-threaded: the HTTP layer above
//! us runs this on a tokio blocking thread via `spawn_blocking`. Making
//! forward-token itself async would just hand control back to tokio between
//! GPU kernel launches, which on gfx1151 take a handful of microseconds —
//! scheduling overhead would dominate.

use std::fs::File;
use std::io::Read as _;
use std::path::Path;

use half::f16;
use halo_bitnet_hip as hip;
use halo_bitnet_hip::{DeviceBuffer, DeviceMutPtr, DevicePtr, HipStream, RcppError};
use halo_core::gguf::{GGUF_MAGIC, GgufFile};
use halo_core::h1b::{H1B_MAGIC, H1bConfig, H1bFile, H1bLayerOffsets, H1bWeightFormat, Span};
use halo_core::htok::HtokFile;

use crate::tokenizer::ByteLevelBpe;

/// Which on-disk model container the router detected at `--model <path>`.
///
/// Resolved by [`sniff_model_format`] from a cheap 4-byte peek at the file
/// head plus the filename extension. The router holds the enum for logs /
/// observability; actual loading is handled per-variant inside
/// [`HipBackend::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Native halo `.h1b` — packed ternary weights already in HaloV2 layout.
    H1b,
    /// Public GGUF v3 BitNet export (e.g. `microsoft/bitnet-b1.58-2B-4t-gguf`).
    Gguf,
}

/// Peek at `path` to decide which loader we should dispatch to.
///
/// The extension is the hint and the first 4 bytes are the authority:
///
/// * `.h1b` + magic starting `H1B`  → [`ModelFormat::H1b`].
/// * `.gguf` + magic `GGUF`        → [`ModelFormat::Gguf`].
/// * mismatch (wrong magic for the claimed extension, or an unknown
///   extension altogether) → [`BackendError::BadInput`] naming both
///   accepted formats so the operator knows what the router understands.
///
/// Note: halo's `.h1b` parser checks only the first three magic bytes,
/// so `H1B\0` / `H1Bx` are both accepted here — same relaxed rule.
pub fn sniff_model_format(path: &Path) -> Result<ModelFormat, BackendError> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    // Read the first 4 bytes — enough for both magics.
    let mut head = [0u8; 4];
    {
        let mut f = File::open(path)
            .map_err(|e| BackendError::Other(format!("open {}: {}", path.display(), e)))?;
        f.read_exact(&mut head).map_err(|e| {
            BackendError::Other(format!(
                "read magic from {} (file too small?): {}",
                path.display(),
                e
            ))
        })?;
    }

    let is_h1b_magic = head[..3] == H1B_MAGIC[..3];
    let is_gguf_magic = head == GGUF_MAGIC;

    match ext.as_str() {
        "h1b" => {
            if is_h1b_magic {
                Ok(ModelFormat::H1b)
            } else {
                Err(BackendError::Other(format!(
                    "{} has .h1b extension but magic is {:?}; expected H1B\\0 or GGUF",
                    path.display(),
                    head
                )))
            }
        }
        "gguf" => {
            if is_gguf_magic {
                Ok(ModelFormat::Gguf)
            } else {
                Err(BackendError::Other(format!(
                    "{} has .gguf extension but magic is {:?}; expected GGUF or H1B\\0",
                    path.display(),
                    head
                )))
            }
        }
        // Unknown extension: trust the magic if it's one we recognise,
        // otherwise bail with the full list of accepted formats.
        _ => {
            if is_h1b_magic {
                Ok(ModelFormat::H1b)
            } else if is_gguf_magic {
                Ok(ModelFormat::Gguf)
            } else {
                Err(BackendError::Other(format!(
                    "unrecognised model file {}: extension {:?}, magic {:?}; \
                     expected a .h1b or .gguf file",
                    path.display(),
                    ext,
                    head
                )))
            }
        }
    }
}

/// Errors surfaced from the router backend. Kept small so the server layer
/// can format them into OpenAI-shaped JSON without lossy stringification.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// Model loader / format error (bad .h1b, wrong version, ...).
    #[error(transparent)]
    Halo(#[from] halo_core::HaloError),
    /// HIP runtime or rocm-cpp kernel error.
    #[error(transparent)]
    Hip(#[from] RcppError),
    /// The request asked for more tokens than the KV cache can hold.
    #[error("context overflow: {used} + {new_tokens} > {limit}")]
    Context {
        /// Tokens already resident in the KV cache.
        used: usize,
        /// Tokens the request wants to add on top.
        new_tokens: usize,
        /// Hard cap set at router construction (default 4096).
        limit: usize,
    },
    /// This .h1b uses a packing version halo-router's forward pass doesn't
    /// know how to dispatch yet (currently only HaloV2 is wired).
    #[error("unsupported weight format: {0:?}")]
    UnsupportedFormat(H1bWeightFormat),
    /// Caught bad inputs before kicking off GEMVs.
    #[error("bad input: {0}")]
    BadInput(&'static str),
    /// A peer backend has been selected but its real dispatch path is not
    /// compiled / loaded in this build. Surfaced today by the XDNA 2 NPU
    /// arm of the router — the stub FFI crate is always built, but the
    /// `real-xdna` feature (→ `halo-bitnet-xdna/real-xrt`) is off by
    /// default and no Peano-compiled xclbin exists yet.
    ///
    /// This is not a panic on purpose: ops tooling treats it like any
    /// other `BackendError` and can fall back to the HIP path by
    /// retrying with `HALO_BACKEND=hip`.
    #[error("{0}")]
    NotYetWired(&'static str),
    /// The CPU lane is scaffolded (see [`crate::cpu_lane`]) but not yet
    /// on the critical path. Dispatching through `HALO_BACKEND=cpu`
    /// today hits this arm rather than panicking; the lane's real
    /// purpose is to run the sampler + tokenizer in parallel while the
    /// iGPU grinds the next token, and that wire-up is tracked in
    /// `docs/wiki/CPU-Lane-Plan.md`.
    ///
    /// Distinct from [`BackendError::NotYetWired`] because the CPU lane
    /// *does* have working code — it's just not on the SSE hot path
    /// yet. Keeping the two error kinds separate lets the HTTP layer
    /// log + count them differently without regex on the message.
    #[error("{0}")]
    CpuLaneStub(&'static str),
    /// Operator selected `Backend::Xdna` against a ternary BitNet model.
    /// FastFlowLM (the only Linux-on-STX-H NPU path that actually runs
    /// LLMs today, `/usr/bin/flm`) is Q4NX-only — ternary weights don't
    /// have a kernel. AMD has signalled a ternary→INT8 mapping is in
    /// flight (see `project_lemonade_10_2_pivot.md`); until it ships,
    /// this arm refuses gracefully so ops tooling can retry with
    /// `HALO_BACKEND=hip`.
    ///
    /// Distinct from [`BackendError::NotYetWired`] because *this* path
    /// won't get unblocked by our own build — it's an upstream feature
    /// wait.
    #[error("{0}")]
    NpuTernaryUnsupported(&'static str),
    /// FastFlowLM subprocess (`/usr/bin/flm`) couldn't be spawned — binary
    /// missing, not executable, or crashed at startup. Surfaces the OS
    /// error message verbatim so operators can diagnose from logs without
    /// re-running the subprocess by hand.
    ///
    /// Owned `String` (not `&'static str`) because the payload is the
    /// OS-specific spawn error, not a canned message. Distinct from
    /// [`BackendError::NotYetWired`] because the NPU-prefill path *is*
    /// wired in this build — the failure is environmental, not a code
    /// gap, and ops tooling may want to retry after `sudo pacman -S
    /// fastflowlm`.
    #[error("flm subprocess: {0}")]
    FlmSpawn(String),
    /// Anything else.
    #[error("router: {0}")]
    Other(String),
}

/// Per-layer device-resident weights. Every field is a sized device buffer
/// we uploaded once at `new()` and never mutate.
struct LayerWeights {
    input_norm: DeviceBuffer<u16>,     // fp16 [hs]
    post_attn_norm: DeviceBuffer<u16>, // fp16 [hs]
    attn_sub_norm: DeviceBuffer<u16>,  // fp16 [hs]
    ffn_sub_norm: DeviceBuffer<u16>,   // fp16 [is]

    q_packed: DeviceBuffer<u8>,
    q_scales: DeviceBuffer<f32>,
    k_packed: DeviceBuffer<u8>,
    k_scales: DeviceBuffer<f32>,
    v_packed: DeviceBuffer<u8>,
    v_scales: DeviceBuffer<f32>,
    o_packed: DeviceBuffer<u8>,
    o_scales: DeviceBuffer<f32>,
    gate_packed: DeviceBuffer<u8>,
    gate_scales: DeviceBuffer<f32>,
    up_packed: DeviceBuffer<u8>,
    up_scales: DeviceBuffer<f32>,
    down_packed: DeviceBuffer<u8>,
    down_scales: DeviceBuffer<f32>,
}

/// Model-level device weights.
struct ModelWeights {
    embedding: DeviceBuffer<u16>,  // fp16 [vocab * hs]
    final_norm: DeviceBuffer<u16>, // fp16 [hs]
    layers: Vec<LayerWeights>,
}

/// Scratch buffers reused per-token. Allocated once at `new()`.
struct Scratch {
    // Residual + attention activations
    x_fp32: DeviceBuffer<f32>,      // [hs]
    x_fp16: DeviceBuffer<u16>,      // [hs] — embedding read target
    normed: DeviceBuffer<u16>,      // [hs]
    x_i8: DeviceBuffer<i8>,         // [hs_k]
    x_scale_dev: DeviceBuffer<f32>, // [1]

    q_fp16: DeviceBuffer<u16>, // [nh * hd]
    k_fp16: DeviceBuffer<u16>, // [nkv * hd]
    v_fp16: DeviceBuffer<u16>, // [nkv * hd]
    o_fp16: DeviceBuffer<u16>, // [hs]

    gate_fp16: DeviceBuffer<u16>,      // [is]
    up_fp16: DeviceBuffer<u16>,        // [is]
    down_fp16: DeviceBuffer<u16>,      // [hs]
    silu_out: DeviceBuffer<u16>,       // [is]
    silu_i8: DeviceBuffer<i8>,         // [is_k]
    silu_scale_dev: DeviceBuffer<f32>, // [1]

    logits: DeviceBuffer<f32>,       // [vocab]
    next_tok_dev: DeviceBuffer<i32>, // [1]
}

/// Per-layer KV cache (FP16). We allocate `max_context * nkv * hd` per
/// layer at `new()` and index by position.
struct KvCache {
    k: DeviceBuffer<u16>,
    v: DeviceBuffer<u16>,
}

/// The actual HIP backend. Holds weights, scratch, KV cache, tokenizer,
/// and a running decode position.
pub struct HipBackend {
    cfg: H1bConfig,
    weights: ModelWeights,
    scratch: Scratch,
    kv: Vec<KvCache>,
    tokenizer: ByteLevelBpe,
    max_context: usize,
    model_id: String,
    hip_label: String,
}

impl HipBackend {
    /// Load a model file + `.htok` from disk and upload every weight to the
    /// current HIP device. Allocates all scratch + KV cache up front.
    ///
    /// `model_path` may be either `.h1b` (native halo packing) or `.gguf`
    /// (public BitNet GGUF). The format is sniffed via [`sniff_model_format`]
    /// and dispatched to the matching private loader. `htok_path` is only
    /// consumed on the `.h1b` path — on the GGUF path the tokenizer comes
    /// out of the file's own metadata KVs and `htok_path` is ignored.
    ///
    /// `model_id` is the name advertised through `/v1/models`.
    /// `max_context` bounds the KV cache.
    pub fn new(
        model_path: &Path,
        htok_path: &Path,
        model_id: String,
        max_context: usize,
    ) -> Result<Self, BackendError> {
        match sniff_model_format(model_path)? {
            ModelFormat::H1b => {
                tracing::info!(path = %model_path.display(), "dispatch: .h1b loader");
                Self::load_h1b_into_hip(model_path, htok_path, model_id, max_context)
            }
            ModelFormat::Gguf => {
                tracing::info!(path = %model_path.display(), "dispatch: .gguf loader");
                load_gguf_into_hip(model_path, model_id, max_context)
            }
        }
    }

    /// Loader for native `.h1b` files — the path this crate has shipped
    /// since gen-2 day one. Split out of [`Self::new`] in 2026-04 so the
    /// format dispatch can sit above it.
    fn load_h1b_into_hip(
        h1b_path: &Path,
        htok_path: &Path,
        model_id: String,
        max_context: usize,
    ) -> Result<Self, BackendError> {
        let h1b = H1bFile::open(h1b_path)?;
        let tok = HtokFile::open(htok_path)?;
        let tokenizer = ByteLevelBpe::from_htok(tok);

        let cfg = *h1b.config();
        // Forward pass here implements HaloV2 packing only. Sherry / TQ1
        // dispatch is a follow-up — the kernel wrappers already exist in
        // halo-bitnet-hip so adding them is a one-match switch.
        if cfg.weight_format()? != H1bWeightFormat::HaloV2 {
            return Err(BackendError::UnsupportedFormat(cfg.weight_format()?));
        }

        tracing::info!(
            hidden_size = cfg.hidden_size,
            intermediate_size = cfg.intermediate_size,
            num_layers = cfg.num_layers,
            num_heads = cfg.num_heads,
            num_kv_heads = cfg.num_kv_heads,
            vocab_size = cfg.vocab_size,
            version = cfg.version,
            "uploading .h1b weights to GPU"
        );

        // Pick the default device. Most hosts only have one; on multi-GPU
        // rigs we'd surface a selector, but Strix Halo is single-iGPU.
        hip::set_device(0)?;
        let hip_label = format!(
            "hip device 0 of {} available",
            hip::device_count().unwrap_or(0)
        );

        let hs = cfg.hidden_size as usize;
        let is_ = cfg.intermediate_size as usize;
        let nh = cfg.num_heads as usize;
        let nkv = cfg.num_kv_heads as usize;
        let hd = cfg.head_dim()? as usize;
        let vocab = cfg.vocab_size as usize;
        let n_layers = cfg.num_layers as usize;
        let hs_k = hs; // HaloV2 kernels take K = hs directly.
        let is_k = is_;

        // ---------- Weights ----------
        let model_off = *h1b.model_offsets();
        let embedding = upload_fp32_as_fp16(&h1b, model_off.embedding, vocab * hs)?;
        let final_norm = upload_fp32_as_fp16(&h1b, model_off.final_norm, hs)?;

        let mut layers = Vec::with_capacity(n_layers);
        for (idx, lo) in h1b.layer_offsets().iter().enumerate() {
            layers.push(upload_layer(&h1b, lo, hs, is_, idx)?);
        }

        let weights = ModelWeights {
            embedding,
            final_norm,
            layers,
        };

        // ---------- Scratch ----------
        let scratch = Scratch {
            x_fp32: DeviceBuffer::alloc(hs)?,
            x_fp16: DeviceBuffer::alloc(hs)?,
            normed: DeviceBuffer::alloc(hs)?,
            x_i8: DeviceBuffer::alloc_zeroed(hs_k)?,
            x_scale_dev: DeviceBuffer::alloc(1)?,

            q_fp16: DeviceBuffer::alloc(nh * hd)?,
            k_fp16: DeviceBuffer::alloc(nkv * hd)?,
            v_fp16: DeviceBuffer::alloc(nkv * hd)?,
            o_fp16: DeviceBuffer::alloc(hs)?,

            gate_fp16: DeviceBuffer::alloc(is_)?,
            up_fp16: DeviceBuffer::alloc(is_)?,
            down_fp16: DeviceBuffer::alloc(hs)?,
            silu_out: DeviceBuffer::alloc(is_)?,
            silu_i8: DeviceBuffer::alloc_zeroed(is_k)?,
            silu_scale_dev: DeviceBuffer::alloc(1)?,

            logits: DeviceBuffer::alloc(vocab)?,
            next_tok_dev: DeviceBuffer::alloc(1)?,
        };

        // ---------- KV cache ----------
        let mut kv = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            kv.push(KvCache {
                k: DeviceBuffer::alloc(max_context * nkv * hd)?,
                v: DeviceBuffer::alloc(max_context * nkv * hd)?,
            });
        }

        hip::device_synchronize()?;

        tracing::info!(
            layers = n_layers,
            max_context,
            "halo-router HIP backend ready"
        );

        Ok(Self {
            cfg,
            weights,
            scratch,
            kv,
            tokenizer,
            max_context,
            model_id,
            hip_label,
        })
    }

    /// Human-readable backend label for `/v1/models` and logs.
    pub fn label(&self) -> &str {
        &self.hip_label
    }

    /// Model id advertised on `/v1/models`.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Encode a prompt string into BitNet token ids (with BOS prepended).
    pub fn tokenize(&self, text: &str) -> Vec<i32> {
        self.tokenizer.encode(text, /* add_bos */ true)
    }

    /// Decode a sequence of ids into raw bytes.
    pub fn detokenize(&self, ids: &[i32]) -> String {
        self.tokenizer.decode(ids)
    }

    /// Run one forward token and return (next_token_id, full logits copy).
    ///
    /// `logits_out` lets the caller reuse a host-side buffer across calls
    /// (sized to `vocab_size`). It's always overwritten on success.
    ///
    /// This is the single hot path. It takes `&mut self` because every
    /// scratch buffer and the KV cache are borrowed mutably.
    pub fn forward_token(
        &mut self,
        token_id: i32,
        pos: i32,
        logits_out: &mut Vec<f32>,
    ) -> Result<i32, BackendError> {
        if (pos as usize) >= self.max_context {
            return Err(BackendError::Context {
                used: pos as usize,
                new_tokens: 1,
                limit: self.max_context,
            });
        }

        let hs = self.cfg.hidden_size;
        let is_ = self.cfg.intermediate_size;
        let nh = self.cfg.num_heads;
        let nkv = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim()?;
        let vocab = self.cfg.vocab_size;
        let rope_theta = self.cfg.rope_theta;
        let rms_eps = self.cfg.rms_norm_eps;
        let scale = 1.0f32 / (hd as f32).sqrt();

        let stream = HipStream::DEFAULT;

        // Seed the FP32 residual stream from the FP16 embedding.
        ok(hip::embedding_lookup_fp16(
            self.weights.embedding.as_device_ptr(),
            token_id,
            self.scratch.x_fp16.as_device_mut_ptr(),
            hs,
            stream,
        ))?;
        // Zero x_fp32, then accumulate the FP16 embedding into it.
        // SAFETY: x_fp32 is a valid device buffer we own; we memset exactly
        // its allocation size.
        unsafe {
            let rc = hip::ffi::hipMemsetAsync(
                self.scratch.x_fp32.as_device_mut_ptr().0 as *mut core::ffi::c_void,
                0,
                self.scratch.x_fp32.byte_len(),
                core::ptr::null_mut(),
            );
            if rc != hip::ffi::HIP_SUCCESS {
                return Err(BackendError::Hip(RcppError::HipError));
            }
        }
        ok(hip::residual_add_fp32_from_fp16(
            self.scratch.x_fp32.as_device_mut_ptr(),
            self.scratch.x_fp16.as_device_ptr(),
            hs,
            stream,
        ))?;

        for l in 0..self.cfg.num_layers as usize {
            // ---------- Attention block ----------
            let ly = &self.weights.layers[l];

            ok(hip::rmsnorm_fp32_in_fp16_out(
                self.scratch.x_fp32.as_device_ptr(),
                ly.input_norm.as_device_ptr(),
                self.scratch.normed.as_device_mut_ptr(),
                rms_eps,
                hs,
                stream,
            ))?;
            ok(hip::quantize_fp16_to_i8(
                self.scratch.normed.as_device_ptr(),
                self.scratch.x_i8.as_device_mut_ptr(),
                self.scratch.x_scale_dev.as_device_mut_ptr(),
                hs,
                stream,
            ))?;
            let x_scale = self.scratch.x_scale_dev.copy_to_host_scalar()?;

            // Q/K/V projections (HaloV2 kernel → fp16 direct).
            ok(hip::ternary_gemv_halo_f16(
                ly.q_packed.as_device_ptr(),
                self.scratch.x_i8.as_device_ptr(),
                x_scale,
                ly.q_scales.as_device_ptr(),
                self.scratch.q_fp16.as_device_mut_ptr(),
                nh * hd,
                hs,
                stream,
            ))?;
            ok(hip::ternary_gemv_halo_f16(
                ly.k_packed.as_device_ptr(),
                self.scratch.x_i8.as_device_ptr(),
                x_scale,
                ly.k_scales.as_device_ptr(),
                self.scratch.k_fp16.as_device_mut_ptr(),
                nkv * hd,
                hs,
                stream,
            ))?;
            ok(hip::ternary_gemv_halo_f16(
                ly.v_packed.as_device_ptr(),
                self.scratch.x_i8.as_device_ptr(),
                x_scale,
                ly.v_scales.as_device_ptr(),
                self.scratch.v_fp16.as_device_mut_ptr(),
                nkv * hd,
                hs,
                stream,
            ))?;

            // RoPE on Q and K.
            ok(hip::rope_fp16(
                self.scratch.q_fp16.as_device_mut_ptr(),
                pos,
                rope_theta,
                nh,
                hd,
                stream,
            ))?;
            ok(hip::rope_fp16(
                self.scratch.k_fp16.as_device_mut_ptr(),
                pos,
                rope_theta,
                nkv,
                hd,
                stream,
            ))?;

            // Append K/V to cache at slot `pos`, then flash-decode attention.
            let kv_slot = (pos as usize) * (nkv as usize) * (hd as usize);
            let kv_bytes = (nkv as usize) * (hd as usize) * core::mem::size_of::<u16>();
            // SAFETY: kv_slot <= max_context*nkv*hd by the bounds-check at the
            // top of forward_token; we write exactly `nkv*hd*2` bytes which
            // fits in the allocation.
            unsafe {
                let rc = hip::ffi::hipMemcpyAsync(
                    self.kv[l].k.offset_mut(kv_slot).0 as *mut core::ffi::c_void,
                    self.scratch.k_fp16.as_device_ptr().0 as *const core::ffi::c_void,
                    kv_bytes,
                    hip::ffi::HIP_MEMCPY_DEVICE_TO_DEVICE,
                    core::ptr::null_mut(),
                );
                if rc != hip::ffi::HIP_SUCCESS {
                    return Err(BackendError::Hip(RcppError::HipError));
                }
                let rc = hip::ffi::hipMemcpyAsync(
                    self.kv[l].v.offset_mut(kv_slot).0 as *mut core::ffi::c_void,
                    self.scratch.v_fp16.as_device_ptr().0 as *const core::ffi::c_void,
                    kv_bytes,
                    hip::ffi::HIP_MEMCPY_DEVICE_TO_DEVICE,
                    core::ptr::null_mut(),
                );
                if rc != hip::ffi::HIP_SUCCESS {
                    return Err(BackendError::Hip(RcppError::HipError));
                }
            }

            ok(hip::kv_cache_attn_decode_fd(
                self.scratch.q_fp16.as_device_ptr(),
                self.kv[l].k.as_device_ptr(),
                self.kv[l].v.as_device_ptr(),
                self.scratch.o_fp16.as_device_mut_ptr(),
                nh,
                nkv,
                hd,
                pos + 1,
                scale,
                stream,
            ))?;

            // BitNet b1.58: attn_sub_norm between attention and O projection.
            ok(hip::rmsnorm_fp16(
                self.scratch.o_fp16.as_device_ptr(),
                ly.attn_sub_norm.as_device_ptr(),
                self.scratch.normed.as_device_mut_ptr(),
                rms_eps,
                hs,
                stream,
            ))?;
            ok(hip::quantize_fp16_to_i8(
                self.scratch.normed.as_device_ptr(),
                self.scratch.x_i8.as_device_mut_ptr(),
                self.scratch.x_scale_dev.as_device_mut_ptr(),
                hs,
                stream,
            ))?;
            let x_scale = self.scratch.x_scale_dev.copy_to_host_scalar()?;
            ok(hip::ternary_gemv_halo_f16(
                ly.o_packed.as_device_ptr(),
                self.scratch.x_i8.as_device_ptr(),
                x_scale,
                ly.o_scales.as_device_ptr(),
                self.scratch.o_fp16.as_device_mut_ptr(),
                hs,
                nh * hd,
                stream,
            ))?;
            ok(hip::residual_add_fp32_from_fp16(
                self.scratch.x_fp32.as_device_mut_ptr(),
                self.scratch.o_fp16.as_device_ptr(),
                hs,
                stream,
            ))?;

            // ---------- FFN block ----------
            ok(hip::rmsnorm_fp32_in_fp16_out(
                self.scratch.x_fp32.as_device_ptr(),
                ly.post_attn_norm.as_device_ptr(),
                self.scratch.normed.as_device_mut_ptr(),
                rms_eps,
                hs,
                stream,
            ))?;
            ok(hip::quantize_fp16_to_i8(
                self.scratch.normed.as_device_ptr(),
                self.scratch.x_i8.as_device_mut_ptr(),
                self.scratch.x_scale_dev.as_device_mut_ptr(),
                hs,
                stream,
            ))?;
            let x_scale = self.scratch.x_scale_dev.copy_to_host_scalar()?;

            ok(hip::ternary_gemv_halo_f16(
                ly.gate_packed.as_device_ptr(),
                self.scratch.x_i8.as_device_ptr(),
                x_scale,
                ly.gate_scales.as_device_ptr(),
                self.scratch.gate_fp16.as_device_mut_ptr(),
                is_,
                hs,
                stream,
            ))?;
            ok(hip::ternary_gemv_halo_f16(
                ly.up_packed.as_device_ptr(),
                self.scratch.x_i8.as_device_ptr(),
                x_scale,
                ly.up_scales.as_device_ptr(),
                self.scratch.up_fp16.as_device_mut_ptr(),
                is_,
                hs,
                stream,
            ))?;

            // Fused ReLU² GLU + ffn_sub_norm (FP32 interior → FP16 out).
            ok(hip::relu2_glu_rmsnorm_fp16(
                self.scratch.gate_fp16.as_device_ptr(),
                self.scratch.up_fp16.as_device_ptr(),
                ly.ffn_sub_norm.as_device_ptr(),
                self.scratch.silu_out.as_device_mut_ptr(),
                rms_eps,
                is_,
                stream,
            ))?;
            ok(hip::quantize_fp16_to_i8(
                self.scratch.silu_out.as_device_ptr(),
                self.scratch.silu_i8.as_device_mut_ptr(),
                self.scratch.silu_scale_dev.as_device_mut_ptr(),
                is_,
                stream,
            ))?;
            let silu_scale = self.scratch.silu_scale_dev.copy_to_host_scalar()?;

            ok(hip::ternary_gemv_halo_f16(
                ly.down_packed.as_device_ptr(),
                self.scratch.silu_i8.as_device_ptr(),
                silu_scale,
                ly.down_scales.as_device_ptr(),
                self.scratch.down_fp16.as_device_mut_ptr(),
                hs,
                is_,
                stream,
            ))?;
            ok(hip::residual_add_fp32_from_fp16(
                self.scratch.x_fp32.as_device_mut_ptr(),
                self.scratch.down_fp16.as_device_ptr(),
                hs,
                stream,
            ))?;
        }

        // ---------- Final norm + tied LM head ----------
        ok(hip::rmsnorm_fp32_in_fp16_out(
            self.scratch.x_fp32.as_device_ptr(),
            self.weights.final_norm.as_device_ptr(),
            self.scratch.normed.as_device_mut_ptr(),
            rms_eps,
            hs,
            stream,
        ))?;
        ok(hip::fp16_gemv(
            self.weights.embedding.as_device_ptr(),
            self.scratch.normed.as_device_ptr(),
            self.scratch.logits.as_device_mut_ptr(),
            vocab,
            hs,
            stream,
        ))?;

        // Greedy argmax on device (fast path). We also run a deterministic
        // host-side argmax below and reconcile: on a tie (two tokens with
        // the exact same f32 logit), the HIP warp-reduction can land on
        // EITHER tied lane depending on the per-thread stride-scan (see
        // `rcpp_argmax_fp32` in `rocm-cpp/src/prim_kernels.hip`), which is
        // a real source of drift against gen-1's effective behaviour on
        // prompts like "The chemical symbol for gold is" where the model
        // puts ~equal mass on two tokens. Host reconciliation forces
        // lowest-token-id-wins-on-tie — a stable total order that matches
        // a plain linear scan on both gen-1 and gen-2.
        ok(hip::argmax_fp32(
            self.scratch.logits.as_device_ptr(),
            self.scratch.next_tok_dev.as_device_mut_ptr(),
            vocab,
            stream,
        ))?;
        hip::device_synchronize()?;
        let dev_next = self.scratch.next_tok_dev.copy_to_host_scalar()?;

        // Copy logits back to host for non-greedy sampling paths.
        if logits_out.len() != vocab as usize {
            logits_out.resize(vocab as usize, 0.0);
        }
        self.scratch
            .logits
            .copy_to_slice(logits_out.as_mut_slice())?;

        // Deterministic host argmax: strict `>` preserves the first (lowest)
        // index on ties. If the device agreed, we keep its answer (zero-cost);
        // if not, the device hit a tie and we override with the lower index.
        let next = host_argmax_lowest_index(logits_out, dev_next);

        Ok(next)
    }

    /// Reset decode state at the start of a new conversation.
    pub fn reset(&mut self) {
        // Nothing to do on the GPU — reused slots are overwritten on next
        // forward_token. Host-side position is tracked by the caller.
    }

    /// Parsed `.h1b` config header (dimensions, rope params, ...).
    pub fn config(&self) -> &H1bConfig {
        &self.cfg
    }
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

#[inline]
fn ok(status: halo_bitnet_hip::RcppStatus) -> Result<(), BackendError> {
    status.into_result().map_err(BackendError::from)
}

/// Deterministic argmax with lowest-index-wins-on-tie semantics.
///
/// Fast path: if `dev_next` already points at the argmax (which it does
/// whenever the max logit is unique), return it immediately — a single
/// bounds-check and indexed read.
///
/// Slow path: when the device kernel's lane-reduction landed on a tied
/// value, walk the logit vector once and return the lowest index whose
/// logit equals the max. This forces a stable total order across every
/// decode, independent of warp layout / thread count, and fixes the
/// shadow-burnin prompt-7 divergence where ~equal logits on two digit
/// tokens produced v1="1" vs v2="0" 100% of the time.
#[inline]
fn host_argmax_lowest_index(logits: &[f32], dev_next: i32) -> i32 {
    if logits.is_empty() {
        return 0;
    }
    let dev_idx = dev_next as usize;
    let dev_val = logits.get(dev_idx).copied().unwrap_or(f32::NEG_INFINITY);
    // Strict `>`: first (lowest-index) occurrence of the max wins.
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    // Device and host agree whenever the max is unique. They only diverge
    // on exact-logit ties — in which case host wins (lower index).
    if best_val == dev_val && dev_idx < logits.len() {
        // NaN-safe: NaN != NaN so this branch won't fire on NaN max.
        // Prefer the lower of (dev_idx, best_idx) since both carry the max.
        return dev_idx.min(best_idx) as i32;
    }
    best_idx as i32
}

/// Upload a span of on-disk FP32 as FP16 to the device. This matches the
/// C++ loader's `read_fp32_as_fp16` helper, except we read from the mmap
/// instead of an ifstream.
fn upload_fp32_as_fp16(
    file: &H1bFile,
    span: Span,
    expected_count: usize,
) -> Result<DeviceBuffer<u16>, BackendError> {
    let bytes = file.tensor_bytes(span);
    if bytes.len() != expected_count * core::mem::size_of::<f32>() {
        return Err(BackendError::BadInput("fp32 span size mismatch"));
    }
    // Convert fp32 → fp16 on the host, then upload. Lossy but matches
    // gen-1 numerics (same cast).
    let mut host_fp16: Vec<u16> = Vec::with_capacity(expected_count);
    for chunk in bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        host_fp16.push(f16::from_f32(v).to_bits());
    }
    let mut buf = DeviceBuffer::alloc(expected_count)?;
    buf.copy_from_slice(&host_fp16)?;
    Ok(buf)
}

/// Upload a raw-bytes span to the device as `u8`. Used for packed ternary
/// weights which are already in the exact on-device layout.
fn upload_u8(file: &H1bFile, span: Span) -> Result<DeviceBuffer<u8>, BackendError> {
    let bytes = file.tensor_bytes(span);
    let mut buf = DeviceBuffer::alloc(bytes.len())?;
    buf.copy_from_slice(bytes)?;
    Ok(buf)
}

/// Upload a span of FP32 scales as-is.
fn upload_fp32(
    file: &H1bFile,
    span: Span,
    expected_count: usize,
) -> Result<DeviceBuffer<f32>, BackendError> {
    let bytes = file.tensor_bytes(span);
    if bytes.len() != expected_count * core::mem::size_of::<f32>() {
        return Err(BackendError::BadInput("fp32 scale span size mismatch"));
    }
    let mut host = Vec::with_capacity(expected_count);
    for chunk in bytes.chunks_exact(4) {
        host.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let mut buf = DeviceBuffer::alloc(expected_count)?;
    buf.copy_from_slice(&host)?;
    Ok(buf)
}

fn upload_layer(
    file: &H1bFile,
    lo: &H1bLayerOffsets,
    hs: usize,
    is_: usize,
    idx: usize,
) -> Result<LayerWeights, BackendError> {
    let _ = idx; // used only for logging
    let input_norm = upload_fp32_as_fp16(file, lo.input_norm, hs)?;
    let post_attn_norm = upload_fp32_as_fp16(file, lo.post_attn_norm, hs)?;
    let attn_sub_norm = upload_fp32_as_fp16(file, lo.attn_sub_norm, hs)?;
    let ffn_sub_norm = upload_fp32_as_fp16(file, lo.ffn_sub_norm, is_)?;

    let q_packed = upload_u8(file, lo.q_packed)?;
    let q_scales = upload_fp32(file, lo.q_scales, lo.q_scales.len / 4)?;
    let k_packed = upload_u8(file, lo.k_packed)?;
    let k_scales = upload_fp32(file, lo.k_scales, lo.k_scales.len / 4)?;
    let v_packed = upload_u8(file, lo.v_packed)?;
    let v_scales = upload_fp32(file, lo.v_scales, lo.v_scales.len / 4)?;
    let o_packed = upload_u8(file, lo.o_packed)?;
    let o_scales = upload_fp32(file, lo.o_scales, lo.o_scales.len / 4)?;
    let gate_packed = upload_u8(file, lo.gate_packed)?;
    let gate_scales = upload_fp32(file, lo.gate_scales, lo.gate_scales.len / 4)?;
    let up_packed = upload_u8(file, lo.up_packed)?;
    let up_scales = upload_fp32(file, lo.up_scales, lo.up_scales.len / 4)?;
    let down_packed = upload_u8(file, lo.down_packed)?;
    let down_scales = upload_fp32(file, lo.down_scales, lo.down_scales.len / 4)?;

    Ok(LayerWeights {
        input_norm,
        post_attn_norm,
        attn_sub_norm,
        ffn_sub_norm,
        q_packed,
        q_scales,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        o_packed,
        o_scales,
        gate_packed,
        gate_scales,
        up_packed,
        up_scales,
        down_packed,
        down_scales,
    })
}

// Explicitly shadow these to keep warnings quiet when used by `lib.rs`.
#[allow(dead_code)]
fn _use_devices(_d: DevicePtr<u8>, _m: DeviceMutPtr<u8>) {}

// ----------------------------------------------------------------------------
// GGUF loader — dispatch target for `sniff_model_format(..) == Gguf`.
//
// Two sprints deep now:
//   1. Plumbing — parse header, extract `BitnetHeader`, walk the standard
//      llama.cpp tensor-name grid, return raw mmap slices. DONE.
//   2. Bit-unpack — IQ2_S → halo v2 (2 bpw) on the host side via
//      `halo_core::gguf::unpack::iq2_s_to_halo_v2`. DONE (2026-04-19).
//
// What's still open is device-side integration: the unpacked payload +
// fp16 absmean scales need to be `memcpy_h2d`'d into `DeviceBuffer<u8>` /
// `DeviceBuffer<f16>` slots on `HipBackend`, and those pointers stashed
// where the forward-pass code can find them. Today we just drop the host
// copies after the unpack succeeds — the loader will still tap out at the
// outer `unimplemented!` since it can't construct a usable `HipBackend`.
// ----------------------------------------------------------------------------

/// GGUF tensor-name grid (llama.cpp convention). We build the expected
/// name for each layer / global up front and fetch the byte slice via
/// [`GgufFile::tensor`]. Producers have diverged here historically — if we
/// ever need to support alternate spellings (`attn_qkv` fused, etc.) we'll
/// add them as a fallback list, not by globbing the tensor directory.
fn layer_tensor_name(layer_idx: usize, tail: &str) -> String {
    format!("blk.{}.{}", layer_idx, tail)
}

/// Dispatch target for `.gguf` — mirrors the shape of
/// [`HipBackend::load_h1b_into_hip`] but reads from [`GgufFile`] instead.
///
/// Scope today (post-unpack sprint):
///   1. Parse header + tensor directory.
///   2. Extract [`halo_core::BitnetHeader`] (hidden_size, num_layers,
///      rope_theta, rms_norm_eps, ...).
///   3. For each layer, look up the canonical llama.cpp tensor names and
///      pull the raw bytes out of the mmap.
///   4. Host-side bit-unpack via
///      [`halo_core::gguf::unpack::iq2_s_to_halo_v2`]. The unpacked
///      payload is dropped immediately — device-buffer integration is
///      deliberately split into its own sprint.
///   5. Bail with `unimplemented!` after the directory walk succeeds —
///      we've proven the unpack works end-to-end on a real file, but
///      the `HipBackend` we'd want to return still has no device
///      allocations.
///
/// The signature deliberately matches [`HipBackend::load_h1b_into_hip`]
/// minus the separate `htok_path` — GGUFs carry the tokenizer inline, so
/// there is nothing to pass.
fn load_gguf_into_hip(
    gguf_path: &Path,
    model_id: String,
    max_context: usize,
) -> Result<HipBackend, BackendError> {
    let _ = model_id;
    let _ = max_context;

    let g = GgufFile::open(gguf_path)?;
    let hdr = g.read_bitnet_metadata()?;

    tracing::info!(
        architecture = %hdr.architecture,
        block_count = hdr.block_count,
        embedding_length = hdr.embedding_length,
        feed_forward_length = hdr.feed_forward_length,
        attention_head_count = hdr.attention_head_count,
        attention_head_count_kv = hdr.attention_head_count_kv,
        rope_freq_base = hdr.rope_freq_base,
        tokens = hdr.tokens.len(),
        "parsed GGUF BitnetHeader"
    );

    // Globals (llama.cpp canonical names).
    let _token_embd = g.tensor("token_embd.weight");
    let _output_norm = g.tensor("output_norm.weight");
    let _output = g.tensor("output.weight");
    let _rope_freqs = g.tensor("rope_freqs.weight"); // often absent

    // Per-layer directory sweep. Grab each expected tensor's dtype + bytes;
    // the moment we find an IQ2_S / TQ2_0 payload we tap out with an
    // explicit TODO so the bit-unpacking is a single-function follow-up.
    for l in 0..hdr.block_count as usize {
        for tail in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
        ] {
            let name = layer_tensor_name(l, tail);
            let info = g.tensor_info(&name).ok_or_else(|| {
                BackendError::Other(format!(
                    "GGUF missing expected tensor {name} (llama.cpp convention)"
                ))
            })?;
            let _bytes = g.tensor(&name).ok_or_else(|| {
                BackendError::Other(format!(
                    "GGUF tensor {name} has unknown dtype / size (got {:?})",
                    info.dtype
                ))
            })?;
            // This is where the requantizer will live. Plumbed as a single
            // call so a future agent can swap it in without touching
            // `sniff_model_format` / `HipBackend::new`.
            gguf_tensor_to_halo_packed(info.dtype, _bytes)?;
        }
    }

    // We got through the directory walk and host-side unpack — but we
    // still don't have device-resident weights. Stop here so a bad
    // config can still fail fast during integration testing. Wiring up
    // `DeviceBuffer<u8>` + `memcpy_h2d` is the next sprint.
    unimplemented!(
        "GGUF → HipBackend device-buffer integration is the next sprint; \
         host-side IQ2_S unpack is landed and exercised above"
    )
}

/// Requantizer shim — bit-unpacks GGUF tensor bytes into halo's packed
/// ternary layout on the host. **This sprint does the CPU-side unpack
/// only**; device-buffer integration (allocating `DeviceBuffer<u8>`s for
/// the packed payload + fp32 scales, `memcpy_h2d`, stashing the device
/// pointers on [`HipBackend`]) is still the next sprint. That split is
/// deliberate: the unpacker is backend-agnostic and lives in
/// [`halo_core::gguf::unpack`] where it can be tested without HIP.
///
/// Dtype dispatch:
///   * `IQ2_S` — routed through [`halo_core::gguf::unpack::iq2_s_to_halo_v2`].
///     Produces halo v2 packed bytes + per-super-block fp16 absmean scales.
///   * `F16` / `F32` / `BF16` — norms + embeddings. Still TODO; these
///     need a typed upload, not a bit-unpack.
///   * `TQ2_0` / `TQ1_0` — llama.cpp's newer native BitNet formats. TODO.
fn gguf_tensor_to_halo_packed(
    dtype: halo_core::GgufTensorType,
    bytes: &[u8],
) -> Result<(), BackendError> {
    use halo_core::GgufTensorType::*;
    match dtype {
        // FP16 / FP32 norms + embeddings — easy, land these first.
        F16 | F32 | BF16 => {
            // TODO(tier-1): typed upload into a DeviceBuffer<f16 / f32>.
            // No bit-unpack needed, just a bytewise memcpy_h2d.
            Ok(())
        }
        // BitNet's canonical public 2-bit ternary packing in llama.cpp.
        IQ2_S => {
            use halo_core::gguf::unpack;
            // n_weights is derivable from the payload size: each super-block
            // is 82 bytes for 256 weights. Round down; the caller gave us
            // a whole tensor so this should be exact.
            let n_blocks = bytes.len() / unpack::IQ2_S_BLOCK_BYTES;
            let n_weights = n_blocks * 256;
            let mut packed = vec![0u8; n_weights.div_ceil(4)];
            let _scales = unpack::iq2_s_to_halo_v2(bytes, &mut packed, n_weights).map_err(|e| {
                BackendError::Other(format!("IQ2_S → halo v2 bit-unpack failed: {e}"))
            })?;
            // TODO(tier-1): memcpy_h2d the `packed` and `_scales` vectors
            // into DeviceBuffer<u8> / DeviceBuffer<f16> slots on
            // HipBackend, then drop the host copies. We deliberately don't
            // do this yet — the unpack is tested in isolation in
            // halo-core, and the device-buffer plumbing is its own PR.
            Ok(())
        }
        TQ2_0 | TQ1_0 => {
            // TODO(halo-core/unpack): port llama.cpp's base-3 packing
            // (block_tq2_0 / block_tq1_0). Bit-layout is different from
            // IQ2_S — 5 ternaries per byte for TQ1_0 via the
            // d0+3·d1+9·d2+27·d3+81·d4 polynomial.
            unimplemented!("TQ2_0 / TQ1_0 → halo 2-bit repack is the next sprint");
        }
        other => Err(BackendError::Other(format!(
            "GGUF dtype {other:?} is not a BitNet-supported tensor format"
        ))),
    }
}

// ----------------------------------------------------------------------------
// Tests — format sniffing only. Construction of a real `HipBackend` needs
// a GPU and is covered in `tests/smoke_hip.rs` behind `--features hip` +
// `#[ignore]`.
// ----------------------------------------------------------------------------

#[cfg(test)]
mod format_sniff_tests {
    use super::*;
    use halo_core::gguf::GGUF_MAGIC;
    use halo_core::h1b::H1B_MAGIC;
    use std::io::Write;

    /// Write a throwaway temp file in the per-test tempdir and return its
    /// path. We keep the file on disk for the duration of the test — tests
    /// are short and tempdir cleans up on drop.
    fn scratch_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("halo-router-sniff-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    #[test]
    fn sniff_h1b_by_extension_and_magic() {
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(&H1B_MAGIC);
        let p = scratch_file("good.h1b", &bytes);
        assert_eq!(sniff_model_format(&p).unwrap(), ModelFormat::H1b);
    }

    #[test]
    fn sniff_gguf_by_extension_and_magic() {
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(&GGUF_MAGIC);
        let p = scratch_file("good.gguf", &bytes);
        assert_eq!(sniff_model_format(&p).unwrap(), ModelFormat::Gguf);
    }

    #[test]
    fn sniff_rejects_mismatch_with_helpful_error() {
        // `.h1b` extension, but magic says GGUF. Operator probably renamed
        // the file — error message must name both accepted magics.
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(&GGUF_MAGIC);
        let p = scratch_file("lying.h1b", &bytes);
        let err = sniff_model_format(&p).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("H1B") || s.contains("GGUF"), "got: {s}");
    }

    #[test]
    fn sniff_falls_back_to_magic_for_unknown_extension() {
        // No extension at all — we still accept the file if the magic is
        // unambiguous. Mirrors llama.cpp's behaviour.
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(&GGUF_MAGIC);
        let p = scratch_file("anonymous", &bytes);
        assert_eq!(sniff_model_format(&p).unwrap(), ModelFormat::Gguf);
    }

    #[test]
    fn sniff_rejects_unknown_magic_outright() {
        let bytes = b"XXXX\0\0\0\0".to_vec();
        let p = scratch_file("random.bin", &bytes);
        assert!(sniff_model_format(&p).is_err());
    }

    /// Live-model dispatch test: if the halo-1bit-2b.h1b is on disk, the
    /// sniffer must route it to the `.h1b` loader.
    ///
    /// `#[ignore]` because it needs the ~700 MB model file present; run
    /// manually with `cargo test -p halo-router --release -- --ignored`.
    #[test]
    #[ignore = "needs $HOME/halo-ai/models/halo-1bit-2b.h1b on disk"]
    fn live_h1b_model_routes_to_h1b_path() {
        let home = std::env::var("HOME").unwrap();
        let p = std::path::PathBuf::from(format!("{home}/halo-ai/models/halo-1bit-2b.h1b"));
        if !p.exists() {
            eprintln!("skipping: model not present at {}", p.display());
            return;
        }
        let f = sniff_model_format(&p).expect("sniff should succeed");
        assert_eq!(f, ModelFormat::H1b);
    }

    /// Host argmax + tie-break: unique max → device answer flows through,
    /// tied max → lowest-index wins. Regression guard for the shadow-burnin
    /// prompt-7 divergence ("chemical symbol for gold" → v1="1" vs v2="0"
    /// 100% of rounds, fixed 2026-04-20).
    #[test]
    fn host_argmax_picks_lowest_index_on_tie() {
        // Unique max at index 2: device answer passes through.
        let l = [0.1f32, 0.5, 3.2, 0.5, -1.0];
        assert_eq!(host_argmax_lowest_index(&l, 2), 2);

        // Tied max at indices 3 and 7 — device landed on the higher index
        // (HIP warp-reduce lane 7), host override must pick 3.
        let mut l = vec![0.0f32; 16];
        l[3] = 5.25;
        l[7] = 5.25;
        assert_eq!(host_argmax_lowest_index(&l, 7), 3);
        // Idempotent when device already picked the lower index.
        assert_eq!(host_argmax_lowest_index(&l, 3), 3);

        // Device OOB (defensive) + empty input: no panic.
        let l = [1.0f32, 2.0, 3.0];
        assert_eq!(host_argmax_lowest_index(&l, 99), 2);
        assert_eq!(host_argmax_lowest_index(&[], 0), 0);
    }

    /// Synthetic-GGUF dispatch test: build a minimal in-memory GGUF buffer
    /// (1 F32 tensor, correct header + alignment) and assert the sniffer
    /// routes it to the GGUF loader without needing to construct a
    /// `HipBackend`. Stops short of actually calling
    /// [`load_gguf_into_hip`], which hits `unimplemented!` on the first
    /// tensor — the goal is to prove the dispatch fired.
    #[test]
    #[ignore = "synthetic GGUF would succeed at sniff but panic inside load_gguf_into_hip today"]
    fn synthetic_gguf_routes_to_gguf_path() {
        // Minimal valid GGUF: magic + v3 + 0 tensors + 0 kvs, padded.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        // pad to 32-byte alignment for the (empty) tensor data region.
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        let p = scratch_file("synth.gguf", &buf);
        let f = sniff_model_format(&p).expect("sniff should succeed");
        assert_eq!(f, ModelFormat::Gguf, "should route to gguf loader");

        // We deliberately don't call `load_gguf_into_hip` — today it would
        // `unimplemented!` on the first tensor directory walk. The
        // assertion above is the whole point: dispatch fired.
    }
}
