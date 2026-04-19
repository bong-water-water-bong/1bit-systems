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

use std::path::Path;

use halo_bitnet_hip as hip;
use halo_bitnet_hip::{DeviceBuffer, DevicePtr, DeviceMutPtr, HipStream, RcppError};
use halo_core::h1b::{H1bConfig, H1bFile, H1bLayerOffsets, H1bWeightFormat, Span};
use halo_core::htok::HtokFile;
use half::f16;

use crate::tokenizer::ByteLevelBpe;

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
    /// Anything else.
    #[error("router: {0}")]
    Other(String),
}

/// Per-layer device-resident weights. Every field is a sized device buffer
/// we uploaded once at `new()` and never mutate.
struct LayerWeights {
    input_norm: DeviceBuffer<u16>,        // fp16 [hs]
    post_attn_norm: DeviceBuffer<u16>,    // fp16 [hs]
    attn_sub_norm: DeviceBuffer<u16>,     // fp16 [hs]
    ffn_sub_norm: DeviceBuffer<u16>,      // fp16 [is]

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
    embedding: DeviceBuffer<u16>,   // fp16 [vocab * hs]
    final_norm: DeviceBuffer<u16>,  // fp16 [hs]
    layers: Vec<LayerWeights>,
}

/// Scratch buffers reused per-token. Allocated once at `new()`.
struct Scratch {
    // Residual + attention activations
    x_fp32: DeviceBuffer<f32>,           // [hs]
    x_fp16: DeviceBuffer<u16>,           // [hs] — embedding read target
    normed: DeviceBuffer<u16>,           // [hs]
    x_i8: DeviceBuffer<i8>,              // [hs_k]
    x_scale_dev: DeviceBuffer<f32>,      // [1]

    q_fp16: DeviceBuffer<u16>,           // [nh * hd]
    k_fp16: DeviceBuffer<u16>,           // [nkv * hd]
    v_fp16: DeviceBuffer<u16>,           // [nkv * hd]
    o_fp16: DeviceBuffer<u16>,           // [hs]

    gate_fp16: DeviceBuffer<u16>,        // [is]
    up_fp16: DeviceBuffer<u16>,          // [is]
    down_fp16: DeviceBuffer<u16>,        // [hs]
    silu_out: DeviceBuffer<u16>,         // [is]
    silu_i8: DeviceBuffer<i8>,           // [is_k]
    silu_scale_dev: DeviceBuffer<f32>,   // [1]

    logits: DeviceBuffer<f32>,           // [vocab]
    next_tok_dev: DeviceBuffer<i32>,     // [1]
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
    /// Load `.h1b` + `.htok` from disk and upload every weight to the
    /// current HIP device. Allocates all scratch + KV cache up front.
    ///
    /// `model_id` is the name advertised through `/v1/models`.
    /// `max_context` bounds the KV cache.
    pub fn new(
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

        let weights = ModelWeights { embedding, final_norm, layers };

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

        // Greedy argmax on device (matches gen-1 fast path). Higher
        // temperatures are handled via a host-side sampler in `lib.rs`.
        ok(hip::argmax_fp32(
            self.scratch.logits.as_device_ptr(),
            self.scratch.next_tok_dev.as_device_mut_ptr(),
            vocab,
            stream,
        ))?;
        hip::device_synchronize()?;
        let next = self.scratch.next_tok_dev.copy_to_host_scalar()?;

        // Copy logits back to host for non-greedy sampling paths.
        if logits_out.len() != vocab as usize {
            logits_out.resize(vocab as usize, 0.0);
        }
        self.scratch.logits.copy_to_slice(logits_out.as_mut_slice())?;

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
