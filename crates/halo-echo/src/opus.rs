//! Opus encoding for halo-echo.
//!
//! Kokoro emits 24 kHz mono PCM s16le wrapped in a RIFF WAV container.
//! Browsers want Opus to avoid the 3-4x bandwidth cost of raw PCM, and
//! libopus likes 48 kHz, so this module: parses the 44-byte RIFF header,
//! linearly upsamples 24→48 kHz, splits into 20 ms (960-sample) frames,
//! and encodes each through `audiopus` at 24 kbps VBR. One
//! [`encode_wav_to_opus`] call returns one packet per frame.

use anyhow::{Context, Result, anyhow};
use audiopus::{Application, Channels, SampleRate, coder::Encoder as OpusEncoder};

pub const TARGET_SR: u32 = 48_000;
pub const FRAME_MS: u32 = 20;
pub const BITRATE_BPS: i32 = 24_000;

/// Parsed header fields from a standard 44-byte PCM WAV file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub data_offset: usize,
    pub data_len: usize,
}

impl WavInfo {
    /// Parse the 44-byte RIFF layout kokoro emits. Anything exotic
    /// (extensible fmt, non-PCM, aux chunks before `data`) is rejected.
    pub fn parse(wav: &[u8]) -> Result<Self> {
        if wav.len() < 44 {
            return Err(anyhow!("wav too short: {} bytes", wav.len()));
        }
        if &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" || &wav[12..16] != b"fmt " {
            return Err(anyhow!("not a RIFF/WAVE/fmt file"));
        }
        let fmt = u16::from_le_bytes([wav[20], wav[21]]);
        if fmt != 1 {
            return Err(anyhow!("unsupported WAV format {fmt} (want PCM=1)"));
        }
        if &wav[36..40] != b"data" {
            return Err(anyhow!("expected `data` chunk at offset 36"));
        }
        Ok(Self {
            channels: u16::from_le_bytes([wav[22], wav[23]]),
            sample_rate: u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            bits_per_sample: u16::from_le_bytes([wav[34], wav[35]]),
            data_offset: 44,
            data_len: u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]) as usize,
        })
    }

    pub fn pcm<'a>(&self, wav: &'a [u8]) -> &'a [u8] {
        let end = (self.data_offset + self.data_len).min(wav.len());
        &wav[self.data_offset..end]
    }
}

/// Encode one RIFF WAV blob into a sequence of 20 ms Opus packets.
/// Only mono s16le is supported — kokoro emits nothing else.
pub fn encode_wav_to_opus(wav: &[u8], target_sr: u32) -> Result<Vec<Vec<u8>>> {
    let info = WavInfo::parse(wav).context("parse WAV header")?;
    if info.channels != 1 {
        return Err(anyhow!("expected mono, got {} channels", info.channels));
    }
    if info.bits_per_sample != 16 {
        return Err(anyhow!("expected 16-bit PCM, got {}", info.bits_per_sample));
    }

    let samples: Vec<i16> = info
        .pcm(wav)
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    // Linear interpolation is fine for a 2x integer ratio on speech; the
    // vocoder has already rolled off everything above 12 kHz.
    let resampled = if info.sample_rate == target_sr {
        samples
    } else {
        linear_resample(&samples, info.sample_rate, target_sr)
    };

    let sr = <SampleRate as audiopus::TryFrom<i32>>::try_from(target_sr as i32)
        .map_err(|e| anyhow!("unsupported opus sample rate {target_sr}: {e}"))?;
    let frame_samples = (target_sr as usize * FRAME_MS as usize) / 1000;

    let mut encoder = OpusEncoder::new(sr, Channels::Mono, Application::Voip)
        .map_err(|e| anyhow!("opus encoder new: {e}"))?;
    // Best-effort tuning; libopus' defaults are fine if CTLs are rejected.
    let _ = encoder.set_bitrate(audiopus::Bitrate::BitsPerSecond(BITRATE_BPS));
    let _ = encoder.set_vbr(true);

    let mut packets = Vec::with_capacity(resampled.len() / frame_samples + 1);
    let mut out = vec![0u8; 4000]; // libopus' documented max packet size.
    let mut i = 0;
    while i < resampled.len() {
        let end = (i + frame_samples).min(resampled.len());
        let frame: Vec<i16> = if end - i == frame_samples {
            resampled[i..end].to_vec()
        } else {
            let mut f = vec![0i16; frame_samples]; // zero-pad the tail
            f[..end - i].copy_from_slice(&resampled[i..end]);
            f
        };
        let n = encoder
            .encode(&frame, &mut out)
            .map_err(|e| anyhow!("opus encode: {e}"))?;
        if n > 0 {
            packets.push(out[..n].to_vec());
        }
        i += frame_samples;
    }
    Ok(packets)
}

fn linear_resample(input: &[i16], src_sr: u32, dst_sr: u32) -> Vec<i16> {
    if input.is_empty() || src_sr == dst_sr {
        return input.to_vec();
    }
    let ratio = dst_sr as f64 / src_sr as f64;
    let out_len = ((input.len() as f64) * ratio) as usize;
    let last = input.len() - 1;
    (0..out_len)
        .map(|j| {
            let src_pos = (j as f64) / ratio;
            let idx = src_pos.floor() as usize;
            let frac = src_pos - (idx as f64);
            let a = input[idx.min(last)] as f64;
            let b = input[(idx + 1).min(last)] as f64;
            (a + (b - a) * frac).clamp(i16::MIN as f64, i16::MAX as f64) as i16
        })
        .collect()
}

/// Build a minimal RIFF WAV for test fixtures.
#[cfg(test)]
pub(crate) fn build_wav(sample_rate: u32, channels: u16, pcm: &[i16]) -> Vec<u8> {
    let byte_rate = sample_rate * channels as u32 * 2;
    let data_len = (pcm.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + pcm.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&(channels * 2).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_riff_header() {
        let wav = build_wav(24_000, 1, &[0i16, 123i16]);
        let info = WavInfo::parse(&wav).expect("parse");
        assert_eq!(info.sample_rate, 24_000);
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 16);
        assert_eq!(info.data_offset, 44);
        assert_eq!(info.data_len, 4);
        assert_eq!(info.pcm(&wav), &[0, 0, 123, 0]);
    }

    #[test]
    fn parse_rejects_short_and_non_riff() {
        assert!(WavInfo::parse(b"too short").is_err());
        let mut not_riff = build_wav(24_000, 1, &[0]);
        not_riff[0] = b'X';
        assert!(WavInfo::parse(&not_riff).is_err());
    }

    #[test]
    fn encode_sine_yields_nonempty_packet() {
        // 500 ms of 440 Hz sine at 24 kHz → ~25 frames after 2x upsample.
        let sr = 24_000u32;
        let n = (sr as usize) / 2;
        let pcm: Vec<i16> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (16_000.0 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()) as i16
            })
            .collect();
        let wav = build_wav(sr, 1, &pcm);
        let pkts = encode_wav_to_opus(&wav, TARGET_SR).expect("encode");
        assert!(!pkts.is_empty(), "expected at least one opus packet");
        assert!(pkts.iter().any(|p| !p.is_empty()));
        assert!(
            pkts.len() >= 24 && pkts.len() <= 26,
            "got {} packets",
            pkts.len()
        );
    }

    #[test]
    fn encode_rejects_stereo() {
        let wav = build_wav(24_000, 2, &[0, 0, 0, 0]);
        assert!(encode_wav_to_opus(&wav, TARGET_SR).is_err());
    }
}
