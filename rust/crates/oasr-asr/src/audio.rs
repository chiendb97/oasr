// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Audio decoding to raw f32 mono PCM (little-endian bytes).

use bytes::{BufMut, Bytes, BytesMut};
use hound::{SampleFormat, WavReader};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("wav decode: {0}")]
    Wav(#[from] hound::Error),
    #[error("unsupported sample format")]
    Unsupported,
    #[error("buffer not a multiple of sample width (got {0} bytes, expected multiple of {1})")]
    Misaligned(usize, usize),
    #[error("missing sample rate for raw PCM input")]
    MissingSampleRate,
}

/// Decoded audio ready to send as a ZMQ payload frame.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Raw little-endian f32 mono PCM samples.
    pub samples: Bytes,
}

/// Raw PCM input encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmEncoding {
    /// Little-endian f32.
    F32Le,
    /// Little-endian i16; converted to f32 by dividing by 32768.
    I16Le,
}

/// Decode an audio blob into f32 mono PCM.
///
/// `content_type` is sniffed first; if unrecognized, the blob is interpreted
/// as raw PCM using `default_encoding` and `default_sample_rate`.
pub fn decode_audio(
    content_type: Option<&str>,
    body: &[u8],
    default_encoding: PcmEncoding,
    default_sample_rate: Option<u32>,
) -> Result<DecodedAudio, AudioError> {
    let kind = content_type.unwrap_or("").to_ascii_lowercase();
    if kind.contains("wav") || looks_like_wav(body) {
        return decode_wav(body);
    }
    let sr = default_sample_rate.ok_or(AudioError::MissingSampleRate)?;
    decode_raw_pcm(body, default_encoding, sr)
}

/// Decode a WAV container.
pub fn decode_wav(body: &[u8]) -> Result<DecodedAudio, AudioError> {
    let mut reader = WavReader::new(std::io::Cursor::new(body))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let bps = spec.bits_per_sample;
    let mut out = BytesMut::with_capacity(body.len()); // upper bound

    match (spec.sample_format, bps) {
        (SampleFormat::Float, 32) => {
            // Interleaved float32; average channels.
            let samples: Vec<f32> = reader.samples::<f32>().collect::<Result<_, _>>()?;
            for frame in samples.chunks_exact(channels) {
                let mean: f32 = frame.iter().sum::<f32>() / channels as f32;
                out.put_f32_le(mean);
            }
            let rem = samples.len() % channels;
            if rem != 0 {
                let frame = &samples[samples.len() - rem..];
                let mean: f32 = frame.iter().sum::<f32>() / rem as f32;
                out.put_f32_le(mean);
            }
        }
        (SampleFormat::Int, 16) => {
            let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<_, _>>()?;
            const SCALE: f32 = 32768.0;
            for frame in samples.chunks_exact(channels) {
                let acc: i32 = frame.iter().map(|&s| s as i32).sum();
                let mean = (acc as f32) / (channels as f32 * SCALE);
                out.put_f32_le(mean);
            }
        }
        (SampleFormat::Int, 24) | (SampleFormat::Int, 32) => {
            let samples: Vec<i32> = reader.samples::<i32>().collect::<Result<_, _>>()?;
            // hound's 24/32-bit ints saturate to i32 range; scale to [-1,1).
            const SCALE: f32 = 2_147_483_648.0;
            for frame in samples.chunks_exact(channels) {
                let mean = (frame.iter().map(|&s| s as f64).sum::<f64>()
                    / channels as f64
                    / SCALE as f64) as f32;
                out.put_f32_le(mean);
            }
        }
        _ => return Err(AudioError::Unsupported),
    }

    Ok(DecodedAudio {
        sample_rate: spec.sample_rate,
        samples: out.freeze(),
    })
}

/// Decode raw PCM bytes.  No header parsing.
pub fn decode_raw_pcm(
    body: &[u8],
    encoding: PcmEncoding,
    sample_rate: u32,
) -> Result<DecodedAudio, AudioError> {
    let samples = match encoding {
        PcmEncoding::F32Le => {
            if body.len() % 4 != 0 {
                return Err(AudioError::Misaligned(body.len(), 4));
            }
            // Caller-provided f32 → pass through (assume mono).
            Bytes::copy_from_slice(body)
        }
        PcmEncoding::I16Le => {
            if body.len() % 2 != 0 {
                return Err(AudioError::Misaligned(body.len(), 2));
            }
            let n = body.len() / 2;
            let mut out = BytesMut::with_capacity(n * 4);
            for i in 0..n {
                let lo = body[2 * i] as i16;
                let hi = body[2 * i + 1] as i16;
                let s = (hi << 8) | (lo & 0xff);
                out.put_f32_le((s as f32) / 32768.0);
            }
            out.freeze()
        }
    };
    Ok(DecodedAudio {
        sample_rate,
        samples,
    })
}

fn looks_like_wav(body: &[u8]) -> bool {
    body.len() >= 12 && &body[0..4] == b"RIFF" && &body[8..12] == b"WAVE"
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn write_wav_i16(samples: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut w = hound::WavWriter::new(std::io::Cursor::new(&mut buf), spec).unwrap();
            for s in samples {
                w.write_sample(*s).unwrap();
            }
            w.finalize().unwrap();
        }
        buf
    }

    #[test]
    fn wav_i16_mono_roundtrip() {
        let src: Vec<i16> = (0..32).map(|i| (i as i16) * 100).collect();
        let wav = write_wav_i16(&src, 16000, 1);
        let dec = decode_wav(&wav).unwrap();
        assert_eq!(dec.sample_rate, 16000);
        let samples: Vec<f32> = dec
            .samples
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(samples.len(), src.len());
        for (i, s) in samples.iter().enumerate() {
            let expected = (src[i] as f32) / 32768.0;
            assert!((s - expected).abs() < 1e-6, "mismatch at {i}: {s} != {expected}");
        }
    }

    #[test]
    fn wav_i16_stereo_averaged_to_mono() {
        // [L, R, L, R, ...] interleaved.  Mean of each frame.
        let src: Vec<i16> = vec![1000, 3000, 2000, -2000];
        let wav = write_wav_i16(&src, 16000, 2);
        let dec = decode_wav(&wav).unwrap();
        let samples: Vec<f32> = dec
            .samples
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - (1000.0 + 3000.0) / 2.0 / 32768.0).abs() < 1e-6);
        assert!((samples[1] - (2000.0 + -2000.0) / 2.0 / 32768.0).abs() < 1e-6);
    }

    #[test]
    fn raw_f32_passthrough() {
        let src: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dec = decode_raw_pcm(&bytes, PcmEncoding::F32Le, 16000).unwrap();
        assert_eq!(dec.sample_rate, 16000);
        let back: Vec<f32> = dec
            .samples
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(back, src);
    }

    #[test]
    fn raw_i16_scaled_to_f32() {
        let src: Vec<i16> = vec![16384, -16384, 0];
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dec = decode_raw_pcm(&bytes, PcmEncoding::I16Le, 16000).unwrap();
        let back: Vec<f32> = dec
            .samples
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!((back[0] - 0.5).abs() < 1e-3);
        assert!((back[1] - -0.5).abs() < 1e-3);
        assert!((back[2] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn detect_wav_by_magic() {
        let wav = write_wav_i16(&[0, 0, 0], 8000, 1);
        let dec = decode_audio(None, &wav, PcmEncoding::F32Le, None).unwrap();
        assert_eq!(dec.sample_rate, 8000);
    }
}
