// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Audio decoding to raw f32 mono PCM (little-endian bytes).

use bytes::{BufMut, Bytes, BytesMut};
use hound::{SampleFormat, WavReader};
use thiserror::Error;

use crate::resample::{resample_mono, Resampler, MAX_SAMPLE_RATE, MIN_SAMPLE_RATE};

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
    #[error(
        "sample rate {0} Hz is outside the supported range \
         [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}]"
    )]
    UnsupportedSampleRate(u32),
    #[error("resample: {0}")]
    Resample(String),
}

/// Decoded audio ready to hand to the engine.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Raw little-endian f32 mono PCM samples.
    pub samples: Bytes,
}

impl DecodedAudio {
    /// Convert to `target` Hz, returning `self` untouched when already there.
    ///
    /// Both the source and target rates are validated, so a client that
    /// declares a nonsense `sampleRateHertz` is rejected here rather than
    /// producing a nonsense conversion.
    pub fn resampled_to(self, target: u32) -> Result<Self, AudioError> {
        if self.sample_rate == target {
            crate::resample::validate_sample_rate(target)?;
            return Ok(self);
        }
        let src = bytes_to_f32(&self.samples);
        let out = resample_mono(&src, self.sample_rate, target)?;
        Ok(Self {
            sample_rate: target,
            samples: f32_to_bytes(&out),
        })
    }
}

/// Reinterpret f32-LE bytes as samples.  A trailing partial sample (which the
/// decoders never produce) is ignored.
fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32_to_bytes(v: &[f32]) -> Bytes {
    let mut out = BytesMut::with_capacity(v.len() * 4);
    for &s in v {
        out.put_f32_le(s);
    }
    out.freeze()
}

/// Chunk-by-chunk PCM decode + resample for the streaming RPCs.
///
/// Holds the resampler's filter state across chunks, which is the whole reason
/// this type exists: calling [`decode_audio`] per chunk would restart the
/// filter every time and stamp a discontinuity into the waveform at each chunk
/// boundary.  When the client's rate already matches the model's, no resampler
/// is built and `decode_chunk` is the same byte conversion as
/// [`decode_raw_pcm`].
pub struct PcmStream {
    encoding: PcmEncoding,
    source_rate: u32,
    target_rate: u32,
    resampler: Option<Resampler>,
}

impl PcmStream {
    /// Open a stream decoding `encoding` at `source_rate` and emitting
    /// `target_rate`.  Fails on an implausible rate at *open* time, so a bad
    /// `streaming_config` is rejected before the client sends any audio.
    pub fn new(
        encoding: PcmEncoding,
        source_rate: u32,
        target_rate: u32,
    ) -> Result<Self, AudioError> {
        crate::resample::validate_sample_rate(source_rate)?;
        crate::resample::validate_sample_rate(target_rate)?;
        let resampler = if source_rate == target_rate {
            None
        } else {
            Some(Resampler::new(source_rate, target_rate)?)
        };
        Ok(Self {
            encoding,
            source_rate,
            target_rate,
            resampler,
        })
    }

    /// Source sample rate as declared by the client.
    pub fn source_rate(&self) -> u32 {
        self.source_rate
    }

    /// Sample rate of the bytes this stream emits (the model's rate).
    pub fn target_rate(&self) -> u32 {
        self.target_rate
    }

    /// Whether this stream is converting rather than passing through.
    pub fn is_resampling(&self) -> bool {
        self.resampler.is_some()
    }

    /// Decode one inbound chunk to f32-LE bytes at the target rate.
    ///
    /// When resampling, the returned chunk is not the same duration as the
    /// input: the resampler holds up to one internal block back, released by
    /// the next call or by [`flush`](Self::flush).
    pub fn decode_chunk(&mut self, body: &[u8]) -> Result<Bytes, AudioError> {
        let Some(r) = self.resampler.as_mut() else {
            return Ok(decode_raw_pcm(body, self.encoding, self.source_rate)?.samples);
        };
        let src = pcm_to_f32(body, self.encoding)?;
        let mut out = Vec::with_capacity(src.len() + 64);
        r.push(&src, &mut out)?;
        Ok(f32_to_bytes(&out))
    }

    /// Emit the frames still inside the resampler.  Empty for a passthrough
    /// stream.  Call once, after the client half-closes.
    pub fn flush(&mut self) -> Result<Bytes, AudioError> {
        let Some(r) = self.resampler.as_mut() else {
            return Ok(Bytes::new());
        };
        let mut out = Vec::new();
        r.flush(&mut out)?;
        Ok(f32_to_bytes(&out))
    }
}

/// Widen raw PCM bytes to f32 samples in `[-1, 1)`.
fn pcm_to_f32(body: &[u8], encoding: PcmEncoding) -> Result<Vec<f32>, AudioError> {
    match encoding {
        PcmEncoding::F32Le => {
            if !body.len().is_multiple_of(4) {
                return Err(AudioError::Misaligned(body.len(), 4));
            }
            Ok(bytes_to_f32(body))
        }
        PcmEncoding::I16Le => {
            if !body.len().is_multiple_of(2) {
                return Err(AudioError::Misaligned(body.len(), 2));
            }
            Ok(body
                .chunks_exact(2)
                .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
                .collect())
        }
    }
}

/// Raw PCM input encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmEncoding {
    /// Little-endian f32.
    F32Le,
    /// Little-endian i16; converted to f32 by dividing by 32768.
    I16Le,
}

/// Decode an audio blob into f32 mono PCM, optionally converted to
/// `target_sample_rate`.
///
/// `content_type` is sniffed first; if unrecognized, the blob is interpreted
/// as raw PCM using `default_encoding` and `default_sample_rate`.
///
/// `target_sample_rate` is where a *serving* caller passes the model's own rate:
/// the engine derives every frame count from `FeatureConfig.sample_rate` and
/// ignores the request's, so audio arriving at another rate must be converted
/// before it crosses PyO3.  `None` decodes without conversion — for callers that
/// only want the samples, and for tests.
pub fn decode_audio(
    content_type: Option<&str>,
    body: &[u8],
    default_encoding: PcmEncoding,
    default_sample_rate: Option<u32>,
    target_sample_rate: Option<u32>,
) -> Result<DecodedAudio, AudioError> {
    let kind = content_type.unwrap_or("").to_ascii_lowercase();
    let decoded = if kind.contains("wav") || looks_like_wav(body) {
        decode_wav(body)?
    } else {
        let sr = default_sample_rate.ok_or(AudioError::MissingSampleRate)?;
        decode_raw_pcm(body, default_encoding, sr)?
    };
    match target_sample_rate {
        Some(target) => decoded.resampled_to(target),
        None => Ok(decoded),
    }
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
            if !body.len().is_multiple_of(4) {
                return Err(AudioError::Misaligned(body.len(), 4));
            }
            // Caller-provided f32 → pass through (assume mono).
            Bytes::copy_from_slice(body)
        }
        PcmEncoding::I16Le => {
            if !body.len().is_multiple_of(2) {
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
            assert!(
                (s - expected).abs() < 1e-6,
                "mismatch at {i}: {s} != {expected}"
            );
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
        let dec = decode_audio(None, &wav, PcmEncoding::F32Le, None, None).unwrap();
        assert_eq!(dec.sample_rate, 8000);
    }

    /// A WAV carries its own rate, so a client that posts 44.1 kHz media never
    /// even sets `sampleRateHertz` — the header is the only signal, and before
    /// the target-rate argument existed it went straight to an engine that
    /// assumed 16 kHz.
    #[test]
    fn wav_header_rate_is_converted_to_the_target() {
        let src: Vec<i16> = (0..44_100).map(|i| ((i % 100) as i16) * 100).collect();
        let wav = write_wav_i16(&src, 44_100, 1);
        let dec = decode_audio(None, &wav, PcmEncoding::F32Le, None, Some(16_000)).unwrap();
        assert_eq!(dec.sample_rate, 16_000);
        assert_eq!(dec.samples.len() / 4, 16_000);
    }

    #[test]
    fn raw_pcm_is_converted_to_the_target() {
        let src: Vec<i16> = (0..8_000).map(|i| ((i % 50) as i16) * 200).collect();
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dec =
            decode_audio(None, &bytes, PcmEncoding::I16Le, Some(8_000), Some(16_000)).unwrap();
        assert_eq!(dec.sample_rate, 16_000);
        assert_eq!(dec.samples.len() / 4, 16_000);
    }

    #[test]
    fn matching_target_rate_leaves_the_samples_alone() {
        let src: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dec =
            decode_audio(None, &bytes, PcmEncoding::F32Le, Some(16_000), Some(16_000)).unwrap();
        assert_eq!(dec.sample_rate, 16_000);
        assert_eq!(&dec.samples[..], &bytes[..]);
    }

    #[test]
    fn implausible_declared_rate_is_rejected() {
        let bytes = vec![0u8; 32];
        let err = decode_audio(None, &bytes, PcmEncoding::F32Le, Some(50), Some(16_000));
        assert!(matches!(err, Err(AudioError::UnsupportedSampleRate(50))));
    }

    /// The streaming decoder must produce the same waveform as one-shot decode
    /// of the concatenated chunks; without carried filter state it produces a
    /// discontinuity per chunk instead, which nothing else in the stack checks.
    #[test]
    fn pcm_stream_chunked_matches_one_shot() {
        let src: Vec<i16> = (0..8_000)
            .map(|i| ((i as f32 * 0.05).sin() * 8000.0) as i16)
            .collect();
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();

        let one_shot =
            decode_audio(None, &bytes, PcmEncoding::I16Le, Some(8_000), Some(16_000)).unwrap();

        let mut stream = PcmStream::new(PcmEncoding::I16Le, 8_000, 16_000).unwrap();
        assert!(stream.is_resampling());
        let mut streamed = BytesMut::new();
        // Ragged, but whole samples: a chunk that splits an i16 is a client
        // framing bug and stays an error, same as before this type existed.
        let sizes = [2usize, 778, 4096, 30, 1000];
        let mut off = 0;
        let mut i = 0;
        while off < bytes.len() {
            let n = sizes[i % sizes.len()].min(bytes.len() - off);
            streamed.extend_from_slice(&stream.decode_chunk(&bytes[off..off + n]).unwrap());
            off += n;
            i += 1;
        }
        streamed.extend_from_slice(&stream.flush().unwrap());

        assert_eq!(streamed.len(), one_shot.samples.len());
        for (i, (a, b)) in streamed
            .chunks_exact(4)
            .zip(one_shot.samples.chunks_exact(4))
            .enumerate()
        {
            let (a, b) = (
                f32::from_le_bytes(a.try_into().unwrap()),
                f32::from_le_bytes(b.try_into().unwrap()),
            );
            assert!((a - b).abs() < 1e-6, "chunk mismatch at {i}: {a} vs {b}");
        }
    }

    /// The matched-rate stream must stay a pure byte conversion — the serving
    /// hot path is 16 kHz in, 16 kHz out, and it should not build a filter.
    #[test]
    fn pcm_stream_passthrough_when_rates_match() {
        let mut stream = PcmStream::new(PcmEncoding::I16Le, 16_000, 16_000).unwrap();
        assert!(!stream.is_resampling());
        let src: Vec<i16> = vec![16384, -16384, 0];
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = stream.decode_chunk(&bytes).unwrap();
        assert_eq!(out.len(), src.len() * 4);
        assert!(stream.flush().unwrap().is_empty());
    }

    #[test]
    fn pcm_stream_rejects_an_implausible_rate_at_open() {
        assert!(PcmStream::new(PcmEncoding::I16Le, 0, 16_000).is_err());
        assert!(PcmStream::new(PcmEncoding::I16Le, 16_000, 999_999).is_err());
    }
}
