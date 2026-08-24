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
    /// The body carries no header this build recognises, and the caller did not
    /// declare a raw-PCM encoding to fall back on.
    #[error(
        "unrecognised audio container; supported: WAV, FLAC, MP3, AAC/M4A, \
         OGG (Vorbis), AIFF, CAF, WebM/MKV, or raw PCM with an explicit encoding"
    )]
    UnknownContainer,
    /// The container demuxed but its bitstream could not be decoded.
    #[error("container decode: {0}")]
    Container(String),
    /// The container demuxed and named a codec this build cannot decode —
    /// Opus, in practice.  Declared rather than silently mis-decoded.
    #[error("unsupported codec: {0}")]
    UnsupportedCodec(String),
    /// The decoded waveform exceeded the configured ceiling.  Reported in
    /// samples because that is what the caller can convert to seconds; a byte
    /// cap on a compressed body says nothing about the decode's cost.
    #[error("decoded audio is too long ({0} samples > the {1}-sample limit)")]
    TooLong(usize, usize),
    /// A compressed container reached a build with the `codecs` feature off.
    #[error("{0} decoding requires the `codecs` feature, which this build has disabled")]
    CodecsDisabled(String),
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
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

pub(crate) fn f32_to_bytes(v: &[f32]) -> Bytes {
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
    let width = encoding.sample_width();
    if !body.len().is_multiple_of(width) {
        return Err(AudioError::Misaligned(body.len(), width));
    }
    Ok(match encoding {
        PcmEncoding::F32Le => bytes_to_f32(body),
        PcmEncoding::I16Le => body
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| f32::from(i16::from_le_bytes(c)) / 32768.0)
            .collect(),
        PcmEncoding::MuLaw => body
            .iter()
            .map(|&b| f32::from(mulaw_to_i16(b)) / 32768.0)
            .collect(),
        PcmEncoding::ALaw => body
            .iter()
            .map(|&b| f32::from(alaw_to_i16(b)) / 32768.0)
            .collect(),
    })
}

/// Raw PCM input encoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PcmEncoding {
    /// Little-endian f32.
    #[default]
    F32Le,
    /// Little-endian i16; converted to f32 by dividing by 32768.
    I16Le,
    /// ITU-T G.711 µ-law, one byte per sample.  What telephony actually sends,
    /// and the reason `MULAW` sat in the proto as `UNIMPLEMENTED`.
    MuLaw,
    /// ITU-T G.711 A-law, one byte per sample (the European telephony pair of
    /// µ-law).
    ALaw,
}

impl PcmEncoding {
    /// Bytes per sample in this encoding.
    pub fn sample_width(self) -> usize {
        match self {
            PcmEncoding::F32Le => 4,
            PcmEncoding::I16Le => 2,
            PcmEncoding::MuLaw | PcmEncoding::ALaw => 1,
        }
    }
}

/// G.711 µ-law byte → 14-bit linear sample (ITU-T G.711, the reference
/// `ulaw2linear`).  Exact, table-free, and branch-light enough that a table
/// would only cost a cache line.
fn mulaw_to_i16(byte: u8) -> i16 {
    const BIAS: i32 = 0x84;
    let u = !byte;
    let mut t = (i32::from(u & 0x0F) << 3) + BIAS;
    t <<= (u & 0x70) >> 4;
    if u & 0x80 != 0 {
        (BIAS - t) as i16
    } else {
        (t - BIAS) as i16
    }
}

/// G.711 A-law byte → 13-bit linear sample (ITU-T G.711 `alaw2linear`).
fn alaw_to_i16(byte: u8) -> i16 {
    let a = byte ^ 0x55;
    let mut t = i32::from(a & 0x0F) << 4;
    let seg = (a & 0x70) >> 4;
    match seg {
        0 => t += 8,
        1 => t += 0x108,
        _ => {
            t += 0x108;
            t <<= seg - 1;
        }
    }
    if a & 0x80 != 0 {
        t as i16
    } else {
        -t as i16
    }
}

/// How the caller says the body is encoded.
///
/// Declared PCM permits only unambiguous container overrides. Container input
/// uses full sniffing and never falls back to a guessed raw format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SourceEncoding {
    /// The body carries its own container header.
    #[default]
    Container,
    /// Headerless PCM in this sample format.
    Pcm(PcmEncoding),
}

/// Inputs to [`decode_audio`], grouped rather than passed as six positionals.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeOptions<'a> {
    /// `Content-Type` or bare format name from the transport, if any.  Used as
    /// the first signal; the body's magic bytes are the fallback (and the
    /// tiebreaker, since clients mislabel uploads routinely).
    pub hint: Option<&'a str>,
    /// What the caller says the body is.
    pub encoding: SourceEncoding,
    /// Rate of the body's samples when `encoding` is [`SourceEncoding::Pcm`].
    /// Ignored for a container, which carries its own.
    pub source_sample_rate: Option<u32>,
    /// Rate to convert to — where a *serving* caller passes the model's own:
    /// the engine derives every frame count from `FeatureConfig.sample_rate`
    /// and ignores the request's, so audio arriving at another rate must be
    /// converted before it crosses PyO3.  `None` decodes without conversion.
    pub target_sample_rate: Option<u32>,
    /// Ceiling on the decoded waveform, in samples at its **source** rate.
    /// `None` disables.  A byte cap on the request body cannot stand in for
    /// this once compressed containers are accepted: 1 MiB of Opus is an hour
    /// of audio, and the allocation happens before anything could notice.
    pub max_samples: Option<usize>,
}

/// Decode an audio blob into f32 mono PCM, optionally converted to
/// `opts.target_sample_rate`.
pub fn decode_audio(body: &[u8], opts: &DecodeOptions<'_>) -> Result<DecodedAudio, AudioError> {
    let decoded = decode_to_source_rate(body, opts)?;
    if let Some(cap) = opts.max_samples {
        let n = decoded.samples.len() / 4;
        if n > cap {
            return Err(AudioError::TooLong(n, cap));
        }
    }
    match opts.target_sample_rate {
        Some(target) => decoded.resampled_to(target),
        None => Ok(decoded),
    }
}

/// Decode without rate conversion: pick the container (or raw PCM) and run it.
fn decode_to_source_rate(
    body: &[u8],
    opts: &DecodeOptions<'_>,
) -> Result<DecodedAudio, AudioError> {
    let hinted = opts.hint.and_then(crate::codec::container_from_hint);
    let container = match opts.encoding {
        // A declared container: the hint wins, the magic is the fallback, and
        // an ambiguous frame sync is allowed because there is no PCM reading
        // of this body to protect.
        SourceEncoding::Container => hinted
            .or_else(|| crate::codec::sniff(body))
            .ok_or(AudioError::UnknownContainer)?,
        // Declared PCM: only a header we are sure about may override it.
        SourceEncoding::Pcm(_) => match hinted.or_else(|| crate::codec::sniff_unambiguous(body)) {
            Some(c) => c,
            None => {
                let sr = opts
                    .source_sample_rate
                    .ok_or(AudioError::MissingSampleRate)?;
                let SourceEncoding::Pcm(enc) = opts.encoding else {
                    unreachable!("matched SourceEncoding::Pcm above")
                };
                return decode_raw_pcm(body, enc, sr);
            }
        },
    };
    match container {
        // WAV stays on `hound`: it is the hot path for the raw-PCM surfaces,
        // it handles 24-bit containers symphonia's reader treats differently,
        // and it keeps WAV working with the `codecs` feature off.
        crate::codec::Container::Wav => decode_wav(body),
        other => decode_compressed(body, other, opts.max_samples),
    }
}

#[cfg(feature = "codecs")]
fn decode_compressed(
    body: &[u8],
    container: crate::codec::Container,
    max_samples: Option<usize>,
) -> Result<DecodedAudio, AudioError> {
    crate::codec::decode_container(body, container, max_samples)
}

#[cfg(not(feature = "codecs"))]
fn decode_compressed(
    _body: &[u8],
    container: crate::codec::Container,
    _max_samples: Option<usize>,
) -> Result<DecodedAudio, AudioError> {
    Err(AudioError::CodecsDisabled(format!("{container:?}")))
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
    // f32 is already the engine's own format: pass the bytes through rather
    // than widening them into an identical copy.
    if encoding == PcmEncoding::F32Le {
        if !body.len().is_multiple_of(4) {
            return Err(AudioError::Misaligned(body.len(), 4));
        }
        return Ok(DecodedAudio {
            sample_rate,
            samples: Bytes::copy_from_slice(body),
        });
    }
    Ok(DecodedAudio {
        sample_rate,
        samples: f32_to_bytes(&pcm_to_f32(body, encoding)?),
    })
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        assert!((back[0] - 0.5).abs() < 1e-3);
        assert!((back[1] - -0.5).abs() < 1e-3);
        assert!((back[2] - 0.0).abs() < 1e-3);
    }

    /// Options for a declared-PCM body, the shape both Google-STT-style
    /// surfaces build.
    fn pcm_opts(enc: PcmEncoding, src: Option<u32>, target: Option<u32>) -> DecodeOptions<'static> {
        DecodeOptions {
            hint: None,
            encoding: SourceEncoding::Pcm(enc),
            source_sample_rate: src,
            target_sample_rate: target,
            max_samples: None,
        }
    }

    #[test]
    fn detect_wav_by_magic() {
        let wav = write_wav_i16(&[0, 0, 0], 8000, 1);
        let dec = decode_audio(&wav, &pcm_opts(PcmEncoding::F32Le, None, None)).unwrap();
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
        let dec = decode_audio(&wav, &pcm_opts(PcmEncoding::F32Le, None, Some(16_000))).unwrap();
        assert_eq!(dec.sample_rate, 16_000);
        assert_eq!(dec.samples.len() / 4, 16_000);
    }

    #[test]
    fn raw_pcm_is_converted_to_the_target() {
        let src: Vec<i16> = (0..8_000).map(|i| ((i % 50) as i16) * 200).collect();
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dec = decode_audio(
            &bytes,
            &pcm_opts(PcmEncoding::I16Le, Some(8_000), Some(16_000)),
        )
        .unwrap();
        assert_eq!(dec.sample_rate, 16_000);
        assert_eq!(dec.samples.len() / 4, 16_000);
    }

    #[test]
    fn matching_target_rate_leaves_the_samples_alone() {
        let src: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dec = decode_audio(
            &bytes,
            &pcm_opts(PcmEncoding::F32Le, Some(16_000), Some(16_000)),
        )
        .unwrap();
        assert_eq!(dec.sample_rate, 16_000);
        assert_eq!(&dec.samples[..], &bytes[..]);
    }

    #[test]
    fn implausible_declared_rate_is_rejected() {
        let bytes = vec![0u8; 32];
        let err = decode_audio(
            &bytes,
            &pcm_opts(PcmEncoding::F32Le, Some(50), Some(16_000)),
        );
        assert!(matches!(err, Err(AudioError::UnsupportedSampleRate(50))));
    }

    // -- G.711 -----------------------------------------------------------

    /// G.711 reference points, including µ-law's complemented sign convention.
    #[test]
    fn mulaw_matches_the_g711_reference_points() {
        assert_eq!(mulaw_to_i16(0x7F), 0); // -0
        assert_eq!(mulaw_to_i16(0xFF), 0); // +0
        assert_eq!(mulaw_to_i16(0x00), -32124); // negative full scale
        assert_eq!(mulaw_to_i16(0x80), 32124); // positive full scale
        assert_eq!(mulaw_to_i16(0xD5), 716);
        assert_eq!(mulaw_to_i16(0x55), -716);
    }

    #[test]
    fn alaw_matches_the_g711_reference_points() {
        assert_eq!(alaw_to_i16(0xD5), 8); // smallest positive step
        assert_eq!(alaw_to_i16(0x55), -8);
        assert_eq!(alaw_to_i16(0xAA), 32256); // positive full scale
        assert_eq!(alaw_to_i16(0x2A), -32256);
        assert_eq!(alaw_to_i16(0x00), -5504);
        assert_eq!(alaw_to_i16(0x80), 5504);
    }

    #[test]
    fn mulaw_body_decodes_one_byte_per_sample() {
        let body = [0xFFu8, 0x80, 0x00, 0x7F];
        let dec = decode_raw_pcm(&body, PcmEncoding::MuLaw, 8_000).unwrap();
        let back: Vec<f32> = dec
            .samples
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        assert_eq!(back.len(), 4, "µ-law is one byte per sample");
        assert_eq!(back[0], 0.0);
        assert!((back[1] - 32124.0 / 32768.0).abs() < 1e-6);
        assert!((back[2] + 32124.0 / 32768.0).abs() < 1e-6);
    }

    /// A µ-law body is a byte stream: no alignment constraint, unlike i16/f32.
    #[test]
    fn mulaw_accepts_an_odd_length_body() {
        assert!(decode_raw_pcm(&[0xFFu8; 5], PcmEncoding::MuLaw, 8_000).is_ok());
        assert!(matches!(
            decode_raw_pcm(&[0u8; 5], PcmEncoding::I16Le, 8_000),
            Err(AudioError::Misaligned(5, 2))
        ));
    }

    // -- container routing ------------------------------------------------

    /// The whole point of the ceiling: a compressed body's *byte* length says
    /// nothing about how much waveform it decodes to, so the cap has to be on
    /// the decoded samples and has to fire before the request is admitted.
    #[test]
    fn the_decoded_length_ceiling_rejects_rather_than_truncates() {
        let src: Vec<i16> = vec![0; 16_000];
        let wav = write_wav_i16(&src, 16_000, 1);
        let opts = DecodeOptions {
            encoding: SourceEncoding::Pcm(PcmEncoding::F32Le),
            max_samples: Some(8_000),
            ..Default::default()
        };
        assert!(matches!(
            decode_audio(&wav, &opts),
            Err(AudioError::TooLong(16_000, 8_000))
        ));
        // One sample under the cap still passes.
        let opts = DecodeOptions {
            max_samples: Some(16_000),
            ..opts
        };
        assert!(decode_audio(&wav, &opts).is_ok());
    }

    /// A file upload declares no encoding, so a body matching no container is
    /// an error — never a blob reinterpreted as PCM at a guessed rate.
    #[test]
    fn a_container_upload_that_matches_nothing_is_rejected() {
        let opts = DecodeOptions {
            encoding: SourceEncoding::Container,
            ..Default::default()
        };
        assert!(matches!(
            decode_audio(b"not audio at all", &opts),
            Err(AudioError::UnknownContainer)
        ));
    }

    /// A declared LINEAR16 body whose first bytes happen to look like an MPEG
    /// frame sync must still decode as PCM.  The reverse — sniffing winning —
    /// silently mangles real telephony audio.
    #[test]
    fn declared_pcm_survives_an_accidental_frame_sync() {
        let src: Vec<i16> = vec![-1281, 300, -20, 7]; // 0xFF 0xFA ... little-endian
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(bytes[0], 0xFF, "test fixture must start with a frame sync");
        assert_eq!(bytes[1] & 0xE0, 0xE0);
        let dec = decode_audio(&bytes, &pcm_opts(PcmEncoding::I16Le, Some(16_000), None)).unwrap();
        assert_eq!(dec.samples.len() / 4, src.len());
    }

    /// WAV declared as a container still goes through `hound`, not symphonia:
    /// the raw-PCM surfaces post WAV constantly and the two readers disagree
    /// about 24-bit scaling.  (Real compressed bitstreams are decoded end to
    /// end in `tests/codecs.rs`, against fixtures.)
    #[test]
    fn a_declared_wav_container_takes_the_hound_path() {
        let wav = write_wav_i16(&[1000, -1000, 500], 22_050, 1);
        assert_eq!(
            crate::codec::sniff(&wav),
            Some(crate::codec::Container::Wav)
        );
        let opts = DecodeOptions {
            encoding: SourceEncoding::Container,
            hint: Some("audio/wav"),
            ..Default::default()
        };
        let dec = decode_audio(&wav, &opts).unwrap();
        assert_eq!(dec.sample_rate, 22_050);
        assert_eq!(dec.samples.len() / 4, 3);
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

        let one_shot = decode_audio(
            &bytes,
            &pcm_opts(PcmEncoding::I16Le, Some(8_000), Some(16_000)),
        )
        .unwrap();

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
        for (i, (&a, &b)) in streamed
            .as_chunks::<4>()
            .0
            .iter()
            .zip(one_shot.samples.as_chunks::<4>().0.iter())
            .enumerate()
        {
            let (a, b) = (f32::from_le_bytes(a), f32::from_le_bytes(b));
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
