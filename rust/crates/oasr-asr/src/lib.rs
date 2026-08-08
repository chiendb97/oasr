// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Audio decoding helpers for the OASR Rust frontend.
//!
//! The engine consumes raw little-endian f32 mono PCM **at the model's own
//! sample rate** — it does not resample, and it derives every frame count from
//! `FeatureConfig.sample_rate` regardless of what a request declares.  Rate
//! conversion is therefore this crate's job, on the way in.  It decodes:
//!
//! - **WAV** containers via `hound` (any bit-depth, multi-channel averaged
//!   down to mono).
//! - **Compressed containers** via `symphonia` (`codecs` feature, on by
//!   default): MP3, FLAC, AAC/M4A, ALAC, OGG-Vorbis, AIFF, CAF and MKV/WebM.
//! - **Raw PCM** in `f32_le`, `i16_le`, µ-law or A-law with a caller-specified
//!   sample rate.
//!
//! and converts the result to a target rate ([`decode_audio`] for whole clips,
//! [`PcmStream`] for chunk-by-chunk streaming, which carries the resampler's
//! filter state across chunks).
//!
//! **Opus is the one gap**, and a declared one: there is no pure-Rust decoder,
//! and pulling in libopus would add a C dependency to every build.  An Opus
//! track demuxes and then fails with [`AudioError::UnsupportedCodec`].

pub mod audio;
pub mod codec;
pub mod encoding;
pub mod resample;

pub use audio::{
    decode_audio, decode_raw_pcm, decode_wav, AudioError, DecodeOptions, DecodedAudio, PcmEncoding,
    PcmStream, SourceEncoding,
};
pub use codec::{container_from_hint, sniff, Container};
pub use encoding::{parse_encoding, EncodingError};
pub use resample::{
    resample_mono, validate_sample_rate, Resampler, MAX_SAMPLE_RATE, MIN_SAMPLE_RATE,
};
