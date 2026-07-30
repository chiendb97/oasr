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
//! - **Raw PCM** in `f32_le` or `i16_le` formats with a caller-specified
//!   sample rate.
//!
//! and converts the result to a target rate ([`decode_audio`] for whole clips,
//! [`PcmStream`] for chunk-by-chunk streaming, which carries the resampler's
//! filter state across chunks).
//!
//! MP3 / Opus / FLAC are deliberately out of scope for v1.  Add them later
//! behind a `symphonia` feature.

pub mod audio;
pub mod resample;

pub use audio::{decode_audio, decode_raw_pcm, AudioError, DecodedAudio, PcmEncoding, PcmStream};
pub use resample::{
    resample_mono, validate_sample_rate, Resampler, MAX_SAMPLE_RATE, MIN_SAMPLE_RATE,
};
