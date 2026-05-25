// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Audio decoding helpers for the OASR Rust frontend.
//!
//! The engine consumes raw little-endian f32 mono PCM samples at any sample
//! rate (it resamples internally).  This crate decodes:
//!
//! - **WAV** containers via `hound` (any bit-depth, multi-channel averaged
//!   down to mono).
//! - **Raw PCM** in `f32_le` or `i16_le` formats with a caller-specified
//!   sample rate.
//!
//! MP3 / Opus / FLAC are deliberately out of scope for v1.  Add them later
//! behind a `symphonia` feature.

pub mod audio;

pub use audio::{decode_audio, decode_raw_pcm, AudioError, DecodedAudio, PcmEncoding};
