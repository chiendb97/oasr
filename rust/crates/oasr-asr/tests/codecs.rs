// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! End-to-end decode of real compressed bitstreams.
//!
//! The unit tests in `src/` cover sniffing and routing decisions; this file
//! covers the thing they cannot fake — that an actual MP3 / FLAC / OGG / M4A
//! file produces the *same waveform* as the WAV it was encoded from.  Without
//! it, "we support MP3" rests on a `match` arm.
//!
//! Fixtures are one second of a 440 Hz sine at 16 kHz mono, encoded by ffmpeg
//! (`tests/fixtures/`).  FLAC is lossless, so it is asserted sample-exact; the
//! lossy formats are asserted on rate, duration and energy, which is what
//! survives a 32 kbit/s encode.

#![cfg(feature = "codecs")]

use oasr_asr::{
    decode_audio, decode_wav, resample_mono, AudioError, DecodeOptions, PcmEncoding, SourceEncoding,
};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIXTURES}/{name}")).unwrap_or_else(|e| panic!("read {name}: {e}"))
}

fn samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|s| s * s).sum::<f32>() / v.len() as f32).sqrt()
}

/// Decode a container upload — the shape `POST /v1/audio/transcriptions` builds.
fn decode_upload(name: &str, target: Option<u32>) -> Result<Vec<f32>, AudioError> {
    let body = fixture(name);
    let dec = decode_audio(
        &body,
        &DecodeOptions {
            encoding: SourceEncoding::Container,
            target_sample_rate: target,
            ..Default::default()
        },
    )?;
    if target.is_none() {
        assert_eq!(dec.sample_rate, 16_000, "{name} kept its own rate");
    }
    Ok(samples(&dec.samples))
}

#[test]
fn flac_is_bit_exact_against_the_source_wav() {
    let reference = samples(&decode_wav(&fixture("tone.wav")).unwrap().samples);
    let decoded = decode_upload("tone.flac", None).unwrap();
    assert_eq!(decoded.len(), reference.len());
    for (i, (a, b)) in decoded.iter().zip(&reference).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "FLAC differs at sample {i}: {a} vs {b}"
        );
    }
}

#[test]
fn lossy_containers_decode_to_the_same_signal() {
    let reference = samples(&decode_wav(&fixture("tone.wav")).unwrap().samples);
    let want_rms = rms(&reference);
    for name in ["tone.mp3", "tone.ogg", "tone.m4a"] {
        let decoded = decode_upload(name, None).unwrap();
        // Encoder delay and frame padding move the length by up to a frame or
        // two; anything beyond 10% means we decoded the wrong thing.
        let ratio = decoded.len() as f32 / reference.len() as f32;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "{name}: {} samples vs {} in the source",
            decoded.len(),
            reference.len()
        );
        let got = rms(&decoded);
        assert!(
            (got / want_rms - 1.0).abs() < 0.15,
            "{name}: RMS {got} vs {want_rms} — decoded, but not this signal"
        );
    }
}

#[test]
fn a_container_is_resampled_to_the_model_rate() {
    let decoded = decode_upload("tone.mp3", Some(8_000)).unwrap();
    // One second in, one second out — at half the rate.
    assert!(
        (7_200..=8_800).contains(&decoded.len()),
        "got {} samples",
        decoded.len()
    );
}

/// Opus is the declared gap: symphonia has no pure-Rust decoder and libopus
/// would put a C dependency in every build.  The failure must *name* it — a
/// browser posting `audio/webm;codecs=opus` deserves a message it can act on,
/// not "unrecognised container".
#[test]
fn webm_opus_fails_naming_the_codec_rather_than_guessing() {
    let err = decode_upload("tone.webm", None).unwrap_err();
    assert!(
        matches!(err, AudioError::UnsupportedCodec(_)),
        "expected an UnsupportedCodec error, got {err:?}"
    );
}

#[test]
fn the_decoded_length_cap_applies_to_compressed_bodies() {
    let body = fixture("tone.mp3");
    let err = decode_audio(
        &body,
        &DecodeOptions {
            encoding: SourceEncoding::Container,
            max_samples: Some(4_000), // a quarter of the file's one second
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, AudioError::TooLong(_, 4_000)), "{err:?}");
}

/// A container upload decoded at its own rate (no target), with that rate.
fn decode_at_source_rate(name: &str) -> (u32, Vec<f32>) {
    let dec = decode_audio(
        &fixture(name),
        &DecodeOptions {
            encoding: SourceEncoding::Container,
            ..Default::default()
        },
    )
    .unwrap();
    (dec.sample_rate, samples(&dec.samples))
}

/// `max_samples` counts the waveform the decode *returns*.  A serving caller
/// converts to its model's rate and sizes the cap as `--max-audio-seconds x`
/// that rate; counted at the source rate, as it was, a 4-hour cap refused a
/// 48 kHz upload after 80 minutes.  `tone_48k.flac` is one second at 48 kHz:
/// `ffmpeg -f lavfi -i sine=frequency=440:sample_rate=48000:duration=1 -ac 1
/// -c:a flac`.
#[test]
fn the_cap_counts_samples_at_the_rate_the_decode_returns() {
    let body = fixture("tone_48k.flac");
    let opts = |cap| DecodeOptions {
        encoding: SourceEncoding::Container,
        target_sample_rate: Some(16_000),
        max_samples: Some(cap),
        ..Default::default()
    };
    let dec = decode_audio(&body, &opts(16_000)).unwrap();
    assert_eq!(samples(&dec.samples).len(), 16_000);
    let err = decode_audio(&body, &opts(15_999)).unwrap_err();
    assert!(
        matches!(err, AudioError::TooLong(16_000, 15_999)),
        "{err:?}"
    );
}

/// A container is converted a packet at a time, so that its waveform is never
/// held at the source rate; that must not change a sample, since the
/// resampler feeds its filter the same chunks however the packets cut it.
#[test]
fn a_container_converted_while_it_decodes_equals_the_one_shot_resample() {
    for (name, to) in [
        ("tone_48k.flac", 16_000),
        ("tone.flac", 48_000),
        ("tone.mp3", 8_000),
        ("tone.ogg", 44_100),
        ("tone.m4a", 22_050),
    ] {
        let (from, source) = decode_at_source_rate(name);
        let want = resample_mono(&source, from, to).unwrap();
        let got = decode_upload(name, Some(to)).unwrap();
        assert!(
            got == want,
            "{name}: {from} -> {to} Hz differs from the one-shot resample"
        );
    }
}

/// Sample-exact G.711 checks against reference decodes; aggregate energy would
/// not detect an inverted sign convention.
#[test]
fn g711_decodes_sample_exactly_against_ffmpeg() {
    for (name, reference, enc) in [
        ("tone_8k.ulaw", "tone_8k_ulaw_ref.wav", PcmEncoding::MuLaw),
        ("tone_8k.alaw", "tone_8k_alaw_ref.wav", PcmEncoding::ALaw),
    ] {
        let body = fixture(name);
        assert_eq!(
            body.len(),
            8_000,
            "{name}: one second at 8 kHz, one byte per sample"
        );
        let dec = decode_audio(
            &body,
            &DecodeOptions {
                encoding: SourceEncoding::Pcm(enc),
                source_sample_rate: Some(8_000),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(dec.sample_rate, 8_000);
        let got = samples(&dec.samples);
        let want = samples(&decode_wav(&fixture(reference)).unwrap().samples);
        assert_eq!(got.len(), want.len(), "{name}: sample count");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-6, "{name}: sample {i}: {a} vs {b}");
        }
    }
}

/// Telephony arrives at 8 kHz and the model wants 16; the conversion is the
/// same one the WAV path uses, so this only pins that it happens at all — and
/// that it preserves the signal's energy rather than emitting silence.
#[test]
fn g711_is_resampled_to_the_model_rate() {
    let opts = |target| DecodeOptions {
        encoding: SourceEncoding::Pcm(PcmEncoding::MuLaw),
        source_sample_rate: Some(8_000),
        target_sample_rate: target,
        ..Default::default()
    };
    let native = decode_audio(&fixture("tone_8k.ulaw"), &opts(None)).unwrap();
    let up = decode_audio(&fixture("tone_8k.ulaw"), &opts(Some(16_000))).unwrap();
    assert_eq!(up.sample_rate, 16_000);
    assert_eq!(up.samples.len() / 4, 16_000);
    let (before, after) = (rms(&samples(&native.samples)), rms(&samples(&up.samples)));
    assert!(
        (after / before - 1.0).abs() < 0.05,
        "resampling changed the energy: {before} -> {after}"
    );
}

/// A mislabelled upload is the common case, not the exception: browsers send
/// `application/octet-stream`, curl sends nothing.  The magic bytes decide.
#[test]
fn a_wrong_content_type_does_not_break_the_decode() {
    let body = fixture("tone.flac");
    let dec = decode_audio(
        &body,
        &DecodeOptions {
            encoding: SourceEncoding::Container,
            hint: Some("application/octet-stream"),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(dec.sample_rate, 16_000);
}
