// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Compressed-container decoding via [symphonia].
//!
//! Everything here is *demux + decode to f32 mono*; rate conversion stays in
//! [`crate::resample`] and the raw-PCM paths stay in [`crate::audio`].  The
//! whole module is behind the `codecs` feature, which is on by default: real
//! callers have MP3 podcasts, M4A voice memos and FLAC archives, not headerless
//! PCM, and requiring every one of them to transcode first was the single
//! largest gap between the API and what clients actually hold.
//!
//! **Opus is the one common codec missing.** symphonia has no pure-Rust Opus
//! decoder, and the alternative is a C library (libopus) that would change the
//! build requirements for every user.  A WebM/Ogg body carrying an Opus track
//! therefore demuxes and then fails with [`AudioError::UnsupportedCodec`]
//! naming the codec — a declared gap rather than a silent wrong answer.
//!
//! Container *identification* is unconditional: with the feature off the
//! sniffer still recognises an MP3 and the request is rejected saying so,
//! rather than being reinterpreted as headerless PCM.

#[cfg(feature = "codecs")]
use symphonia::core::audio::GenericAudioBufferRef;
#[cfg(feature = "codecs")]
use symphonia::core::codecs::audio::AudioDecoderOptions;
#[cfg(feature = "codecs")]
use symphonia::core::codecs::CodecParameters;
#[cfg(feature = "codecs")]
use symphonia::core::errors::Error as SymphoniaError;
#[cfg(feature = "codecs")]
use symphonia::core::formats::probe::Hint;
#[cfg(feature = "codecs")]
use symphonia::core::formats::{FormatOptions, TrackType};
#[cfg(feature = "codecs")]
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
#[cfg(feature = "codecs")]
use symphonia::core::meta::MetadataOptions;

#[cfg(feature = "codecs")]
use crate::audio::{f32_to_bytes, AudioError, DecodedAudio};

/// Container families this crate recognises by magic bytes.
///
/// Sniffing exists because the two OpenAI-shaped routes carry no encoding
/// parameter at all — the container *is* the declaration — and because a
/// client that mislabels its upload should still get the right answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    /// RIFF/WAVE — decoded by `hound`, not symphonia, so WAV keeps working
    /// with the `codecs` feature off.
    Wav,
    Flac,
    /// Ogg bitstream: Vorbis is decodable, Opus is not (see the module docs).
    Ogg,
    /// MPEG audio (MP3), either bare frame-sync or behind an ID3v2 tag.
    Mp3,
    /// ISO base media (MP4 / M4A) — AAC or ALAC inside.
    IsoMp4,
    /// Matroska / WebM — what a browser's `MediaRecorder` produces.
    Matroska,
    /// ADTS-framed AAC.
    Aac,
    Aiff,
    Caf,
}

impl Container {
    /// symphonia's extension hint for this family (`None` for WAV, which never
    /// reaches symphonia).
    #[cfg(feature = "codecs")]
    fn hint(self) -> Option<&'static str> {
        match self {
            Container::Wav => None,
            Container::Flac => Some("flac"),
            Container::Ogg => Some("ogg"),
            Container::Mp3 => Some("mp3"),
            Container::IsoMp4 => Some("m4a"),
            Container::Matroska => Some("webm"),
            Container::Aac => Some("aac"),
            Container::Aiff => Some("aiff"),
            Container::Caf => Some("caf"),
        }
    }

    /// Whether the magic identifying this family is specific enough to
    /// override a caller's declared raw-PCM encoding.
    ///
    /// `Mp3` and `Aac` are not: both are identified by an 11-bit frame sync
    /// (`0xFF 0xEx`..`0xFF 0xFx`), which headerless PCM hits by chance roughly
    /// once every 2^11 sample pairs.  Treating that as a container would
    /// mis-decode real PCM, so those two are honoured only when the caller did
    /// not declare a PCM encoding (the multipart routes, which have no
    /// encoding field).
    fn is_unambiguous(self) -> bool {
        !matches!(self, Container::Mp3 | Container::Aac)
    }
}

/// Identify the container in `body` from its leading bytes.
///
/// Returns `None` for a blob that matches nothing — which is the normal
/// outcome for headerless PCM.
pub fn sniff(body: &[u8]) -> Option<Container> {
    let b = body;
    if b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WAVE" {
        return Some(Container::Wav);
    }
    if b.len() >= 4 && &b[0..4] == b"fLaC" {
        return Some(Container::Flac);
    }
    if b.len() >= 4 && &b[0..4] == b"OggS" {
        return Some(Container::Ogg);
    }
    if b.len() >= 12 && &b[4..8] == b"ftyp" {
        return Some(Container::IsoMp4);
    }
    if b.len() >= 4 && b[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        return Some(Container::Matroska);
    }
    if b.len() >= 12 && &b[0..4] == b"FORM" && (&b[8..12] == b"AIFF" || &b[8..12] == b"AIFC") {
        return Some(Container::Aiff);
    }
    if b.len() >= 4 && &b[0..4] == b"caff" {
        return Some(Container::Caf);
    }
    // An ID3v2 tag is unambiguous even though the MPEG frames behind it are
    // not: no PCM stream starts with "ID3" *and* a valid version/flag byte.
    if b.len() >= 5 && &b[0..3] == b"ID3" && b[3] < 0xFF && b[4] < 0xFF {
        return Some(Container::Mp3);
    }
    if b.len() >= 2 && b[0] == 0xFF && (b[1] & 0xE0) == 0xE0 {
        // 11-bit frame sync, shared by MPEG audio and ADTS AAC.  Layer bits
        // 00 mean "reserved" for MPEG, which is how ADTS marks itself.
        let layer = (b[1] >> 1) & 0x03;
        return Some(if layer == 0 {
            Container::Aac
        } else {
            Container::Mp3
        });
    }
    None
}

/// `true` when `body`'s magic is specific enough to override a declared
/// raw-PCM encoding.
pub fn sniff_unambiguous(body: &[u8]) -> Option<Container> {
    sniff(body).filter(|c| c.is_unambiguous())
}

/// Map a `Content-Type` / format name to a container family.
///
/// Accepts both MIME types (`audio/mpeg`) and bare names (`mp3`), because the
/// two front-end surfaces carry each: an upload's part header on one, an
/// `encoding=` value on the other.
pub fn container_from_hint(hint: &str) -> Option<Container> {
    let h = hint.to_ascii_lowercase();
    // Order matters: "audio/x-wav" contains "wav", "audio/webm" contains "web".
    for (needle, c) in [
        ("wav", Container::Wav),
        ("wave", Container::Wav),
        ("flac", Container::Flac),
        ("webm", Container::Matroska),
        ("matroska", Container::Matroska),
        ("mkv", Container::Matroska),
        ("mpeg3", Container::Mp3),
        ("mp3", Container::Mp3),
        ("mpga", Container::Mp3),
        ("mpeg", Container::Mp3),
        ("ogg", Container::Ogg),
        ("oga", Container::Ogg),
        ("opus", Container::Ogg),
        ("vorbis", Container::Ogg),
        ("m4a", Container::IsoMp4),
        ("mp4", Container::IsoMp4),
        ("aac", Container::Aac),
        ("aiff", Container::Aiff),
        ("aif", Container::Aiff),
        ("caf", Container::Caf),
    ] {
        if h.contains(needle) {
            return Some(c);
        }
    }
    None
}

/// Decode a compressed container to f32 mono PCM at the container's own rate.
///
/// `max_samples` bounds the **decoded** waveform, which is the only bound that
/// means anything for a compressed body: a 1 MiB Opus file is minutes of audio,
/// so a byte cap on the request body says nothing about how much memory the
/// decode will take.  Hitting the cap is an error, not a truncation — a
/// silently shortened transcript is worse than a rejected upload.
#[cfg(feature = "codecs")]
pub fn decode_container(
    body: &[u8],
    container: Container,
    max_samples: Option<usize>,
) -> Result<DecodedAudio, AudioError> {
    let mut hint = Hint::new();
    if let Some(ext) = container.hint() {
        hint.with_extension(ext);
    }
    // `Bytes` would avoid this copy, but `MediaSource` is implemented for
    // `Cursor<T: AsRef<[u8]>>` and the source has to be `'static`-free only in
    // the sense of outliving the reader, which a borrowed cursor already does.
    let mss = MediaSourceStream::new(
        Box::new(std::io::Cursor::new(body)),
        MediaSourceStreamOptions::default(),
    );

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| AudioError::Container(format!("{container:?}: {e}")))?;

    let track = format
        .first_track_known_codec(TrackType::Audio)
        .ok_or_else(|| AudioError::Container(format!("{container:?}: no decodable audio track")))?;
    let track_id = track.id;
    let params = match &track.codec_params {
        Some(CodecParameters::Audio(p)) => p.clone(),
        _ => {
            return Err(AudioError::Container(format!(
                "{container:?}: audio track carries no codec parameters"
            )))
        }
    };
    // The container's declared rate; the decoded buffers carry it too, and the
    // buffer wins (a track header can lie, a decoded frame cannot).
    let mut sample_rate = params.sample_rate.unwrap_or(0);

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|_| {
            AudioError::UnsupportedCodec(format!(
                "{:?} in a {container:?} container is not supported by this build",
                params.codec
            ))
        })?;

    let mut out: Vec<f32> = Vec::new();
    let mut scratch: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            // A truncated upload is common enough (a cancelled browser
            // recording) that the frames already decoded are worth keeping.
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            Err(e) => return Err(AudioError::Container(format!("{container:?}: {e}"))),
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // Per symphonia's contract these two are per-packet and recoverable:
            // drop the packet and keep going rather than failing the request.
            Err(SymphoniaError::DecodeError(_)) | Err(SymphoniaError::ResetRequired) => continue,
            Err(e) => return Err(AudioError::Container(format!("{container:?}: {e}"))),
        };
        append_mono(&decoded, &mut out, &mut scratch);
        sample_rate = decoded.spec().rate();
        if let Some(cap) = max_samples {
            if out.len() > cap {
                return Err(AudioError::TooLong(out.len(), cap));
            }
        }
    }

    if out.is_empty() {
        return Err(AudioError::Container(format!(
            "{container:?}: decoded to zero audio frames"
        )));
    }
    if sample_rate == 0 {
        return Err(AudioError::Container(format!(
            "{container:?}: stream declares no sample rate"
        )));
    }
    Ok(DecodedAudio {
        sample_rate,
        samples: f32_to_bytes(&out),
    })
}

/// Append one decoded buffer to `out`, averaging its channels down to mono.
#[cfg(feature = "codecs")]
fn append_mono(buf: &GenericAudioBufferRef<'_>, out: &mut Vec<f32>, scratch: &mut Vec<f32>) {
    let channels = buf.spec().channels().count().max(1);
    if channels == 1 {
        buf.copy_to_vec_interleaved(scratch);
        out.append(scratch);
        return;
    }
    buf.copy_to_vec_interleaved(scratch);
    let inv = 1.0 / channels as f32;
    out.reserve(scratch.len() / channels);
    for frame in scratch.chunks_exact(channels) {
        out.push(frame.iter().sum::<f32>() * inv);
    }
    scratch.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniffs_the_container_families() {
        let mut wav = b"RIFF\0\0\0\0WAVE".to_vec();
        wav.extend_from_slice(&[0u8; 8]);
        assert_eq!(sniff(&wav), Some(Container::Wav));
        assert_eq!(sniff(b"fLaC\0\0\0\0"), Some(Container::Flac));
        assert_eq!(sniff(b"OggS\0\0\0\0"), Some(Container::Ogg));
        assert_eq!(sniff(b"\0\0\0\x18ftypM4A "), Some(Container::IsoMp4));
        assert_eq!(
            sniff(&[0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0]),
            Some(Container::Matroska)
        );
        assert_eq!(sniff(b"FORM\0\0\0\0AIFF"), Some(Container::Aiff));
        assert_eq!(sniff(b"caff\0\0\0\0"), Some(Container::Caf));
        assert_eq!(sniff(b"ID3\x04\x00rest"), Some(Container::Mp3));
        // MPEG layer III frame sync, and the ADTS spelling of the same sync.
        assert_eq!(sniff(&[0xFF, 0xFB, 0x90, 0x00]), Some(Container::Mp3));
        assert_eq!(sniff(&[0xFF, 0xF1, 0x50, 0x80]), Some(Container::Aac));
        assert_eq!(sniff(b"just some bytes"), None);
    }

    /// Headerless PCM hits the MPEG frame sync by chance; treating that as a
    /// container would mis-decode a perfectly valid LINEAR16 body, so the
    /// frame-sync families must not override a declared PCM encoding.
    #[test]
    fn frame_sync_alone_does_not_override_a_declared_pcm_encoding() {
        let pcm = [0xFFu8, 0xFB, 0x00, 0x01];
        assert_eq!(sniff(&pcm), Some(Container::Mp3));
        assert_eq!(sniff_unambiguous(&pcm), None);
        // A real container header still wins.
        assert_eq!(sniff_unambiguous(b"fLaC\0\0\0\0"), Some(Container::Flac));
    }

    #[test]
    fn maps_mime_types_and_bare_names() {
        assert_eq!(container_from_hint("audio/mpeg"), Some(Container::Mp3));
        assert_eq!(container_from_hint("audio/x-flac"), Some(Container::Flac));
        assert_eq!(
            container_from_hint("audio/webm;codecs=opus"),
            Some(Container::Matroska)
        );
        assert_eq!(container_from_hint("m4a"), Some(Container::IsoMp4));
        assert_eq!(container_from_hint("audio/wav"), Some(Container::Wav));
        assert_eq!(container_from_hint("application/json"), None);
    }

    #[cfg(feature = "codecs")]
    #[test]
    fn undecodable_bytes_are_an_error_not_a_panic() {
        let err = decode_container(b"fLaC not really a flac file", Container::Flac, None);
        assert!(matches!(err, Err(AudioError::Container(_))), "{err:?}");
    }
}
