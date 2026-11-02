// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Compressed-container decoding via [symphonia].
//!
//! Everything here is *demux + decode to f32 mono*, converted to the caller's
//! rate a packet at a time by [`crate::resample`]'s streaming resampler; the
//! raw-PCM paths stay in [`crate::audio`].  The
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
use crate::audio::{AudioError, DecodedAudio, Waveform};
#[cfg(feature = "codecs")]
use crate::resample::{expected_out, Resampler};

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

    /// Whether this magic can override declared raw PCM. Short frame-sync
    /// patterns are ambiguous and apply only when no PCM encoding was declared.
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
/// `max_samples` bounds decoded memory independently of compressed body size.
/// Exceeding it is an error, never silent truncation.
#[cfg(feature = "codecs")]
pub fn decode_container(
    body: &[u8],
    container: Container,
    max_samples: Option<usize>,
) -> Result<DecodedAudio, AudioError> {
    container_waveform(body, container, None, max_samples).map(Waveform::into_decoded)
}

/// [`decode_container`] up to the samples, converted to `target` Hz while they
/// decode when it is set, and held to `max_samples` at that rate.
///
/// A packet at a time, so a long upload is never held at its source rate: the
/// ceiling bounds the decode's memory as well as its duration, and a 48 kHz
/// waveform held whole until a one-shot resample was three times the 16 kHz
/// one the ceiling is sized for.  The output is the one-shot resample's, sample
/// for sample, because [`Resampler`] feeds its filter the same chunks however
/// the input is cut.
#[cfg(feature = "codecs")]
pub(crate) fn container_waveform(
    body: &[u8],
    container: Container,
    target: Option<u32>,
    max_samples: Option<usize>,
) -> Result<Waveform, AudioError> {
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
    let num_frames = track.num_frames;
    let params = match &track.codec_params {
        Some(CodecParameters::Audio(p)) => p.clone(),
        _ => {
            return Err(AudioError::Container(format!(
                "{container:?}: audio track carries no codec parameters"
            )))
        }
    };
    // Reserve what the track says it holds, at the rate the decode returns, so
    // the waveform is not regrown -- and copied -- a doubling at a time.  The
    // header is only a hint, and an untrusted one: the reservation never
    // exceeds the cap the decode is held to anyway, and without a cap (or a
    // declared rate to convert the count with) nothing is reserved.  The
    // decoded buffers carry the rate the samples are actually read at.
    let reserve = match (num_frames, params.sample_rate, max_samples) {
        (Some(n), Some(rate), Some(cap)) if rate > 0 => {
            let returned = match target {
                Some(t) => expected_out(n, rate, t),
                None => n,
            };
            usize::try_from(returned).unwrap_or(usize::MAX).min(cap)
        }
        _ => 0,
    };

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|_| {
            AudioError::UnsupportedCodec(format!(
                "{:?} in a {container:?} container is not supported by this build",
                params.codec
            ))
        })?;

    let mut sink = Sink::new(target, max_samples, reserve);
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
        let rate = decoded.spec().rate();
        if rate == 0 {
            return Err(AudioError::Container(format!(
                "{container:?}: stream declares no sample rate"
            )));
        }
        append_mono(&decoded, sink.packet(rate)?, &mut scratch);
        sink.commit()?;
    }

    let waveform = sink.finish()?;
    if waveform.samples.is_empty() {
        return Err(AudioError::Container(format!(
            "{container:?}: decoded to zero audio frames"
        )));
    }
    Ok(waveform)
}

/// Where a container's decoded packets go: straight into the waveform while
/// they are at the rate the decode returns, through a [`Resampler`] when not.
///
/// The returned rate is the target's, or without one the first packet's.  A
/// decoded buffer at another rate than the one before it opens a segment that
/// is converted from its own rate; labelled with the last packet's rate
/// instead, as the waveform was, every earlier sample played back at the wrong
/// speed.  symphonia's readers were not seen to change rate within a track (it
/// drops concatenated MP3 frames at another rate, as ffmpeg does), but nothing
/// in the decoder's contract rules it out.
#[cfg(feature = "codecs")]
struct Sink {
    target: Option<u32>,
    max_samples: Option<usize>,
    /// The rate the waveform is returned at: the target's, or without one the
    /// first packet's.  0 until a packet arrives.
    rate: u32,
    out: Vec<f32>,
    /// The packet a resampled segment decodes into before it is pushed.
    packet: Vec<f32>,
    segment: Option<Segment>,
}

#[cfg(feature = "codecs")]
struct Segment {
    rate: u32,
    /// `None` while the segment is already at the returned rate.
    resampler: Option<Resampler>,
    /// `out.len()` when the segment began: every earlier segment, flushed.
    start: usize,
    /// Source samples the segment's resampler has taken.
    consumed: u64,
}

#[cfg(feature = "codecs")]
impl Sink {
    fn new(target: Option<u32>, max_samples: Option<usize>, reserve: usize) -> Self {
        let mut out = Vec::new();
        // A failed reservation is not an error: the decode grows `out` as before.
        let _ = out.try_reserve_exact(reserve);
        Self {
            target,
            max_samples,
            rate: 0,
            out,
            packet: Vec::new(),
            segment: None,
        }
    }

    /// The buffer a packet at `rate` Hz decodes into, opening a segment when
    /// the rate is not the current one's.
    fn packet(&mut self, rate: u32) -> Result<&mut Vec<f32>, AudioError> {
        if self.segment.as_ref().map(|seg| seg.rate) != Some(rate) {
            self.close_segment()?;
            if self.rate == 0 {
                self.rate = self.target.unwrap_or(rate);
            }
            let resampler = if rate == self.rate {
                None
            } else {
                Some(Resampler::new(rate, self.rate)?)
            };
            self.segment = Some(Segment {
                rate,
                resampler,
                start: self.out.len(),
                consumed: 0,
            });
        }
        if self
            .segment
            .as_ref()
            .is_some_and(|seg| seg.resampler.is_some())
        {
            self.packet.clear();
            Ok(&mut self.packet)
        } else {
            Ok(&mut self.out)
        }
    }

    /// Take the packet [`packet`](Self::packet) handed out, and refuse the
    /// decode as soon as what it has read would come out longer than the cap.
    fn commit(&mut self) -> Result<(), AudioError> {
        let Some(seg) = self.segment.as_mut() else {
            return Ok(());
        };
        let returned_len = match seg.resampler.as_mut() {
            Some(resampler) => {
                resampler.push(&self.packet, &mut self.out)?;
                seg.consumed += self.packet.len() as u64;
                // What the segment comes to once flushed, which is exactly
                // `round(consumed x to / from)`.  The resampler holds up to a
                // chunk back, so `out.len()` alone would let the decode run
                // past the cap and then hand back more than it allows.
                seg.start as u64 + expected_out(seg.consumed, seg.rate, self.rate)
            }
            None => self.out.len() as u64,
        };
        if let Some(cap) = self.max_samples {
            let returned_len = usize::try_from(returned_len).unwrap_or(usize::MAX);
            if returned_len > cap {
                return Err(AudioError::TooLong(returned_len, cap));
            }
        }
        Ok(())
    }

    /// Flush the current segment's resampler into the waveform.
    fn close_segment(&mut self) -> Result<(), AudioError> {
        if let Some(Segment {
            resampler: Some(mut resampler),
            ..
        }) = self.segment.take()
        {
            resampler.flush(&mut self.out)?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Waveform, AudioError> {
        self.close_segment()?;
        Ok(Waveform {
            sample_rate: self.rate,
            samples: self.out,
        })
    }
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

    /// `(rate, samples)` packets through a [`Sink`], as the decode loop feeds it.
    #[cfg(feature = "codecs")]
    fn through_sink(
        target: Option<u32>,
        cap: Option<usize>,
        packets: &[(u32, &[f32])],
    ) -> Result<Waveform, AudioError> {
        let mut sink = Sink::new(target, cap, 0);
        for &(rate, samples) in packets {
            sink.packet(rate)?.extend_from_slice(samples);
            sink.commit()?;
        }
        sink.finish()
    }

    #[cfg(feature = "codecs")]
    fn tone(n: usize, rate: u32) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 * (std::f32::consts::TAU * 440.0 * i as f32 / rate as f32).sin())
            .collect()
    }

    /// Packets of any size give the one-shot resample, and the cap refuses the
    /// packet whose output would cross it rather than finishing the decode.
    #[cfg(feature = "codecs")]
    #[test]
    fn the_sink_resamples_packet_by_packet_and_holds_the_cap_at_the_returned_rate() {
        use crate::resample::resample_mono;

        let src = tone(48_000 + 333, 48_000);
        let packets: Vec<_> = src.chunks(1_151).map(|c| (48_000, c)).collect();
        let want = resample_mono(&src, 48_000, 16_000).unwrap();

        let got = through_sink(Some(16_000), Some(want.len()), &packets).unwrap();
        assert_eq!(got.sample_rate, 16_000);
        assert!(got.samples == want, "differs from the one-shot resample");

        let err = through_sink(Some(16_000), Some(want.len() - 1), &packets).unwrap_err();
        assert!(
            matches!(err, AudioError::TooLong(n, _) if n == want.len()),
            "{err:?}"
        );

        // Half the length: refused within one packet's output of the cap.
        let cap = want.len() / 2;
        match through_sink(Some(16_000), Some(cap), &packets) {
            Err(AudioError::TooLong(n, c)) => assert!(c == cap && n <= cap + 1_151 / 3 + 1),
            other => panic!("expected TooLong, got {other:?}"),
        }
    }

    /// A rate change opens a segment converted from its own rate, whether the
    /// decode has a target or returns the first packet's rate.
    #[cfg(feature = "codecs")]
    #[test]
    fn a_rate_change_opens_a_segment_converted_from_its_own_rate() {
        use crate::resample::resample_mono;

        let a = tone(8_000, 16_000);
        let b = tone(4_000, 8_000);
        let c = tone(3_000, 16_000);
        let packets = [
            (16_000, &a[..5_000]),
            (16_000, &a[5_000..]),
            (8_000, &b[..1_000]),
            (8_000, &b[1_000..]),
            (16_000, &c[..]),
        ];

        let mut at_16k = a.clone();
        at_16k.extend(resample_mono(&b, 8_000, 16_000).unwrap());
        at_16k.extend(&c);
        for target in [None, Some(16_000)] {
            let got = through_sink(target, None, &packets).unwrap();
            assert_eq!(got.sample_rate, 16_000);
            assert!(got.samples == at_16k, "target {target:?}");
        }

        let mut at_48k = resample_mono(&a, 16_000, 48_000).unwrap();
        at_48k.extend(resample_mono(&b, 8_000, 48_000).unwrap());
        at_48k.extend(resample_mono(&c, 16_000, 48_000).unwrap());
        let got = through_sink(Some(48_000), Some(at_48k.len()), &packets).unwrap();
        assert!(got.samples == at_48k, "segments converted to 48 kHz");
        let err = through_sink(Some(48_000), Some(at_48k.len() - 1), &packets).unwrap_err();
        assert!(matches!(err, AudioError::TooLong(..)), "{err:?}");
    }

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
