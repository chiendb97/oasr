// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! The one place an `encoding` name is turned into a decode plan.
//!
//! Both frontends use this mapping so adding a codec cannot make HTTP and gRPC
//! interpret the same encoding name differently.

use crate::audio::{PcmEncoding, SourceEncoding};

/// Why an encoding name was rejected.  Kept apart because the surfaces map
/// them to different statuses: an unset encoding is the caller's mistake
/// (`INVALID_ARGUMENT`), an unsupported one is ours (`UNIMPLEMENTED`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodingError {
    /// No encoding given, on a surface that requires one.
    Unspecified,
    /// A known name this build cannot decode.
    Unsupported(String),
    /// A name we do not know at all.
    Unknown(String),
}

impl std::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodingError::Unspecified => write!(f, "encoding must be set"),
            EncodingError::Unsupported(name) => write!(f, "encoding {name} is not supported"),
            EncodingError::Unknown(name) => write!(f, "unknown encoding {name}"),
        }
    }
}

impl std::error::Error for EncodingError {}

/// Parse an encoding name into `(how to decode, container hint)`.
///
/// The hint is what a caller passes as [`crate::DecodeOptions::hint`]; it
/// matters for the containers whose magic is ambiguous, and is harmless
/// otherwise (the body's own header still wins if the two disagree in a way
/// the decoder can detect).
pub fn parse_encoding(name: &str) -> Result<(SourceEncoding, Option<&'static str>), EncodingError> {
    let upper = name.trim().to_ascii_uppercase();
    Ok(match upper.as_str() {
        "" | "ENCODING_UNSPECIFIED" => return Err(EncodingError::Unspecified),

        // Headerless PCM.
        "LINEAR16" => (SourceEncoding::Pcm(PcmEncoding::I16Le), None),
        "LINEAR32F" => (SourceEncoding::Pcm(PcmEncoding::F32Le), None),
        "MULAW" => (SourceEncoding::Pcm(PcmEncoding::MuLaw), None),
        "ALAW" => (SourceEncoding::Pcm(PcmEncoding::ALaw), None),

        // Containers.  `AUTO` is the one to reach for when relaying files you
        // did not create; the named values exist so a caller that *knows* can
        // say so, and so a Google-STT client's vocabulary keeps working.
        "AUTO" => (SourceEncoding::Container, None),
        "WAV" => (SourceEncoding::Container, Some("wav")),
        "FLAC" => (SourceEncoding::Container, Some("flac")),
        "MP3" => (SourceEncoding::Container, Some("mp3")),
        "M4A" | "MP4" => (SourceEncoding::Container, Some("m4a")),
        "AIFF" => (SourceEncoding::Container, Some("aiff")),
        "CAF" => (SourceEncoding::Container, Some("caf")),
        "OGG" | "OGG_VORBIS" => (SourceEncoding::Container, Some("ogg")),
        "WEBM" | "MKV" => (SourceEncoding::Container, Some("webm")),

        // Named, understood, and genuinely not decodable here.
        "OGG_OPUS" | "WEBM_OPUS" | "AMR" | "AMR_WB" | "SPEEX_WITH_HEADER_BYTE" => {
            return Err(EncodingError::Unsupported(upper))
        }
        other => return Err(EncodingError::Unknown(other.to_string())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_names_carry_no_container_hint() {
        for (name, want) in [
            ("LINEAR16", PcmEncoding::I16Le),
            ("linear32f", PcmEncoding::F32Le),
            ("MULAW", PcmEncoding::MuLaw),
            ("ALAW", PcmEncoding::ALaw),
        ] {
            let (enc, hint) = parse_encoding(name).unwrap();
            assert_eq!(enc, SourceEncoding::Pcm(want), "{name}");
            assert!(hint.is_none(), "{name}");
        }
    }

    #[test]
    fn container_names_map_to_a_hint() {
        for (name, want) in [
            ("WAV", "wav"),
            ("FLAC", "flac"),
            ("MP3", "mp3"),
            ("M4A", "m4a"),
            ("OGG", "ogg"),
        ] {
            let (enc, hint) = parse_encoding(name).unwrap();
            assert_eq!(enc, SourceEncoding::Container);
            assert_eq!(hint, Some(want), "{name}");
        }
        // AUTO defers entirely to the body's own header.
        assert_eq!(
            parse_encoding("AUTO").unwrap(),
            (SourceEncoding::Container, None)
        );
    }

    /// The three rejections are distinct because the surfaces map them to
    /// different HTTP statuses / gRPC codes.
    #[test]
    fn rejections_keep_their_kind() {
        assert_eq!(parse_encoding(""), Err(EncodingError::Unspecified));
        assert_eq!(
            parse_encoding("ENCODING_UNSPECIFIED"),
            Err(EncodingError::Unspecified)
        );
        assert_eq!(
            parse_encoding("OGG_OPUS"),
            Err(EncodingError::Unsupported("OGG_OPUS".into()))
        );
        assert_eq!(
            parse_encoding("nonsense"),
            Err(EncodingError::Unknown("NONSENSE".into()))
        );
    }

    /// Opus is the declared gap; it must be named as unsupported rather than
    /// falling through to "unknown", which would read as a typo.
    #[test]
    fn opus_is_a_declared_gap_not_a_typo() {
        for name in ["OGG_OPUS", "WEBM_OPUS"] {
            assert!(matches!(
                parse_encoding(name),
                Err(EncodingError::Unsupported(_))
            ));
        }
    }
}
