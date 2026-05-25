// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! MessagePack wire schema shared with `oasr/serving/ipc.py`.
//!
//! Each logical message is one or two ZMQ frames:
//! - Frame 1: MessagePack-encoded header (a tagged enum, see [`Cmd`] / [`Event`]).
//! - Frame 2 (optional): raw little-endian f32 mono PCM samples — only for
//!   commands that carry audio ([`Cmd::CreateOffline`] and [`Cmd::FeedChunk`]).
//!
//! Wire compatibility with Python is achieved by using serde with
//! `#[serde(tag = "type")]` on the enums; the Python side mirrors the same
//! `{"type": ..., ...}` shape.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Commands sent from the Rust frontend to the Python worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum Cmd {
    /// Submit a fully-buffered offline transcription.  Audio rides in a
    /// separate ZMQ frame after the header.
    CreateOffline {
        request_id: String,
        sample_rate: u32,
        #[serde(default)]
        priority: i32,
    },

    /// Open a streaming request.  Audio chunks arrive via [`Cmd::FeedChunk`].
    CreateStreaming {
        request_id: String,
        sample_rate: u32,
        #[serde(default)]
        priority: i32,
    },

    /// Push one audio chunk into an open streaming request.  Audio in a
    /// separate ZMQ frame.
    FeedChunk {
        request_id: String,
        is_last: bool,
    },

    /// Abort a request; the worker frees its cache and emits a single
    /// [`Event::Error`] with code [`ErrorCode::Shutdown`] or just drops.
    Cancel { request_id: String },

    /// Health check round-trip.
    Ping { seq: u64 },
}

impl Cmd {
    /// True if this command carries a binary audio payload in a second frame.
    pub fn has_audio_payload(&self) -> bool {
        matches!(self, Cmd::CreateOffline { .. } | Cmd::FeedChunk { .. })
    }

    /// Owning request id for routing in the per-worker `RouterActor`.
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Cmd::CreateOffline { request_id, .. }
            | Cmd::CreateStreaming { request_id, .. }
            | Cmd::FeedChunk { request_id, .. }
            | Cmd::Cancel { request_id } => Some(request_id.as_str()),
            Cmd::Ping { .. } => None,
        }
    }
}

/// Events emitted by the Python worker back to the Rust frontend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum Event {
    /// Request was successfully admitted (or queued).
    Accepted { request_id: String },

    /// Streaming partial transcript; more updates expected.
    Partial {
        request_id: String,
        text: String,
        tokens: Vec<Vec<u32>>,
        scores: Option<Vec<f32>>,
    },

    /// Final transcript; no further events for this request.
    Final {
        request_id: String,
        text: String,
        tokens: Vec<Vec<u32>>,
        scores: Option<Vec<f32>>,
    },

    /// Per-request error (also used for shutdown notifications).
    Error {
        request_id: String,
        code: ErrorCode,
        message: String,
    },

    /// Heartbeat response with load + model metadata.
    Pong {
        seq: u64,
        model_info: Option<ModelInfo>,
        num_running: u32,
        num_waiting: u32,
    },

    /// Worker is over capacity; frontend should shed load until clear.
    Overloaded { reason: String, queue_depth: u32 },
}

impl Event {
    /// Owning request id (for events that target a specific request).
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Event::Accepted { request_id }
            | Event::Partial { request_id, .. }
            | Event::Final { request_id, .. }
            | Event::Error { request_id, .. } => Some(request_id.as_str()),
            Event::Pong { .. } | Event::Overloaded { .. } => None,
        }
    }

    /// Whether this event terminates the per-request channel.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Event::Final { .. } | Event::Error { .. })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ErrorCode {
    #[serde(rename = "BUSY")]
    Busy,
    #[serde(rename = "UNKNOWN_REQUEST")]
    UnknownRequest,
    #[serde(rename = "INVALID_CMD")]
    InvalidCmd,
    #[serde(rename = "INTERNAL")]
    Internal,
    #[serde(rename = "SHUTDOWN")]
    Shutdown,
    #[serde(rename = "WORKER_LOST")]
    WorkerLost,
}

/// Static model metadata returned in `Pong.model_info`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    #[serde(default)]
    pub ckpt_dir: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub chunk_size: Option<u32>,
    #[serde(default)]
    pub max_batch_size: Option<u32>,
    #[serde(default)]
    pub decoder_type: Option<String>,
    #[serde(default)]
    pub vocab_size: Option<u32>,
}

// ---------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum WireError {
    #[error("msgpack encode: {0}")]
    Encode(#[from] rmp_serde::encode::Error),
    #[error("msgpack decode: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
}

/// MessagePack-encode a command header.
pub fn encode_cmd(cmd: &Cmd) -> Result<Vec<u8>, WireError> {
    // ``to_vec_named`` writes struct/variant field names as map keys, matching
    // python-msgpack's default decode shape.
    Ok(rmp_serde::to_vec_named(cmd)?)
}

/// MessagePack-decode an event header.
pub fn decode_event(bytes: &[u8]) -> Result<Event, WireError> {
    Ok(rmp_serde::from_slice(bytes)?)
}

/// MessagePack-encode an event header (used by tests and the optional
/// in-process echo backend).
pub fn encode_event(event: &Event) -> Result<Vec<u8>, WireError> {
    Ok(rmp_serde::to_vec_named(event)?)
}

/// MessagePack-decode a command header.
pub fn decode_cmd(bytes: &[u8]) -> Result<Cmd, WireError> {
    Ok(rmp_serde::from_slice(bytes)?)
}

// ---------------------------------------------------------------------------
// Tests — exercise round-trips and check the tag layout matches Python.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmd_roundtrip() {
        let cases = vec![
            Cmd::CreateOffline {
                request_id: "abc".into(),
                sample_rate: 16000,
                priority: 0,
            },
            Cmd::CreateStreaming {
                request_id: "def".into(),
                sample_rate: 8000,
                priority: -1,
            },
            Cmd::FeedChunk {
                request_id: "ghi".into(),
                is_last: true,
            },
            Cmd::Cancel {
                request_id: "jkl".into(),
            },
            Cmd::Ping { seq: 7 },
        ];
        for c in &cases {
            let bytes = encode_cmd(c).unwrap();
            let back = decode_cmd(&bytes).unwrap();
            assert_eq!(*c, back, "cmd roundtrip mismatch");
        }
    }

    #[test]
    fn event_roundtrip() {
        let cases = vec![
            Event::Accepted { request_id: "r1".into() },
            Event::Partial {
                request_id: "r2".into(),
                text: "hello".into(),
                tokens: vec![vec![1, 2, 3]],
                scores: Some(vec![-0.1]),
            },
            Event::Final {
                request_id: "r3".into(),
                text: "done".into(),
                tokens: vec![vec![4]],
                scores: None,
            },
            Event::Error {
                request_id: "r4".into(),
                code: ErrorCode::Busy,
                message: "queue full".into(),
            },
            Event::Pong {
                seq: 42,
                model_info: Some(ModelInfo {
                    ckpt_dir: Some("/tmp/m".into()),
                    device: Some("cuda".into()),
                    dtype: Some("float16".into()),
                    chunk_size: Some(16),
                    max_batch_size: Some(32),
                    decoder_type: Some("ctc_prefix_beam".into()),
                    vocab_size: Some(5002),
                }),
                num_running: 3,
                num_waiting: 5,
            },
            Event::Overloaded {
                reason: "in-flight 257 >= cap 256".into(),
                queue_depth: 257,
            },
        ];
        for e in &cases {
            let bytes = encode_event(e).unwrap();
            let back = decode_event(&bytes).unwrap();
            assert_eq!(*e, back, "event roundtrip mismatch");
        }
    }

    #[test]
    fn has_audio_payload_is_correct() {
        assert!(Cmd::CreateOffline {
            request_id: "".into(),
            sample_rate: 0,
            priority: 0,
        }
        .has_audio_payload());
        assert!(Cmd::FeedChunk {
            request_id: "".into(),
            is_last: false,
        }
        .has_audio_payload());
        assert!(!Cmd::CreateStreaming {
            request_id: "".into(),
            sample_rate: 0,
            priority: 0,
        }
        .has_audio_payload());
        assert!(!Cmd::Cancel { request_id: "".into() }.has_audio_payload());
        assert!(!Cmd::Ping { seq: 0 }.has_audio_payload());
    }
}
