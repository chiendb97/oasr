// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Shared event / command types between the Rust serving frontend and the
//! embedded Python `ASREngine`.
//!
//! Previously these types doubled as a msgpack wire schema for the ZMQ
//! worker boundary; that transport is gone and the types are now used
//! purely in-process (the dispatcher constructs `Event` values directly
//! from `ASREngine.step()` output and the HTTP/gRPC adapters consume
//! them).  `serde` is retained because the gRPC + HTTP layers convert
//! `Event` payloads into their own response shapes via `serde_json`.

use serde::{Deserialize, Serialize};

/// Commands sent into the engine dispatcher.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum Cmd {
    /// Submit a fully-buffered offline transcription.  Audio bytes ride
    /// alongside the command in [`crate::CmdEnvelope::payload`].
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

    /// Push one audio chunk into an open streaming request.
    FeedChunk {
        request_id: String,
        is_last: bool,
    },

    /// Abort a request; the engine frees its cache and emits a final
    /// [`Event::Error`] with code [`ErrorCode::Shutdown`].
    Cancel { request_id: String },

    /// Health probe — returns a [`Event::Pong`] containing engine load.
    Ping { seq: u64 },
}

impl Cmd {
    /// Owning request id for routing in the per-engine `RouterActor`.
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

/// Events emitted by the engine dispatcher back to HTTP / gRPC handlers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum Event {
    /// Request was admitted (or queued).
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

    /// Per-request error (also used for shutdown / worker-lost notifications).
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

    /// Engine is over capacity; frontend should shed load until clear.
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
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
