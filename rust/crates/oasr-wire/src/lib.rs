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

/// Per-request decoding options forwarded verbatim into the Python engine's
/// `DecodingOptions` (see `oasr/engine/request.py`).  Every field is
/// optional; `None` keeps the engine default, so a default-constructed
/// params value is indistinguishable from not sending one.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DecodingParams {
    /// Hypotheses to detokenize into `Event::Final::nbest_texts`
    /// (maps from the proto `max_alternatives`).
    #[serde(default)]
    pub n_best: Option<u32>,
    /// Per-request generation cap for the AR families (AED / LLM).
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
    /// `> 0` enables sampling for the AR families (default greedy).
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-k filter (0 = disabled); only meaningful with `temperature > 0`.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Nucleus filter in (0, 1]; only meaningful with `temperature > 0`.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Speech-LLM user-prompt override (ignored by other families).
    #[serde(default)]
    pub prompt: Option<String>,
}

impl DecodingParams {
    /// Whether every field is `None` — callers skip building the Python-side
    /// options dict entirely in that case.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

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
        #[serde(default)]
        decoding: Option<DecodingParams>,
    },

    /// Open a streaming request.  Audio chunks arrive via [`Cmd::FeedChunk`].
    CreateStreaming {
        request_id: String,
        sample_rate: u32,
        #[serde(default)]
        priority: i32,
        #[serde(default)]
        decoding: Option<DecodingParams>,
    },

    /// Push one audio chunk into an open streaming request.
    FeedChunk { request_id: String, is_last: bool },

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
        /// Detokenized top-N transcripts aligned with `tokens` rows
        /// (`nbest_texts[0] == text`); present only when the request asked
        /// for more than one alternative and the decode family produced
        /// multiple hypotheses.
        #[serde(default)]
        nbest_texts: Option<Vec<String>>,
        /// End time (seconds) of the last decoded token, from the engine's
        /// per-token timestamps (decode families with alignments —
        /// Paraformer CIF).  Maps to the proto `result_end_time`.
        #[serde(default)]
        end_time_s: Option<f32>,
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

/// Softmax-normalized posteriors over the returned n-best scores.
///
/// The engine's `scores` are per-hypothesis log-probabilities; the serving
/// `confidence` fields are [0, 1] estimates, so both front-ends report each
/// hypothesis's posterior among the n-best.  With fewer than two hypotheses
/// there is nothing to normalize against — returns `None` and the caller
/// leaves `confidence` at 0.0 (Google's "unset when unavailable").
pub fn score_posteriors(scores: &Option<Vec<f32>>) -> Option<Vec<f32>> {
    let s = scores.as_ref()?;
    if s.len() < 2 {
        return None;
    }
    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = s.iter().map(|&v| (v - m).exp()).collect();
    let z: f32 = exps.iter().sum();
    Some(exps.into_iter().map(|e| e / z).collect())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoding_params_is_empty() {
        assert!(DecodingParams::default().is_empty());
        assert!(!DecodingParams {
            n_best: Some(3),
            ..Default::default()
        }
        .is_empty());
    }

    #[test]
    fn score_posteriors_normalizes() {
        // Equal scores → uniform posterior.
        let p = score_posteriors(&Some(vec![-5.0, -5.0])).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6);
        // Ordering preserved, sums to 1.
        let p = score_posteriors(&Some(vec![-1.0, -2.0, -4.0])).unwrap();
        assert!(p[0] > p[1] && p[1] > p[2]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn score_posteriors_unavailable() {
        assert!(score_posteriors(&None).is_none());
        assert!(score_posteriors(&Some(vec![])).is_none());
        // Single hypothesis: nothing to normalize against.
        assert!(score_posteriors(&Some(vec![-3.0])).is_none());
    }

    #[test]
    fn cmd_decoding_defaults_deserialize() {
        // Old-shape command JSON (no `decoding` key) must still deserialize.
        let j = r#"{"type":"CreateOffline","request_id":"r","sample_rate":16000}"#;
        let cmd: Cmd = serde_json::from_str(j).unwrap();
        match cmd {
            Cmd::CreateOffline {
                priority, decoding, ..
            } => {
                assert_eq!(priority, 0);
                assert!(decoding.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }
}
