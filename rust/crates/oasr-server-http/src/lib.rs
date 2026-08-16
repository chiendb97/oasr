// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! axum routes for the OASR HTTP API.
//!
//! Two request shapes over one engine:
//!
//! * the **Google STT v1**-shaped surface (`POST /v1/speech:recognize`) — raw
//!   PCM body, config in the query string, lowest overhead;
//! * the **OpenAI**-shaped surface (`POST /v1/audio/transcriptions`,
//!   `/v1/audio/translations`, `GET /v1/realtime`) — multipart uploads and a
//!   WebSocket, which is what every ASR client library and LLM app framework
//!   already speaks.
//!
//! Both are thin adapters over the same `EnginePool` handles; neither is
//! privileged.

pub mod engine_call;
pub mod http_metrics;
pub mod openai;
pub mod realtime;
pub mod recognize;
pub mod router;

pub use router::{build_router, AppState, RouterLimits, ServiceMode, READY_STALE_AFTER};

use std::sync::Arc;

use metrics_exporter_prometheus::PrometheusHandle;
use oasr_engine_client::EnginePool;

/// Shared state injected into every handler.
pub struct ServerState {
    pub pool: Arc<EnginePool>,
    pub prometheus: Option<PrometheusHandle>,
    pub service_mode: ServiceMode,
    /// The engine's waveform sample rate in Hz.  Handlers resample client audio
    /// to this before submitting — the engine ignores a request's declared rate
    /// and computes every frame count from its own.
    pub sample_rate: u32,
    /// Largest accepted request body, in bytes.  Shared with the gRPC
    /// `max_decoding_message_size` so the two surfaces accept the same audio.
    pub max_body_bytes: usize,
    /// Ceiling on the **decoded** waveform, in samples at the model's rate.
    /// `None` disables.  Separate from `max_body_bytes` because a compressed
    /// container breaks the relationship between the two: a few MiB of MP3 is
    /// hours of audio, and the allocation happens before anything could notice.
    pub max_audio_samples: Option<usize>,
    /// Names this process answers to on the OpenAI surface's `model` field.
    /// Empty means "accept anything" — the single-model default, which keeps a
    /// client pointed at `whisper-1` working after nothing but a base-URL
    /// change.  Non-empty makes an unknown name a 404, as OpenAI does.
    pub served_model_names: Vec<String>,
    /// The id reported by `GET /v1/models` and echoed in responses.
    pub model_id: String,
}

impl ServerState {
    /// Whether this process will answer to `name` (empty = unspecified).
    pub fn serves_model(&self, name: &str) -> bool {
        let name = name.trim();
        if name.is_empty() || self.served_model_names.is_empty() {
            return true;
        }
        self.served_model_names.iter().any(|m| m == name) || name == self.model_id
    }
}
