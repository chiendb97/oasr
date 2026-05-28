// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Synchronous recognition: `POST /v1/speech:recognize`.
//!
//! Body is JSON mirroring the gRPC `RecognizeRequest` / Google STT v1
//! transcoding convention.  Audio is carried inline as base64 in
//! `audio.content`.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use bytes::Bytes;
use oasr_asr::{decode_audio, PcmEncoding};
use oasr_wire::{ErrorCode, Event};
use serde::{Deserialize, Serialize};
use tracing::error;

use crate::router::{AppState, ServiceMode};

pub const MAX_BODY: usize = 256 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecognitionConfig {
    #[serde(default)]
    pub encoding: String,
    #[serde(default)]
    pub sample_rate_hertz: u32,
    #[serde(default)]
    pub language_code: String,
    #[serde(default)]
    pub max_alternatives: u32,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub audio_channel_count: u32,
    #[serde(default)]
    pub priority: i32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecognitionAudio {
    /// Base64-encoded audio bytes.
    #[serde(default)]
    pub content: String,
    /// Currently unsupported; presence triggers `UNIMPLEMENTED`.
    #[serde(default)]
    pub uri: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecognizeRequest {
    pub config: RecognitionConfig,
    pub audio: RecognitionAudio,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechRecognitionAlternative {
    pub transcript: String,
    pub confidence: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tokens: Vec<u32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechRecognitionResult {
    pub alternatives: Vec<SpeechRecognitionAlternative>,
    #[serde(skip_serializing_if = "is_zero_i32")]
    pub channel_tag: i32,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub language_code: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecognizeResponse {
    pub results: Vec<SpeechRecognitionResult>,
    pub request_id: String,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    code: u16,
    status: &'static str,
    message: String,
}

fn is_zero_i32(v: &i32) -> bool {
    *v == 0
}

/// Map the JSON encoding string to a `(pcm_encoding, content_type_hint)`
/// pair.  Names follow the proto enum spelling so the REST + gRPC surfaces
/// agree.
fn map_encoding(s: &str) -> Result<(PcmEncoding, Option<&'static str>), (StatusCode, &'static str, String)> {
    match s.to_ascii_uppercase().as_str() {
        "" | "ENCODING_UNSPECIFIED" => Err((
            StatusCode::BAD_REQUEST,
            "INVALID_ARGUMENT",
            "config.encoding must be set".into(),
        )),
        "LINEAR16" => Ok((PcmEncoding::I16Le, None)),
        "LINEAR32F" => Ok((PcmEncoding::F32Le, None)),
        "WAV" => Ok((PcmEncoding::F32Le, Some("audio/wav"))),
        other => Err((
            StatusCode::NOT_IMPLEMENTED,
            "UNIMPLEMENTED",
            format!("encoding {other} is not supported"),
        )),
    }
}

fn build_alternatives(
    text: String,
    tokens: Vec<Vec<u32>>,
    scores: Option<Vec<f32>>,
    max_alts: u32,
) -> Vec<SpeechRecognitionAlternative> {
    let cap = if max_alts == 0 { 1 } else { max_alts as usize };
    let rows = if tokens.is_empty() { vec![Vec::new()] } else { tokens };
    rows.into_iter()
        .take(cap)
        .enumerate()
        .map(|(i, ids)| SpeechRecognitionAlternative {
            transcript: if i == 0 { text.clone() } else { String::new() },
            confidence: scores.as_ref().and_then(|s| s.get(i).copied()).unwrap_or(0.0),
            tokens: ids,
        })
        .collect()
}

fn error_response(
    status: StatusCode,
    code: &'static str,
    message: impl Into<String>,
) -> axum::response::Response {
    let body = ErrorBody {
        error: ErrorDetail {
            code: status.as_u16(),
            status: code,
            message: message.into(),
        },
    };
    (status, Json(body)).into_response()
}

fn error_from_event_code(code: ErrorCode) -> (StatusCode, &'static str) {
    match code {
        ErrorCode::Busy => (StatusCode::SERVICE_UNAVAILABLE, "RESOURCE_EXHAUSTED"),
        ErrorCode::UnknownRequest => (StatusCode::NOT_FOUND, "NOT_FOUND"),
        ErrorCode::InvalidCmd => (StatusCode::BAD_REQUEST, "INVALID_ARGUMENT"),
        ErrorCode::Shutdown | ErrorCode::WorkerLost => {
            (StatusCode::SERVICE_UNAVAILABLE, "UNAVAILABLE")
        }
        ErrorCode::Internal => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL"),
    }
}

pub async fn handle_recognize(
    State(s): State<AppState>,
    body: axum::body::Body,
) -> axum::response::Response {
    if s.service_mode != ServiceMode::Offline {
        return error_response(
            StatusCode::BAD_REQUEST,
            "FAILED_PRECONDITION",
            "server is running in streaming mode; use the gRPC StreamingRecognize RPC",
        );
    }

    let bytes = match axum::body::to_bytes(body, MAX_BODY).await {
        Ok(b) => b,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "INVALID_ARGUMENT",
                format!("body read: {e}"),
            );
        }
    };

    let req: RecognizeRequest = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "INVALID_ARGUMENT",
                format!("invalid request JSON: {e}"),
            );
        }
    };

    if !req.audio.uri.is_empty() {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "UNIMPLEMENTED",
            "audio.uri is not supported",
        );
    }
    if req.audio.content.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "INVALID_ARGUMENT",
            "audio.content (base64) is required",
        );
    }

    let audio_bytes = match STANDARD.decode(req.audio.content.as_bytes()) {
        Ok(b) => b,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "INVALID_ARGUMENT",
                format!("audio.content base64 decode: {e}"),
            );
        }
    };

    let (pcm_enc, ct_hint) = match map_encoding(&req.config.encoding) {
        Ok(p) => p,
        Err((status, code, msg)) => return error_response(status, code, msg),
    };
    let sr = if req.config.sample_rate_hertz == 0 {
        16_000
    } else {
        req.config.sample_rate_hertz
    };
    let decoded = match decode_audio(ct_hint, &audio_bytes, pcm_enc, Some(sr)) {
        Ok(d) => d,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "INVALID_ARGUMENT",
                format!("audio decode: {e}"),
            );
        }
    };

    let max_alts = req.config.max_alternatives;
    let priority = req.config.priority;
    let audio_buf: Bytes = decoded.samples;

    let handle = match s
        .pool
        .submit_offline(audio_buf, decoded.sample_rate, priority)
        .await
    {
        Ok(h) => h,
        Err(e) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "RESOURCE_EXHAUSTED",
                format!("submit failed: {e}"),
            );
        }
    };
    let rid = handle.request_id.clone();
    let ev = match handle.finish().await {
        Ok(e) => e,
        Err(_) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL",
                "engine channel closed before result",
            );
        }
    };
    s.pool.release(&rid);

    match ev {
        Event::Final {
            request_id,
            text,
            tokens,
            scores,
        } => Json(RecognizeResponse {
            results: vec![SpeechRecognitionResult {
                alternatives: build_alternatives(text, tokens, scores, max_alts),
                channel_tag: 0,
                language_code: String::new(),
            }],
            request_id,
        })
        .into_response(),
        Event::Error { code, message, .. } => {
            let (status, code_name) = error_from_event_code(code);
            error_response(status, code_name, message)
        }
        other => {
            error!("unexpected non-terminal event for offline rid: {other:?}");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL",
                "unexpected event type",
            )
        }
    }
}
