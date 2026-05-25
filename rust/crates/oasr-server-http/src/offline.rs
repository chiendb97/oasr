// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Offline transcription: `POST /v1/transcriptions` (raw body).
//!
//! Accepts a raw body whose content-type selects the decoder:
//! - `audio/wav` or unset + RIFF magic → WAV decode
//! - `application/octet-stream` + `?sample_rate=<N>[&encoding=f32_le|i16_le]`
//!   → raw PCM decode
//!
//! Multipart uploads belong to the Whisper-compat route in `whisper.rs`.

use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json};
use bytes::Bytes;
use oasr_asr::{decode_audio, AudioError, PcmEncoding};
use oasr_wire::Event;
use serde::Deserialize;
use serde_json::json;
use tracing::error;

use crate::router::AppState;

pub const MAX_BODY: usize = 256 * 1024 * 1024;

#[derive(Debug, Deserialize)]
pub struct OfflineQuery {
    pub sample_rate: Option<u32>,
    pub encoding: Option<String>,
    pub priority: Option<i32>,
}

pub fn parse_encoding(s: Option<&str>) -> PcmEncoding {
    match s.unwrap_or("").to_ascii_lowercase().as_str() {
        "i16_le" | "linear16_i16" => PcmEncoding::I16Le,
        _ => PcmEncoding::F32Le,
    }
}

pub async fn handle_offline(
    State(s): State<AppState>,
    Query(q): Query<OfflineQuery>,
    headers: HeaderMap,
    body: axum::body::Body,
) -> axum::response::Response {
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    let bytes = match axum::body::to_bytes(body, MAX_BODY).await {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("body read: {e}")).into_response();
        }
    };

    let decoded = match decode_audio(
        content_type.as_deref(),
        &bytes,
        parse_encoding(q.encoding.as_deref()),
        q.sample_rate,
    ) {
        Ok(d) => d,
        Err(e) => return audio_error(e),
    };
    submit_offline_response(s, decoded.samples, decoded.sample_rate, q.priority.unwrap_or(0))
        .await
}

pub fn audio_error(e: AudioError) -> axum::response::Response {
    (StatusCode::BAD_REQUEST, format!("audio decode: {e}")).into_response()
}

pub async fn submit_offline_response(
    state: AppState,
    audio: Bytes,
    sample_rate: u32,
    priority: i32,
) -> axum::response::Response {
    let handle = match state
        .pool
        .submit_offline(audio, sample_rate, priority)
        .await
    {
        Ok(h) => h,
        Err(e) => {
            return (StatusCode::SERVICE_UNAVAILABLE, format!("submit failed: {e}"))
                .into_response();
        }
    };
    let rid = handle.request_id.clone();
    let ev = match handle.finish().await {
        Ok(e) => e,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "engine channel closed before result".to_owned(),
            )
                .into_response();
        }
    };
    state.pool.release(&rid);

    match ev {
        Event::Final {
            request_id,
            text,
            tokens,
            scores,
        } => Json(json!({
            "request_id": request_id,
            "text": text,
            "tokens": tokens,
            "scores": scores,
        }))
        .into_response(),
        Event::Error {
            request_id,
            code,
            message,
        } => {
            let status = match code {
                oasr_wire::ErrorCode::Busy => StatusCode::SERVICE_UNAVAILABLE,
                oasr_wire::ErrorCode::UnknownRequest => StatusCode::NOT_FOUND,
                oasr_wire::ErrorCode::InvalidCmd => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (
                status,
                Json(json!({
                    "request_id": request_id,
                    "code": format!("{code:?}"),
                    "message": message,
                })),
            )
                .into_response()
        }
        other => {
            error!("unexpected non-terminal event for offline rid: {other:?}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "unexpected event type".to_owned(),
            )
                .into_response()
        }
    }
}
