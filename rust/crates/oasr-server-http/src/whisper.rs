// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! OpenAI Whisper-compatible offline endpoint.
//!
//! `POST /v1/audio/transcriptions` accepts multipart/form-data with a `file`
//! (or `audio`) part and an optional `response_format` field.  `language`,
//! `prompt`, `temperature`, `timestamp_granularities` are accepted and
//! ignored in v1.

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use bytes::Bytes;
use oasr_asr::{decode_audio, PcmEncoding};
use oasr_wire::Event;
use serde_json::json;
use tracing::error;

use crate::router::AppState;

pub async fn handle_whisper(
    State(s): State<AppState>,
    mut multipart: Multipart,
) -> axum::response::Response {
    let mut audio_part: Option<(Bytes, Option<String>)> = None;
    let mut response_format = "json".to_owned();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_owned();
        let ct = field.content_type().map(str::to_owned);
        match name.as_str() {
            "file" | "audio" => {
                let data = match field.bytes().await {
                    Ok(b) => b,
                    Err(e) => {
                        return (StatusCode::BAD_REQUEST, format!("part read: {e}"))
                            .into_response();
                    }
                };
                audio_part = Some((data, ct));
            }
            "response_format" => {
                if let Ok(v) = field.text().await {
                    response_format = v.trim().to_lowercase();
                }
            }
            // accepted-and-ignored Whisper params
            "language" | "prompt" | "temperature" | "timestamp_granularities"
            | "model" => {
                let _ = field.text().await;
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let Some((bytes, ct)) = audio_part else {
        return (
            StatusCode::BAD_REQUEST,
            "missing 'file' part in multipart body",
        )
            .into_response();
    };

    let decoded = match decode_audio(ct.as_deref(), &bytes, PcmEncoding::F32Le, Some(16000)) {
        Ok(d) => d,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("audio decode: {e}")).into_response(),
    };

    let handle = match s
        .pool
        .submit_offline(decoded.samples, decoded.sample_rate, 0)
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
                "engine channel closed before result",
            )
                .into_response();
        }
    };
    s.pool.release(&rid);

    match ev {
        Event::Final { text, .. } => match response_format.as_str() {
            "text" => (StatusCode::OK, text).into_response(),
            "verbose_json" => Json(json!({
                "text": text,
                "language": "",
                "duration": 0.0,
                "segments": [],
            }))
            .into_response(),
            _ => Json(json!({ "text": text })).into_response(),
        },
        Event::Error { code, message, .. } => {
            error!(?code, %message, "whisper transcription failed");
            let status = match code {
                oasr_wire::ErrorCode::Busy => StatusCode::SERVICE_UNAVAILABLE,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (status, Json(json!({ "error": { "message": message } }))).into_response()
        }
        other => {
            error!("unexpected non-terminal event from offline: {other:?}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "unexpected event type",
            )
                .into_response()
        }
    }
}
