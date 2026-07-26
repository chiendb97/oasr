// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Synchronous recognition: `POST /v1/speech:recognize`.
//!
//! The request body is the **raw PCM payload** (no base64, no JSON envelope)
//! and the recognition config rides in the query string
//! (`?encoding=LINEAR16&sample_rate=16000`).  Dropping the base64 + JSON request
//! framing avoids the ~33% wire inflation and a multi-MB JSON parse (~2× the
//! throughput of a base64-JSON body under load).  The response is a small JSON
//! `RecognizeResponse` (transcript + tokens — no base64).

use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use bytes::Bytes;
use oasr_asr::{decode_audio, PcmEncoding};
use oasr_wire::{score_posteriors, DecodingParams, ErrorCode, Event};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, field, info, info_span, warn, Instrument, Span};

use crate::router::{AppState, ServiceMode};

pub const MAX_BODY: usize = 256 * 1024 * 1024;

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
    /// End time (seconds) of the last decoded token; present only for decode
    /// families with token alignments (Paraformer CIF).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_end_time_s: Option<f32>,
    /// Why generation stopped — `"stop"` (EOS) or `"length"` (hit the
    /// generation cap).  Present only for the AR families; without it a
    /// truncated transcript looks exactly like a complete one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
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

/// Map the `encoding` query value to a `(pcm_encoding, content_type_hint)`
/// pair.  Names follow the proto enum spelling so the REST + gRPC surfaces
/// agree.
fn map_encoding(
    s: &str,
) -> Result<(PcmEncoding, Option<&'static str>), (StatusCode, &'static str, String)> {
    match s.to_ascii_uppercase().as_str() {
        "" | "ENCODING_UNSPECIFIED" => Err((
            StatusCode::BAD_REQUEST,
            "INVALID_ARGUMENT",
            "encoding query parameter must be set".into(),
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
    nbest_texts: Option<Vec<String>>,
    max_alts: u32,
) -> Vec<SpeechRecognitionAlternative> {
    let cap = if max_alts == 0 { 1 } else { max_alts as usize };
    let rows = if tokens.is_empty() {
        vec![Vec::new()]
    } else {
        tokens
    };
    let confidences = score_posteriors(&scores);
    rows.into_iter()
        .take(cap)
        .enumerate()
        .map(|(i, ids)| SpeechRecognitionAlternative {
            transcript: if i == 0 {
                text.clone()
            } else {
                nbest_texts
                    .as_ref()
                    .and_then(|ts| ts.get(i).cloned())
                    .unwrap_or_default()
            },
            confidence: confidences
                .as_ref()
                .and_then(|c| c.get(i).copied())
                .unwrap_or(0.0),
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

/// Build a 4xx client-error response and log the reason at DEBUG.  Used for
/// request-validation rejections (bad JSON, unsupported encoding, decode
/// failures, …) which are the caller's fault, not the server's.
fn reject(
    status: StatusCode,
    code: &'static str,
    message: impl Into<String>,
) -> axum::response::Response {
    let message = message.into();
    debug!(status = status.as_u16(), code, reason = %message, "recognize rejected");
    error_response(status, code, message)
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

/// Query parameters for `POST /v1/speech:recognize`.  The request body is the
/// raw PCM audio; all recognition config travels here in the query string.
#[derive(Debug, Deserialize)]
pub struct RawParams {
    /// PCM encoding name: `LINEAR16` (i16-LE), `LINEAR32F` (f32-LE), or `WAV`
    /// (RIFF container in the body).  Same spelling as the proto enum.
    #[serde(default)]
    pub encoding: String,
    /// Sample rate in Hz (default 16000; ignored for `WAV`, which carries its
    /// own header).
    #[serde(default)]
    pub sample_rate: u32,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub max_alternatives: u32,
    /// Per-request DecodingOptions (autoregressive decode families only —
    /// AED / speech-LLM; CTC ignores them).  Mirror the gRPC
    /// `RecognitionConfig` extension fields.
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub prompt: Option<String>,
}

impl RawParams {
    /// Map the query-string knobs to the engine's per-request
    /// [`DecodingParams`]; `Ok(None)` when every knob is at its default, and a
    /// client-facing message for out-of-range values.
    ///
    /// Shares [`DecodingParams::validated`] with the gRPC surface so the two
    /// cannot drift, and so a bad value fails only its own request — the Python
    /// `DecodingOptions` raise would take down the whole coalesced admit batch.
    fn decoding_params(&self) -> Result<Option<DecodingParams>, String> {
        DecodingParams {
            n_best: (self.max_alternatives > 1).then_some(self.max_alternatives),
            max_new_tokens: self.max_new_tokens.filter(|&v| v > 0),
            temperature: self.temperature.filter(|&v| v > 0.0),
            top_k: self.top_k.filter(|&v| v > 0),
            top_p: self.top_p.filter(|&v| v > 0.0),
            prompt: self.prompt.clone().filter(|s| !s.is_empty()),
        }
        .validated()
    }
}

/// `POST /v1/speech:recognize?encoding=LINEAR16&sample_rate=16000`
///
/// Offline unary recognition.  The request **body is the raw PCM payload** (no
/// base64, no JSON envelope); config rides in the query string.  Avoids the
/// ~33% base64 inflation and the JSON parse of a multi-MB body.  The response is
/// a small JSON `RecognizeResponse`.
pub async fn handle_recognize(
    State(s): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<RawParams>,
    body: axum::body::Body,
) -> axum::response::Response {
    // Per-request span; `rid` is recorded once the engine assigns one (after
    // admission), so all downstream events for this request carry it.
    let span = info_span!("http.recognize", rid = field::Empty);
    async move {
        let start = Instant::now();

        if s.service_mode != ServiceMode::Offline {
            return reject(
                StatusCode::BAD_REQUEST,
                "FAILED_PRECONDITION",
                "server is running in streaming mode; use the gRPC StreamingRecognize RPC",
            );
        }

        let bytes = match axum::body::to_bytes(body, MAX_BODY).await {
            Ok(b) => b,
            Err(e) => {
                return reject(
                    StatusCode::BAD_REQUEST,
                    "INVALID_ARGUMENT",
                    format!("body read: {e}"),
                );
            }
        };
        if bytes.is_empty() {
            return reject(
                StatusCode::BAD_REQUEST,
                "INVALID_ARGUMENT",
                "request body (raw PCM audio) is required",
            );
        }

        let (pcm_enc, ct_hint) = match map_encoding(&params.encoding) {
            Ok(p) => p,
            Err((status, code, msg)) => return reject(status, code, msg),
        };
        let sr = if params.sample_rate == 0 {
            16_000
        } else {
            params.sample_rate
        };
        let decoded = match decode_audio(ct_hint, &bytes, pcm_enc, Some(sr)) {
            Ok(d) => d,
            Err(e) => {
                return reject(
                    StatusCode::BAD_REQUEST,
                    "INVALID_ARGUMENT",
                    format!("audio decode: {e}"),
                );
            }
        };

        let audio_buf: Bytes = decoded.samples;
        let decoding = match params.decoding_params() {
            Ok(d) => d,
            Err(msg) => return reject(StatusCode::BAD_REQUEST, "INVALID_ARGUMENT", msg),
        };
        run_offline(
            &s,
            audio_buf,
            decoded.sample_rate,
            params.priority,
            params.max_alternatives,
            decoding,
            start,
        )
        .await
    }
    .instrument(span)
    .await
}

/// Shared offline submit → await → response tail for both the JSON and raw-PCM
/// recognise handlers.  ``audio_buf`` is f32-LE mono samples (already decoded);
/// records ``rid`` on the current span once the engine assigns one.
async fn run_offline(
    s: &AppState,
    audio_buf: Bytes,
    sample_rate: u32,
    priority: i32,
    max_alts: u32,
    decoding: Option<DecodingParams>,
    start: Instant,
) -> axum::response::Response {
    let n_samples = audio_buf.len() / 4;
    let handle = match s
        .pool
        .submit_offline(audio_buf, sample_rate, priority, decoding)
        .await
    {
        Ok(h) => h,
        Err(e) => {
            warn!(%e, "recognize submit rejected");
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "RESOURCE_EXHAUSTED",
                format!("submit failed: {e}"),
            );
        }
    };
    let rid = handle.request_id.clone();
    Span::current().record("rid", rid.as_str());
    let ev = match handle.finish().await {
        Ok(e) => e,
        Err(_) => {
            error!(rid = %rid, "engine channel closed before result");
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL",
                "engine channel closed before result",
            );
        }
    };
    s.pool.release(&rid);

    let elapsed_ms = start.elapsed().as_millis() as u64;
    match ev {
        Event::Final {
            request_id,
            text,
            tokens,
            scores,
            nbest_texts,
            end_time_s,
            finish_reason,
        } => {
            let n_tokens = tokens.first().map_or(0, |t| t.len());
            info!(
                rid = %request_id,
                sample_rate,
                n_samples,
                n_tokens,
                elapsed_ms,
                transcript = %text,
                "recognize ok"
            );
            Json(RecognizeResponse {
                results: vec![SpeechRecognitionResult {
                    alternatives: build_alternatives(text, tokens, scores, nbest_texts, max_alts),
                    channel_tag: 0,
                    language_code: String::new(),
                    result_end_time_s: end_time_s,
                    finish_reason,
                }],
                request_id,
            })
            .into_response()
        }
        Event::Error { code, message, .. } => {
            let (status, code_name) = error_from_event_code(code);
            warn!(
                rid = %rid,
                code = ?code,
                status = status.as_u16(),
                elapsed_ms,
                reason = %message,
                "recognize error"
            );
            error_response(status, code_name, message)
        }
        other => {
            error!(rid = %rid, elapsed_ms, "unexpected non-terminal event for offline request: {other:?}");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL",
                "unexpected event type",
            )
        }
    }
}
