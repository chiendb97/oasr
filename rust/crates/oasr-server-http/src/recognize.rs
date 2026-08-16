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
use oasr_asr::{decode_audio, parse_encoding, DecodeOptions, EncodingError, SourceEncoding};
use oasr_wire::{normalize_language, score_posteriors, DecodingParams, WordTiming};
use serde::{Deserialize, Serialize};
use tracing::{debug, field, info, info_span, Instrument, Span};

use crate::engine_call::submit_offline_and_wait;
use crate::http_metrics::google_offline;
use crate::router::{AppState, ServiceMode};

/// Fallback body cap, used only when a caller builds a [`ServerState`] without
/// one.  The served value comes from `--max-audio-mib` via
/// [`crate::ServerState::max_body_bytes`], which the gRPC surface reads too.
pub const MAX_BODY: usize = 256 * 1024 * 1024;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechRecognitionAlternative {
    pub transcript: String,
    pub confidence: f32,
    /// Per-word timings, on the **top** alternative only and only when the
    /// request set `enable_word_time_offsets` — the engine aligns the
    /// hypothesis it returns as `transcript`, not the whole beam.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<WordInfo>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tokens: Vec<u32>,
}

/// One word of a transcript, in Google's `WordInfo` shape.  Seconds rather
/// than the proto `Duration` message, matching how the rest of this surface
/// reports time (`resultEndTimeS`).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WordInfo {
    pub word: String,
    pub start_time_s: f32,
    pub end_time_s: f32,
    pub confidence: f32,
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

/// Map the `encoding` query value to a `(source_encoding, container_hint)`
/// pair.  Names follow the proto enum spelling — and the mapping itself is
/// `oasr_asr::parse_encoding`, shared with the gRPC surface, so the two cannot
/// disagree about which codecs exist.
fn map_encoding(
    s: &str,
) -> Result<(SourceEncoding, Option<&'static str>), (StatusCode, &'static str, String)> {
    parse_encoding(s).map_err(|e| match e {
        EncodingError::Unspecified => (
            StatusCode::BAD_REQUEST,
            "INVALID_ARGUMENT",
            "encoding query parameter must be set".into(),
        ),
        other => (
            StatusCode::NOT_IMPLEMENTED,
            "UNIMPLEMENTED",
            other.to_string(),
        ),
    })
}

fn build_alternatives(
    text: String,
    tokens: Vec<Vec<u32>>,
    scores: Option<Vec<f32>>,
    nbest_texts: Option<Vec<String>>,
    max_alts: u32,
    words: Option<Vec<WordTiming>>,
) -> Vec<SpeechRecognitionAlternative> {
    let cap = if max_alts == 0 { 1 } else { max_alts as usize };
    let rows = if tokens.is_empty() {
        vec![Vec::new()]
    } else {
        tokens
    };
    let confidences = score_posteriors(&scores);
    let mut words = words;
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
            words: if i == 0 {
                words.take().map(|w| w.into_iter().map(word_info).collect())
            } else {
                None
            }
            .unwrap_or_default(),
            tokens: ids,
        })
        .collect()
}

fn word_info(w: WordTiming) -> WordInfo {
    WordInfo {
        word: w.word,
        start_time_s: w.start,
        end_time_s: w.end,
        confidence: w.confidence,
    }
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

/// Query parameters for `POST /v1/speech:recognize`.  The request body is the
/// raw PCM audio; all recognition config travels here in the query string.
#[derive(Debug, Deserialize)]
pub struct RawParams {
    /// PCM encoding name: `LINEAR16` (i16-LE), `LINEAR32F` (f32-LE), or `WAV`
    /// (RIFF container in the body).  Same spelling as the proto enum.
    #[serde(default)]
    pub encoding: String,
    /// Sample rate in Hz of the body's PCM (ignored for `WAV`, which carries
    /// its own header).  Unset means the model's own rate.  Anything else is
    /// resampled to the model's rate server-side; rates outside
    /// `[4000, 384000]` are rejected.
    #[serde(default)]
    pub sample_rate: u32,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub max_alternatives: u32,
    /// Per-word start/end times and confidences on the top alternative
    /// (Google's own field name).  A decode family that cannot align in this
    /// request's mode rejects the request rather than answering without it.
    #[serde(default)]
    pub enable_word_time_offsets: bool,
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
    /// `transcribe` (default) or `translate` — Whisper's task token.
    #[serde(default)]
    pub task: Option<String>,
    /// BCP-47 or ISO-639 tag for the families with language control; reduced to
    /// its primary subtag server-side.
    #[serde(default)]
    pub language: Option<String>,
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
            task: normalize_task(self.task.as_deref()),
            language: normalize_optional_language(self.language.as_deref())?,
            word_timestamps: self.enable_word_time_offsets.then_some(true),
        }
        .validated()
    }
}

/// `Some(lowercased)` for a set task; `None` when unset.  Validation of the
/// *value* is `DecodingParams::validate`'s job, in one place for both surfaces.
pub(crate) fn normalize_task(task: Option<&str>) -> Option<String> {
    task.map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
}

/// Reduce a client's language tag to a primary subtag, rejecting junk.
///
/// Dropping an unparseable tag would transcribe in the checkpoint's own
/// language while the client believes it asked for another — the response
/// carries nothing that would reveal it.
pub(crate) fn normalize_optional_language(lang: Option<&str>) -> Result<Option<String>, String> {
    match lang.map(str::trim).filter(|s| !s.is_empty()) {
        None => Ok(None),
        Some(tag) => normalize_language(tag)
            .map(Some)
            .ok_or_else(|| format!("language {tag:?} is not a language tag")),
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

        let bytes = match axum::body::to_bytes(body, s.max_body_bytes).await {
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

        let (source_enc, ct_hint) = match map_encoding(&params.encoding) {
            Ok(p) => p,
            Err((status, code, msg)) => return reject(status, code, msg),
        };
        // An unset `sample_rate` means "the model's own rate" — the only value
        // that could ever have worked before resampling existed.
        let sr = if params.sample_rate == 0 {
            s.sample_rate
        } else {
            params.sample_rate
        };
        // Resample to the engine's rate here, in the front-end: the engine
        // derives every frame count from its own `FeatureConfig.sample_rate` and
        // ignores the request's, so 8 kHz telephony or 44.1 kHz media reaching it
        // unconverted produced a confident, wrong transcript.  A WAV body carries
        // its own rate in the header, which is what `decode_audio` converts from.
        let decoded = match decode_audio(
            &bytes,
            &DecodeOptions {
                hint: ct_hint,
                encoding: source_enc,
                source_sample_rate: Some(sr),
                target_sample_rate: Some(s.sample_rate),
                max_samples: s.max_audio_samples,
            },
        ) {
            Ok(d) => d,
            Err(e) => {
                return reject(
                    StatusCode::BAD_REQUEST,
                    "INVALID_ARGUMENT",
                    format!("audio decode: {e}"),
                );
            }
        };

        if decoded.sample_rate != sr {
            debug!(
                from_hz = sr,
                to_hz = decoded.sample_rate,
                "resampled request audio to the model rate"
            );
        }

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

/// Offline submit → await → Google-shaped response.  ``audio_buf`` is f32-LE
/// mono samples (already decoded); records ``rid`` on the current span once the
/// engine assigns one.
///
/// The submit/await half is [`submit_offline_and_wait`], shared with the
/// OpenAI-shaped routes; only the rendering below is specific to this surface.
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
    let final_ = match submit_offline_and_wait(
        s,
        audio_buf,
        sample_rate,
        priority,
        decoding,
        start,
        google_offline(),
    )
    .await
    {
        Ok(f) => f,
        Err(e) => return error_response(e.status, e.code, e.message),
    };
    Span::current().record("rid", final_.request_id.as_str());
    let n_tokens = final_.tokens.first().map_or(0, |t| t.len());
    info!(
        rid = %final_.request_id,
        sample_rate,
        n_samples,
        n_tokens,
        elapsed_ms = final_.elapsed_ms,
        transcript = %final_.text,
        "recognize ok"
    );
    Json(RecognizeResponse {
        results: vec![SpeechRecognitionResult {
            alternatives: build_alternatives(
                final_.text,
                final_.tokens,
                final_.scores,
                final_.nbest_texts,
                max_alts,
                final_.words,
            ),
            channel_tag: 0,
            language_code: String::new(),
            result_end_time_s: final_.end_time_s,
            finish_reason: final_.finish_reason,
        }],
        request_id: final_.request_id,
    })
    .into_response()
}
