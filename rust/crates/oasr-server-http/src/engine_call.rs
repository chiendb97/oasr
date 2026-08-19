// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Submit → await → classify, once, for every HTTP shape.
//!
//! The Google-shaped route, the two OpenAI-shaped routes and the WebSocket all
//! do the same three things to the engine and differ only in how they *render*
//! the result.  Keeping the middle step here is what lets a new response shape
//! be a serializer rather than another copy of the orchestration — the failure
//! classification in particular, which is where two copies quietly disagree
//! about whether a busy engine is a 429 or a 503.

use std::time::Instant;

use axum::http::StatusCode;
use bytes::Bytes;
use oasr_metrics::{f32_pcm_seconds, Outcome, RequestRecorder};
use oasr_wire::{DecodingParams, ErrorCode, Event, WordTiming};
use tracing::{error, warn};

use crate::router::AppState;

/// The terminal payload of a completed offline request.
#[derive(Debug, Clone)]
pub struct FinalTranscript {
    pub request_id: String,
    pub text: String,
    pub tokens: Vec<Vec<u32>>,
    pub scores: Option<Vec<f32>>,
    pub nbest_texts: Option<Vec<String>>,
    /// End time (s) of the last decoded token, for families with alignments.
    pub end_time_s: Option<f32>,
    /// Per-word timings for the best hypothesis, when the request asked.
    pub words: Option<Vec<WordTiming>>,
    /// Mean per-token posterior of the best hypothesis, in [0, 1].
    pub confidence: Option<f32>,
    /// `"stop"` / `"length"` for the autoregressive families.
    pub finish_reason: Option<String>,
    /// Wall time from the handler's start to the terminal event.
    pub elapsed_ms: u64,
}

/// A request that did not produce a transcript, already classified.
#[derive(Debug, Clone)]
pub struct EngineFailure {
    pub status: StatusCode,
    /// Canonical status name (`RESOURCE_EXHAUSTED`, `INTERNAL`, …).
    pub code: &'static str,
    pub message: String,
}

impl EngineFailure {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }
}

/// Map an engine error code to the HTTP status + canonical name it reports as.
pub fn error_from_event_code(code: ErrorCode) -> (StatusCode, &'static str) {
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

/// Submit one fully-buffered utterance and await its terminal event.
///
/// `audio` is f32-LE mono at the engine rate. Releases the pool slot on every
/// path; `metrics` identifies the calling API surface.
pub async fn submit_offline_and_wait(
    state: &AppState,
    audio: Bytes,
    sample_rate: u32,
    priority: i32,
    decoding: Option<DecodingParams>,
    start: Instant,
    metrics: &RequestRecorder,
) -> Result<FinalTranscript, EngineFailure> {
    // Counted before the submit, so audio a saturated engine refuses still
    // lands in the denominator: RTFx computed only over accepted work makes a
    // server look most efficient exactly when it is shedding load.
    let audio_seconds = f32_pcm_seconds(audio.len(), sample_rate);
    metrics.audio_ingested(audio_seconds);
    metrics.audio_duration(audio_seconds);

    let handle = match state
        .pool
        .submit_offline(audio, sample_rate, priority, decoding)
        .await
    {
        Ok(h) => h,
        Err(e) => {
            warn!(%e, "offline submit rejected");
            metrics.finished(Outcome::Error, start.elapsed());
            return Err(EngineFailure::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "RESOURCE_EXHAUSTED",
                format!("submit failed: {e}"),
            ));
        }
    };
    let rid = handle.request_id.clone();
    let ev = match handle.finish().await {
        Ok(e) => e,
        Err(_) => {
            error!(rid = %rid, "engine channel closed before result");
            metrics.finished(Outcome::Error, start.elapsed());
            return Err(EngineFailure::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL",
                "engine channel closed before result",
            ));
        }
    };
    state.pool.release(&rid);

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_millis() as u64;
    match ev {
        Event::Final {
            request_id,
            text,
            tokens,
            scores,
            nbest_texts,
            end_time_s,
            words,
            confidence,
            finish_reason,
        } => Ok(FinalTranscript {
            request_id,
            text,
            tokens,
            scores,
            nbest_texts,
            end_time_s,
            words,
            confidence,
            finish_reason,
            elapsed_ms,
        })
        .inspect(|_| metrics.finished(Outcome::Ok, elapsed)),
        Event::Error { code, message, .. } => {
            let (status, code_name) = error_from_event_code(code);
            // A shutdown-coded terminal is the engine draining out from under
            // an in-flight request, which is a cancellation from the caller's
            // point of view rather than a request that failed on its merits.
            // Splitting them here is what keeps an ordinary rolling restart
            // out of the error-rate panel.
            let kind = match code {
                ErrorCode::Shutdown => Outcome::Cancelled,
                _ => Outcome::Error,
            };
            metrics.finished(kind, elapsed);
            warn!(
                rid = %rid,
                code = ?code,
                status = status.as_u16(),
                elapsed_ms,
                reason = %message,
                "recognize error"
            );
            Err(EngineFailure::new(status, code_name, message))
        }
        other => {
            metrics.finished(Outcome::Error, elapsed);
            error!(rid = %rid, elapsed_ms, "unexpected non-terminal event for offline request: {other:?}");
            Err(EngineFailure::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL",
                "unexpected event type",
            ))
        }
    }
}
