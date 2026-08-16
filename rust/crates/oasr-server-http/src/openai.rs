// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! OpenAI-compatible audio endpoints.
//!
//! * `POST /v1/audio/transcriptions` — multipart upload, transcribe.
//! * `POST /v1/audio/translations` — the same, with `task=translate`.
//!
//! These exist for one reason: every ASR client library, LLM app framework and
//! "swap the endpoint" migration already speaks this shape.  Serving it turns
//! adoption from "rewrite your client" into "change one base URL".  The
//! Google-shaped `POST /v1/speech:recognize` is unchanged and still the
//! lowest-overhead path (raw body, no multipart framing); this is the one
//! people will actually point their existing code at.
//!
//! Two places where OASR is deliberately *not* silent about a difference:
//!
//! * `timestamp_granularities[]=word` is honoured by the decode families that
//!   can align, and **rejected at admission** by the ones that cannot (the WFST
//!   decoder, and any family running in streaming mode without a
//!   frame-synchronous emission). A `verbose_json` response missing the array a
//!   client explicitly asked for reads as "this audio had no words", which is a
//!   worse answer than an error naming the gap.
//! * `segments` carries one segment spanning the utterance, with only the
//!   fields OASR can actually compute. `avg_logprob` / `no_speech_prob` /
//!   `compression_ratio` are omitted rather than filled with plausible numbers.

use std::time::Instant;

use axum::extract::{Multipart, State};
use axum::http::{header, StatusCode};
use axum::response::sse::Event as SseEvent;
use axum::response::{IntoResponse, Json, Response, Sse};
use bytes::Bytes;
use oasr_asr::{decode_audio, AudioError, DecodeOptions, SourceEncoding};
use oasr_wire::{DecodingParams, Event};
use serde::Serialize;
use serde_json::json;
use tokio_stream::StreamExt;
use tracing::{debug, field, info, info_span, warn, Instrument, Span};

use crate::engine_call::submit_offline_and_wait;
use crate::http_metrics::openai_offline;
use crate::recognize::{normalize_optional_language, normalize_task};
use crate::router::{AppState, ServiceMode};

/// `POST /v1/audio/transcriptions`
pub async fn handle_transcriptions(state: State<AppState>, form: Multipart) -> Response {
    handle_audio(state, form, Endpoint::Transcriptions).await
}

/// `POST /v1/audio/translations`
pub async fn handle_translations(state: State<AppState>, form: Multipart) -> Response {
    handle_audio(state, form, Endpoint::Translations).await
}

/// Which of the two routes is being served.  They differ only in the task they
/// force and in the `task` field of a `verbose_json` response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endpoint {
    Transcriptions,
    Translations,
}

impl Endpoint {
    /// The task to *send*, or `None` to leave the checkpoint's own.
    ///
    /// `/v1/audio/transcriptions` sends nothing. Transcription is what every
    /// decode family does by default, so asserting `task=transcribe` would make
    /// a plain upload carry an option that a family without a task control has
    /// to reject — turning "transcribe this file" into a 400 on a CTC engine.
    /// The absence of a `task` field in the request means "the default", not
    /// "explicitly transcribe".
    ///
    /// `/v1/audio/translations` is the opposite: translation *is* the request,
    /// so a family that cannot do it must say so rather than transcribe and
    /// call the result a translation.
    fn task(self) -> Option<&'static str> {
        match self {
            Endpoint::Transcriptions => None,
            Endpoint::Translations => Some("translate"),
        }
    }

    /// The `task` field of a `verbose_json` response — always concrete, since
    /// this reports what happened rather than what was asked for.
    fn reported_task(self) -> &'static str {
        match self {
            Endpoint::Transcriptions => "transcribe",
            Endpoint::Translations => "translate",
        }
    }
}

// ---------------------------------------------------------------------------
// Errors — OpenAI's envelope, not the Google one
// ---------------------------------------------------------------------------

/// OpenAI's error shape.  Clients parse `error.message` and branch on
/// `error.type`, so these routes speak it even though the rest of the server
/// uses the Google envelope.
fn api_error(
    status: StatusCode,
    kind: &str,
    param: Option<&str>,
    message: impl Into<String>,
) -> Response {
    let message = message.into();
    debug!(status = status.as_u16(), kind, param, reason = %message, "openai request rejected");
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": kind,
                "param": param,
                "code": serde_json::Value::Null,
            }
        })),
    )
        .into_response()
}

fn invalid_request(param: Option<&str>, message: impl Into<String>) -> Response {
    api_error(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        param,
        message,
    )
}

// ---------------------------------------------------------------------------
// Request form
// ---------------------------------------------------------------------------

/// `response_format` — OpenAI's five.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ResponseFormat {
    #[default]
    Json,
    Text,
    Srt,
    Vtt,
    VerboseJson,
}

impl ResponseFormat {
    fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "json" => ResponseFormat::Json,
            "text" => ResponseFormat::Text,
            "srt" => ResponseFormat::Srt,
            "vtt" => ResponseFormat::Vtt,
            "verbose_json" => ResponseFormat::VerboseJson,
            _ => return None,
        })
    }

    /// Whether this format can be produced incrementally.  The subtitle and
    /// verbose formats cannot: both need the utterance's total duration, which
    /// does not exist until the last token.
    fn streamable(self) -> bool {
        matches!(self, ResponseFormat::Json | ResponseFormat::Text)
    }
}

/// The parsed multipart form.
struct AudioForm {
    file: Bytes,
    /// Filename and part content-type, used as container hints.  Both are
    /// advisory: the body's own magic bytes decide (clients mislabel uploads
    /// constantly, and `curl -F` sends `application/octet-stream`).
    filename: Option<String>,
    content_type: Option<String>,
    model: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
    response_format: ResponseFormat,
    temperature: Option<f32>,
    granularities: Vec<String>,
    stream: bool,
}

/// Read the multipart body into an [`AudioForm`].
///
/// Unknown fields are ignored: OpenAI's clients send a moving set of them, and
/// a 400 on a field that does not change the transcript would break callers for
/// no benefit.  Fields that *would* change the answer are all named here.
async fn read_form(mut form: Multipart) -> Result<AudioForm, Response> {
    let mut out = AudioForm {
        file: Bytes::new(),
        filename: None,
        content_type: None,
        model: None,
        language: None,
        prompt: None,
        response_format: ResponseFormat::default(),
        temperature: None,
        granularities: Vec::new(),
        stream: false,
    };
    loop {
        let field = match form.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                return Err(invalid_request(
                    None,
                    format!("could not read the multipart body: {e}"),
                ))
            }
        };
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                out.filename = field.file_name().map(str::to_owned);
                out.content_type = field.content_type().map(str::to_owned);
                out.file = field.bytes().await.map_err(|e| {
                    // Hitting the body limit lands here; say which limit, since
                    // "multipart error" is unactionable.
                    invalid_request(
                        Some("file"),
                        format!("could not read the uploaded file: {e}"),
                    )
                })?;
            }
            "timestamp_granularities" | "timestamp_granularities[]" => {
                if let Ok(v) = field.text().await {
                    out.granularities.push(v.trim().to_ascii_lowercase());
                }
            }
            "model" | "language" | "prompt" | "response_format" | "temperature" | "stream" => {
                let value = match field.text().await {
                    Ok(v) => v,
                    Err(e) => {
                        return Err(invalid_request(
                            Some(&name),
                            format!("could not read field: {e}"),
                        ))
                    }
                };
                match name.as_str() {
                    "model" => out.model = Some(value),
                    "language" => out.language = Some(value),
                    "prompt" => out.prompt = Some(value),
                    "response_format" => {
                        out.response_format = ResponseFormat::parse(&value).ok_or_else(|| {
                            invalid_request(
                                Some("response_format"),
                                format!(
                                    "unknown response_format {value:?}; expected one of \
                                     json, text, srt, vtt, verbose_json"
                                ),
                            )
                        })?
                    }
                    "temperature" => {
                        out.temperature = Some(value.trim().parse::<f32>().map_err(|_| {
                            invalid_request(
                                Some("temperature"),
                                format!("temperature must be a number, got {value:?}"),
                            )
                        })?)
                    }
                    "stream" => out.stream = matches!(value.trim(), "true" | "1" | "yes"),
                    _ => unreachable!("outer match limits the names"),
                }
            }
            // Everything else (`chunking_strategy`, `include[]`, …) is read and
            // dropped so the body is fully consumed.
            _ => {
                let _ = field.bytes().await;
            }
        }
    }
    if out.file.is_empty() {
        return Err(invalid_request(
            Some("file"),
            "a `file` part carrying the audio is required",
        ));
    }
    Ok(out)
}

impl AudioForm {
    /// Container hint from the part's content-type, falling back to the
    /// filename extension.
    fn hint(&self) -> Option<&str> {
        if let Some(ct) = self.content_type.as_deref() {
            // `application/octet-stream` names no container; skip it so the
            // sniffer gets the decision rather than a useless hint.
            if oasr_asr::container_from_hint(ct).is_some() {
                return Some(ct);
            }
        }
        self.filename
            .as_deref()
            .and_then(|f| f.rsplit_once('.'))
            .map(|(_, ext)| ext)
            .filter(|ext| oasr_asr::container_from_hint(ext).is_some())
    }

    /// Whether `timestamp_granularities[]` asked for word-level times.
    ///
    /// OpenAI only honours the parameter with `response_format=verbose_json`,
    /// and so does this: the other formats have nowhere to put the array, and
    /// paying for an alignment whose result is then dropped is worse than
    /// ignoring a parameter that could not have been rendered anyway.
    fn wants_words(&self) -> bool {
        matches!(self.response_format, ResponseFormat::VerboseJson)
            && self.granularities.iter().any(|g| g == "word")
    }

    /// Map the form's knobs to the engine's per-request decoding options.
    fn decoding_params(&self, endpoint: Endpoint) -> Result<Option<DecodingParams>, String> {
        DecodingParams {
            n_best: None,
            max_new_tokens: None,
            // OpenAI's temperature 0 means greedy, same as the engine's.
            temperature: self.temperature.filter(|&v| v > 0.0),
            top_k: None,
            top_p: None,
            prompt: self.prompt.clone().filter(|s| !s.is_empty()),
            // `/v1/audio/translations` *is* the task; there is no form field.
            task: normalize_task(endpoint.task()),
            language: normalize_optional_language(self.language.as_deref())?,
            word_timestamps: self.wants_words().then_some(true),
        }
        .validated()
    }
}

// ---------------------------------------------------------------------------
// Response bodies
// ---------------------------------------------------------------------------

/// One segment of a `verbose_json` response.
///
/// OASR emits exactly one, spanning the utterance: no decode family produces
/// segment boundaries today. The fields it cannot compute are **omitted**
/// rather than defaulted — a `no_speech_prob` of 0.0 that was never measured is
/// a number a client will act on.
#[derive(Debug, Serialize)]
struct VerboseSegment {
    id: u32,
    seek: u32,
    start: f32,
    end: f32,
    text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tokens: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

/// One word of a `verbose_json` response — OpenAI's shape exactly.
#[derive(Debug, Serialize)]
struct VerboseWord {
    word: String,
    start: f32,
    end: f32,
    /// OASR extension: the mean per-token posterior over the word's tokens.
    /// OpenAI does not return a per-word confidence, and a client that does not
    /// know about it ignores the field.
    confidence: f32,
}

#[derive(Debug, Serialize)]
struct VerboseResponse {
    task: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    duration: f32,
    text: String,
    segments: Vec<VerboseSegment>,
    /// Present only when `timestamp_granularities[]=word` was asked for —
    /// absent and empty mean different things, so the field is omitted rather
    /// than serialised as `[]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    words: Option<Vec<VerboseWord>>,
    /// OASR extensions: the server-assigned id (for correlating with logs and
    /// `oasr_requests_*` metrics) and the generation stop reason.
    request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

/// `hh:mm:ss,mmm` (SRT) or `hh:mm:ss.mmm` (WebVTT).
fn timecode(seconds: f32, millis_sep: char) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let (ms, s) = (total_ms % 1000, total_ms / 1000);
    let (secs, mins, hours) = (s % 60, (s / 60) % 60, s / 3600);
    format!("{hours:02}:{mins:02}:{secs:02}{millis_sep}{ms:03}")
}

fn srt_body(text: &str, end: f32) -> String {
    format!(
        "1\n{} --> {}\n{}\n\n",
        timecode(0.0, ','),
        timecode(end, ','),
        text
    )
}

fn vtt_body(text: &str, end: f32) -> String {
    format!(
        "WEBVTT\n\n{} --> {}\n{}\n\n",
        timecode(0.0, '.'),
        timecode(end, '.'),
        text
    )
}

fn plain_text(body: String, content_type: &'static str) -> Response {
    ([(header::CONTENT_TYPE, content_type)], body).into_response()
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

async fn handle_audio(State(s): State<AppState>, form: Multipart, endpoint: Endpoint) -> Response {
    let span = info_span!("http.openai_audio", rid = field::Empty);
    async move {
        let start = Instant::now();
        if s.service_mode != ServiceMode::Offline {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                None,
                "this server is running in streaming mode; open a WebSocket to \
                 /v1/realtime instead",
            );
        }

        let form = match read_form(form).await {
            Ok(f) => f,
            Err(resp) => return resp,
        };

        if let Some(model) = form.model.as_deref() {
            if !s.serves_model(model) {
                return api_error(
                    StatusCode::NOT_FOUND,
                    "invalid_request_error",
                    Some("model"),
                    format!(
                        "the model {model:?} does not exist; this server serves {:?}",
                        s.served_model_names
                    ),
                );
            }
        }

        // `timestamp_granularities[]` is only meaningful for verbose_json, and
        // saying so beats silently dropping it: a client that asked for word
        // times and got `{"text": ...}` has no way to tell the request was
        // understood.  (The *decode family's* ability to align is checked at
        // admission, where the engine knows which one is running.)
        if !form.granularities.is_empty()
            && !matches!(form.response_format, ResponseFormat::VerboseJson)
        {
            return invalid_request(
                Some("timestamp_granularities"),
                "timestamp_granularities[] requires response_format=verbose_json",
            );
        }
        if form
            .granularities
            .iter()
            .any(|g| g != "word" && g != "segment")
        {
            return invalid_request(
                Some("timestamp_granularities"),
                "timestamp_granularities[] accepts \"word\" and \"segment\"",
            );
        }
        if form.stream && !form.response_format.streamable() {
            return invalid_request(
                Some("stream"),
                "stream=true supports response_format json or text only \
                 (srt, vtt and verbose_json need the utterance's total duration, \
                 which is not known until the last token)",
            );
        }

        let decoded = match decode_audio(
            &form.file,
            &DecodeOptions {
                hint: form.hint(),
                encoding: SourceEncoding::Container,
                source_sample_rate: None,
                target_sample_rate: Some(s.sample_rate),
                max_samples: s.max_audio_samples,
            },
        ) {
            Ok(d) => d,
            Err(e) => {
                let status = match e {
                    AudioError::TooLong(..) => StatusCode::PAYLOAD_TOO_LARGE,
                    AudioError::UnsupportedCodec(_) | AudioError::CodecsDisabled(_) => {
                        StatusCode::NOT_IMPLEMENTED
                    }
                    _ => StatusCode::BAD_REQUEST,
                };
                return api_error(
                    status,
                    "invalid_request_error",
                    Some("file"),
                    format!("could not decode the uploaded audio: {e}"),
                );
            }
        };

        let decoding = match form.decoding_params(endpoint) {
            Ok(d) => d,
            Err(msg) => return invalid_request(None, msg),
        };
        let audio: Bytes = decoded.samples;
        let duration_s = (audio.len() / 4) as f32 / decoded.sample_rate.max(1) as f32;

        if form.stream {
            return stream_transcription(s, audio, decoded.sample_rate, decoding, form).await;
        }

        let final_ = match submit_offline_and_wait(
            &s,
            audio,
            decoded.sample_rate,
            0,
            decoding,
            start,
            openai_offline(),
        )
        .await
        {
            Ok(f) => f,
            Err(e) => {
                let kind = if e.status.is_server_error() {
                    "server_error"
                } else {
                    "invalid_request_error"
                };
                return api_error(e.status, kind, None, e.message);
            }
        };
        Span::current().record("rid", final_.request_id.as_str());
        info!(
            rid = %final_.request_id,
            task = endpoint.reported_task(),
            duration_s,
            elapsed_ms = final_.elapsed_ms,
            format = ?form.response_format,
            transcript = %final_.text,
            "openai transcription ok"
        );

        // The end of the last token when the family aligns them, the audio's
        // own duration otherwise — never a fabricated value.
        let end = final_.end_time_s.unwrap_or(duration_s);
        match form.response_format {
            ResponseFormat::Json => Json(json!({ "text": final_.text })).into_response(),
            ResponseFormat::Text => plain_text(final_.text, "text/plain; charset=utf-8"),
            ResponseFormat::Srt => plain_text(
                srt_body(&final_.text, end),
                "application/x-subrip; charset=utf-8",
            ),
            ResponseFormat::Vtt => {
                plain_text(vtt_body(&final_.text, end), "text/vtt; charset=utf-8")
            }
            ResponseFormat::VerboseJson => Json(VerboseResponse {
                task: endpoint.reported_task(),
                language: form
                    .language
                    .as_deref()
                    .and_then(oasr_wire::normalize_language),
                duration: duration_s,
                segments: vec![VerboseSegment {
                    id: 0,
                    seek: 0,
                    start: 0.0,
                    end,
                    text: final_.text.clone(),
                    tokens: final_.tokens.first().cloned().unwrap_or_default(),
                    temperature: form.temperature,
                }],
                words: form.wants_words().then(|| {
                    final_
                        .words
                        .unwrap_or_default()
                        .into_iter()
                        .map(|w| VerboseWord {
                            word: w.word,
                            start: w.start,
                            end: w.end,
                            confidence: w.confidence,
                        })
                        .collect()
                }),
                text: final_.text,
                request_id: final_.request_id,
                finish_reason: final_.finish_reason,
            })
            .into_response(),
        }
    }
    .instrument(span)
    .await
}

// ---------------------------------------------------------------------------
// `stream=true` — server-sent events
// ---------------------------------------------------------------------------

/// Serve one transcription as SSE, in OpenAI's `transcript.text.*` shape.
///
/// The engine's partials carry the transcript *so far*, while the protocol
/// wants the increment, so each delta is the suffix the partial added.  A
/// partial that rewrites rather than extends (the frame-synchronous families
/// revise a beam) yields no delta: the client's running concatenation then lags
/// the truth instead of contradicting it, and `transcript.text.done` — which
/// always carries the complete text — settles it.
async fn stream_transcription(
    s: AppState,
    audio: Bytes,
    sample_rate: u32,
    decoding: Option<DecodingParams>,
    form: AudioForm,
) -> Response {
    let mut handle = match s
        .pool
        .submit_offline_streaming(audio, sample_rate, 0, decoding)
        .await
    {
        Ok(h) => h,
        Err(e) => {
            warn!(%e, "openai stream submit rejected");
            return api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                None,
                format!("submit failed: {e}"),
            );
        }
    };
    // Both streamable formats carry the same events: `json` and `text` differ
    // only in how a *non*-streaming response is framed.
    let _ = form;
    let stream = async_stream::stream! {
        let mut emitted = String::new();
        while let Some(ev) = handle.events.next().await {
            match ev {
                Event::Partial { text, .. } => {
                    let Some(delta) = text.strip_prefix(emitted.as_str()) else {
                        debug!("non-monotone partial; deferring to the final event");
                        continue;
                    };
                    if delta.is_empty() {
                        continue;
                    }
                    let payload = json!({"type": "transcript.text.delta", "delta": delta});
                    emitted = text;
                    yield Ok(SseEvent::default().data(payload.to_string()));
                }
                Event::Final { text, .. } => {
                    handle.finish();
                    let payload = json!({"type": "transcript.text.done", "text": text});
                    yield Ok(SseEvent::default().data(payload.to_string()));
                    break;
                }
                Event::Error { code, message, .. } => {
                    handle.finish();
                    let payload = json!({
                        "type": "error",
                        "error": {"message": message, "type": format!("{code:?}")},
                    });
                    yield Ok(SseEvent::default().data(payload.to_string()));
                    break;
                }
                _ => {}
            }
        }
        s.pool.release(&handle.request_id);
        // OpenAI's clients stop on this sentinel; without it they wait for
        // the connection to close.
        yield Ok::<SseEvent, std::convert::Infallible>(SseEvent::default().data("[DONE]"));
    };
    Sse::new(stream).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_formats_parse_and_reject() {
        assert_eq!(ResponseFormat::parse("json"), Some(ResponseFormat::Json));
        assert_eq!(
            ResponseFormat::parse(" VERBOSE_JSON "),
            Some(ResponseFormat::VerboseJson)
        );
        assert_eq!(ResponseFormat::parse("srt"), Some(ResponseFormat::Srt));
        assert_eq!(ResponseFormat::parse("yaml"), None);
    }

    /// Only the two formats that need no total duration can be streamed.
    #[test]
    fn only_json_and_text_are_streamable() {
        assert!(ResponseFormat::Json.streamable());
        assert!(ResponseFormat::Text.streamable());
        for f in [
            ResponseFormat::Srt,
            ResponseFormat::Vtt,
            ResponseFormat::VerboseJson,
        ] {
            assert!(!f.streamable(), "{f:?}");
        }
    }

    #[test]
    fn timecodes_match_the_subtitle_formats() {
        assert_eq!(timecode(0.0, ','), "00:00:00,000");
        assert_eq!(timecode(3.42, ','), "00:00:03,420");
        assert_eq!(timecode(3661.5, '.'), "01:01:01.500");
        // A negative end time (a malformed alignment) clamps rather than
        // producing a subtitle no player will load.
        assert_eq!(timecode(-1.0, ','), "00:00:00,000");
    }

    #[test]
    fn srt_and_vtt_carry_one_cue_over_the_utterance() {
        let srt = srt_body("hello world", 2.5);
        assert!(srt.starts_with("1\n00:00:00,000 --> 00:00:02,500\nhello world"));
        let vtt = vtt_body("hello world", 2.5);
        assert!(vtt.starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nhello world"));
    }

    /// The filename is the fallback container hint, and only when it names one
    /// we can actually decode — otherwise the sniffer should get the decision.
    #[test]
    fn container_hints_come_from_the_content_type_then_the_filename() {
        let form = |ct: Option<&str>, fname: Option<&str>| AudioForm {
            file: Bytes::from_static(b"x"),
            filename: fname.map(str::to_owned),
            content_type: ct.map(str::to_owned),
            model: None,
            language: None,
            prompt: None,
            response_format: ResponseFormat::Json,
            temperature: None,
            granularities: Vec::new(),
            stream: false,
        };
        assert_eq!(form(Some("audio/mpeg"), None).hint(), Some("audio/mpeg"));
        // `octet-stream` names nothing: fall through to the extension.
        assert_eq!(
            form(Some("application/octet-stream"), Some("memo.m4a")).hint(),
            Some("m4a")
        );
        // Neither is usable: let the magic bytes decide.
        assert_eq!(form(None, Some("recording")).hint(), None);
        assert_eq!(form(Some("text/plain"), Some("a.txt")).hint(), None);
    }

    /// `/v1/audio/translations` *is* the task — there is no form field for it,
    /// so the endpoint must set it.
    #[test]
    fn the_translations_endpoint_asserts_the_task() {
        let form = AudioForm {
            file: Bytes::from_static(b"x"),
            filename: None,
            content_type: None,
            model: None,
            language: Some("fr-FR".into()),
            prompt: None,
            response_format: ResponseFormat::Json,
            temperature: None,
            granularities: Vec::new(),
            stream: false,
        };
        let p = form
            .decoding_params(Endpoint::Translations)
            .unwrap()
            .expect("task is always set here, so params are never empty");
        assert_eq!(p.task.as_deref(), Some("translate"));
        // BCP-47 is reduced to the primary subtag the models' tokens use.
        assert_eq!(p.language.as_deref(), Some("fr"));
    }

    /// A plain transcription must send **no** `task`.
    ///
    /// It looks harmless to set `task=transcribe` — it is what the endpoint
    /// does — but a family with no task control rejects any set `task`, so
    /// asserting it turned `POST /v1/audio/transcriptions` into a 400 on every
    /// CTC engine. Caught by running a real server, not by this file's first
    /// draft, which asserted the bug.
    #[test]
    fn a_plain_transcription_sends_no_task() {
        let bare = AudioForm {
            file: Bytes::from_static(b"x"),
            filename: None,
            content_type: None,
            model: None,
            language: None,
            prompt: None,
            response_format: ResponseFormat::Json,
            temperature: None,
            granularities: Vec::new(),
            stream: false,
        };
        assert_eq!(
            bare.decoding_params(Endpoint::Transcriptions).unwrap(),
            None,
            "an upload with no options must carry no options"
        );
        // With a language, the language travels and the task still does not.
        let with_lang = AudioForm {
            language: Some("de".into()),
            ..bare
        };
        let p = with_lang
            .decoding_params(Endpoint::Transcriptions)
            .unwrap()
            .unwrap();
        assert_eq!(p.task, None);
        assert_eq!(p.language.as_deref(), Some("de"));
    }

    /// The response still reports a concrete task: it says what happened.
    #[test]
    fn the_reported_task_is_always_concrete() {
        assert_eq!(Endpoint::Transcriptions.reported_task(), "transcribe");
        assert_eq!(Endpoint::Translations.reported_task(), "translate");
    }

    #[test]
    fn an_unparseable_language_is_rejected_rather_than_dropped() {
        let form = AudioForm {
            file: Bytes::from_static(b"x"),
            filename: None,
            content_type: None,
            model: None,
            language: Some("!!".into()),
            prompt: None,
            response_format: ResponseFormat::Json,
            temperature: None,
            granularities: Vec::new(),
            stream: false,
        };
        let err = form
            .decoding_params(Endpoint::Transcriptions)
            .expect_err("junk must not be silently dropped");
        assert!(err.contains("language"), "{err:?}");
    }
}
