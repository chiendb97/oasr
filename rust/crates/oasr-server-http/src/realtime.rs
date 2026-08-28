// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! `GET /v1/realtime` — streaming transcription over a WebSocket.
//!
//! This is the surface OASR was actually missing.  The gRPC
//! `StreamingRecognize` RPC has always existed and is the better transport for
//! a server-to-server client, but a browser cannot open a gRPC bidi stream, and
//! neither can most script-level clients — which is why `examples/web` shipped
//! a FastAPI process whose entire job was to relay between a WebSocket and
//! gRPC.  That bridge is deleted by this file.
//!
//! ## Protocol
//!
//! Event names follow OpenAI's realtime *transcription* session, so a client
//! written against that API needs a URL change and nothing else.  Two OASR
//! additions, both strictly optional:
//!
//! * **binary frames** carry raw audio directly, skipping base64 (a third of
//!   the bytes, and no JSON parse of a multi-MiB string).  `input_audio_buffer.append`
//!   with base64 works identically for clients that cannot send binary.
//! * `session.update` may set `sample_rate`, `encoding`, `language`, `task` and
//!   `prompt`; anything unset keeps the server's default.
//!
//! Client → server:
//!
//! | Message | Meaning |
//! |---|---|
//! | `{"type":"session.update","session":{…}}` | Configure. Only honoured before the first audio. |
//! | `{"type":"input_audio_buffer.append","audio":"<base64>"}` | One audio chunk. |
//! | *binary frame* | One audio chunk (same thing, unwrapped). |
//! | `{"type":"input_audio_buffer.commit"}` | End of utterance — the half-close. |
//!
//! Server → client:
//!
//! | Message | Meaning |
//! |---|---|
//! | `{"type":"transcription_session.created","session":{…}}` | Handshake, echoing the resolved config. |
//! | `{"type":"conversation.item.input_audio_transcription.delta","delta":"…"}` | Interim transcript. |
//! | `{"type":"conversation.item.input_audio_transcription.completed","transcript":"…"}` | Final. |
//! | `{"type":"error","error":{…}}` | Terminal failure. |
//!
//! ## Both service modes
//!
//! A `streaming` engine consumes chunks as they arrive.  An `offline` engine
//! cannot (its decode families need the whole utterance), so the audio is
//! buffered and submitted on commit, and the *text* streams back — the same
//! trade the gRPC surface makes, and the reason a speech-LLM still feels
//! incremental here.

use std::time::{Duration, Instant};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use bytes::{Bytes, BytesMut};
use oasr_asr::{parse_encoding, PcmEncoding, PcmStream, SourceEncoding};
use oasr_engine_client::handle::{OfflineStreamHandle, StreamingHandle};
use oasr_metrics as om;
use oasr_wire::{normalize_language, DecodingParams, Event};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::StreamExt;
use tracing::{debug, info, info_span, warn, Instrument};

use crate::http_metrics::realtime_streaming;
use crate::router::{AppState, ServiceMode};

/// `GET /v1/realtime` — upgrade and run one transcription session.
pub async fn handle_realtime(State(s): State<AppState>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| async move {
        if let Err(e) = run_session(s, socket).await {
            debug!(reason = %e, "realtime session ended");
        }
    })
}

/// Per-session configuration, from `session.update`.
#[derive(Debug, Deserialize, Default)]
struct SessionUpdate {
    /// Rate of the audio the client will send.  Unset means the model's own.
    #[serde(default)]
    sample_rate: Option<u32>,
    /// Encoding name, in the same spelling the REST surface uses
    /// (`LINEAR16` — the default — `LINEAR32F`, `MULAW`, `ALAW`).
    #[serde(default)]
    encoding: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    /// Suppress interim results and send only the final transcript.
    #[serde(default)]
    interim_results: Option<bool>,
    #[serde(default)]
    model: Option<String>,
    /// Server-side turn detection, in OpenAI's shape.  Absent or `null` keeps
    /// the session on manual `input_audio_buffer.commit`, which is what it did
    /// before server VAD existed — so an existing client is unaffected.
    #[serde(default)]
    turn_detection: Option<TurnDetection>,
}

/// OpenAI's `turn_detection` object.
///
/// Only `server_vad` is accepted.  `semantic_vad` decides a turn is over from
/// *what was said*, which needs a model trained to predict end-of-utterance;
/// there is no such head in this engine, and accepting the value would mean
/// silently doing acoustic detection under a name that promises otherwise.
///
/// `threshold` and `prefix_padding_ms` are rejected rather than ignored. They
/// are engine-level segmenter settings here (`--vad-option threshold=...`), and
/// a client that sets one per request and sees no change would reasonably
/// conclude it had tuned something.
#[derive(Debug, Clone, Deserialize)]
struct TurnDetection {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    threshold: Option<f32>,
    #[serde(default)]
    prefix_padding_ms: Option<u32>,
    #[serde(default)]
    silence_duration_ms: Option<u32>,
}

impl TurnDetection {
    fn validate(&self) -> Result<(), SessionError> {
        match self.kind.as_deref() {
            None | Some("server_vad") => {}
            Some(other) => {
                return Err(SessionError::new(
                    Some("turn_detection.type"),
                    format!(
                        "{other:?} is not supported; this engine detects turns \
                         acoustically, so only \"server_vad\" is available"
                    ),
                ))
            }
        }
        // `param` is `&'static str`, so the field names are literals rather
        // than formatted — which also keeps the error's `param` a value a client
        // can match on instead of a sentence.
        for (param, name, set) in [
            (
                "turn_detection.threshold",
                "threshold",
                self.threshold.is_some(),
            ),
            (
                "turn_detection.prefix_padding_ms",
                "prefix_padding_ms",
                self.prefix_padding_ms.is_some(),
            ),
        ] {
            if set {
                return Err(SessionError::new(
                    Some(param),
                    format!(
                        "{name} is an engine-level setting on this server; start it \
                         with --vad-option {name}=... instead of sending it per session"
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// The resolved session: what the handshake echoes back.
struct Session {
    pcm: PcmStream,
    decoding: Option<DecodingParams>,
    interim_results: bool,
}

/// Per-session metric state for the streaming SLIs.
///
/// Time to first partial is measured from the **first inbound audio byte**, not
/// from the socket upgrade. A client may hold an open realtime socket for
/// minutes before it starts speaking, and timing from the upgrade would report
/// that silence as latency — a p99 that tracks user behaviour instead of
/// server behaviour, and moves when nothing about the server has changed.
struct SessionMetrics {
    started: Instant,
    first_audio_at: Option<Instant>,
    partial_recorded: bool,
    /// Accumulated so the per-request duration can be recorded at the end.
    /// A live stream has no total until it closes, while the RTFx denominator
    /// has to accrue as audio arrives — so the two are counted separately
    /// rather than one being derived from the other.
    audio_seconds: f64,
}

impl SessionMetrics {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            first_audio_at: None,
            partial_recorded: false,
            audio_seconds: 0.0,
        }
    }

    /// Note `bytes` of engine-rate f32 PCM arriving from the client.
    fn audio(&mut self, bytes: usize, sample_rate: u32) {
        self.first_audio_at.get_or_insert_with(Instant::now);
        let seconds = om::f32_pcm_seconds(bytes, sample_rate);
        self.audio_seconds += seconds;
        realtime_streaming().audio_ingested(seconds);
    }

    /// Record TTFP, once, on the first partial that follows some audio.
    fn partial(&mut self) {
        if self.partial_recorded {
            return;
        }
        if let Some(t0) = self.first_audio_at {
            self.partial_recorded = true;
            realtime_streaming().first_partial(t0.elapsed());
        }
    }

    fn finished(&self, outcome: om::Outcome) {
        realtime_streaming().audio_duration(self.audio_seconds);
        realtime_streaming().finished(outcome, self.started.elapsed());
    }
}

/// A session that failed to configure — reported to the client, then closed.
struct SessionError {
    param: Option<&'static str>,
    message: String,
}

impl SessionError {
    fn new(param: Option<&'static str>, message: impl Into<String>) -> Self {
        Self {
            param,
            message: message.into(),
        }
    }
}

fn resolve_session(s: &AppState, update: &SessionUpdate) -> Result<Session, SessionError> {
    if let Some(model) = update.model.as_deref() {
        if !s.serves_model(model) {
            return Err(SessionError::new(
                Some("model"),
                format!("the model {model:?} does not exist on this server"),
            ));
        }
    }
    // A container header arrives once, at the front of a stream whose chunks
    // are decoded independently — so, as on the gRPC surface, the realtime
    // session takes headerless PCM only.
    let encoding = match update.encoding.as_deref() {
        None => PcmEncoding::I16Le,
        Some(name) => match parse_encoding(name) {
            Ok((SourceEncoding::Pcm(p), _)) => p,
            Ok((SourceEncoding::Container, _)) => {
                return Err(SessionError::new(
                    Some("encoding"),
                    format!(
                        "{name} is a container; a realtime session takes headerless \
                         PCM (LINEAR16, LINEAR32F, MULAW, ALAW). Post containers to \
                         /v1/audio/transcriptions."
                    ),
                ))
            }
            Err(e) => return Err(SessionError::new(Some("encoding"), e.to_string())),
        },
    };
    let source_rate = update.sample_rate.unwrap_or(s.sample_rate);
    // Built before anything is admitted, so an implausible rate is rejected at
    // open rather than after the client has streamed a minute of audio.
    let pcm = PcmStream::new(encoding, source_rate, s.sample_rate)
        .map_err(|e| SessionError::new(Some("sample_rate"), e.to_string()))?;

    let language = match update
        .language
        .as_deref()
        .map(str::trim)
        .filter(|l| !l.is_empty())
    {
        None => None,
        Some(tag) => Some(normalize_language(tag).ok_or_else(|| {
            SessionError::new(Some("language"), format!("{tag:?} is not a language tag"))
        })?),
    };
    let turn = update.turn_detection.as_ref();
    if let Some(t) = turn {
        t.validate()?;
    }
    let decoding = DecodingParams {
        n_best: None,
        max_new_tokens: None,
        temperature: None,
        top_k: None,
        top_p: None,
        prompt: update.prompt.clone().filter(|p| !p.is_empty()),
        task: update
            .task
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(|t| t.to_ascii_lowercase()),
        language,
        // A realtime session has no place to put word timings: the transcript
        // arrives as deltas, and the alignment (where a family has one) is a
        // property of the finished utterance.
        word_timestamps: None,
        // Server-side turn detection, in OpenAI's own shape: `turn_detection`
        // absent or null keeps this session on manual `input_audio_buffer.commit`,
        // which is what it has always done.
        single_utterance: turn.as_ref().map(|_| true),
        vad_events: turn.as_ref().map(|_| true),
        endpoint_silence_ms: turn.as_ref().and_then(|t| t.silence_duration_ms),
    }
    .validated()
    .map_err(|msg| SessionError::new(None, msg))?;

    Ok(Session {
        pcm,
        decoding,
        interim_results: update.interim_results.unwrap_or(true),
    })
}

/// Anything that ends a session; only reported at DEBUG, since a client
/// hanging up mid-stream is the normal case.
type SessionEnd = String;

async fn run_session(s: AppState, mut socket: WebSocket) -> Result<(), SessionEnd> {
    // Configuration is optional: a client that opens the socket and starts
    // sending 16 kHz LINEAR16 immediately gets the defaults.  So the first
    // message is *peeked*, not required — unlike the gRPC RPC, whose first
    // message must carry the config.
    let mut update = SessionUpdate::default();
    let mut pending_audio: Option<Bytes> = None;
    let mut committed = false;
    loop {
        match next_client_message(&mut socket).await? {
            ClientMessage::Configure(u) => {
                update = u;
                break;
            }
            ClientMessage::Audio(bytes) => {
                pending_audio = Some(bytes);
                break;
            }
            ClientMessage::Commit => {
                committed = true;
                break;
            }
            ClientMessage::Ignored => continue,
        }
    }

    let session = match resolve_session(&s, &update) {
        Ok(sess) => sess,
        Err(e) => {
            let _ = send_json(
                &mut socket,
                json!({
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "param": e.param,
                        "message": e.message,
                    }
                }),
            )
            .await;
            return Err(e.message);
        }
    };
    send_json(
        &mut socket,
        json!({
            "type": "transcription_session.created",
            "session": {
                "model": s.model_id,
                "sample_rate": session.pcm.source_rate(),
                "model_sample_rate": session.pcm.target_rate(),
                "interim_results": session.interim_results,
                "mode": match s.service_mode {
                    ServiceMode::Streaming => "streaming",
                    ServiceMode::Offline => "offline",
                },
            }
        }),
    )
    .await?;

    let span = info_span!("ws.realtime");
    match s.service_mode {
        ServiceMode::Streaming => {
            chunked_session(s, socket, session, pending_audio, committed)
                .instrument(span)
                .await
        }
        ServiceMode::Offline => {
            buffered_session(s, socket, session, pending_audio, committed)
                .instrument(span)
                .await
        }
    }
}

/// One inbound message, already classified.
#[derive(Debug)]
enum ClientMessage {
    Configure(SessionUpdate),
    Audio(Bytes),
    Commit,
    /// A ping, an empty frame, or a control message this server does not act
    /// on — read and dropped.
    Ignored,
}

async fn next_client_message(socket: &mut WebSocket) -> Result<ClientMessage, SessionEnd> {
    loop {
        let msg = socket
            .recv()
            .await
            .ok_or_else(|| "client closed the socket".to_string())?
            .map_err(|e| format!("websocket receive: {e}"))?;
        return Ok(match msg {
            // The fast path: audio as it comes off the wire.
            Message::Binary(b) if !b.is_empty() => ClientMessage::Audio(Bytes::from(b)),
            Message::Binary(_) => ClientMessage::Ignored,
            Message::Text(t) => match serde_json::from_str::<Value>(&t) {
                Ok(v) => classify_text(v)?,
                Err(e) => return Err(format!("malformed JSON message: {e}")),
            },
            Message::Close(_) => return Err("client closed the socket".into()),
            Message::Ping(_) | Message::Pong(_) => continue,
        });
    }
}

fn classify_text(v: Value) -> Result<ClientMessage, SessionEnd> {
    match v.get("type").and_then(Value::as_str).unwrap_or_default() {
        "session.update" | "transcription_session.update" => {
            let inner = v.get("session").cloned().unwrap_or(Value::Null);
            let update = if inner.is_null() {
                SessionUpdate::default()
            } else {
                serde_json::from_value(inner)
                    .map_err(|e| format!("malformed session.update: {e}"))?
            };
            Ok(ClientMessage::Configure(update))
        }
        "input_audio_buffer.append" => {
            let b64 = v.get("audio").and_then(Value::as_str).unwrap_or_default();
            let bytes = BASE64
                .decode(b64)
                .map_err(|e| format!("input_audio_buffer.append carries invalid base64: {e}"))?;
            Ok(ClientMessage::Audio(Bytes::from(bytes)))
        }
        "input_audio_buffer.commit" | "input_audio_buffer.end" => Ok(ClientMessage::Commit),
        // `input_audio_buffer.clear`, `response.create`, … — accepted and
        // ignored on purpose: they have no meaning for a transcription-only
        // session, and erroring would break clients that send them by habit.
        _ => Ok(ClientMessage::Ignored),
    }
}

async fn send_json(socket: &mut WebSocket, value: Value) -> Result<(), SessionEnd> {
    socket
        .send(Message::Text(value.to_string()))
        .await
        .map_err(|e| format!("websocket send: {e}"))
}

/// Tracks what the client has already been told, so each event can carry a
/// true increment.
///
/// Converts cumulative engine partials into protocol increments.
#[derive(Default)]
struct Emitted(String);

impl Emitted {
    /// The event for one cumulative partial, or `None` when it added nothing.
    ///
    /// Revisions carry corrected full text with an empty delta because they have
    /// no append-only representation.
    fn event(&mut self, partial: &str) -> Option<Value> {
        let delta = match partial.strip_prefix(self.0.as_str()) {
            Some("") => return None,
            Some(suffix) => suffix.to_string(),
            None => String::new(),
        };
        self.0 = partial.to_string();
        Some(json!({
            "type": "conversation.item.input_audio_transcription.delta",
            "delta": delta,
            // OASR extension: the authoritative transcript so far.  Present on
            // every delta so a caller never has to reconstruct it, and the only
            // way a revision can be communicated at all.
            "text": partial,
        }))
    }
}

fn completed_event(text: &str) -> Value {
    json!({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": text,
    })
}

fn error_event(message: &str, kind: &str) -> Value {
    json!({"type": "error", "error": {"type": kind, "message": message}})
}

/// Streaming engine: feed chunks as they arrive, forward events as they come.
async fn chunked_session(
    s: AppState,
    mut socket: WebSocket,
    mut session: Session,
    pending_audio: Option<Bytes>,
    already_committed: bool,
) -> Result<(), SessionEnd> {
    let mut handle: StreamingHandle = s
        .pool
        .open_streaming(s.sample_rate, 0, session.decoding.clone())
        .await
        .map_err(|e| {
            warn!(%e, "realtime open rejected");
            format!("submit failed: {e}")
        })?;
    let rid = handle.request_id.clone();
    let mut emitted = Emitted::default();
    let mut metrics = SessionMetrics::new();
    info!(rid = %rid, resampling = session.pcm.is_resampling(), "realtime stream opened");

    if let Some(chunk) = pending_audio {
        feed(&mut session, &handle, &chunk, &mut metrics, s.sample_rate).await?;
    }
    let mut inbound_done = already_committed;
    if inbound_done {
        let tail = session.pcm.flush().unwrap_or_default();
        let _ = handle.flush_last(tail).await;
    }

    let result = loop {
        tokio::select! {
            ev = handle.events.next() => match ev {
                Some(Event::Partial { text, speech_events, .. }) => {
                    // Timed on the engine's partial, before the interim-results
                    // filter: the SLI is how fast the engine produced a
                    // transcript, not whether this client asked to see it.
                    metrics.partial();
                    for ev in speech_activity_events(speech_events) {
                        send_json(&mut socket, ev).await?;
                    }
                    if session.interim_results {
                        if let Some(ev) = emitted.event(&text) {
                            send_json(&mut socket, ev).await?;
                        }
                    }
                }
                Some(Event::Final { text, speech_events, endpoint_reason, .. }) => {
                    handle.finish();
                    for ev in speech_activity_events(speech_events) {
                        send_json(&mut socket, ev).await?;
                    }
                    // A turn the endpointer closed is *committed* — the buffer
                    // became a conversation item without the client asking.
                    // A turn the audio closed was already committed by the
                    // client's own `input_audio_buffer.commit`, so re-announcing
                    // it would double-count the item.
                    if endpoint_reason.is_some() {
                        send_json(&mut socket, json!({"type": "input_audio_buffer.committed"}))
                            .await?;
                    }
                    send_json(&mut socket, completed_event(&text)).await?;
                    info!(
                        rid = %rid,
                        transcript = %text,
                        endpoint = endpoint_reason.as_deref().unwrap_or("audio_end"),
                        "realtime final"
                    );
                    break Ok(());
                }
                Some(Event::Error { code, message, .. }) => {
                    handle.finish();
                    let _ = send_json(&mut socket, error_event(&message, &format!("{code:?}"))).await;
                    break Err(message);
                }
                Some(_) => {}
                None => {
                    let _ = send_json(&mut socket, error_event("event stream closed", "server_error")).await;
                    break Err("event stream closed".into());
                }
            },
            msg = next_client_message(&mut socket), if !inbound_done => match msg {
                Ok(ClientMessage::Audio(bytes)) => {
                    feed(&mut session, &handle, &bytes, &mut metrics, s.sample_rate).await?
                }
                Ok(ClientMessage::Commit) => {
                    inbound_done = true;
                    // The resampler's tail rides out on the final chunk;
                    // dropping it cuts the last word.
                    let tail = session.pcm.flush().unwrap_or_else(|e| {
                        warn!(reason = %e, "resampler flush failed; dropping tail");
                        Bytes::new()
                    });
                    let _ = handle.flush_last(tail).await;
                }
                // A client that hangs up without committing gets its request
                // cancelled by `CancelOnDrop`, which is the right answer: there
                // is nothing to transcribe and nobody to send it to.
                Ok(ClientMessage::Configure(_)) => {
                    debug!("ignoring session.update after audio has started");
                }
                Ok(ClientMessage::Ignored) => {}
                Err(end) => break Err(end),
            },
        }
    };
    metrics.finished(outcome_of(&result));
    s.pool.release(&rid);
    let _ = socket.send(Message::Close(None)).await;
    result
}

/// Map engine speech-activity transitions onto OpenAI's realtime events.
///
/// The engine already names them `speech_started` / `speech_stopped` — the
/// kinds are OpenAI's, chosen there precisely so this is a prefix and not a
/// translation table that can drift.  `audio_start_ms` / `audio_end_ms` carry
/// the transition time in **audio** milliseconds, which is what the client can
/// seek to; wall clock would move with the uplink.
fn speech_activity_events(events: Option<Vec<oasr_wire::SpeechEvent>>) -> Vec<serde_json::Value> {
    let Some(events) = events else {
        return Vec::new();
    };
    events
        .into_iter()
        .filter_map(|e| {
            let ms = (e.time.max(0.0) * 1000.0).round() as i64;
            match e.kind.as_str() {
                "speech_started" => Some(json!({
                    "type": "input_audio_buffer.speech_started",
                    "audio_start_ms": ms,
                })),
                "speech_stopped" => Some(json!({
                    "type": "input_audio_buffer.speech_stopped",
                    "audio_end_ms": ms,
                })),
                _ => None,
            }
        })
        .collect()
}

/// Classify a session's end for the `outcome` label.
///
/// A realtime session that ends without a final transcript is almost always a
/// client that hung up, which is a cancellation rather than a server error —
/// and counting every browser tab close as an error would make the error-rate
/// panel useless on exactly the surface where disconnects are normal.
fn outcome_of(result: &Result<(), SessionEnd>) -> om::Outcome {
    match result {
        Ok(()) => om::Outcome::Ok,
        Err(_) => om::Outcome::Cancelled,
    }
}

async fn feed(
    session: &mut Session,
    handle: &StreamingHandle,
    chunk: &[u8],
    metrics: &mut SessionMetrics,
    sample_rate: u32,
) -> Result<(), SessionEnd> {
    let decoded = session
        .pcm
        .decode_chunk(chunk)
        .map_err(|e| format!("pcm decode: {e}"))?;
    // A resampling stream can hold a chunk back inside the filter; feeding an
    // empty one would just cost the engine a step.
    if decoded.is_empty() {
        return Ok(());
    }
    // Counted on the *decoded* bytes, at the engine's rate: a client sending
    // 8 kHz µ-law and one sending 48 kHz f32 have wildly different byte counts
    // for the same second of speech, and RTFx has to be about the speech.
    metrics.audio(decoded.len(), sample_rate);
    handle
        .push_chunk(decoded)
        .await
        .map_err(|_| "audio channel dropped".to_string())
}

/// Offline engine: buffer the audio, submit on commit, stream the text back.
///
/// The constraint is on *audio in*, not on *text out* — the AR families emit a
/// partial per engine tick, which is the token-streaming UX a speech-LLM client
/// expects.  Identical to what the gRPC surface does for the same reason.
async fn buffered_session(
    s: AppState,
    mut socket: WebSocket,
    mut session: Session,
    pending_audio: Option<Bytes>,
    already_committed: bool,
) -> Result<(), SessionEnd> {
    let mut metrics = SessionMetrics::new();
    let mut buffered = BytesMut::new();
    if let Some(chunk) = pending_audio {
        let decoded = session
            .pcm
            .decode_chunk(&chunk)
            .map_err(|e| format!("pcm decode: {e}"))?;
        metrics.audio(decoded.len(), s.sample_rate);
        buffered.extend_from_slice(&decoded);
    }
    let mut committed = already_committed;
    while !committed {
        match next_client_message(&mut socket).await? {
            ClientMessage::Audio(chunk) => {
                let decoded = session
                    .pcm
                    .decode_chunk(&chunk)
                    .map_err(|e| format!("pcm decode: {e}"))?;
                metrics.audio(decoded.len(), s.sample_rate);
                buffered.extend_from_slice(&decoded);
                if let Some(cap) = s.max_audio_samples {
                    if buffered.len() / 4 > cap {
                        let msg = format!("audio exceeds the {cap}-sample limit");
                        let _ = send_json(&mut socket, error_event(&msg, "invalid_request_error"))
                            .await;
                        return Err(msg);
                    }
                }
            }
            ClientMessage::Commit => committed = true,
            ClientMessage::Configure(_) => {
                debug!("ignoring session.update after the session has started")
            }
            ClientMessage::Ignored => {}
        }
    }
    match session.pcm.flush() {
        Ok(tail) => buffered.extend_from_slice(&tail),
        Err(e) => warn!(reason = %e, "resampler flush failed; dropping tail"),
    }
    if buffered.is_empty() {
        let msg = "no audio received before input_audio_buffer.commit";
        let _ = send_json(&mut socket, error_event(msg, "invalid_request_error")).await;
        return Err(msg.into());
    }

    let mut handle: OfflineStreamHandle = s
        .pool
        .submit_offline_streaming(
            buffered.freeze(),
            s.sample_rate,
            0,
            session.decoding.clone(),
        )
        .await
        .map_err(|e| {
            warn!(%e, "realtime (offline) submit rejected");
            format!("submit failed: {e}")
        })?;
    let rid = handle.request_id.clone();
    let mut emitted = Emitted::default();

    let result = loop {
        match handle.events.next().await {
            Some(Event::Partial { text, .. }) => {
                metrics.partial();
                if session.interim_results {
                    if let Some(ev) = emitted.event(&text) {
                        send_json(&mut socket, ev).await?;
                    }
                }
            }
            Some(Event::Final { text, .. }) => {
                handle.finish();
                send_json(&mut socket, completed_event(&text)).await?;
                info!(rid = %rid, transcript = %text, "realtime final");
                break Ok(());
            }
            Some(Event::Error { code, message, .. }) => {
                handle.finish();
                let _ = send_json(&mut socket, error_event(&message, &format!("{code:?}"))).await;
                break Err(message);
            }
            Some(_) => {}
            None => break Err("event stream closed".into()),
        }
    };
    metrics.finished(outcome_of(&result));
    s.pool.release(&rid);
    let _ = socket.send(Message::Close(None)).await;
    result
}

/// How long a realtime session may sit idle before the server gives up.  Kept
/// here rather than on `RouterLimits` because axum's timeout layer applies to
/// the *upgrade* request, not to the socket that outlives it.
pub const DEFAULT_IDLE: Duration = Duration::from_secs(300);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_and_base64_audio_are_the_same_message() {
        let audio = [1u8, 2, 3, 4];
        let msg = classify_text(json!({
            "type": "input_audio_buffer.append",
            "audio": BASE64.encode(audio),
        }))
        .unwrap();
        match msg {
            ClientMessage::Audio(b) => assert_eq!(&b[..], &audio[..]),
            _ => panic!("expected audio"),
        }
    }

    #[test]
    fn commit_ends_the_utterance_under_either_spelling() {
        for t in ["input_audio_buffer.commit", "input_audio_buffer.end"] {
            assert!(matches!(
                classify_text(json!({"type": t})).unwrap(),
                ClientMessage::Commit
            ));
        }
    }

    /// Control messages a realtime *conversation* client sends by habit must
    /// not fail a transcription session.
    #[test]
    fn unknown_control_messages_are_ignored_not_fatal() {
        for t in ["response.create", "input_audio_buffer.clear", "nonsense"] {
            assert!(matches!(
                classify_text(json!({"type": t})).unwrap(),
                ClientMessage::Ignored
            ));
        }
    }

    #[test]
    fn session_update_parses_both_spellings_and_an_absent_body() {
        for t in ["session.update", "transcription_session.update"] {
            let msg = classify_text(json!({
                "type": t,
                "session": {"sample_rate": 8000, "encoding": "MULAW", "language": "fr-FR"},
            }))
            .unwrap();
            match msg {
                ClientMessage::Configure(u) => {
                    assert_eq!(u.sample_rate, Some(8000));
                    assert_eq!(u.encoding.as_deref(), Some("MULAW"));
                    assert_eq!(u.language.as_deref(), Some("fr-FR"));
                }
                _ => panic!("expected configure"),
            }
        }
        // No `session` object at all: defaults, not an error.
        assert!(matches!(
            classify_text(json!({"type": "session.update"})).unwrap(),
            ClientMessage::Configure(_)
        ));
    }

    /// Base64 that does not decode has to fail loudly: silently dropping the
    /// chunk would produce a transcript with a hole in it.
    #[test]
    fn invalid_base64_audio_fails_the_session() {
        let err = classify_text(json!({
            "type": "input_audio_buffer.append",
            "audio": "not base64!!",
        }))
        .unwrap_err();
        assert!(err.contains("base64"), "{err}");
    }

    #[test]
    fn events_use_the_openai_realtime_names() {
        let mut emitted = Emitted::default();
        assert_eq!(
            emitted.event("hi").unwrap()["type"],
            "conversation.item.input_audio_transcription.delta"
        );
        assert_eq!(
            completed_event("hi there")["type"],
            "conversation.item.input_audio_transcription.completed"
        );
        assert_eq!(completed_event("hi there")["transcript"], "hi there");
    }

    /// The engine's partials are cumulative and the protocol's `delta` is an
    /// increment.  Forwarding the cumulative text as a delta makes a client
    /// that concatenates render every prefix again — "he", "hehe llo", …
    #[test]
    fn deltas_are_increments_of_the_cumulative_partials() {
        let mut emitted = Emitted::default();
        let ev = emitted.event("he").unwrap();
        assert_eq!(ev["delta"], "he");
        assert_eq!(ev["text"], "he");

        let ev = emitted.event("hello").unwrap();
        assert_eq!(ev["delta"], "llo", "only what the partial added");
        assert_eq!(ev["text"], "hello", "and the full transcript alongside");

        // An unchanged partial says nothing rather than sending an empty event.
        assert!(emitted.event("hello").is_none());
    }

    /// A frame-synchronous family can re-rank its beam, so a partial may
    /// *revise* rather than extend.  There is no increment to send; the event
    /// has to carry the corrected text or the client keeps the wrong words
    /// until the final.
    #[test]
    fn a_revised_partial_carries_the_corrected_text_with_an_empty_delta() {
        let mut emitted = Emitted::default();
        emitted.event("their").unwrap();
        let ev = emitted.event("there are").unwrap();
        assert_eq!(ev["delta"], "");
        assert_eq!(ev["text"], "there are");
        // And the next extension is relative to the corrected text.
        let ev = emitted.event("there are two").unwrap();
        assert_eq!(ev["delta"], " two");
    }
}
