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

use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use bytes::{Bytes, BytesMut};
use oasr_asr::{parse_encoding, PcmEncoding, PcmStream, SourceEncoding};
use oasr_engine_client::handle::{OfflineStreamHandle, StreamingHandle};
use oasr_wire::{normalize_language, DecodingParams, Event};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::StreamExt;
use tracing::{debug, info, info_span, warn, Instrument};

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
}

/// The resolved session: what the handshake echoes back.
struct Session {
    pcm: PcmStream,
    decoding: Option<DecodingParams>,
    interim_results: bool,
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
/// The engine's partials are *cumulative* — each one is the transcript so far —
/// while the protocol's `delta` is an increment.  Sending the cumulative text
/// as a delta makes a client that concatenates deltas render every prefix
/// again, which is what a naive mapping does.
#[derive(Default)]
struct Emitted(String);

impl Emitted {
    /// The event for one cumulative partial, or `None` when it added nothing.
    ///
    /// A partial that *revises* rather than extends (the frame-synchronous
    /// families re-rank a beam, so a word can change) has no increment to
    /// express: it carries an empty `delta` and the corrected `text`, and a
    /// client that reads `text` stays right where a delta-concatenating one
    /// would have to wait for the final.
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
    info!(rid = %rid, resampling = session.pcm.is_resampling(), "realtime stream opened");

    if let Some(chunk) = pending_audio {
        feed(&mut session, &handle, &chunk).await?;
    }
    let mut inbound_done = already_committed;
    if inbound_done {
        let tail = session.pcm.flush().unwrap_or_default();
        let _ = handle.flush_last(tail).await;
    }

    let result = loop {
        tokio::select! {
            ev = handle.events.next() => match ev {
                Some(Event::Partial { text, .. }) => {
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
                None => {
                    let _ = send_json(&mut socket, error_event("event stream closed", "server_error")).await;
                    break Err("event stream closed".into());
                }
            },
            msg = next_client_message(&mut socket), if !inbound_done => match msg {
                Ok(ClientMessage::Audio(bytes)) => feed(&mut session, &handle, &bytes).await?,
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
    s.pool.release(&rid);
    let _ = socket.send(Message::Close(None)).await;
    result
}

async fn feed(
    session: &mut Session,
    handle: &StreamingHandle,
    chunk: &[u8],
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
    let mut buffered = BytesMut::new();
    if let Some(chunk) = pending_audio {
        let decoded = session
            .pcm
            .decode_chunk(&chunk)
            .map_err(|e| format!("pcm decode: {e}"))?;
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
