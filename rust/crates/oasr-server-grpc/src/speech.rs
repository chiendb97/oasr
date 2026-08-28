// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! `oasr.speech.v1.Speech` service implementation.

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::Stream;
use oasr_asr::{
    decode_audio, parse_encoding, DecodeOptions, EncodingError, PcmStream, SourceEncoding,
};
use oasr_engine_client::EnginePool;
use oasr_wire::{
    normalize_language, score_posteriors, DecodingParams, ErrorCode, Event, WordTiming,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Code, Request, Response, Status, Streaming};
use tracing::{debug, error, field, info, info_span, warn, Instrument, Span};

use oasr_metrics::f32_pcm_seconds;

use crate::grpc_metrics::{
    code_of, method, outcome_for, record_rpc, streaming as grpc_streaming, unary as grpc_unary,
};
use crate::pb;

/// Service-wide configuration for the gRPC Speech handlers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceMode {
    /// Engine only accepts streaming requests.
    Streaming,
    /// Engine only accepts full-audio (unary) requests.
    Offline,
}

impl std::str::FromStr for ServiceMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "streaming" => Ok(Self::Streaming),
            "offline" => Ok(Self::Offline),
            other => Err(format!(
                "unknown service mode {other:?}: expected 'streaming' or 'offline'"
            )),
        }
    }
}

pub struct SpeechService {
    pool: Arc<EnginePool>,
    mode: ServiceMode,
    /// The engine's waveform sample rate in Hz; inbound audio is resampled to
    /// it before submission (the engine does not resample).
    sample_rate: u32,
    /// Ceiling on the **decoded** waveform, in samples.  `None` disables.
    /// Distinct from the message-size cap: a compressed body's byte length
    /// says nothing about how much audio it becomes.
    max_audio_samples: Option<usize>,
    /// Abort a streaming RPC idle this long.  `None` disables.
    stream_idle_timeout: Option<Duration>,
}

impl SpeechService {
    pub fn new(pool: Arc<EnginePool>, mode: ServiceMode, sample_rate: u32) -> Self {
        Self {
            pool,
            mode,
            sample_rate,
            max_audio_samples: None,
            stream_idle_timeout: None,
        }
    }

    /// Bound the decoded waveform at `n` samples (at the source rate).
    pub fn with_max_audio_samples(mut self, n: Option<usize>) -> Self {
        self.max_audio_samples = n;
        self
    }

    /// Abort a streaming RPC that goes `d` without an inbound audio message
    /// (before half-close) or a decode event (after).
    ///
    /// This is deliberately *not* a `tonic::transport::Server::timeout`: that
    /// applies to every request including `StreamingRecognize`, so it would cut
    /// off healthy long-lived streams.  A live stream cannot be bounded by
    /// total duration, only by inactivity.
    pub fn with_stream_idle_timeout(mut self, d: Option<Duration>) -> Self {
        self.stream_idle_timeout = d;
        self
    }
}

/// Await `fut`, mapping an idle-timeout elapse to `DEADLINE_EXCEEDED`.
/// `None` waits forever, which is what an operator gets by setting the knob
/// to 0.
async fn with_idle<F, T>(timeout: Option<Duration>, fut: F) -> Result<T, Status>
where
    F: std::future::Future<Output = T>,
{
    match timeout {
        None => Ok(fut.await),
        Some(d) => tokio::time::timeout(d, fut)
            .await
            .map_err(|_| Status::deadline_exceeded(format!("stream idle for {}s", d.as_secs()))),
    }
}

/// A timer for a `select!` arm that simply never fires when the bound is off.
async fn sleep_opt(d: Option<Duration>) {
    match d {
        Some(d) => tokio::time::sleep(d).await,
        None => std::future::pending::<()>().await,
    }
}

/// One interim `StreamingRecognizeResponse` (`is_final = false`).
fn partial_response(
    rid: &str,
    text: String,
    tokens: Vec<Vec<u32>>,
    scores: Option<Vec<f32>>,
    max_alts: u32,
) -> pb::StreamingRecognizeResponse {
    pb::StreamingRecognizeResponse {
        results: vec![pb::StreamingRecognitionResult {
            alternatives: build_alternatives(text, tokens, scores, None, max_alts, None),
            is_final: false,
            stability: 0.0,
            result_end_time: None,
            language_code: String::new(),
            finish_reason: String::new(),
        }],
        speech_event_type: pb::SpeechEventType::SpeechEventUnspecified as i32,
        speech_event_offset: None,
        request_id: rid.to_string(),
    }
}

/// A response carrying only a speech event.
///
/// Separate from the transcript responses on purpose: Google's own surface
/// sends the event on its own message with an empty `results`, and folding it
/// onto the next interim would delay a `SPEECH_ACTIVITY_END` until the decoder
/// happened to produce one — which during silence is exactly when it does not.
fn event_response(
    rid: &str,
    event: pb::SpeechEventType,
    offset_s: f32,
) -> pb::StreamingRecognizeResponse {
    pb::StreamingRecognizeResponse {
        results: Vec::new(),
        speech_event_type: event as i32,
        speech_event_offset: Some(duration_from_secs(offset_s)),
        request_id: rid.to_string(),
    }
}

/// Engine speech-activity transitions → Google's event messages.
fn speech_activity_responses(
    rid: &str,
    events: Option<Vec<oasr_wire::SpeechEvent>>,
) -> Vec<pb::StreamingRecognizeResponse> {
    let Some(events) = events else {
        return Vec::new();
    };
    events
        .into_iter()
        .filter_map(|e| match e.kind.as_str() {
            "speech_started" => Some(event_response(
                rid,
                pb::SpeechEventType::SpeechActivityBegin,
                e.time,
            )),
            "speech_stopped" => Some(event_response(
                rid,
                pb::SpeechEventType::SpeechActivityEnd,
                e.time,
            )),
            _ => None,
        })
        .collect()
}

/// The terminal `StreamingRecognizeResponse` (`is_final = true`).
#[allow(clippy::too_many_arguments)]
fn final_response(
    rid: &str,
    text: String,
    tokens: Vec<Vec<u32>>,
    scores: Option<Vec<f32>>,
    nbest_texts: Option<Vec<String>>,
    end_time_s: Option<f32>,
    finish_reason: Option<String>,
    max_alts: u32,
    words: Option<Vec<WordTiming>>,
) -> pb::StreamingRecognizeResponse {
    pb::StreamingRecognizeResponse {
        results: vec![pb::StreamingRecognitionResult {
            alternatives: build_alternatives(text, tokens, scores, nbest_texts, max_alts, words),
            is_final: true,
            stability: 1.0,
            result_end_time: end_time_s.map(duration_from_secs),
            language_code: String::new(),
            finish_reason: finish_reason.unwrap_or_default(),
        }],
        speech_event_type: pb::SpeechEventType::SpeechEventUnspecified as i32,
        speech_event_offset: None,
        request_id: rid.to_string(),
    }
}

/// Map the proto encoding enum to a `(source_encoding, container_hint)` pair.
///
/// The enum's own `as_str_name()` is the bridge to `oasr_asr::parse_encoding`,
/// which both surfaces share — the two used to carry separate `match`es and
/// drifted, so a codec added to one returned `UNIMPLEMENTED` on the other.
///
/// Unsupported codecs return `UNIMPLEMENTED`; `ENCODING_UNSPECIFIED` returns
/// `INVALID_ARGUMENT` (Google STT v1 does the same).
fn map_encoding(enc: i32) -> Result<(SourceEncoding, Option<&'static str>), Status> {
    use pb::recognition_config::AudioEncoding;
    let ae = AudioEncoding::try_from(enc).unwrap_or(AudioEncoding::EncodingUnspecified);
    parse_encoding(ae.as_str_name()).map_err(|e| match e {
        EncodingError::Unspecified => Status::invalid_argument(e.to_string()),
        EncodingError::Unsupported(_) | EncodingError::Unknown(_) => {
            Status::unimplemented(e.to_string())
        }
    })
}

/// The chunked RPC needs a headerless PCM format: a container header arrives
/// once, at the front of a stream whose chunks are decoded independently.
///
/// Rejecting is the honest answer.  Before this, `encoding=WAV` on
/// `StreamingRecognize` silently mapped to raw f32 and the client's 44-byte
/// RIFF header was decoded as eleven samples of noise at the start of every
/// stream.
fn streaming_pcm_encoding(enc: i32) -> Result<oasr_asr::PcmEncoding, Status> {
    match map_encoding(enc)?.0 {
        SourceEncoding::Pcm(p) => Ok(p),
        SourceEncoding::Container => Err(Status::invalid_argument(
            "StreamingRecognize needs a headerless PCM encoding (LINEAR16, \
             LINEAR32F, MULAW or ALAW); a container cannot be fed chunk by \
             chunk. Send containers to the unary Recognize RPC.",
        )),
    }
}

fn map_error(code: ErrorCode, message: String) -> Status {
    match code {
        ErrorCode::Busy => Status::resource_exhausted(message),
        ErrorCode::UnknownRequest => Status::not_found(message),
        ErrorCode::InvalidCmd => Status::invalid_argument(message),
        ErrorCode::Shutdown | ErrorCode::WorkerLost => Status::unavailable(message),
        ErrorCode::Internal => Status::internal(message),
    }
}

/// Log a client-side request rejection at DEBUG and return the `Status`
/// unchanged.  Used for request-validation failures (missing config,
/// unsupported encoding, decode errors) so they can be threaded through
/// `?` / `map_err` without losing observability.
fn log_reject(st: Status) -> Status {
    debug!(code = ?st.code(), reason = st.message(), "grpc request rejected");
    st
}

/// Map the proto `RecognitionConfig` decoding extensions to the engine's
/// per-request [`DecodingParams`].  Returns `Ok(None)` when every knob is at its
/// proto default (0 / empty) so the common path sends nothing, and
/// `INVALID_ARGUMENT` for out-of-range values.
///
/// Validating here rather than letting the Python `DecodingOptions` raise is
/// what keeps a bad value scoped to its own request: bulk admission coalesces
/// many envelopes into one Python call, so a raise there fails the whole batch.
fn decoding_params(cfg: &pb::RecognitionConfig) -> Result<Option<DecodingParams>, Status> {
    // `language_code` is BCP-47 by Google's contract and a primary subtag by
    // the models' — reduce here, and reject junk rather than dropping it, since
    // a dropped language decodes confidently in the checkpoint's own.
    let language = if cfg.language_code.is_empty() {
        None
    } else {
        Some(normalize_language(&cfg.language_code).ok_or_else(|| {
            Status::invalid_argument(format!(
                "language_code {:?} is not a language tag",
                cfg.language_code
            ))
        })?)
    };
    DecodingParams {
        n_best: (cfg.max_alternatives > 1).then_some(cfg.max_alternatives),
        max_new_tokens: (cfg.max_new_tokens > 0).then_some(cfg.max_new_tokens),
        temperature: (cfg.temperature > 0.0).then_some(cfg.temperature),
        top_k: (cfg.top_k > 0).then_some(cfg.top_k),
        top_p: (cfg.top_p > 0.0).then_some(cfg.top_p),
        prompt: (!cfg.prompt.is_empty()).then(|| cfg.prompt.clone()),
        task: (!cfg.task.is_empty()).then(|| cfg.task.trim().to_ascii_lowercase()),
        language,
        word_timestamps: cfg.enable_word_time_offsets.then_some(true),
        // Filled by the streaming caller, which is the only one with a
        // `StreamingRecognitionConfig` to read them from — the unary RPC has no
        // turn to end and no interim stream to annotate.
        single_utterance: None,
        vad_events: None,
        endpoint_silence_ms: None,
    }
    .validated()
    .map_err(Status::invalid_argument)
}

/// Overlay the streaming-only voice-activity controls onto the shared params.
///
/// They live on `StreamingRecognitionConfig` rather than `RecognitionConfig`
/// because they have no meaning for the unary RPC: one buffered utterance has
/// no turn to end and no interim stream to annotate.  `voice_activity_timeout`
/// is *not* forwarded as a decoding option — the engine owns those timers, and
/// they are configured for the process rather than per request; a request that
/// sets one is told so rather than having it dropped.
fn apply_streaming_vad(
    params: Option<DecodingParams>,
    single_utterance: bool,
    voice_activity_events: bool,
    voice_activity_timeout_set: bool,
) -> Result<Option<DecodingParams>, Status> {
    if voice_activity_timeout_set {
        return Err(Status::unimplemented(
            "voice_activity_timeout is configured per process on this server (--vad-option speech_start_timeout_s=... / speech_end_timeout_s=...), not per request",
        ));
    }
    if !single_utterance && !voice_activity_events {
        return Ok(params);
    }
    let mut p = params.unwrap_or_default();
    if single_utterance {
        p.single_utterance = Some(true);
    }
    if voice_activity_events {
        p.vad_events = Some(true);
    }
    p.validated().map_err(Status::invalid_argument)
}

/// Seconds → protobuf `Duration` (audio times are small and non-negative).
fn duration_from_secs(t: f32) -> prost_types::Duration {
    let secs = t.max(0.0);
    prost_types::Duration {
        seconds: secs as i64,
        nanos: ((secs - secs.floor()) * 1e9) as i32,
    }
}

/// Build STT v1 alternatives from a single Final/Partial event payload.
///
/// `text` is the canonical decoded transcript (top hypothesis).  `tokens` is
/// the engine's per-hypothesis token-id list; row 0 aligns with `text`, and
/// rows the engine detokenized (per the request's `max_alternatives`) carry
/// their transcript in `nbest_texts`.
fn build_alternatives(
    text: String,
    tokens: Vec<Vec<u32>>,
    scores: Option<Vec<f32>>,
    nbest_texts: Option<Vec<String>>,
    max_alternatives: u32,
    words: Option<Vec<WordTiming>>,
) -> Vec<pb::SpeechRecognitionAlternative> {
    let cap = if max_alternatives == 0 {
        1
    } else {
        max_alternatives as usize
    };
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
        .map(|(i, ids)| pb::SpeechRecognitionAlternative {
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
            // Only the top alternative is timed; the engine aligns the
            // hypothesis it returns as `text`, not the whole beam.  `take`
            // rather than `clone` so the copy happens once.
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

/// Engine word timing → STT v1 `WordInfo`.
fn word_info(w: WordTiming) -> pb::WordInfo {
    pb::WordInfo {
        start_time: Some(duration_from_secs(w.start)),
        end_time: Some(duration_from_secs(w.end)),
        word: w.word,
        confidence: w.confidence,
    }
}

/// Inputs for [`SpeechService::streaming_over_offline`], grouped so the call site
/// does not take eight positional arguments.
struct StreamOverOfflineCfg {
    /// Chunk decoder, already bound to the client's rate and the model's.  It
    /// carries the resampler's filter state, so it must live for the whole
    /// stream rather than be rebuilt per chunk.
    pcm: PcmStream,
    priority: i32,
    decoding: Option<DecodingParams>,
    want_partials: bool,
    max_alts: u32,
    /// Abort if the client goes this long without sending audio.
    idle_timeout: Option<Duration>,
}

impl SpeechService {
    /// Buffer streaming input for an offline engine, then forward incremental text.
    /// Client half-close triggers submission because fixed-window frontends cannot
    /// process the incomplete utterance.
    async fn streaming_over_offline(
        &self,
        mut inbound: Streaming<pb::StreamingRecognizeRequest>,
        out_tx: mpsc::Sender<Result<pb::StreamingRecognizeResponse, Status>>,
        out_rx: mpsc::Receiver<Result<pb::StreamingRecognizeResponse, Status>>,
        mut cfg: StreamOverOfflineCfg,
    ) -> Result<Response<<Self as pb::speech_server::Speech>::StreamingRecognizeStream>, Status>
    {
        let pool = Arc::clone(&self.pool);
        let span = info_span!("grpc.stream_offline", rid = field::Empty);
        let source_rate = cfg.pcm.source_rate();
        let target_rate = cfg.pcm.target_rate();
        info!(
            parent: &span,
            sample_rate = source_rate,
            model_sample_rate = target_rate,
            resampling = cfg.pcm.is_resampling(),
            want_partials = cfg.want_partials,
            "stream opened (offline engine: buffering audio, streaming text)"
        );

        tokio::spawn(
            async move {
                let start = Instant::now();

                // ---- 1. drain the inbound audio -------------------------------
                let mut buffered: Vec<u8> = Vec::new();
                loop {
                    // A client that opens a stream and then vanishes without
                    // closing the connection would otherwise hold this task —
                    // and, once admitted, an engine slot — forever.
                    let next = match with_idle(cfg.idle_timeout, inbound.next()).await {
                        Ok(n) => n,
                        Err(status) => {
                            warn!(reason = %status, "stream idle timeout before half-close");
                            let _ = out_tx.send(Err(status)).await;
                            return;
                        }
                    };
                    match next {
                        Some(Ok(m)) => match m.streaming_request {
                            Some(
                                pb::streaming_recognize_request::StreamingRequest::AudioContent(
                                    bytes,
                                ),
                            ) => match cfg.pcm.decode_chunk(&bytes) {
                                Ok(samples) => buffered.extend_from_slice(&samples),
                                Err(e) => {
                                    debug!(reason = %e, "stream chunk pcm decode failed");
                                    let _ = out_tx
                                        .send(Err(Status::invalid_argument(format!(
                                            "pcm decode: {e}"
                                        ))))
                                        .await;
                                    return;
                                }
                            },
                            Some(_) => {
                                debug!("ignoring a second streaming_config mid-stream");
                            }
                            None => {
                                let _ = out_tx
                                    .send(Err(Status::invalid_argument("expected audio_content")))
                                    .await;
                                return;
                            }
                        },
                        Some(Err(e)) => {
                            warn!(reason = %e, "stream inbound error");
                            let _ = out_tx
                                .send(Err(Status::internal(format!("inbound: {e}"))))
                                .await;
                            return;
                        }
                        // Half-closed: the utterance is complete.
                        None => break,
                    }
                }

                // Release the frames still inside the resampler; without this the
                // utterance loses its last ~filter-length of audio, which is the
                // final word.
                match cfg.pcm.flush() {
                    Ok(tail) => buffered.extend_from_slice(&tail),
                    Err(e) => {
                        debug!(reason = %e, "stream resampler flush failed");
                        let _ = out_tx
                            .send(Err(Status::internal(format!("resample: {e}"))))
                            .await;
                        return;
                    }
                }

                if buffered.is_empty() {
                    let _ = out_tx
                        .send(Err(Status::invalid_argument(
                            "no audio received before half-close",
                        )))
                        .await;
                    return;
                }
                let n_samples = buffered.len() / 4;

                // ---- 2. submit as one offline request --------------------------
                let mut handle = match pool
                    .submit_offline_streaming(
                        Bytes::from(buffered),
                        target_rate,
                        cfg.priority,
                        cfg.decoding,
                    )
                    .await
                {
                    Ok(h) => h,
                    Err(e) => {
                        warn!(%e, "grpc stream-over-offline submit rejected");
                        let _ = out_tx
                            .send(Err(Status::resource_exhausted(format!(
                                "submit failed: {e}"
                            ))))
                            .await;
                        return;
                    }
                };
                let rid = handle.request_id.clone();
                Span::current().record("rid", rid.as_str());
                debug!(n_samples, "audio buffered; awaiting generation");

                // ---- 3. stream the text out -----------------------------------
                let mut n_partials: u64 = 0;
                loop {
                    // Same bound after half-close: an engine that stops
                    // producing events for this request (wedged, or a lost
                    // terminal) must not park the RPC indefinitely.
                    let ev = match with_idle(cfg.idle_timeout, handle.events.next()).await {
                        Ok(Some(ev)) => ev,
                        Ok(None) => {
                            error!("event stream closed before terminal event");
                            let _ = out_tx
                                .send(Err(Status::internal("event stream closed")))
                                .await;
                            break;
                        }
                        Err(status) => {
                            warn!(rid = %rid, reason = %status, "no decode event within the idle timeout");
                            let _ = out_tx.send(Err(status)).await;
                            break;
                        }
                    };
                    match ev {
                        Event::Partial {
                            text,
                            tokens,
                            scores,
                            ..
                        } => {
                            if !cfg.want_partials {
                                continue;
                            }
                            n_partials += 1;
                            if out_tx
                                .send(Ok(partial_response(
                                    &rid,
                                    text,
                                    tokens,
                                    scores,
                                    cfg.max_alts,
                                )))
                                .await
                                .is_err()
                            {
                                // Client is gone; dropping the handle cancels the
                                // request so the AR row stops burning decode slots.
                                debug!("client dropped mid-generation; cancelling");
                                return;
                            }
                        }
                        Event::Final {
                            text,
                            tokens,
                            scores,
                            nbest_texts,
                            end_time_s,
                            words,
                            finish_reason,
                            ..
                        } => {
                            let transcript = text.clone();
                            let _ = out_tx
                                .send(Ok(final_response(
                                    &rid,
                                    text,
                                    tokens,
                                    scores,
                                    nbest_texts,
                                    end_time_s,
                                    finish_reason,
                                    cfg.max_alts,
                                    words,
                                )))
                                .await;
                            handle.finish();
                            info!(
                                n_partials,
                                n_samples,
                                elapsed_ms = start.elapsed().as_millis() as u64,
                                transcript = %transcript,
                                "stream final"
                            );
                            break;
                        }
                        Event::Error { code, message, .. } => {
                            warn!(
                                code = ?code,
                                elapsed_ms = start.elapsed().as_millis() as u64,
                                reason = %message,
                                "stream error"
                            );
                            let _ = out_tx.send(Err(map_error(code, message))).await;
                            handle.finish();
                            break;
                        }
                        // Accepted / Pong / Overloaded — nothing to forward.
                        _ => {}
                    }
                }
                pool.release(&rid);
            }
            .instrument(span),
        );

        Ok(Response::new(Box::pin(ReceiverStream::new(out_rx))))
    }
}

#[tonic::async_trait]
impl pb::speech_server::Speech for SpeechService {
    async fn recognize(
        &self,
        req: Request<pb::RecognizeRequest>,
    ) -> Result<Response<pb::RecognizeResponse>, Status> {
        // Per-request span; `rid` is recorded once the engine admits the
        // request so all downstream events carry it.
        let span = info_span!("grpc.recognize", rid = field::Empty);
        let rpc_start = Instant::now();
        let result = async move {
            let start = Instant::now();

            if self.mode != ServiceMode::Offline {
                return Err(log_reject(Status::failed_precondition(
                    "server is running in streaming mode; use StreamingRecognize",
                )));
            }

            let pb::RecognizeRequest { config, audio } = req.into_inner();
            let cfg =
                config.ok_or_else(|| log_reject(Status::invalid_argument("config required")))?;
            let max_alts = cfg.max_alternatives;
            let decoding = decoding_params(&cfg).map_err(log_reject)?;

            let audio_bytes = match audio.and_then(|a| a.audio_source) {
                Some(pb::recognition_audio::AudioSource::Content(b)) => b,
                Some(pb::recognition_audio::AudioSource::Uri(_)) => {
                    return Err(log_reject(Status::unimplemented("audio.uri is not supported")));
                }
                None => return Err(log_reject(Status::invalid_argument("audio.content required"))),
            };

            let sr = if cfg.sample_rate_hertz == 0 {
                self.sample_rate
            } else {
                cfg.sample_rate_hertz
            };
            let (source_enc, ct_hint) = map_encoding(cfg.encoding).map_err(log_reject)?;
            // Convert to the engine's rate before submitting: the engine derives
            // every frame count from its own feature config and ignores the
            // request's rate, so unconverted 8 kHz / 44.1 kHz audio decoded to a
            // confident, wrong transcript.
            let decoded = decode_audio(
                &audio_bytes,
                &DecodeOptions {
                    hint: ct_hint,
                    encoding: source_enc,
                    source_sample_rate: Some(sr),
                    target_sample_rate: Some(self.sample_rate),
                    max_samples: self.max_audio_samples,
                },
            )
            .map_err(|e| log_reject(Status::invalid_argument(format!("audio decode: {e}"))))?;
            if decoded.sample_rate != sr {
                debug!(
                    from_hz = sr,
                    to_hz = decoded.sample_rate,
                    "resampled request audio to the model rate"
                );
            }

            let sample_rate = decoded.sample_rate;
            let n_samples = decoded.samples.len() / 4;
            let audio_seconds = f32_pcm_seconds(decoded.samples.len(), sample_rate);
            grpc_unary().audio_ingested(audio_seconds);
            grpc_unary().audio_duration(audio_seconds);
            let handle = self
                .pool
                .submit_offline(decoded.samples, sample_rate, cfg.priority, decoding)
                .await
                .map_err(|e| {
                    warn!(%e, "grpc recognize submit rejected");
                    Status::resource_exhausted(format!("submit failed: {e}"))
                })?;
            let rid = handle.request_id.clone();
            Span::current().record("rid", rid.as_str());
            let ev = handle.finish().await.map_err(|_| {
                error!(rid = %rid, "engine channel closed");
                Status::internal("engine channel closed")
            })?;
            self.pool.release(&rid);

            let elapsed_ms = start.elapsed().as_millis() as u64;
            match ev {
                Event::Final {
                    request_id,
                    text,
                    tokens,
                    scores,
                    nbest_texts,
                    end_time_s,
                    words,
                    finish_reason,
                    ..
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
                    Ok(Response::new(pb::RecognizeResponse {
                        results: vec![pb::SpeechRecognitionResult {
                            alternatives: build_alternatives(
                                text,
                                tokens,
                                scores,
                                nbest_texts,
                                max_alts,
                                words,
                            ),
                            channel_tag: 0,
                            finish_reason: finish_reason.unwrap_or_default(),
                            result_end_time: end_time_s.map(duration_from_secs),
                            language_code: String::new(),
                        }],
                        request_id,
                    }))
                }
                Event::Error { code, message, .. } => {
                    warn!(rid = %rid, code = ?code, elapsed_ms, reason = %message, "recognize error");
                    Err(map_error(code, message))
                }
                other => {
                    error!(rid = %rid, elapsed_ms, "unexpected non-terminal event for offline request: {other:?}");
                    Err(Status::internal("unexpected event type"))
                }
            }
        }
        .instrument(span)
        .await;

        // Recorded here rather than in a tower layer: this is where the RPC's
        // terminal status actually exists.
        let code = code_of(&result);
        let elapsed = rpc_start.elapsed();
        record_rpc(method::RECOGNIZE, code, elapsed);
        grpc_unary().finished(outcome_for(code), elapsed);
        result
    }

    type StreamingRecognizeStream =
        Pin<Box<dyn Stream<Item = Result<pb::StreamingRecognizeResponse, Status>> + Send>>;

    async fn streaming_recognize(
        &self,
        req: Request<Streaming<pb::StreamingRecognizeRequest>>,
    ) -> Result<Response<Self::StreamingRecognizeStream>, Status> {
        let mut inbound = req.into_inner();
        let (out_tx, out_rx) = mpsc::channel::<Result<pb::StreamingRecognizeResponse, Status>>(64);

        // First inbound MUST carry streaming_config.
        let first = inbound
            .next()
            .await
            .ok_or_else(|| {
                log_reject(Status::invalid_argument(
                    "missing streaming_config first message",
                ))
            })?
            .map_err(|e| log_reject(Status::internal(format!("stream recv: {e}"))))?;

        let scfg = match first.streaming_request {
            Some(pb::streaming_recognize_request::StreamingRequest::StreamingConfig(c)) => c,
            _ => {
                return Err(log_reject(Status::invalid_argument(
                    "first message must carry streaming_config",
                )))
            }
        };
        // Read the streaming-only controls before `config` is moved out of
        // `scfg`: they live on the outer message, and taking the inner one
        // partially moves it.
        let want_events = scfg.enable_voice_activity_events;
        let want_single = scfg.single_utterance;
        let vad_timeout_set = scfg.voice_activity_timeout.is_some();
        let rcfg = scfg
            .config
            .ok_or_else(|| log_reject(Status::invalid_argument("missing recognition config")))?;
        // An unset rate means "the model's own" — the only value that could ever
        // have worked before resampling existed.
        let sr = if rcfg.sample_rate_hertz == 0 {
            self.sample_rate
        } else {
            rcfg.sample_rate_hertz
        };
        let pcm_enc = streaming_pcm_encoding(rcfg.encoding).map_err(log_reject)?;
        // One decoder for the whole stream: it holds the resampler's filter
        // state, so per-chunk construction would stamp a discontinuity into the
        // waveform at every chunk boundary.  Built here, before anything is
        // admitted, so an implausible `sample_rate_hertz` is rejected at open.
        let mut pcm = PcmStream::new(pcm_enc, sr, self.sample_rate)
            .map_err(|e| log_reject(Status::invalid_argument(format!("audio config: {e}"))))?;
        let want_partials = scfg.interim_results;
        let max_alts = rcfg.max_alternatives;
        let decoding = decoding_params(&rcfg).map_err(log_reject)?;
        let decoding = apply_streaming_vad(decoding, want_single, want_events, vad_timeout_set)
            .map_err(log_reject)?;

        // An offline-pinned engine cannot take CreateStreaming + FeedChunk — the
        // decode families it serves (`aed`, `llm`, `paraformer`,
        // `ctc_aed_rescoring`) need the whole utterance before they can start, and
        // `whisper_logmel` normalises over a fixed window.  But that is a
        // constraint on *audio in*, not on *text out*: those same families emit one
        // partial per engine tick, which is exactly the token-streaming UX a
        // speech-LLM client wants.  So serve the RPC by buffering the inbound audio
        // and streaming the generated text back.
        if self.mode == ServiceMode::Offline {
            return self
                .streaming_over_offline(
                    inbound,
                    out_tx,
                    out_rx,
                    StreamOverOfflineCfg {
                        pcm,
                        priority: rcfg.priority,
                        decoding,
                        want_partials,
                        max_alts,
                        idle_timeout: self.stream_idle_timeout,
                    },
                )
                .await;
        }

        // The engine is told the rate it will actually receive — the model's —
        // not the client's, because `pcm` converts on the way in.
        let mut handle = self
            .pool
            .open_streaming(self.sample_rate, rcfg.priority, decoding)
            .await
            .map_err(|e| {
                warn!(%e, "grpc streaming open rejected");
                Status::resource_exhausted(format!("submit failed: {e}"))
            })?;

        let pool = Arc::clone(&self.pool);
        let rid = handle.request_id.clone();

        // Per-request span carrying `rid`; the streaming work runs in the
        // spawned task below, instrumented with this span so every per-chunk
        // / per-event log line is correlated.
        let span = info_span!("grpc.stream", rid = %rid);
        info!(
            parent: &span,
            sample_rate = sr,
            model_sample_rate = self.sample_rate,
            resampling = pcm.is_resampling(),
            want_partials,
            "stream opened"
        );

        let idle_timeout = self.stream_idle_timeout;
        let engine_rate = self.sample_rate;
        tokio::spawn(async move {
            let start = Instant::now();
            let mut n_partials: u64 = 0;
            // Time to first partial is clocked from the first inbound audio,
            // not from the RPC opening: a client may open the stream and then
            // stay silent, and timing that silence would report the caller's
            // behaviour as the server's latency.
            let mut first_audio_at: Option<Instant> = None;
            let mut ttfp_recorded = false;
            // Accumulated so the per-request audio duration can be recorded at
            // the end: a live stream has no total until it closes, while the
            // RTFx denominator has to accrue as chunks arrive.
            let mut audio_seconds = 0.0f64;
            // The status this RPC really ends with, which a middleware cannot
            // see: it rides out in the trailers after this task finishes.
            // Named `rpc_code` because `code` is bound by the `Event::Error`
            // pattern below — that is the engine's error code, not the RPC's.
            let mut rpc_code = Code::Ok;
            // ``inbound_done`` flips once the client half-closes so we stop
            // polling the inbound stream — otherwise tokio::select! keeps
            // racing on a fused-None stream and we'd call ``flush_last``
            // repeatedly, which the engine rejects with "feed_chunk after
            // is_last=True".
            let mut inbound_done = false;
            // Reset on every event or chunk; firing means neither side has
            // moved, so the RPC is holding an engine slot for nobody.
            let mut idle = std::pin::pin!(sleep_opt(idle_timeout));
            loop {
                tokio::select! {
                    _ = &mut idle => {
                        warn!(
                            rid = %rid,
                            elapsed_ms = start.elapsed().as_millis() as u64,
                            "stream idle timeout; cancelling"
                        );
                        let _ = out_tx.send(Err(Status::deadline_exceeded("stream idle"))).await;
                        rpc_code = Code::DeadlineExceeded;
                        // Fall through to `pool.release`; dropping `handle`
                        // un-disarmed cancels the request engine-side.
                        break;
                    }
                    ev = handle.events.next() => {
                        idle.set(sleep_opt(idle_timeout));
                        match ev {
                            Some(Event::Partial { text, tokens, scores, speech_events, .. }) => {
                                if !ttfp_recorded {
                                    if let Some(t0) = first_audio_at {
                                        ttfp_recorded = true;
                                        grpc_streaming().first_partial(t0.elapsed());
                                    }
                                }
                                if want_events {
                                    for resp in speech_activity_responses(&rid, speech_events) {
                                        let _ = out_tx.send(Ok(resp)).await;
                                    }
                                }
                                // The filter is about what this client wants to
                                // see; the SLI is about what the engine
                                // produced, so it is timed above the filter.
                                if !want_partials { continue; }
                                n_partials += 1;
                                let resp = partial_response(&rid, text, tokens, scores, max_alts);
                                let _ = out_tx.send(Ok(resp)).await;
                            }
                            Some(Event::Final {
                                text, tokens, scores, nbest_texts, end_time_s, words,
                                finish_reason, speech_events, endpoint_reason, ..
                            }) => {
                                let transcript = text.clone();
                                let ended_at = end_time_s.unwrap_or(0.0);
                                if want_events {
                                    for resp in speech_activity_responses(&rid, speech_events) {
                                        let _ = out_tx.send(Ok(resp)).await;
                                    }
                                }
                                let resp = final_response(
                                    &rid, text, tokens, scores, nbest_texts, end_time_s,
                                    finish_reason, max_alts, words,
                                );
                                let _ = out_tx.send(Ok(resp)).await;
                                // The event the enum has declared since this
                                // surface landed, and that nothing ever sent.
                                // After the final result, matching Google: the
                                // transcript for the utterance comes first, then
                                // the notice that the utterance is over.
                                if want_single && endpoint_reason.is_some() {
                                    let _ = out_tx
                                        .send(Ok(event_response(
                                            &rid,
                                            pb::SpeechEventType::EndOfSingleUtterance,
                                            ended_at,
                                        )))
                                        .await;
                                }
                                handle.finish();
                                info!(
                                    n_partials,
                                    elapsed_ms = start.elapsed().as_millis() as u64,
                                    transcript = %transcript,
                                    "stream final"
                                );
                                break;
                            }
                            Some(Event::Error { code, message, .. }) => {
                                warn!(
                                    code = ?code,
                                    elapsed_ms = start.elapsed().as_millis() as u64,
                                    reason = %message,
                                    "stream error"
                                );
                                let status = map_error(code, message);
                                let terminal = status.code();
                                let _ = out_tx.send(Err(status)).await;
                                handle.finish();
                                rpc_code = terminal;
                                break;
                            }
                            Some(_) => {} // Accepted / Pong / Overloaded — ignored at this layer.
                            None => {
                                error!("event stream closed before terminal event");
                                let _ = out_tx.send(Err(Status::internal("event stream closed"))).await;
                                rpc_code = Code::Internal;
                                break;
                            }
                        }
                    }
                    msg = inbound.next(), if !inbound_done => {
                        idle.set(sleep_opt(idle_timeout));
                        match msg {
                            Some(Ok(m)) => {
                                if let Some(pb::streaming_recognize_request::StreamingRequest::AudioContent(bytes)) = m.streaming_request {
                                    let chunk = match pcm.decode_chunk(&bytes) {
                                        Ok(samples) => samples,
                                        Err(e) => {
                                            debug!(reason = %e, "stream chunk pcm decode failed");
                                            let _ = out_tx.send(Err(Status::invalid_argument(format!("pcm decode: {e}")))).await;
                                            continue;
                                        }
                                    };
                                    // A resampling stream can hold a chunk back
                                    // inside the filter; feeding an empty chunk
                                    // would just cost the engine a step.
                                    if chunk.is_empty() {
                                        continue;
                                    }
                                    first_audio_at.get_or_insert_with(Instant::now);
                                    let seconds = f32_pcm_seconds(chunk.len(), engine_rate);
                                    audio_seconds += seconds;
                                    grpc_streaming().audio_ingested(seconds);
                                    if handle.push_chunk(chunk).await.is_err() {
                                        warn!("grpc bidi: audio channel dropped");
                                        rpc_code = Code::Internal;
                                        break;
                                    }
                                } else {
                                    debug!("stream chunk missing audio_content");
                                    let _ = out_tx.send(Err(Status::invalid_argument("expected audio_content"))).await;
                                }
                            }
                            Some(Err(e)) => {
                                warn!(reason = %e, "stream inbound error");
                                let _ = out_tx.send(Err(Status::internal(format!("inbound: {e}")))).await;
                                rpc_code = Code::Internal;
                                break;
                            }
                            None => {
                                // Client half-closed: send is_last once and
                                // stop polling the inbound stream.  Keep
                                // draining events until Final / Error.
                                // The resampler's tail rides out on the final
                                // chunk; dropping it would cut the last word.
                                inbound_done = true;
                                let tail = pcm.flush().unwrap_or_else(|e| {
                                    warn!(reason = %e, "resampler flush failed; dropping tail");
                                    Bytes::new()
                                });
                                debug!(tail_samples = tail.len() / 4, "client half-closed; flushing final chunk");
                                let _ = handle.flush_last(tail).await;
                            }
                        }
                    }
                }
            }
            pool.release(&rid);
            let elapsed = start.elapsed();
            record_rpc(method::STREAMING_RECOGNIZE, rpc_code, elapsed);
            grpc_streaming().audio_duration(audio_seconds);
            grpc_streaming().finished(outcome_for(rpc_code), elapsed);
        }.instrument(span));

        let out_stream = ReceiverStream::new(out_rx);
        Ok(Response::new(Box::pin(out_stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These tests pin the field values that distinguish an
    /// interim response from a terminal one — a client renders on `is_final`, so
    /// getting it wrong turns a partial into a premature answer.
    #[test]
    fn partial_response_is_not_final() {
        let r = partial_response("rid-1", "hello".into(), vec![vec![1, 2]], None, 1);
        let res = &r.results[0];
        assert!(!res.is_final);
        assert_eq!(res.stability, 0.0);
        assert!(res.result_end_time.is_none());
        assert_eq!(r.request_id, "rid-1");
        assert_eq!(res.alternatives[0].transcript, "hello");
    }

    #[test]
    fn final_response_is_final_and_carries_end_time() {
        let r = final_response(
            "rid-2",
            "hello world".into(),
            vec![vec![1, 2, 3]],
            None,
            None,
            Some(1.5),
            Some("length".into()),
            1,
            None,
        );
        let res = &r.results[0];
        assert!(res.is_final);
        assert_eq!(res.stability, 1.0);
        let d = res.result_end_time.as_ref().expect("end time");
        assert_eq!(d.seconds, 1);
        assert_eq!(d.nanos, 500_000_000);
        // A truncated transcript must be distinguishable from a complete one.
        assert_eq!(res.finish_reason, "length");
    }

    /// The one-shot families set no `finish_reason`; proto3 has no optional
    /// string, so "absent" is the empty string — never the word "stop".
    #[test]
    fn final_response_without_a_finish_reason_leaves_it_empty() {
        let r = final_response(
            "rid-4",
            "x".into(),
            vec![vec![1]],
            None,
            None,
            None,
            None,
            1,
            None,
        );
        assert_eq!(r.results[0].finish_reason, "");
    }

    /// Interim results are mid-generation by definition.
    #[test]
    fn partial_response_never_carries_a_finish_reason() {
        let r = partial_response("rid-5", "partial".into(), vec![vec![1]], None, 1);
        assert_eq!(r.results[0].finish_reason, "");
        assert!(!r.results[0].is_final);
    }

    /// A partial must never advertise alternatives it has no transcript for:
    /// `nbest_texts` is `None` until the final, so a client asking for
    /// `max_alternatives > 1` would otherwise see empty-transcript rows mid-stream.
    #[test]
    fn partial_with_max_alternatives_does_not_invent_transcripts() {
        let r = partial_response(
            "rid-3",
            "top".into(),
            vec![vec![1], vec![2], vec![3]],
            None,
            3,
        );
        let alts = &r.results[0].alternatives;
        assert_eq!(alts.len(), 3, "rows are still reported");
        assert_eq!(alts[0].transcript, "top");
        assert!(
            alts[1].transcript.is_empty() && alts[2].transcript.is_empty(),
            "non-top rows have no detokenized text until the final"
        );
    }

    /// Only the transcript the caller is shown is timed.  Copying the word
    /// list onto every alternative would attach one hypothesis's clock to
    /// another's text.
    #[test]
    fn words_ride_only_on_the_top_alternative() {
        let words = vec![
            WordTiming {
                word: "hello".into(),
                start: 0.25,
                end: 0.75,
                confidence: 0.9,
            },
            WordTiming {
                word: "world".into(),
                start: 0.75,
                end: 1.25,
                confidence: 0.8,
            },
        ];
        let alts = build_alternatives(
            "hello world".into(),
            vec![vec![1], vec![2]],
            Some(vec![-1.0, -2.0]),
            Some(vec!["hello world".into(), "hello word".into()]),
            2,
            Some(words),
        );
        assert_eq!(alts.len(), 2);
        assert_eq!(alts[0].words.len(), 2);
        assert!(alts[1].words.is_empty(), "alternatives are not timed");
        let w = &alts[0].words[0];
        assert_eq!(w.word, "hello");
        assert_eq!(w.confidence, 0.9);
        let start = w.start_time.as_ref().expect("start");
        assert_eq!((start.seconds, start.nanos), (0, 250_000_000));
    }

    /// A request that did not ask leaves the field empty, not zero-filled.
    #[test]
    fn no_words_means_an_empty_list() {
        let alts = build_alternatives("hi".into(), vec![vec![1]], None, None, 1, None);
        assert!(alts[0].words.is_empty());
    }

    #[test]
    fn enable_word_time_offsets_maps_to_the_engine_option() {
        let cfg = pb::RecognitionConfig {
            enable_word_time_offsets: true,
            ..Default::default()
        };
        let params = decoding_params(&cfg).expect("valid").expect("some");
        assert_eq!(params.word_timestamps, Some(true));
        // ...and stays unset otherwise, so the engine sees no options dict at all.
        assert!(decoding_params(&pb::RecognitionConfig::default())
            .expect("valid")
            .is_none());
    }

    #[test]
    fn max_alternatives_zero_means_one() {
        let r = final_response(
            "r",
            "x".into(),
            vec![vec![1], vec![2]],
            None,
            None,
            None,
            None,
            0,
            None,
        );
        assert_eq!(r.results[0].alternatives.len(), 1);
    }

    #[test]
    fn service_mode_parses_both_and_rejects_junk() {
        use std::str::FromStr;
        assert_eq!(
            ServiceMode::from_str("streaming").unwrap(),
            ServiceMode::Streaming
        );
        assert_eq!(
            ServiceMode::from_str("offline").unwrap(),
            ServiceMode::Offline
        );
        assert!(ServiceMode::from_str("both").is_err());
    }
}
