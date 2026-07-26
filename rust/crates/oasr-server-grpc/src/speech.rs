// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! `oasr.speech.v1.Speech` service implementation.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use futures::Stream;
use oasr_asr::{decode_audio, decode_raw_pcm, PcmEncoding};
use oasr_engine_client::EnginePool;
use oasr_wire::{score_posteriors, DecodingParams, ErrorCode, Event};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, error, field, info, info_span, warn, Instrument, Span};

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
}

impl SpeechService {
    pub fn new(pool: Arc<EnginePool>, mode: ServiceMode) -> Self {
        Self { pool, mode }
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
            alternatives: build_alternatives(text, tokens, scores, None, max_alts),
            is_final: false,
            stability: 0.0,
            result_end_time: None,
            language_code: String::new(),
            finish_reason: String::new(),
        }],
        speech_event_type: pb::SpeechEventType::SpeechEventUnspecified as i32,
        request_id: rid.to_string(),
    }
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
) -> pb::StreamingRecognizeResponse {
    pb::StreamingRecognizeResponse {
        results: vec![pb::StreamingRecognitionResult {
            alternatives: build_alternatives(text, tokens, scores, nbest_texts, max_alts),
            is_final: true,
            stability: 1.0,
            result_end_time: end_time_s.map(duration_from_secs),
            language_code: String::new(),
            finish_reason: finish_reason.unwrap_or_default(),
        }],
        speech_event_type: pb::SpeechEventType::SpeechEventUnspecified as i32,
        request_id: rid.to_string(),
    }
}

/// Map the proto encoding enum to a `(pcm_encoding, content_type_hint)` pair.
///
/// Unsupported codecs return `UNIMPLEMENTED`; `ENCODING_UNSPECIFIED` returns
/// `INVALID_ARGUMENT` (Google STT v1 does the same).
fn map_encoding(enc: i32) -> Result<(PcmEncoding, Option<&'static str>), Status> {
    use pb::recognition_config::AudioEncoding;
    let ae = AudioEncoding::try_from(enc).unwrap_or(AudioEncoding::EncodingUnspecified);
    match ae {
        AudioEncoding::EncodingUnspecified => Err(Status::invalid_argument("encoding must be set")),
        AudioEncoding::Linear16 => Ok((PcmEncoding::I16Le, None)),
        AudioEncoding::Linear32f => Ok((PcmEncoding::F32Le, None)),
        AudioEncoding::Wav => Ok((PcmEncoding::F32Le, Some("audio/wav"))),
        other => Err(Status::unimplemented(format!(
            "encoding {other:?} not supported"
        ))),
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
    DecodingParams {
        n_best: (cfg.max_alternatives > 1).then_some(cfg.max_alternatives),
        max_new_tokens: (cfg.max_new_tokens > 0).then_some(cfg.max_new_tokens),
        temperature: (cfg.temperature > 0.0).then_some(cfg.temperature),
        top_k: (cfg.top_k > 0).then_some(cfg.top_k),
        top_p: (cfg.top_p > 0.0).then_some(cfg.top_p),
        prompt: (!cfg.prompt.is_empty()).then(|| cfg.prompt.clone()),
    }
    .validated()
    .map_err(Status::invalid_argument)
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
            tokens: ids,
        })
        .collect()
}

/// Inputs for [`SpeechService::streaming_over_offline`], grouped so the call site
/// does not take eight positional arguments.
struct StreamOverOfflineCfg {
    sample_rate: u32,
    pcm_enc: PcmEncoding,
    priority: i32,
    decoding: Option<DecodingParams>,
    want_partials: bool,
    max_alts: u32,
}

impl SpeechService {
    /// `StreamingRecognize` against an **offline-pinned** engine: buffer the
    /// inbound audio, submit it as one offline request, stream the text out.
    ///
    /// This is what makes the autoregressive strategies' per-tick partials
    /// reachable. They were built (`llm` emits one `Event::Partial` per advanced
    /// request per tick), they crossed PyO3, they reached the router — and then the
    /// unary path's drain task threw every non-terminal event away, one layer from
    /// the wire. The inter-token cadence a client sees here is set by
    /// `EngineConfig.max_tick_ms`, which is why that knob is a user-visible latency
    /// feature and not merely an internal bound on GIL hold time.
    ///
    /// Half-close is the submit trigger: there is nothing useful to do with a
    /// partial utterance when the frontend has a fixed window. A client that never
    /// half-closes gets no result, same as a unary client that never finishes its
    /// request body.
    async fn streaming_over_offline(
        &self,
        mut inbound: Streaming<pb::StreamingRecognizeRequest>,
        out_tx: mpsc::Sender<Result<pb::StreamingRecognizeResponse, Status>>,
        out_rx: mpsc::Receiver<Result<pb::StreamingRecognizeResponse, Status>>,
        cfg: StreamOverOfflineCfg,
    ) -> Result<Response<<Self as pb::speech_server::Speech>::StreamingRecognizeStream>, Status>
    {
        let pool = Arc::clone(&self.pool);
        let span = info_span!("grpc.stream_offline", rid = field::Empty);
        info!(
            parent: &span,
            sample_rate = cfg.sample_rate,
            want_partials = cfg.want_partials,
            "stream opened (offline engine: buffering audio, streaming text)"
        );

        tokio::spawn(
            async move {
                let start = Instant::now();

                // ---- 1. drain the inbound audio -------------------------------
                let mut buffered: Vec<u8> = Vec::new();
                loop {
                    match inbound.next().await {
                        Some(Ok(m)) => match m.streaming_request {
                            Some(
                                pb::streaming_recognize_request::StreamingRequest::AudioContent(
                                    bytes,
                                ),
                            ) => match decode_raw_pcm(&bytes, cfg.pcm_enc, cfg.sample_rate) {
                                Ok(d) => buffered.extend_from_slice(&d.samples),
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
                        cfg.sample_rate,
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
                while let Some(ev) = handle.events.next().await {
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
        async move {
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
                16_000
            } else {
                cfg.sample_rate_hertz
            };
            let (pcm_enc, ct_hint) = map_encoding(cfg.encoding).map_err(log_reject)?;
            let decoded = decode_audio(ct_hint, &audio_bytes, pcm_enc, Some(sr))
                .map_err(|e| log_reject(Status::invalid_argument(format!("audio decode: {e}"))))?;

            let sample_rate = decoded.sample_rate;
            let n_samples = decoded.samples.len() / 4;
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
                    Ok(Response::new(pb::RecognizeResponse {
                        results: vec![pb::SpeechRecognitionResult {
                            alternatives: build_alternatives(
                                text,
                                tokens,
                                scores,
                                nbest_texts,
                                max_alts,
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
        .await
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
        let rcfg = scfg
            .config
            .ok_or_else(|| log_reject(Status::invalid_argument("missing recognition config")))?;
        let sr = if rcfg.sample_rate_hertz == 0 {
            16_000
        } else {
            rcfg.sample_rate_hertz
        };
        let (pcm_enc, _ct_hint) = map_encoding(rcfg.encoding).map_err(log_reject)?;
        let want_partials = scfg.interim_results;
        let max_alts = rcfg.max_alternatives;
        let decoding = decoding_params(&rcfg).map_err(log_reject)?;

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
                        sample_rate: sr,
                        pcm_enc,
                        priority: rcfg.priority,
                        decoding,
                        want_partials,
                        max_alts,
                    },
                )
                .await;
        }

        let mut handle = self
            .pool
            .open_streaming(sr, rcfg.priority, decoding)
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
        info!(parent: &span, sample_rate = sr, want_partials, "stream opened");

        tokio::spawn(async move {
            let start = Instant::now();
            let mut n_partials: u64 = 0;
            // ``inbound_done`` flips once the client half-closes so we stop
            // polling the inbound stream — otherwise tokio::select! keeps
            // racing on a fused-None stream and we'd call ``flush_last``
            // repeatedly, which the engine rejects with "feed_chunk after
            // is_last=True".
            let mut inbound_done = false;
            loop {
                tokio::select! {
                    ev = handle.events.next() => {
                        match ev {
                            Some(Event::Partial { text, tokens, scores, .. }) => {
                                if !want_partials { continue; }
                                n_partials += 1;
                                let resp = partial_response(&rid, text, tokens, scores, max_alts);
                                let _ = out_tx.send(Ok(resp)).await;
                            }
                            Some(Event::Final {
                                text, tokens, scores, nbest_texts, end_time_s, finish_reason, ..
                            }) => {
                                let transcript = text.clone();
                                let resp = final_response(
                                    &rid, text, tokens, scores, nbest_texts, end_time_s,
                                    finish_reason, max_alts,
                                );
                                let _ = out_tx.send(Ok(resp)).await;
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
                                let _ = out_tx.send(Err(map_error(code, message))).await;
                                handle.finish();
                                break;
                            }
                            Some(_) => {} // Accepted / Pong / Overloaded — ignored at this layer.
                            None => {
                                error!("event stream closed before terminal event");
                                let _ = out_tx.send(Err(Status::internal("event stream closed"))).await;
                                break;
                            }
                        }
                    }
                    msg = inbound.next(), if !inbound_done => {
                        match msg {
                            Some(Ok(m)) => {
                                if let Some(pb::streaming_recognize_request::StreamingRequest::AudioContent(bytes)) = m.streaming_request {
                                    let chunk = match decode_raw_pcm(&bytes, pcm_enc, sr) {
                                        Ok(d) => d.samples,
                                        Err(e) => {
                                            debug!(reason = %e, "stream chunk pcm decode failed");
                                            let _ = out_tx.send(Err(Status::invalid_argument(format!("pcm decode: {e}")))).await;
                                            continue;
                                        }
                                    };
                                    if handle.push_chunk(chunk).await.is_err() {
                                        warn!("grpc bidi: audio channel dropped");
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
                                break;
                            }
                            None => {
                                // Client half-closed: send is_last once and
                                // stop polling the inbound stream.  Keep
                                // draining events until Final / Error.
                                inbound_done = true;
                                debug!("client half-closed; flushing final chunk");
                                let _ = handle.flush_last(Bytes::new()).await;
                            }
                        }
                    }
                }
            }
            pool.release(&rid);
        }.instrument(span));

        let out_stream = ReceiverStream::new(out_rx);
        Ok(Response::new(Box::pin(out_stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The response builders were factored out of the streaming loop so the
    /// offline-engine path (S1) could reuse them instead of duplicating ~40 lines
    /// of proto construction.  These pin the field values that distinguish an
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
