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
        if self.mode != ServiceMode::Streaming {
            return Err(log_reject(Status::failed_precondition(
                "server is running in offline mode; use Recognize",
            )));
        }

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
                                let resp = pb::StreamingRecognizeResponse {
                                    results: vec![pb::StreamingRecognitionResult {
                                        alternatives: build_alternatives(text, tokens, scores, None, max_alts),
                                        is_final: false,
                                        stability: 0.0,
                                        result_end_time: None,
                                        language_code: String::new(),
                                    }],
                                    speech_event_type: pb::SpeechEventType::SpeechEventUnspecified as i32,
                                    request_id: rid.clone(),
                                };
                                let _ = out_tx.send(Ok(resp)).await;
                            }
                            Some(Event::Final { text, tokens, scores, nbest_texts, end_time_s, .. }) => {
                                let transcript = text.clone();
                                let resp = pb::StreamingRecognizeResponse {
                                    results: vec![pb::StreamingRecognitionResult {
                                        alternatives: build_alternatives(text, tokens, scores, nbest_texts, max_alts),
                                        is_final: true,
                                        stability: 1.0,
                                        result_end_time: end_time_s.map(duration_from_secs),
                                        language_code: String::new(),
                                    }],
                                    speech_event_type: pb::SpeechEventType::SpeechEventUnspecified as i32,
                                    request_id: rid.clone(),
                                };
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
