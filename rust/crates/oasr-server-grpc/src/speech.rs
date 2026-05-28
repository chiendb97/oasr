// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! `oasr.speech.v1.Speech` service implementation.

use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;
use futures::Stream;
use oasr_asr::{decode_audio, decode_raw_pcm, PcmEncoding};
use oasr_engine_client::EnginePool;
use oasr_wire::{ErrorCode, Event};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};
use tracing::{error, warn};

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
        AudioEncoding::EncodingUnspecified => {
            Err(Status::invalid_argument("encoding must be set"))
        }
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

/// Build STT v1 alternatives from a single Final/Partial event payload.
///
/// `text` is the canonical decoded transcript (top hypothesis).  `tokens` is
/// the engine's per-hypothesis token-id list; row 0 aligns with `text`,
/// additional rows surface as alternatives with empty transcripts.
fn build_alternatives(
    text: String,
    tokens: Vec<Vec<u32>>,
    scores: Option<Vec<f32>>,
    max_alternatives: u32,
) -> Vec<pb::SpeechRecognitionAlternative> {
    let cap = if max_alternatives == 0 {
        1
    } else {
        max_alternatives as usize
    };
    let rows = if tokens.is_empty() { vec![Vec::new()] } else { tokens };
    rows.into_iter()
        .take(cap)
        .enumerate()
        .map(|(i, ids)| pb::SpeechRecognitionAlternative {
            transcript: if i == 0 { text.clone() } else { String::new() },
            confidence: scores.as_ref().and_then(|s| s.get(i).copied()).unwrap_or(0.0),
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
        if self.mode != ServiceMode::Offline {
            return Err(Status::failed_precondition(
                "server is running in streaming mode; use StreamingRecognize",
            ));
        }

        let pb::RecognizeRequest { config, audio } = req.into_inner();
        let cfg = config.ok_or_else(|| Status::invalid_argument("config required"))?;
        let max_alts = cfg.max_alternatives;

        let audio_bytes = match audio.and_then(|a| a.audio_source) {
            Some(pb::recognition_audio::AudioSource::Content(b)) => b,
            Some(pb::recognition_audio::AudioSource::Uri(_)) => {
                return Err(Status::unimplemented("audio.uri is not supported"));
            }
            None => return Err(Status::invalid_argument("audio.content required")),
        };

        let sr = if cfg.sample_rate_hertz == 0 {
            16_000
        } else {
            cfg.sample_rate_hertz
        };
        let (pcm_enc, ct_hint) = map_encoding(cfg.encoding)?;
        let decoded = decode_audio(ct_hint, &audio_bytes, pcm_enc, Some(sr))
            .map_err(|e| Status::invalid_argument(format!("audio decode: {e}")))?;

        let handle = self
            .pool
            .submit_offline(decoded.samples, decoded.sample_rate, cfg.priority)
            .await
            .map_err(|e| Status::resource_exhausted(format!("submit failed: {e}")))?;
        let rid = handle.request_id.clone();
        let ev = handle
            .finish()
            .await
            .map_err(|_| Status::internal("engine channel closed"))?;
        self.pool.release(&rid);

        match ev {
            Event::Final {
                request_id,
                text,
                tokens,
                scores,
            } => Ok(Response::new(pb::RecognizeResponse {
                results: vec![pb::SpeechRecognitionResult {
                    alternatives: build_alternatives(text, tokens, scores, max_alts),
                    channel_tag: 0,
                    result_end_time: None,
                    language_code: String::new(),
                }],
                request_id,
            })),
            Event::Error { code, message, .. } => Err(map_error(code, message)),
            other => {
                error!("unexpected non-terminal event for offline rid: {other:?}");
                Err(Status::internal("unexpected event type"))
            }
        }
    }

    type StreamingRecognizeStream =
        Pin<Box<dyn Stream<Item = Result<pb::StreamingRecognizeResponse, Status>> + Send>>;

    async fn streaming_recognize(
        &self,
        req: Request<Streaming<pb::StreamingRecognizeRequest>>,
    ) -> Result<Response<Self::StreamingRecognizeStream>, Status> {
        if self.mode != ServiceMode::Streaming {
            return Err(Status::failed_precondition(
                "server is running in offline mode; use Recognize",
            ));
        }

        let mut inbound = req.into_inner();
        let (out_tx, out_rx) = mpsc::channel::<Result<pb::StreamingRecognizeResponse, Status>>(64);

        // First inbound MUST carry streaming_config.
        let first = inbound
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("missing streaming_config first message"))?
            .map_err(|e| Status::internal(format!("stream recv: {e}")))?;

        let scfg = match first.streaming_request {
            Some(pb::streaming_recognize_request::StreamingRequest::StreamingConfig(c)) => c,
            _ => {
                return Err(Status::invalid_argument(
                    "first message must carry streaming_config",
                ))
            }
        };
        let rcfg = scfg
            .config
            .ok_or_else(|| Status::invalid_argument("missing recognition config"))?;
        let sr = if rcfg.sample_rate_hertz == 0 {
            16_000
        } else {
            rcfg.sample_rate_hertz
        };
        let (pcm_enc, _ct_hint) = map_encoding(rcfg.encoding)?;
        let want_partials = scfg.interim_results;
        let max_alts = rcfg.max_alternatives;

        let mut handle = self
            .pool
            .open_streaming(sr, rcfg.priority)
            .await
            .map_err(|e| Status::resource_exhausted(format!("submit failed: {e}")))?;

        let pool = Arc::clone(&self.pool);
        let rid = handle.request_id.clone();

        tokio::spawn(async move {
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
                                let resp = pb::StreamingRecognizeResponse {
                                    results: vec![pb::StreamingRecognitionResult {
                                        alternatives: build_alternatives(text, tokens, scores, max_alts),
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
                            Some(Event::Final { text, tokens, scores, .. }) => {
                                let resp = pb::StreamingRecognizeResponse {
                                    results: vec![pb::StreamingRecognitionResult {
                                        alternatives: build_alternatives(text, tokens, scores, max_alts),
                                        is_final: true,
                                        stability: 1.0,
                                        result_end_time: None,
                                        language_code: String::new(),
                                    }],
                                    speech_event_type: pb::SpeechEventType::SpeechEventUnspecified as i32,
                                    request_id: rid.clone(),
                                };
                                let _ = out_tx.send(Ok(resp)).await;
                                handle.finish();
                                break;
                            }
                            Some(Event::Error { code, message, .. }) => {
                                let _ = out_tx.send(Err(map_error(code, message))).await;
                                handle.finish();
                                break;
                            }
                            Some(_) => {} // Accepted / Pong / Overloaded — ignored at this layer.
                            None => {
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
                                            let _ = out_tx.send(Err(Status::invalid_argument(format!("pcm decode: {e}")))).await;
                                            continue;
                                        }
                                    };
                                    if handle.push_chunk(chunk).await.is_err() {
                                        warn!("grpc bidi: audio channel dropped");
                                        break;
                                    }
                                } else {
                                    let _ = out_tx.send(Err(Status::invalid_argument("expected audio_content"))).await;
                                }
                            }
                            Some(Err(e)) => {
                                let _ = out_tx.send(Err(Status::internal(format!("inbound: {e}")))).await;
                                break;
                            }
                            None => {
                                // Client half-closed: send is_last once and
                                // stop polling the inbound stream.  Keep
                                // draining events until Final / Error.
                                inbound_done = true;
                                let _ = handle.flush_last(Bytes::new()).await;
                            }
                        }
                    }
                }
            }
            pool.release(&rid);
        });

        let out_stream = ReceiverStream::new(out_rx);
        Ok(Response::new(Box::pin(out_stream)))
    }
}
