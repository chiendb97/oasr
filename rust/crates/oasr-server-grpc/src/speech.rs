// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! `oasr.asr.v1.Speech` service implementation.

use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;
use futures::Stream;
use oasr_asr::{decode_audio, decode_raw_pcm, PcmEncoding};
use oasr_engine_client::EnginePool;
use oasr_wire::Event;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};
use tracing::{error, warn};

use crate::pb;

pub struct SpeechService {
    pool: Arc<EnginePool>,
}

impl SpeechService {
    pub fn new(pool: Arc<EnginePool>) -> Self {
        Self { pool }
    }
}

fn pick_encoding(enc: i32) -> PcmEncoding {
    use pb::recognition_config::AudioEncoding;
    match AudioEncoding::try_from(enc).unwrap_or(AudioEncoding::EncodingUnspecified) {
        AudioEncoding::Linear16I16 => PcmEncoding::I16Le,
        _ => PcmEncoding::F32Le,
    }
}

fn content_type_for(enc: i32) -> Option<&'static str> {
    use pb::recognition_config::AudioEncoding;
    match AudioEncoding::try_from(enc).unwrap_or(AudioEncoding::EncodingUnspecified) {
        AudioEncoding::Wav => Some("audio/wav"),
        _ => None,
    }
}

fn token_lists(tokens: Vec<Vec<u32>>) -> Vec<pb::TokenList> {
    tokens.into_iter().map(|ids| pb::TokenList { ids }).collect()
}

#[tonic::async_trait]
impl pb::speech_server::Speech for SpeechService {
    async fn recognize(
        &self,
        req: Request<pb::RecognizeRequest>,
    ) -> Result<Response<pb::RecognizeResponse>, Status> {
        let pb::RecognizeRequest { config, audio } = req.into_inner();
        let cfg = config.ok_or_else(|| Status::invalid_argument("config required"))?;
        let sr = if cfg.sample_rate_hertz == 0 {
            16000
        } else {
            cfg.sample_rate_hertz
        };
        let decoded = decode_audio(
            content_type_for(cfg.encoding),
            &audio,
            pick_encoding(cfg.encoding),
            Some(sr),
        )
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
                request_id,
                text,
                tokens: token_lists(tokens),
                scores: scores.unwrap_or_default(),
            })),
            Event::Error { code, message, .. } => Err(match code {
                oasr_wire::ErrorCode::Busy => Status::resource_exhausted(message),
                oasr_wire::ErrorCode::UnknownRequest => Status::not_found(message),
                oasr_wire::ErrorCode::InvalidCmd => Status::invalid_argument(message),
                _ => Status::internal(message),
            }),
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
        let mut inbound = req.into_inner();
        let (out_tx, out_rx) = mpsc::channel::<Result<pb::StreamingRecognizeResponse, Status>>(64);

        // Read the first message; must be a streaming_config.
        let first = inbound
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("missing streaming_config first message"))?
            .map_err(|e| Status::internal(format!("stream recv: {e}")))?;

        let cfg = match first.streaming_request {
            Some(pb::streaming_recognize_request::StreamingRequest::StreamingConfig(c)) => c,
            _ => {
                return Err(Status::invalid_argument(
                    "first message must carry streaming_config",
                ))
            }
        };
        let rcfg = cfg
            .config
            .ok_or_else(|| Status::invalid_argument("missing recognition config"))?;
        let sr = if rcfg.sample_rate_hertz == 0 {
            16000
        } else {
            rcfg.sample_rate_hertz
        };
        let encoding = pick_encoding(rcfg.encoding);
        let want_partials = cfg.interim_results;

        let mut handle = self
            .pool
            .open_streaming(sr, rcfg.priority)
            .await
            .map_err(|e| Status::resource_exhausted(format!("submit failed: {e}")))?;

        let pool = Arc::clone(&self.pool);
        let rid = handle.request_id.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    ev = handle.events.next() => {
                        match ev {
                            Some(Event::Partial { text, tokens, .. }) => {
                                if !want_partials { continue; }
                                let _ = out_tx.send(Ok(pb::StreamingRecognizeResponse {
                                    request_id: rid.clone(),
                                    is_final: false,
                                    text,
                                    tokens: token_lists(tokens),
                                    scores: vec![],
                                })).await;
                            }
                            Some(Event::Final { text, tokens, scores, .. }) => {
                                let _ = out_tx.send(Ok(pb::StreamingRecognizeResponse {
                                    request_id: rid.clone(),
                                    is_final: true,
                                    text,
                                    tokens: token_lists(tokens),
                                    scores: scores.unwrap_or_default(),
                                })).await;
                                handle.finish();
                                break;
                            }
                            Some(Event::Error { code, message, .. }) => {
                                let status = match code {
                                    oasr_wire::ErrorCode::Busy => Status::resource_exhausted(message),
                                    oasr_wire::ErrorCode::UnknownRequest => Status::not_found(message),
                                    oasr_wire::ErrorCode::InvalidCmd => Status::invalid_argument(message),
                                    _ => Status::internal(message),
                                };
                                let _ = out_tx.send(Err(status)).await;
                                handle.finish();
                                break;
                            }
                            Some(_) => {} // Accepted / others ignored.
                            None => {
                                let _ = out_tx.send(Err(Status::internal("event stream closed"))).await;
                                break;
                            }
                        }
                    }
                    msg = inbound.next() => {
                        match msg {
                            Some(Ok(m)) => {
                                if let Some(pb::streaming_recognize_request::StreamingRequest::AudioContent(bytes)) = m.streaming_request {
                                    let chunk = match decode_raw_pcm(&bytes, encoding, sr) {
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
                                // Client half-closed: send is_last and continue reading events.
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
