// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! WebSocket streaming endpoint at `GET /v1/stream`.
//!
//! Client → server framing:
//! - First frame: text JSON
//!   `{"type":"start","sample_rate":<u32>,"format":"pcm_f32le"|"pcm_i16le","priority":<i32?>}`.
//! - Subsequent: binary frames = raw PCM chunks in the declared format.
//! - Text `{"type":"end"}` marks the last chunk (flush + finalize).
//! - Text `{"type":"cancel"}` aborts the request.
//!
//! Server → client framing:
//! - Text `{"type":"accepted","request_id":...}` once admitted.
//! - Text `{"type":"partial","text":...,"tokens":...}` per partial.
//! - Text `{"type":"final","text":...,"tokens":...,"scores":...}` once finalized.
//! - Text `{"type":"error","code":...,"message":...}` on failure.

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use oasr_asr::{decode_raw_pcm, PcmEncoding};
use oasr_wire::Event;
use serde::Deserialize;
use serde_json::json;
use tracing::warn;

use crate::router::AppState;

#[derive(Debug, Deserialize)]
struct StartFrame {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    type_: String,
    sample_rate: u32,
    #[serde(default = "default_format")]
    format: String,
    #[serde(default)]
    priority: i32,
}

fn default_format() -> String {
    "pcm_f32le".to_owned()
}

#[derive(Debug, Deserialize)]
struct ControlFrame {
    #[serde(rename = "type")]
    type_: String,
}

pub async fn handle_ws(ws: WebSocketUpgrade, State(s): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws_session(socket, s))
}

async fn ws_session(socket: WebSocket, state: AppState) {
    let (mut tx, mut rx) = socket.split();

    // 1. Wait for the start frame.
    let start_text = match rx.next().await {
        Some(Ok(Message::Text(t))) => t,
        Some(Ok(Message::Close(_))) | None => return,
        Some(Ok(other)) => {
            let _ = tx
                .send(ws_error("INVALID_CMD", &format!("expected start text frame, got {other:?}")))
                .await;
            return;
        }
        Some(Err(e)) => {
            warn!("ws first recv error: {e}");
            return;
        }
    };
    let start: StartFrame = match serde_json::from_str(&start_text) {
        Ok(s) => s,
        Err(e) => {
            let _ = tx.send(ws_error("INVALID_CMD", &format!("bad start JSON: {e}"))).await;
            return;
        }
    };
    let encoding = match start.format.to_ascii_lowercase().as_str() {
        "pcm_f32le" | "f32_le" | "linear16_f32" => PcmEncoding::F32Le,
        "pcm_i16le" | "i16_le" | "linear16_i16" => PcmEncoding::I16Le,
        other => {
            let _ = tx.send(ws_error("INVALID_CMD", &format!("unsupported format {other:?}"))).await;
            return;
        }
    };

    // 2. Open streaming request via the pool.
    let mut handle = match state
        .pool
        .open_streaming(start.sample_rate, start.priority)
        .await
    {
        Ok(h) => h,
        Err(e) => {
            let _ = tx.send(ws_error("INTERNAL", &format!("submit failed: {e}"))).await;
            return;
        }
    };
    let rid = handle.request_id.clone();
    let _ = tx
        .send(Message::Text(
            json!({"type":"accepted","request_id": rid}).to_string(),
        ))
        .await;

    let mut explicit_end = false;

    loop {
        tokio::select! {
            // Outbound — engine events → ws frames.
            ev_opt = handle.events.next() => {
                match ev_opt {
                    Some(Event::Partial { text, tokens, scores, .. }) => {
                        let msg = json!({
                            "type": "partial",
                            "request_id": rid,
                            "text": text,
                            "tokens": tokens,
                            "scores": scores,
                        });
                        if tx.send(Message::Text(msg.to_string())).await.is_err() {
                            break;
                        }
                    }
                    Some(Event::Final { text, tokens, scores, .. }) => {
                        let msg = json!({
                            "type": "final",
                            "request_id": rid,
                            "text": text,
                            "tokens": tokens,
                            "scores": scores,
                        });
                        let _ = tx.send(Message::Text(msg.to_string())).await;
                        // Axum will send a Close frame for us when the
                        // handler returns; avoid double-Close which races
                        // the underlying TCP socket on heavy concurrency.
                        handle.finish();
                        break;
                    }
                    Some(Event::Error { code, message, .. }) => {
                        let _ = tx.send(ws_error(&format!("{code:?}"), &message)).await;
                        handle.finish();
                        break;
                    }
                    Some(Event::Accepted { .. }) => {}
                    Some(_) => {}
                    None => {
                        let _ = tx.send(ws_error("INTERNAL", "event channel closed")).await;
                        break;
                    }
                }
            }

            // Inbound — client → engine.
            recv = rx.next() => {
                match recv {
                    Some(Ok(Message::Binary(bin))) => {
                        let chunk = match decode_raw_pcm(&bin, encoding, start.sample_rate) {
                            Ok(d) => d.samples,
                            Err(e) => {
                                let _ = tx
                                    .send(ws_error("INVALID_CMD", &format!("pcm decode: {e}")))
                                    .await;
                                continue;
                            }
                        };
                        if let Err(_dropped) = handle.push_chunk(chunk).await {
                            warn!("audio channel dropped");
                            break;
                        }
                    }
                    Some(Ok(Message::Text(t))) => {
                        if let Ok(ctl) = serde_json::from_str::<ControlFrame>(&t) {
                            match ctl.type_.as_str() {
                                "end" => {
                                    explicit_end = true;
                                    let _ = handle.flush_last(Bytes::new()).await;
                                }
                                "cancel" => {
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(p))) => {
                        let _ = tx.send(Message::Pong(p)).await;
                    }
                    Some(Ok(Message::Pong(_))) => {}
                    Some(Err(e)) => {
                        warn!("ws recv error: {e}");
                        break;
                    }
                }
            }
        }
    }

    // If the client closed without an explicit end, dropping `handle` will
    // emit a Cancel via its Drop impl.  ``explicit_end`` is intentionally
    // unused after the loop — it exists so reviewers can see the intent.
    let _ = explicit_end;
    state.pool.release(&rid);
}

fn ws_error(code: &str, message: &str) -> Message {
    Message::Text(json!({"type":"error","code":code,"message":message}).to_string())
}
