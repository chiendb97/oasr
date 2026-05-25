// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Single-worker async client over the pure-Rust `zeromq` crate.
//!
//! Architecture:
//! - One [`zeromq::DealerSocket`] per worker.  `zeromq` is tokio-native so we
//!   drive it directly from a pair of tokio tasks.
//! - A `tokio::sync::mpsc::channel::<CmdEnvelope>(N)` is the system-wide
//!   backpressure point — handlers `try_send` and full → HTTP 503.
//! - A writer task pulls envelopes and emits `ZmqMessage`s.
//! - A reader task pulls `ZmqMessage`s, decodes events, updates load + routes.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use oasr_wire::{Cmd, Event};
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, warn};
use uuid::Uuid;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use crate::handle::{OfflineHandle, StreamingHandle};
use crate::router::RouterActor;
use crate::EngineClientError;

/// One outbound command + optional binary payload.
pub struct CmdEnvelope {
    pub cmd: Cmd,
    pub payload: Option<Bytes>,
}

impl CmdEnvelope {
    pub fn new(cmd: Cmd, payload: Option<Bytes>) -> Self {
        Self { cmd, payload }
    }
}

pub const DEFAULT_CMD_CHANNEL_CAP: usize = 1024;
pub const DEFAULT_EVENT_CHANNEL_CAP: usize = 64;
pub const DEFAULT_AUDIO_CHANNEL_CAP: usize = 64;

#[derive(Debug, Clone)]
pub struct EngineClientConfig {
    pub endpoint: String,
    pub cmd_channel_cap: usize,
    pub event_channel_cap: usize,
    pub audio_channel_cap: usize,
    pub label: String,
}

impl EngineClientConfig {
    pub fn new(endpoint: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            cmd_channel_cap: DEFAULT_CMD_CHANNEL_CAP,
            event_channel_cap: DEFAULT_EVENT_CHANNEL_CAP,
            audio_channel_cap: DEFAULT_AUDIO_CHANNEL_CAP,
            label: label.into(),
        }
    }
}

/// One worker, one DEALER socket.  Cheap to clone (internal `Arc`s).
#[derive(Clone)]
pub struct EngineClient {
    cfg: Arc<EngineClientConfig>,
    cmd_tx: mpsc::Sender<CmdEnvelope>,
    router: RouterActor,
    load: Arc<AtomicU32>,
    model_info: Arc<Mutex<Option<oasr_wire::ModelInfo>>>,
    last_pong_at: Arc<Mutex<Option<Instant>>>,
}

impl EngineClient {
    /// Connect to a worker and spawn the reader/writer tokio tasks.
    ///
    /// The DEALER identity is auto-assigned by the `zeromq` crate.  The
    /// worker addresses replies back using whatever identity arrived on the
    /// inbound frame — it doesn't matter what value we pick.
    pub async fn connect(cfg: EngineClientConfig) -> Result<Self, EngineClientError> {
        let mut socket = DealerSocket::new();
        socket.connect(&cfg.endpoint).await.map_err(zmq_err)?;

        let cfg = Arc::new(cfg);
        let (cmd_tx, cmd_rx) = mpsc::channel::<CmdEnvelope>(cfg.cmd_channel_cap);
        let router = RouterActor::new();
        let load = Arc::new(AtomicU32::new(0));
        let model_info = Arc::new(Mutex::new(None));
        let last_pong_at = Arc::new(Mutex::new(None));

        // One I/O task owns the socket and select!s on both directions.
        // Splitting `DealerSocket` into independent read/write halves isn't
        // supported by the `zeromq` crate at this version; sharing it across
        // tasks via a Mutex deadlocks (reader holds the lock awaiting recv,
        // writer can never acquire it to send).  A single task is simplest.
        {
            let router = router.clone();
            let load = Arc::clone(&load);
            let model_info = Arc::clone(&model_info);
            let last_pong_at = Arc::clone(&last_pong_at);
            let label = cfg.label.clone();
            tokio::spawn(io_task(
                socket, cmd_rx, router, load, model_info, last_pong_at, label,
            ));
        }

        Ok(Self {
            cfg,
            cmd_tx,
            router,
            load,
            model_info,
            last_pong_at,
        })
    }

    pub fn label(&self) -> &str {
        &self.cfg.label
    }

    pub fn load(&self) -> u32 {
        self.load.load(Ordering::Relaxed)
    }

    pub fn model_info(&self) -> Option<oasr_wire::ModelInfo> {
        self.model_info.lock().clone()
    }

    pub fn last_pong_at(&self) -> Option<Instant> {
        *self.last_pong_at.lock()
    }

    pub fn router(&self) -> &RouterActor {
        &self.router
    }

    pub fn cmd_tx(&self) -> mpsc::Sender<CmdEnvelope> {
        self.cmd_tx.clone()
    }

    /// Send a `Ping` and await the matching `Pong`.  Uses `last_pong_at` as
    /// the signal channel since `Pong` has no per-request id.
    pub async fn ping(&self, timeout: Duration) -> Result<Event, EngineClientError> {
        let start = Instant::now();
        let pre = self.last_pong_at();
        self.send_envelope(CmdEnvelope::new(Cmd::Ping { seq: 0 }, None)).await?;
        let deadline = start + timeout;
        loop {
            if let Some(t) = self.last_pong_at() {
                if Some(t) != pre {
                    let mi = self.model_info();
                    return Ok(Event::Pong {
                        seq: 0,
                        model_info: mi,
                        num_running: 0,
                        num_waiting: self.load(),
                    });
                }
            }
            if Instant::now() >= deadline {
                return Err(EngineClientError::Other(format!(
                    "ping timeout after {timeout:?}"
                )));
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    pub async fn submit_offline(
        &self,
        audio: Bytes,
        sample_rate: u32,
        priority: i32,
    ) -> Result<OfflineHandle, EngineClientError> {
        let request_id = Uuid::new_v4().simple().to_string();
        let (tx, rx) = oneshot::channel::<Event>();
        let mut event_rx = self.router.register(request_id.clone(), 8);
        tokio::spawn(async move {
            while let Some(ev) = event_rx.recv().await {
                if matches!(ev, Event::Final { .. } | Event::Error { .. }) {
                    let _ = tx.send(ev);
                    break;
                }
            }
        });
        let envelope = CmdEnvelope::new(
            Cmd::CreateOffline {
                request_id: request_id.clone(),
                sample_rate,
                priority,
            },
            Some(audio),
        );
        self.send_envelope(envelope).await?;
        self.load.fetch_add(1, Ordering::Relaxed);
        Ok(OfflineHandle::new(request_id, rx))
    }

    pub async fn open_streaming(
        &self,
        sample_rate: u32,
        priority: i32,
    ) -> Result<StreamingHandle, EngineClientError> {
        let request_id = Uuid::new_v4().simple().to_string();
        self.open_streaming_with_id(request_id, sample_rate, priority).await
    }

    pub async fn open_streaming_with_id(
        &self,
        request_id: String,
        sample_rate: u32,
        priority: i32,
    ) -> Result<StreamingHandle, EngineClientError> {
        let event_rx = self.router.register(request_id.clone(), self.cfg.event_channel_cap);
        let stream = ReceiverStream::new(event_rx);
        let envelope = CmdEnvelope::new(
            Cmd::CreateStreaming {
                request_id: request_id.clone(),
                sample_rate,
                priority,
            },
            None,
        );
        self.send_envelope(envelope).await?;
        self.load.fetch_add(1, Ordering::Relaxed);

        // Audio channel carries (chunk, is_last) so the forwarder preserves the
        // order between in-flight push_chunk + flush_last operations.  If the
        // handle is dropped before any is_last=true frame is enqueued,
        // ``CancelOnDrop`` emits ``Cmd::Cancel`` and the worker aborts.
        let (audio_tx, mut audio_rx) =
            mpsc::channel::<(Bytes, bool)>(self.cfg.audio_channel_cap);
        let cmd_tx_for_audio = self.cmd_tx.clone();
        let rid_for_task = request_id.clone();
        tokio::spawn(async move {
            while let Some((chunk, is_last)) = audio_rx.recv().await {
                let envelope = CmdEnvelope::new(
                    Cmd::FeedChunk {
                        request_id: rid_for_task.clone(),
                        is_last,
                    },
                    Some(chunk),
                );
                if cmd_tx_for_audio.send(envelope).await.is_err() {
                    break;
                }
                if is_last {
                    break;
                }
            }
        });

        Ok(StreamingHandle::new(
            request_id,
            audio_tx,
            stream,
            self.cmd_tx.clone(),
            self.router.clone(),
        ))
    }

    pub async fn cancel(&self, request_id: &str) -> Result<(), EngineClientError> {
        self.router.remove(request_id);
        let envelope = CmdEnvelope::new(
            Cmd::Cancel {
                request_id: request_id.to_owned(),
            },
            None,
        );
        self.send_envelope(envelope).await
    }

    async fn send_envelope(&self, env: CmdEnvelope) -> Result<(), EngineClientError> {
        self.cmd_tx
            .send(env)
            .await
            .map_err(|_| EngineClientError::WorkerDown(self.cfg.label.clone()))
    }
}

fn zmq_err(e: zeromq::ZmqError) -> EngineClientError {
    EngineClientError::Other(format!("zmq: {e}"))
}

// ---------------------------------------------------------------------------
// Background tasks
// ---------------------------------------------------------------------------

async fn io_task(
    mut socket: DealerSocket,
    mut cmd_rx: mpsc::Receiver<CmdEnvelope>,
    router: RouterActor,
    load: Arc<AtomicU32>,
    model_info: Arc<Mutex<Option<oasr_wire::ModelInfo>>>,
    last_pong_at: Arc<Mutex<Option<Instant>>>,
    label: String,
) {
    info!(label = %label, "zmq i/o task started");
    loop {
        tokio::select! {
            // Outbound: a new command to send.
            cmd_opt = cmd_rx.recv() => {
                let Some(env) = cmd_opt else {
                    info!(label = %label, "command channel closed; exiting io_task");
                    break;
                };
                let header_bytes = match oasr_wire::encode_cmd(&env.cmd) {
                    Ok(b) => b,
                    Err(e) => {
                        error!("writer: encode failed: {e}");
                        continue;
                    }
                };
                let mut frames: Vec<Bytes> = Vec::with_capacity(2);
                frames.push(Bytes::from(header_bytes));
                if let Some(payload) = env.payload {
                    frames.push(payload);
                }
                let msg: ZmqMessage = match ZmqMessage::try_from(frames) {
                    Ok(m) => m,
                    Err(_) => {
                        error!("writer: empty frames vector");
                        continue;
                    }
                };
                if let Err(e) = socket.send(msg).await {
                    error!(label = %label, "zmq send error: {e}");
                }
            }

            // Inbound: a frame arrived from the peer.
            recv_res = socket.recv() => {
                let msg = match recv_res {
                    Ok(m) => m,
                    Err(e) => {
                        warn!(label = %label, "zmq recv error: {e}");
                        // Brief backoff; the peer may have dropped and
                        // we'll reconnect via a fresh client.
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                };
                let header = match msg.iter().next() {
                    Some(b) => b.clone(),
                    None => continue,
                };
                let event = match oasr_wire::decode_event(&header) {
                    Ok(e) => e,
                    Err(e) => {
                        warn!(label = %label, "bad event header: {e}");
                        continue;
                    }
                };
                match &event {
                    Event::Pong { num_running, num_waiting, model_info: mi, .. } => {
                        load.store(num_running.saturating_add(*num_waiting), Ordering::Relaxed);
                        *last_pong_at.lock() = Some(Instant::now());
                        if let Some(m) = mi {
                            *model_info.lock() = Some(m.clone());
                        }
                    }
                    Event::Overloaded { queue_depth, .. } => {
                        load.store(*queue_depth, Ordering::Relaxed);
                    }
                    Event::Final { .. } | Event::Error { .. } => {
                        let prev = load.load(Ordering::Relaxed);
                        if prev > 0 {
                            load.store(prev - 1, Ordering::Relaxed);
                        }
                    }
                    _ => {}
                }
                router.route(event).await;
            }
        }
    }
}
