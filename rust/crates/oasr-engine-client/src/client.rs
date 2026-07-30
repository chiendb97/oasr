// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! `EngineClient` — async-facing facade for one in-process `PyEngine`.
//!
//! Each client owns:
//! - a [`crate::pyengine::PyEngine`] (lives on the dispatcher thread; the
//!   client never touches it directly), and
//! - a `tokio::mpsc::Sender<CmdEnvelope>` that async handlers push commands
//!   into.  The dispatcher thread drains the channel each iteration.
//!
//! Public API is intentionally identical to the previous ZMQ-backed client
//! so the HTTP / gRPC adapters and `EnginePool` see no behaviour change.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use oasr_wire::{Cmd, DecodingParams, Event};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tracing::trace;
use uuid::Uuid;

use crate::dispatcher::{
    spawn as spawn_dispatcher, CmdEnvelope, DispatcherConfig, DispatcherShared,
};
use crate::handle::{OfflineHandle, OfflineStreamHandle, StreamingHandle};
use crate::pyengine::PyEngine;
use crate::router::RouterActor;
use crate::EngineClientError;

pub const DEFAULT_EVENT_CHANNEL_CAP: usize = 64;

/// Floor on the command channel so a tiny `max_concurrent_requests` still
/// leaves room for the non-admission traffic (chunks, cancels, pings).
pub const MIN_CMD_CHANNEL_CAP: usize = 64;

/// Command-channel depth for a given admission cap.
///
/// The two bounds have to be related, because every buffered `CreateOffline`
/// carries its **full audio payload** in `CmdEnvelope::payload`: a fixed 1024
/// against a default cap of 256 meant up to 4× the admissible work could sit in
/// the channel — each up to `--max-audio-mib` — before a sender blocked, and
/// that wait is undeadlined.  Two per admissible request covers one in-flight
/// `FeedChunk`/`Cancel` per open stream on top of its admission, which is the
/// most a well-behaved front-end has outstanding at once.
pub fn cmd_channel_cap_for(max_concurrent_requests: u32) -> usize {
    (max_concurrent_requests as usize)
        .saturating_mul(2)
        .max(MIN_CMD_CHANNEL_CAP)
}

#[derive(Debug, Clone)]
pub struct EngineClientConfig {
    /// Depth of the dispatcher command channel.  Defaults to
    /// [`cmd_channel_cap_for`] of the dispatcher's `max_concurrent_requests`
    /// and is recomputed by [`EngineClient::start`] unless a caller overrode it
    /// after construction.
    pub cmd_channel_cap: usize,
    pub event_channel_cap: usize,
    pub label: String,
    pub dispatcher: DispatcherConfig,
}

impl EngineClientConfig {
    pub fn new(label: impl Into<String>) -> Self {
        let dispatcher = DispatcherConfig::default();
        Self {
            cmd_channel_cap: cmd_channel_cap_for(dispatcher.max_concurrent_requests),
            event_channel_cap: DEFAULT_EVENT_CHANNEL_CAP,
            label: label.into(),
            dispatcher,
        }
    }

    /// Re-derive [`Self::cmd_channel_cap`] from the current dispatcher cap.
    ///
    /// Callers build the config, then overwrite `dispatcher` wholesale from the
    /// CLI; without this the channel would keep the depth implied by the
    /// *default* cap rather than the configured one.
    pub fn sync_cmd_channel_cap(&mut self) {
        self.cmd_channel_cap = cmd_channel_cap_for(self.dispatcher.max_concurrent_requests);
    }
}

/// One in-process engine, cheap to clone (internal `Arc`s).
#[derive(Clone)]
pub struct EngineClient {
    cfg: Arc<EngineClientConfig>,
    cmd_tx: mpsc::Sender<CmdEnvelope>,
    shared: Arc<DispatcherShared>,
    /// Anchor for `last_pong_at` computation — `last_event_at_ms` is the
    /// number of millis since this instant.
    epoch: Instant,
}

impl EngineClient {
    /// Start a dispatcher thread for `engine` and return the async-facing client.
    pub fn start(engine: PyEngine, cfg: EngineClientConfig) -> Self {
        let cfg = Arc::new(cfg);
        let (shared, cmd_tx) = spawn_dispatcher(
            engine,
            cfg.label.clone(),
            cfg.dispatcher.clone(),
            cfg.cmd_channel_cap,
        );
        Self {
            cfg,
            cmd_tx,
            shared,
            epoch: Instant::now(),
        }
    }

    pub fn label(&self) -> &str {
        &self.cfg.label
    }

    pub fn load(&self) -> u32 {
        self.shared.load.load(Ordering::Relaxed)
    }

    pub fn model_info(&self) -> Option<oasr_wire::ModelInfo> {
        self.shared.model_info.lock().clone()
    }

    /// Mirrors the legacy ZMQ-backed API.  Returns the instant of the last
    /// successful dispatcher tick; falls back to `None` before the first
    /// step has run.
    pub fn last_pong_at(&self) -> Option<Instant> {
        let ms = self.shared.last_event_at_ms.load(Ordering::Relaxed);
        if ms == 0 {
            None
        } else {
            Some(self.epoch + Duration::from_millis(ms))
        }
    }

    pub fn router(&self) -> &RouterActor {
        &self.shared.router
    }

    pub fn cmd_tx(&self) -> mpsc::Sender<CmdEnvelope> {
        self.cmd_tx.clone()
    }

    /// Send a `Ping` and wait for the dispatcher to refresh
    /// `last_pong_at`.  The dispatcher updates the heartbeat on every tick,
    /// so under load this returns immediately; under idle it blocks until
    /// the dispatcher wakes from `idle_sleep` and processes the Ping.
    pub async fn ping(&self, timeout: Duration) -> Result<Event, EngineClientError> {
        let pre = self.last_pong_at();
        self.send_envelope(CmdEnvelope::new(Cmd::Ping { seq: 0 }, None))
            .await?;
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(t) = self.last_pong_at() {
                if Some(t) != pre {
                    return Ok(Event::Pong {
                        seq: 0,
                        model_info: self.model_info(),
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
        decoding: Option<DecodingParams>,
    ) -> Result<OfflineHandle, EngineClientError> {
        let request_id = Uuid::new_v4().simple().to_string();
        let (tx, rx) = oneshot::channel::<Event>();
        let mut event_rx = self.shared.router.register(request_id.clone(), 8);
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
                decoding,
            },
            Some(audio),
        );
        self.send_envelope(envelope).await?;
        Ok(OfflineHandle::new(
            request_id,
            rx,
            self.cmd_tx.clone(),
            self.shared.router.clone(),
        ))
    }

    /// Submit an offline request and keep the **whole** event stream.
    ///
    /// [`Self::submit_offline`] registers a small channel and spawns a drain task
    /// that throws away every non-terminal event, so the per-tick
    /// `Event::Partial`s the autoregressive strategies produce could never reach a
    /// client — token streaming was built end-to-end and then discarded one layer
    /// from the wire. This variant hands the stream to the caller instead.
    ///
    /// Audio still arrives in one shot: an offline-only decode family (`aed`,
    /// `llm`, `paraformer`, `ctc_aed_rescoring`) cannot consume a growing buffer.
    /// What is decoupled here is *chunked audio in* from *incremental text out* —
    /// only the latter is what a speech-LLM client actually wants.
    pub async fn submit_offline_streaming(
        &self,
        audio: Bytes,
        sample_rate: u32,
        priority: i32,
        decoding: Option<DecodingParams>,
    ) -> Result<OfflineStreamHandle, EngineClientError> {
        let request_id = Uuid::new_v4().simple().to_string();
        self.submit_offline_streaming_with_id(request_id, audio, sample_rate, priority, decoding)
            .await
    }

    pub async fn submit_offline_streaming_with_id(
        &self,
        request_id: String,
        audio: Bytes,
        sample_rate: u32,
        priority: i32,
        decoding: Option<DecodingParams>,
    ) -> Result<OfflineStreamHandle, EngineClientError> {
        // Full ``event_channel_cap`` rather than the unary path's 8: an AR request
        // emits {Accepted, Partial x N, Final}, and N is the token count.
        let event_rx = self
            .shared
            .router
            .register(request_id.clone(), self.cfg.event_channel_cap);
        let stream = ReceiverStream::new(event_rx);
        let envelope = CmdEnvelope::new(
            Cmd::CreateOffline {
                request_id: request_id.clone(),
                sample_rate,
                priority,
                decoding,
            },
            Some(audio),
        );
        self.send_envelope(envelope).await?;
        Ok(OfflineStreamHandle::new(
            request_id,
            stream,
            self.cmd_tx.clone(),
            self.shared.router.clone(),
        ))
    }

    pub async fn open_streaming(
        &self,
        sample_rate: u32,
        priority: i32,
        decoding: Option<DecodingParams>,
    ) -> Result<StreamingHandle, EngineClientError> {
        let request_id = Uuid::new_v4().simple().to_string();
        self.open_streaming_with_id(request_id, sample_rate, priority, decoding)
            .await
    }

    pub async fn open_streaming_with_id(
        &self,
        request_id: String,
        sample_rate: u32,
        priority: i32,
        decoding: Option<DecodingParams>,
    ) -> Result<StreamingHandle, EngineClientError> {
        let event_rx = self
            .shared
            .router
            .register(request_id.clone(), self.cfg.event_channel_cap);
        let stream = ReceiverStream::new(event_rx);
        let envelope = CmdEnvelope::new(
            Cmd::CreateStreaming {
                request_id: request_id.clone(),
                sample_rate,
                priority,
                decoding,
            },
            None,
        );
        self.send_envelope(envelope).await?;

        // The handle holds its own `cmd_tx` clone and builds FeedChunk
        // envelopes inline — no separate forwarder task. Ordering between
        // `push_chunk` / `flush_last` is preserved by sequential awaits in
        // the caller (HTTP WS handler / gRPC bidi stream). If the handle
        // is dropped before an `is_last=true` flush, `CancelOnDrop` emits
        // `Cmd::Cancel` via the same `cmd_tx`.
        Ok(StreamingHandle::new(
            request_id,
            stream,
            self.cmd_tx.clone(),
            self.shared.router.clone(),
        ))
    }

    pub async fn cancel(&self, request_id: &str) -> Result<(), EngineClientError> {
        self.shared.router.remove(request_id);
        let envelope = CmdEnvelope::new(
            Cmd::Cancel {
                request_id: request_id.to_owned(),
            },
            None,
        );
        self.send_envelope(envelope).await
    }

    async fn send_envelope(&self, env: CmdEnvelope) -> Result<(), EngineClientError> {
        self.cmd_tx.send(env).await.map_err(|_| {
            trace!(label = %self.cfg.label, "send_envelope: receiver dropped");
            EngineClientError::WorkerDown(self.cfg.label.clone())
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The two bounds must stay related.  A fixed 1024-deep command channel
    /// against a default admission cap of 256 meant up to 4x the admissible
    /// work could buffer — **each envelope carrying its full audio payload** —
    /// before a sender blocked, on an undeadlined wait.
    #[test]
    fn the_channel_cap_follows_the_admission_cap() {
        for cap in [1u32, 8, 64, 256, 1024] {
            let depth = cmd_channel_cap_for(cap);
            assert!(
                depth >= cap as usize,
                "cap {cap} must not be able to block its own admissions (depth {depth})"
            );
            assert!(
                depth <= (cap as usize).saturating_mul(2).max(MIN_CMD_CHANNEL_CAP),
                "cap {cap} buffers far more than the admissible work (depth {depth})"
            );
        }
    }

    /// A tiny cap still needs room for chunks, cancels and pings.
    #[test]
    fn a_small_admission_cap_keeps_a_usable_floor() {
        assert_eq!(cmd_channel_cap_for(1), MIN_CMD_CHANNEL_CAP);
    }

    /// Callers overwrite `dispatcher` wholesale after `new()`, so the derived
    /// depth has to be re-synced or it silently keeps the *default* cap's value.
    #[test]
    fn sync_picks_up_a_dispatcher_cap_set_after_construction() {
        let mut cfg = EngineClientConfig::new("t");
        cfg.dispatcher.max_concurrent_requests = 16;
        cfg.sync_cmd_channel_cap();
        assert_eq!(cfg.cmd_channel_cap, cmd_channel_cap_for(16));
    }
}
