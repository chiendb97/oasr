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
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bytes::Bytes;
use oasr_wire::{Cmd, DecodingParams, Event};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{trace, warn};
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

/// How often [`EngineClient::shutdown`] looks for the dispatcher thread's exit.
const JOIN_POLL: Duration = Duration::from_millis(10);

/// One in-process engine, cheap to clone (internal `Arc`s).
#[derive(Clone)]
pub struct EngineClient {
    cfg: Arc<EngineClientConfig>,
    cmd_tx: mpsc::Sender<CmdEnvelope>,
    shared: Arc<DispatcherShared>,
    /// The dispatcher thread, until [`Self::shutdown`] has joined it.
    thread: Arc<parking_lot::Mutex<Option<JoinHandle<()>>>>,
    /// Anchor for `last_pong_at` computation — `last_event_at_ms` is the
    /// number of millis since this instant.
    epoch: Instant,
}

impl EngineClient {
    /// Start a dispatcher thread for `engine` and return the async-facing client.
    pub fn start(engine: PyEngine, cfg: EngineClientConfig) -> Self {
        let cfg = Arc::new(cfg);
        let (shared, cmd_tx, thread) = spawn_dispatcher(
            engine,
            cfg.label.clone(),
            cfg.dispatcher.clone(),
            cfg.cmd_channel_cap,
        );
        Self {
            cfg,
            cmd_tx,
            shared,
            thread: Arc::new(parking_lot::Mutex::new(Some(thread))),
            epoch: Instant::now(),
        }
    }

    /// Ask the dispatcher to exit after its current tick.  Returns at once;
    /// [`Self::shutdown`] waits for it.
    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::Release);
        // Wakes a dispatcher blocked on an idle receive, which would otherwise
        // notice the flag only when that wait times out.  A full channel means
        // it is not idle, and it will see the flag on its next iteration.
        let _ = self
            .cmd_tx
            .try_send(CmdEnvelope::new(Cmd::Ping { seq: 0 }, None));
    }

    /// Stop the dispatcher and wait up to `timeout` for its thread to exit.
    ///
    /// On the way out the thread fails every request still in flight with
    /// `SHUTDOWN`, aborts them and calls the engine's `shutdown()`.  Returns
    /// whether the thread exited; a later call keeps waiting for it.
    ///
    /// The thread has to be gone before the tokio runtime is: it waits on the
    /// runtime's timers, and a process that exited under it cut whatever tick
    /// it was in.
    pub async fn shutdown(&self, timeout: Duration) -> bool {
        self.stop();
        let deadline = Instant::now() + timeout;
        loop {
            let finished = match self.thread.lock().as_ref() {
                None => return true,
                Some(t) => t.is_finished(),
            };
            if finished {
                if let Some(t) = self.thread.lock().take() {
                    // Already finished, so this returns at once.
                    if t.join().is_err() {
                        warn!(label = %self.cfg.label, "dispatcher thread panicked on its way out");
                    }
                }
                return true;
            }
            if Instant::now() >= deadline {
                warn!(
                    label = %self.cfg.label,
                    timeout_secs = timeout.as_secs_f64(),
                    "dispatcher thread did not exit in time"
                );
                return false;
            }
            tokio::time::sleep(JOIN_POLL).await;
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
        let (registration, events) = self.shared.router.register(request_id.clone(), 8);
        let envelope = CmdEnvelope::new(
            Cmd::CreateOffline {
                request_id: request_id.clone(),
                sample_rate,
                priority,
                decoding,
            },
            Some(audio),
        );
        // A send that fails, or whose future is dropped while the channel is
        // full, drops `registration` here and takes the router entry with it.
        self.send_envelope(envelope).await?;
        Ok(OfflineHandle::new(
            registration,
            events,
            self.cmd_tx.clone(),
        ))
    }

    /// Submit an offline request and keep the **whole** event stream.
    ///
    /// Unlike [`Self::submit_offline`], returns non-terminal events to the caller.
    /// Audio still arrives in one shot; only decoded text is incremental.
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
        let (registration, event_rx) = self
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
            registration,
            stream,
            self.cmd_tx.clone(),
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
        let (registration, event_rx) = self
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
            registration,
            stream,
            self.cmd_tx.clone(),
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

    /// A client whose "dispatcher" is the returned receiver, so a test can
    /// fill, drain or drop the command channel itself.
    fn detached(cap: usize) -> (EngineClient, mpsc::Receiver<CmdEnvelope>) {
        let (cmd_tx, cmd_rx) = mpsc::channel(cap);
        let client = EngineClient {
            cfg: Arc::new(EngineClientConfig::new("t")),
            cmd_tx,
            shared: Arc::new(DispatcherShared::new("t".into(), None)),
            thread: Arc::new(parking_lot::Mutex::new(None)),
            epoch: Instant::now(),
        };
        (client, cmd_rx)
    }

    fn audio() -> Bytes {
        Bytes::from_static(&[0u8; 16])
    }

    /// The submit registered its event channel, then failed to send.  No
    /// handle was ever built to remove it, and the unary path's forwarding
    /// task waited on it for the life of the process.
    #[tokio::test]
    async fn a_submit_the_engine_never_receives_leaves_no_entry() {
        let (client, cmd_rx) = detached(4);
        drop(cmd_rx);
        assert!(client
            .submit_offline(audio(), 16_000, 0, None)
            .await
            .is_err());
        assert!(client
            .submit_offline_streaming(audio(), 16_000, 0, None)
            .await
            .is_err());
        assert!(client.open_streaming(16_000, 0, None).await.is_err());
        assert!(
            client.router().is_empty(),
            "{} entries outlived their submits",
            client.router().len()
        );
    }

    /// A handler dropped while its submit waits for room -- a client
    /// disconnect, or a deadline, against a backed-up dispatcher.  The command
    /// was never sent, so nothing would ever route a terminal event for it.
    #[tokio::test]
    async fn a_submit_abandoned_on_a_full_channel_leaves_no_entry() {
        let (client, _cmd_rx) = detached(1);
        client
            .cmd_tx
            .try_send(CmdEnvelope::new(Cmd::Ping { seq: 0 }, None))
            .unwrap();
        let wait = Duration::from_millis(50);
        assert!(
            tokio::time::timeout(wait, client.submit_offline(audio(), 16_000, 0, None))
                .await
                .is_err(),
            "the full channel should have held the submit"
        );
        assert!(
            tokio::time::timeout(wait, client.open_streaming(16_000, 0, None))
                .await
                .is_err()
        );
        assert!(
            client.router().is_empty(),
            "{} entries outlived their submits",
            client.router().len()
        );
    }

    /// A stream that sent its last chunk is disarmed, and its handler can still
    /// return before the final arrives -- on a failed send to its client, say.
    #[tokio::test]
    async fn a_stream_dropped_after_its_last_chunk_leaves_no_entry() {
        let (client, mut cmd_rx) = detached(4);
        let handle = client.open_streaming(16_000, 0, None).await.unwrap();
        handle.flush_last(Bytes::new()).await.unwrap();
        assert_eq!(client.router().len(), 1);
        drop(handle);
        assert!(client.router().is_empty(), "the entry outlived its handle");
        // Created, then the last chunk; no cancel for a finished stream.
        for _ in 0..2 {
            let env = cmd_rx.recv().await.unwrap();
            assert!(!matches!(env.cmd, Cmd::Cancel { .. }), "{:?}", env.cmd);
        }
        assert!(cmd_rx.try_recv().is_err());
    }

    /// An unfinished request dropped by its handler is cancelled and removed.
    #[tokio::test]
    async fn a_request_dropped_mid_flight_is_cancelled_and_removed() {
        let (client, mut cmd_rx) = detached(4);
        let handle = client
            .submit_offline(audio(), 16_000, 0, None)
            .await
            .unwrap();
        let rid = handle.request_id.clone();
        drop(handle);
        assert!(client.router().is_empty(), "the entry outlived its handle");
        assert!(matches!(
            cmd_rx.recv().await.unwrap().cmd,
            Cmd::CreateOffline { .. }
        ));
        match cmd_rx.recv().await.unwrap().cmd {
            Cmd::Cancel { request_id } => assert_eq!(request_id, rid),
            other => panic!("expected a cancel, got {other:?}"),
        }
    }

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
