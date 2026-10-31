// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Per-request handles returned by [`EngineClient`] / [`EnginePool`].
//!
//! [`StreamingHandle`] holds a clone of the dispatcher command sender + the
//! event receiver for one streaming session.  `push_chunk` / `flush_last`
//! build `Cmd::FeedChunk` envelopes inline and send them directly to the
//! dispatcher — there's no intermediate forwarder task.  Dropping the
//! handle without an explicit `finish` emits a `Cmd::Cancel`.
//!
//! [`OfflineHandle`] resolves with the Final or Error event.  All three
//! handles arm [`CancelOnDrop`]: a client that goes away is the normal case,
//! and the engine has to hear about it or the request keeps computing on a
//! slot nobody is waiting for.  Each also owns the request's router
//! [`Registration`], so the entry goes when the handle does, however the
//! handler that held it returned.

use bytes::Bytes;
use oasr_wire::{Cmd, Event};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tracing::{debug, trace};

use crate::dispatcher::CmdEnvelope;
use crate::router::Registration;
use crate::{EngineClientError, EventStream};

/// Carrier of a cancellation tail when the WS / gRPC stream drops early.
///
/// Disarming stops the cancel, not the cleanup: the registration is dropped
/// with the guard either way.  A streaming handle disarms at `flush_last`,
/// before its final arrives, and a handler that returned in that window used to
/// leave its entry until the engine got round to the terminal event.
struct CancelOnDrop {
    registration: Registration,
    cmd_tx: mpsc::Sender<CmdEnvelope>,
    finished: bool,
}

impl CancelOnDrop {
    fn arm(registration: Registration, cmd_tx: mpsc::Sender<CmdEnvelope>) -> Self {
        Self {
            registration,
            cmd_tx,
            finished: false,
        }
    }

    fn disarm(&mut self) {
        self.finished = true;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let request_id = self.registration.request_id();
        let cancel = Cmd::Cancel {
            request_id: request_id.to_owned(),
        };
        let envelope = CmdEnvelope::new(cancel, None);
        match self.cmd_tx.try_send(envelope) {
            Ok(()) => {}
            // A full channel is exactly when this matters: a reconnect storm
            // drops handles in bulk while the dispatcher is backed up, and
            // nothing else will ever tell the engine — a cancel that was only
            // logged left the request holding its slot and its KV blocks for
            // the life of the process.  So it waits for room instead.
            Err(TrySendError::Full(envelope)) => {
                debug!(rid = %request_id, "dispatcher channel full; deferring cancel");
                send_when_room(self.cmd_tx.clone(), envelope);
            }
            // The dispatcher is gone, and the engine with it.
            Err(TrySendError::Closed(_)) => {
                trace!(rid = %request_id, "dispatcher closed; cancel dropped");
            }
        }
        // The registration is a field, so it is removed after the cancel is
        // queued, whether or not the guard was armed.
    }
}

/// Send `envelope` once the channel has room, without blocking the dropper.
///
/// `Drop` cannot await.  It runs on a tokio worker in the front-ends but on
/// whatever thread drops the handle otherwise, so the wait goes to a task when
/// there is a runtime and to a thread when there is not.
fn send_when_room(cmd_tx: mpsc::Sender<CmdEnvelope>, envelope: CmdEnvelope) {
    match tokio::runtime::Handle::try_current() {
        Ok(runtime) => {
            runtime.spawn(async move {
                let _ = cmd_tx.send(envelope).await;
            });
        }
        Err(_) => {
            std::thread::spawn(move || {
                let _ = cmd_tx.blocking_send(envelope);
            });
        }
    }
}

/// Streaming request handle: push audio chunks, pull events.
pub struct StreamingHandle {
    pub request_id: String,
    cmd_tx: mpsc::Sender<CmdEnvelope>,
    pub events: EventStream,
    _cancel: Arc<parking_lot::Mutex<CancelOnDrop>>,
}

impl StreamingHandle {
    pub(crate) fn new(
        registration: Registration,
        events: EventStream,
        cmd_tx: mpsc::Sender<CmdEnvelope>,
    ) -> Self {
        Self {
            request_id: registration.request_id().to_owned(),
            _cancel: Arc::new(parking_lot::Mutex::new(CancelOnDrop::arm(
                registration,
                cmd_tx.clone(),
            ))),
            cmd_tx,
            events,
        }
    }

    /// Push one audio chunk with `is_last=false`.
    ///
    /// On send failure (dispatcher gone, channel closed), returns the chunk
    /// bytes back to the caller so they can decide whether to retry or
    /// surface an error to the API client.
    pub async fn push_chunk(&self, audio: Bytes) -> Result<(), Bytes> {
        let envelope = CmdEnvelope::new(
            Cmd::FeedChunk {
                request_id: self.request_id.clone(),
                is_last: false,
            },
            Some(audio),
        );
        self.cmd_tx
            .send(envelope)
            .await
            .map_err(|e| e.0.payload.unwrap_or_default())
    }

    /// Push the final audio chunk with `is_last=true` and disarm
    /// cancel-on-drop.  Pass an empty `Bytes` if the caller already
    /// exhausted audio.
    pub async fn flush_last(&self, audio: Bytes) -> Result<(), Bytes> {
        let envelope = CmdEnvelope::new(
            Cmd::FeedChunk {
                request_id: self.request_id.clone(),
                is_last: true,
            },
            Some(audio),
        );
        let result = self
            .cmd_tx
            .send(envelope)
            .await
            .map_err(|e| e.0.payload.unwrap_or_default());
        self.finish();
        result
    }

    /// Mark the handle as completed so dropping it won't emit a Cancel.
    pub fn finish(&self) {
        self._cancel.lock().disarm();
    }
}

/// Offline request handle that streams **every** event, not just the terminal one.
///
/// Audio arrives in one shot, while decoded text is emitted incrementally.
/// Cancel-on-drop prevents a disconnected client from retaining a decode slot.
pub struct OfflineStreamHandle {
    pub request_id: String,
    pub events: EventStream,
    _cancel: Arc<parking_lot::Mutex<CancelOnDrop>>,
}

impl OfflineStreamHandle {
    pub(crate) fn new(
        registration: Registration,
        events: EventStream,
        cmd_tx: mpsc::Sender<CmdEnvelope>,
    ) -> Self {
        Self {
            request_id: registration.request_id().to_owned(),
            _cancel: Arc::new(parking_lot::Mutex::new(CancelOnDrop::arm(
                registration,
                cmd_tx,
            ))),
            events,
        }
    }

    /// Mark the request as completed so dropping the handle won't emit a Cancel.
    pub fn finish(&self) {
        self._cancel.lock().disarm();
    }
}

/// Offline request handle: await a single final result.
///
/// Cancel-on-drop is armed like the streaming handles', and it matters just as
/// much here even though the client sends its audio in one shot: this handle
/// backs **both** `POST /v1/speech:recognize` and the gRPC unary `Recognize`,
/// and both front-ends' handler futures are dropped when the client
/// disconnects.  Without the guard the request kept computing and kept holding
/// its `--max-concurrent-requests` slot — for an autoregressive family, the
/// whole `max_new_tokens` generation, burned on nobody.
///
/// The handle reads its own events rather than leaving them to a forwarding
/// task: a task spawned per request is one more thing that outlives the request
/// when it ends early, and this one did, waiting forever on a registration no
/// terminal event would ever remove.
pub struct OfflineHandle {
    pub request_id: String,
    events: mpsc::Receiver<Event>,
    cancel: CancelOnDrop,
}

impl OfflineHandle {
    pub(crate) fn new(
        registration: Registration,
        events: mpsc::Receiver<Event>,
        cmd_tx: mpsc::Sender<CmdEnvelope>,
    ) -> Self {
        Self {
            request_id: registration.request_id().to_owned(),
            cancel: CancelOnDrop::arm(registration, cmd_tx),
            events,
        }
    }

    /// Await the final result event, skipping the non-terminal ones.
    ///
    /// The guard is disarmed only once the event stream has ended.  Dropping
    /// this future before then — which is what a client disconnect does — leaves
    /// it armed and cancels the request.
    pub async fn finish(self) -> Result<Event, EngineClientError> {
        let Self {
            mut events,
            mut cancel,
            ..
        } = self;
        let terminal = loop {
            match events.recv().await {
                Some(ev) if ev.is_terminal() => break Some(ev),
                Some(_) => continue,
                None => break None,
            }
        };
        cancel.disarm();
        terminal.ok_or_else(|| {
            EngineClientError::Other("event stream closed before a terminal event".into())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::RouterActor;
    use std::time::Duration;

    /// A guard for `rid` on its own router, which the caller keeps to inspect.
    fn armed(rid: &str, cmd_tx: mpsc::Sender<CmdEnvelope>) -> (CancelOnDrop, RouterActor) {
        let router = RouterActor::default();
        let (registration, _events) = router.register(rid.into(), 1);
        (CancelOnDrop::arm(registration, cmd_tx), router)
    }

    fn cancel_for(rid: &str) -> CmdEnvelope {
        CmdEnvelope::new(
            Cmd::Cancel {
                request_id: rid.into(),
            },
            None,
        )
    }

    /// A one-slot channel that is already full, as under a reconnect storm.
    fn full_channel() -> (mpsc::Sender<CmdEnvelope>, mpsc::Receiver<CmdEnvelope>) {
        let (tx, rx) = mpsc::channel(1);
        tx.try_send(cancel_for("queued-ahead")).unwrap();
        (tx, rx)
    }

    async fn next_cancelled(rx: &mut mpsc::Receiver<CmdEnvelope>) -> String {
        let envelope = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("the cancel never arrived")
            .expect("channel closed");
        match envelope.cmd {
            Cmd::Cancel { request_id } => request_id,
            other => panic!("expected a cancel, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_cancel_dropped_into_a_full_channel_still_arrives() {
        let (tx, mut rx) = full_channel();
        drop(armed("gone", tx).0);
        assert_eq!(next_cancelled(&mut rx).await, "queued-ahead");
        assert_eq!(next_cancelled(&mut rx).await, "gone");
    }

    #[tokio::test]
    async fn a_handle_dropped_off_the_runtime_still_cancels() {
        let (tx, mut rx) = full_channel();
        std::thread::spawn(move || drop(armed("gone", tx).0))
            .join()
            .unwrap();
        assert_eq!(next_cancelled(&mut rx).await, "queued-ahead");
        assert_eq!(next_cancelled(&mut rx).await, "gone");
    }

    #[tokio::test]
    async fn a_disarmed_guard_sends_nothing() {
        let (tx, mut rx) = mpsc::channel(1);
        let (mut guard, _router) = armed("done", tx);
        guard.disarm();
        drop(guard);
        assert!(
            rx.recv().await.is_none(),
            "a finished request must not be cancelled"
        );
    }

    /// Disarming is about the cancel.  A streaming handle disarms at
    /// `flush_last`, before its final arrives, and its router entry used to
    /// stay until that final was routed -- or for good, if it never was.
    #[tokio::test]
    async fn a_disarmed_guard_still_takes_its_router_entry() {
        let (tx, _rx) = mpsc::channel(1);
        let (mut guard, router) = armed("done", tx);
        assert_eq!(router.len(), 1);
        guard.disarm();
        drop(guard);
        assert!(router.is_empty(), "the entry outlived its handle");
    }

    #[tokio::test]
    async fn an_armed_guard_takes_its_router_entry_too() {
        let (tx, _rx) = mpsc::channel(1);
        let (guard, router) = armed("gone", tx);
        drop(guard);
        assert!(router.is_empty(), "the entry outlived its handle");
    }

    /// The unary handle reads past `Accepted` and partials to the terminal.
    #[tokio::test]
    async fn an_offline_handle_resolves_with_the_terminal_event() {
        let router = RouterActor::default();
        let (registration, events) = router.register("r".into(), 8);
        let (cmd_tx, mut cmd_rx) = mpsc::channel(1);
        let handle = OfflineHandle::new(registration, events, cmd_tx);
        router
            .route(Event::Accepted {
                request_id: "r".into(),
            })
            .await;
        router
            .route(Event::Error {
                request_id: "r".into(),
                code: oasr_wire::ErrorCode::Busy,
                message: "full".into(),
            })
            .await;
        let ev = handle.finish().await.expect("no terminal event");
        assert!(matches!(ev, Event::Error { .. }), "{ev:?}");
        assert!(router.is_empty());
        assert!(
            cmd_rx.recv().await.is_none(),
            "a request that finished must not be cancelled"
        );
    }
}
