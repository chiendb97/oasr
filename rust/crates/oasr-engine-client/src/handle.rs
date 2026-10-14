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
//! [`OfflineHandle`] is a oneshot future that resolves with the Final or
//! Error event.  All three handles arm [`CancelOnDrop`]: a client that goes
//! away is the normal case, and the engine has to hear about it or the request
//! keeps computing on a slot nobody is waiting for.

use bytes::Bytes;
use oasr_wire::{Cmd, Event};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::oneshot;
use tracing::{debug, trace};

use crate::dispatcher::CmdEnvelope;
use crate::router::RouterActor;
use crate::EventStream;

/// Carrier of a cancellation tail when the WS / gRPC stream drops early.
struct CancelOnDrop {
    request_id: String,
    cmd_tx: mpsc::Sender<CmdEnvelope>,
    router: RouterActor,
    finished: bool,
}

impl CancelOnDrop {
    fn arm(request_id: String, cmd_tx: mpsc::Sender<CmdEnvelope>, router: RouterActor) -> Self {
        Self {
            request_id,
            cmd_tx,
            router,
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
        let cancel = Cmd::Cancel {
            request_id: self.request_id.clone(),
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
                debug!(rid = %self.request_id, "dispatcher channel full; deferring cancel");
                send_when_room(self.cmd_tx.clone(), envelope);
            }
            // The dispatcher is gone, and the engine with it.
            Err(TrySendError::Closed(_)) => {
                trace!(rid = %self.request_id, "dispatcher closed; cancel dropped");
            }
        }
        self.router.remove(&self.request_id);
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
        request_id: String,
        events: EventStream,
        cmd_tx: mpsc::Sender<CmdEnvelope>,
        router: RouterActor,
    ) -> Self {
        Self {
            _cancel: Arc::new(parking_lot::Mutex::new(CancelOnDrop::arm(
                request_id.clone(),
                cmd_tx.clone(),
                router,
            ))),
            request_id,
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
        request_id: String,
        events: EventStream,
        cmd_tx: mpsc::Sender<CmdEnvelope>,
        router: RouterActor,
    ) -> Self {
        Self {
            _cancel: Arc::new(parking_lot::Mutex::new(CancelOnDrop::arm(
                request_id.clone(),
                cmd_tx,
                router,
            ))),
            request_id,
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
pub struct OfflineHandle {
    pub request_id: String,
    rx: oneshot::Receiver<Event>,
    cancel: CancelOnDrop,
}

impl OfflineHandle {
    pub(crate) fn new(
        request_id: String,
        rx: oneshot::Receiver<Event>,
        cmd_tx: mpsc::Sender<CmdEnvelope>,
        router: RouterActor,
    ) -> Self {
        Self {
            cancel: CancelOnDrop::arm(request_id.clone(), cmd_tx, router),
            request_id,
            rx,
        }
    }

    /// Await the final result event.
    ///
    /// The guard is disarmed only once the terminal event is in hand.  Dropping
    /// this future before then — which is what a client disconnect does — leaves
    /// it armed and cancels the request.
    pub async fn finish(self) -> Result<Event, oneshot::error::RecvError> {
        let Self { rx, mut cancel, .. } = self;
        let ev = rx.await;
        cancel.disarm();
        ev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

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
        drop(CancelOnDrop::arm("gone".into(), tx, RouterActor::default()));
        assert_eq!(next_cancelled(&mut rx).await, "queued-ahead");
        assert_eq!(next_cancelled(&mut rx).await, "gone");
    }

    #[tokio::test]
    async fn a_handle_dropped_off_the_runtime_still_cancels() {
        let (tx, mut rx) = full_channel();
        std::thread::spawn(move || {
            drop(CancelOnDrop::arm("gone".into(), tx, RouterActor::default()))
        })
        .join()
        .unwrap();
        assert_eq!(next_cancelled(&mut rx).await, "queued-ahead");
        assert_eq!(next_cancelled(&mut rx).await, "gone");
    }

    #[tokio::test]
    async fn a_disarmed_guard_sends_nothing() {
        let (tx, mut rx) = mpsc::channel(1);
        let mut guard = CancelOnDrop::arm("done".into(), tx, RouterActor::default());
        guard.disarm();
        drop(guard);
        assert!(
            rx.recv().await.is_none(),
            "a finished request must not be cancelled"
        );
    }
}
