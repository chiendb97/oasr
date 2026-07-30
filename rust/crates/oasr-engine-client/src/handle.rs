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
use tokio::sync::oneshot;
use tracing::trace;

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
        // Best effort — if the dispatcher has gone away, cmd_tx is closed.
        if let Err(e) = self.cmd_tx.try_send(envelope) {
            trace!(rid = %self.request_id, "could not send cancel on drop: {e}");
        }
        self.router.remove(&self.request_id);
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
/// The audio arrives in one shot (an offline-only decode family cannot consume a
/// growing buffer), but the *text* comes out incrementally: the autoregressive
/// strategies emit one `Event::Partial` per advanced request per engine tick, so a
/// client can render tokens as they are generated. This is what
/// [`OfflineHandle`] cannot express — it resolves once, with the final.
///
/// Cancel-on-drop is armed exactly as for a streaming request, and matters more
/// here: a client that disconnects mid-generation would otherwise leave an AR row
/// occupying a decode slot until it hits its `max_new_tokens` cap.
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
