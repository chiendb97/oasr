// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Per-request handles returned by [`EngineClient`] / [`EnginePool`].
//!
//! [`StreamingHandle`] holds the audio sender + event receiver for a
//! streaming session; dropping it sends a `Cmd::Cancel` to the worker.
//!
//! [`OfflineHandle`] is a oneshot future that resolves with the Final or
//! Error event.

use bytes::Bytes;
use oasr_wire::{Cmd, Event};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::trace;

use crate::client::CmdEnvelope;
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
        // Best effort — if the worker has already died, the cmd_tx is closed.
        if let Err(e) = self.cmd_tx.try_send(envelope) {
            trace!(rid = %self.request_id, "could not send cancel on drop: {e}");
        }
        self.router.remove(&self.request_id);
    }
}

/// Streaming request handle: push audio chunks, pull events.
pub struct StreamingHandle {
    pub request_id: String,
    audio_tx: mpsc::Sender<(Bytes, bool)>,
    pub events: EventStream,
    _cancel: Arc<parking_lot::Mutex<CancelOnDrop>>,
}

impl StreamingHandle {
    pub(crate) fn new(
        request_id: String,
        audio_tx: mpsc::Sender<(Bytes, bool)>,
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
            audio_tx,
            events,
        }
    }

    /// Push one audio chunk with `is_last=false`.
    pub async fn push_chunk(&self, audio: Bytes) -> Result<(), bytes::Bytes> {
        self.audio_tx.send((audio, false)).await.map_err(|e| e.0 .0)
    }

    /// Push the final audio chunk with `is_last=true` and disarm cancel-on-drop.
    /// Pass an empty `Bytes` if the caller already exhausted audio.  The
    /// envelope is enqueued behind any in-flight `push_chunk` calls so
    /// ordering is preserved.
    pub async fn flush_last(&self, audio: Bytes) -> Result<(), bytes::Bytes> {
        let result = self.audio_tx.send((audio, true)).await.map_err(|e| e.0 .0);
        self.finish();
        result
    }

    /// Mark the handle as completed so dropping it won't emit a Cancel.
    pub fn finish(&self) {
        self._cancel.lock().disarm();
    }
}

/// Offline request handle: await a single final result.
pub struct OfflineHandle {
    pub request_id: String,
    rx: oneshot::Receiver<Event>,
}

impl OfflineHandle {
    pub(crate) fn new(request_id: String, rx: oneshot::Receiver<Event>) -> Self {
        Self { request_id, rx }
    }

    /// Await the final result event.
    pub async fn finish(self) -> Result<Event, oneshot::error::RecvError> {
        self.rx.await
    }
}
