// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Per-worker router: maps `request_id` to the `tokio::mpsc::Sender<Event>`
//! that delivers events to the API-layer handler (an axum WebSocket or a
//! tonic streaming RPC).

use dashmap::DashMap;
use oasr_wire::Event;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::warn;

/// Owned per-worker; thread-safe via `Arc<...>` clones.
#[derive(Clone, Default)]
pub struct RouterActor {
    inner: Arc<DashMap<String, mpsc::Sender<Event>>>,
}

impl RouterActor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a per-request channel; returns the matching receiver.
    pub fn register(&self, request_id: String, buffer: usize) -> mpsc::Receiver<Event> {
        let (tx, rx) = mpsc::channel(buffer);
        self.inner.insert(request_id, tx);
        rx
    }

    /// Remove the channel for `request_id`.  No-op if absent.
    pub fn remove(&self, request_id: &str) {
        self.inner.remove(request_id);
    }

    /// Number of in-flight registrations.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Snapshot of all known request ids (used on worker-death failover).
    pub fn all_request_ids(&self) -> Vec<String> {
        self.inner.iter().map(|kv| kv.key().clone()).collect()
    }

    /// Route an event to the matching per-request channel.  If the event is
    /// terminal (Final/Error), the channel is dropped after delivery.
    pub async fn route(&self, event: Event) {
        let Some(rid) = event.request_id().map(|s| s.to_owned()) else {
            // Worker-broadcast events (Pong/Overloaded) are handled by the
            // caller — the router just ignores them.
            return;
        };
        let terminal = event.is_terminal();
        let sender = self.inner.get(&rid).map(|kv| kv.value().clone());
        if let Some(tx) = sender {
            if let Err(e) = tx.send(event).await {
                warn!(rid = %rid, "router: receiver dropped before delivery: {e}");
            }
        }
        if terminal {
            self.inner.remove(&rid);
        }
    }

    /// Synchronous-context variant that uses `try_send` and won't await.
    /// Used by the ZMQ reader thread (a std::thread) that doesn't have a
    /// Tokio runtime handle.
    pub fn route_blocking(&self, event: Event) {
        let Some(rid) = event.request_id().map(|s| s.to_owned()) else {
            return;
        };
        let terminal = event.is_terminal();
        let sender = self.inner.get(&rid).map(|kv| kv.value().clone());
        if let Some(tx) = sender {
            if let Err(e) = tx.try_send(event) {
                warn!(rid = %rid, "router: per-request channel full or closed: {e}");
            }
        }
        if terminal {
            self.inner.remove(&rid);
        }
    }
}
