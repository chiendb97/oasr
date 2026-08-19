// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Per-worker router: maps `request_id` to the `tokio::mpsc::Sender<Event>`
//! that delivers events to the API-layer handler (an axum WebSocket or a
//! tonic streaming RPC).

use dashmap::DashMap;
use metrics::Counter;
use oasr_metrics as om;
use oasr_wire::Event;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::engine_metrics::EngineLabels;

/// Owned per-worker; thread-safe via `Arc<...>` clones.
///
/// The two channel-pressure counters are resolved to handles at construction
/// for the same reason the dispatcher's are: this runs once per routed event,
/// and re-resolving a labelled key each time would hash and allocate on the
/// path that delivers every partial.
#[derive(Clone)]
pub struct RouterActor {
    inner: Arc<DashMap<String, mpsc::Sender<Event>>>,
    dropped: Counter,
    deferred: Counter,
}

impl Default for RouterActor {
    fn default() -> Self {
        Self::new(EngineLabels::none())
    }
}

impl RouterActor {
    pub fn new(labels: EngineLabels) -> Self {
        let l = || labels.iter();
        Self {
            inner: Arc::new(DashMap::new()),
            dropped: metrics::counter!(om::EVENTS_DROPPED, l()),
            deferred: metrics::counter!(om::EVENTS_DEFERRED, l()),
        }
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

    /// Synchronous-context variant used by the dispatcher thread (a
    /// `std::thread`), which must never block on a slow client.
    ///
    /// Uses `try_send`: partials may be dropped when full, but terminal events
    /// are handed to a background task and registration persists until delivery.
    pub fn route_blocking(&self, event: Event) {
        let Some(rid) = event.request_id().map(|s| s.to_owned()) else {
            return;
        };
        let terminal = event.is_terminal();
        let sender = self.inner.get(&rid).map(|kv| kv.value().clone());
        if let Some(tx) = sender {
            if let Err(e) = tx.try_send(event) {
                match e {
                    mpsc::error::TrySendError::Full(ev) if terminal => {
                        // Deliver out-of-band rather than lose the result. The
                        // dispatcher thread runs inside the runtime context
                        // (`Handle::enter`), so a handle is available here.
                        match tokio::runtime::Handle::try_current() {
                            Ok(handle) => {
                                self.deferred.increment(1);
                                warn!(
                                    rid = %rid,
                                    "router: channel full on a terminal event; deferring delivery"
                                );
                                handle.spawn(async move {
                                    let _ = tx.send(ev).await;
                                });
                            }
                            Err(_) => {
                                self.dropped.increment(1);
                                warn!(
                                    rid = %rid,
                                    "router: channel full on a terminal event and no runtime \
                                     handle to defer to; result lost"
                                );
                            }
                        }
                    }
                    mpsc::error::TrySendError::Full(_) => {
                        // A partial; the next one carries the same transcript.
                        self.dropped.increment(1);
                        debug!(rid = %rid, "router: dropped a partial (channel full)");
                    }
                    mpsc::error::TrySendError::Closed(_) => {
                        self.dropped.increment(1);
                        debug!(rid = %rid, "router: receiver gone; event discarded");
                    }
                }
            }
        }
        if terminal {
            self.inner.remove(&rid);
        }
    }
}
