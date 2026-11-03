// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Per-worker router: maps `request_id` to the `tokio::mpsc::Sender<Event>`
//! that delivers events to the API-layer handler (an axum WebSocket or a
//! tonic streaming RPC).
//!
//! # A client that reads slower than the engine emits
//!
//! The dispatcher thread must never wait on a client, so it hands events over
//! with `try_send`, and a full channel leaves it two choices per event:
//!
//! - **Drop it**, when the next event carries everything it does. That is a
//!   partial with no speech transition: its `text` is the transcript *so far*,
//!   and `/v1/realtime` computes deltas against what it last sent.
//! - **Defer it**, when nothing later repeats it: the terminal event, and a
//!   partial carrying `speech_started` / `speech_stopped`. Those are state
//!   changes, so a lost `speech_stopped` left the client in "speaking" for the
//!   rest of the stream.
//!
//! Deferred events wait in the request's [`Backlog`], which one task delivers
//! in order. Until it is empty, every later event for the request queues behind
//! it, so a partial cannot overtake a deferred `speech_started`, nor the
//! terminal a deferred `speech_stopped`. Partials fold together as they queue,
//! which keeps the backlog bounded for a client that never reads.

use dashmap::DashMap;
use metrics::Counter;
use oasr_metrics as om;
use oasr_wire::Event;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::engine_metrics::EngineLabels;

/// One registered request: its channel, and what is waiting to go down it.
#[derive(Clone)]
struct Route {
    tx: mpsc::Sender<Event>,
    backlog: Arc<Mutex<Backlog>>,
}

impl Route {
    fn new(tx: mpsc::Sender<Event>) -> Self {
        Self {
            tx,
            backlog: Arc::new(Mutex::new(Backlog::default())),
        }
    }
}

/// Events that found the request's channel full and must still arrive, oldest
/// first.
///
/// A partial queued behind another partial is folded into it
/// ([`fold_partial`]), so this holds at most one partial between the
/// non-partials (`Accepted`, the terminal) — three events, however long the
/// client stops reading.
#[derive(Default)]
struct Backlog {
    events: VecDeque<Event>,
    /// A task is delivering `events`, and may be waiting with one it popped.
    /// While set, nothing is handed to the channel directly.
    draining: bool,
}

/// A partial with no speech transition: the next partial, or the terminal,
/// carries everything it does, so it may be dropped for a client that is
/// behind.
fn supersedable(event: &Event) -> bool {
    matches!(
        event,
        Event::Partial { speech_events, .. }
            if speech_events.as_ref().is_none_or(Vec::is_empty)
    )
}

/// Fold `newer` into `queued` when both are partials, or hand it back.
///
/// The result is the newer transcript with the speech transitions of both, in
/// order. That is what a partial means — the transcript so far, and the
/// transitions since the previous one *delivered* — so a client that sees the
/// fold ends in the state it would have reached seeing both.
fn fold_partial(queued: &mut Event, newer: Event) -> Option<Event> {
    match (queued, newer) {
        (
            Event::Partial {
                text,
                tokens,
                scores,
                speech_events,
                ..
            },
            Event::Partial {
                text: newer_text,
                tokens: newer_tokens,
                scores: newer_scores,
                speech_events: newer_events,
                ..
            },
        ) => {
            *text = newer_text;
            *tokens = newer_tokens;
            *scores = newer_scores;
            if let Some(newer_events) = newer_events {
                speech_events
                    .get_or_insert_with(Vec::new)
                    .extend(newer_events);
            }
            None
        }
        (_, newer) => Some(newer),
    }
}

/// Deliver a request's backlog in order, then stand down.
///
/// The only consumer of the backlog: it pops one event at a time and waits for
/// room, so whatever the dispatcher queues meanwhile goes out after it.
async fn drain(tx: mpsc::Sender<Event>, backlog: Arc<Mutex<Backlog>>, dropped: Counter) {
    loop {
        let next = {
            let mut b = backlog.lock();
            match b.events.pop_front() {
                Some(event) => event,
                None => {
                    b.draining = false;
                    return;
                }
            }
        };
        if tx.send(next).await.is_err() {
            // The receiver is gone, and with it every reason to deliver the
            // rest.  Later events find the channel closed and are discarded.
            let mut b = backlog.lock();
            dropped.increment(1 + b.events.len() as u64);
            b.events.clear();
            b.draining = false;
            return;
        }
    }
}

/// One request's entry in a [`RouterActor`], removed when this is dropped.
///
/// The entry used to be removed only by a terminal event or a cancel, so every
/// other way a request ends left it behind: a submit whose send failed, or
/// whose future was dropped while the command channel was full, never got a
/// handle to cancel it, and its entry (with the forwarding task waiting on it)
/// stayed for the life of the process.  Every handle owns one of these from
/// the moment the request is registered, so no exit path can skip it.
pub struct Registration {
    router: RouterActor,
    request_id: String,
}

impl Registration {
    pub fn request_id(&self) -> &str {
        &self.request_id
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        // Request ids are fresh UUIDs, so nothing else can have registered
        // the same one in the meantime.
        self.router.remove(&self.request_id);
    }
}

/// Owned per-worker; thread-safe via `Arc<...>` clones.
///
/// The two channel-pressure counters are resolved to handles at construction
/// for the same reason the dispatcher's are: this runs once per routed event,
/// and re-resolving a labelled key each time would hash and allocate on the
/// path that delivers every partial.
#[derive(Clone)]
pub struct RouterActor {
    inner: Arc<DashMap<String, Route>>,
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

    /// Register a per-request channel; returns the entry's guard and the
    /// matching receiver.
    ///
    /// The entry lives exactly as long as the [`Registration`]: it is removed
    /// when that is dropped, if a terminal event has not removed it first.
    pub fn register(
        &self,
        request_id: String,
        buffer: usize,
    ) -> (Registration, mpsc::Receiver<Event>) {
        let (tx, rx) = mpsc::channel(buffer);
        self.inner.insert(request_id.clone(), Route::new(tx));
        let registration = Registration {
            router: self.clone(),
            request_id,
        };
        (registration, rx)
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
        let route = self.inner.get(&rid).map(|kv| kv.value().clone());
        if let Some(route) = route {
            // Behind anything the dispatcher deferred, so as not to overtake it.
            let direct = {
                let mut backlog = route.backlog.lock();
                if backlog.draining {
                    self.enqueue(&mut backlog, event);
                    None
                } else {
                    Some(event)
                }
            };
            if let Some(event) = direct {
                if let Err(e) = route.tx.send(event).await {
                    warn!(rid = %rid, "router: receiver dropped before delivery: {e}");
                }
            }
        }
        if terminal {
            self.inner.remove(&rid);
        }
    }

    /// Synchronous-context variant used by the dispatcher thread (a
    /// `std::thread`), which must never block on a slow client.
    ///
    /// Uses `try_send`. On a full channel a speech-free partial is dropped and
    /// anything else is deferred to the request's backlog (see the module
    /// docs); a deferred terminal is delivered after its entry is gone.
    pub fn route_blocking(&self, event: Event) {
        let Some(rid) = event.request_id().map(|s| s.to_owned()) else {
            return;
        };
        let terminal = event.is_terminal();
        let route = self.inner.get(&rid).map(|kv| kv.value().clone());
        if let Some(route) = route {
            self.deliver_now_or_defer(&rid, &route, event);
        }
        if terminal {
            self.inner.remove(&rid);
        }
    }

    fn deliver_now_or_defer(&self, rid: &str, route: &Route, event: Event) {
        // Held across `try_send`, which never waits: the drain task takes the
        // same lock to stand down, so `draining` cannot change underneath.
        let mut backlog = route.backlog.lock();
        if backlog.draining {
            self.enqueue(&mut backlog, event);
            return;
        }
        match route.tx.try_send(event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(event)) if supersedable(&event) => {
                self.dropped.increment(1);
                debug!(rid = %rid, "router: dropped a partial (channel full)");
            }
            Err(mpsc::error::TrySendError::Full(event)) => {
                // The dispatcher thread runs inside the runtime context
                // (`Handle::enter`), so a handle is available here.
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => {
                        // A terminal is once per request; a speech partial can
                        // start a backlog every few ticks for a slow client,
                        // which the deferred counter already shows.
                        if event.is_terminal() {
                            warn!(
                                rid = %rid,
                                "router: channel full on a terminal event; deferring delivery"
                            );
                        } else {
                            debug!(
                                rid = %rid,
                                "router: channel full on a speech transition; deferring delivery"
                            );
                        }
                        self.enqueue(&mut backlog, event);
                        backlog.draining = true;
                        handle.spawn(drain(
                            route.tx.clone(),
                            route.backlog.clone(),
                            self.dropped.clone(),
                        ));
                    }
                    Err(_) => {
                        self.dropped.increment(1);
                        warn!(
                            rid = %rid,
                            terminal = event.is_terminal(),
                            "router: channel full and no runtime handle to defer to; event lost"
                        );
                    }
                }
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.dropped.increment(1);
                debug!(rid = %rid, "router: receiver gone; event discarded");
            }
        }
    }

    /// Queue `event` behind what is already deferred, folding a partial into
    /// a partial at the tail.  The folded partial's transcript is superseded,
    /// which counts as a drop; its speech transitions carry over.
    fn enqueue(&self, backlog: &mut Backlog, event: Event) {
        self.deferred.increment(1);
        let event = match backlog.events.back_mut() {
            Some(tail) => fold_partial(tail, event),
            None => Some(event),
        };
        match event {
            Some(event) => backlog.events.push_back(event),
            None => self.dropped.increment(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_entry_lives_as_long_as_its_registration() {
        let router = RouterActor::default();
        let (registration, _events) = router.register("r".into(), 1);
        assert_eq!(router.all_request_ids(), vec!["r".to_string()]);
        drop(registration);
        assert!(router.is_empty());
    }

    /// The terminal event removes the entry first; the registration's drop
    /// then has nothing left to do, and must not mind.
    #[tokio::test]
    async fn a_terminal_event_then_the_drop_is_fine() {
        let router = RouterActor::default();
        let (registration, mut events) = router.register("r".into(), 1);
        router
            .route(Event::Error {
                request_id: "r".into(),
                code: oasr_wire::ErrorCode::Shutdown,
                message: "bye".into(),
            })
            .await;
        assert!(router.is_empty());
        assert!(events.recv().await.is_some());
        assert!(
            events.recv().await.is_none(),
            "the channel should be closed"
        );
        drop(registration);
        assert!(router.is_empty());
    }

    // -- a full channel (audit L9, G15) ---------------------------------------
    //
    // These run on a current-thread runtime, so a drain task spawned by
    // `route_blocking` does not run until the test awaits: the dispatcher's
    // side of a race is played out first, deterministically.

    use oasr_wire::SpeechEvent;
    use std::time::Duration;

    fn transition(kind: &str, time: f32) -> SpeechEvent {
        SpeechEvent {
            kind: kind.into(),
            time,
        }
    }

    fn partial(text: &str, speech: Vec<SpeechEvent>) -> Event {
        Event::Partial {
            request_id: "r".into(),
            text: text.into(),
            tokens: vec![],
            scores: None,
            speech_events: (!speech.is_empty()).then_some(speech),
        }
    }

    fn final_event(text: &str) -> Event {
        Event::Final {
            request_id: "r".into(),
            text: text.into(),
            tokens: vec![],
            scores: None,
            nbest_texts: None,
            end_time_s: None,
            words: None,
            confidence: None,
            finish_reason: None,
            speech_events: None,
            segments: None,
            no_speech_prob: None,
            endpoint_reason: None,
        }
    }

    /// Everything the client would read, up to the channel closing.
    async fn read_all(events: &mut mpsc::Receiver<Event>) -> Vec<Event> {
        let mut out = Vec::new();
        loop {
            match tokio::time::timeout(Duration::from_secs(5), events.recv()).await {
                Ok(Some(event)) => out.push(event),
                Ok(None) => return out,
                Err(_) => panic!("the channel neither delivered nor closed; read {out:?}"),
            }
        }
    }

    fn transitions(events: &[Event]) -> Vec<(String, f32)> {
        events
            .iter()
            .flat_map(|e| match e {
                Event::Partial { speech_events, .. } | Event::Final { speech_events, .. } => {
                    speech_events.clone().unwrap_or_default()
                }
                _ => vec![],
            })
            .map(|t| (t.kind, t.time))
            .collect()
    }

    fn text(event: &Event) -> &str {
        match event {
            Event::Partial { text, .. } | Event::Final { text, .. } => text,
            other => panic!("not a transcript event: {other:?}"),
        }
    }

    fn backlog_len(router: &RouterActor, rid: &str) -> usize {
        router.inner.get(rid).unwrap().backlog.lock().events.len()
    }

    /// A speech transition rides on a partial, and nothing later repeats it:
    /// dropped with the partial, `speech_stopped` left the client "speaking".
    #[tokio::test]
    async fn a_speech_transition_on_a_full_channel_still_arrives() {
        let router = RouterActor::default();
        let (_registration, mut events) = router.register("r".into(), 1);
        router.route_blocking(partial("a", vec![])); // fills the channel
        router.route_blocking(partial("a b", vec![transition("speech_started", 0.5)]));
        router.route_blocking(partial("a b c", vec![]));
        router.route_blocking(partial("a b c", vec![transition("speech_stopped", 1.5)]));
        router.route_blocking(final_event("a b c d"));

        let got = read_all(&mut events).await;
        assert_eq!(
            transitions(&got),
            vec![
                ("speech_started".to_string(), 0.5),
                ("speech_stopped".to_string(), 1.5)
            ],
            "read {got:?}"
        );
        assert_eq!(text(&got[0]), "a");
        assert!(matches!(got.last(), Some(Event::Final { .. })), "{got:?}");
        assert_eq!(
            text(&got[got.len() - 2]),
            "a b c",
            "the newest partial text"
        );
    }

    /// The slot frees before the deferred partial's task has run.  Handed over
    /// directly, the next partial would arrive first; and a `speech_stopped`
    /// deferred behind a `speech_started` could arrive before it.
    #[tokio::test]
    async fn nothing_overtakes_a_deferred_transition() {
        let router = RouterActor::default();
        let (_registration, mut events) = router.register("r".into(), 1);
        router.route_blocking(partial("a", vec![]));
        router.route_blocking(partial("a b", vec![transition("speech_started", 0.5)]));
        assert_eq!(text(&events.try_recv().unwrap()), "a"); // the slot is free
        router.route_blocking(partial("a b c", vec![]));

        let next = tokio::time::timeout(Duration::from_secs(5), events.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            transitions(std::slice::from_ref(&next)),
            vec![("speech_started".to_string(), 0.5)],
            "a later partial overtook the deferred transition: {next:?}"
        );
        assert_eq!(text(&next), "a b c", "folded into the newest transcript");
    }

    /// Partials fold as they queue, so a client that stops reading costs one
    /// partial's worth of backlog, not one per tick — and loses no transition.
    #[tokio::test]
    async fn a_client_that_never_reads_holds_a_bounded_backlog() {
        let router = RouterActor::default();
        let (_registration, mut events) = router.register("r".into(), 1);
        router.route_blocking(partial("", vec![]));
        let n = 1000;
        for i in 0..n {
            let kind = if i % 2 == 0 {
                "speech_started"
            } else {
                "speech_stopped"
            };
            router.route_blocking(partial(&format!("w{i}"), vec![transition(kind, i as f32)]));
            router.route_blocking(partial(&format!("w{i}"), vec![])); // and plain partials between
        }
        assert_eq!(backlog_len(&router, "r"), 1);
        router.route_blocking(final_event("done"));

        let got = read_all(&mut events).await;
        let seen = transitions(&got);
        assert_eq!(seen.len(), n, "transitions were lost");
        assert!(
            seen.iter().enumerate().all(|(i, (kind, time))| {
                *time == i as f32 && (kind == "speech_started") == (i % 2 == 0)
            }),
            "transitions out of order"
        );
        assert_eq!(text(&got[got.len() - 2]), format!("w{}", n - 1));
        assert!(matches!(got.last(), Some(Event::Final { .. })));
    }

    /// The cheap case stays cheap: a partial with no transition is superseded
    /// by the next one, so it is dropped, not deferred.
    #[tokio::test]
    async fn a_speech_free_partial_is_still_dropped() {
        let router = RouterActor::default();
        let (_registration, mut events) = router.register("r".into(), 1);
        router.route_blocking(partial("a", vec![]));
        router.route_blocking(partial("a b", vec![]));
        assert!(!router.inner.get("r").unwrap().backlog.lock().draining);
        assert_eq!(text(&events.try_recv().unwrap()), "a");
        tokio::task::yield_now().await;
        assert!(events.try_recv().is_err(), "the partial was deferred");
    }

    /// A drain task whose client has gone discards what it holds and stands
    /// down, rather than waiting on a channel nobody reads.
    #[tokio::test]
    async fn a_drain_stands_down_when_its_receiver_goes() {
        let router = RouterActor::default();
        let (_registration, events) = router.register("r".into(), 1);
        router.route_blocking(partial("a", vec![]));
        router.route_blocking(partial("a b", vec![transition("speech_started", 0.5)]));
        let route = router.inner.get("r").unwrap().value().clone();
        assert!(route.backlog.lock().draining);
        drop(events);
        tokio::time::timeout(Duration::from_secs(5), async {
            while route.backlog.lock().draining {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the drain task kept waiting on a closed channel");
        assert!(route.backlog.lock().events.is_empty());
        // And the dispatcher's next event is discarded, not queued.
        router.route_blocking(partial("a b c", vec![transition("speech_stopped", 1.0)]));
        assert!(route.backlog.lock().events.is_empty());
    }
}
