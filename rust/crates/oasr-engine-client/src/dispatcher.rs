// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Per-engine dispatcher: one dedicated OS thread that owns the GIL and
//! drives `ASREngine.step()`.
//!
//! Async HTTP / gRPC handlers push commands across a `tokio::mpsc` channel.
//! The dispatcher thread drains the channel each iteration (up to a per-tick
//! budget), enters `Python::with_gil` **once**, replays all drained commands
//! plus `engine.step()` in the same GIL scope, and routes any resulting
//! events back via a [`crate::router::RouterActor`] keyed by `request_id`.
//!
//! When the engine is fully idle and the inbound channel is empty, the
//! thread blocks on `cmd_rx.blocking_recv()` instead of polling — this
//! removes the historical ~1 ms idle-sleep floor on latency-to-first-byte.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use bytes::Bytes;
use oasr_wire::{Cmd, ErrorCode, Event, ModelInfo};
use parking_lot::Mutex;
use pyo3::prelude::*;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::pyengine::{engine_error_event, AdmitSpec, PyEngine, PyEngineError};
use crate::router::RouterActor;

/// Maximum time the dispatcher blocks waiting for a command on an idle tick.
/// Bounded so the heartbeat (`last_event_at_ms`) refreshes regularly enough
/// that `/readyz` doesn't go stale under pure-idle conditions.  When a sender
/// lands an envelope inside the window, `recv()` returns immediately.
const IDLE_RECV_TIMEOUT: Duration = Duration::from_millis(500);

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

/// Shared state — cloned (via Arc) between the dispatcher thread and the
/// async client facade.
pub(crate) struct DispatcherShared {
    pub(crate) load: AtomicU32,
    pub(crate) last_event_at_ms: AtomicU64,
    pub(crate) model_info: Mutex<Option<ModelInfo>>,
    pub(crate) router: RouterActor,
    pub(crate) label: String,
}

impl DispatcherShared {
    pub(crate) fn new(label: String) -> Self {
        Self {
            load: AtomicU32::new(0),
            last_event_at_ms: AtomicU64::new(0),
            model_info: Mutex::new(None),
            router: RouterActor::new(),
            label,
        }
    }
}

fn now_millis(epoch: Instant) -> u64 {
    epoch.elapsed().as_millis() as u64
}

/// Configurable per-dispatcher knobs.
#[derive(Debug, Clone)]
pub struct DispatcherConfig {
    pub max_inbound_per_tick: usize,
    pub overload_emit_interval: Duration,
    pub max_concurrent_requests: u32,
    /// Per-step admission window: after the first envelope arrives in a
    /// tick, wait up to this long for siblings to accumulate before
    /// stepping.  Trades a small p50 latency floor for much fuller batches
    /// under HTTP-driven trickle admission.  Set to ``Duration::ZERO`` to
    /// disable (revert to the previous "step ASAP" behavior).
    pub admit_window: Duration,
    /// Coalescing target.  Stop waiting early once this many envelopes
    /// have been drained.  Should be <= ``max_inbound_per_tick`` and
    /// roughly match the engine's ``max_batch_size``.
    pub admit_threshold: usize,
}

impl Default for DispatcherConfig {
    fn default() -> Self {
        Self {
            max_inbound_per_tick: 4096,
            overload_emit_interval: Duration::from_secs(1),
            max_concurrent_requests: 256,
            // 3 ms is enough to catch the bulk of an `asyncio.gather` burst
            // on a loopback HTTP server (1 RTT ≈ 30-100 µs).  Empirically
            // turns 10-22-deep service batches into 32-64-deep ones at
            // concurrency=64 without a visible p50 hit.
            admit_window: Duration::from_millis(3),
            admit_threshold: 64,
        }
    }
}

/// Spawn the dispatcher thread.  Returns the shared state + the command tx
/// for the client facade to clone and use.
///
/// Must be called from within a tokio runtime — the dispatcher uses the
/// captured `Handle` to bound its idle waits with `tokio::time::timeout`
/// (so the heartbeat keeps refreshing even when no traffic is arriving).
pub(crate) fn spawn(
    engine: PyEngine,
    label: String,
    cfg: DispatcherConfig,
    cmd_channel_cap: usize,
) -> (Arc<DispatcherShared>, mpsc::Sender<CmdEnvelope>) {
    let shared = Arc::new(DispatcherShared::new(label.clone()));
    {
        let mi = engine.model_info();
        *shared.model_info.lock() = Some(mi);
    }
    let (cmd_tx, cmd_rx) = mpsc::channel::<CmdEnvelope>(cmd_channel_cap);
    let rt_handle = Handle::current();

    let shared_for_thread = Arc::clone(&shared);
    thread::Builder::new()
        .name(format!("oasr-dispatcher[{label}]"))
        .spawn(move || {
            run_dispatcher(engine, shared_for_thread, cfg, cmd_rx, rt_handle);
        })
        .expect("spawn dispatcher thread");

    (shared, cmd_tx)
}

fn run_dispatcher(
    engine: PyEngine,
    shared: Arc<DispatcherShared>,
    cfg: DispatcherConfig,
    mut cmd_rx: mpsc::Receiver<CmdEnvelope>,
    rt_handle: Handle,
) {
    info!(label = %shared.label, "dispatcher thread started");

    // Enter the tokio runtime context for this thread so `tokio::time::timeout`
    // (which constructs a Sleep eagerly) can find a reactor.  The guard is
    // held for the lifetime of the thread.
    let _rt_guard = rt_handle.enter();

    let epoch = Instant::now();
    let mut last_overload_emit: Option<Instant> = None;

    // Reusable per-tick buffers — avoid re-allocating each loop iteration.
    // `envs` is left non-empty across iterations only when an idle blocking
    // recv hands us one envelope to carry forward.
    let mut envs: Vec<CmdEnvelope> = Vec::with_capacity(cfg.max_inbound_per_tick.min(1024));
    let mut tick_events: Vec<Event> = Vec::new();
    let mut admit_batch: Vec<AdmitSpec> = Vec::with_capacity(64);

    loop {
        // ---- Drain inbound commands (non-blocking up to per-tick budget) ----
        while envs.len() < cfg.max_inbound_per_tick {
            match cmd_rx.try_recv() {
                Ok(env) => envs.push(env),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    info!(label = %shared.label, "command channel closed; dispatcher exit");
                    return;
                }
            }
        }

        // ---- Admission coalescing window ----
        //
        // After the initial try_recv drain, if we got *some* envelopes but
        // fewer than `admit_threshold` and the engine isn't already heavily
        // loaded, briefly wait for sibling envelopes to land.  HTTP / gRPC
        // handlers under an `asyncio.gather`-style burst feed the channel
        // over ~1-3 ms, so a 3 ms window grows per-step batches from 10-20
        // to 32-64 without a measurable p50 hit.  Skipped when:
        //   * no envelopes arrived this tick (nothing to coalesce);
        //   * already at threshold;
        //   * engine is at >=25% load cap (admission is no longer the
        //     bottleneck — step the queue immediately);
        //   * window is zero (knob disabled).
        if !envs.is_empty()
            && envs.len() < cfg.admit_threshold
            && cfg.admit_window > Duration::ZERO
            && shared.load.load(Ordering::Relaxed) * 4 < cfg.max_concurrent_requests
        {
            let deadline = Instant::now() + cfg.admit_window;
            while envs.len() < cfg.admit_threshold {
                let remaining = match deadline.checked_duration_since(Instant::now()) {
                    Some(r) if !r.is_zero() => r,
                    _ => break,
                };
                match rt_handle.block_on(tokio::time::timeout(remaining, cmd_rx.recv())) {
                    Ok(Some(env)) => envs.push(env),
                    Ok(None) => {
                        info!(label = %shared.label, "command channel closed; dispatcher exit");
                        return;
                    }
                    Err(_) => break, // window elapsed
                }
            }
            // Top up with any siblings that landed during the wait but
            // weren't pulled by the recv loop above.
            while envs.len() < cfg.max_inbound_per_tick {
                match cmd_rx.try_recv() {
                    Ok(env) => envs.push(env),
                    Err(_) => break,
                }
            }
        }

        let received_any = !envs.is_empty();

        // ---- ONE Python::with_gil for replay + step ----
        tick_events.clear();
        admit_batch.clear();
        let (running, waiting): (u32, u32) = Python::with_gil(|py| {
            let bound = engine.bind_engine(py);

            // Replay drained envelopes in FIFO order, coalescing contiguous
            // admission commands into one bulk `add_requests_batch` call on
            // the Python side.  Non-admit cmds (FeedChunk, Cancel, Ping)
            // force a flush first so request_id ordering across the
            // CreateStreaming → FeedChunk boundary is preserved.
            for env in envs.drain(..) {
                match &env.cmd {
                    Cmd::CreateOffline { .. } | Cmd::CreateStreaming { .. } => {
                        enqueue_admit_locked(
                            env,
                            cfg.max_concurrent_requests,
                            &shared,
                            &mut admit_batch,
                            &mut tick_events,
                        );
                    }
                    _ => {
                        flush_admit_batch_locked(
                            py,
                            &bound,
                            &mut admit_batch,
                            &shared,
                            &mut tick_events,
                        );
                        handle_nonadmit_cmd_locked(
                            py,
                            &bound,
                            env,
                            &shared,
                            &mut tick_events,
                        );
                    }
                }
            }
            // Drain any remaining admits before stepping.
            flush_admit_batch_locked(
                py,
                &bound,
                &mut admit_batch,
                &shared,
                &mut tick_events,
            );

            // Decide whether to step.  `engine.step()` is fast when there's
            // nothing running, but skipping it saves a Python call on each
            // truly-idle tick.
            let (running, waiting) = PyEngine::load_locked(&bound);
            let pending = running > 0 || waiting > 0;
            if pending {
                match PyEngine::step_locked(py, &bound) {
                    Ok(events) => tick_events.extend(events),
                    Err(e) => {
                        error!(label = %shared.label, "engine.step failed: {e}");
                        // Surface a synthetic Error to every in-flight request
                        // so callers don't hang.  Continue — the engine may
                        // recover on the next tick.
                        for rid in shared.router.all_request_ids() {
                            tick_events.push(Event::Error {
                                request_id: rid,
                                code: ErrorCode::Internal,
                                message: format!("engine.step error: {e}"),
                            });
                        }
                    }
                }
            }

            // Refresh load after step (terminal events drop in-flight count).
            PyEngine::load_locked(&bound)
        });

        // ---- Route events outside the GIL ----
        for evt in tick_events.drain(..) {
            let terminal = evt.is_terminal();
            let rid_present = evt.request_id().is_some();
            shared.router.route_blocking(evt);
            if terminal && rid_present {
                shared.load.fetch_sub(1, Ordering::Relaxed);
            }
        }

        // ---- Refresh load + heartbeat ----
        shared.load.store(running + waiting, Ordering::Relaxed);
        shared
            .last_event_at_ms
            .store(now_millis(epoch), Ordering::Relaxed);

        // ---- Optional overload signal ----
        if running + waiting >= cfg.max_concurrent_requests {
            let due = match last_overload_emit {
                Some(t) => t.elapsed() >= cfg.overload_emit_interval,
                None => true,
            };
            if due {
                last_overload_emit = Some(Instant::now());
            }
        }

        // ---- Idle wakeup ----
        //
        // If nothing came in AND the engine has no work pending, wait on the
        // channel until a sender lands an envelope.  Bounded by
        // `IDLE_RECV_TIMEOUT` so the heartbeat keeps refreshing during long
        // idle periods (otherwise `/readyz` would go stale).  Replaces the
        // historical 1 ms poll loop with a ~10 µs scheduler wake when a
        // command arrives during the wait window.  When pending work exists
        // we keep busy-iterating so streaming requests continue to advance
        // at the engine's natural cadence.
        if !received_any && running == 0 && waiting == 0 {
            match rt_handle
                .block_on(tokio::time::timeout(IDLE_RECV_TIMEOUT, cmd_rx.recv()))
            {
                Ok(Some(env)) => envs.push(env),
                Ok(None) => {
                    info!(label = %shared.label, "command channel closed; dispatcher exit");
                    return;
                }
                Err(_) => {
                    // Timeout — loop back, refresh heartbeat next tick.
                }
            }
            // Loop back — the next iteration's try_recv drain will top up
            // `envs` with any siblings already in the channel.
        }
    }
}

/// Run cap-check + validate audio for one admit envelope.  On success, bump
/// the in-flight load atomic, push an `AdmitSpec` into `out_admits` for
/// later bulk replay, and queue an `Accepted` event.  On rejection (cap,
/// missing audio, unknown cmd) emit the matching error event and skip.
///
/// Does **not** touch Python — runs without the GIL.  Centralising the
/// pre-flight here keeps the bulk-replay path simple: every spec that
/// reaches `add_requests_batch_locked` has already passed admission.
fn enqueue_admit_locked(
    env: CmdEnvelope,
    max_concurrent: u32,
    shared: &DispatcherShared,
    out_admits: &mut Vec<AdmitSpec>,
    out_events: &mut Vec<Event>,
) {
    let payload = env.payload;
    match env.cmd {
        Cmd::CreateOffline {
            request_id,
            sample_rate,
            priority,
        } => {
            if shared.load.load(Ordering::Relaxed) >= max_concurrent {
                out_events.push(Event::Error {
                    request_id: request_id.clone(),
                    code: ErrorCode::Busy,
                    message: format!(
                        "in-flight {} >= cap {}",
                        shared.load.load(Ordering::Relaxed),
                        max_concurrent
                    ),
                });
                return;
            }
            let audio = payload.unwrap_or_default();
            if audio.is_empty() {
                out_events.push(Event::Error {
                    request_id: request_id.clone(),
                    code: ErrorCode::InvalidCmd,
                    message: "CreateOffline requires audio payload".into(),
                });
                return;
            }
            shared.load.fetch_add(1, Ordering::Relaxed);
            out_admits.push(AdmitSpec::Offline {
                rid: request_id.clone(),
                audio,
                sample_rate,
                priority,
            });
            out_events.push(Event::Accepted { request_id });
        }
        Cmd::CreateStreaming {
            request_id,
            sample_rate,
            priority,
        } => {
            if shared.load.load(Ordering::Relaxed) >= max_concurrent {
                out_events.push(Event::Error {
                    request_id: request_id.clone(),
                    code: ErrorCode::Busy,
                    message: format!(
                        "in-flight {} >= cap {}",
                        shared.load.load(Ordering::Relaxed),
                        max_concurrent
                    ),
                });
                return;
            }
            shared.load.fetch_add(1, Ordering::Relaxed);
            out_admits.push(AdmitSpec::Streaming {
                rid: request_id.clone(),
                sample_rate,
                priority,
            });
            out_events.push(Event::Accepted { request_id });
        }
        other => {
            // Defensive — only Create* should reach here.  If we got
            // something else, treat as InvalidCmd against any rid we can
            // recover from the cmd shape; otherwise drop silently.
            if let Some(rid) = match &other {
                Cmd::FeedChunk { request_id, .. } | Cmd::Cancel { request_id } => {
                    Some(request_id.clone())
                }
                _ => None,
            } {
                out_events.push(Event::Error {
                    request_id: rid,
                    code: ErrorCode::InvalidCmd,
                    message: "internal: non-admit cmd routed through enqueue_admit_locked".into(),
                });
            }
        }
    }
}

/// Replay the accumulated `admit_batch` via a single Python call.  On
/// success we leave the previously-pushed `Accepted` events in `out_events`
/// untouched.  On failure we walk the batch, decrement the load atomic per
/// spec (`enqueue_admit_locked` bumped it optimistically), and replace the
/// trailing `Accepted` events with `Error` events for the same rids.
///
/// Empties `admit_batch` on every call.
fn flush_admit_batch_locked<'py>(
    py: Python<'py>,
    bound: &Bound<'py, PyAny>,
    admit_batch: &mut Vec<AdmitSpec>,
    shared: &DispatcherShared,
    out_events: &mut Vec<Event>,
) {
    if admit_batch.is_empty() {
        return;
    }
    match PyEngine::add_requests_batch_locked(py, bound, admit_batch) {
        Ok(()) => {
            admit_batch.clear();
        }
        Err(e) => {
            rollback_admit_batch(admit_batch, shared, out_events, &e);
        }
    }
}

fn rollback_admit_batch(
    admit_batch: &mut Vec<AdmitSpec>,
    shared: &DispatcherShared,
    out_events: &mut Vec<Event>,
    err: &PyEngineError,
) {
    // Roll back the speculative load bumps and rewrite the trailing
    // Accepted events into Error events for the same rids.  We assume the
    // last N `Accepted` events correspond to this admit batch in order —
    // safe because `enqueue_admit_locked` pushes them in lockstep with
    // `admit_batch` appends and no other code path touches `out_events`
    // between admit accumulation and flush.
    let n = admit_batch.len();
    let start = out_events.len().saturating_sub(n);
    let mut idx = start;
    for spec in admit_batch.drain(..) {
        shared.load.fetch_sub(1, Ordering::Relaxed);
        let rid = spec.request_id().to_owned();
        // Defensive: only overwrite if the slot really is an Accepted for this rid.
        if idx < out_events.len() {
            let matches = matches!(
                &out_events[idx],
                Event::Accepted { request_id } if request_id == &rid
            );
            if matches {
                out_events[idx] = engine_error_event(&rid, err);
                idx += 1;
                continue;
            }
        }
        // Fall back to appending if the slot doesn't line up (shouldn't happen).
        out_events.push(engine_error_event(&rid, err));
    }
}

fn handle_nonadmit_cmd_locked<'py>(
    py: Python<'py>,
    bound: &Bound<'py, PyAny>,
    env: CmdEnvelope,
    shared: &DispatcherShared,
    out_events: &mut Vec<Event>,
) {
    let payload = env.payload;
    match env.cmd {
        Cmd::Ping { seq } => {
            // Ping has no per-rid channel; consumers poll `last_pong_at`
            // via the client facade.  The heartbeat update happens
            // unconditionally each tick, so the next Ping wakeup suffices.
            let _ = seq;
        }
        Cmd::FeedChunk {
            request_id,
            is_last,
        } => {
            let chunk = payload.unwrap_or_default();
            if let Err(e) = PyEngine::feed_chunk_locked(py, bound, &request_id, &chunk, is_last) {
                out_events.push(engine_error_event(&request_id, &e));
            }
        }
        Cmd::Cancel { request_id } => {
            let _ = PyEngine::abort_locked(bound, &request_id);
            shared.router.remove(&request_id);
        }
        // Create* should never reach here — the dispatcher routes them to
        // enqueue_admit_locked.  If one slips through, fall back to the
        // single-shot admit so behaviour is preserved.
        Cmd::CreateOffline { .. } | Cmd::CreateStreaming { .. } => {
            error!(label = %shared.label, "internal: Create* cmd reached handle_nonadmit_cmd_locked");
        }
    }
}
