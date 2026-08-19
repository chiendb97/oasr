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
use oasr_metrics as om;
use oasr_wire::{Cmd, ErrorCode, Event, ModelInfo};
use parking_lot::Mutex;
use pyo3::prelude::*;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::engine_metrics::{replay, EngineLabels, TickHandles, DRAIN_INTERVAL};
use crate::pyengine::{engine_error_event, AdmitSpec, PyEngine};
use crate::router::RouterActor;

/// Maximum time the dispatcher blocks waiting for a command on an idle tick.
/// Bounded so the heartbeat (`last_event_at_ms`) refreshes regularly enough
/// that `/readyz` doesn't go stale under pure-idle conditions.  When a sender
/// lands an envelope inside the window, `recv()` returns immediately.
const IDLE_RECV_TIMEOUT: Duration = Duration::from_millis(500);

/// How long the dispatcher waits on the command channel after a tick that did
/// no work while the engine still has requests in flight.
///
/// Without this the loop spun: the idle block below only engages when
/// `running == 0 && waiting == 0`, so a single open streaming session — which
/// spends most of its life waiting for the *client's* next 640 ms chunk — kept
/// the thread stepping an engine with nothing to do, pinning a core and
/// thrashing the GIL for the whole session.  This is a `recv` with a timeout,
/// not a sleep: an arriving `FeedChunk` wakes it in ~10 µs, so the streaming
/// path pays nothing.
const NO_WORK_BACKOFF: Duration = Duration::from_millis(2);

/// A tick faster than this, that produced nothing and received nothing, is
/// treated as a no-op and backs off.
///
/// Gating on tick *duration* is what keeps the backoff off the working paths: a
/// tick that is genuinely computing — an autoregressive decode group grinding
/// through `decode_steps_per_tick`, a paged streaming forward — costs far more
/// than this and never qualifies, even when it emits no output that tick.
const NO_WORK_TICK_MAX: Duration = Duration::from_millis(1);

/// Consecutive `engine.step()` failures after which this process stops
/// advertising readiness.  A single failure is recoverable (the offending
/// requests are errored and aborted); a run of them means the engine is wedged,
/// and staying "ready" would turn every arriving request into an INTERNAL error.
const MAX_CONSECUTIVE_STEP_FAILURES: u32 = 3;

// Metric names, kinds, help text and buckets are declared once in
// `oasr-metrics` and imported here as `om::*`.  The tick histogram is the
// load-bearing one: one engine tick holds the GIL for admit + step + extract,
// so its p99 *is* the worst-case latency a cancel, a new admission, or a
// streaming partial can experience.  Autoregressive decode makes that number
// model-dependent (a batched 7B decoder step is orders of magnitude slower than
// a CTC chunk), so it has to be observable rather than assumed.

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
    /// `{engine, model, decode_method}` for every engine-scope series.
    pub(crate) metric_labels: EngineLabels,
}

impl DispatcherShared {
    pub(crate) fn new(label: String, model: Option<&ModelInfo>) -> Self {
        let metric_labels = EngineLabels::new(&label, model);
        Self {
            load: AtomicU32::new(0),
            last_event_at_ms: AtomicU64::new(0),
            model_info: Mutex::new(model.cloned()),
            router: RouterActor::new(metric_labels.clone()),
            label,
            metric_labels,
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
    /// When true, accumulate per-tick sub-stage timings (intake, admit incl.
    /// audio→numpy, step, output extract, route) + effective batch size and
    /// log a rolling summary every ~2 s at INFO.  Off by default — pure
    /// diagnostics for the service↔engine gap decomposition.
    pub trace_dispatch: bool,
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
            trace_dispatch: false,
        }
    }
}

/// Rolling accumulator for per-tick dispatcher sub-stage timings.  Only
/// touched on ticks that actually stepped the engine; logged + reset every
/// ~2 s when `DispatcherConfig::trace_dispatch` is set.  All `*_us` are
/// summed microseconds over the window.
#[derive(Debug)]
struct DispatchTrace {
    ticks: u64,
    n_admit: u64,
    n_out: u64,
    intake_us: u64,
    admit_us: u64,
    step_us: u64,
    extract_us: u64,
    route_us: u64,
    last_log: Instant,
}

impl DispatchTrace {
    fn new() -> Self {
        Self {
            ticks: 0,
            n_admit: 0,
            n_out: 0,
            intake_us: 0,
            admit_us: 0,
            step_us: 0,
            extract_us: 0,
            route_us: 0,
            last_log: Instant::now(),
        }
    }

    fn reset(&mut self) {
        // Start a fresh ~2 s window from *now* (Self::new() stamps last_log).
        *self = Self::new();
    }

    /// Log a rolling summary every ~2 s, then reset the window.  Reports
    /// per-tick means (the natural unit — each tick is one `engine.step()`)
    /// plus the effective batch (outputs/tick) and the share of wall time the
    /// dispatcher thread spends *outside* the GPU step (intake+admit+extract+
    /// route) — i.e. the serial overhead that starves the GPU.
    fn maybe_log(&mut self, label: &str) {
        if self.last_log.elapsed() < Duration::from_secs(2) || self.ticks == 0 {
            return;
        }
        let t = self.ticks as f64;
        let intake = self.intake_us as f64 / t / 1000.0;
        let admit = self.admit_us as f64 / t / 1000.0;
        let step = self.step_us as f64 / t / 1000.0;
        let extract = self.extract_us as f64 / t / 1000.0;
        let route = self.route_us as f64 / t / 1000.0;
        let overhead = intake + admit + extract + route;
        let total = overhead + step;
        let overhead_pct = if total > 0.0 {
            100.0 * overhead / total
        } else {
            0.0
        };
        info!(
            label = %label,
            ticks = self.ticks,
            batch = format!("{:.1}", self.n_out as f64 / t),
            admit_per_tick = format!("{:.1}", self.n_admit as f64 / t),
            "dispatch[ms/tick]: intake={intake:.2} admit={admit:.2} step={step:.2} \
             extract={extract:.2} route={route:.2} | non-step overhead={overhead:.2}ms \
             ({overhead_pct:.0}% of {total:.2}ms/tick)"
        );
        self.reset();
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
    // The model has to be known *before* the shared state is built: it
    // supplies two of the three engine-scope label values, and a handle
    // resolved without them would publish into a different series than every
    // later sample.
    let model_info = engine.model_info();
    let shared = Arc::new(DispatcherShared::new(label.clone(), Some(&model_info)));
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

    let trace_enabled = cfg.trace_dispatch;
    let mut trace = DispatchTrace::new();
    // Run of consecutive `engine.step()` failures; see the step-failure block
    // below.  Reaching MAX_CONSECUTIVE_STEP_FAILURES stops the readiness
    // heartbeat so this process is drained instead of erroring every request.
    let mut consecutive_step_failures: u32 = 0;
    // Resolved once: an engine's labels never change, and the loop below
    // records into these handles up to a thousand times a second on the thread
    // that owns the GIL.
    let h = TickHandles::new(&shared.metric_labels);
    let mut next_drain = Instant::now();

    loop {
        // ---- Drain inbound commands (non-blocking up to per-tick budget) ----
        let intake_t0 = Instant::now();
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

        // Briefly wait for sibling admissions when below threshold and lightly
        // loaded. Never coalesce other commands because their latency takes priority.
        if should_coalesce(&envs, &cfg, shared.load.load(Ordering::Relaxed)) {
            let deadline = Instant::now() + cfg.admit_window;
            while envs.len() < cfg.admit_threshold {
                let remaining = match deadline.checked_duration_since(Instant::now()) {
                    Some(r) if !r.is_zero() => r,
                    _ => break,
                };
                match rt_handle.block_on(tokio::time::timeout(remaining, cmd_rx.recv())) {
                    Ok(Some(env)) => {
                        // A non-admission command ends the window immediately.
                        let admit = is_admit(&env.cmd);
                        envs.push(env);
                        if !admit {
                            break;
                        }
                    }
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

        let t_intake = intake_t0.elapsed();
        let received_any = !envs.is_empty();
        // Count admit commands this tick *before* they're drained (effective
        // batch trace).  Cheap O(n) scan, skipped unless tracing.
        let n_admit = if trace_enabled {
            envs.iter()
                .filter(|e| {
                    matches!(
                        e.cmd,
                        Cmd::CreateOffline { .. } | Cmd::CreateStreaming { .. }
                    )
                })
                .count() as u64
        } else {
            0
        };

        // ---- ONE Python::with_gil for replay + step ----
        tick_events.clear();
        admit_batch.clear();
        let tick_t0 = Instant::now();
        #[allow(clippy::type_complexity)]
        let (running, waiting, t_admit, t_step, t_extract, n_out, step_failed, engine_snapshot): (
            u32,
            u32,
            Duration,
            Duration,
            Duration,
            u64,
            bool,
            Option<crate::engine_metrics::EngineSnapshot>,
        ) = Python::with_gil(|py| {
            let bound = engine.bind_engine(py);

            // Replay drained envelopes in FIFO order, coalescing contiguous
            // admission commands into one bulk `add_requests_batch` call on
            // the Python side.  Non-admit cmds (FeedChunk, Cancel, Ping)
            // force a flush first so request_id ordering across the
            // CreateStreaming → FeedChunk boundary is preserved.
            let admit_t0 = Instant::now();
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
                        handle_nonadmit_cmd_locked(py, &bound, env, &shared, &mut tick_events);
                    }
                }
            }
            // Drain any remaining admits before stepping.
            flush_admit_batch_locked(py, &bound, &mut admit_batch, &shared, &mut tick_events);
            let t_admit = admit_t0.elapsed();

            // Decide whether to step.  `engine.step()` is fast when there's
            // nothing running, but skipping it saves a Python call on each
            // truly-idle tick.
            let (running, waiting) = PyEngine::load_locked(&bound);
            let pending = running > 0 || waiting > 0;
            let mut t_step = Duration::ZERO;
            let mut t_extract = Duration::ZERO;
            let mut n_out = 0u64;
            let mut step_failed = false;
            if pending {
                let step_t0 = Instant::now();
                match PyEngine::step_raw(&bound) {
                    Ok(list) => {
                        t_step = step_t0.elapsed();
                        n_out = list.len() as u64;
                        let extract_t0 = Instant::now();
                        match PyEngine::extract_events(&list, &shared.metric_labels) {
                            Ok(events) => tick_events.extend(events),
                            Err(e) => {
                                let rids = shared.router.all_request_ids();
                                error!(label = %shared.label, n_affected = rids.len(), "engine.step extract failed: {e}");
                                for rid in rids {
                                    tick_events.push(Event::Error {
                                        request_id: rid,
                                        code: ErrorCode::Internal,
                                        message: format!("engine.step extract error: {e}"),
                                    });
                                }
                            }
                        }
                        t_extract = extract_t0.elapsed();
                    }
                    Err(e) => {
                        t_step = step_t0.elapsed();
                        let rids = shared.router.all_request_ids();
                        error!(label = %shared.label, n_affected = rids.len(), "engine.step failed: {e}");
                        // Surface a synthetic Error to every in-flight request so
                        // callers don't hang, **and abort them in Python**.
                        // Without the abort the offending state survives — an
                        // incremental strategy's failing decode group stays in its
                        // pending pool and raises again on the next tick, so every
                        // newly admitted request inherits the same error forever.
                        // Aborting is also what the error we just sent implies.
                        for rid in rids {
                            if let Err(abort_err) = PyEngine::abort_locked(&bound, &rid) {
                                warn!(
                                    label = %shared.label,
                                    rid = %rid,
                                    "abort after step failure did not succeed: {abort_err}"
                                );
                            }
                            tick_events.push(Event::Error {
                                request_id: rid,
                                code: ErrorCode::Internal,
                                message: format!("engine.step error: {e}"),
                            });
                        }
                        step_failed = true;
                    }
                }
            }

            // Refresh load after step (terminal events drop in-flight count).
            let (running, waiting) = PyEngine::load_locked(&bound);

            // ---- Drain the engine's own metrics ----
            //
            // Inside the GIL scope the tick already owns, so it costs one
            // Python call rather than a second acquisition — but only every
            // DRAIN_INTERVAL, because at a kilohertz tick rate the call itself
            // would be the expensive part and a Prometheus scrape arrives
            // three orders of magnitude less often.  The *replay* happens
            // after the scope, with the GIL released.
            let engine_snapshot = if tick_t0 >= next_drain {
                next_drain = tick_t0 + DRAIN_INTERVAL;
                PyEngine::metrics_snapshot_locked(&bound)
            } else {
                None
            };

            (
                running,
                waiting,
                t_admit,
                t_step,
                t_extract,
                n_out,
                step_failed,
                engine_snapshot,
            )
        });

        let t_tick = tick_t0.elapsed();

        // ---- Route events outside the GIL ----
        let route_t0 = Instant::now();
        for evt in tick_events.drain(..) {
            let terminal = evt.is_terminal();
            let rid_present = evt.request_id().is_some();
            shared.router.route_blocking(evt);
            if terminal && rid_present {
                shared.load.fetch_sub(1, Ordering::Relaxed);
            }
        }
        let t_route = route_t0.elapsed();

        // ---- Metrics ----
        //
        // Recorded on every non-idle tick.  A tick that did nothing (no admits,
        // no step, no outputs) would otherwise flood the tick histogram with
        // near-zero samples and hide the p99 that matters.
        if t_step > Duration::ZERO || n_admit > 0 || n_out > 0 {
            h.tick.record(t_tick.as_secs_f64());
            h.admit.record(t_admit.as_secs_f64());
            h.extract.record(t_extract.as_secs_f64());
            h.route.record(t_route.as_secs_f64());
            if t_step > Duration::ZERO {
                h.step.record(t_step.as_secs_f64());
            }
            if n_out > 0 {
                h.outputs.increment(n_out);
            }
        }
        h.running.set(running as f64);
        h.waiting.set(waiting as f64);

        // Replayed with the GIL released: pushing a few hundred samples into
        // the exporter is pure Rust, and holding the GIL across it would block
        // every request handler for no reason.
        if let Some(snapshot) = engine_snapshot {
            replay(&snapshot, &shared.metric_labels);
        }

        // ---- Dispatch trace accounting (diagnostics; gated by flag) ----
        if trace_enabled && (t_step > Duration::ZERO || n_out > 0) {
            trace.ticks += 1;
            trace.n_admit += n_admit;
            trace.n_out += n_out;
            trace.intake_us += t_intake.as_micros() as u64;
            trace.admit_us += t_admit.as_micros() as u64;
            trace.step_us += t_step.as_micros() as u64;
            trace.extract_us += t_extract.as_micros() as u64;
            trace.route_us += t_route.as_micros() as u64;
            trace.maybe_log(&shared.label);
        }

        // ---- Step-failure tracking ----
        //
        // A single failure is errored + aborted above and the engine usually
        // recovers.  A *run* of failures means the engine is wedged (a corrupt
        // CUDA context, an OOM we cannot unwind), and continuing to accept
        // traffic just converts every arriving request into an INTERNAL error.
        // Stop refreshing the heartbeat so `/readyz` and the gRPC health check
        // go NotServing and the load balancer drains this process.
        if step_failed {
            consecutive_step_failures += 1;
            h.step_failures.increment(1);
            if consecutive_step_failures == MAX_CONSECUTIVE_STEP_FAILURES {
                error!(
                    label = %shared.label,
                    failures = consecutive_step_failures,
                    "engine.step failed {consecutive_step_failures} times in a row; \
                     marking this process not-ready (restart required)"
                );
            }
        } else {
            if consecutive_step_failures >= MAX_CONSECUTIVE_STEP_FAILURES {
                info!(label = %shared.label, "engine.step recovered; marking ready again");
            }
            consecutive_step_failures = 0;
        }
        let healthy = consecutive_step_failures < MAX_CONSECUTIVE_STEP_FAILURES;

        // ---- Refresh load + heartbeat ----
        shared.load.store(running + waiting, Ordering::Relaxed);
        if healthy {
            shared
                .last_event_at_ms
                .store(now_millis(epoch), Ordering::Relaxed);
        }

        // ---- Optional overload signal ----
        if running + waiting >= cfg.max_concurrent_requests {
            let due = match last_overload_emit {
                Some(t) => t.elapsed() >= cfg.overload_emit_interval,
                None => true,
            };
            if due {
                last_overload_emit = Some(Instant::now());
                warn!(
                    label = %shared.label,
                    running,
                    waiting,
                    cap = cfg.max_concurrent_requests,
                    "engine overloaded"
                );
            }
        }

        // Block on the channel when idle or when an in-flight tick did no work.
        // Timeouts refresh the heartbeat and bound backoff; `NO_WORK_TICK_MAX`
        // prevents compute-heavy ticks with no output from qualifying.
        let idle = !received_any && running == 0 && waiting == 0;
        let spinning = did_nothing(received_any, n_out, t_tick);
        if idle || spinning {
            let wait = if idle {
                IDLE_RECV_TIMEOUT
            } else {
                NO_WORK_BACKOFF
            };
            match rt_handle.block_on(tokio::time::timeout(wait, cmd_rx.recv())) {
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

/// Is this an admission command (as opposed to chunk / cancel / ping)?
fn is_admit(cmd: &Cmd) -> bool {
    matches!(cmd, Cmd::CreateOffline { .. } | Cmd::CreateStreaming { .. })
}

/// Should the dispatcher hold this tick's envelopes open for siblings?
///
/// The window is a *throughput* knob for admissions, so it only applies to a
/// batch that is nothing but admissions.  It used to be entered on envelope
/// count alone, which taxed the two most latency-sensitive commands there are:
/// a lone `Cancel` or `FeedChunk` waited the full `--admit-window-ms` before the
/// dispatcher even entered the GIL.
fn should_coalesce(envs: &[CmdEnvelope], cfg: &DispatcherConfig, load: u32) -> bool {
    !envs.is_empty()
        && envs.len() < cfg.admit_threshold
        && cfg.admit_window > Duration::ZERO
        // Past ~25% of the cap admission is no longer the bottleneck; step the
        // queue instead of growing it.
        && load * 4 < cfg.max_concurrent_requests
        && envs.iter().all(|e| is_admit(&e.cmd))
}

/// Did this tick accomplish nothing, so the loop should wait on the channel
/// rather than immediately step again?
///
/// The engine having requests in flight is not the same as the engine having
/// something to do: an open streaming session spends most of its life waiting
/// for the client's next chunk.  `NO_WORK_TICK_MAX` is what keeps this off the
/// working paths — a tick that is genuinely computing costs far more than a
/// millisecond even when it emits nothing that tick (an AR decode group
/// grinding through its per-tick step budget), so it never qualifies.
fn did_nothing(received_any: bool, n_out: u64, t_tick: Duration) -> bool {
    !received_any && n_out == 0 && t_tick < NO_WORK_TICK_MAX
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
            decoding,
        } => {
            let load = shared.load.load(Ordering::Relaxed);
            if load >= max_concurrent {
                warn!(
                    label = %shared.label,
                    rid = %request_id,
                    load,
                    cap = max_concurrent,
                    "admission rejected: at capacity"
                );
                metrics::counter!(om::REQUESTS_BUSY, shared.metric_labels.iter()).increment(1);
                out_events.push(Event::Error {
                    request_id: request_id.clone(),
                    code: ErrorCode::Busy,
                    message: format!("in-flight {load} >= cap {max_concurrent}"),
                });
                return;
            }
            let audio = payload.unwrap_or_default();
            if audio.is_empty() {
                debug!(label = %shared.label, rid = %request_id, "rejected: empty audio payload");
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
                decoding,
            });
            out_events.push(Event::Accepted { request_id });
        }
        Cmd::CreateStreaming {
            request_id,
            sample_rate,
            priority,
            decoding,
        } => {
            let load = shared.load.load(Ordering::Relaxed);
            if load >= max_concurrent {
                warn!(
                    label = %shared.label,
                    rid = %request_id,
                    load,
                    cap = max_concurrent,
                    "admission rejected: at capacity"
                );
                metrics::counter!(om::REQUESTS_BUSY, shared.metric_labels.iter()).increment(1);
                out_events.push(Event::Error {
                    request_id: request_id.clone(),
                    code: ErrorCode::Busy,
                    message: format!("in-flight {load} >= cap {max_concurrent}"),
                });
                return;
            }
            shared.load.fetch_add(1, Ordering::Relaxed);
            out_admits.push(AdmitSpec::Streaming {
                rid: request_id.clone(),
                sample_rate,
                priority,
                decoding,
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

/// Replay the accumulated `admit_batch` via a single Python call.
///
/// The Python side validates and admits **per spec**, so the common failure
/// (one client's out-of-range option) comes back as a per-spec message and only
/// that request's `Accepted` is rewritten into an `Error`.  An `Err` from the
/// call itself is batch-wide and rewrites every spec in the batch.
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
        Ok(per_spec) => {
            rewrite_rejected_admits(admit_batch, shared, out_events, &per_spec);
        }
        Err(e) => {
            let msg = format!("{e}");
            let all: Vec<Option<String>> = vec![Some(msg); admit_batch.len()];
            rewrite_rejected_admits(admit_batch, shared, out_events, &all);
        }
    }
}

/// Turn per-spec rejection messages into `Error` events for the matching rids.
///
/// Replace rejected specs' accepted events by request id and drain the batch.
/// Load is decremented later by the common terminal-event routing path.
fn rewrite_rejected_admits(
    admit_batch: &mut Vec<AdmitSpec>,
    shared: &DispatcherShared,
    out_events: &mut Vec<Event>,
    errors: &[Option<String>],
) {
    let n = admit_batch.len();
    let mut n_rejected = 0usize;
    for (i, spec) in admit_batch.drain(..).enumerate() {
        let Some(msg) = errors.get(i).and_then(|e| e.as_deref()) else {
            continue; // admitted; its Accepted event stands
        };
        n_rejected += 1;
        let rid = spec.request_id();
        // InvalidCmd maps to INVALID_ARGUMENT / 400 on both front-ends — a
        // rejected option is a client error, not an engine fault.
        let ev = Event::Error {
            request_id: rid.to_owned(),
            code: ErrorCode::InvalidCmd,
            message: msg.to_owned(),
        };
        match out_events
            .iter()
            .rposition(|e| matches!(e, Event::Accepted { request_id } if request_id == rid))
        {
            Some(pos) => out_events[pos] = ev,
            // No Accepted to replace (shouldn't happen): append so the client
            // still gets a terminal event instead of hanging.
            None => out_events.push(ev),
        }
    }
    if n_rejected > 0 {
        metrics::counter!(om::REQUESTS_REJECTED, shared.metric_labels.iter())
            .increment(n_rejected as u64);
        warn!(
            label = %shared.label,
            n_rejected,
            n_batch = n,
            "engine rejected admits (per-request; batch-mates unaffected)"
        );
    }
    let admitted = n.saturating_sub(n_rejected);
    if admitted > 0 {
        metrics::counter!(om::REQUESTS_ADMITTED, shared.metric_labels.iter())
            .increment(admitted as u64);
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
                warn!(label = %shared.label, rid = %request_id, "feed_chunk failed: {e}");
                out_events.push(engine_error_event(&request_id, &e));
            }
        }
        Cmd::Cancel { request_id } => {
            // Logged because the usual source is a client disconnect, which is
            // otherwise invisible: the request simply stops producing events and
            // an operator has no way to tell that from an engine that dropped it.
            debug!(label = %shared.label, rid = %request_id, "cancelling request");
            if let Err(e) = PyEngine::abort_locked(bound, &request_id) {
                warn!(label = %shared.label, rid = %request_id, "abort failed: {e}");
            }
            shared.router.remove(&request_id);
            metrics::counter!(om::REQUESTS_CANCELLED, shared.metric_labels.iter()).increment(1);
        }
        // Create* should never reach here — the dispatcher routes them to
        // enqueue_admit_locked.  If one slips through, fall back to the
        // single-shot admit so behaviour is preserved.
        Cmd::CreateOffline { .. } | Cmd::CreateStreaming { .. } => {
            error!(label = %shared.label, "internal: Create* cmd reached handle_nonadmit_cmd_locked");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn admit(rid: &str) -> CmdEnvelope {
        CmdEnvelope::new(
            Cmd::CreateOffline {
                request_id: rid.into(),
                sample_rate: 16_000,
                priority: 0,
                decoding: None,
            },
            Some(Bytes::from_static(b"\0\0\0\0")),
        )
    }

    fn chunk(rid: &str) -> CmdEnvelope {
        CmdEnvelope::new(
            Cmd::FeedChunk {
                request_id: rid.into(),
                is_last: false,
            },
            Some(Bytes::from_static(b"\0\0\0\0")),
        )
    }

    fn cancel(rid: &str) -> CmdEnvelope {
        CmdEnvelope::new(
            Cmd::Cancel {
                request_id: rid.into(),
            },
            None,
        )
    }

    fn cfg() -> DispatcherConfig {
        DispatcherConfig::default()
    }

    #[test]
    fn a_thin_batch_of_admissions_coalesces() {
        assert!(should_coalesce(&[admit("a"), admit("b")], &cfg(), 0));
    }

    /// The regression this predicate exists for: a cancel or a chunk must reach
    /// the engine on the next tick, not after `--admit-window-ms`.  Cancels are
    /// what free a decode slot, and chunks are the streaming critical path.
    #[test]
    fn a_latency_sensitive_command_skips_the_window() {
        assert!(!should_coalesce(&[cancel("a")], &cfg(), 0));
        assert!(!should_coalesce(&[chunk("a")], &cfg(), 0));
        // Even mixed in with admissions — one delayed cancel is worse than one
        // under-full admission batch.
        assert!(!should_coalesce(&[admit("a"), chunk("b")], &cfg(), 0));
        assert!(!should_coalesce(&[chunk("b"), admit("a")], &cfg(), 0));
    }

    #[test]
    fn nothing_to_coalesce_or_already_full_skips_the_window() {
        assert!(!should_coalesce(&[], &cfg(), 0));
        let full: Vec<CmdEnvelope> = (0..cfg().admit_threshold).map(|_| admit("a")).collect();
        assert!(!should_coalesce(&full, &cfg(), 0));
    }

    #[test]
    fn a_loaded_engine_skips_the_window() {
        let c = cfg();
        assert!(!should_coalesce(
            &[admit("a")],
            &c,
            c.max_concurrent_requests / 4
        ));
    }

    #[test]
    fn a_zero_window_disables_coalescing() {
        let c = DispatcherConfig {
            admit_window: Duration::ZERO,
            ..cfg()
        };
        assert!(!should_coalesce(&[admit("a")], &c, 0));
    }

    /// The spin this guards against: an open streaming session leaves
    /// `running > 0` for its whole life, so the idle block never engaged and the
    /// loop stepped an engine with nothing to do until the client's next chunk.
    #[test]
    fn an_empty_fast_tick_backs_off() {
        assert!(did_nothing(false, 0, Duration::from_micros(50)));
    }

    /// ...and the three ways a tick earns the right to loop straight back.
    #[test]
    fn a_tick_that_did_something_does_not_back_off() {
        // Commands arrived — more may be queued behind them.
        assert!(!did_nothing(true, 0, Duration::from_micros(50)));
        // Outputs were produced — the engine is mid-stride.
        assert!(!did_nothing(false, 3, Duration::from_micros(50)));
        // Slow tick with no output: an AR decode group grinding through its
        // per-tick step budget.  Backing off here would directly slow
        // generation, which is why the gate is tick *duration* and not
        // "running > 0".
        assert!(!did_nothing(false, 0, Duration::from_millis(48)));
    }
}
