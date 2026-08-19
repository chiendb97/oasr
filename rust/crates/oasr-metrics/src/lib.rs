// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! The one place every OASR metric is declared.
//!
//! Name, kind, unit, help text and **histogram buckets** live together in
//! [`METRICS`], and [`install_recorder`] is the only thing that builds the
//! Prometheus exporter.  Keeping the four together is not tidiness — it is what
//! makes the bucket configuration impossible to forget:
//!
//! `metrics-exporter-prometheus` defaults `buckets: None`, and a histogram with
//! no buckets is not rendered as a histogram at all.  It becomes a **rolling
//! summary** with hard-coded quantiles: no `_bucket` series, so
//! `histogram_quantile` cannot be used, no heatmaps, and — the part that
//! actually hurts — quantiles that **cannot be aggregated across replicas**.
//! `avg(oasr_engine_step_seconds{quantile="0.99"})` over four pods is not the
//! fleet p99; it is not any number at all.  Every histogram declared here
//! therefore carries an explicit bucket set, and `metrics_table_is_coherent`
//! fails the build if a new one arrives without.
//!
//! # Scopes
//!
//! Metrics are grouped by *who owns the clock*, and that decides the labels:
//!
//! | Scope | Owner | Labels |
//! |---|---|---|
//! | Transport | the HTTP / gRPC listener | `method`, `route`, `status` |
//! | Request | the front-end handler, arrival → terminal event | `api`, `mode`, `outcome` |
//! | Dispatcher | the GIL-owning dispatcher thread | `engine`, `model`, `decode_method` |
//! | Engine | Python, drained through the dispatcher | `engine`, plus a per-metric key |
//!
//! Only the last two carry `engine`: they are the ones an [`EnginePool`] of
//! several engines would otherwise collide on.  A transport metric has no
//! engine to name — one listener fronts the whole pool — and labelling it with
//! a model would be a guess as soon as the pool is heterogeneous.
//!
//! [`EnginePool`]: https://docs.rs/oasr-engine-client

use std::sync::atomic::{AtomicU64, Ordering};

use metrics::Counter;
use metrics_exporter_prometheus::{BuildError, Matcher, PrometheusBuilder, PrometheusHandle};

pub use metrics::Unit;

/// A fractional-seconds total exported through Prometheus' integer counter.
///
/// `metrics::Counter` is `u64`, and audio durations are not: a 3.7 s clip
/// incremented as `3` loses 0.7 s **per request**, and that error compounds
/// until the RTFx computed from it is simply wrong.  So the exact total is
/// accumulated here in microseconds and republished with `absolute()`, whose
/// truncation is against the running total rather than each addition — bounded
/// below one second forever instead of growing without limit.
pub struct SecondsCounter {
    micros: AtomicU64,
    handle: Counter,
}

impl SecondsCounter {
    pub fn new(handle: Counter) -> Self {
        Self {
            micros: AtomicU64::new(0),
            handle,
        }
    }

    /// Add `seconds` to the total and republish it.  Negative and non-finite
    /// values are ignored — a counter may not go backwards, and a NaN duration
    /// would poison the series permanently.
    pub fn add(&self, seconds: f64) {
        if !(seconds.is_finite() && seconds > 0.0) {
            return;
        }
        let add = (seconds * 1e6) as u64;
        let total = self.micros.fetch_add(add, Ordering::Relaxed) + add;
        self.handle.absolute(total / 1_000_000);
    }

    /// The exact accumulated total, for tests.
    pub fn total_seconds(&self) -> f64 {
        self.micros.load(Ordering::Relaxed) as f64 / 1e6
    }
}

/// What a metric is, which decides how it is described and whether it needs
/// buckets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Counter,
    Gauge,
    Histogram,
}

/// One metric's complete declaration.
#[derive(Debug, Clone, Copy)]
pub struct MetricDef {
    pub name: &'static str,
    pub kind: Kind,
    pub unit: Option<Unit>,
    /// Bucket boundaries — required for [`Kind::Histogram`], forbidden otherwise.
    pub buckets: Option<&'static [f64]>,
    pub help: &'static str,
}

// ---------------------------------------------------------------------------
// Bucket sets
// ---------------------------------------------------------------------------

/// Sub-second machinery: dispatcher ticks, engine stages, `step()`.
///
/// Has to span four orders of magnitude because the same histogram sees a
/// ~50 µs idle-ish CTC chunk tick and a ~600 ms batched 7B decoder tick — that
/// spread is the reason the tick histogram exists at all, so buckets that
/// resolve only one end of it would hide exactly what it was added to show.
pub static BUCKETS_TICK: &[f64] = &[
    50e-6, 100e-6, 250e-6, 500e-6, 1e-3, 2.5e-3, 5e-3, 10e-3, 25e-3, 50e-3, 100e-3, 250e-3, 500e-3,
    1.0, 2.5, 5.0, 10.0,
];

/// End-to-end request latency: milliseconds for a short streaming utterance,
/// minutes for a long-form offline file.  The top bucket sits above the
/// `--request-timeout-secs` default (300) so a timed-out request lands in a
/// real bucket rather than only in `+Inf`.
pub static BUCKETS_REQUEST: &[f64] = &[
    5e-3, 10e-3, 25e-3, 50e-3, 100e-3, 250e-3, 500e-3, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0,
    300.0, 600.0,
];

/// Time to first partial — the streaming SLI.  Dense through 100–750 ms
/// because that is the band every real target sits in; anything past a few
/// seconds is a failure, not a percentile worth resolving.
pub static BUCKETS_TTFP: &[f64] = &[
    10e-3, 25e-3, 50e-3, 100e-3, 150e-3, 200e-3, 300e-3, 500e-3, 750e-3, 1.0, 1.5, 2.0, 3.0, 5.0,
    10.0, 30.0,
];

/// Queue wait — admission to first scheduled.  Healthy is under a tick; the
/// long tail is what starvation looks like.
pub static BUCKETS_QUEUE: &[f64] = &[
    100e-6, 500e-6, 1e-3, 5e-3, 10e-3, 25e-3, 50e-3, 100e-3, 250e-3, 500e-3, 1.0, 2.5, 5.0, 10.0,
    30.0,
];

/// A fraction in `[0, 1]`.
pub static BUCKETS_RATIO: &[f64] = &[
    0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
];

/// Rows in a batch.  Roughly geometric — the interesting question is "am I
/// batching at all", not whether it was 47 or 48.
pub static BUCKETS_BATCH: &[f64] = &[
    1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0,
    512.0,
];

/// Audio duration of one request, in seconds.
pub static BUCKETS_AUDIO: &[f64] = &[
    0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0, 14400.0,
];

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

/// Declare a metric constant and its [`MetricDef`] row in one statement.
///
/// A macro rather than two lists because two lists drift: a constant with no
/// row exports a metric with no buckets and no help, and a row with no constant
/// is a name nothing can record without retyping it.
macro_rules! metrics_table {
    ($(
        $(#[$attr:meta])*
        $ident:ident = $name:literal, $kind:ident, $unit:expr, $buckets:expr, $help:literal;
    )*) => {
        $( $(#[$attr])* pub const $ident: &str = $name; )*

        /// Every metric OASR exports, in declaration order.
        pub static METRICS: &[MetricDef] = &[
            $( MetricDef {
                name: $name,
                kind: Kind::$kind,
                unit: $unit,
                buckets: $buckets,
                help: $help,
            } ),*
        ];
    };
}

metrics_table! {
    // -- Transport: HTTP ----------------------------------------------------
    /// `{method, route, status}` — `route` is the **matched** axum path
    /// pattern, never the request URI.
    HTTP_REQUESTS = "oasr_http_requests_total", Counter, None, None,
        "HTTP requests served, by method, matched route and status class";
    /// `{method, route}`
    HTTP_DURATION = "oasr_http_request_duration_seconds", Histogram, Some(Unit::Seconds), Some(BUCKETS_REQUEST),
        "HTTP request wall time, from the listener accepting to the response head";
    HTTP_IN_FLIGHT = "oasr_http_requests_in_flight", Gauge, None, None,
        "HTTP requests currently being served";

    // -- Transport: gRPC ----------------------------------------------------
    /// `{method, code}` — `code` is the gRPC status name, resolved when the
    /// RPC actually ends (for a server-streaming RPC that is at the trailer,
    /// not at the response head).
    GRPC_REQUESTS = "oasr_grpc_requests_total", Counter, None, None,
        "gRPC RPCs served, by method and terminal status code";
    /// `{method}`
    GRPC_DURATION = "oasr_grpc_request_duration_seconds", Histogram, Some(Unit::Seconds), Some(BUCKETS_REQUEST),
        "gRPC RPC wall time, to the terminal status";

    // -- Request scope ------------------------------------------------------
    /// `{api, mode, outcome}` — the number an SLO is written against.
    REQUEST_DURATION = "oasr_request_duration_seconds", Histogram, Some(Unit::Seconds), Some(BUCKETS_REQUEST),
        "ASR request wall time, from the handler receiving it to the terminal event";
    /// `{api, mode}` — the denominator of RTFx.  Counted as audio is *ingested*
    /// (per chunk for a stream), so it includes audio a request later failed on.
    AUDIO_SECONDS = "oasr_audio_seconds_total", Counter, Some(Unit::Seconds), None,
        "Decoded audio duration accepted by the front-end";
    /// `{api, mode}`
    REQUEST_AUDIO_SECONDS = "oasr_request_audio_seconds", Histogram, Some(Unit::Seconds), Some(BUCKETS_AUDIO),
        "Decoded audio duration of one request";
    /// `{api}` — first inbound audio byte to first partial transcript.  The
    /// streaming SLI; undefined (and not recorded) for a stream that produces
    /// no partial before its final.
    TIME_TO_FIRST_PARTIAL = "oasr_time_to_first_partial_seconds", Histogram, Some(Unit::Seconds), Some(BUCKETS_TTFP),
        "First inbound audio byte to first partial transcript, per streaming request";

    // -- Dispatcher scope ---------------------------------------------------
    // All carry {engine, model, decode_method}.
    DISPATCH_TICK = "oasr_dispatch_tick_seconds", Histogram, Some(Unit::Seconds), Some(BUCKETS_TICK),
        "Wall time of one dispatcher tick (GIL held: admit + step + extract)";
    DISPATCH_ADMIT = "oasr_dispatch_admit_seconds", Histogram, Some(Unit::Seconds), Some(BUCKETS_TICK),
        "Time replaying admission commands into Python per tick";
    ENGINE_STEP = "oasr_engine_step_seconds", Histogram, Some(Unit::Seconds), Some(BUCKETS_TICK),
        "Time in ASREngine.step() per tick";
    DISPATCH_EXTRACT = "oasr_dispatch_extract_seconds", Histogram, Some(Unit::Seconds), Some(BUCKETS_TICK),
        "Time converting RequestOutputs into events per tick";
    DISPATCH_ROUTE = "oasr_dispatch_route_seconds", Histogram, Some(Unit::Seconds), Some(BUCKETS_TICK),
        "Time routing events to per-request channels per tick (GIL released)";
    ENGINE_RUNNING = "oasr_engine_running", Gauge, None, None,
        "Requests the engine reports as running (incl. parked AR generations)";
    ENGINE_WAITING = "oasr_engine_waiting", Gauge, None, None,
        "Requests waiting for admission";
    ENGINE_OUTPUTS = "oasr_engine_outputs_total", Counter, None, None,
        "RequestOutputs returned by step()";
    REQUESTS_ADMITTED = "oasr_requests_admitted_total", Counter, None, None,
        "Requests accepted by the engine";
    REQUESTS_REJECTED = "oasr_requests_rejected_total", Counter, None, None,
        "Requests the engine rejected at admission (invalid options, mode mismatch)";
    REQUESTS_BUSY = "oasr_requests_busy_total", Counter, None, None,
        "Requests refused because max_concurrent_requests was reached";
    ENGINE_STEP_FAILURES = "oasr_engine_step_failures_total", Counter, None, None,
        "ASREngine.step() raised";
    REQUESTS_CANCELLED = "oasr_requests_cancelled_total", Counter, None, None,
        "Requests aborted before completion (usually a client disconnect)";
    /// Additionally labelled `{stage}`.
    REQUESTS_FAILED = "oasr_requests_failed_total", Counter, None, None,
        "Requests the engine finished with an error, labelled by the stage that failed";
    EVENTS_DROPPED = "oasr_events_dropped_total", Counter, None, None,
        "Non-terminal events (partials) dropped because a client's channel was full";
    EVENTS_DEFERRED = "oasr_events_deferred_total", Counter, None, None,
        "Terminal events handed to a background task because the channel was full";

    // -- Engine scope (recorded in Python, drained by the dispatcher) --------
    /// `{engine, stage}` — **host** wall time, deliberately not GPU time.
    /// See the crate docs on `oasr.engine.metrics` for why the distinction
    /// is in the metric name.
    ENGINE_STAGE_HOST = "oasr_engine_stage_host_seconds", Histogram, Some(Unit::Seconds), Some(BUCKETS_TICK),
        "Host wall time inside one engine stage (issue time, not GPU time)";
    /// `{engine, mode}`
    ENGINE_BATCH_SIZE = "oasr_engine_batch_size", Histogram, None, Some(BUCKETS_BATCH),
        "Rows in one executed batch";
    /// `{engine}` — fraction of the padded batch that is padding.  Says
    /// whether the bucketing policy is working, which is otherwise guesswork.
    ENGINE_BATCH_PADDING = "oasr_engine_batch_padding_ratio", Histogram, None, Some(BUCKETS_RATIO),
        "Fraction of a padded offline batch that is padding";
    /// `{engine, mode}` — admission to first scheduled.
    ENGINE_QUEUE_WAIT = "oasr_engine_queue_wait_seconds", Histogram, Some(Unit::Seconds), Some(BUCKETS_QUEUE),
        "Time a request spent queued before the scheduler first picked it up";
    ENGINE_KV_BLOCKS_USED = "oasr_engine_kv_blocks_used", Gauge, None, None,
        "Paged streaming-encoder KV blocks currently allocated";
    /// `_capacity`, not `_total`: Prometheus reserves the `_total` suffix for
    /// counters, and a ceiling that can move when the pool is resized is a
    /// gauge.  `metric_names_follow_prometheus_conventions` enforces the split.
    ENGINE_KV_BLOCKS_TOTAL = "oasr_engine_kv_blocks_capacity", Gauge, None, None,
        "Paged streaming-encoder KV blocks in the pool";
    /// Not "evictions", which is what an LLM KV cache would report: OASR's
    /// paged pool does not evict, and its capacity gate means it is never even
    /// asked for a block it cannot give.  What is observable — and what an
    /// operator actually needs — is the consequence: a stream cut short, its
    /// transcript ending early with `finish_reason="length"`.
    ENGINE_KV_EXHAUSTED = "oasr_engine_kv_exhausted_total", Counter, None, None,
        "Streams finalized early because their paged encoder cache reached capacity";
    ENGINE_DECODE_SLOTS_USED = "oasr_engine_decode_slots_in_use", Gauge, None, None,
        "Autoregressive decode slots currently occupied by a parked generation";
    ENGINE_DECODE_SLOTS_TOTAL = "oasr_engine_decode_slots_capacity", Gauge, None, None,
        "Autoregressive decode slot ceiling (0 when the family is not incremental)";
    ENGINE_TOKENS_GENERATED = "oasr_engine_tokens_generated_total", Counter, None, None,
        "Tokens emitted by the decode strategies";
    ENGINE_AUDIO_SECONDS = "oasr_engine_audio_seconds_total", Counter, Some(Unit::Seconds), None,
        "Audio duration the engine admitted (the in-process RTFx numerator)";
    /// Non-zero means a histogram above is showing a *truncated* distribution:
    /// the engine buffered more samples between two drains than it may hold.
    /// Exported so the cap is visible rather than passing for a complete one.
    ENGINE_SAMPLES_DROPPED = "oasr_engine_metric_samples_dropped_total", Counter, None, None,
        "Histogram samples the engine-side collector discarded because its buffer was full";

    // -- Device -------------------------------------------------------------
    GPU_MEMORY_USED = "oasr_gpu_memory_used_bytes", Gauge, Some(Unit::Bytes), None,
        "Device memory in use on the engine's GPU, process-wide (NVML view)";
    GPU_MEMORY_TOTAL = "oasr_gpu_memory_total_bytes", Gauge, Some(Unit::Bytes), None,
        "Total device memory on the engine's GPU";
    /// NVML's sampled "fraction of the last window in which any kernel was
    /// resident" — a saturation hint, **not** SM occupancy.  A kernel using one
    /// SM of 170 reads as 1.0.
    GPU_UTILIZATION = "oasr_gpu_utilization_ratio", Gauge, None, None,
        "NVML GPU utilization over its sampling window, in [0, 1]";
}

// ---------------------------------------------------------------------------
// Label vocabularies
// ---------------------------------------------------------------------------

/// Which API shape served a request — the `api` label.
///
/// Declared here rather than spelled at each call site so the HTTP and gRPC
/// front-ends, which duplicate their orchestration, cannot disagree about what
/// a value is called.
pub mod api {
    pub const GOOGLE: &str = "google";
    pub const OPENAI: &str = "openai";
    pub const REALTIME: &str = "realtime";
    pub const GRPC: &str = "grpc";
    pub const GRPC_STREAMING: &str = "grpc_streaming";
}

/// The `mode` label.
pub mod mode {
    pub const OFFLINE: &str = "offline";
    pub const STREAMING: &str = "streaming";
}

/// The `outcome` label on [`REQUEST_DURATION`].
pub mod outcome {
    pub const OK: &str = "ok";
    /// The engine returned a terminal error event.
    pub const ERROR: &str = "error";
    /// The client went away, or the deadline fired, before a terminal event.
    pub const CANCELLED: &str = "cancelled";
}

/// Label keys, so a typo is a compile error at one site rather than a silently
/// separate time series.
pub mod label {
    pub const ENGINE: &str = "engine";
    pub const MODEL: &str = "model";
    pub const DECODE_METHOD: &str = "decode_method";
    pub const API: &str = "api";
    pub const MODE: &str = "mode";
    pub const OUTCOME: &str = "outcome";
    pub const STAGE: &str = "stage";
    pub const METHOD: &str = "method";
    pub const ROUTE: &str = "route";
    pub const STATUS: &str = "status";
    pub const CODE: &str = "code";
}

/// The label key an engine-scope series carries *besides* `engine`, if any.
///
/// The engine drains its histograms as `{metric: {key: [samples]}}` without
/// saying what the key means; this is where that is declared, once. A metric
/// with no entry here is **skipped** rather than recorded unlabelled — merging
/// `stage="encode"` into `stage="decode"` would produce a plausible number that
/// is not any stage's latency, and a dropped series with a warning is the
/// failure an operator can see.
pub fn keyed_label_for(metric: &str) -> Option<&'static str> {
    match metric {
        ENGINE_STAGE_HOST => Some(label::STAGE),
        ENGINE_BATCH_SIZE | ENGINE_QUEUE_WAIT => Some(label::MODE),
        _ => None,
    }
}

/// Value used for a label whose real value is not known.
///
/// A literal, because dropping the label instead would put the sample in a
/// *different* time series than its labelled siblings — the series would
/// silently split rather than showing a gap.
pub const UNKNOWN: &str = "unknown";

// ---------------------------------------------------------------------------
// Request-scope recorder
// ---------------------------------------------------------------------------

/// How a request ended, for the `outcome` label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    /// The engine returned a terminal error event.
    Error,
    /// The client disconnected, or a deadline fired, before a terminal event.
    Cancelled,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Ok => outcome::OK,
            Outcome::Error => outcome::ERROR,
            Outcome::Cancelled => outcome::CANCELLED,
        }
    }

    fn index(self) -> usize {
        match self {
            Outcome::Ok => 0,
            Outcome::Error => 1,
            Outcome::Cancelled => 2,
        }
    }
}

/// Pre-resolved request metrics shared by HTTP and gRPC so both surfaces use the
/// same outcome vocabulary. Construct one immutable instance per surface.
pub struct RequestRecorder {
    /// Indexed by [`Outcome::index`].
    duration: [metrics::Histogram; 3],
    audio_total: SecondsCounter,
    audio_per_request: metrics::Histogram,
    ttfp: metrics::Histogram,
}

impl RequestRecorder {
    pub fn new(api: &'static str, mode: &'static str) -> Self {
        let duration = |o: Outcome| {
            metrics::histogram!(
                REQUEST_DURATION,
                label::API => api,
                label::MODE => mode,
                label::OUTCOME => o.as_str(),
            )
        };
        Self {
            duration: [
                duration(Outcome::Ok),
                duration(Outcome::Error),
                duration(Outcome::Cancelled),
            ],
            audio_total: SecondsCounter::new(
                metrics::counter!(AUDIO_SECONDS, label::API => api, label::MODE => mode),
            ),
            audio_per_request: metrics::histogram!(
                REQUEST_AUDIO_SECONDS,
                label::API => api,
                label::MODE => mode,
            ),
            ttfp: metrics::histogram!(TIME_TO_FIRST_PARTIAL, label::API => api),
        }
    }

    /// Count audio the front-end accepted.  Call once per offline request and
    /// once per streaming chunk.
    pub fn audio_ingested(&self, seconds: f64) {
        self.audio_total.add(seconds);
    }

    /// Record one request's audio duration.  Separate from
    /// [`Self::audio_ingested`] because a stream's total is only known at the
    /// end, while the RTFx denominator has to accrue as audio arrives.
    pub fn audio_duration(&self, seconds: f64) {
        if seconds.is_finite() && seconds >= 0.0 {
            self.audio_per_request.record(seconds);
        }
    }

    /// Record a finished request's wall time.
    pub fn finished(&self, outcome: Outcome, elapsed: std::time::Duration) {
        self.duration[outcome.index()].record(elapsed.as_secs_f64());
    }

    /// Record time to first partial for a streaming request.
    pub fn first_partial(&self, elapsed: std::time::Duration) {
        self.ttfp.record(elapsed.as_secs_f64());
    }
}

/// Seconds of f32 little-endian mono PCM.
///
/// The front-ends hold decoded audio as raw `Bytes`, so this is the one
/// conversion between what they have and what the metric wants.  Written once
/// because getting the divisor wrong is silent: a factor-of-four error in
/// `oasr_audio_seconds_total` shows up as a plausible RTFx, not as a failure.
pub fn f32_pcm_seconds(byte_len: usize, sample_rate: u32) -> f64 {
    if sample_rate == 0 {
        return 0.0;
    }
    (byte_len / 4) as f64 / sample_rate as f64
}

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

/// Build and install the process-global Prometheus recorder.
///
/// Applies every bucket set in [`METRICS`] and registers every description, so
/// `GET /metrics` is self-documenting and correctly typed before the first
/// request lands.
pub fn install_recorder() -> Result<PrometheusHandle, BuildError> {
    let handle = builder()?.install_recorder()?;
    describe_all();
    Ok(handle)
}

/// A `PrometheusBuilder` carrying every declared bucket set.
///
/// Shared by [`install_recorder`] and by the tests, so a test can never assert
/// against a differently-configured exporter than the one that ships — which
/// for this crate would mean asserting that buckets work against a builder
/// that happened to have them and shipping one that did not.
pub fn builder() -> Result<PrometheusBuilder, BuildError> {
    let mut builder = PrometheusBuilder::new();
    for def in METRICS {
        if let Some(buckets) = def.buckets {
            builder =
                builder.set_buckets_for_metric(Matcher::Full(def.name.to_owned()), buckets)?;
        }
    }
    Ok(builder)
}

/// Register the help text and unit for every declared metric.
///
/// Separate from [`install_recorder`] so a test can install its own recorder
/// and still get described output.
pub fn describe_all() {
    for def in METRICS {
        match (def.kind, def.unit) {
            (Kind::Counter, Some(u)) => metrics::describe_counter!(def.name, u, def.help),
            (Kind::Counter, None) => metrics::describe_counter!(def.name, def.help),
            (Kind::Gauge, Some(u)) => metrics::describe_gauge!(def.name, u, def.help),
            (Kind::Gauge, None) => metrics::describe_gauge!(def.name, def.help),
            (Kind::Histogram, Some(u)) => metrics::describe_histogram!(def.name, u, def.help),
            (Kind::Histogram, None) => metrics::describe_histogram!(def.name, def.help),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The guard that makes this crate worth existing.
    ///
    /// A histogram declared without buckets silently degrades to a
    /// non-aggregatable rolling summary — the exact defect this crate was
    /// introduced to fix — and nothing about the rendered output announces it.
    #[test]
    fn metrics_table_is_coherent() {
        let mut seen = HashSet::new();
        for def in METRICS {
            assert!(
                seen.insert(def.name),
                "duplicate metric name {:?}",
                def.name
            );
            assert!(
                def.name.starts_with("oasr_"),
                "{:?} is missing the oasr_ prefix",
                def.name
            );
            assert!(!def.help.is_empty(), "{:?} has no help text", def.name);

            match def.kind {
                Kind::Histogram => {
                    let buckets = def.buckets.unwrap_or_else(|| {
                        panic!(
                            "histogram {:?} has no buckets: it would render as a rolling \
                             summary, whose quantiles cannot be aggregated across replicas",
                            def.name
                        )
                    });
                    assert!(
                        !buckets.is_empty(),
                        "{:?} has an empty bucket set",
                        def.name
                    );
                    assert!(
                        buckets.windows(2).all(|w| w[0] < w[1]),
                        "{:?} buckets are not strictly ascending",
                        def.name
                    );
                }
                Kind::Counter | Kind::Gauge => assert!(
                    def.buckets.is_none(),
                    "{:?} is not a histogram but declares buckets",
                    def.name
                ),
            }
        }
    }

    /// Prometheus naming conventions, checked rather than trusted: `_total`
    /// belongs to counters alone, and a `_seconds` name must actually be
    /// declared in seconds or the exporter's unit metadata contradicts it.
    #[test]
    fn metric_names_follow_prometheus_conventions() {
        for def in METRICS {
            assert_eq!(
                def.name.ends_with("_total"),
                def.kind == Kind::Counter,
                "{:?}: the _total suffix is a counter convention",
                def.name
            );
            if def.name.ends_with("_seconds") {
                assert_eq!(
                    def.unit,
                    Some(Unit::Seconds),
                    "{:?} is named in seconds but not declared in them",
                    def.name
                );
            }
            if def.name.ends_with("_bytes") {
                assert_eq!(
                    def.unit,
                    Some(Unit::Bytes),
                    "{:?} is named in bytes but not declared in them",
                    def.name
                );
            }
        }
    }

    /// Every keyed metric must be a declared one, and must be a histogram —
    /// the drain protocol only carries samples for keyed series.
    #[test]
    fn keyed_labels_point_at_declared_histograms() {
        for def in METRICS {
            if let Some(key) = keyed_label_for(def.name) {
                assert_eq!(
                    def.kind,
                    Kind::Histogram,
                    "{:?} declares the {key:?} label but is not a histogram",
                    def.name
                );
            }
        }
        assert_eq!(keyed_label_for("oasr_not_a_metric"), None);
    }

    /// The property the type exists for: truncation must not compound.
    ///
    /// A thousand 3.7 s clips through a plain `Counter::increment(3)` would
    /// report 3000 s of audio instead of 3700 — a 19% error in every RTFx
    /// derived from it, growing with traffic rather than bounded.
    #[test]
    fn seconds_counter_truncation_does_not_accumulate() {
        let c = SecondsCounter::new(Counter::noop());
        for _ in 0..1000 {
            c.add(3.7);
        }
        assert!(
            (c.total_seconds() - 3700.0).abs() < 1.0,
            "{}",
            c.total_seconds()
        );
        // The published integer view is the total truncated once, not a sum of
        // truncations: under a second of error against 3700.
        assert_eq!((c.total_seconds() as u64), 3700);
    }

    #[test]
    fn seconds_counter_ignores_values_a_counter_cannot_take() {
        let c = SecondsCounter::new(Counter::noop());
        c.add(f64::NAN);
        c.add(f64::INFINITY);
        c.add(-5.0);
        c.add(0.0);
        assert_eq!(c.total_seconds(), 0.0);
    }

    #[test]
    fn f32_pcm_seconds_converts_bytes_not_samples() {
        // One second of 16 kHz f32 mono is 64000 bytes, not 16000.
        assert_eq!(f32_pcm_seconds(64_000, 16_000), 1.0);
        assert_eq!(f32_pcm_seconds(0, 16_000), 0.0);
        assert_eq!(
            f32_pcm_seconds(64_000, 0),
            0.0,
            "a zero rate must not divide"
        );
    }

    #[test]
    fn outcome_labels_match_the_declared_vocabulary() {
        assert_eq!(Outcome::Ok.as_str(), outcome::OK);
        assert_eq!(Outcome::Error.as_str(), outcome::ERROR);
        assert_eq!(Outcome::Cancelled.as_str(), outcome::CANCELLED);
        // Distinct indices, or two outcomes would share a handle.
        let idx = [
            Outcome::Ok.index(),
            Outcome::Error.index(),
            Outcome::Cancelled.index(),
        ];
        assert_eq!(idx.iter().collect::<HashSet<_>>().len(), 3);
        assert!(idx.iter().all(|i| *i < 3));
    }

    /// Record into **every** declared metric and check what the exporter
    /// actually renders for it.
    ///
    /// Rendering catches exact-name mismatches between metric declarations and
    /// bucket matchers that table-level checks cannot observe.
    #[test]
    fn every_declared_metric_renders_in_its_declared_form() {
        let recorder = builder().unwrap().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            describe_all();
            for def in METRICS {
                match def.kind {
                    Kind::Counter => metrics::counter!(def.name).increment(1),
                    Kind::Gauge => metrics::gauge!(def.name).set(1.0),
                    Kind::Histogram => metrics::histogram!(def.name).record(0.5),
                }
            }
        });
        let body = handle.render();

        for def in METRICS {
            assert!(
                body.contains(&format!("# HELP {} ", def.name)),
                "{} rendered without help text",
                def.name
            );
            match def.kind {
                Kind::Counter => {
                    assert!(
                        body.contains(&format!("# TYPE {} counter", def.name)),
                        "{} did not render as a counter",
                        def.name
                    );
                }
                Kind::Gauge => {
                    assert!(
                        body.contains(&format!("# TYPE {} gauge", def.name)),
                        "{} did not render as a gauge",
                        def.name
                    );
                }
                Kind::Histogram => {
                    assert!(
                        body.contains(&format!("# TYPE {} histogram", def.name)),
                        "{} rendered as a summary, not a histogram: its bucket set did not \
                         reach it, so its quantiles cannot be aggregated across replicas",
                        def.name
                    );
                    assert!(
                        body.contains(&format!("{}_bucket{{le=", def.name)),
                        "{} rendered no _bucket series",
                        def.name
                    );
                    assert!(
                        body.contains(&format!("{}_sum ", def.name))
                            && body.contains(&format!("{}_count ", def.name)),
                        "{} rendered without _sum/_count",
                        def.name
                    );
                }
            }
        }
        // And nothing rendered as a summary, which is what the whole crate is
        // here to prevent.
        assert!(
            !body.contains("quantile=\""),
            "some metric still renders as a summary:\n{body}"
        );
    }

    #[test]
    fn every_bucket_set_is_usable() {
        for buckets in [
            BUCKETS_TICK,
            BUCKETS_REQUEST,
            BUCKETS_TTFP,
            BUCKETS_QUEUE,
            BUCKETS_RATIO,
            BUCKETS_BATCH,
            BUCKETS_AUDIO,
        ] {
            assert!(buckets.windows(2).all(|w| w[0] < w[1]));
            assert!(buckets.iter().all(|b| b.is_finite() && *b >= 0.0));
        }
    }
}
