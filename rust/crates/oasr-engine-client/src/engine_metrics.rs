// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Per-engine metric labels, pre-resolved tick handles, and the replay of the
//! engine's own metric snapshot.
//!
//! Two things live here that the dispatcher would otherwise pay for on its
//! hottest path.
//!
//! **Labels resolved once.**  `metrics::histogram!(name, "engine" => label)`
//! hashes the name plus labels on every call, and a label value that is not
//! `'static` allocates a `String` each time.  The tick loop records six series
//! per tick at up to ~1 kHz, on the thread that owns the GIL — so the lookup
//! and the allocation land exactly where the process can least afford them.
//! An engine's labels never change after construction, so [`TickHandles`]
//! resolves them all once at thread start and the loop records into handles.
//!
//! **The Python drain.**  Batch widths, padding, stage timings and pool
//! occupancy are only knowable inside the engine.  [`EngineSnapshot`] is what
//! `ASREngine.metrics_snapshot()` hands over, and [`replay`] pushes it into the
//! exporter.  Counters arrive **absolute** and are replayed with
//! `Counter::absolute`, which makes the whole protocol idempotent: a missed
//! drain loses nothing, and a repeated one double-counts nothing.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use metrics::{Counter, Gauge, Histogram, Label};
use oasr_metrics as om;
use oasr_wire::ModelInfo;
use tracing::warn;

/// How often the engine's own metrics are drained.
///
/// Not every tick: the drain is a Python call plus a dict walk under the GIL,
/// and at a 1 kHz tick rate that is pure overhead against a Prometheus scrape
/// that arrives every 15 s.  Four times a second is far fresher than any
/// scrape and costs nothing measurable.
pub(crate) const DRAIN_INTERVAL: Duration = Duration::from_millis(250);

/// The label set every engine-scope series carries.
#[derive(Clone)]
pub struct EngineLabels(Arc<Vec<Label>>);

impl EngineLabels {
    /// Build from the engine's label and whatever the model reported.
    ///
    /// A field the engine did not report becomes [`om::UNKNOWN`] rather than
    /// being dropped: a sample with one fewer label is a *different* time
    /// series, so omitting it would silently split the series in two rather
    /// than leave a visible gap.
    pub fn new(engine: &str, model: Option<&ModelInfo>) -> Self {
        let name = model
            .and_then(|m| m.ckpt_dir.as_deref())
            .map(short_model_name)
            .unwrap_or_else(|| om::UNKNOWN.to_string());
        let decode = model
            .and_then(|m| m.decode_method.as_deref())
            .unwrap_or(om::UNKNOWN)
            .to_string();
        Self(Arc::new(vec![
            Label::new(om::label::ENGINE, engine.to_string()),
            Label::new(om::label::MODEL, name),
            Label::new(om::label::DECODE_METHOD, decode),
        ]))
    }

    /// An empty label set, for tests and for a router with no engine context.
    pub fn none() -> Self {
        Self(Arc::new(Vec::new()))
    }

    pub fn as_slice(&self) -> &[Label] {
        &self.0
    }

    /// The label set in the form the `metrics` macros accept.
    ///
    /// `metrics` implements `IntoLabels` for `slice::Iter<'_, Label>` but not
    /// for `&[Label]`, so this is the borrow that records without building an
    /// intermediate `Vec` at the call site.
    pub fn iter(&self) -> std::slice::Iter<'_, Label> {
        self.0.iter()
    }

    /// This label set plus one more — for the series keyed by stage, mode or
    /// failure stage, which cannot be pre-resolved to a handle.
    pub fn with(&self, key: &'static str, value: impl Into<String>) -> Vec<Label> {
        let mut labels = Vec::with_capacity(self.0.len() + 1);
        labels.extend(self.0.iter().cloned());
        labels.push(Label::new(key, value.into()));
        labels
    }
}

/// Reduce a checkpoint path to something usable as a label value.
///
/// Use the last path component to avoid leaking unstable deployment paths.
fn short_model_name(ckpt_dir: &str) -> String {
    ckpt_dir
        .trim_end_matches('/')
        .rsplit('/')
        .find(|s| !s.is_empty())
        .unwrap_or(ckpt_dir)
        .to_string()
}

/// Metric handles resolved once per engine, recorded on **every tick**.
///
/// Only hot per-tick series are pre-resolved; request-rate counters remain direct.
pub(crate) struct TickHandles {
    pub tick: Histogram,
    pub admit: Histogram,
    pub step: Histogram,
    pub extract: Histogram,
    pub route: Histogram,
    pub running: Gauge,
    pub waiting: Gauge,
    pub outputs: Counter,
    pub step_failures: Counter,
}

impl TickHandles {
    pub fn new(labels: &EngineLabels) -> Self {
        let l = || labels.iter();
        Self {
            tick: metrics::histogram!(om::DISPATCH_TICK, l()),
            admit: metrics::histogram!(om::DISPATCH_ADMIT, l()),
            step: metrics::histogram!(om::ENGINE_STEP, l()),
            extract: metrics::histogram!(om::DISPATCH_EXTRACT, l()),
            route: metrics::histogram!(om::DISPATCH_ROUTE, l()),
            running: metrics::gauge!(om::ENGINE_RUNNING, l()),
            waiting: metrics::gauge!(om::ENGINE_WAITING, l()),
            outputs: metrics::counter!(om::ENGINE_OUTPUTS, l()),
            step_failures: metrics::counter!(om::ENGINE_STEP_FAILURES, l()),
        }
    }
}

/// What `ASREngine.metrics_snapshot()` returns.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct EngineSnapshot {
    /// Monotonic totals, absolute.
    pub counters: HashMap<String, f64>,
    pub gauges: HashMap<String, f64>,
    /// Histogram samples since the previous drain.
    pub hist: HashMap<String, Vec<f64>>,
    /// Histogram samples carrying one extra label, keyed by that label's value.
    pub keyed_hist: HashMap<String, HashMap<String, Vec<f64>>>,
}

impl EngineSnapshot {
    pub fn is_empty(&self) -> bool {
        self.counters.is_empty()
            && self.gauges.is_empty()
            && self.hist.is_empty()
            && self.keyed_hist.is_empty()
    }
}

/// Push one engine snapshot into the exporter.
pub(crate) fn replay(snapshot: &EngineSnapshot, labels: &EngineLabels) {
    for (name, value) in &snapshot.counters {
        // `absolute`, not `increment`: the engine keeps the total, so replaying
        // the same snapshot twice is a no-op and skipping one loses nothing.
        // Truncation is against the running total, so it never accumulates.
        metrics::counter!(clone_name(name), labels.iter()).absolute(*value as u64);
    }
    for (name, value) in &snapshot.gauges {
        metrics::gauge!(clone_name(name), labels.iter()).set(*value);
    }
    for (name, samples) in &snapshot.hist {
        let h = metrics::histogram!(clone_name(name), labels.iter());
        for s in samples {
            h.record(*s);
        }
    }
    for (name, by_key) in &snapshot.keyed_hist {
        let Some(label_key) = om::keyed_label_for(name) else {
            // Recording it unlabelled would merge every key into one series —
            // a plausible number that is not any key's distribution.  Drop it
            // and say so; a missing series with a warning is diagnosable, a
            // silently-merged one is not.
            warn!(
                metric = %name,
                "engine reported a keyed histogram with no declared label in oasr-metrics; \
                 dropping it rather than merging its keys into one series"
            );
            continue;
        };
        for (key, samples) in by_key {
            let h = metrics::histogram!(clone_name(name), labels.with(label_key, key.clone()));
            for s in samples {
                h.record(*s);
            }
        }
    }
}

/// The metric macros want an owned name when it is not a literal.
#[inline]
fn clone_name(name: &str) -> String {
    name.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_label_is_the_checkpoint_directory_name() {
        assert_eq!(short_model_name("/models/conformer-u2pp"), "conformer-u2pp");
        assert_eq!(short_model_name("/models/whisper-tiny/"), "whisper-tiny");
        assert_eq!(short_model_name("relative"), "relative");
        assert_eq!(short_model_name("/"), "/");
    }

    /// A missing model field must produce a labelled sample, not an unlabelled
    /// one — otherwise the same metric lands in two different series depending
    /// on whether the engine had reported its model yet.
    #[test]
    fn absent_model_fields_become_unknown_rather_than_absent() {
        let labels = EngineLabels::new("e0", None);
        let keys: Vec<_> = labels
            .as_slice()
            .iter()
            .map(|l| l.key().to_owned())
            .collect();
        assert_eq!(
            keys,
            vec![
                om::label::ENGINE,
                om::label::MODEL,
                om::label::DECODE_METHOD
            ]
        );
        let values: Vec<_> = labels
            .as_slice()
            .iter()
            .map(|l| l.value().to_owned())
            .collect();
        assert_eq!(values, vec!["e0", om::UNKNOWN, om::UNKNOWN]);
    }

    #[test]
    fn with_appends_without_disturbing_the_base_set() {
        let labels = EngineLabels::new("e0", None);
        let keyed = labels.with(om::label::STAGE, "offline.encode");
        assert_eq!(keyed.len(), labels.as_slice().len() + 1);
        assert_eq!(keyed.last().unwrap().value(), "offline.encode");
        assert_eq!(labels.as_slice().len(), 3, "base set must not be mutated");
    }
}
