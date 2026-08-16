// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! The shipped Grafana dashboard only references metrics that exist.
//!
//! A panel whose query names a metric nobody exports does not error — it draws
//! an empty graph, which is indistinguishable from a healthy system with no
//! traffic. Renaming a metric therefore breaks the dashboard silently, and in
//! the worst way: the panel an operator is watching during an incident is the
//! one that quietly went blank.
//!
//! Compiled into the test binary with `include_str!`, so the dashboard is
//! checked against the same [`oasr_metrics::METRICS`] table the server exports
//! from — not against a copy of the names.

use std::collections::HashSet;

const DASHBOARD: &str = include_str!("../../../../docker/monitoring/grafana/oasr-overview.json");

/// Every `oasr_`-prefixed identifier in `text`.
fn oasr_identifiers(text: &str) -> HashSet<String> {
    let bytes = text.as_bytes();
    let mut found = HashSet::new();
    let mut i = 0;
    while let Some(rel) = text[i..].find("oasr_") {
        let start = i + rel;
        let mut end = start;
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
            end += 1;
        }
        found.insert(text[start..end].to_string());
        i = end.max(start + 1);
    }
    found
}

/// Strip the suffixes Prometheus adds to a histogram's exported series.
fn base_name(name: &str) -> &str {
    for suffix in ["_bucket", "_sum", "_count"] {
        if let Some(stripped) = name.strip_suffix(suffix) {
            return stripped;
        }
    }
    name
}

#[test]
fn dashboard_references_only_declared_metrics() {
    let declared: HashSet<&str> = oasr_metrics::METRICS.iter().map(|d| d.name).collect();
    let referenced = oasr_identifiers(DASHBOARD);
    assert!(
        !referenced.is_empty(),
        "found no oasr_ metrics in the dashboard at all — did the file move?"
    );

    let unknown: Vec<&String> = referenced
        .iter()
        .filter(|name| !declared.contains(base_name(name)))
        .collect();
    assert!(
        unknown.is_empty(),
        "the dashboard queries metrics that are not declared in oasr-metrics: {unknown:?}. \
         Those panels render empty rather than failing, so nothing else would catch this."
    );
}

/// A histogram may only be queried through its exported series.
///
/// `histogram_quantile(0.99, rate(oasr_x_seconds[5m]))` — without `_bucket` —
/// is the single most common Prometheus mistake, and it returns a number
/// rather than an error, so a dashboard can ship with it and look right.
#[test]
fn histograms_are_queried_through_their_exported_series() {
    let histograms: Vec<&str> = oasr_metrics::METRICS
        .iter()
        .filter(|d| d.kind == oasr_metrics::Kind::Histogram)
        .map(|d| d.name)
        .collect();

    for name in oasr_identifiers(DASHBOARD) {
        if histograms.contains(&name.as_str()) {
            panic!(
                "the dashboard references the histogram {name} by its bare name; a histogram is \
                 only readable as {name}_bucket / _sum / _count"
            );
        }
    }
}

/// Counters and gauges are the mirror image: they have no `_bucket` series, so
/// querying one is a panel that can only ever be empty.
#[test]
fn non_histograms_are_not_queried_as_buckets() {
    let non_histograms: HashSet<&str> = oasr_metrics::METRICS
        .iter()
        .filter(|d| d.kind != oasr_metrics::Kind::Histogram)
        .map(|d| d.name)
        .collect();

    for name in oasr_identifiers(DASHBOARD) {
        if let Some(base) = name.strip_suffix("_bucket") {
            assert!(
                !non_histograms.contains(base),
                "{name}: {base} is not a histogram, so it has no bucket series"
            );
        }
    }
}
