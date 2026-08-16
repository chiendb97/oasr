// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Transport-level HTTP metrics, and the per-surface request recorders.
//!
//! Two things, both about labels.
//!
//! The middleware labels by the **matched route pattern**
//! (`/v1/audio/transcriptions`), never `req.uri().path()`. The raw path is the
//! classic way to melt a Prometheus server: it is unbounded, and every
//! `/v1/speech:recognize?...` variant, every probe from a scanner and every
//! typo becomes a permanent time series. A route that did not match gets one
//! fixed label instead, so 404 traffic is counted without being enumerated.
//!
//! The recorders are `OnceLock` statics, one per API surface, because
//! [`oasr_metrics::RequestRecorder`] resolves its handles at construction: the
//! point is to pay the label lookup once per process rather than once per
//! request.

use std::sync::OnceLock;
use std::time::Instant;

use axum::extract::MatchedPath;
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use oasr_metrics as om;
use oasr_metrics::RequestRecorder;

/// Label value for a request that matched no route.
///
/// A literal, not the requested path: the whole point of labelling by matched
/// route is that the label set stays bounded, and 404s are exactly the traffic
/// an attacker (or a broken client) controls the shape of.
const UNMATCHED_ROUTE: &str = "<unmatched>";

/// `POST /v1/speech:recognize` — the Google-shaped surface.
pub fn google_offline() -> &'static RequestRecorder {
    static R: OnceLock<RequestRecorder> = OnceLock::new();
    R.get_or_init(|| RequestRecorder::new(om::api::GOOGLE, om::mode::OFFLINE))
}

/// `POST /v1/audio/{transcriptions,translations}`.
pub fn openai_offline() -> &'static RequestRecorder {
    static R: OnceLock<RequestRecorder> = OnceLock::new();
    R.get_or_init(|| RequestRecorder::new(om::api::OPENAI, om::mode::OFFLINE))
}

/// `GET /v1/realtime` — the streaming WebSocket.
pub fn realtime_streaming() -> &'static RequestRecorder {
    static R: OnceLock<RequestRecorder> = OnceLock::new();
    R.get_or_init(|| RequestRecorder::new(om::api::REALTIME, om::mode::STREAMING))
}

/// Decrements the in-flight gauge when dropped.
///
/// Not a plain `decrement` after the `await`: a handler that panics, or a
/// future dropped when the client disconnects mid-request, never reaches the
/// line after it. Either would leave the gauge permanently one too high, and
/// since it only ever moves up, an "in flight" reading would drift away from
/// reality until a restart — a gauge that slowly becomes a lie is worse than
/// one that was never added.
struct InFlightGuard(metrics::Gauge);

impl InFlightGuard {
    fn new(gauge: metrics::Gauge) -> Self {
        gauge.increment(1.0);
        Self(gauge)
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.decrement(1.0);
    }
}

/// A `&'static` name for a status code, avoiding a per-request allocation.
///
/// Only the codes this server actually returns; anything else falls back to
/// `other`, which keeps the label bounded even if a new code appears.
fn status_label(code: u16) -> &'static str {
    match code {
        200 => "200",
        400 => "400",
        401 => "401",
        403 => "403",
        404 => "404",
        408 => "408",
        413 => "413",
        429 => "429",
        499 => "499",
        500 => "500",
        503 => "503",
        504 => "504",
        101 => "101",
        _ => "other",
    }
}

/// Count and time every HTTP request, by method, matched route and status.
///
/// Method and status are `&'static` (hyper's method table, and the map above),
/// so the only per-request allocations are the route string and the label
/// vectors the `metrics` macros build. Interning the route to make it static
/// too was tried and reverted: it needs a shared table behind a lock, and a
/// mutex taken on every HTTP request is a contention point far more expensive,
/// under the concurrency this server is built for, than the allocation it
/// removes.
pub async fn track_http(req: Request, next: Next) -> Response {
    let method = method_label(req.method());
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_owned())
        .unwrap_or_else(|| UNMATCHED_ROUTE.to_owned());

    let _guard = InFlightGuard::new(metrics::gauge!(
        om::HTTP_IN_FLIGHT,
        om::label::METHOD => method,
        om::label::ROUTE => route.clone(),
    ));

    let start = Instant::now();
    let response = next.run(req).await;
    let elapsed = start.elapsed();

    metrics::counter!(
        om::HTTP_REQUESTS,
        om::label::METHOD => method,
        om::label::ROUTE => route.clone(),
        om::label::STATUS => status_label(response.status().as_u16()),
    )
    .increment(1);
    metrics::histogram!(
        om::HTTP_DURATION,
        om::label::METHOD => method,
        om::label::ROUTE => route,
    )
    .record(elapsed.as_secs_f64());

    response
}

/// A `&'static` name for an HTTP method.
fn method_label(m: &axum::http::Method) -> &'static str {
    match *m {
        axum::http::Method::GET => "GET",
        axum::http::Method::POST => "POST",
        axum::http::Method::PUT => "PUT",
        axum::http::Method::DELETE => "DELETE",
        axum::http::Method::HEAD => "HEAD",
        axum::http::Method::OPTIONS => "OPTIONS",
        axum::http::Method::PATCH => "PATCH",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gauge must come back down on the paths that skip the happy exit —
    /// a panicking handler, or a future dropped when the client disconnects.
    #[test]
    fn in_flight_guard_decrements_when_dropped_early() {
        let recorder = oasr_metrics::builder().unwrap().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let g = || metrics::gauge!(om::HTTP_IN_FLIGHT, om::label::ROUTE => "/t");
            let held = InFlightGuard::new(g());
            assert!(handle
                .render()
                .contains("oasr_http_requests_in_flight{route=\"/t\"} 1"));
            // Unwinding past the guard, as a panicking handler would.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _inner = InFlightGuard::new(g());
                panic!("handler blew up");
            }));
            drop(held);
            assert!(
                handle
                    .render()
                    .contains("oasr_http_requests_in_flight{route=\"/t\"} 0"),
                "gauge did not return to zero:\n{}",
                handle.render()
            );
        });
    }

    #[test]
    fn labels_are_static_and_bounded() {
        assert_eq!(method_label(&axum::http::Method::POST), "POST");
        assert_eq!(status_label(503), "503");
        // An unexpected code must not become its own series.
        assert_eq!(status_label(418), "other");
    }
}
