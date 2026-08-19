// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! What `GET /metrics` actually renders.
//!
//! One test, not several: the recorder is process-global and can only be
//! installed once, so parallel tests would race over the same series. Driving
//! the whole scenario in order and asserting against a single render is both
//! correct and closer to what a scrape sees.
//!
//! Like `routes.rs`, this runs against an [`EnginePool`] with no workers — the
//! submit fails with `RESOURCE_EXHAUSTED`, which is exactly what makes the
//! error-outcome assertions below reachable without a Python interpreter.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use oasr_engine_client::EnginePool;
use oasr_server_http::{build_router, RouterLimits, ServerState, ServiceMode};
use tower::ServiceExt; // `oneshot`

/// 16 kHz mono 16-bit WAV of `n` samples.
fn wav(n: usize) -> Vec<u8> {
    let data_len = (n * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&16_000u32.to_le_bytes());
    out.extend_from_slice(&32_000u32.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend(std::iter::repeat_n(0u8, n * 2));
    out
}

/// Every line of the exposition carrying `metric`, including its labels.
fn series<'a>(body: &'a str, metric: &str) -> Vec<&'a str> {
    body.lines()
        .filter(|l| !l.starts_with('#') && l.starts_with(metric))
        .collect()
}

#[tokio::test]
async fn metrics_endpoint_renders_histograms_and_labels() {
    let prometheus = oasr_metrics::install_recorder().expect("install recorder");

    let state = Arc::new(ServerState {
        pool: Arc::new(EnginePool::new(Vec::new())),
        prometheus: Some(prometheus),
        service_mode: ServiceMode::Offline,
        sample_rate: 16_000,
        max_body_bytes: 1024 * 1024,
        max_audio_samples: Some(16_000 * 60),
        served_model_names: Vec::new(),
        model_id: "/ckpt/u2pp-conformer".into(),
    });
    let app = build_router(state, RouterLimits::default());

    // A Google-shaped recognize: two seconds of audio, submitted to an empty
    // pool, so it reaches the submit and fails there.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/speech:recognize?encoding=LINEAR16&sample_rate_hertz=16000")
                .body(Body::from(wav(32_000)))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);

    // A health probe, and a path that matches no route at all.
    for uri in ["/healthz", "/no/such/route"] {
        let _ = app
            .clone()
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
    }

    let res = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();

    // Require aggregatable histogram buckets, not rolling summary quantiles.
    assert!(
        body.contains("oasr_http_request_duration_seconds_bucket"),
        "no _bucket series: histograms regressed to summaries.\n{body}"
    );
    assert!(
        body.contains("oasr_http_request_duration_seconds_bucket{") && body.contains("le=\""),
        "bucket series carry no `le` label.\n{body}"
    );
    assert!(
        !body.contains("oasr_http_request_duration_seconds{quantile="),
        "histogram is still rendering as a summary.\n{body}"
    );
    assert!(
        body.contains("# TYPE oasr_http_request_duration_seconds histogram"),
        "histogram is not TYPEd as one.\n{body}"
    );

    // ---- Help text is registered ---------------------------------------
    assert!(
        body.contains("# HELP oasr_http_requests_total"),
        "descriptions were not registered.\n{body}"
    );

    // ---- Route labels are matched patterns, never request URIs ----------
    //
    // The raw path is unbounded — every query string and every scanner probe
    // would become a permanent series — so the label must be the route
    // pattern, and an unmatched request must collapse to one fixed value.
    let http = series(&body, "oasr_http_requests_total");
    assert!(
        http.iter()
            .any(|l| l.contains("route=\"/v1/speech:recognize\"")),
        "no matched-route label; MatchedPath is not reaching the middleware.\n{body}"
    );
    assert!(
        http.iter().any(|l| l.contains("route=\"<unmatched>\"")),
        "an unmatched request was not counted under one fixed route label.\n{body}"
    );
    assert!(
        !body.contains("encoding=LINEAR16"),
        "the query string leaked into a label — this is the cardinality bomb.\n{body}"
    );
    assert!(
        !body.contains("route=\"/no/such/route\""),
        "an unmatched path was labelled with its own URI.\n{body}"
    );
    assert!(
        http.iter().any(|l| l.contains("status=\"503\"")),
        "no status label on the failed recognize.\n{body}"
    );

    // ---- Request-scope series ------------------------------------------
    //
    // Audio is counted before the submit, so a request the pool refuses still
    // reaches the RTFx denominator.
    let audio = series(&body, "oasr_audio_seconds_total");
    assert!(
        audio
            .iter()
            .any(|l| l.contains("api=\"google\"") && l.contains("mode=\"offline\"")),
        "audio was not counted, or not labelled by surface.\n{body}"
    );
    assert!(
        audio.iter().any(|l| l.ends_with(" 2")),
        "expected 2 s of audio counted for a 32000-sample request.\n{audio:?}"
    );
    assert!(
        series(&body, "oasr_request_duration_seconds_bucket")
            .iter()
            .any(|l| l.contains("outcome=\"error\"")),
        "a refused request was not recorded with an error outcome.\n{body}"
    );

    // Whether *every* declared metric renders correctly is checked in
    // `oasr-metrics` against a local recorder, where each one can be recorded
    // into first — the exporter emits nothing at all for a metric that has no
    // samples, so a description-only check here would pass vacuously.
}
