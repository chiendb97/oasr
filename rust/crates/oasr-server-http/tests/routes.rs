// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Route-level tests against the real axum router.
//!
//! The engine needs a Python interpreter, so these run against an [`EnginePool`]
//! with **no workers**: everything up to the submit is exercised for real —
//! routing, multipart parsing, model validation, audio decode, body limits,
//! response and error shapes — and the submit itself fails with
//! `RESOURCE_EXHAUSTED`.  That boundary is exactly right for what is being
//! checked: a request that reaches the submit is a request whose upload was
//! parsed and whose audio decoded, which is what the two new endpoints add.
//!
//! Reaching 503 is therefore the *success* condition in several tests below,
//! and the tests that assert 4xx prove the request was rejected before it ever
//! got there.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use oasr_engine_client::EnginePool;
use oasr_server_http::{build_router, RouterLimits, ServerState, ServiceMode};
use serde_json::Value;
use tower::ServiceExt; // `oneshot`

const BOUNDARY: &str = "----oasrtestboundary";

fn state(mode: ServiceMode, served: &[&str]) -> Arc<ServerState> {
    Arc::new(ServerState {
        pool: Arc::new(EnginePool::new(Vec::new())),
        prometheus: None,
        service_mode: mode,
        sample_rate: 16_000,
        max_body_bytes: 1024 * 1024,
        max_audio_samples: Some(16_000 * 60),
        served_model_names: served.iter().map(|s| (*s).to_owned()).collect(),
        model_id: "/ckpt/u2pp-conformer".into(),
    })
}

fn router(mode: ServiceMode, served: &[&str]) -> axum::Router {
    build_router(state(mode, served), RouterLimits::default())
}

/// A valid 16 kHz mono 16-bit WAV of `n` samples, built by hand so the test
/// needs no encoder and no fixture path.
fn wav(n: usize) -> Vec<u8> {
    let data_len = (n * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&16_000u32.to_le_bytes());
    out.extend_from_slice(&32_000u32.to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..n {
        out.extend_from_slice(&(((i % 100) as i16) * 100).to_le_bytes());
    }
    out
}

/// Build a multipart body from `(name, value)` text parts plus an optional file.
fn multipart(parts: &[(&str, &str)], file: Option<(&str, &str, &[u8])>) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in parts {
        body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    if let Some((filename, content_type, bytes)) = file {
        body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n\
                 Content-Type: {content_type}\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    body
}

/// A raw-body request for the Google-shaped route (config in the query string).
fn raw_body(path: &str, bytes: &[u8]) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .body(Body::from(bytes.to_vec()))
        .expect("build request")
}

fn upload(path: &str, parts: &[(&str, &str)], file: Option<(&str, &str, &[u8])>) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(multipart(parts, file)))
        .expect("build request")
}

async fn send(router: axum::Router, req: Request<Body>) -> (StatusCode, String) {
    let resp = router.oneshot(req).await.expect("router response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn json(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("expected JSON, got {body:?}: {e}"))
}

// ---------------------------------------------------------------------------
// Operability + models
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_answers_while_the_engine_is_down() {
    let (status, body) = send(
        router(ServiceMode::Offline, &[]),
        Request::builder()
            .uri("/healthz")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, "ok");

    // Readiness tracks the *engine*, which has no workers here.
    let (status, _) = send(
        router(ServiceMode::Offline, &[]),
        Request::builder()
            .uri("/readyz")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn models_lists_the_checkpoint_and_its_aliases() {
    let (status, body) = send(
        router(ServiceMode::Offline, &["whisper-1"]),
        Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v = json(&body);
    assert_eq!(v["object"], "list", "OpenAI clients branch on this");
    let ids: Vec<&str> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["/ckpt/u2pp-conformer", "whisper-1"]);
    assert_eq!(v["data"][0]["object"], "model");
}

// ---------------------------------------------------------------------------
// POST /v1/audio/transcriptions
// ---------------------------------------------------------------------------

/// The whole point of the endpoint: a plain multipart upload is parsed, its
/// audio decoded, and the request handed to the engine.  With no workers the
/// engine leg fails — reaching *that* failure is the assertion.
#[tokio::test]
async fn a_wav_upload_is_decoded_and_submitted() {
    let audio = wav(16_000);
    let (status, body) = send(
        router(ServiceMode::Offline, &[]),
        upload(
            "/v1/audio/transcriptions",
            &[("model", "whisper-1"), ("response_format", "json")],
            Some(("speech.wav", "audio/wav", &audio)),
        ),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "upload parsed and decoded; only the engine leg should fail: {body}"
    );
    assert_eq!(json(&body)["error"]["type"], "server_error");
}

#[tokio::test]
async fn a_missing_file_part_is_a_400_naming_the_param() {
    let (status, body) = send(
        router(ServiceMode::Offline, &[]),
        upload("/v1/audio/transcriptions", &[("model", "whisper-1")], None),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let v = json(&body);
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["param"], "file");
}

/// Undecodable bytes must be rejected before admission, not turned into a
/// transcript of noise.
#[tokio::test]
async fn an_undecodable_upload_is_rejected() {
    let (status, body) = send(
        router(ServiceMode::Offline, &[]),
        upload(
            "/v1/audio/transcriptions",
            &[],
            Some(("notes.txt", "text/plain", b"this is not audio")),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json(&body)["error"]["param"], "file");
}

/// `--served-model-name` turns an unknown `model` into a 404, as OpenAI does.
#[tokio::test]
async fn an_unknown_model_is_404_when_names_are_declared() {
    let audio = wav(1_600);
    let (status, body) = send(
        router(ServiceMode::Offline, &["whisper-1"]),
        upload(
            "/v1/audio/transcriptions",
            &[("model", "gpt-4o-transcribe")],
            Some(("a.wav", "audio/wav", &audio)),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json(&body)["error"]["param"], "model");
}

/// With no names declared — the single-model default — *any* `model` is
/// accepted, which is what makes "change one base URL" true for a client
/// hardcoded to `whisper-1`.
#[tokio::test]
async fn any_model_name_is_accepted_by_default() {
    let audio = wav(1_600);
    let (status, _) = send(
        router(ServiceMode::Offline, &[]),
        upload(
            "/v1/audio/transcriptions",
            &[("model", "gpt-4o-transcribe")],
            Some(("a.wav", "audio/wav", &audio)),
        ),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "reached the engine"
    );
}

/// H7 landed: `word` granularity is now carried through to the engine instead
/// of being refused.  A worker-less pool means 503 — the marker that the
/// request parsed, decoded and reached submission.  (Whether the *decode
/// family* can align is decided at admission, inside the engine, which is why
/// the route no longer has an opinion.)
#[tokio::test]
async fn word_granularity_reaches_the_engine() {
    let audio = wav(1_600);
    for granularity in ["word", "segment"] {
        let (status, body) = send(
            router(ServiceMode::Offline, &[]),
            upload(
                "/v1/audio/transcriptions",
                &[
                    ("response_format", "verbose_json"),
                    ("timestamp_granularities[]", granularity),
                ],
                Some(("a.wav", "audio/wav", &audio)),
            ),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "granularity={granularity}: {body}"
        );
    }
}

/// The parameter only means something for `verbose_json`; the other formats
/// have nowhere to render an array.  Saying so beats accepting it and returning
/// `{"text": ...}`, which a client cannot distinguish from "no words found".
#[tokio::test]
async fn granularities_outside_verbose_json_are_rejected() {
    let audio = wav(1_600);
    let (status, body) = send(
        router(ServiceMode::Offline, &[]),
        upload(
            "/v1/audio/transcriptions",
            &[
                ("response_format", "json"),
                ("timestamp_granularities[]", "word"),
            ],
            Some(("a.wav", "audio/wav", &audio)),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json(&body)["error"]["param"], "timestamp_granularities");
}

#[tokio::test]
async fn an_unknown_granularity_is_rejected() {
    let audio = wav(1_600);
    let (status, body) = send(
        router(ServiceMode::Offline, &[]),
        upload(
            "/v1/audio/transcriptions",
            &[
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "phoneme"),
            ],
            Some(("a.wav", "audio/wav", &audio)),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json(&body)["error"]["param"], "timestamp_granularities");
}

/// The Google-shaped route carries the same request through its own spelling.
#[tokio::test]
async fn enable_word_time_offsets_reaches_the_engine() {
    let audio = wav(1_600);
    let (status, body) = send(
        router(ServiceMode::Offline, &[]),
        raw_body(
            "/v1/speech:recognize?encoding=WAV&enable_word_time_offsets=true",
            &audio,
        ),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
}

#[tokio::test]
async fn an_unknown_response_format_is_rejected() {
    let audio = wav(1_600);
    let (status, body) = send(
        router(ServiceMode::Offline, &[]),
        upload(
            "/v1/audio/transcriptions",
            &[("response_format", "yaml")],
            Some(("a.wav", "audio/wav", &audio)),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json(&body)["error"]["param"], "response_format");
}

#[tokio::test]
async fn streaming_rejects_the_formats_that_need_a_total_duration() {
    let audio = wav(1_600);
    for format in ["srt", "vtt", "verbose_json"] {
        let (status, body) = send(
            router(ServiceMode::Offline, &[]),
            upload(
                "/v1/audio/transcriptions",
                &[("stream", "true"), ("response_format", format)],
                Some(("a.wav", "audio/wav", &audio)),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{format}");
        assert_eq!(json(&body)["error"]["param"], "stream", "{format}");
    }
}

/// A language the server cannot parse must fail the request: dropping it would
/// transcribe in the checkpoint's own language while the client believes
/// otherwise, and nothing in the response would say so.
#[tokio::test]
async fn an_unparseable_language_is_rejected() {
    let audio = wav(1_600);
    let (status, body) = send(
        router(ServiceMode::Offline, &[]),
        upload(
            "/v1/audio/transcriptions",
            &[("language", "!!")],
            Some(("a.wav", "audio/wav", &audio)),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("language"), "{body}");
}

/// The decoded-length ceiling has to fire on the *decoded* waveform, and
/// report a size status rather than a generic 400.
#[tokio::test]
async fn audio_longer_than_the_ceiling_is_413() {
    let mut st = state(ServiceMode::Offline, &[]);
    Arc::get_mut(&mut st).unwrap().max_audio_samples = Some(8_000);
    let audio = wav(16_000); // one second at 16 kHz
    let (status, body) = send(
        build_router(st, RouterLimits::default()),
        upload(
            "/v1/audio/transcriptions",
            &[],
            Some(("a.wav", "audio/wav", &audio)),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(body.contains("too long"), "{body}");
}

/// axum's own multipart default is 2 MiB — an order of magnitude under
/// `--max-audio-mib`, so without the explicit layer the upload route would
/// reject audio the raw-body route on the same server accepts.
#[tokio::test]
async fn the_body_limit_follows_max_audio_mib() {
    let mut st = state(ServiceMode::Offline, &[]);
    Arc::get_mut(&mut st).unwrap().max_body_bytes = 4 * 1024;
    let audio = wav(16_000); // ~32 KiB, well over the 4 KiB cap
    let (status, _) = send(
        build_router(st, RouterLimits::default()),
        upload(
            "/v1/audio/transcriptions",
            &[],
            Some(("a.wav", "audio/wav", &audio)),
        ),
    )
    .await;
    assert!(
        status == StatusCode::PAYLOAD_TOO_LARGE || status == StatusCode::BAD_REQUEST,
        "an over-sized body must be refused, got {status}"
    );
}

/// The translations route exists and is distinct from transcriptions.
#[tokio::test]
async fn the_translations_route_is_served() {
    let audio = wav(1_600);
    let (status, _) = send(
        router(ServiceMode::Offline, &[]),
        upload(
            "/v1/audio/translations",
            &[],
            Some(("a.wav", "audio/wav", &audio)),
        ),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "reached the engine"
    );
}

/// A streaming-mode server cannot serve a buffered upload; it must say what to
/// use instead.
#[tokio::test]
async fn a_streaming_engine_points_uploads_at_the_websocket() {
    let audio = wav(1_600);
    let (status, body) = send(
        router(ServiceMode::Streaming, &[]),
        upload(
            "/v1/audio/transcriptions",
            &[],
            Some(("a.wav", "audio/wav", &audio)),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("/v1/realtime"), "{body}");
}

// ---------------------------------------------------------------------------
// The Google-shaped route still works
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_google_shaped_route_is_unchanged() {
    let audio = wav(1_600);
    let (status, body) = send(
        router(ServiceMode::Offline, &[]),
        Request::builder()
            .method("POST")
            .uri("/v1/speech:recognize?encoding=WAV")
            .body(Body::from(audio))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    // Google's envelope, not OpenAI's — the two surfaces keep their own shapes.
    assert_eq!(json(&body)["error"]["status"], "RESOURCE_EXHAUSTED");
}

/// New codecs must be reachable from the query-string surface too, not only
/// from the uploads.
#[tokio::test]
async fn the_google_shaped_route_accepts_the_new_encodings() {
    for (encoding, body) in [
        ("MULAW", vec![0xFFu8; 800]),
        ("ALAW", vec![0xD5u8; 800]),
        ("AUTO", wav(800)),
    ] {
        let (status, text) = send(
            router(ServiceMode::Offline, &[]),
            Request::builder()
                .method("POST")
                .uri(format!(
                    "/v1/speech:recognize?encoding={encoding}&sample_rate=8000"
                ))
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "{encoding} should decode and reach the engine: {text}"
        );
    }
}

#[tokio::test]
async fn opus_reports_unimplemented_rather_than_a_decode_failure() {
    let (status, body) = send(
        router(ServiceMode::Offline, &[]),
        Request::builder()
            .method("POST")
            .uri("/v1/speech:recognize?encoding=OGG_OPUS")
            .body(Body::from(vec![0u8; 64]))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(json(&body)["error"]["status"], "UNIMPLEMENTED");
}

// ---------------------------------------------------------------------------
// CORS
// ---------------------------------------------------------------------------

/// A browser cannot call the API cross-origin without this, which is the whole
/// reason `examples/web` needed a same-origin bridge process.
#[tokio::test]
async fn cors_headers_appear_only_when_configured() {
    let preflight = || {
        Request::builder()
            .method("OPTIONS")
            .uri("/v1/audio/transcriptions")
            .header("origin", "http://localhost:5173")
            .header("access-control-request-method", "POST")
            .body(Body::empty())
            .unwrap()
    };

    let off = build_router(state(ServiceMode::Offline, &[]), RouterLimits::default());
    let resp = off.oneshot(preflight()).await.unwrap();
    assert!(
        resp.headers().get("access-control-allow-origin").is_none(),
        "no CORS layer unless an operator asked for one"
    );

    let on = build_router(
        state(ServiceMode::Offline, &[]),
        RouterLimits {
            cors_allow_origins: vec!["*".into()],
            ..Default::default()
        },
    );
    let resp = on.oneshot(preflight()).await.unwrap();
    assert_eq!(
        resp.headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("*")
    );
}
