// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! axum Router assembly.

use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::{HeaderValue, Method, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde_json::json;
use tower::limit::GlobalConcurrencyLimitLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::ServerState;

pub type AppState = Arc<ServerState>;

/// Service mode the engine was launched with.  Mirrors the gRPC
/// [`oasr_server_grpc::ServiceMode`] enum but kept here so this crate
/// doesn't depend on `oasr-server-grpc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceMode {
    /// Engine only accepts streaming requests.
    Streaming,
    /// Engine only accepts full-audio (unary) requests.
    Offline,
}

impl std::str::FromStr for ServiceMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "streaming" => Ok(Self::Streaming),
            "offline" => Ok(Self::Offline),
            other => Err(format!(
                "unknown service mode {other:?}: expected 'streaming' or 'offline'"
            )),
        }
    }
}

/// Bounds applied to the recognition routes.  All optional so a test (or a
/// deployment that terminates policy at a proxy) can turn them off.
#[derive(Clone, Debug, Default)]
pub struct RouterLimits {
    /// Deadline for one request, end to end.
    pub request_timeout: Option<Duration>,
    /// Max requests processed concurrently; the rest queue.
    pub max_inflight: Option<usize>,
    /// Origins allowed to call the browser-reachable routes.  Empty installs no
    /// CORS layer at all (the default, and the right one behind a proxy that
    /// owns the policy); `["*"]` allows any.
    pub cors_allow_origins: Vec<String>,
}

/// Build the axum Router.
///
/// Two request shapes, one engine: the Google-STT-shaped
/// `POST /v1/speech:recognize` (raw body, query-string config — the lowest
/// overhead) and the OpenAI-shaped `/v1/audio/*` uploads plus the
/// `GET /v1/realtime` WebSocket.  Adding the second set did not replace the
/// first: existing clients keep working, and both are thin adapters over the
/// same pool handles.
pub fn build_router(state: AppState, limits: RouterLimits) -> Router {
    // The bounds go on the recognition routes only.  `/healthz`, `/readyz` and
    // `/metrics` are exactly what an operator needs answered *while* the
    // service is saturated, so putting them behind the same concurrency limit
    // would make a busy server look like a dead one to its own probes.
    let body_limit = state.max_body_bytes;
    let mut recognize: Router<AppState> = Router::new()
        .route(
            "/v1/speech:recognize",
            post(crate::recognize::handle_recognize),
        )
        .route(
            "/v1/audio/transcriptions",
            post(crate::openai::handle_transcriptions),
        )
        .route(
            "/v1/audio/translations",
            post(crate::openai::handle_translations),
        )
        // The multipart routes buffer their parts, so the body cap has to be
        // stated here too: axum's own default is 2 MiB, an order of magnitude
        // below `--max-audio-mib`, and it would reject uploads the raw-body
        // route on the same server accepts.
        .layer(DefaultBodyLimit::max(body_limit));
    if let Some(n) = limits.max_inflight {
        // Bounds how many multi-MiB bodies can be resident at once; excess
        // requests wait for a permit rather than each parking a buffer.
        // *Global*, not per-layer-instance: axum clones the service per
        // connection, and the plain `ConcurrencyLimitLayer` would hand each
        // clone its own budget — a limit that scales with connection count is
        // not a limit.
        recognize = recognize.layer(GlobalConcurrencyLimitLayer::new(n));
    }
    if let Some(d) = limits.request_timeout {
        // Applied outside the concurrency limit so the deadline covers time
        // spent *queued* for a permit, not just time being served.
        recognize = recognize.layer(TimeoutLayer::new(d));
    }

    let mut router = Router::new()
        .merge(recognize)
        // Streaming transcription for clients that cannot speak gRPC — every
        // browser, and most scripting clients.
        .route("/v1/realtime", get(crate::realtime::handle_realtime))
        // Models.
        .route("/v1/models", get(handle_models))
        // Operability.
        .route("/healthz", get(handle_health))
        .route("/readyz", get(handle_ready))
        .route("/metrics", get(handle_metrics))
        .with_state(state);
    if let Some(cors) = cors_layer(&limits.cors_allow_origins) {
        router = router.layer(cors);
    }
    // Transport metrics wrap everything, including `/healthz` and the
    // 404 path: a probe that never reaches a handler is exactly the traffic
    // an operator needs counted.
    router
        .layer(axum::middleware::from_fn(crate::http_metrics::track_http))
        .layer(TraceLayer::new_for_http())
}

/// Build the CORS layer, or `None` when no origins were configured.
///
/// A browser cannot call this API cross-origin without one, which is why
/// `examples/web` needed a same-origin bridge process. It stays **off by
/// default**: a public inference endpoint that any page may call is a decision
/// an operator makes, not one a default makes for them.
fn cors_layer(origins: &[String]) -> Option<CorsLayer> {
    if origins.is_empty() {
        return None;
    }
    let base = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);
    Some(if origins.iter().any(|o| o == "*") {
        base.allow_origin(Any)
    } else {
        let parsed: Vec<HeaderValue> = origins
            .iter()
            .filter_map(|o| match HeaderValue::from_str(o) {
                Ok(v) => Some(v),
                Err(_) => {
                    tracing::warn!(origin = %o, "ignoring un-parseable CORS origin");
                    None
                }
            })
            .collect();
        base.allow_origin(parsed)
    })
}

async fn handle_health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// A heartbeat older than this means the engine is not serving.  The gRPC
/// health watcher polls the same signal on the same bound, so `/readyz` and
/// `grpc.health.v1.Health/Check` cannot disagree about this process.
pub const READY_STALE_AFTER: Duration = Duration::from_secs(5);

async fn handle_ready(State(s): State<AppState>) -> impl IntoResponse {
    if s.pool.any_ready(READY_STALE_AFTER) {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not ready")
    }
}

async fn handle_metrics(State(s): State<AppState>) -> impl IntoResponse {
    if let Some(h) = &s.prometheus {
        (StatusCode::OK, h.render())
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            String::from("# metrics exporter not initialised\n"),
        )
    }
}

/// `GET /v1/models` — OpenAI's list shape, with OASR's engine detail attached.
///
/// `object: "list"` and the `object`/`owned_by` fields on each row are what an
/// OpenAI client expects; `info` is the OASR extension carrying what the engine
/// actually loaded. One process serves one model, so `data` names it plus any
/// aliases `--served-model-name` added.
async fn handle_models(State(s): State<AppState>) -> impl IntoResponse {
    let mi = s.pool.model_info();
    let mut ids: Vec<String> = vec![s.model_id.clone()];
    for alias in &s.served_model_names {
        if !ids.contains(alias) {
            ids.push(alias.clone());
        }
    }
    let data: Vec<_> = ids
        .into_iter()
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "owned_by": "oasr",
                "info": mi,
            })
        })
        .collect();
    Json(json!({"object": "list", "data": data}))
}
