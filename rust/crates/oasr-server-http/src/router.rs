// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! axum Router assembly.

use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde_json::json;
use tower::limit::GlobalConcurrencyLimitLayer;
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

/// Bounds applied to the recognition route.  Both are optional so a test (or a
/// deployment that terminates policy at a proxy) can turn them off.
#[derive(Clone, Copy, Debug, Default)]
pub struct RouterLimits {
    /// Deadline for one request, end to end.
    pub request_timeout: Option<Duration>,
    /// Max requests processed concurrently; the rest queue.
    pub max_inflight: Option<usize>,
}

/// Build the axum Router for the Google STT v1-shaped HTTP API.
///
/// REST is synchronous-only: streaming clients must use the gRPC
/// `StreamingRecognize` RPC.
pub fn build_router(state: AppState, limits: RouterLimits) -> Router {
    // The bounds go on the recognition route only.  `/healthz`, `/readyz` and
    // `/metrics` are exactly what an operator needs answered *while* the
    // service is saturated, so putting them behind the same concurrency limit
    // would make a busy server look like a dead one to its own probes.
    let mut recognize: Router<AppState> = Router::new().route(
        "/v1/speech:recognize",
        post(crate::recognize::handle_recognize),
    );
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

    Router::new()
        // Speech-to-Text v1 (Google STT v1-shaped surface).
        // Raw PCM body, config in the query string (no base64/JSON).
        .merge(recognize)
        // Models.
        .route("/v1/models", get(handle_models))
        // Operability.
        .route("/healthz", get(handle_health))
        .route("/readyz", get(handle_ready))
        .route("/metrics", get(handle_metrics))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
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

async fn handle_models(State(s): State<AppState>) -> impl IntoResponse {
    let mi = s.pool.model_info();
    Json(json!({
        "data": [{
            "id": mi.as_ref().and_then(|m| m.ckpt_dir.clone()).unwrap_or_else(|| "oasr".into()),
            "object": "model",
            "owned_by": "oasr",
            "info": mi,
        }]
    }))
}
