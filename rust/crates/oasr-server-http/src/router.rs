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

/// Build the axum Router for the Google STT v1-shaped HTTP API.
///
/// REST is synchronous-only: streaming clients must use the gRPC
/// `StreamingRecognize` RPC.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Speech-to-Text v1 (Google STT v1-shaped surface).
        .route(
            "/v1/speech:recognize",
            post(crate::recognize::handle_recognize),
        )
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

async fn handle_ready(State(s): State<AppState>) -> impl IntoResponse {
    if s.pool.any_ready(Duration::from_secs(5)) {
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
