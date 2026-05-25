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

/// Build the axum Router with all native and Whisper-compat routes.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Native
        .route("/v1/transcriptions", post(crate::offline::handle_offline))
        .route("/v1/stream", get(crate::ws::handle_ws))
        .route("/v1/models", get(handle_models))
        // OpenAI Whisper compatibility
        .route(
            "/v1/audio/transcriptions",
            post(crate::whisper::handle_whisper),
        )
        // Operability
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
