// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! axum routes for the OASR HTTP API.

pub mod offline;
pub mod router;
pub mod whisper;
pub mod ws;

pub use router::{build_router, AppState};

use std::sync::Arc;

use metrics_exporter_prometheus::PrometheusHandle;
use oasr_engine_client::EnginePool;

/// Shared state injected into every handler.
pub struct ServerState {
    pub pool: Arc<EnginePool>,
    pub prometheus: Option<PrometheusHandle>,
}
