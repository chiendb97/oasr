// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! axum routes for the OASR HTTP API (Google STT v1-shaped surface).

pub mod recognize;
pub mod router;

pub use router::{build_router, AppState, ServiceMode};

use std::sync::Arc;

use metrics_exporter_prometheus::PrometheusHandle;
use oasr_engine_client::EnginePool;

/// Shared state injected into every handler.
pub struct ServerState {
    pub pool: Arc<EnginePool>,
    pub prometheus: Option<PrometheusHandle>,
    pub service_mode: ServiceMode,
}
