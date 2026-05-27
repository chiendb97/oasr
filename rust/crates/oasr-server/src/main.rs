// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Entry point for `oasr-server`.
//!
//! Hosts one in-process Python `ASREngine` (via PyO3) and serves it over
//! HTTP + gRPC.  Multi-GPU scaling = launch one `oasr-server` per GPU
//! behind a process manager and set `CUDA_VISIBLE_DEVICES` per launch —
//! same topology as the previous ZMQ supervisor used for its Python
//! workers, minus the IPC hop.

mod config;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use metrics_exporter_prometheus::PrometheusBuilder;
use oasr_engine_client::{
    client::EngineClientConfig, dispatcher::DispatcherConfig, EngineClient, EnginePool, PyEngine,
};
use oasr_server_grpc::pb::speech_server::SpeechServer;
use oasr_server_grpc::SpeechService;
use oasr_server_http::{build_router, ServerState};
use tokio::signal;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use crate::config::Cli;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let filter = EnvFilter::try_new(&cli.log_level)
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let prometheus = match PrometheusBuilder::new().install_recorder() {
        Ok(h) => Some(h),
        Err(e) => {
            error!("prometheus recorder install failed: {e}");
            None
        }
    };

    // ---- Build the in-process engine ----
    let engine_cfg_json = cli.build_engine_config_json().context("build engine config")?;
    info!(label = %cli.engine_label, "loading ASREngine");
    let engine = PyEngine::new(&engine_cfg_json).context("build PyEngine")?;
    info!(label = %cli.engine_label, "ASREngine loaded");

    let mut client_cfg = EngineClientConfig::new(cli.engine_label.clone());
    client_cfg.dispatcher = DispatcherConfig {
        max_concurrent_requests: cli.max_concurrent_requests,
        admit_window: Duration::from_millis(cli.admit_window_ms),
        admit_threshold: cli.admit_threshold,
        ..DispatcherConfig::default()
    };
    let client = Arc::new(EngineClient::start(engine, client_cfg));

    // Wait briefly for the dispatcher to take its first tick so /readyz
    // doesn't flap on startup.
    let _ = client.ping(Duration::from_secs(10)).await;

    let pool = Arc::new(EnginePool::new(vec![client]));

    let state = Arc::new(ServerState {
        pool: Arc::clone(&pool),
        prometheus,
    });

    // ---- HTTP server ----
    let http_router = build_router(Arc::clone(&state));
    let http_bind = cli.http_bind;
    let http_listener = tokio::net::TcpListener::bind(http_bind)
        .await
        .with_context(|| format!("bind http {http_bind}"))?;
    info!("HTTP listening on http://{http_bind}");
    let http_handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(http_listener, http_router).await {
            error!("axum serve: {e}");
        }
    });

    // ---- gRPC server ----
    let grpc_bind = cli.grpc_bind;
    let grpc_pool = Arc::clone(&pool);
    let grpc_handle = tokio::spawn(async move {
        let svc = SpeechService::new(grpc_pool);
        info!("gRPC listening on http://{grpc_bind}");
        if let Err(e) = tonic::transport::Server::builder()
            .add_service(SpeechServer::new(svc))
            .serve(grpc_bind)
            .await
        {
            error!("tonic serve: {e}");
        }
    });

    // ---- Wait for shutdown ----
    wait_for_signal().await;
    info!("shutdown signal received; draining");

    http_handle.abort();
    grpc_handle.abort();

    // Brief grace period for in-flight handlers.
    tokio::time::sleep(Duration::from_millis(500)).await;

    info!("bye");
    Ok(())
}

async fn wait_for_signal() {
    #[cfg(unix)]
    {
        use signal::unix::{signal as unix_signal, SignalKind};
        let mut sigterm = unix_signal(SignalKind::terminate()).expect("install SIGTERM");
        let mut sigint = unix_signal(SignalKind::interrupt()).expect("install SIGINT");
        tokio::select! {
            _ = sigterm.recv() => {}
            _ = sigint.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = signal::ctrl_c().await;
    }
}
