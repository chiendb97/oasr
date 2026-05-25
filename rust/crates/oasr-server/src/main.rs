// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Entry point for `oasr-server`.

mod config;
mod worker_supervisor;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use metrics_exporter_prometheus::PrometheusBuilder;
use oasr_engine_client::EnginePool;
use oasr_server_grpc::pb::speech_server::SpeechServer;
use oasr_server_grpc::SpeechService;
use oasr_server_http::{build_router, ServerState};
use tokio::signal;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use crate::config::Cli;
use crate::worker_supervisor::{shutdown_workers, spawn_workers};

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

    let (procs, clients) = spawn_workers(&cli).await.context("spawn workers")?;
    let pool = Arc::new(EnginePool::new(clients));

    let state = Arc::new(ServerState {
        pool: Arc::clone(&pool),
        prometheus,
    });

    // --- HTTP server ---
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

    // --- gRPC server ---
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

    // --- Wait for shutdown ---
    wait_for_signal().await;
    info!("shutdown signal received; draining");

    http_handle.abort();
    grpc_handle.abort();

    // Give in-flight requests a moment to settle.
    tokio::time::sleep(Duration::from_millis(500)).await;
    shutdown_workers(procs).await;

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
