// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! OASR serving core.
//!
//! Builds one in-process Python `ASREngine` (via PyO3) and serves it over
//! HTTP + gRPC.  This logic is shared by two front-ends:
//!
//!   * the `oasr-server` binary, which embeds Python (`pyo3/auto-initialize`)
//!     and calls [`run`] from `fn main`; and
//!   * the `oasr._core` extension module, which is imported into a running
//!     interpreter (`pyo3/extension-module`) and calls [`run`] under
//!     `Python::allow_threads`.
//!
//! Multi-GPU scaling = launch one front-end per GPU behind a process manager
//! and set `CUDA_VISIBLE_DEVICES` per launch.

mod config;

pub use config::Cli;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use metrics_exporter_prometheus::PrometheusBuilder;
use oasr_engine_client::{
    client::EngineClientConfig, dispatcher::DispatcherConfig, EngineClient, EnginePool, PyEngine,
};
use oasr_server_grpc::pb::speech_server::SpeechServer;
use oasr_server_grpc::{ServiceMode as GrpcServiceMode, SpeechService, SPEECH_SERVICE_NAME};
use oasr_server_http::{build_router, ServerState, ServiceMode as HttpServiceMode};
use tokio::signal;
use tonic_health::ServingStatus;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

/// Run the server to completion: build the engine, start the HTTP + gRPC
/// listeners, and block until a shutdown signal arrives.
///
/// Builds its own multi-threaded tokio runtime so it can be driven from a
/// synchronous entry point (`fn main` or an extension-module function).
pub fn run(cli: Cli) -> Result<()> {
    init_tracing(&cli.log_level);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;
    runtime.block_on(serve(cli))
}

/// Install the tracing subscriber.  Uses `try_init` because when imported as
/// an extension module the host interpreter may already have one — in that
/// case we keep the existing subscriber instead of panicking.
fn init_tracing(log_level: &str) {
    let filter = EnvFilter::try_new(log_level)
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

async fn serve(cli: Cli) -> Result<()> {
    let prometheus = match PrometheusBuilder::new().install_recorder() {
        Ok(h) => Some(h),
        Err(e) => {
            error!("prometheus recorder install failed: {e}");
            None
        }
    };

    let grpc_mode: GrpcServiceMode = cli
        .service_mode
        .parse()
        .map_err(|e: String| anyhow::anyhow!("invalid --service-mode: {e}"))?;
    let http_mode: HttpServiceMode = cli
        .service_mode
        .parse()
        .expect("validated by GrpcServiceMode parse above");

    // ---- Build the in-process engine ----
    let engine_cfg_json = cli
        .build_engine_config_json()
        .context("build engine config")?;
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
        service_mode: http_mode,
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

    // ---- gRPC server (Speech + standard Health) ----
    let grpc_bind = cli.grpc_bind;
    let grpc_pool = Arc::clone(&pool);

    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    // Mark the Speech service as serving from the start: the engine
    // dispatcher has already completed its first tick above and the pool is
    // accepting work.  An empty service name flips the overall process
    // health, which Kubernetes / gRPC LBs probe by default.
    health_reporter
        .set_service_status(SPEECH_SERVICE_NAME, ServingStatus::Serving)
        .await;
    health_reporter
        .set_service_status("", ServingStatus::Serving)
        .await;

    let grpc_handle = tokio::spawn(async move {
        let svc = SpeechService::new(grpc_pool, grpc_mode);
        info!("gRPC listening on {grpc_bind}");
        if let Err(e) = tonic::transport::Server::builder()
            .add_service(SpeechServer::new(svc))
            .add_service(health_service)
            .serve(grpc_bind)
            .await
        {
            error!("tonic serve: {e}");
        }
    });

    // ---- Wait for shutdown ----
    wait_for_signal().await;
    info!("shutdown signal received; draining");

    // Flip health to NOT_SERVING so probes in-flight see the transition.
    health_reporter
        .set_service_status(SPEECH_SERVICE_NAME, ServingStatus::NotServing)
        .await;
    health_reporter
        .set_service_status("", ServingStatus::NotServing)
        .await;

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
