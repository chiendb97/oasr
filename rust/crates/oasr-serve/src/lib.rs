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
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use oasr_engine_client::{
    client::EngineClientConfig, dispatcher::DispatcherConfig, EngineClient, EnginePool, PyEngine,
};
use oasr_server_grpc::pb::speech_server::SpeechServer;
use oasr_server_grpc::{ServiceMode as GrpcServiceMode, SpeechService, SPEECH_SERVICE_NAME};
use oasr_server_http::{
    build_router, RouterLimits, ServerState, ServiceMode as HttpServiceMode, READY_STALE_AFTER,
};
use tokio::signal;
use tonic_health::ServingStatus;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;

/// How often the health watcher re-evaluates engine readiness.
const HEALTH_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Run the server to completion: build the engine, start the HTTP + gRPC
/// listeners, and block until a shutdown signal arrives.
///
/// Builds its own multi-threaded tokio runtime so it can be driven from a
/// synchronous entry point (`fn main` or an extension-module function).
pub fn run(cli: Cli) -> Result<()> {
    init_tracing(&cli.log_level, &cli.log_format);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;
    runtime.block_on(serve(cli))
}

/// Install the tracing subscriber.  Uses `try_init` because when imported as
/// an extension module the host interpreter may already have one — in that
/// case we keep the existing subscriber instead of panicking.
///
/// `log_format` selects the output formatter: `"json"` emits one JSON object
/// per line (with span fields such as `rid` inlined) for log aggregators; any
/// other value uses the human-readable text formatter.
fn init_tracing(log_level: &str, log_format: &str) {
    let filter = EnvFilter::try_new(log_level)
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new("info"));
    match log_format.to_ascii_lowercase().as_str() {
        "json" => {
            let _ = tracing_subscriber::fmt()
                .json()
                .with_env_filter(filter)
                .try_init();
        }
        "text" => {
            let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
        }
        other => {
            let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
            warn!(log_format = %other, "unknown --log-format; falling back to text");
        }
    }
}

async fn serve(cli: Cli) -> Result<()> {
    // `oasr_metrics::install_recorder` rather than `PrometheusBuilder::new()`
    // directly: the builder defaults `buckets: None`, and a histogram with no
    // buckets is not exported as a histogram at all — it becomes a rolling
    // summary with fixed quantiles, no `_bucket` series, and quantiles that
    // cannot be aggregated across replicas.  The bucket table lives beside the
    // metric declarations so a new histogram cannot arrive without one.
    let prometheus = match oasr_metrics::install_recorder() {
        Ok(h) => Some(h),
        Err(e) => {
            error!("prometheus recorder install failed: {e}");
            None
        }
    };

    // Validate the flag early (a typo should fail before we load a model), but
    // the *authoritative* mode comes from the engine once it is built — see below.
    let _: GrpcServiceMode = cli
        .service_mode
        .parse()
        .map_err(|e: String| anyhow::anyhow!("invalid --service-mode: {e}"))?;

    info!(
        label = %cli.engine_label,
        service_mode = %cli.service_mode,
        max_concurrent_requests = cli.max_concurrent_requests,
        http_bind = %cli.http_bind,
        grpc_bind = %cli.grpc_bind,
        "starting oasr-server"
    );

    // ---- Build the in-process engine ----
    let engine_cfg_json = cli
        .build_engine_config_json()
        .context("build engine config")?;
    info!(label = %cli.engine_label, "loading ASREngine");
    let load_t0 = Instant::now();
    let engine = PyEngine::new(&engine_cfg_json).context("build PyEngine")?;
    let model = engine.model_info();
    info!(
        label = %cli.engine_label,
        load_ms = load_t0.elapsed().as_millis() as u64,
        device = ?model.device,
        dtype = ?model.dtype,
        chunk_size = ?model.chunk_size,
        max_batch_size = ?model.max_batch_size,
        decoder_type = ?model.decoder_type,
        decode_method = ?model.decode_method,
        capabilities = ?model.capabilities,
        service_mode = ?model.service_mode,
        vocab_size = ?model.vocab_size,
        sample_rate = ?model.sample_rate,
        "ASREngine loaded"
    );

    // The rate every handler resamples client audio to.  An engine that did not
    // report one (an older/stubbed engine object) falls back to 16 kHz, which is
    // what every checkpoint in tree uses — but say so, because getting this
    // wrong is silent: the transcript comes back confident and wrong.
    let engine_sample_rate = match model.sample_rate {
        Some(sr) => sr,
        None => {
            warn!(
                label = %cli.engine_label,
                "engine did not report a sample rate; assuming 16000 Hz for client resampling"
            );
            16_000
        }
    };

    // ---- Engine-authoritative service mode ----
    //
    // The engine's mode can differ from `--service-mode`: `--engine-config` JSON
    // wins on the Python side, and several decode families are offline-only.  If
    // the front-ends trusted the flag they would reject requests this engine can
    // serve (and accept ones it cannot, which then fail deep inside admission).
    // Take the engine's answer and say so when it disagrees with the flag.
    let effective_mode = match model.service_mode.as_deref() {
        Some(m) => {
            if m != cli.service_mode {
                warn!(
                    label = %cli.engine_label,
                    flag = %cli.service_mode,
                    engine = %m,
                    decode_method = ?model.decode_method,
                    "--service-mode disagrees with the engine (engine-config JSON or an \
                     offline-only decode family); using the engine's mode"
                );
            }
            m.to_string()
        }
        None => cli.service_mode.clone(),
    };
    let grpc_mode: GrpcServiceMode = effective_mode
        .parse()
        .map_err(|e: String| anyhow::anyhow!("engine reported an unknown service_mode: {e}"))?;
    let http_mode: HttpServiceMode = effective_mode
        .parse()
        .expect("validated by GrpcServiceMode parse above");

    let mut client_cfg = EngineClientConfig::new(cli.engine_label.clone());
    client_cfg.dispatcher = DispatcherConfig {
        max_concurrent_requests: cli.max_concurrent_requests,
        admit_window: Duration::from_millis(cli.admit_window_ms),
        admit_threshold: cli.admit_threshold,
        trace_dispatch: cli.trace_dispatch,
        ..DispatcherConfig::default()
    };
    // The command channel buffers whole audio payloads, so its depth has to
    // follow the admission cap rather than sit at an unrelated constant.
    client_cfg.sync_cmd_channel_cap();
    info!(
        label = %cli.engine_label,
        max_concurrent_requests = cli.max_concurrent_requests,
        cmd_channel_cap = client_cfg.cmd_channel_cap,
        max_audio_mib = cli.max_audio_mib,
        request_timeout_secs = cli.request_timeout_secs,
        stream_idle_timeout_secs = cli.stream_idle_timeout_secs,
        max_inflight_connections = ?cli.inflight_limit(),
        shutdown_grace_secs = cli.shutdown_grace_secs,
        "serving limits"
    );
    // Worker 0 is the engine already built above (its `model_info` is what the
    // front-ends are configured from); `--engine-workers` adds the rest.  They
    // are identical by construction — same config JSON — so the router can treat
    // them as interchangeable and only the load differs.
    let mut workers: Vec<Arc<EngineClient>> =
        Vec::with_capacity(cli.engine_workers.max(1) as usize);
    workers.push(Arc::new(EngineClient::start(engine, client_cfg.clone())));
    for i in 1..cli.engine_workers.max(1) {
        let t0 = Instant::now();
        let extra = PyEngine::new(&engine_cfg_json)
            .with_context(|| format!("build PyEngine for worker {i}"))?;
        info!(
            label = %cli.engine_label,
            worker = i,
            load_ms = t0.elapsed().as_millis() as u64,
            "additional ASREngine loaded"
        );
        workers.push(Arc::new(EngineClient::start(extra, client_cfg.clone())));
    }
    if workers.len() > 1 {
        // Each worker is a full copy — weights, KV pool, graph pools — so this is
        // the line that explains the VRAM.  `--max-num-blocks` is per worker and
        // is *not* divided for you.
        info!(
            label = %cli.engine_label,
            workers = workers.len(),
            max_batch_size = ?model.max_batch_size,
            "running multiple engine workers: VRAM, weights and graph pools are              replicated per worker; --max-num-blocks applies to each"
        );
    }

    // Wait briefly for each dispatcher to take its first tick so /readyz
    // doesn't flap on startup.
    for (i, w) in workers.iter().enumerate() {
        match w.ping(Duration::from_secs(10)).await {
            Ok(_) => debug!(label = %cli.engine_label, worker = i, "dispatcher ready"),
            Err(e) => {
                warn!(label = %cli.engine_label, worker = i, "dispatcher not ready within 10s: {e}")
            }
        }
    }

    let pool = Arc::new(EnginePool::new(workers));

    // The id the OpenAI surface reports and echoes.  The checkpoint path is the
    // only stable name a single-model process has; `--served-model-name` is how
    // an operator gives it the one their clients already send.
    let model_id = model
        .ckpt_dir
        .clone()
        .unwrap_or_else(|| cli.engine_label.clone());
    let max_audio_samples = cli.max_audio_samples(engine_sample_rate);
    let state = Arc::new(ServerState {
        pool: Arc::clone(&pool),
        prometheus,
        service_mode: http_mode,
        sample_rate: engine_sample_rate,
        max_body_bytes: cli.max_audio_bytes(),
        max_audio_samples,
        served_model_names: cli.served_model_name.clone(),
        model_id,
    });

    // One broadcast of the shutdown signal to both listeners; each stops
    // accepting and lets its in-flight requests finish.
    let (shutdown_tx, mut http_shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);
    let mut grpc_shutdown_rx = shutdown_tx.subscribe();

    // ---- HTTP server ----
    let http_router = build_router(
        Arc::clone(&state),
        RouterLimits {
            request_timeout: cli.request_timeout(),
            max_inflight: cli.inflight_limit(),
            cors_allow_origins: cli.cors_allow_origin.clone(),
        },
    );
    let http_bind = cli.http_bind;
    let http_listener = tokio::net::TcpListener::bind(http_bind)
        .await
        .with_context(|| format!("bind http {http_bind}"))?;
    info!("HTTP listening on http://{http_bind}");
    let http_handle = tokio::spawn(async move {
        let serve = axum::serve(http_listener, http_router).with_graceful_shutdown(async move {
            let _ = http_shutdown_rx.recv().await;
        });
        if let Err(e) = serve.await {
            error!("axum serve: {e}");
        }
        debug!("HTTP listener drained");
    });

    // ---- gRPC server (Speech + standard Health) ----
    let grpc_bind = cli.grpc_bind;
    let grpc_pool = Arc::clone(&pool);
    let grpc_idle = cli.stream_idle_timeout();
    let grpc_max_message = cli.max_audio_bytes();
    let grpc_conn_limit = cli.inflight_limit();
    let grpc_max_audio_samples = max_audio_samples;

    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    // Start NOT_SERVING and let the watcher below promote us: the reporter has
    // to track the *engine*, not the process.  It used to be set once at
    // startup and once at shutdown, so after the dispatcher gave up on a wedged
    // engine — at which point `/readyz` correctly 503s — `grpc.health.v1
    // Health/Check` still answered SERVING forever, and a deployment following
    // this doc's `grpc-health-probe` advice would never drain the pod.
    health_reporter
        .set_service_status(SPEECH_SERVICE_NAME, ServingStatus::NotServing)
        .await;
    let health_pool = Arc::clone(&pool);
    let mut health_shutdown_rx = shutdown_tx.subscribe();
    let health_handle = tokio::spawn(async move {
        let mut reporter = health_reporter;
        let mut serving: Option<bool> = None;
        loop {
            // Same signal `/readyz` reads, on the same staleness bound, so the
            // two probes cannot disagree about this process.
            let ready = health_pool.any_ready(READY_STALE_AFTER);
            if serving != Some(ready) {
                let status = if ready {
                    ServingStatus::Serving
                } else {
                    ServingStatus::NotServing
                };
                if serving.is_some() {
                    warn!(ready, "engine readiness changed; updating gRPC health");
                }
                reporter
                    .set_service_status(SPEECH_SERVICE_NAME, status)
                    .await;
                // The empty service name is the overall-process health that
                // Kubernetes and gRPC load balancers probe by default.
                reporter.set_service_status("", status).await;
                serving = Some(ready);
            }
            tokio::select! {
                _ = tokio::time::sleep(HEALTH_POLL_INTERVAL) => {}
                _ = health_shutdown_rx.recv() => {
                    reporter
                        .set_service_status(SPEECH_SERVICE_NAME, ServingStatus::NotServing)
                        .await;
                    reporter.set_service_status("", ServingStatus::NotServing).await;
                    return;
                }
            }
        }
    });

    let grpc_handle = tokio::spawn(async move {
        let svc = SpeechService::new(grpc_pool, grpc_mode, engine_sample_rate)
            .with_stream_idle_timeout(grpc_idle)
            .with_max_audio_samples(grpc_max_audio_samples);
        info!("gRPC listening on {grpc_bind}");
        // Both message-size limits are set explicitly.  tonic's default
        // decoding cap is 4 MiB, which silently rejected any unary `Recognize`
        // carrying more than ~2 minutes of 16 kHz PCM while HTTP accepted
        // 256 MiB of the same audio — a 64x asymmetry on the surface the docs
        // recommend for offline throughput.
        let speech = SpeechServer::new(svc)
            .max_decoding_message_size(grpc_max_message)
            .max_encoding_message_size(grpc_max_message);
        let mut builder = tonic::transport::Server::builder();
        if let Some(n) = grpc_conn_limit {
            builder = builder.concurrency_limit_per_connection(n);
        }
        let serve = builder
            .add_service(speech)
            .add_service(health_service)
            .serve_with_shutdown(grpc_bind, async move {
                let _ = grpc_shutdown_rx.recv().await;
            });
        if let Err(e) = serve.await {
            error!("tonic serve: {e}");
        }
        debug!("gRPC listener drained");
    });

    // ---- Wait for shutdown ----
    wait_for_signal().await;
    let grace = Duration::from_secs(cli.shutdown_grace_secs);
    info!(
        grace_secs = cli.shutdown_grace_secs,
        "shutdown signal received; draining"
    );

    // Tell every listener to stop accepting.  Both then finish their in-flight
    // requests; the health watcher flips to NOT_SERVING on the same signal, so
    // a load balancer sees the transition while the drain is still running.
    let _ = shutdown_tx.send(());

    // Wait for the drain, bounded.  The previous code called `abort()` on both
    // tasks — hard-cancelling every in-flight request — and *then* slept 500 ms,
    // which helped nothing because there was no longer anything to wait for.
    let drained = tokio::time::timeout(grace, async {
        let _ = tokio::join!(http_handle, grpc_handle, health_handle);
    })
    .await;
    match drained {
        Ok(()) => info!("listeners drained cleanly"),
        Err(_) => warn!(
            grace_secs = cli.shutdown_grace_secs,
            "drain deadline elapsed with requests still in flight; exiting anyway"
        ),
    }

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
