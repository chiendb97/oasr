// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! `--gpus`: several GPUs behind one server's ports.
//!
//! One engine process saturates one GPU, and two engines in one process share
//! one GIL, so the topology that scales is one `oasr-server` per GPU -- two
//! servers on two GPUs measured 2.02x one at saturation.  It used to be the
//! operator's to assemble: a process per device, a port each, a load balancer
//! in front.  With `--gpus` this process does it:
//!
//! * it starts one replica of itself per GPU (`--processes-per-gpu` for
//!   more), with `CUDA_VISIBLE_DEVICES` set and every other flag passed
//!   through, bound to loopback ports it reports back ([`supervisor`]);
//! * it serves the configured ports itself and forwards each request, gRPC
//!   stream and realtime session to the least-loaded ready replica
//!   ([`router`]);
//! * it restarts a replica that exits or wedges, merges their metrics, and on
//!   SIGTERM drains the router and the replicas together.
//!
//! The launcher never builds an engine and never touches CUDA.

mod exposition;
mod plan;
mod replica;
mod router;
mod supervisor;
#[cfg(test)]
mod tests;

pub use replica::BoundAddrs;

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use hyper_util::rt::TokioExecutor;
use hyper_util::server::conn::auto;
use tokio::sync::{mpsc, watch};
use tonic_health::ServingStatus;
use tracing::{info, warn};

use crate::{Cli, SelfCommand};
use replica::Replica;
use router::{GrpcConnection, Router};
use supervisor::{FirstStarts, LaunchContext, LogSinks};

pub async fn run(cli: Cli, argv: Vec<OsString>, relaunch: SelfCommand) -> Result<()> {
    if !cfg!(unix) {
        anyhow::bail!("--gpus starts replica processes, which needs a Unix host");
    }
    // Installed before any replica resolves its metric handles.
    let prometheus = match oasr_metrics::install_recorder() {
        Ok(h) => Some(h),
        Err(e) => {
            warn!("prometheus recorder install failed: {e}");
            None
        }
    };
    if let Some(handle) = prometheus.clone() {
        crate::spawn_metrics_upkeep(handle);
    }

    let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    let specs = plan::plan_replicas(
        cli.gpus.as_deref(),
        cli.processes_per_gpu,
        visible.as_deref(),
        plan::count_gpus_with_nvidia_smi,
    )?;

    // Bound before any replica starts: a port in use fails in a second, not
    // after every replica has loaded a model.
    let http_listener = crate::bind_listener(cli.http_bind, cli.listen_backlog)
        .with_context(|| format!("bind http {}", cli.http_bind))?;
    let grpc_listener = crate::bind_listener(cli.grpc_bind, cli.listen_backlog)
        .with_context(|| format!("bind grpc {}", cli.grpc_bind))?;
    let http_addr = http_listener.local_addr()?;
    let grpc_addr = grpc_listener.local_addr()?;

    let run_dir = RunDir::create()?;
    let replicas: Vec<Arc<Replica>> = specs.into_iter().map(Replica::new).collect();
    let client = router::proxy_client();
    let router = Arc::new(Router::new(replicas.clone(), client.clone(), prometheus));

    // A replica drains for `--shutdown-grace-secs`, stops its engines, and
    // gives the cut sessions a moment on either side.
    let stop_timeout = Duration::from_secs(cli.shutdown_grace_secs)
        + crate::ENGINE_STOP_TIMEOUT
        + 2 * crate::CUT_GRACE
        + Duration::from_secs(5);
    let (fatal_tx, mut fatal_rx) = mpsc::unbounded_channel();
    let ctx = Arc::new(LaunchContext::new(
        relaunch.program,
        relaunch.prefix_args,
        argv,
        run_dir.0.clone(),
        client,
        LogSinks::start(cli.log_format.eq_ignore_ascii_case("json")),
        stop_timeout,
        FirstStarts::new(replicas.len(), fatal_tx),
    ));
    let (stop_tx, stop_rx) = watch::channel(false);

    info!(
        replicas = replicas.len(),
        devices = ?replicas.iter().map(|r| r.spec.device.as_str()).collect::<Vec<_>>(),
        http = %http_addr,
        grpc = %grpc_addr,
        "launching replicas; every flag but the binds applies to each replica"
    );
    let supervisors: Vec<_> = replicas
        .iter()
        .map(|r| {
            tokio::spawn(supervisor::supervise(
                Arc::clone(r),
                Arc::clone(&ctx),
                stop_rx.clone(),
            ))
        })
        .collect();

    // ---- the two ports ----
    let http_task = {
        let router = Arc::clone(&router);
        tokio::spawn(router::serve_listener(
            http_listener,
            stop_rx.clone(),
            auto::Builder::new(TokioExecutor::new()),
            move || {
                let router = Arc::clone(&router);
                move |req| Arc::clone(&router).serve_http(req)
            },
        ))
    };
    let (mut reporter, health_server) = tonic_health::server::health_reporter();
    for service in [oasr_server_grpc::SPEECH_SERVICE_NAME, ""] {
        reporter
            .set_service_status(service, ServingStatus::NotServing)
            .await;
    }
    let health = router::health_service(health_server);
    let grpc_task = {
        let router = Arc::clone(&router);
        let mut builder = auto::Builder::new(TokioExecutor::new());
        // What tonic sets on the replicas: no cap on concurrent streams per
        // connection.  hyper's own default is 200, and a client that ran more
        // streams than that on one connection would have queued here only.
        builder.http2().max_concurrent_streams(None);
        tokio::spawn(router::serve_listener(
            grpc_listener,
            stop_rx.clone(),
            builder.http2_only(),
            move || {
                let conn = GrpcConnection::new(Arc::clone(&router), health.clone());
                move |req| Arc::clone(&conn).serve(req)
            },
        ))
    };
    let health_task = tokio::spawn(report_health(
        Arc::clone(&router),
        reporter,
        stop_rx.clone(),
    ));
    info!("HTTP listening on http://{http_addr}; gRPC on {grpc_addr}");

    // ---- until a signal, or until no replica could start at all ----
    let outcome = tokio::select! {
        _ = crate::wait_for_signal() => {
            info!(grace_secs = cli.shutdown_grace_secs, "shutdown signal received; draining");
            Ok(())
        }
        Some(e) = fatal_rx.recv() => Err(e),
    };

    // Not ready from here on; stop accepting, and let what is open finish
    // while the replicas drain theirs.
    router.drain();
    let _ = stop_tx.send(true);
    let drained = tokio::time::timeout(stop_timeout + Duration::from_secs(5), async {
        futures::future::join_all(supervisors).await;
        let _ = tokio::join!(http_task, grpc_task, health_task);
        router.tunnels.wait_closed().await;
    })
    .await
    .is_ok();
    if !drained {
        warn!(
            open_sessions = router.tunnels.open(),
            "the drain did not finish in time; exiting with work in flight"
        );
    }
    drop(run_dir);
    info!("bye");
    outcome
}

/// Resolves once `stop` is set, or its sender is gone.  The guard `wait_for`
/// returns is dropped in here: left as a `select!` arm's output across an
/// await, it makes the task `!Send`.
async fn stopped(stop: &mut watch::Receiver<bool>) {
    let _ = stop.wait_for(|s| *s).await;
}

/// Keep `grpc.health.v1.Health` in step with the router: SERVING while any
/// replica is ready, NOT_SERVING from the shutdown signal on.
async fn report_health(
    router: Arc<Router>,
    mut reporter: tonic_health::server::HealthReporter,
    mut stop: watch::Receiver<bool>,
) {
    let mut serving: Option<bool> = None;
    loop {
        let ready = router.is_ready();
        if serving != Some(ready) {
            let status = if ready {
                ServingStatus::Serving
            } else {
                ServingStatus::NotServing
            };
            reporter
                .set_service_status(oasr_server_grpc::SPEECH_SERVICE_NAME, status)
                .await;
            reporter.set_service_status("", status).await;
            serving = Some(ready);
        }
        tokio::select! {
            _ = tokio::time::sleep(crate::HEALTH_POLL_INTERVAL) => {}
            _ = stopped(&mut stop) => {
                for service in [oasr_server_grpc::SPEECH_SERVICE_NAME, ""] {
                    reporter.set_service_status(service, ServingStatus::NotServing).await;
                }
                return;
            }
        }
    }
}

/// The launcher's private directory for the replicas' reports, removed on
/// exit.
struct RunDir(PathBuf);

impl RunDir {
    fn create() -> Result<Self> {
        Self::create_in(std::env::temp_dir().join(format!("oasr-launcher-{}", std::process::id())))
    }

    fn create_in(dir: PathBuf) -> Result<Self> {
        // A directory a previous launcher with this pid left behind holds
        // nothing of ours.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create the launcher's directory {}", dir.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
        }
        Ok(Self(dir))
    }
}

impl Drop for RunDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
