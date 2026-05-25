// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Worker process supervisor: spawn the N Python engine workers, wait for
//! READY, and connect [`EngineClient`]s to them.

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use oasr_engine_client::client::EngineClientConfig;
use oasr_engine_client::EngineClient;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::timeout;
use tracing::{error, info};

use crate::config::Cli;

pub struct WorkerProcess {
    pub index: usize,
    #[allow(dead_code)] // retained for diagnostic logging
    pub endpoint: String,
    pub child: Child,
}

/// Spawn `cli.num_workers` Python workers and connect a [`EngineClient`] to
/// each.  Returns (children, clients).
pub async fn spawn_workers(cli: &Cli) -> Result<(Vec<WorkerProcess>, Vec<Arc<EngineClient>>)> {
    let cudas = cli.resolved_cuda_devices();
    if cudas.len() < cli.num_workers {
        return Err(anyhow!(
            "expected at least {} cuda devices, got {}",
            cli.num_workers,
            cudas.len()
        ));
    }

    let pid = std::process::id();
    let engine_cfg_json = cli.build_engine_config_json()?;

    let mut children: Vec<WorkerProcess> = Vec::new();
    let mut clients: Vec<Arc<EngineClient>> = Vec::new();

    for k in 0..cli.num_workers {
        let port = cli.worker_port_base + (k as u16);
        let endpoint = cli
            .zmq_endpoint_base
            .replace("{pid}", &pid.to_string())
            .replace("{k}", &k.to_string())
            .replace("{port}", &port.to_string());

        let label = format!("w{k}");

        if cli.spawn_workers {
            let mut cmd = Command::new(&cli.python);
            if let Some(script) = &cli.worker_script {
                cmd.arg(script);
            } else {
                cmd.arg("-m").arg(&cli.worker_module);
            }
            cmd.arg("--zmq-endpoint")
                .arg(&endpoint)
                .arg("--engine-config-json")
                .arg(&engine_cfg_json)
                .arg("--worker-threads")
                .arg(cli.worker_threads.to_string())
                .arg("--max-concurrent-requests")
                .arg(cli.max_concurrent_requests.to_string())
                .arg("--log-level")
                .arg(&cli.log_level);
            // Pin to a specific GPU.
            cmd.env("CUDA_VISIBLE_DEVICES", cudas[k].to_string());
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::inherit());

            let mut child = cmd
                .spawn()
                .with_context(|| format!("spawn worker {k}: {} -m {}", cli.python, cli.worker_module))?;

            // Wait for READY on stdout, with a soft timeout.
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| anyhow!("worker stdout is not captured"))?;
            let mut lines = BufReader::new(stdout).lines();
            let ready = timeout(Duration::from_secs(cli.ready_timeout_s), async {
                while let Some(line) = lines.next_line().await? {
                    info!(worker = %label, "[stdout] {line}");
                    if line.trim() == "READY" {
                        return Ok::<bool, std::io::Error>(true);
                    }
                }
                Ok(false)
            })
            .await
            .map_err(|_| anyhow!("worker {k} READY timeout after {}s", cli.ready_timeout_s))?
            .context("read worker stdout")?;
            if !ready {
                return Err(anyhow!("worker {k} exited before sending READY"));
            }

            children.push(WorkerProcess {
                index: k,
                endpoint: endpoint.clone(),
                child,
            });
        } else {
            info!(worker = %label, endpoint = %endpoint, "assuming external worker");
        }

        let client_cfg = EngineClientConfig::new(endpoint.clone(), label.clone());
        let client = Arc::new(EngineClient::connect(client_cfg).await?);
        // One Ping to verify connectivity and capture model_info for `/v1/models`.
        match client.ping(Duration::from_secs(10)).await {
            Ok(_) => info!(worker = %label, "ping OK"),
            Err(e) => {
                error!(worker = %label, "initial ping failed: {e}");
            }
        }
        clients.push(client);
    }

    Ok((children, clients))
}

pub async fn shutdown_workers(mut procs: Vec<WorkerProcess>) {
    for p in &mut procs {
        if let Err(e) = p.child.start_kill() {
            error!(worker = p.index, "kill: {e}");
        }
    }
    for mut p in procs {
        let _ = p.child.wait().await;
    }
}
