// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! CLI / runtime config.

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde_json::{Map, Value};

#[derive(Debug, Parser)]
#[command(name = "oasr-server", version, about = "OASR HTTP + gRPC frontend")]
pub struct Cli {
    // ---- Worker fleet ----
    /// Number of Python worker processes to spawn.  Defaults to 1.
    #[arg(long, default_value_t = 1)]
    pub num_workers: usize,
    /// Comma-separated CUDA device list mapping worker `k` to GPU `cuda_devices[k]`.
    /// Defaults to "0,1,...,num_workers-1".
    #[arg(long)]
    pub cuda_devices: Option<String>,
    /// Python worker thread mode: 1 (default) or 2.
    #[arg(long, default_value_t = 1)]
    pub worker_threads: u32,
    /// Path to the Python interpreter and module spec.
    #[arg(long, default_value = "python")]
    pub python: String,
    /// Module spec passed after the python interpreter.
    #[arg(long, default_value = "oasr.serving")]
    pub worker_module: String,
    /// If set, the supervisor spawns ``python <worker_script>`` instead of
    /// ``python -m <worker_module>``.  Useful for tests and benchmarks that
    /// drive a stub worker (e.g. ``scripts/fake_engine_worker.py``).
    #[arg(long)]
    pub worker_script: Option<PathBuf>,
    /// Base ZMQ endpoint; `{k}` is replaced with worker index, `{pid}` with
    /// the server pid.  Default uses TCP because the pure-Rust `zeromq` crate
    /// interoperates reliably with pyzmq only over TCP at this version.
    #[arg(long, default_value = "tcp://127.0.0.1:{port}")]
    pub zmq_endpoint_base: String,
    /// Base TCP port for worker endpoints; worker `k` binds `--worker-port-base + k`.
    #[arg(long, default_value_t = 55580)]
    pub worker_port_base: u16,
    /// Spawn workers (default) or assume they're already listening on
    /// `zmq_endpoint_base` (then `num_workers` endpoints must already exist).
    #[arg(long, default_value_t = true)]
    pub spawn_workers: bool,

    // ---- Engine config (mirror EngineConfig essentials) ----
    /// Required: WeNet checkpoint directory.
    #[arg(long)]
    pub ckpt_dir: Option<PathBuf>,
    /// torch.dtype string ("float16" | "bfloat16" | "float32").
    #[arg(long, default_value = "float16")]
    pub dtype: String,
    /// Service mode — the engine runs in exactly one mode per lifecycle.
    /// "streaming" (default) accepts chunk-by-chunk requests via /v1/stream
    /// and the gRPC `StreamingRecognize`; "offline" accepts full-audio
    /// requests via /v1/transcriptions and `Recognize`.  Mismatched
    /// requests are rejected at admission.
    #[arg(long, default_value = "streaming")]
    pub service_mode: String,
    /// Optional: max batch size override.
    #[arg(long)]
    pub max_batch_size: Option<u32>,
    /// Encoder chunk size (frames).
    #[arg(long)]
    pub chunk_size: Option<u32>,
    /// Decoder type.
    #[arg(long)]
    pub decoder_type: Option<String>,
    /// Full EngineConfig JSON file; values override individual flags above.
    #[arg(long)]
    pub engine_config: Option<PathBuf>,

    // ---- Server ----
    #[arg(long, default_value = "0.0.0.0:8080")]
    pub http_bind: SocketAddr,
    #[arg(long, default_value = "0.0.0.0:50051")]
    pub grpc_bind: SocketAddr,
    #[arg(long, default_value_t = 256)]
    pub max_concurrent_requests: u32,
    #[arg(long, default_value = "info")]
    pub log_level: String,
    /// Soft timeout (s) for the READY handshake when spawning workers.
    #[arg(long, default_value_t = 120)]
    pub ready_timeout_s: u64,
}

impl Cli {
    pub fn resolved_cuda_devices(&self) -> Vec<i32> {
        if let Some(s) = self.cuda_devices.as_deref() {
            s.split(',')
                .filter(|t| !t.is_empty())
                .filter_map(|t| t.trim().parse::<i32>().ok())
                .collect()
        } else {
            (0..self.num_workers as i32).collect()
        }
    }

    /// Build the full EngineConfig JSON object that goes to each worker.
    pub fn build_engine_config_json(&self) -> Result<String> {
        let mut obj: Map<String, Value> = if let Some(p) = &self.engine_config {
            let bytes = std::fs::read(p).with_context(|| format!("read engine_config {p:?}"))?;
            let parsed: Value = serde_json::from_slice(&bytes)?;
            match parsed {
                Value::Object(m) => m,
                _ => return Err(anyhow!("engine_config JSON must be an object")),
            }
        } else {
            Map::new()
        };

        if !obj.contains_key("ckpt_dir") {
            let ck = self
                .ckpt_dir
                .as_ref()
                .ok_or_else(|| anyhow!("--ckpt-dir or engine_config.ckpt_dir is required"))?;
            obj.insert(
                "ckpt_dir".into(),
                Value::String(ck.to_string_lossy().into_owned()),
            );
        }
        if !obj.contains_key("dtype") {
            obj.insert("dtype".into(), Value::String(self.dtype.clone()));
        }
        if !obj.contains_key("service_mode") {
            obj.insert(
                "service_mode".into(),
                Value::String(self.service_mode.clone()),
            );
        }
        if let Some(v) = self.max_batch_size {
            obj.entry("max_batch_size")
                .or_insert(Value::Number(v.into()));
        }
        if let Some(v) = self.chunk_size {
            obj.entry("chunk_size").or_insert(Value::Number(v.into()));
        }
        if let Some(s) = &self.decoder_type {
            obj.entry("decoder_type")
                .or_insert(Value::String(s.clone()));
        }
        // device defaults to "cuda" — let EngineConfig fall back if absent.

        Ok(serde_json::to_string(&Value::Object(obj))?)
    }
}
