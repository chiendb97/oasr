// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! CLI / runtime config.
//!
//! After the move to PyO3 in-process engines (one engine per process, one
//! process per GPU), the supervisor + subprocess-spawning options went away
//! and the binary now reads engine config like `engine_worker.py` did:
//! optional JSON file + flag overrides.

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde_json::{Map, Value};

#[derive(Debug, Parser)]
#[command(name = "oasr-server", version, about = "OASR HTTP + gRPC frontend with in-process Python engine")]
pub struct Cli {
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
    /// Preferred batch sizes (comma-separated, e.g. `1,4,16,32,64`).  Drives
    /// the encoder CUDA-Graph pre-warm so the first request at each B value
    /// replays a captured graph instead of triggering capture mid-traffic.
    /// Values must be <= max_batch_size; the engine dedupes/sorts internally.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    pub preferred_batch_sizes: Option<Vec<u32>>,
    /// Offline scheduling policy.  ``"bucket"`` (engine default) groups by
    /// audio length using ``max_offline_pad_ratio`` as the safety cap;
    /// ``"fcfs"`` is strict FIFO with no bucketing (bigger batches under
    /// HTTP-trickle admission but more padded compute waste); ``"sjf"`` is
    /// shortest-job-first.
    #[arg(long)]
    pub schedule_policy: Option<String>,
    /// Padded-waste ratio cap for the bucket policy: a candidate is rejected
    /// if adding it would push ``(max_len * batch_size) / sum_len`` above
    /// this value.  Engine default is 4.0; raise to 8-16 for service
    /// workloads where the per-batch padding cost is much smaller than the
    /// per-batch dispatch overhead — directly grows per-step batches from
    /// 10-20 to 30-60 on mixed-length traffic.
    #[arg(long)]
    pub max_offline_pad_ratio: Option<f64>,
    /// Full EngineConfig JSON file; values override individual flags above.
    #[arg(long)]
    pub engine_config: Option<PathBuf>,
    /// Display label for tracing / logs (defaults to "engine").
    #[arg(long, default_value = "engine")]
    pub engine_label: String,

    // ---- Server ----
    #[arg(long, default_value = "0.0.0.0:8080")]
    pub http_bind: SocketAddr,
    #[arg(long, default_value = "0.0.0.0:50051")]
    pub grpc_bind: SocketAddr,
    #[arg(long, default_value_t = 256)]
    pub max_concurrent_requests: u32,
    /// Dispatcher admission coalescing window in milliseconds.  After the
    /// first envelope arrives in a tick, wait up to this long for siblings
    /// to land before stepping.  ``0`` disables (step ASAP).  Default 3 ms
    /// — empirically grows per-step batches from 10-20 to 32-64 under
    /// `asyncio.gather` HTTP bursts without a measurable p50 hit.
    #[arg(long, default_value_t = 3)]
    pub admit_window_ms: u64,
    /// Coalescing target — stop waiting early once this many envelopes
    /// have been drained.  Default 64 (matches the typical max_batch_size).
    #[arg(long, default_value_t = 64)]
    pub admit_threshold: usize,
    #[arg(long, default_value = "info")]
    pub log_level: String,
}

impl Cli {
    /// Build the full EngineConfig JSON object handed to `PyEngine::new`.
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
        if let Some(sizes) = &self.preferred_batch_sizes {
            let arr: Vec<Value> = sizes
                .iter()
                .map(|&v| Value::Number(v.into()))
                .collect();
            obj.entry("preferred_batch_size")
                .or_insert(Value::Array(arr));
        }
        if let Some(s) = &self.schedule_policy {
            obj.entry("schedule_policy")
                .or_insert(Value::String(s.clone()));
        }
        if let Some(r) = self.max_offline_pad_ratio {
            if let Some(n) = serde_json::Number::from_f64(r) {
                obj.entry("max_offline_pad_ratio")
                    .or_insert(Value::Number(n));
            }
        }
        // device defaults to "cuda" — EngineConfig falls back if absent.

        Ok(serde_json::to_string(&Value::Object(obj))?)
    }
}
