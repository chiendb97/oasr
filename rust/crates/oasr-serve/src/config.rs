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
#[command(
    name = "oasr-server",
    version,
    about = "OASR HTTP + gRPC frontend with in-process Python engine"
)]
pub struct Cli {
    // ---- Engine config (mirror EngineConfig essentials) ----
    /// Required: WeNet checkpoint directory.
    #[arg(long)]
    pub ckpt_dir: Option<PathBuf>,
    /// torch.dtype string ("float16" | "bfloat16" | "float32").
    #[arg(long, default_value = "bfloat16")]
    pub dtype: String,
    /// Service mode — the engine runs in exactly one mode per lifecycle.
    /// "streaming" (default) accepts chunk-by-chunk requests via the gRPC
    /// `StreamingRecognize` RPC; "offline" accepts full-audio requests via
    /// `POST /v1/speech:recognize` and the gRPC `Recognize` RPC.  The
    /// mismatched RPC returns `FAILED_PRECONDITION` at the service layer.
    #[arg(long, default_value = "streaming")]
    pub service_mode: String,
    /// Optional: max batch size override.
    #[arg(long)]
    pub max_batch_size: Option<u32>,
    /// Encoder chunk size (frames).
    #[arg(long)]
    pub chunk_size: Option<u32>,
    /// Decoder type: `ctc_cuda` (GPU CTC beam, engine default) or `ctc_wfst`
    /// (k2 WFST). Forwarded verbatim to the Python `EngineConfig.decoder_type`.
    #[arg(long)]
    pub decoder_type: Option<String>,
    /// Offline only: overlap per-request admission prep (waveform load + scale
    /// + frame stamp) with the GPU ``step()`` on a daemon prep thread.  Helps
    /// at high concurrency (a deep backlog to pipeline); can slightly regress
    /// at low concurrency due to GIL contention, so it is opt-in.  No effect in
    /// streaming mode.
    #[arg(long, default_value_t = false)]
    pub overlap_admit: bool,
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
    /// Offline length-aware batching: hard cap on **padded** input frames per
    /// micro-batch (``max_len * batch_size`` in pre-subsampling frames).  Unset
    /// bounds each micro-batch solely by ``max_batch_size``.  Exact-equivalent
    /// to the padded forward — only batch composition changes — so it trims
    /// padded GPU waste on mixed-length offline traffic without accuracy drift.
    #[arg(long)]
    pub max_batch_frames: Option<u32>,
    /// Offline length-bucket tolerance: group requests so ``min_len/max_len >=
    /// this`` within a batch.  ``0`` disables (engine default), relying solely
    /// on ``max_offline_pad_ratio``.
    #[arg(long)]
    pub length_bucket_ratio: Option<f64>,
    /// Max seconds a waiting request may sit before it is force-admitted even
    /// without an ideal length-bucket peer (starvation bound).  Engine default
    /// 0.2.
    #[arg(long)]
    pub max_wait_time: Option<f64>,
    /// Streaming: when true (engine default) admission length-sorts the waiting
    /// queue so each batched paged forward is length-similar.  Pass ``false``
    /// for ``schedule_policy``-ordered admission (lower per-stream latency, more
    /// padded compute).
    #[arg(long)]
    pub streaming_cohort_admit: Option<bool>,
    /// Streaming: capture the CTC decode step into a per-state CUDA graph.
    /// Engine default false (the per-non-blank D2H of log-prob slices outweighs
    /// the launch saving at production scale); enable for small-B / many-short-
    /// utterance deployments.
    #[arg(long)]
    pub use_ctc_cuda_graphs: Option<bool>,
    /// Streaming: capture the batched fbank/mfcc feature extraction into a CUDA
    /// graph per B bucket.  Engine default false; enable for fixed preferred-B
    /// deployments where the launch saving beats the per-replay copy.
    #[arg(long)]
    pub use_feature_cuda_graphs: Option<bool>,
    /// Streaming interim-partial cadence: ``1`` (engine default) emits a partial
    /// every step (one batched D2H read-back), ``N>1`` every N-th step, ``<=0``
    /// disables interim partials (final transcript only) for throughput.
    #[arg(long)]
    pub partial_decode_interval: Option<i64>,
    /// Streaming: issue the interim-partial read-back non-blocking and emit the
    /// previous step's partial (one-chunk lag) so the blocking sync leaves the
    /// critical path.  Engine default false (lowest first-token latency).
    #[arg(long)]
    pub overlap_partial_readback: Option<bool>,
    /// Offline: pack several utterances into one gapless varlen encoder forward
    /// instead of padding each micro-batch.  Requires ``service_mode=offline``.
    #[arg(long)]
    pub enable_sequence_packing: Option<bool>,
    /// Token budget (post-subsampling encoder frames) for one packed row when
    /// ``enable_sequence_packing`` is set.  Engine default 8192.
    #[arg(long)]
    pub max_packed_frames: Option<u32>,
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
    /// Diagnostics: log a rolling per-tick dispatcher sub-stage timing
    /// breakdown (intake / admit / step / extract / route) + effective batch
    /// every ~2 s at INFO.  Off by default.  Used to decompose the
    /// service↔engine gap; needs `--log-level info`.
    #[arg(long, default_value_t = false)]
    pub trace_dispatch: bool,
    #[arg(long, default_value = "info")]
    pub log_level: String,
    /// Log output format: ``text`` (default, human-readable) or ``json``
    /// (one JSON object per line, including span fields such as ``rid``, for
    /// ingestion by log aggregators).  Unknown values fall back to ``text``.
    #[arg(long, default_value = "text")]
    pub log_format: String,
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
        if self.overlap_admit {
            obj.entry("overlap_admit").or_insert(Value::Bool(true));
        }
        if let Some(sizes) = &self.preferred_batch_sizes {
            let arr: Vec<Value> = sizes.iter().map(|&v| Value::Number(v.into())).collect();
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
        if let Some(v) = self.max_batch_frames {
            obj.entry("max_batch_frames").or_insert(Value::Number(v.into()));
        }
        if let Some(r) = self.length_bucket_ratio {
            if let Some(n) = serde_json::Number::from_f64(r) {
                obj.entry("length_bucket_ratio").or_insert(Value::Number(n));
            }
        }
        if let Some(r) = self.max_wait_time {
            if let Some(n) = serde_json::Number::from_f64(r) {
                obj.entry("max_wait_time").or_insert(Value::Number(n));
            }
        }
        if let Some(b) = self.streaming_cohort_admit {
            obj.entry("streaming_cohort_admit").or_insert(Value::Bool(b));
        }
        if let Some(b) = self.use_ctc_cuda_graphs {
            obj.entry("use_ctc_cuda_graphs").or_insert(Value::Bool(b));
        }
        if let Some(b) = self.use_feature_cuda_graphs {
            obj.entry("use_feature_cuda_graphs").or_insert(Value::Bool(b));
        }
        if let Some(v) = self.partial_decode_interval {
            obj.entry("partial_decode_interval").or_insert(Value::Number(v.into()));
        }
        if let Some(b) = self.overlap_partial_readback {
            obj.entry("overlap_partial_readback").or_insert(Value::Bool(b));
        }
        if let Some(b) = self.enable_sequence_packing {
            obj.entry("enable_sequence_packing").or_insert(Value::Bool(b));
        }
        if let Some(v) = self.max_packed_frames {
            obj.entry("max_packed_frames").or_insert(Value::Number(v.into()));
        }
        // device defaults to "cuda" — EngineConfig falls back if absent.

        Ok(serde_json::to_string(&Value::Object(obj))?)
    }
}
