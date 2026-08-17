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
use serde_json::{Map, Number, Value};

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
    /// Streaming: physical blocks in the shared paged KV pool.  Pass `0` to
    /// **derive it from free VRAM** at startup (the engine's
    /// `max_num_blocks=None`) instead of hand-computing it from layers x heads x
    /// head_dim x dtype; unset keeps the engine default (2048).
    ///
    /// An undersized pool raises `BlockPool exhausted` from inside the encoder
    /// forward — where it takes out the tick for every concurrent stream — and an
    /// oversized one OOMs at startup, so deriving is how one config moves between
    /// a 24 GB and an 80 GB card.  Inert in `--service-mode offline`, which
    /// allocates no pool.
    #[arg(long)]
    pub max_num_blocks: Option<u32>,
    /// Share of the device the engine may occupy in total — weights, caches and
    /// activations together — when it derives a capacity from VRAM
    /// (`--max-num-blocks 0`, or an AR decode family with no
    /// `--decode-kv-budget-gib`).  Engine default 0.90; the unspent remainder is
    /// headroom for CUDA-graph pools, AR prefill transients and allocator
    /// fragmentation.
    #[arg(long)]
    pub gpu_memory_utilization: Option<f64>,
    /// Decoder type: `ctc_cuda` (GPU CTC beam, engine default) or `ctc_wfst`
    /// (in-tree GPU WFST beam search). Forwarded verbatim to the Python
    /// `EngineConfig.decoder_type`.
    #[arg(long)]
    pub decoder_type: Option<String>,
    /// WFST decoding graph for `--decoder-type ctc_wfst`: a prebuilt `.img`
    /// image or a k2 `HLG.pt` (exported + cached on first use). Forwarded to
    /// `EngineConfig.fst_path`.
    #[arg(long)]
    pub fst_path: Option<String>,
    /// Decode-method selection among the checkpoint's capabilities (e.g.
    /// `ctc_aed_rescoring` on a U2++ hybrid, `llm` on a speech-LLM).  Unset
    /// runs the model's default decode family.  Forwarded to
    /// `EngineConfig.decode_method`; the engine validates the name against
    /// `model.capabilities` at startup.
    #[arg(long)]
    pub decode_method: Option<String>,
    /// Speech-LLM only: the user prompt placed in the checkpoint's chat
    /// template next to the audio.  Unset uses the checkpoint's default ASR
    /// prompt.  Forwarded to `EngineConfig.llm_prompt`; per-request `prompt`
    /// decoding options override it.
    #[arg(long)]
    pub llm_prompt: Option<String>,
    /// AR generation length cap per request (AED / LLM decode families).
    /// Engine default 448.  Per-request `max_new_tokens` decoding options
    /// override it.
    #[arg(long)]
    pub max_new_tokens: Option<u32>,
    /// Ceiling on total decoder-KV bytes across in-flight AR requests, in GiB.
    ///
    /// `--max-decode-slots` bounds admission by request *count*, which does not
    /// bound memory: a row's KV footprint is its position budget (prompt +
    /// generation cap) times the model's per-token rate, and prefill
    /// preallocates all of it.  Unset **derives** the ceiling from free VRAM
    /// (see `--gpu-memory-utilization`); pass `0` to turn the byte budget off
    /// and leave the slot cap as the only limit.
    #[arg(long)]
    pub decode_kv_budget_gib: Option<f64>,
    /// Streaming: recycle the oldest KV block when a stream reaches its cache
    /// ceiling, instead of finalising it with `finish_reason="length"`.
    ///
    /// Makes streaming memory bounded by construction and lets a stream run
    /// indefinitely.  Measured identical (0.00% WER) for audio inside the
    /// retained window; past it the recycling run decodes the whole file where
    /// unlimited history truncates.  Off by default because it does change the
    /// attention span for very long streams.
    #[arg(long, default_value_t = false)]
    pub recycle_streaming_history: bool,
    /// Decode audio longer than a fixed-window frontend's window (Whisper /
    /// Qwen2-Audio's 30 s) by splitting it into windows and stitching the
    /// transcripts, instead of rejecting it.
    ///
    /// Windows decode in parallel, so a long file costs about one window of
    /// wall clock rather than N sequential decodes; the price is boundary
    /// accuracy, which `--long-form-overlap-seconds` mitigates.
    #[arg(long, default_value_t = false)]
    pub long_form: bool,
    /// Audio shared between adjacent long-form windows, in seconds.  Overlapping
    /// lets the stitcher drop words duplicated at a cut instead of losing one.
    /// Engine default 1.0; 0 disables.
    #[arg(long)]
    pub long_form_overlap_seconds: Option<f64>,
    /// Generic per-family decode knob, repeatable: `--decode-option k=v`.
    ///
    /// Forwarded verbatim to `EngineConfig.decode_options` and validated
    /// against the **active** decode family's option set at engine
    /// construction, so an unknown or misspelled key is a startup error naming
    /// the valid ones rather than a silently ignored flag.  This is what lets a
    /// newly registered decode family expose its configuration without a new
    /// flag here — and it reaches the three knobs that never got one
    /// (`rescoring_ctc_weight`, `rescoring_reverse_weight`,
    /// `transducer_max_sym_per_frame`) as `ctc_weight`, `reverse_weight` and
    /// `max_sym_per_frame`.  Values are typed from the option's declared
    /// default, so `--decode-option ctc_weight=0.3` arrives as a float.
    #[arg(long = "decode-option", value_name = "KEY=VALUE")]
    pub decode_option: Vec<String>,
    /// Incremental (AED / LLM) decode: max batched decoder steps one engine
    /// tick runs across all pending requests — the bounded-work-per-tick
    /// contract that keeps AR decode from starving the dispatcher.  Engine
    /// default 32.
    #[arg(long)]
    pub decode_steps_per_tick: Option<u32>,
    /// Incremental (AED / LLM) decode: wall-clock cap on one tick's decode phase,
    /// in milliseconds.  The step cap above bounds work, not time, and step cost
    /// is model-dependent (measured: ~1.5 ms/step for whisper-tiny at B=8 vs
    /// ~18 ms/step for Qwen2-Audio-7B at B=4), so this is what actually bounds
    /// cancel latency, admission latency and the streaming-partial interval —
    /// the dispatcher holds the GIL for a whole tick.  Engine default 25;
    /// 0 disables (step cap only).
    #[arg(long)]
    pub max_tick_ms: Option<f64>,
    /// Incremental (AED / LLM) decode: hold a thin waiting queue this many
    /// milliseconds so near-simultaneous arrivals prefill as **one** decode
    /// batch.  An AR decoder step is weight-read bound, so its cost barely
    /// depends on how many rows it carries — two decode groups cost roughly
    /// twice one group of the same total rows, and groups cannot be merged after
    /// the fact (both decoder surfaces keep a shared scalar generation offset).
    /// Trades first-token latency for throughput; engine default 0 (off).
    #[arg(long)]
    pub decode_admit_window_ms: Option<f64>,
    /// Incremental (AED / LLM) decode: max AR requests in flight before
    /// new-batch admission pauses.  Unset defaults to the engine's
    /// max_batch_size.
    #[arg(long)]
    pub max_decode_slots: Option<u32>,
    /// Offline only: overlap per-request admission prep (waveform load, scale,
    /// frame stamp) with the GPU ``step()`` on a daemon prep thread.  Helps at
    /// high concurrency (a deep backlog to pipeline); can slightly regress at
    /// low concurrency due to GIL contention, so it is opt-in.  No effect in
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
    /// Offline: longest single request the front-end may hand over in
    /// **page-locked** host memory, in seconds of audio.  Engine default 300.
    ///
    /// Pinning is what lets the engine DMA each row of a micro-batch straight
    /// into the padded device batch instead of packing it into staging first —
    /// one copy of the waveform after the codec instead of two.  Page-locked
    /// memory is process-global and the allocator keeps what it takes, so the
    /// high-water is this cap times `--max-concurrent-requests`; `0` turns the
    /// hand-off off entirely and everything falls back to ordinary heap
    /// buffers, which is the older, slower, still-correct path.
    #[arg(long)]
    pub max_pinned_audio_seconds: Option<f64>,
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
    /// Largest audio payload one request may carry, in MiB.  Drives **both**
    /// the HTTP body cap and the gRPC `max_decoding_message_size`, because
    /// tonic's undeclared 4 MiB default against HTTP's 256 MiB meant the same
    /// audio was accepted on one surface and rejected on the other — and the
    /// docs recommend gRPC as the higher-throughput offline path.
    #[arg(long, default_value_t = 256)]
    pub max_audio_mib: usize,
    /// Longest **decoded** audio one request may carry, in seconds.  0
    /// disables.
    ///
    /// `--max-audio-mib` bounds the *encoded* body, and once compressed
    /// containers are accepted the two stop being related: a few MiB of MP3 is
    /// hours of waveform, allocated before anything could notice.  The default
    /// is generous — this is a backstop against a decode bomb, not a product
    /// limit.
    #[arg(long, default_value_t = 4 * 3600)]
    pub max_audio_seconds: u64,
    /// Names this server answers to on the OpenAI surface's `model` field.
    /// Repeatable.
    ///
    /// Unset (the default) accepts **any** name, which is what keeps a client
    /// pointed at `whisper-1` working after nothing but a base-URL change. Set
    /// it and an unrecognised name is a 404, as OpenAI does — worth turning on
    /// behind a router that fans out to several models.
    #[arg(long)]
    pub served_model_name: Vec<String>,
    /// Origin allowed to call the HTTP API from a browser, repeatable; `*`
    /// allows any.
    ///
    /// Off by default: a browser cannot call this API cross-origin without it,
    /// and whether an inference endpoint should be callable from any page is an
    /// operator's decision. Needed for the `examples/web` demo, which talks to
    /// the server directly.
    #[arg(long)]
    pub cors_allow_origin: Vec<String>,
    /// Deadline for a single **unary** request (HTTP `speech:recognize`,
    /// gRPC `Recognize`), in seconds.  0 disables.  Streaming RPCs are bounded
    /// by `--stream-idle-timeout-secs` instead: a blanket deadline would kill
    /// healthy long-lived streams.
    #[arg(long, default_value_t = 300)]
    pub request_timeout_secs: u64,
    /// Abort a streaming RPC that has gone this long without an inbound audio
    /// message (before half-close) or a decode event (after), in seconds.
    /// 0 disables.  Without it a client that opens a stream and vanishes
    /// without closing the connection holds an engine slot indefinitely.
    #[arg(long, default_value_t = 300)]
    pub stream_idle_timeout_secs: u64,
    /// Max requests either listener will process concurrently; excess ones
    /// queue (and are eventually cut off by the timeouts above) rather than
    /// each parking a multi-MiB body.  0 disables.  Defaults to
    /// 4x `--max-concurrent-requests`, i.e. deep enough that the engine's own
    /// admission cap is what clients actually see.
    #[arg(long)]
    pub max_inflight_connections: Option<usize>,
    /// How long to let in-flight requests finish after a shutdown signal
    /// before the listeners are dropped, in seconds.
    #[arg(long, default_value_t = 10)]
    pub shutdown_grace_secs: u64,
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
    /// Largest accepted audio payload in bytes (shared by both front-ends).
    pub fn max_audio_bytes(&self) -> usize {
        self.max_audio_mib.saturating_mul(1024 * 1024)
    }

    /// Ceiling on the decoded waveform in samples at `model_sample_rate`, or
    /// `None` when disabled.
    pub fn max_audio_samples(&self, model_sample_rate: u32) -> Option<usize> {
        (self.max_audio_seconds > 0)
            .then(|| (self.max_audio_seconds as usize).saturating_mul(model_sample_rate as usize))
    }

    /// Concurrency limit for each listener, defaulted from the admission cap.
    pub fn inflight_limit(&self) -> Option<usize> {
        let n = self
            .max_inflight_connections
            .unwrap_or_else(|| (self.max_concurrent_requests as usize).saturating_mul(4));
        (n > 0).then_some(n)
    }

    /// Per-request deadline for the unary surfaces, if enabled.
    pub fn request_timeout(&self) -> Option<std::time::Duration> {
        (self.request_timeout_secs > 0)
            .then(|| std::time::Duration::from_secs(self.request_timeout_secs))
    }

    /// Idle bound for the streaming RPCs, if enabled.
    pub fn stream_idle_timeout(&self) -> Option<std::time::Duration> {
        (self.stream_idle_timeout_secs > 0)
            .then(|| std::time::Duration::from_secs(self.stream_idle_timeout_secs))
    }

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
        if let Some(v) = self.max_num_blocks {
            // `0` is the CLI spelling of "derive from VRAM": clap cannot pass a
            // three-state Option<Option<u32>>, and the engine's sentinel for
            // derive is `null`.
            obj.entry("max_num_blocks").or_insert(if v == 0 {
                Value::Null
            } else {
                Value::Number(v.into())
            });
        }
        if let Some(v) = self.gpu_memory_utilization {
            if let Some(num) = serde_json::Number::from_f64(v) {
                obj.entry("gpu_memory_utilization")
                    .or_insert(Value::Number(num));
            }
        }
        if let Some(s) = &self.decoder_type {
            obj.entry("decoder_type")
                .or_insert(Value::String(s.clone()));
        }
        if let Some(s) = &self.fst_path {
            obj.entry("fst_path").or_insert(Value::String(s.clone()));
        }
        if let Some(s) = &self.decode_method {
            obj.entry("decode_method")
                .or_insert(Value::String(s.clone()));
        }
        if let Some(s) = &self.llm_prompt {
            obj.entry("llm_prompt").or_insert(Value::String(s.clone()));
        }
        if let Some(v) = self.max_new_tokens {
            obj.entry("max_new_tokens")
                .or_insert(Value::Number(v.into()));
        }
        if self.recycle_streaming_history {
            obj.entry("recycle_streaming_history")
                .or_insert(Value::Bool(true));
        }
        if self.long_form {
            obj.entry("long_form").or_insert(Value::Bool(true));
        }
        if let Some(v) = self.long_form_overlap_seconds {
            if let Some(num) = serde_json::Number::from_f64(v) {
                obj.entry("long_form_overlap_seconds")
                    .or_insert(Value::Number(num));
            }
        }
        if let Some(v) = self.decode_kv_budget_gib {
            if let Some(num) = serde_json::Number::from_f64(v) {
                obj.entry("decode_kv_budget_gib")
                    .or_insert(Value::Number(num));
            }
        }
        if !self.decode_option.is_empty() {
            // Pass the raw `k=v` strings through; the engine types each value
            // from the active family's declared default and rejects unknown
            // keys.  Doing the typing here would mean this crate tracking every
            // family's option table — exactly the drift S9 is about.
            let mut opts = serde_json::Map::new();
            for pair in &self.decode_option {
                let (k, v) = pair.split_once('=').ok_or_else(|| {
                    anyhow::anyhow!("--decode-option expects KEY=VALUE, got {pair:?}")
                })?;
                opts.insert(k.trim().to_string(), Value::String(v.to_string()));
            }
            obj.entry("decode_options").or_insert(Value::Object(opts));
        }
        if let Some(v) = self.decode_steps_per_tick {
            obj.entry("decode_steps_per_tick")
                .or_insert(Value::Number(v.into()));
        }
        if let Some(v) = self.max_decode_slots {
            obj.entry("max_decode_slots")
                .or_insert(Value::Number(v.into()));
        }
        if let Some(v) = self.max_tick_ms {
            if let Some(num) = serde_json::Number::from_f64(v) {
                obj.entry("max_tick_ms").or_insert(Value::Number(num));
            }
        }
        if let Some(v) = self.decode_admit_window_ms {
            if let Some(num) = serde_json::Number::from_f64(v) {
                obj.entry("decode_admit_window_ms")
                    .or_insert(Value::Number(num));
            }
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
            obj.entry("max_batch_frames")
                .or_insert(Value::Number(v.into()));
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
            obj.entry("streaming_cohort_admit")
                .or_insert(Value::Bool(b));
        }
        if let Some(b) = self.use_ctc_cuda_graphs {
            obj.entry("use_ctc_cuda_graphs").or_insert(Value::Bool(b));
        }
        if let Some(b) = self.use_feature_cuda_graphs {
            obj.entry("use_feature_cuda_graphs")
                .or_insert(Value::Bool(b));
        }
        if let Some(v) = self.partial_decode_interval {
            obj.entry("partial_decode_interval")
                .or_insert(Value::Number(v.into()));
        }
        if let Some(b) = self.overlap_partial_readback {
            obj.entry("overlap_partial_readback")
                .or_insert(Value::Bool(b));
        }
        if let Some(v) = self.max_pinned_audio_seconds {
            if let Some(n) = Number::from_f64(v) {
                obj.entry("max_pinned_audio_seconds")
                    .or_insert(Value::Number(n));
            }
        }
        if let Some(b) = self.enable_sequence_packing {
            obj.entry("enable_sequence_packing")
                .or_insert(Value::Bool(b));
        }
        if let Some(v) = self.max_packed_frames {
            obj.entry("max_packed_frames")
                .or_insert(Value::Number(v.into()));
        }
        // device defaults to "cuda" — EngineConfig falls back if absent.

        Ok(serde_json::to_string(&Value::Object(obj))?)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn cli(extra: &[&str]) -> Cli {
        let mut argv = vec!["oasr-server", "--ckpt-dir", "/tmp/ckpt"];
        argv.extend_from_slice(extra);
        Cli::try_parse_from(argv).expect("parse")
    }

    /// One knob feeds both surfaces.  They used to disagree by 64x — tonic's
    /// undeclared 4 MiB default against HTTP's explicit 256 MiB — so the same
    /// audio was accepted on one and rejected on the other, on the surface the
    /// docs recommend for offline throughput.
    #[test]
    fn the_audio_cap_is_one_number_for_both_front_ends() {
        assert_eq!(cli(&[]).max_audio_bytes(), 256 * 1024 * 1024);
        assert_eq!(
            cli(&["--max-audio-mib", "8"]).max_audio_bytes(),
            8 * 1024 * 1024
        );
    }

    /// The connection bound defaults above the engine's admission cap so the
    /// engine's own `Busy` is what clients see, not a queue in the listener.
    #[test]
    fn the_inflight_limit_defaults_above_the_admission_cap() {
        let c = cli(&["--max-concurrent-requests", "256"]);
        assert_eq!(c.inflight_limit(), Some(1024));
        assert!(c.inflight_limit().unwrap() > c.max_concurrent_requests as usize);
    }

    /// The pinned-audio cap must be **absent** from the JSON when the flag is
    /// unset, or the CLI silently pins the engine default in place and a later
    /// change to it never reaches a served process (the `--dtype` /
    /// `--service-mode` drift channel, which this flag deliberately avoids by
    /// being `Option`).  `0` is a real operator choice — "stop page-locking" —
    /// so it has to survive as `0`, not be read as "unset".
    #[test]
    fn the_pinned_audio_cap_is_only_forwarded_when_asked_for() {
        let unset: Value =
            serde_json::from_str(&cli(&[]).build_engine_config_json().unwrap()).unwrap();
        assert_eq!(unset.get("max_pinned_audio_seconds"), None);

        let off: Value = serde_json::from_str(
            &cli(&["--max-pinned-audio-seconds", "0"])
                .build_engine_config_json()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            off.get("max_pinned_audio_seconds").and_then(|v| v.as_f64()),
            Some(0.0)
        );

        let set: Value = serde_json::from_str(
            &cli(&["--max-pinned-audio-seconds", "12.5"])
                .build_engine_config_json()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            set.get("max_pinned_audio_seconds").and_then(|v| v.as_f64()),
            Some(12.5)
        );
    }

    #[test]
    fn limits_are_individually_disablable() {
        assert_eq!(
            cli(&["--max-inflight-connections", "0"]).inflight_limit(),
            None
        );
        assert_eq!(
            cli(&["--request-timeout-secs", "0"]).request_timeout(),
            None
        );
        assert_eq!(
            cli(&["--stream-idle-timeout-secs", "0"]).stream_idle_timeout(),
            None
        );
    }

    /// `--max-num-blocks 0` is the CLI spelling of the engine's `None` sentinel
    /// ("derive the paged KV pool from free VRAM").  A `0` forwarded literally
    /// would be rejected by `EngineConfig` as an empty pool, and an omitted flag
    /// must leave the engine default alone rather than deriving behind the
    /// operator's back.
    #[test]
    fn zero_blocks_asks_the_engine_to_derive_the_pool() {
        let derive: Value = serde_json::from_str(
            &cli(&["--max-num-blocks", "0"])
                .build_engine_config_json()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(derive.get("max_num_blocks"), Some(&Value::Null));

        let explicit: Value = serde_json::from_str(
            &cli(&["--max-num-blocks", "4096"])
                .build_engine_config_json()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(explicit["max_num_blocks"], 4096);

        let untouched: Value =
            serde_json::from_str(&cli(&[]).build_engine_config_json().unwrap()).unwrap();
        assert!(untouched.get("max_num_blocks").is_none());
        assert!(untouched.get("gpu_memory_utilization").is_none());
    }

    #[test]
    fn utilization_reaches_the_engine() {
        let cfg: Value = serde_json::from_str(
            &cli(&["--gpu-memory-utilization", "0.75"])
                .build_engine_config_json()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(cfg["gpu_memory_utilization"], 0.75);
    }

    #[test]
    fn timeouts_parse_into_durations() {
        assert_eq!(
            cli(&["--request-timeout-secs", "30"]).request_timeout(),
            Some(std::time::Duration::from_secs(30))
        );
        assert_eq!(
            cli(&["--stream-idle-timeout-secs", "45"]).stream_idle_timeout(),
            Some(std::time::Duration::from_secs(45))
        );
    }
}
