// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Shared event / command types between the Rust serving frontend and the
//! embedded Python `ASREngine`.
//!
//! Previously these types doubled as a msgpack wire schema for the ZMQ
//! worker boundary; that transport is gone and the types are now used
//! purely in-process (the dispatcher constructs `Event` values directly
//! from `ASREngine.step()` output and the HTTP/gRPC adapters consume
//! them).  `serde` is retained because the gRPC + HTTP layers convert
//! `Event` payloads into their own response shapes via `serde_json`.

use serde::{Deserialize, Serialize};

/// Per-request decoding options forwarded verbatim into the Python engine's
/// `DecodingOptions` (see `oasr/engine/request.py`).  Every field is
/// optional; `None` keeps the engine default, so a default-constructed
/// params value is indistinguishable from not sending one.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DecodingParams {
    /// Hypotheses to detokenize into `Event::Final::nbest_texts`
    /// (maps from the proto `max_alternatives`).
    #[serde(default)]
    pub n_best: Option<u32>,
    /// Per-request generation cap for the AR families (AED / LLM).
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
    /// `> 0` enables sampling for the AR families (default greedy).
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-k filter (0 = disabled); only meaningful with `temperature > 0`.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Nucleus filter in (0, 1]; only meaningful with `temperature > 0`.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Speech-LLM user-prompt override (ignored by other families).
    #[serde(default)]
    pub prompt: Option<String>,
    /// `"transcribe"` or `"translate"` — Whisper's task token.  Frozen at
    /// checkpoint-conversion time before this existed, which is why
    /// `/v1/audio/translations` could not be served.
    #[serde(default)]
    pub task: Option<String>,
    /// Language for the families that can select one (Whisper's language
    /// token).  A primary subtag (`"en"`), not a BCP-47 tag — the front-ends
    /// reduce `"en-US"` before it gets here.
    #[serde(default)]
    pub language: Option<String>,
}

/// Sampling-temperature bounds, mirroring `oasr.engine.request.MIN_TEMPERATURE`
/// / `MAX_TEMPERATURE`.  Outside this range (and non-zero) the Python side
/// raises, so the front-ends reject here instead — a raise from inside
/// `add_requests_batch` used to fail every coalesced admit in the same batch.
pub const MIN_TEMPERATURE: f32 = 0.01;
pub const MAX_TEMPERATURE: f32 = 100.0;
/// Upper bound on `n_best` (proto `max_alternatives`).  Google's STT caps
/// alternatives at 30; anything beyond that is a client bug, and `u32::MAX`
/// would be accepted silently otherwise.
pub const MAX_N_BEST: u32 = 30;
/// Upper bound on a speech-LLM prompt override, in bytes.  A long prompt eats
/// the AR generation budget (the LM's position capacity is shared between
/// prompt and output), so an unbounded one silently clamps generation to a
/// single token.
pub const MAX_PROMPT_BYTES: usize = 4096;
/// The task values any decode family understands.  Whisper's own pair; a
/// checkpoint that cannot do one of them rejects it at admission, where the
/// token table lives.
pub const TASKS: &[&str] = &["transcribe", "translate"];
/// Upper bound on a language tag.  Whisper's are two or three letters; the
/// slack is for the handful of longer ISO-639-3 codes.
pub const MAX_LANGUAGE_LEN: usize = 16;

/// Reduce a language tag to the primary subtag, lowercased: `"en-US"` → `"en"`.
///
/// Both API surfaces take BCP-47 (`language_code` on the Google shape,
/// `language` on the OpenAI one) while the models' language tokens are ISO-639
/// primary subtags. Returns `None` for an empty or non-alphabetic tag, which
/// the caller turns into `INVALID_ARGUMENT` — silently dropping a language the
/// client asked for is how a request gets transcribed in the wrong one.
pub fn normalize_language(tag: &str) -> Option<String> {
    let primary = tag.trim().split(['-', '_']).next().unwrap_or("");
    if primary.is_empty()
        || primary.len() > MAX_LANGUAGE_LEN
        || !primary.chars().all(|c| c.is_ascii_alphabetic())
    {
        return None;
    }
    Some(primary.to_ascii_lowercase())
}

impl DecodingParams {
    /// The per-request option names, in declaration order.
    ///
    /// The one place the option table is written down on this side of the PyO3
    /// boundary. `decoding_params_dict` builds the Python dict from these keys,
    /// and the engine asserts at startup that they equal
    /// `oasr.engine.DecodingOptions.option_keys()`. Adding a field to this
    /// struct without adding it here fails `option_keys_cover_every_field`
    /// below; adding it on only one side of the boundary fails at startup.
    /// Neither can degrade into an option that is accepted and ignored.
    pub const OPTION_KEYS: &'static [&'static str] = &[
        "n_best",
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "prompt",
        "task",
        "language",
    ];

    /// Whether every field is `None` — callers skip building the Python-side
    /// options dict entirely in that case.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Validate the ranges the Python `DecodingOptions` enforces.
    ///
    /// Returns a client-facing message on the first violation.  Both front-ends
    /// call this at request-mapping time so an out-of-range value becomes
    /// `INVALID_ARGUMENT` for *that* request, rather than an `INTERNAL` error
    /// for every request in the admit batch it happened to land in.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(n) = self.n_best {
            if n > MAX_N_BEST {
                return Err(format!("max_alternatives must be <= {MAX_N_BEST}, got {n}"));
            }
        }
        if let Some(t) = self.temperature {
            if !t.is_finite() {
                return Err(format!("temperature must be finite, got {t}"));
            }
            if t < 0.0 {
                return Err(format!("temperature must be >= 0, got {t}"));
            }
            if t > 0.0 && t < MIN_TEMPERATURE {
                return Err(format!(
                    "temperature must be 0 (greedy) or >= {MIN_TEMPERATURE}, got {t}"
                ));
            }
            if t > MAX_TEMPERATURE {
                return Err(format!("temperature must be <= {MAX_TEMPERATURE}, got {t}"));
            }
        }
        if let Some(p) = self.top_p {
            if !p.is_finite() || p <= 0.0 || p > 1.0 {
                return Err(format!("top_p must be in (0, 1], got {p}"));
            }
        }
        if let Some(prompt) = &self.prompt {
            if prompt.len() > MAX_PROMPT_BYTES {
                return Err(format!(
                    "prompt must be <= {MAX_PROMPT_BYTES} bytes, got {}",
                    prompt.len()
                ));
            }
        }
        if let Some(task) = &self.task {
            if !TASKS.contains(&task.as_str()) {
                return Err(format!("task must be one of {TASKS:?}, got {task:?}"));
            }
        }
        if let Some(lang) = &self.language {
            // Already normalized by the front-ends; re-checked because a
            // mis-normalized tag reaching the engine picks a *different*
            // language token, which reads as a confident wrong transcript.
            if normalize_language(lang).as_deref() != Some(lang.as_str()) {
                return Err(format!(
                    "language must be a primary subtag like \"en\", got {lang:?}"
                ));
            }
        }
        Ok(())
    }

    /// `Some(params)` when anything was set and the ranges check out;
    /// `None` when nothing was set (the allocation-free fast path).
    ///
    /// Front-ends build params with the "unset or zero means default" filters
    /// and then funnel through this one place, so the two surfaces cannot drift.
    pub fn validated(self) -> Result<Option<Self>, String> {
        if self.is_empty() {
            return Ok(None);
        }
        self.validate()?;
        Ok(Some(self))
    }
}

/// Commands sent into the engine dispatcher.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum Cmd {
    /// Submit a fully-buffered offline transcription.  Audio bytes ride
    /// alongside the command in [`crate::CmdEnvelope::payload`].
    CreateOffline {
        request_id: String,
        sample_rate: u32,
        #[serde(default)]
        priority: i32,
        #[serde(default)]
        decoding: Option<DecodingParams>,
    },

    /// Open a streaming request.  Audio chunks arrive via [`Cmd::FeedChunk`].
    CreateStreaming {
        request_id: String,
        sample_rate: u32,
        #[serde(default)]
        priority: i32,
        #[serde(default)]
        decoding: Option<DecodingParams>,
    },

    /// Push one audio chunk into an open streaming request.
    FeedChunk { request_id: String, is_last: bool },

    /// Abort a request; the engine frees its cache and emits a final
    /// [`Event::Error`] with code [`ErrorCode::Shutdown`].
    Cancel { request_id: String },

    /// Health probe — returns a [`Event::Pong`] containing engine load.
    Ping { seq: u64 },
}

impl Cmd {
    /// Owning request id for routing in the per-engine `RouterActor`.
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Cmd::CreateOffline { request_id, .. }
            | Cmd::CreateStreaming { request_id, .. }
            | Cmd::FeedChunk { request_id, .. }
            | Cmd::Cancel { request_id } => Some(request_id.as_str()),
            Cmd::Ping { .. } => None,
        }
    }
}

/// Events emitted by the engine dispatcher back to HTTP / gRPC handlers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum Event {
    /// Request was admitted (or queued).
    Accepted { request_id: String },

    /// Streaming partial transcript; more updates expected.
    Partial {
        request_id: String,
        text: String,
        tokens: Vec<Vec<u32>>,
        scores: Option<Vec<f32>>,
    },

    /// Final transcript; no further events for this request.
    Final {
        request_id: String,
        text: String,
        tokens: Vec<Vec<u32>>,
        scores: Option<Vec<f32>>,
        /// Detokenized top-N transcripts aligned with `tokens` rows
        /// (`nbest_texts[0] == text`); present only when the request asked
        /// for more than one alternative and the decode family produced
        /// multiple hypotheses.
        #[serde(default)]
        nbest_texts: Option<Vec<String>>,
        /// End time (seconds) of the last decoded token, from the engine's
        /// per-token timestamps (decode families with alignments —
        /// Paraformer CIF).  Maps to the proto `result_end_time`.
        #[serde(default)]
        end_time_s: Option<f32>,
        /// Why generation stopped: `"stop"` (EOS) or `"length"` (hit
        /// `max_new_tokens` / the cache ceiling).  Without it a transcript
        /// truncated by the generation cap is indistinguishable from a
        /// complete one at the API — the client cannot tell that asking for
        /// more tokens would have produced more text.  `None` for the
        /// one-shot families, which have no such distinction.
        #[serde(default)]
        finish_reason: Option<String>,
    },

    /// Per-request error (also used for shutdown / worker-lost notifications).
    Error {
        request_id: String,
        code: ErrorCode,
        message: String,
    },

    /// Heartbeat response with load + model metadata.
    Pong {
        seq: u64,
        model_info: Option<ModelInfo>,
        num_running: u32,
        num_waiting: u32,
    },

    /// Engine is over capacity; frontend should shed load until clear.
    Overloaded { reason: String, queue_depth: u32 },
}

impl Event {
    /// Owning request id (for events that target a specific request).
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Event::Accepted { request_id }
            | Event::Partial { request_id, .. }
            | Event::Final { request_id, .. }
            | Event::Error { request_id, .. } => Some(request_id.as_str()),
            Event::Pong { .. } | Event::Overloaded { .. } => None,
        }
    }

    /// Whether this event terminates the per-request channel.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Event::Final { .. } | Event::Error { .. })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ErrorCode {
    #[serde(rename = "BUSY")]
    Busy,
    #[serde(rename = "UNKNOWN_REQUEST")]
    UnknownRequest,
    #[serde(rename = "INVALID_CMD")]
    InvalidCmd,
    #[serde(rename = "INTERNAL")]
    Internal,
    #[serde(rename = "SHUTDOWN")]
    Shutdown,
    #[serde(rename = "WORKER_LOST")]
    WorkerLost,
}

/// Softmax-normalized posteriors over the returned n-best scores.
///
/// The engine's `scores` are per-hypothesis log-probabilities; the serving
/// `confidence` fields are [0, 1] estimates, so both front-ends report each
/// hypothesis's posterior among the n-best.  With fewer than two hypotheses
/// there is nothing to normalize against — returns `None` and the caller
/// leaves `confidence` at 0.0 (Google's "unset when unavailable").
pub fn score_posteriors(scores: &Option<Vec<f32>>) -> Option<Vec<f32>> {
    let s = scores.as_ref()?;
    if s.len() < 2 {
        return None;
    }
    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = s.iter().map(|&v| (v - m).exp()).collect();
    let z: f32 = exps.iter().sum();
    Some(exps.into_iter().map(|e| e / z).collect())
}

/// Static model metadata returned in `Pong.model_info`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    #[serde(default)]
    pub ckpt_dir: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub chunk_size: Option<u32>,
    #[serde(default)]
    pub max_batch_size: Option<u32>,
    /// CTC *kernel* selector (`ctc_cuda` / `ctc_wfst`) — not the decode family.
    #[serde(default)]
    pub decoder_type: Option<String>,
    #[serde(default)]
    pub vocab_size: Option<u32>,
    /// The mode the engine was actually built for, read back from it rather than
    /// from the CLI flag: several decode families are offline-only, and the two
    /// sources used to be able to disagree silently.
    #[serde(default)]
    pub service_mode: Option<String>,
    /// The resolved decode family running in this process (`ctc`, `transducer`,
    /// `aed`, `llm`, `paraformer`, `ctc_aed_rescoring`).
    #[serde(default)]
    pub decode_method: Option<String>,
    /// Every decode family this checkpoint could serve.
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// The waveform sample rate the engine accepts, in Hz.  Read off the
    /// engine's resolved feature config: the engine does **not** resample, so
    /// the front-ends convert client audio to this rate before submitting.
    #[serde(default)]
    pub sample_rate: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `OPTION_KEYS` must name every field of `DecodingParams`.
    ///
    /// The first link of the S9 chain: struct → `OPTION_KEYS` → the Python
    /// dataclass (asserted at engine startup). Without this, adding a field
    /// here and forgetting the const gives an option that serialises over the
    /// wire, never reaches the dict, and reports nothing.
    #[test]
    fn option_keys_cover_every_field() {
        let all = DecodingParams {
            n_best: Some(1),
            max_new_tokens: Some(1),
            temperature: Some(1.0),
            top_k: Some(1),
            top_p: Some(1.0),
            prompt: Some("x".into()),
            task: Some("transcribe".into()),
            language: Some("en".into()),
        };
        let json = serde_json::to_value(&all).expect("serialize");
        let mut fields: Vec<&str> = json
            .as_object()
            .expect("object")
            .keys()
            .map(|s| s.as_str())
            .collect();
        let mut keys: Vec<&str> = DecodingParams::OPTION_KEYS.to_vec();
        fields.sort_unstable();
        keys.sort_unstable();
        assert_eq!(
            fields, keys,
            "DecodingParams fields and OPTION_KEYS disagree; update both"
        );
    }

    #[test]
    fn decoding_params_is_empty() {
        assert!(DecodingParams::default().is_empty());
        assert!(!DecodingParams {
            n_best: Some(3),
            ..Default::default()
        }
        .is_empty());
    }

    #[test]
    fn validated_passes_through_empty_and_ok() {
        // Nothing set → no params dict is built at all.
        assert!(DecodingParams::default().validated().unwrap().is_none());
        let ok = DecodingParams {
            n_best: Some(5),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            max_new_tokens: Some(64),
            prompt: Some("transcribe".into()),
            task: Some("translate".into()),
            language: Some("fr".into()),
        };
        assert_eq!(ok.clone().validated().unwrap(), Some(ok));
    }

    #[test]
    fn language_tags_reduce_to_their_primary_subtag() {
        assert_eq!(normalize_language("en-US").as_deref(), Some("en"));
        assert_eq!(normalize_language("zh_Hans").as_deref(), Some("zh"));
        assert_eq!(normalize_language("  FR  ").as_deref(), Some("fr"));
        assert_eq!(normalize_language("yue").as_deref(), Some("yue"));
        // Junk is rejected rather than passed through to become a token lookup
        // miss deep inside the decode strategy.
        assert_eq!(normalize_language(""), None);
        assert_eq!(normalize_language("-"), None);
        assert_eq!(normalize_language("e1"), None);
        assert_eq!(normalize_language(&"a".repeat(MAX_LANGUAGE_LEN + 1)), None);
    }

    /// A misspelled task must fail its own request.  Accepting it and running
    /// the checkpoint's default is the "accepted and ignored" failure this
    /// option table exists to prevent.
    #[test]
    fn an_unknown_task_is_rejected() {
        let err = DecodingParams {
            task: Some("summarize".into()),
            ..Default::default()
        }
        .validated()
        .expect_err("unknown task should be rejected");
        assert!(err.contains("task"), "{err:?}");
        for t in TASKS {
            assert!(DecodingParams {
                task: Some((*t).into()),
                ..Default::default()
            }
            .validated()
            .is_ok());
        }
    }

    #[test]
    fn an_unnormalized_language_is_rejected() {
        let err = DecodingParams {
            language: Some("en-US".into()),
            ..Default::default()
        }
        .validated()
        .expect_err("a full BCP-47 tag must be normalized by the front-end first");
        assert!(err.contains("language"), "{err:?}");
    }

    #[test]
    fn validated_rejects_out_of_range() {
        // Each of these used to reach Python and raise from inside
        // `add_requests_batch`, failing every coalesced admit in the batch.
        let cases = [
            (
                "top_p",
                DecodingParams {
                    top_p: Some(1.5),
                    ..Default::default()
                },
            ),
            (
                "temperature",
                DecodingParams {
                    temperature: Some(1e-30),
                    ..Default::default()
                },
            ),
            (
                "temperature",
                DecodingParams {
                    temperature: Some(1e6),
                    ..Default::default()
                },
            ),
            (
                "temperature",
                DecodingParams {
                    temperature: Some(f32::NAN),
                    ..Default::default()
                },
            ),
            (
                "max_alternatives",
                DecodingParams {
                    n_best: Some(u32::MAX),
                    ..Default::default()
                },
            ),
            (
                "prompt",
                DecodingParams {
                    prompt: Some("x".repeat(MAX_PROMPT_BYTES + 1)),
                    ..Default::default()
                },
            ),
        ];
        for (field, params) in cases {
            let err = params
                .validated()
                .expect_err(&format!("{field} should have been rejected"));
            assert!(err.contains(field), "message {err:?} should name {field}");
        }
    }

    #[test]
    fn validated_accepts_the_bounds_themselves() {
        for t in [MIN_TEMPERATURE, MAX_TEMPERATURE] {
            assert!(DecodingParams {
                temperature: Some(t),
                ..Default::default()
            }
            .validated()
            .is_ok());
        }
        assert!(DecodingParams {
            top_p: Some(1.0),
            ..Default::default()
        }
        .validated()
        .is_ok());
        assert!(DecodingParams {
            n_best: Some(MAX_N_BEST),
            ..Default::default()
        }
        .validated()
        .is_ok());
    }

    #[test]
    fn score_posteriors_normalizes() {
        // Equal scores → uniform posterior.
        let p = score_posteriors(&Some(vec![-5.0, -5.0])).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6);
        // Ordering preserved, sums to 1.
        let p = score_posteriors(&Some(vec![-1.0, -2.0, -4.0])).unwrap();
        assert!(p[0] > p[1] && p[1] > p[2]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn score_posteriors_unavailable() {
        assert!(score_posteriors(&None).is_none());
        assert!(score_posteriors(&Some(vec![])).is_none());
        // Single hypothesis: nothing to normalize against.
        assert!(score_posteriors(&Some(vec![-3.0])).is_none());
    }

    #[test]
    fn cmd_decoding_defaults_deserialize() {
        // Old-shape command JSON (no `decoding` key) must still deserialize.
        let j = r#"{"type":"CreateOffline","request_id":"r","sample_rate":16000}"#;
        let cmd: Cmd = serde_json::from_str(j).unwrap();
        match cmd {
            Cmd::CreateOffline {
                priority, decoding, ..
            } => {
                assert_eq!(priority, 0);
                assert!(decoding.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }
}
