// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! PyO3 wrapper around `oasr.engine.ASREngine`.
//!
//! `PyEngine` is **not** clone-safe and is owned by the dispatcher thread.
//! It is constructed once at startup with the JSON engine config, and every
//! method call acquires the GIL.  The dispatcher serialises access by virtue
//! of being a single thread.

use bytes::Bytes;
use numpy::{PyArray1, PyArrayMethods};
use oasr_wire::{DecodingParams, ErrorCode, Event, ModelInfo};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use thiserror::Error;

/// One unit of admission for [`PyEngine::add_requests_batch_locked`].
///
/// Owns its audio buffer (when present) so the dispatcher can drain
/// `CmdEnvelope::payload` directly into the spec vec without lifetime
/// gymnastics.  Audio bytes are widened into a fresh numpy array inside
/// the bulk call; the `Bytes` reference can be dropped immediately after.
pub enum AdmitSpec {
    Offline {
        rid: String,
        audio: Bytes,
        sample_rate: u32,
        priority: i32,
        decoding: Option<DecodingParams>,
    },
    Streaming {
        rid: String,
        sample_rate: u32,
        priority: i32,
        decoding: Option<DecodingParams>,
    },
}

impl AdmitSpec {
    pub fn request_id(&self) -> &str {
        match self {
            AdmitSpec::Offline { rid, .. } => rid,
            AdmitSpec::Streaming { rid, .. } => rid,
        }
    }
}

#[derive(Debug, Error)]
pub enum PyEngineError {
    #[error("python error: {0}")]
    Py(String),
    #[error("invalid engine config: {0}")]
    InvalidConfig(String),
}

impl From<PyErr> for PyEngineError {
    fn from(e: PyErr) -> Self {
        PyEngineError::Py(format!("{e}"))
    }
}

impl<'py> From<pyo3::DowncastIntoError<'py>> for PyEngineError {
    fn from(e: pyo3::DowncastIntoError<'py>) -> Self {
        PyEngineError::Py(format!("downcast: {e}"))
    }
}

impl<'py> From<pyo3::DowncastError<'_, 'py>> for PyEngineError {
    fn from(e: pyo3::DowncastError<'_, 'py>) -> Self {
        PyEngineError::Py(format!("downcast: {e}"))
    }
}

/// Thin Rust handle around `oasr.engine.ASREngine`.
pub struct PyEngine {
    /// `oasr.engine.ASREngine` instance.
    engine: Py<PyAny>,
    /// Cached model info collected at construction time.
    model_info: ModelInfo,
}

impl PyEngine {
    /// Build an `ASREngine` from a JSON config string.
    ///
    /// The JSON is decoded into a Python dict, then handed to
    /// `oasr.engine.EngineConfig(**cfg)`.  The `dtype` field is mapped from a
    /// string (`"float16"` / `"bfloat16"` / `"float32"`) to the matching
    /// `torch.dtype` value before the `EngineConfig` is constructed —
    /// `engine_worker._load_engine_config` did the same.
    pub fn new(engine_config_json: &str) -> Result<Self, PyEngineError> {
        Python::with_gil(|py| {
            // Before anything else: the per-request option table must agree
            // across the boundary, or options silently vanish (S9).
            assert_decoding_option_keys_match(py)?;

            // Parse the JSON into a Python dict using `json.loads` so we keep
            // serde out of the GIL critical section.
            let json_mod = PyModule::import_bound(py, "json")?;
            let cfg_obj = json_mod.getattr("loads")?.call1((engine_config_json,))?;
            let cfg: Bound<'_, PyDict> = cfg_obj.downcast_into()?;

            // Map string dtype → torch dtype.
            if let Some(dtype_val) = cfg.get_item("dtype")? {
                if let Ok(dtype_str) = dtype_val.extract::<String>() {
                    let torch = PyModule::import_bound(py, "torch")?;
                    let normalized = dtype_str.to_lowercase().replace("torch.", "");
                    let attr = match normalized.as_str() {
                        "float16" | "fp16" | "half" => "float16",
                        "bfloat16" | "bf16" => "bfloat16",
                        "float32" | "fp32" | "float" => "float32",
                        other => {
                            return Err(PyEngineError::InvalidConfig(format!(
                                "unsupported dtype string: {other:?}"
                            )));
                        }
                    };
                    let dtype = torch.getattr(attr)?;
                    cfg.set_item("dtype", dtype)?;
                }
            }

            let engine_mod = PyModule::import_bound(py, "oasr.engine")?;
            let engine_cfg_cls = engine_mod.getattr("EngineConfig")?;
            let engine_cfg = engine_cfg_cls.call((), Some(&cfg))?;
            let engine_cls = engine_mod.getattr("ASREngine")?;
            let engine = engine_cls.call1((engine_cfg.clone(),))?;

            let model_info = collect_model_info(py, &engine_cfg, &engine).unwrap_or_default();

            Ok(Self {
                engine: engine.unbind(),
                model_info,
            })
        })
    }

    pub fn model_info(&self) -> ModelInfo {
        self.model_info.clone()
    }

    /// Bind the engine handle under the caller's GIL token.  Cheap — used by
    /// the dispatcher to enter `Python::with_gil` once per tick and call the
    /// `*_locked` methods repeatedly under the same scope.
    pub fn bind_engine<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.engine.bind(py).clone()
    }

    /// Admission probe — returns `(num_running, num_waiting)`.
    pub fn load(&self) -> (u32, u32) {
        Python::with_gil(|py| {
            let bound = self.engine.bind(py);
            Self::load_locked(bound)
        })
    }

    /// GIL-already-held variant of [`load`].
    pub fn load_locked(bound: &Bound<'_, PyAny>) -> (u32, u32) {
        let r = bound
            .getattr("num_running")
            .and_then(|x| x.extract::<u32>())
            .unwrap_or(0);
        let w = bound
            .getattr("num_waiting")
            .and_then(|x| x.extract::<u32>())
            .unwrap_or(0);
        (r, w)
    }

    pub fn add_offline(
        &self,
        rid: &str,
        audio: &[u8],
        sample_rate: u32,
        priority: i32,
        decoding: Option<&DecodingParams>,
    ) -> Result<(), PyEngineError> {
        Python::with_gil(|py| {
            let bound = self.engine.bind(py);
            Self::add_offline_locked(py, bound, rid, audio, sample_rate, priority, decoding)
        })
    }

    /// GIL-already-held variant of [`add_offline`].
    #[allow(clippy::too_many_arguments)]
    pub fn add_offline_locked<'py>(
        py: Python<'py>,
        bound: &Bound<'py, PyAny>,
        rid: &str,
        audio: &[u8],
        sample_rate: u32,
        priority: i32,
        decoding: Option<&DecodingParams>,
    ) -> Result<(), PyEngineError> {
        let arr = audio_bytes_to_numpy(py, audio)?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("audio", arr)?;
        kwargs.set_item("request_id", rid)?;
        kwargs.set_item("sample_rate", sample_rate)?;
        kwargs.set_item("streaming", false)?;
        kwargs.set_item("priority", priority)?;
        if let Some(dict) = decoding_params_dict(py, decoding)? {
            kwargs.set_item("decoding", dict)?;
        }
        bound.call_method("add_request", (), Some(&kwargs))?;
        Ok(())
    }

    pub fn add_streaming(
        &self,
        rid: &str,
        sample_rate: u32,
        priority: i32,
        decoding: Option<&DecodingParams>,
    ) -> Result<(), PyEngineError> {
        Python::with_gil(|py| {
            let bound = self.engine.bind(py);
            Self::add_streaming_locked(py, bound, rid, sample_rate, priority, decoding)
        })
    }

    /// GIL-already-held variant of [`add_streaming`].
    pub fn add_streaming_locked<'py>(
        py: Python<'py>,
        bound: &Bound<'py, PyAny>,
        rid: &str,
        sample_rate: u32,
        priority: i32,
        decoding: Option<&DecodingParams>,
    ) -> Result<(), PyEngineError> {
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("request_id", rid)?;
        kwargs.set_item("sample_rate", sample_rate)?;
        kwargs.set_item("priority", priority)?;
        if let Some(dict) = decoding_params_dict(py, decoding)? {
            kwargs.set_item("decoding", dict)?;
        }
        bound.call_method("add_streaming_request", (), Some(&kwargs))?;
        Ok(())
    }

    /// Bulk admission entry — calls `ASREngine.add_requests_batch_checked(list)`
    /// in **one** Python method invocation across all `specs`.  GIL must already
    /// be held; intended to be called by the dispatcher inside its tick's
    /// `Python::with_gil` scope after draining contiguous admit envelopes.
    ///
    /// Returns one entry per spec, in order: `None` when that spec was admitted,
    /// `Some(message)` when the engine rejected it.  A rejection is scoped to its
    /// own request — the `_checked` Python entry point validates and admits per
    /// spec instead of raising for the batch, so one client's bad `top_p` can no
    /// longer error every request that happened to coalesce with it.  `Err` is
    /// reserved for a genuinely batch-wide failure (the Python call itself
    /// raising, e.g. an engine-level fault).
    pub fn add_requests_batch_locked<'py>(
        py: Python<'py>,
        bound: &Bound<'py, PyAny>,
        specs: &[AdmitSpec],
    ) -> Result<Vec<Option<String>>, PyEngineError> {
        if specs.is_empty() {
            return Ok(Vec::new());
        }
        let list = PyList::empty_bound(py);
        for spec in specs {
            let d = PyDict::new_bound(py);
            match spec {
                AdmitSpec::Offline {
                    rid,
                    audio,
                    sample_rate,
                    priority,
                    decoding,
                } => {
                    let arr = audio_bytes_to_numpy(py, audio)?;
                    d.set_item("audio", arr)?;
                    d.set_item("request_id", rid.as_str())?;
                    d.set_item("sample_rate", *sample_rate)?;
                    d.set_item("streaming", false)?;
                    d.set_item("priority", *priority)?;
                    if let Some(dict) = decoding_params_dict(py, decoding.as_ref())? {
                        d.set_item("decoding", dict)?;
                    }
                }
                AdmitSpec::Streaming {
                    rid,
                    sample_rate,
                    priority,
                    decoding,
                } => {
                    d.set_item("request_id", rid.as_str())?;
                    d.set_item("sample_rate", *sample_rate)?;
                    d.set_item("streaming", true)?;
                    d.set_item("priority", *priority)?;
                    if let Some(dict) = decoding_params_dict(py, decoding.as_ref())? {
                        d.set_item("decoding", dict)?;
                    }
                }
            }
            list.append(d)?;
        }
        let out = bound.call_method1("add_requests_batch_checked", (list,))?;
        let rows: Vec<Bound<'py, PyAny>> = out.extract()?;
        let mut errors: Vec<Option<String>> = Vec::with_capacity(specs.len());
        for row in rows.iter() {
            // `{"request_id": str}` on success, `{..., "error": str}` on rejection.
            let err = row
                .get_item("error")
                .ok()
                .and_then(|v| v.extract::<Option<String>>().ok())
                .flatten();
            errors.push(err);
        }
        // Defensive: a length mismatch would misattribute errors, so treat it as
        // batch-wide rather than guessing which spec each row belongs to.
        if errors.len() != specs.len() {
            return Err(PyEngineError::Py(format!(
                "add_requests_batch_checked returned {} rows for {} specs",
                errors.len(),
                specs.len()
            )));
        }
        Ok(errors)
    }

    pub fn feed_chunk(&self, rid: &str, chunk: &[u8], is_last: bool) -> Result<(), PyEngineError> {
        Python::with_gil(|py| {
            let bound = self.engine.bind(py);
            Self::feed_chunk_locked(py, bound, rid, chunk, is_last)
        })
    }

    /// GIL-already-held variant of [`feed_chunk`].
    pub fn feed_chunk_locked<'py>(
        py: Python<'py>,
        bound: &Bound<'py, PyAny>,
        rid: &str,
        chunk: &[u8],
        is_last: bool,
    ) -> Result<(), PyEngineError> {
        let arr = audio_bytes_to_numpy(py, chunk)?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("is_last", is_last)?;
        bound.call_method("feed_chunk", (rid, arr), Some(&kwargs))?;
        Ok(())
    }

    pub fn abort(&self, rid: &str) -> Result<(), PyEngineError> {
        Python::with_gil(|py| {
            let bound = self.engine.bind(py);
            Self::abort_locked(bound, rid)
        })
    }

    /// GIL-already-held variant of [`abort`].
    pub fn abort_locked(bound: &Bound<'_, PyAny>, rid: &str) -> Result<(), PyEngineError> {
        bound.call_method1("abort_request", (rid,))?;
        Ok(())
    }

    /// Run one engine step and return any `RequestOutput`s as native events.
    pub fn step(&self) -> Result<Vec<Event>, PyEngineError> {
        Python::with_gil(|py| {
            let bound = self.engine.bind(py);
            Self::step_locked(py, bound)
        })
    }

    /// GIL-already-held variant of [`step`].
    pub fn step_locked<'py>(
        _py: Python<'py>,
        bound: &Bound<'py, PyAny>,
    ) -> Result<Vec<Event>, PyEngineError> {
        let list = Self::step_raw(bound)?;
        Self::extract_events(&list)
    }

    /// Call `ASREngine.step()` and return the raw `RequestOutput` list without
    /// marshaling its fields.  Split out from [`step_locked`] so callers (the
    /// dispatcher) can time the GPU-bound step separately from the per-output
    /// PyO3 extraction in [`extract_events`].
    pub fn step_raw<'py>(bound: &Bound<'py, PyAny>) -> Result<Bound<'py, PyList>, PyEngineError> {
        let outputs = bound.call_method0("step")?;
        let list: Bound<'py, PyList> = outputs.downcast_into()?;
        Ok(list)
    }

    /// Marshal a `RequestOutput` list (from [`step_raw`]) into native events.
    /// This is the GIL-held per-output `getattr` + `Vec` materialization that
    /// runs on the dispatcher thread after each step.
    pub fn extract_events(list: &Bound<'_, PyList>) -> Result<Vec<Event>, PyEngineError> {
        let mut events = Vec::with_capacity(list.len());
        for item in list.iter() {
            let rid: String = item.getattr("request_id")?.extract()?;
            let text: String = item
                .getattr("text")
                .and_then(|x| x.extract())
                .unwrap_or_default();
            let finished: bool = item.getattr("finished")?.extract()?;
            let tokens: Vec<Vec<u32>> = item
                .getattr("tokens")
                .and_then(|x| x.extract::<Vec<Vec<u32>>>())
                .unwrap_or_default();
            let scores: Option<Vec<f32>> = item
                .getattr("scores")
                .ok()
                .and_then(|x| x.extract::<Option<Vec<f32>>>().ok())
                .unwrap_or(None);
            let evt = if finished {
                let nbest_texts: Option<Vec<String>> = item
                    .getattr("nbest_texts")
                    .ok()
                    .and_then(|x| x.extract::<Option<Vec<String>>>().ok())
                    .unwrap_or(None);
                // Last-token end time from the engine's per-token timestamps
                // (families with alignments — Paraformer CIF); None otherwise.
                let end_time_s: Option<f32> = item
                    .getattr("timestamps")
                    .ok()
                    .and_then(|x| x.extract::<Option<Vec<(f32, f32)>>>().ok())
                    .unwrap_or(None)
                    .and_then(|ts| ts.last().map(|&(_, end)| end));
                let finish_reason: Option<String> = item
                    .getattr("finish_reason")
                    .ok()
                    .and_then(|x| x.extract::<Option<String>>().ok())
                    .unwrap_or(None);
                Event::Final {
                    request_id: rid,
                    text,
                    tokens,
                    scores,
                    nbest_texts,
                    end_time_s,
                    finish_reason,
                }
            } else {
                Event::Partial {
                    request_id: rid,
                    text,
                    tokens,
                    scores,
                }
            };
            events.push(evt);
        }
        Ok(events)
    }
}

/// Build the Python-side `decoding` kwargs dict from [`DecodingParams`].
/// Returns `Ok(None)` when there are no params (or all fields are unset) so
/// callers skip the kwarg and the engine sees `decoding=None` — the fast
/// path stays allocation-free.
fn decoding_params_dict<'py>(
    py: Python<'py>,
    params: Option<&DecodingParams>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let p = match params {
        Some(p) if !p.is_empty() => p,
        _ => return Ok(None),
    };
    // Keys come from `DecodingParams::OPTION_KEYS`, the single source of truth
    // on this side, so a field added to the struct cannot quietly miss the dict
    // (`option_keys_cover_every_field` in oasr-wire enforces the match, and
    // `assert_decoding_option_keys_match` enforces it against Python).
    let d = PyDict::new_bound(py);
    for key in DecodingParams::OPTION_KEYS {
        match *key {
            "n_best" => {
                if let Some(v) = p.n_best {
                    d.set_item(key, v)?;
                }
            }
            "max_new_tokens" => {
                if let Some(v) = p.max_new_tokens {
                    d.set_item(key, v)?;
                }
            }
            "temperature" => {
                if let Some(v) = p.temperature {
                    d.set_item(key, v)?;
                }
            }
            "top_k" => {
                if let Some(v) = p.top_k {
                    d.set_item(key, v)?;
                }
            }
            "top_p" => {
                if let Some(v) = p.top_p {
                    d.set_item(key, v)?;
                }
            }
            "prompt" => {
                if let Some(v) = &p.prompt {
                    d.set_item(key, v.as_str())?;
                }
            }
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "DecodingParams::OPTION_KEYS names {other:?} but \
                     decoding_params_dict has no arm for it"
                )))
            }
        }
    }
    Ok(Some(d))
}

/// Cross-check the per-request option table against Python, once, at startup.
///
/// Silent drift is the failure mode S9 catalogued: a field added on one side
/// only makes requests carrying that option accepted and ignored, with nothing
/// logged at either end. One call turns that into a startup error naming the
/// keys that disagree.
pub fn assert_decoding_option_keys_match(py: Python<'_>) -> PyResult<()> {
    let request_mod = py.import_bound("oasr.engine.request")?;
    let options = request_mod.getattr("DecodingOptions")?;
    options.call_method1(
        "assert_matches_wire_keys",
        (DecodingParams::OPTION_KEYS.to_vec(),),
    )?;
    Ok(())
}

/// Decode raw little-endian f32 audio bytes into a writable numpy array on
/// the Python heap.  Mirrors the worker's `np.frombuffer(payload, ...).copy()`
/// fallback — the engine concatenates this with `audio_tail` and needs a
/// writable buffer.
fn audio_bytes_to_numpy<'py>(py: Python<'py>, audio: &[u8]) -> PyResult<Bound<'py, PyArray1<f32>>> {
    // The payload is contiguous little-endian f32 (already produced by the
    // front-end / `oasr-asr`).  Fill a freshly-allocated numpy array with a
    // single bulk memcpy into its (f32-aligned) backing store, instead of the
    // old per-element `from_le_bytes` decode + second copy through a Vec —
    // that loop dominated the dispatcher's per-tick admit cost on offline
    // batches.  x86 is little-endian, so the source byte layout matches the
    // destination; ragged tail bytes (len % 4 != 0) are dropped, mirroring
    // `np.frombuffer`.
    let elem = std::mem::size_of::<f32>();
    let n = audio.len() / elem;
    // SAFETY: the array is left uninitialized only until the copy below fills
    // every one of its `n * elem` bytes.  `f32` has no drop glue, so the early
    // return on the (unreachable) contiguity error frees it safely too.
    let arr = unsafe { PyArray1::<f32>::new_bound(py, n, false) };
    {
        // Contiguity is an invariant of a freshly allocated 1-D array, but it
        // comes back as a `Result` and this runs on the **dispatcher's OS
        // thread** — the one thread that owns the GIL and the engine.  An
        // `expect` here would unwind, drop the command receiver, and kill the
        // single GPU worker for the life of the process (every later submit
        // failing `WorkerDown`) without crashing the process or logging a
        // cause: a silent one-way failure. Propagating turns a violated
        // invariant into one failed request.
        // SAFETY: no other reference to `arr` exists; the slice is dropped
        // before `arr` is handed back.
        let dst = unsafe { arr.as_slice_mut() }.map_err(|e| {
            PyRuntimeError::new_err(format!("fresh 1-D numpy array is not contiguous: {e}"))
        })?;
        // SAFETY: `dst` is `n` contiguous, f32-aligned elements owned by a
        // freshly allocated array; `audio` is a distinct allocation of at
        // least `n * elem` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(audio.as_ptr(), dst.as_mut_ptr().cast::<u8>(), n * elem);
        }
    }
    Ok(arr)
}

fn collect_model_info(
    _py: Python<'_>,
    cfg: &Bound<'_, PyAny>,
    engine: &Bound<'_, PyAny>,
) -> PyResult<ModelInfo> {
    /// `obj.attr` as `T`, or the type's default when absent / not convertible.
    fn attr_or_default<T: Default + for<'p> pyo3::FromPyObject<'p>>(
        obj: &Bound<'_, PyAny>,
        name: &str,
    ) -> T {
        obj.getattr(name)
            .ok()
            .and_then(|x| x.extract::<T>().ok())
            .unwrap_or_default()
    }

    let vocab_size = match cfg.getattr("_model_config") {
        Ok(mc) if !mc.is_none() => mc
            .getattr("vocab_size")
            .ok()
            .and_then(|x| x.extract::<u32>().ok()),
        _ => None,
    };

    Ok(ModelInfo {
        ckpt_dir: attr_or_default(cfg, "ckpt_dir"),
        device: attr_or_default(cfg, "device"),
        dtype: cfg.getattr("dtype").ok().map(|x| format!("{x}")),
        chunk_size: attr_or_default(cfg, "chunk_size"),
        max_batch_size: attr_or_default(cfg, "max_batch_size"),
        decoder_type: attr_or_default(cfg, "decoder_type"),
        vocab_size,
        // Read the *resolved* mode / family off the engine, not the config: the
        // engine validates `decode_method` against the checkpoint's capabilities
        // and rejects offline-only families in streaming mode, so it is the
        // authority on what this process can actually serve.
        service_mode: attr_or_default(engine, "service_mode"),
        decode_method: attr_or_default(engine, "decode_method"),
        capabilities: attr_or_default(engine, "capabilities"),
        // Also engine-authoritative: `feature_config` is materialized from the
        // checkpoint's `FeatureSpec` during construction unless the caller pinned
        // one, so the CLI/JSON config is not a reliable source.
        sample_rate: attr_or_default(engine, "sample_rate"),
    })
}

// Engine-level errors thrown by the Python side surface as Event::Error via
// the dispatcher; this helper lets the dispatcher convert a PyEngineError on
// admission into an Event so the client sees a real BUSY / INTERNAL code
// instead of a hard channel close.
pub fn engine_error_event(rid: &str, err: &PyEngineError) -> Event {
    Event::Error {
        request_id: rid.to_owned(),
        code: ErrorCode::Internal,
        message: format!("{err}"),
    }
}
