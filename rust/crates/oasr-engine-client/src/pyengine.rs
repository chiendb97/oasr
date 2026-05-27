// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! PyO3 wrapper around `oasr.engine.ASREngine`.
//!
//! `PyEngine` is **not** clone-safe and is owned by the dispatcher thread.
//! It is constructed once at startup with the JSON engine config, and every
//! method call acquires the GIL.  The dispatcher serialises access by virtue
//! of being a single thread.

use bytes::Bytes;
use numpy::PyArray1;
use oasr_wire::{ErrorCode, Event, ModelInfo};
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
    },
    Streaming {
        rid: String,
        sample_rate: u32,
        priority: i32,
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

            let model_info = collect_model_info(py, &engine_cfg).unwrap_or_default();

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
            Self::load_locked(&bound)
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
    ) -> Result<(), PyEngineError> {
        Python::with_gil(|py| {
            let bound = self.engine.bind(py);
            Self::add_offline_locked(py, &bound, rid, audio, sample_rate, priority)
        })
    }

    /// GIL-already-held variant of [`add_offline`].
    pub fn add_offline_locked<'py>(
        py: Python<'py>,
        bound: &Bound<'py, PyAny>,
        rid: &str,
        audio: &[u8],
        sample_rate: u32,
        priority: i32,
    ) -> Result<(), PyEngineError> {
        let arr = audio_bytes_to_numpy(py, audio)?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("audio", arr)?;
        kwargs.set_item("request_id", rid)?;
        kwargs.set_item("sample_rate", sample_rate)?;
        kwargs.set_item("streaming", false)?;
        kwargs.set_item("priority", priority)?;
        bound.call_method("add_request", (), Some(&kwargs))?;
        Ok(())
    }

    pub fn add_streaming(
        &self,
        rid: &str,
        sample_rate: u32,
        priority: i32,
    ) -> Result<(), PyEngineError> {
        Python::with_gil(|py| {
            let bound = self.engine.bind(py);
            Self::add_streaming_locked(py, &bound, rid, sample_rate, priority)
        })
    }

    /// GIL-already-held variant of [`add_streaming`].
    pub fn add_streaming_locked<'py>(
        py: Python<'py>,
        bound: &Bound<'py, PyAny>,
        rid: &str,
        sample_rate: u32,
        priority: i32,
    ) -> Result<(), PyEngineError> {
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("request_id", rid)?;
        kwargs.set_item("sample_rate", sample_rate)?;
        kwargs.set_item("priority", priority)?;
        bound.call_method("add_streaming_request", (), Some(&kwargs))?;
        Ok(())
    }

    /// Bulk admission entry — calls `ASREngine.add_requests_batch(list)` in
    /// **one** Python method invocation across all `specs`.  GIL must already
    /// be held; intended to be called by the dispatcher inside its tick's
    /// `Python::with_gil` scope after draining contiguous admit envelopes.
    /// Returns `Ok(())` when the Python batch call succeeds — callers that
    /// need per-request error fan-out can call the single-shot
    /// `add_*_locked` helpers in a loop instead.
    pub fn add_requests_batch_locked<'py>(
        py: Python<'py>,
        bound: &Bound<'py, PyAny>,
        specs: &[AdmitSpec],
    ) -> Result<(), PyEngineError> {
        if specs.is_empty() {
            return Ok(());
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
                } => {
                    let arr = audio_bytes_to_numpy(py, audio)?;
                    d.set_item("audio", arr)?;
                    d.set_item("request_id", rid.as_str())?;
                    d.set_item("sample_rate", *sample_rate)?;
                    d.set_item("streaming", false)?;
                    d.set_item("priority", *priority)?;
                }
                AdmitSpec::Streaming {
                    rid,
                    sample_rate,
                    priority,
                } => {
                    d.set_item("request_id", rid.as_str())?;
                    d.set_item("sample_rate", *sample_rate)?;
                    d.set_item("streaming", true)?;
                    d.set_item("priority", *priority)?;
                }
            }
            list.append(d)?;
        }
        bound.call_method1("add_requests_batch", (list,))?;
        Ok(())
    }

    pub fn feed_chunk(
        &self,
        rid: &str,
        chunk: &[u8],
        is_last: bool,
    ) -> Result<(), PyEngineError> {
        Python::with_gil(|py| {
            let bound = self.engine.bind(py);
            Self::feed_chunk_locked(py, &bound, rid, chunk, is_last)
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
            Self::abort_locked(&bound, rid)
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
            Self::step_locked(py, &bound)
        })
    }

    /// GIL-already-held variant of [`step`].
    pub fn step_locked<'py>(
        _py: Python<'py>,
        bound: &Bound<'py, PyAny>,
    ) -> Result<Vec<Event>, PyEngineError> {
        let outputs = bound.call_method0("step")?;
        let list: Bound<'_, PyList> = outputs.downcast_into()?;
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
                Event::Final {
                    request_id: rid,
                    text,
                    tokens,
                    scores,
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

/// Decode raw little-endian f32 audio bytes into a writable numpy array on
/// the Python heap.  Mirrors the worker's `np.frombuffer(payload, ...).copy()`
/// fallback — the engine concatenates this with `audio_tail` and needs a
/// writable buffer.
fn audio_bytes_to_numpy<'py>(py: Python<'py>, audio: &[u8]) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let n = audio.len() / std::mem::size_of::<f32>();
    let samples = unsafe {
        std::slice::from_raw_parts(audio.as_ptr() as *const f32, n)
    };
    Ok(PyArray1::<f32>::from_slice_bound(py, samples))
}

fn collect_model_info(_py: Python<'_>, cfg: &Bound<'_, PyAny>) -> PyResult<ModelInfo> {
    let mut info = ModelInfo::default();
    info.ckpt_dir = cfg
        .getattr("ckpt_dir")
        .ok()
        .and_then(|x| x.extract::<Option<String>>().ok())
        .unwrap_or_default();
    info.device = cfg
        .getattr("device")
        .ok()
        .and_then(|x| x.extract::<Option<String>>().ok())
        .unwrap_or_default();
    info.dtype = cfg.getattr("dtype").ok().map(|x| format!("{x}"));
    info.chunk_size = cfg
        .getattr("chunk_size")
        .ok()
        .and_then(|x| x.extract::<u32>().ok());
    info.max_batch_size = cfg
        .getattr("max_batch_size")
        .ok()
        .and_then(|x| x.extract::<u32>().ok());
    info.decoder_type = cfg
        .getattr("decoder_type")
        .ok()
        .and_then(|x| x.extract::<Option<String>>().ok())
        .unwrap_or_default();
    if let Ok(mc) = cfg.getattr("_model_config") {
        if !mc.is_none() {
            info.vocab_size = mc
                .getattr("vocab_size")
                .ok()
                .and_then(|x| x.extract::<u32>().ok());
        }
    }
    Ok(info)
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
