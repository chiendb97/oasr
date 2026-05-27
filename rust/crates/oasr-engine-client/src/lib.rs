// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! In-process driver for the OASR Python `ASREngine` via PyO3.
//!
//! - [`EngineClient`] owns one Python `ASREngine` and a dedicated OS thread
//!   that drives `engine.step()` while holding the GIL.  Async HTTP / gRPC
//!   handlers push commands across a tokio mpsc channel; the dispatcher
//!   thread runs them in order, then routes the resulting events back via
//!   per-request channels.
//! - [`EnginePool`] groups N clients (one per GPU) with sticky-by-rid
//!   routing for streaming and least-loaded routing for offline.
//!
//! The previous ZMQ + msgpack hop is gone: there is no separate Python
//! worker process, no DEALER socket, no wire codec.  Events are
//! constructed natively from `engine.step()` output.

pub mod client;
pub mod dispatcher;
pub mod handle;
pub mod pool;
pub mod pyengine;
pub mod router;

pub use client::{EngineClient, EngineClientConfig};
pub use dispatcher::CmdEnvelope;
pub use handle::{OfflineHandle, StreamingHandle};
pub use pool::EnginePool;
pub use pyengine::{PyEngine, PyEngineError};

use oasr_wire::Event;
use thiserror::Error;

/// Errors surfaced from the client / pool API.
#[derive(Debug, Error)]
pub enum EngineClientError {
    #[error("engine channel full; backpressure: {0}")]
    Backpressure(String),
    #[error("engine is down: {0}")]
    WorkerDown(String),
    #[error("unknown stream id: {0}")]
    UnknownStream(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("engine rejected request: {request_id} {message}")]
    Rejected { request_id: String, message: String },
    #[error("python: {0}")]
    Python(#[from] PyEngineError),
    #[error("{0}")]
    Other(String),
}

/// Per-request channel item: events from the engine for this rid.
pub type EventStream = tokio_stream::wrappers::ReceiverStream<Event>;
