// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Async client to the OASR Python engine worker over ZeroMQ + MessagePack.
//!
//! - [`EngineClient`] talks to one worker (one DEALER socket).  It owns a
//!   pair of background threads (reader/writer) and a [`router::RouterActor`]
//!   that demuxes incoming events back to per-request channels.
//! - [`EnginePool`] manages N [`EngineClient`]s (one per GPU worker) with
//!   sticky-by-request_id routing for streaming and least-loaded routing for
//!   offline.

pub mod client;
pub mod handle;
pub mod pool;
pub mod router;

pub use client::EngineClient;
pub use handle::{OfflineHandle, StreamingHandle};
pub use pool::EnginePool;

use oasr_wire::Event;
use thiserror::Error;

/// Errors surfaced from the client / pool API.
#[derive(Debug, Error)]
pub enum EngineClientError {
    #[error("worker channel full; backpressure: {0}")]
    Backpressure(String),
    #[error("worker is down: {0}")]
    WorkerDown(String),
    #[error("unknown stream id: {0}")]
    UnknownStream(String),
    #[error("wire error: {0}")]
    Wire(#[from] oasr_wire::WireError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("worker rejected request: {request_id} {message}")]
    Rejected { request_id: String, message: String },
    #[error("{0}")]
    Other(String),
}

/// Per-request channel item: events from the worker for this rid.
pub type EventStream = tokio_stream::wrappers::ReceiverStream<Event>;
