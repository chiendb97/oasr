// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Multi-worker pool: least-loaded admission, and a request stays where it was
//! admitted.
//!
//! "Stays" needs no table.  A handle carries the command sender of the worker
//! that admitted it, and that worker's router holds its event channel.  The
//! pool used to keep a `request_id -> worker` map besides, which nothing read:
//! entries went in on every streaming open and came out only when a front-end
//! reached its normal-completion `release` call, so every error, timeout and
//! disconnect left one behind for the life of the process.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use oasr_wire::DecodingParams;

use crate::client::EngineClient;
use crate::handle::{OfflineHandle, OfflineStreamHandle, StreamingHandle};
use crate::EngineClientError;

/// Owned by `oasr-server`; shared across HTTP and gRPC stacks via `Arc<_>`.
pub struct EnginePool {
    workers: Vec<Arc<EngineClient>>,
}

impl EnginePool {
    pub fn new(workers: Vec<Arc<EngineClient>>) -> Self {
        Self { workers }
    }

    pub fn workers(&self) -> &[Arc<EngineClient>] {
        &self.workers
    }

    /// Number of workers in the pool.
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Pick the least-loaded healthy worker.  Returns `None` if all are down.
    pub fn pick_least_loaded(&self) -> Option<usize> {
        let mut best: Option<(usize, u32)> = None;
        for (i, w) in self.workers.iter().enumerate() {
            // Skip workers that haven't ponged in 10s (best-effort liveness).
            if let Some(t) = w.last_pong_at() {
                if t.elapsed() > Duration::from_secs(10) {
                    continue;
                }
            } else {
                // No pong yet — accept it as healthy for the first ~30s of life.
                // Production code should require a successful ping before the
                // worker is added to the pool.
            }
            let load = w.load();
            match best {
                None => best = Some((i, load)),
                Some((_, b)) if load < b => best = Some((i, load)),
                _ => {}
            }
        }
        best.map(|(i, _)| i)
    }

    /// Submit an offline request to the least-loaded worker.
    pub async fn submit_offline(
        &self,
        audio: Bytes,
        sample_rate: u32,
        priority: i32,
        decoding: Option<DecodingParams>,
    ) -> Result<OfflineHandle, EngineClientError> {
        let k = self
            .pick_least_loaded()
            .ok_or_else(|| EngineClientError::WorkerDown("no healthy workers".into()))?;
        self.workers[k]
            .submit_offline(audio, sample_rate, priority, decoding)
            .await
    }

    /// Submit an offline request whose *events* stream back incrementally, to
    /// the least-loaded worker.  The caller keeps pulling events from that
    /// worker through the handle.
    pub async fn submit_offline_streaming(
        &self,
        audio: Bytes,
        sample_rate: u32,
        priority: i32,
        decoding: Option<DecodingParams>,
    ) -> Result<OfflineStreamHandle, EngineClientError> {
        let k = self
            .pick_least_loaded()
            .ok_or_else(|| EngineClientError::WorkerDown("no healthy workers".into()))?;
        self.workers[k]
            .submit_offline_streaming(audio, sample_rate, priority, decoding)
            .await
    }

    /// Open a streaming request on the least-loaded worker.  Its chunks follow
    /// through the handle, which sends them to the worker that admitted it.
    pub async fn open_streaming(
        &self,
        sample_rate: u32,
        priority: i32,
        decoding: Option<DecodingParams>,
    ) -> Result<StreamingHandle, EngineClientError> {
        let k = self
            .pick_least_loaded()
            .ok_or_else(|| EngineClientError::WorkerDown("no healthy workers".into()))?;
        self.workers[k]
            .open_streaming(sample_rate, priority, decoding)
            .await
    }

    /// Stop every worker's dispatcher and wait, up to `timeout` in all, for
    /// their threads to exit.  Returns whether every one did.
    ///
    /// Each fails what it still has in flight with `SHUTDOWN` and releases its
    /// engine before exiting, so call this once the listeners have drained.
    pub async fn shutdown(&self, timeout: Duration) -> bool {
        for w in &self.workers {
            w.stop();
        }
        let joined =
            futures::future::join_all(self.workers.iter().map(|w| w.shutdown(timeout))).await;
        joined.into_iter().all(|ok| ok)
    }

    pub fn worker(&self, i: usize) -> Option<&Arc<EngineClient>> {
        self.workers.get(i)
    }

    /// First model info across workers (all should be identical when running
    /// the same checkpoint).
    pub fn model_info(&self) -> Option<oasr_wire::ModelInfo> {
        for w in &self.workers {
            if let Some(mi) = w.model_info() {
                return Some(mi);
            }
        }
        None
    }

    /// True if any worker has produced at least one Pong recently.
    pub fn any_ready(&self, recent: Duration) -> bool {
        self.workers.iter().any(|w| {
            w.last_pong_at()
                .map(|t| t.elapsed() <= recent)
                .unwrap_or(false)
        })
    }
}

// Re-export so the `oasr-server` binary doesn't need to depend on tokio.
pub use crate::dispatcher::CmdEnvelope as CmdEnvelopePub;

// Allow tests to introspect.
#[doc(hidden)]
pub fn _new_pool_for_test(workers: Vec<Arc<EngineClient>>) -> EnginePool {
    EnginePool::new(workers)
}
