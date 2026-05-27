// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Multi-worker pool with sticky-by-rid streaming routing.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use dashmap::DashMap;
use oasr_wire::Event;
use tracing::warn;
use uuid::Uuid;

use crate::client::EngineClient;
use crate::handle::{OfflineHandle, StreamingHandle};
use crate::EngineClientError;

/// Owned by `oasr-server`; shared across HTTP and gRPC stacks via `Arc<_>`.
pub struct EnginePool {
    workers: Vec<Arc<EngineClient>>,
    /// Streaming `request_id` → worker index.
    sticky: DashMap<String, usize>,
}

impl EnginePool {
    pub fn new(workers: Vec<Arc<EngineClient>>) -> Self {
        Self {
            workers,
            sticky: DashMap::new(),
        }
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
    ) -> Result<OfflineHandle, EngineClientError> {
        let k = self
            .pick_least_loaded()
            .ok_or_else(|| EngineClientError::WorkerDown("no healthy workers".into()))?;
        self.workers[k].submit_offline(audio, sample_rate, priority).await
    }

    /// Open a streaming request, sticky to the chosen worker for subsequent
    /// chunks.  The returned handle wraps the worker's `StreamingHandle`.
    pub async fn open_streaming(
        &self,
        sample_rate: u32,
        priority: i32,
    ) -> Result<StreamingHandle, EngineClientError> {
        let k = self
            .pick_least_loaded()
            .ok_or_else(|| EngineClientError::WorkerDown("no healthy workers".into()))?;
        let rid = Uuid::new_v4().simple().to_string();
        self.sticky.insert(rid.clone(), k);
        let handle = self.workers[k]
            .open_streaming_with_id(rid.clone(), sample_rate, priority)
            .await?;
        Ok(handle)
    }

    /// Remove the sticky entry for `rid` (typically after Final/Error).
    pub fn release(&self, rid: &str) {
        self.sticky.remove(rid);
    }

    /// On worker death, mark all rids belonging to it as failed and emit a
    /// synthetic [`Event::Error{code: WorkerLost}`] into their channels.
    pub async fn fail_worker(&self, worker_index: usize, reason: &str) {
        let to_kill: Vec<String> = self
            .sticky
            .iter()
            .filter(|kv| *kv.value() == worker_index)
            .map(|kv| kv.key().clone())
            .collect();
        if to_kill.is_empty() {
            return;
        }
        warn!(
            worker = worker_index,
            n = to_kill.len(),
            "fanning WORKER_LOST to in-flight streams: {reason}"
        );
        let router = self.workers[worker_index].router().clone();
        for rid in to_kill {
            router
                .route(Event::Error {
                    request_id: rid.clone(),
                    code: oasr_wire::ErrorCode::WorkerLost,
                    message: reason.to_owned(),
                })
                .await;
            self.sticky.remove(&rid);
        }
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
