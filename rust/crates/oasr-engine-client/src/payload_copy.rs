// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Copying admitted audio into the engine's buffers, off the GIL.
//!
//! Every offline admission copies its waveform (little-endian f32 bytes from
//! the front-end) into the buffer the engine offered for it: page-locked when
//! the engine can DMA from it, a fresh numpy array otherwise.  The copy touches
//! no Python object, but it used to run inside the dispatcher tick that holds
//! the GIL, on one thread below 4 MiB a tick and on eight threads spawned for
//! the occasion above it.
//!
//! [`CopyInFlight::start`] cuts the payloads into chunks and wakes helpers from
//! the runtime's blocking pool, which start copying at once.  The dispatcher
//! meanwhile builds the admission specs under the GIL, then joins the copy
//! with [`CopyInFlight::finish`], with the GIL released for a large one.  The
//! caller copies too, so a helper that is slow to wake only means the caller
//! does more of the work: it never waits for a chunk nobody has claimed.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A destination pointer that may cross into a copy thread.
#[derive(Clone, Copy)]
pub(crate) struct DstPtr(pub(crate) *mut f32);

// SAFETY: each pointer addresses a distinct, freshly allocated buffer that only
// its own copy job writes, and the copy is finished before the buffer is handed
// to Python (see `CopyInFlight`).
unsafe impl Send for DstPtr {}
unsafe impl Sync for DstPtr {}

/// One payload copy: `src` (little-endian f32 bytes) into `dst`, which holds
/// exactly `src.len() / 4` elements.  Copied byte for byte: x86 is
/// little-endian, so the bytes are the floats.
pub(crate) struct PayloadCopy<'a> {
    pub(crate) dst: DstPtr,
    pub(crate) src: &'a [u8],
}

/// The unit a thread claims: 256 KiB is 4 s of 16 kHz audio, a few tens of
/// microseconds of copying, so the tail one thread can leave the others waiting
/// on stays short.
const CHUNK_BYTES: usize = 256 << 10;
/// One helper per this many bytes past the first: a helper takes some tens of
/// microseconds to wake, which a smaller share does not repay.
const BYTES_PER_HELPER: usize = 1 << 20;
/// Helpers per batch, besides the caller.  A memcpy into pinned pages measured
/// ~6 GB/s on one thread; a handful of threads reaches memory bandwidth.
const MAX_HELPERS: usize = 7;
/// Below this a copy is not worth giving the GIL up for: it costs less than
/// handing the GIL to another thread and waiting to get it back.
pub(crate) const RELEASE_GIL_MIN_BYTES: usize = 1 << 20;

/// One contiguous piece of one payload.
struct Chunk {
    src: *const u8,
    dst: *mut u8,
    len: usize,
}

/// The chunks of one admission batch, claimed in order by whichever thread
/// asks next.
struct Batch {
    chunks: Vec<Chunk>,
    /// Index of the next unclaimed chunk; past the end once all are claimed.
    next: AtomicUsize,
    /// Chunks copied.
    done: AtomicUsize,
}

// SAFETY: the pointers are only dereferenced for a claimed chunk, every chunk is
// claimed exactly once, and `CopyInFlight` does not return (or drop) until
// every claimed chunk is copied.  A helper that starts after that finds no
// chunk left to claim and touches nothing.
unsafe impl Send for Batch {}
unsafe impl Sync for Batch {}

impl Batch {
    /// The next unclaimed chunk, now this thread's alone; `None` once every
    /// chunk is claimed.
    fn claim(&self) -> Option<&Chunk> {
        self.chunks.get(self.next.fetch_add(1, Ordering::Relaxed))
    }

    /// Copy a chunk this thread claimed, and count it done.
    fn copy(&self, c: &Chunk) {
        // SAFETY: `c` was claimed by this thread alone; its source and
        // destination are distinct live allocations of at least `len` bytes
        // (see `CopyInFlight::start`).
        unsafe { std::ptr::copy_nonoverlapping(c.src, c.dst, c.len) };
        self.done.fetch_add(1, Ordering::Release);
    }

    /// Claim and copy chunks until none is left.
    fn work(&self) {
        while let Some(c) = self.claim() {
            self.copy(c);
        }
    }

    fn finished(&self) -> bool {
        self.done.load(Ordering::Acquire) == self.chunks.len()
    }
}

/// A copy that has been started and must be finished before its destinations
/// are handed to Python or freed.
///
/// Borrows the sources for `'a`.  The destinations are the caller's to keep
/// alive: create them before this, so they drop after it.  Dropping an
/// unfinished copy finishes it, so an early return cannot leave a helper
/// writing into a buffer that is gone.
pub(crate) struct CopyInFlight<'a> {
    batch: Option<Arc<Batch>>,
    bytes: usize,
    _src: PhantomData<&'a [u8]>,
}

impl<'a> CopyInFlight<'a> {
    /// Start copying every job, with helpers from the current runtime's
    /// blocking pool when the batch is large enough to repay them.
    ///
    /// No runtime (a test, a caller outside the server) means no helpers: the
    /// whole copy then runs in [`Self::finish`].
    pub(crate) fn start(jobs: &[PayloadCopy<'a>]) -> Self {
        let copy = Self::plan(jobs);
        let batch = copy.batch.as_ref().expect("a planned copy has its batch");
        let helpers = helpers_for(copy.bytes, batch.chunks.len());
        if helpers > 0 {
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                for _ in 0..helpers {
                    let b = Arc::clone(batch);
                    // Detached: a helper the pool runs late finds nothing left.
                    drop(rt.spawn_blocking(move || b.work()));
                }
            }
        }
        copy
    }

    /// Cut every job into chunks, copying nothing yet.
    fn plan(jobs: &[PayloadCopy<'a>]) -> Self {
        let bytes: usize = jobs.iter().map(|j| j.src.len()).sum();
        let mut chunks = Vec::with_capacity(bytes / CHUNK_BYTES + jobs.len());
        for job in jobs {
            let dst = job.dst.0.cast::<u8>();
            for (k, piece) in job.src.chunks(CHUNK_BYTES).enumerate() {
                chunks.push(Chunk {
                    src: piece.as_ptr(),
                    // SAFETY: `dst` holds `src.len()` bytes (the caller's
                    // contract), so offset `k * CHUNK_BYTES` stays within it.
                    dst: unsafe { dst.add(k * CHUNK_BYTES) },
                    len: piece.len(),
                });
            }
        }
        let batch = Arc::new(Batch {
            chunks,
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
        });
        Self {
            batch: Some(batch),
            bytes,
            _src: PhantomData,
        }
    }

    /// Total bytes this copy moves.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    /// Copy whatever no helper has claimed yet, then wait for the chunks the
    /// helpers did claim.  Every destination is filled when this returns.
    pub(crate) fn finish(mut self) {
        self.join();
    }

    fn join(&mut self) {
        let Some(batch) = self.batch.take() else {
            return;
        };
        batch.work();
        // Only chunks a helper is copying right now remain, each a few tens of
        // microseconds of work.
        let mut spins = 0u32;
        while !batch.finished() {
            if spins < 64 {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

impl Drop for CopyInFlight<'_> {
    fn drop(&mut self) {
        self.join();
    }
}

/// Helpers worth waking for `bytes` cut into `chunks`.
fn helpers_for(bytes: usize, chunks: usize) -> usize {
    (bytes / BYTES_PER_HELPER)
        .min(MAX_HELPERS)
        .min(chunks.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Payloads of awkward lengths: chunk boundaries inside a payload, a payload
    /// shorter than a chunk, and an empty one.
    fn payloads() -> Vec<Vec<u8>> {
        let lens = [
            0usize,
            4,
            CHUNK_BYTES - 4,
            CHUNK_BYTES,
            CHUNK_BYTES + 4,
            3 * CHUNK_BYTES + 12,
            5 << 20,
        ];
        lens.iter()
            .enumerate()
            .map(|(i, &n)| (0..n).map(|k| (k * 31 + i * 7) as u8).collect())
            .collect()
    }

    fn copy_all(rt: Option<&tokio::runtime::Runtime>) {
        let srcs = payloads();
        let mut dsts: Vec<Vec<f32>> = srcs.iter().map(|s| vec![-1.0; s.len() / 4]).collect();
        {
            let jobs: Vec<PayloadCopy<'_>> = srcs
                .iter()
                .zip(dsts.iter_mut())
                .map(|(s, d)| PayloadCopy {
                    dst: DstPtr(d.as_mut_ptr()),
                    src: s,
                })
                .collect();
            let _guard = rt.map(|r| r.enter());
            let copy = CopyInFlight::start(&jobs);
            assert_eq!(copy.bytes(), srcs.iter().map(Vec::len).sum::<usize>());
            copy.finish();
        }
        for (s, d) in srcs.iter().zip(&dsts) {
            let got: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
            assert_eq!(&got, s);
        }
    }

    #[test]
    fn every_byte_lands_without_helpers() {
        copy_all(None);
    }

    #[test]
    fn every_byte_lands_with_helpers() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .unwrap();
        for _ in 0..20 {
            copy_all(Some(&rt));
        }
    }

    /// Dropped without `finish` (an early return): the copy still completes
    /// before the destinations can go.
    #[test]
    fn dropping_an_unfinished_copy_completes_it() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .unwrap();
        let src: Vec<u8> = (0..(8usize << 20)).map(|k| k as u8).collect();
        let mut dst = vec![0f32; src.len() / 4];
        {
            let _guard = rt.enter();
            let jobs = [PayloadCopy {
                dst: DstPtr(dst.as_mut_ptr()),
                src: &src,
            }];
            let copy = CopyInFlight::start(&jobs);
            drop(copy);
        }
        let got: Vec<u8> = dst.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(got, src);
    }

    /// A helper still copying the chunk it claimed when the caller runs out of
    /// chunks to claim: `finish` must wait for it, or the destination is handed
    /// to Python half written.  Deterministic, unlike a race with real helpers,
    /// which mostly wake after the caller has claimed everything.
    #[test]
    fn finish_waits_for_a_chunk_a_helper_is_still_copying() {
        let src: Vec<u8> = (0..4 * CHUNK_BYTES).map(|k| (k % 251) as u8).collect();
        let mut dst = vec![0f32; src.len() / 4];
        {
            let jobs = [PayloadCopy {
                dst: DstPtr(dst.as_mut_ptr()),
                src: &src,
            }];
            let copy = CopyInFlight::plan(&jobs);
            let batch = Arc::clone(copy.batch.as_ref().unwrap());
            let (claimed_tx, claimed_rx) = std::sync::mpsc::channel();
            let helper = std::thread::spawn(move || {
                let c = batch.claim().expect("a chunk to claim");
                claimed_tx.send(()).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(100));
                batch.copy(c);
            });
            claimed_rx.recv().unwrap();
            copy.finish();
            // Read before the helper is joined, asserted after: a failure must
            // not unwind past a thread still writing into `dst`.
            let got: Vec<u8> = dst.iter().flat_map(|x| x.to_le_bytes()).collect();
            helper.join().unwrap();
            assert_eq!(got, src, "finish returned before the helper's chunk landed");
        }
    }

    #[test]
    fn small_batches_wake_no_helper() {
        assert_eq!(helpers_for(BYTES_PER_HELPER - 1, 100), 0);
        assert_eq!(helpers_for(3 * BYTES_PER_HELPER, 100), 3);
        assert_eq!(helpers_for(64 * BYTES_PER_HELPER, 100), MAX_HELPERS);
        // Never more helpers than chunks the caller would leave them.
        assert_eq!(helpers_for(64 * BYTES_PER_HELPER, 3), 2);
    }
}
