// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Whole-clip decodes off the async runtime.
//!
//! [`decode_audio`] is CPU work for as long as the clip is: an hour of MP3
//! decodes and resamples in about five seconds on one core.  Run inline in a
//! request handler it holds that tokio worker for the whole time, and with it
//! every health check, metrics scrape and streaming forward queued behind it, so
//! a few long uploads stalled the server's own probes.  [`DecodePool`] runs each
//! decode on tokio's blocking pool instead, behind a semaphore: a burst of long
//! uploads becomes a queue of waiting requests rather than one busy thread --
//! and one decoded waveform in memory -- per request.
//!
//! Not every decode goes there.  Raw PCM already at the target rate is one
//! linear pass over the body -- about 10 us for a typical short request --
//! while the hop to a blocking thread and back costs more than that alone, and
//! milliseconds at its tail under load.  Such a body runs inline up to
//! [`INLINE_MAX_SAMPLES`].

use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::Semaphore;

use crate::audio::{decode_audio, linear_pass_samples, AudioError, DecodeOptions, DecodedAudio};

/// Most samples [`DecodePool::decode`] decodes on the calling thread, when the
/// decode is a single pass over raw PCM with no rate conversion: 16 s at
/// 16 kHz, which covers most unary requests.  Counted in samples, not bytes,
/// because the encodings differ per sample: f32 and LINEAR16 widen at a fraction
/// of a nanosecond each, G.711 at about one, so this bound is tens of
/// microseconds of work for the former and a few hundred for the latter.
pub const INLINE_MAX_SAMPLES: usize = 1 << 18;

/// Runs [`decode_audio`] on tokio's blocking pool, at most
/// [`max_concurrent`](Self::max_concurrent) at a time.
///
/// Cheap to clone; clones share the bound, so one pool covers every surface of
/// a process.
#[derive(Clone, Debug)]
pub struct DecodePool {
    permits: Arc<Semaphore>,
    max_concurrent: usize,
}

impl DecodePool {
    /// A pool running at most `max_concurrent` decodes at once (at least one).
    pub fn new(max_concurrent: usize) -> Self {
        let max_concurrent = max_concurrent.max(1);
        Self {
            permits: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }

    /// A pool bounded by the number of CPUs this process may use: decoding is
    /// CPU-bound, so more at once only shares the same cores among more
    /// resident waveforms.
    pub fn per_cpu() -> Self {
        Self::new(std::thread::available_parallelism().map_or(1, |n| n.get()))
    }

    /// Decodes this pool runs at once.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// [`decode_audio`] off the calling task's worker thread, unless it is a
    /// short raw-PCM pass (see [`INLINE_MAX_SAMPLES`]), which runs in place.
    ///
    /// Waits for a slot first.  The slot is held by the decode, not by the
    /// caller, so a request dropped mid-decode (a client disconnect, a timeout)
    /// frees it only when that decode has actually finished: the bound counts
    /// work in progress, never just the requests still waiting for it.
    pub async fn decode(
        &self,
        body: Bytes,
        opts: &DecodeOptions<'_>,
    ) -> Result<DecodedAudio, AudioError> {
        if linear_pass_samples(&body, opts).is_some_and(|n| n <= INLINE_MAX_SAMPLES) {
            return decode_audio(&body, opts);
        }
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .map_err(|_| AudioError::DecodeTask("the decode pool is closed".into()))?;
        let hint = opts.hint.map(str::to_owned);
        let (encoding, source_sample_rate, target_sample_rate, max_samples) = (
            opts.encoding,
            opts.source_sample_rate,
            opts.target_sample_rate,
            opts.max_samples,
        );
        let task = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            decode_audio(
                &body,
                &DecodeOptions {
                    hint: hint.as_deref(),
                    encoding,
                    source_sample_rate,
                    target_sample_rate,
                    max_samples,
                },
            )
        });
        match task.await {
            Ok(result) => result,
            Err(e) => {
                // A decoder that panics on some input is a bug to fix, but it
                // is that request's failure: answer it, and keep the worker.
                let reason = if e.is_panic() {
                    panic_message(e.into_panic())
                } else {
                    e.to_string()
                };
                tracing::error!(%reason, "audio decode task failed");
                Err(AudioError::DecodeTask(reason))
            }
        }
    }
}

impl Default for DecodePool {
    fn default() -> Self {
        Self::per_cpu()
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(s) => *s,
        Err(payload) => payload
            .downcast_ref::<&str>()
            .map_or_else(|| "decoder panicked".to_owned(), |s| (*s).to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{PcmEncoding, SourceEncoding};

    fn pcm_i16(n: usize) -> Bytes {
        (0..n)
            .flat_map(|i| (((i % 200) as i16) * 100).to_le_bytes())
            .collect::<Vec<u8>>()
            .into()
    }

    fn opts(target: u32) -> DecodeOptions<'static> {
        DecodeOptions {
            encoding: SourceEncoding::Pcm(PcmEncoding::I16Le),
            source_sample_rate: Some(8_000),
            target_sample_rate: Some(target),
            ..Default::default()
        }
    }

    /// The pool is a different thread, not a different answer.
    #[tokio::test]
    async fn decodes_exactly_what_decode_audio_does() {
        let body = pcm_i16(8_000);
        let pool = DecodePool::new(2);
        for target in [8_000, 16_000] {
            let inline = decode_audio(&body, &opts(target)).unwrap();
            let pooled = pool.decode(body.clone(), &opts(target)).await.unwrap();
            assert_eq!(pooled.sample_rate, inline.sample_rate);
            assert_eq!(pooled.samples, inline.samples);
        }
        let err = pool
            .decode(body.slice(..3), &opts(16_000))
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::Misaligned(3, 2)), "{err:?}");
    }

    /// The decode must leave the worker that asked for it.  On a
    /// current-thread runtime there is exactly one worker, so a decode run
    /// inline would hold it until done and no other task could be polled.
    #[tokio::test(flavor = "current_thread")]
    async fn the_runtime_keeps_running_while_a_decode_does() {
        let pool = DecodePool::new(1);
        // Milliseconds of work in an optimized build, far more than a yield.
        let body = pcm_i16(8_000 * 20);
        let decode = tokio::spawn({
            let pool = pool.clone();
            async move { pool.decode(body, &opts(16_000)).await.map(|_| ()) }
        });
        // Let the decode start: it is the only other task, so one yield runs
        // it up to its first await.
        tokio::task::yield_now().await;
        // Count how often this task gets polled while the decode runs.
        let mut ticks = 0u32;
        while !decode.is_finished() {
            tokio::task::yield_now().await;
            ticks += 1;
        }
        decode.await.unwrap().unwrap();
        assert!(ticks > 0, "the decode ran on the runtime's only worker");
    }

    /// Whether `decode` finished on its first poll, i.e. without leaving the
    /// calling thread.
    fn ran_in_place(pool: &DecodePool, body: Bytes, opts: &DecodeOptions<'_>) -> bool {
        use std::future::Future;
        let mut fut = std::pin::pin!(pool.decode(body, opts));
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        fut.as_mut().poll(&mut cx).is_ready()
    }

    /// Only a short raw-PCM pass with no rate conversion stays on the caller's
    /// thread: the hop costs more than that work.  A resample, a container or a
    /// long body each goes to the pool.
    #[tokio::test]
    async fn only_a_short_raw_pass_runs_in_place() {
        let pool = DecodePool::new(4);
        let short = pcm_i16(8_000);
        assert!(ran_in_place(&pool, short.clone(), &opts(8_000)));
        assert!(!ran_in_place(&pool, short.clone(), &opts(16_000)));
        let long = pcm_i16(INLINE_MAX_SAMPLES + 1);
        assert!(!ran_in_place(&pool, long, &opts(8_000)));
        // The same samples behind a WAV header: a container, so pooled.
        let mut wav = b"RIFF\0\0\0\0WAVE".to_vec();
        wav.extend_from_slice(&short);
        assert!(!ran_in_place(&pool, wav.into(), &opts(8_000)));
    }

    #[test]
    fn a_pool_always_has_a_slot() {
        assert_eq!(DecodePool::new(0).max_concurrent(), 1);
        assert!(DecodePool::per_cpu().max_concurrent() >= 1);
    }

    #[test]
    fn a_panic_payload_keeps_its_message() {
        assert_eq!(panic_message(Box::new("boom")), "boom");
        assert_eq!(panic_message(Box::new(String::from("bang"))), "bang");
        assert_eq!(panic_message(Box::new(7u8)), "decoder panicked");
    }
}
