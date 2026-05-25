# Engine concurrency and scaling

## Thread safety

`oasr.engine.ASREngine` is **thread-safe** as of the v0.1 serving release.
The historical `OfflineEngine` subclass was removed — pass `streaming=False`
or use `transcribe_offline(...)` for the batched path.  Every public entry —

- `add_request`
- `add_streaming_request`
- `feed_chunk`
- `abort_request`
- `step`
- `run`
- `num_running` / `num_waiting` (properties)
- `transcribe` (composes the above)

— acquires a process-wide `threading.RLock` on the engine.  `run()` re-enters
to call `step()`, so the lock is re-entrant.  Uncontended cost is ~50 ns per
call, invisible next to GPU work.

### What the lock protects

Two pieces of mutable state had no guard before:

1. **Scheduler queues** — `_streaming_waiting`, `_offline_waiting`, `_running`,
   `_index` in `oasr/engine/scheduler.py:85-94`.
2. **Per-request audio** — `request.audio_chunks` (a `Deque[Tensor]`) and
   `request.audio_final` (a `bool`) at `oasr/engine/request.py:114-160`,
   mutated by `feed_chunk` (`oasr/engine/input_processor.py:602`) and
   concurrently read+popped by `step()` (`oasr/engine/input_processor.py:661`).

Both are now safe to access from any thread.

### What it does *not* fix

The lock is **coarse**: `step()` holds it for the full step duration
(10–100 ms, GPU-bound).  Concurrent `feed_chunk` calls from another thread
wait up to one step.  For Python serving workers this matters little because
the GIL serializes Python anyway and CUDA work releases the GIL — the lock
just makes the data-structure consistency explicit.

A finer-grained split (scheduler-only lock + per-request audio locks + drop
the lock during forward/decode) would trim that to a few microseconds.  It is
designed but deferred — the multi-process fleet (below) addresses the bigger
throughput concern.

## Is the single-threaded engine a bottleneck?

Short version: **no, for throughput; small for streaming tail latency; the
real scaling answer is multi-process.**

| Concern | Single engine | What to do |
|---|---|---|
| Throughput | Not a bottleneck — GPU-bound, batching already saturates compute. | Don't add threads inside one engine. |
| Streaming tail latency | Modest issue — `step()` blocks new chunk ingestion. | Use the worker's `--worker-threads 2` mode (overlaps ZMQ recv with `step()` under the coarse lock). |
| Horizontal scale | One engine = one GPU.  Stuck. | Run multiple worker processes, one per GPU. |

CPython's GIL means adding threads *inside* one Python process never
increases compute throughput.  CUDA calls release the GIL, so a separate
I/O thread *can* overlap ZMQ ingestion with `step()`'s GPU work; that's what
`oasr/serving/engine_worker.py::_run_two_thread` does.

## Multi-process scaling

The Rust serving frontend (`rust/`) supervises a fleet of N independent
Python workers — one per GPU — and routes:

- **offline** requests to the least-loaded worker;
- **streaming** requests with sticky affinity: the worker that admits a
  request keeps all subsequent chunks for that `request_id` until
  Final/Error/Cancel.

When a worker process dies, the supervisor surfaces
`Event::Error{code: WORKER_LOST}` to every in-flight stream pinned to that
worker and (eventually — v1 leaves auto-restart to the operator) spawns a
replacement.

This is the same pattern vLLM and sglang use: one scheduler per GPU, route at
the frontend.

## Validation

Concurrent-access tests live at `tests/test_engine_concurrent.py` and run
under the `concurrent` pytest mark:

```bash
pytest tests/test_engine_concurrent.py -m concurrent -v
```

Covered:

- 16-thread stress on `add_streaming_request` — no duplicate ids, scheduler
  index consistent.
- 8 feeders × 4 aborters racing on a shared rid pool — no dangling state.
- 4 writers + 4 readers on `num_running` / `num_waiting` — no exceptions, no
  negative counts.

An end-to-end test (`#[ignore]`) feeds 4 streaming requests from 4 threads
against a real engine driven by the main thread and asserts the transcripts
match the single-threaded baseline (gated on `--ckpt-dir` and `--wav-dir`).
