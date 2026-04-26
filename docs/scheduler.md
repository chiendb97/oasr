# Scheduler

The scheduler (`oasr/engine/scheduler.py`) is the admission-control and
batching policy module of the ASR engine. Each engine `step()` calls
`Scheduler.schedule()` exactly once; the returned `SchedulerOutput`
specifies which offline requests should be forwarded as a batch this step,
which streaming requests should be promoted from waiting to running, and
which streaming requests are currently active.

## 1. Purpose and Responsibilities

The scheduler:

1. **Holds two waiting queues** — one for streaming, one for offline — and
   one running pool of admitted streaming requests.
2. **Decides per step** which requests to admit, applying length-aware
   bucketing for offline batches and a cohort-admission policy for
   streaming.
3. **Bounds starvation** via `max_wait_time`: any request that has waited
   too long becomes a forced-flush anchor regardless of policy.
4. **Manages priorities** so urgent requests bypass FIFO ordering.
5. **Provides O(1) request lookup** so the engine's `feed_chunk` does not
   pay an O(N) scan per audio chunk on backlog workloads.

The scheduler is *purely* a policy and bookkeeping object — it never
allocates GPU memory, never runs feature extraction, and never touches the
model.

## 2. High-Level Architecture

```
                add_request(req)
                       │
        ┌──────────────┴───────────────┐
   streaming?                       offline?
        ▼                              ▼
  _streaming_waiting (deque)      _offline_waiting (deque)
        │                              │
        │              schedule()      │
        │       ┌──────────┴──────┐    │
        │       │                 │    │
        ▼       ▼                 ▼    ▼
   admit()  build_offline_batch()      ─── forced flush gate
        │       │                 │
        ▼       ▼
  _running   offline_batch
  (OrderedDict<id, Request>)    (List[Request])

  schedule() → SchedulerOutput {running_streams, newly_admitted, offline_batch}
```

`_streaming_waiting`, `_offline_waiting`, and `_running` are kept consistent
via `_index: Dict[str, Request]` so `find_request(rid)` and `abort_request(rid)`
are O(1) regardless of which queue the request is in.

## 3. Internal Structure

### Data structures

| Field | Type | Role |
|-------|------|------|
| `_streaming_waiting` | `Deque[Request]` | Streaming requests not yet admitted. Priority-aware insertion; FIFO within the same priority. |
| `_offline_waiting` | `Deque[Request]` | Offline requests awaiting a batched forward pass. |
| `_running` | `OrderedDict[str, Request]` | Admitted streaming requests with allocated KV cache. Iteration order = admission order. |
| `_index` | `Dict[str, Request]` | O(1) lookup across all three queues. Kept in sync on every add / finish / abort. |
| `_next_stream_id` | `int` | Monotonic stream ID counter assigned during streaming admission. |

### `SchedulerOutput`

```python
@dataclass
class SchedulerOutput:
    running_streams: List[Request]   # all currently-active streaming requests
    newly_admitted:  List[Request]   # promoted to RUNNING this step
    offline_batch:   List[Request]   # one length-bucketed offline batch
```

The engine's step loop reads:

- `offline_batch` → forwarded through `OfflinePipeline.run` and finalised.
- `newly_admitted` (streaming subset) → `ModelRunner.allocate_stream`.
- `running_streams` → drives the per-step fbank-extract / forward / decode
  passes; the engine, not the scheduler, decides per request whether to
  run feature extraction, an encoder chunk, or finalisation.

## 4. Core Algorithms

### 4.1 Offline batching: `_build_offline_batch`

```
q = _offline_waiting
if q empty: return []
cap = max(1, max_offline_batch_size)
policy = config.schedule_policy

force_flush = q[0].waited_for >= max_wait_time

if policy == "fcfs":
    return [popleft() up to cap]

if policy == "sjf" and not force_flush:
    sort q by (priority, num_frames)        # ascending length

# "bucket" (default) and "sjf" share length-aware selection:
anchor = q.popleft()                         # oldest (or shortest if SJF)
batch  = [anchor]
min_len, max_len = anchor.num_frames, anchor.num_frames

if force_flush:
    return _fill_batch_fifo(batch, q, cap)   # strict FIFO refill

for cand in q:
    new_min = min(min_len, cand.num_frames)
    new_max = max(max_len, cand.num_frames)

    # bucket guard: candidates within length_bucket_ratio of anchor
    if length_bucket_ratio > 0 and new_min/new_max < length_bucket_ratio:
        continue
    # pad-waste guard: reject if padded/useful would exceed max_offline_pad_ratio
    useful = sum(r.num_frames for r in batch) + cand.num_frames
    padded = new_max * (len(batch) + 1)
    if max_offline_pad_ratio > 0 and padded/useful > max_offline_pad_ratio:
        continue

    batch.append(cand)
    min_len, max_len = new_min, new_max
    if len(batch) == cap: break
return batch
```

Two independent guards bound padded-compute waste:

- **`length_bucket_ratio`** — soft floor on `min_len / max_len` within a
  batch. Disabled (`0`) by default because splitting one bursty
  `transcribe(list_of_N)` call into sub-batches multiplies CPU
  feature-extraction cost while saving only a few percent of GPU
  compute on real workloads with moderate length spread.
- **`max_offline_pad_ratio`** — hard cap on `(max_len × B) / sum_len`.
  Default `4.0` admits LJSpeech-scale spreads in one batch but rejects
  pathological mixes of ~1 s and ~30 s clips.

### 4.2 Streaming admission: `_admit_streaming`

```
budget = max_batch_size - len(_running)
if budget <= 0: return []

if streaming_cohort_admit and _running:
    # Only admit when all running streams are still at offset 0;
    # otherwise the new stream would lag behind the cohort and break
    # the lockstep that makes _forward_batched_paged efficient.
    if any(r.offset != 0 for r in _running.values()):
        return []

if streaming_cohort_admit or policy == "sjf":
    sort _streaming_waiting by (priority, num_frames)

admitted = []
while _streaming_waiting and len(admitted) < budget:
    req = _streaming_waiting.popleft()
    req.stream_id = _next_stream_id; _next_stream_id += 1
    req.state = RUNNING
    admitted.append(req)
return admitted
```

### 4.3 Cohort admission rationale

Streaming throughput is dominated by `_forward_batched_paged`, which
fuses up to `B = max_batch_size` per-stream encoder calls into one
batched forward. That fusion is only possible when all `B` streams
share `(offset, window_size)`. Cohort admission enforces this:

- If the running pool is empty, anything goes — the next admitted
  cohort defines the offset.
- Otherwise the gate stays closed until every running stream has
  reached the end of its audio (after which it is finalised and
  removed). When the pool empties, the next batch admits in lockstep
  again.

The cost is brief GPU idle time during cohort transitions; the win is
per-step launch-amortised forwards across `B` streams. On backlog
workloads this is the single largest streaming throughput knob.

When `streaming_cohort_admit=False`, admission falls back to the
configured `schedule_policy` ordering (FCFS / bucket / SJF) and each
freed slot is filled immediately — better tail latency for the *next*
request, worse aggregate throughput.

### 4.4 Priority handling

Both queues use `_insert_ordered` on `add_request`:

- `priority == 0` (default) → `append` (O(1)).
- Higher-priority requests (lower numeric value) are inserted before
  the first slot whose priority is strictly worse — O(N) but queues
  are expected to stay small (< 10³).

Priority is a stable secondary key in `_sort_by_length`: SJF / cohort
sorting orders by `(priority, num_frames)` so high-priority work is
scheduled first within each length bucket.

### 4.5 Forced flush (starvation bound)

```
force_flush = q[0].waited_for >= cfg.max_wait_time
```

When the oldest offline request has been waiting too long:

- Skip the SJF re-sort (it would push this request back).
- Skip bucket and pad-waste guards — fill the batch via FIFO from the
  current head.

This guarantees that under sustained heavy load no request waits longer
than ~`max_wait_time + step_duration`. The default `max_wait_time=0.2s`
keeps p99 latency bounded while still letting normal-load batches
collect ideal length-similar peers.

## 5. Data Flow

```
                add_request                 step()
                     │                        │
                     ▼                        ▼
              ┌──────────────┐         ┌─────────────┐
   streaming → _streaming_w │         │ schedule()  │
   offline   → _offline_w   │ ──────▶ │             │
              │ _running    │         │             │
              │ _index      │         └──────┬──────┘
              └──────────────┘                │
                                              ▼
                                      SchedulerOutput
                                              │
                                              ▼
                                  ┌─────────────────────┐
                                  │  Engine step loop   │
                                  │  (allocate / fbank / │
                                  │   forward / decode) │
                                  └──────────┬──────────┘
                                             │
                                             ▼
                              finish_request | abort_request
                                             │
                                             ▼
                                   _running.pop, _index.pop
```

## 6. Configuration Options

All scheduler-relevant knobs live on `EngineConfig`:

| Option | Default | Effect |
|--------|---------|--------|
| `max_batch_size` | 32 | Cap on concurrent running streaming requests. |
| `max_offline_batch_size` | 1024 | Max offline requests admitted per step. The pipeline then splits this into `offline_micro_batch_size` micro-batches internally. |
| `length_bucket_ratio` | 0.0 | Soft floor on `min_len/max_len` inside a bucket. `0` disables. |
| `max_offline_pad_ratio` | 4.0 | Hard cap on `(max_len × B) / sum_len`. `0` disables. |
| `max_wait_time` | 0.2 s | Starvation bound: oldest offline request triggers forced flush. |
| `schedule_policy` | `"bucket"` | One of `"fcfs"`, `"bucket"`, `"sjf"`. |
| `streaming_cohort_admit` | `True` | Gate streaming admission on cohort offset alignment. |

### Choosing a policy

| Workload | Recommended policy |
|----------|-------------------|
| Ordering matters (live transcription, real-time pipelines) | `fcfs` |
| Mixed length, throughput-sensitive (default) | `bucket` |
| Heavy backlog of mixed lengths, latency-tolerant for outliers | `sjf` |

`sjf` can starve long requests indefinitely without `max_wait_time` —
keep it ≤ a few hundred ms unless you explicitly want long-tail
latency.

## 7. Usage Examples

The scheduler is owned by `ASREngine`; users do not construct it
directly. Direct interaction is mostly useful in tests or in a custom
engine.

### 7.1 Inside the engine step loop

```python
sched: SchedulerOutput = self._scheduler.schedule()

for req in sched.newly_admitted:
    if req.streaming:
        self._model_runner.allocate_stream(req)

if sched.offline_batch:
    outputs.extend(self._run_offline_batch(sched.offline_batch))

for req in sched.running_streams:
    # extract, forward, decode, finalise...
```

### 7.2 Aborting a request

```python
req = engine._scheduler.abort_request(rid)
if req is not None and req.stream_context is not None:
    engine._model_runner.free_stream(req)
```

### 7.3 Looking up a request by id (e.g. for `feed_chunk`)

```python
req = self._scheduler.find_request(request_id)   # O(1)
if req is None:
    raise KeyError(f"unknown or finished request_id {request_id!r}")
self._input_processor.append_streaming_chunk(req, chunk, is_last=is_last)
```

### 7.4 Custom direct usage (test scenarios)

```python
sched = Scheduler(EngineConfig(max_batch_size=4, schedule_policy="sjf"))
for r in requests:
    sched.add_request(r)

while sched.has_pending():
    out = sched.schedule()
    # ... drive forward, then mark finished:
    for req in finished:
        sched.finish_request(req.request_id)
```

## 8. Error Handling and Edge Cases

| Condition | Behaviour |
|-----------|-----------|
| `finish_request(rid)` for an id not in `_running` | `KeyError` from `OrderedDict.pop`. The engine ensures finishes only target running streams. |
| `abort_request(rid)` for an unknown id | Returns `None` — caller decides whether to raise. |
| Empty waiting queues | `schedule()` returns an empty `SchedulerOutput` (no offline batch, no admissions). `running_streams` may still be non-empty. |
| `max_batch_size == len(_running)` | No streaming admission this step (`budget <= 0`). |
| Cohort-admit gate blocking | Empty `newly_admitted` list with non-zero waiting queue — recover when running pool drains. |
| `length_bucket_ratio=0` and `max_offline_pad_ratio=0` | Effectively disables bucketing; behaves like FCFS within each step but still respects cap. |
| Forced flush fires | All bucket guards skipped; the resulting batch may be highly padded. |
| Priority insert into already long queue | O(N) walk, but priority is rare and queues are small. |

## 9. Performance Considerations

1. **`add_request` is O(1)** when `priority == 0` (default), O(N) only
   for non-default priorities.
2. **`schedule()` is O(N_offline)** per call where N_offline is the
   waiting queue length, bounded by `max_offline_batch_size`. It does
   one `_sort_by_length` (O(N log N)) for SJF or cohort admission with
   length-similar streams.
3. **`find_request` / `abort_request` are O(1)** thanks to `_index`.
4. **No GPU work.** The scheduler is pure Python on dataclasses; it is
   rarely the bottleneck. Profile with NVTX (`engine.step → schedule`)
   to confirm.
5. **Cohort admission has a measurable throughput effect**: on backlog
   workloads, enabling it doubles streaming throughput because
   `_forward_batched_paged` can take the full `B` path on every step.
   On low-concurrency or interactive workloads it adds idle time at
   cohort boundaries — measure both on your traffic before deciding.
6. **Avoid huge `max_offline_batch_size`** if the GPU cannot keep up —
   the scheduler will admit them, the pipeline will queue them, and
   memory pressure rises. Pair admission size with
   `offline_micro_batch_size × offline_pipeline_depth`.

## 10. Extension Points

- **Custom policy.** Add a new `schedule_policy` value and branch in
  `_build_offline_batch` / `_admit_streaming`. Keep the `force_flush`
  invariant so starvation stays bounded.
- **Custom priorities.** Replace `_insert_ordered` with a heap if you
  need many priority levels and large queues. Keep the `_index`
  invariant — `feed_chunk` depends on it.
- **Multi-tenant fairness.** Add a per-tenant `Request` field and group
  the offline batch by tenant before length bucketing. Cohort admission
  could be scoped per tenant similarly.
- **Preemption.** The current scheduler has no preempt path: if the
  block pool runs out, the engine fails. A real preemption policy would
  swap a low-priority running stream out of `_running`, free its
  blocks, and reinsert it into `_streaming_waiting` — implementable
  here without touching the engine if you also expose a hook for
  releasing the cache.

## 11. Quick Reference

```text
add_request(req)        # route to streaming/offline waiting queue
schedule() → output     # one-step decision
finish_request(rid)     # running → finished, drop from index
abort_request(rid)      # remove from any queue, return req or None
find_request(rid)       # O(1) lookup across all queues
has_pending()           # any waiting or running? (engine loop guard)
num_waiting / num_waiting_streaming / num_waiting_offline / num_running
```
