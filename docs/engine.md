# ASR Engine

The engine subsystem (`oasr/engine/`) is the top-level inference façade. It
takes raw audio in and produces transcripts out, hiding the complexity of
feature extraction, dynamic batching, paged KV-cache management, encoder
forward passes, and CTC decoding behind a small public API.

OASR's engine design is vLLM-inspired: a **single step loop** drives both
offline (single-pass batched) and streaming (chunk-by-chunk with paged KV
cache) work in one pool, with length-aware bucketing and CPU/GPU overlap.

## 1. Purpose and Responsibilities

`ASREngine` is responsible for:

1. **Loading the model and its config** from a WeNet-format checkpoint
   directory.
2. **Owning shared GPU resources** — the paged KV block pool, CNN cache
   manager, CTC state manager — through `ModelRunner`.
3. **Routing requests** through the scheduler into either the offline
   pipeline or the streaming step path.
4. **Driving the step loop**: schedule → ingest fbank → forward → decode
   → finalise.
5. **Exposing a high-level API**:
   - `transcribe(audio | List[audio])` for one-shot use.
   - `add_request` / `feed_chunk` / `step` / `run` for explicit control.

`OfflineEngine` is a thin subclass that defaults `transcribe` to
`streaming=False` for batch-only workflows.

## 2. High-Level Architecture

```
                       ASREngine
   ┌──────────────────────────────────────────────────────────┐
   │                                                          │
   │   ┌───────────────┐   ┌─────────────┐                    │
   │   │ InputProcessor│   │  Scheduler  │                    │
   │   │ (load/fbank/  │   │ (admission, │                    │
   │   │  chunk split) │   │  batching)  │                    │
   │   └───────┬───────┘   └──────┬──────┘                    │
   │           │                  │                           │
   │           ▼                  ▼                           │
   │       Request*          SchedulerOutput                  │
   │           │                  │                           │
   │           │   ┌──────────────┴────────────┐              │
   │           │   ▼                           ▼              │
   │           │ OfflinePipeline       streaming step path    │
   │           │ (micro-batches,       (per-step batched      │
   │           │  CPU/GPU overlap)      paged forward)        │
   │           │   │                           │              │
   │           │   ▼                           ▼              │
   │           │  ┌──────────────────────────────┐            │
   │           └─▶│         ModelRunner          │            │
   │              │  (offline forward,           │            │
   │              │   streaming forward+cache)   │            │
   │              └──────────────┬───────────────┘            │
   │                             ▼                            │
   │                  ┌─────────────────────┐                 │
   │                  │   OutputProcessor   │                 │
   │                  │ (CTC decode +       │                 │
   │                  │  detokenization)    │                 │
   │                  └─────────────────────┘                 │
   └──────────────────────────────────────────────────────────┘
```

## 3. Internal Structure

| File | Class | Role |
|------|-------|------|
| `engine.py` | `ASREngine` | Top-level façade. Owns one of each subsystem and the step loop. |
| `offline.py` | `OfflineEngine` | `ASREngine` with `streaming=False` default; also exposes legacy `transcribe_streaming` (no paged cache). |
| `config.py` | `EngineConfig` | Unified dataclass aggregating model / cache / feature / decoding / detokenization settings. Auto-detects SentencePiece model and `units.txt`. |
| `request.py` | `Request`, `RequestOutput`, `RequestState` | Single-request representation, output container, lifecycle enum (`WAITING → RUNNING → FINISHED`). |
| `scheduler.py` | `Scheduler`, `SchedulerOutput` | Dynamic-batching admission control and length bucketing. See `scheduler.md`. |
| `input_processor.py` | `InputProcessor` | Audio loading, batched fbank/MFCC (CPU pool or GPU stream), streaming chunk-split, CMVN-free (CMVN is in the model). |
| `model_runner.py` | `ModelRunner` | Wraps `ConformerModel`. Owns the cache managers; runs `forward_offline`, `forward_streaming_step`, and the batched paged path. |
| `pipeline.py` | `OfflinePipeline` | Producer/consumer pipeline that splits an offline batch into length-bucketed micro-batches and overlaps fbank with forward+decode. |
| `output_processor.py` | `OutputProcessor` | CTC decode (greedy / prefix beam / GPU / WFST) and SentencePiece-or-units detokenization. |

## 4. Core Algorithms and Workflows

### 4.1 Offline transcription (single pass)

```
add_request(audio, streaming=False):
    waveform = load + scale(audio_scale)
    num_frames = exact Kaldi snip_edges count
    enqueue Request in scheduler._offline_waiting

step():
    sched = scheduler.schedule()
    if sched.offline_batch:
        outputs = OfflinePipeline.run(sched.offline_batch)
```

`OfflinePipeline.run` then:

1. Length-bucket and split into micro-batches of size
   `offline_micro_batch_size` each.
2. Pad and ship features to the device (or run batched GPU fbank on a
   dedicated CUDA stream).
3. Pipeline depth `D` allows up to `D` micro-batches in flight: a
   producer thread runs fbank for chunk `k+1` while the main thread
   forwards + decodes chunk `k` on the default stream.
4. Restore the original input order before returning.

### 4.2 Streaming transcription (chunk-by-chunk)

`add_request(audio, streaming=True)` does **no fbank work**:

```
prepare_streaming(req):
    waveform = load + scale
    samples_per_chunk = ceil(stride * frame_shift_samples)
    audio_chunks = deque of CPU float32 chunks
    samples_enqueued = total
    num_frames = exact Kaldi snip_edges count   # for length bucketing
    audio_final = True                          # all audio attached up-front
```

Per step:

```
1. schedule()
   → newly_admitted streaming → ModelRunner.allocate_stream
2. For every running stream with pending audio:
       extract_streaming_batch  → batched fbank on _feat_stream
       (one GPU kernel call across all streams)
       record event; main stream waits before reading feature_buffer
3. For every stream with a full encoder window in feature_buffer:
       group by offset
       _forward_batched_paged(group)   if all paged & full window
         else
       _forward_single(req)            for partial / final / mismatched
4. For every stream whose audio is exhausted and feature_buffer is drained:
       output = OutputProcessor.finalize_streaming
       free_stream + scheduler.finish_request
```

### 4.3 Batched paged forward (`_forward_batched_paged`)

Pre-condition: every request in `group` has identical `offset`, a full
`window` of features, and is using paged attention.

```
1. att_mgr.prepare_chunks_batched([sid for sid in group])
       # one BlockPool.allocate(B), B writes into per-stream block_table

2. block_tables, cache_seqlens = stack each stream's view
   batched_bt = cat(block_tables, dim=0)        # (B, blocks_per_seq)
   batched_cs = cat(cache_seqlens, dim=0)       # (B,)

3. batched_caches = [PagedKVCache(...) per layer with shared bt/cs]

4. xs   = torch.stack(feature_chunk_per_req)    # (B, window, F)
   cnn  = stack per-stream cnn_cache or placeholder

5. log_probs, new_cnn = model.forward_chunk_paged(xs, offset, ...)

6. For each stream in group:
       commit_chunk_paged(actual_frames, new_cnn[:, b:b+1])
       offset += actual_frames; feature_cursor += stride
       results[req.request_id] = log_probs[b:b+1]
```

This is the single biggest streaming throughput lever — the per-layer
matmuls were launch-bound at `B=1`. Batching the lockstep cohort
amortises ~`num_layers × (linear + attention + conv + ffn)` kernel
launches across all in-flight streams.

### 4.4 Chunk-by-chunk streaming with `feed_chunk`

For real-time serving where audio is produced incrementally:

```python
rid = engine.add_streaming_request()
while not eof:
    chunk = mic.read(...)
    engine.feed_chunk(rid, chunk, is_last=eof)
    outs = engine.step()       # may return partial RequestOutput
```

`feed_chunk`:

- Looks up the request via `scheduler.find_request(rid)` (O(1)).
- Appends to `req.audio_chunks` and updates `samples_enqueued`.
- Sets `req.audio_final = is_last` so the engine knows when to flush.

The engine tolerates feed-before-admission (chunks queue up) and
feed-after-admission (consumed by the next step).

### 4.5 Step loop (annotated)

```python
def step(self) -> List[RequestOutput]:
    sched = self._scheduler.schedule()
    outputs = []

    # 1. allocate cache for newly admitted streaming requests
    for req in sched.newly_admitted:
        if req.streaming:
            self._model_runner.allocate_stream(req)

    # 2. offline batch (if any) — pipelined CPU/GPU overlap
    if sched.offline_batch:
        outputs.extend(self._offline_pipeline.run(sched.offline_batch))

    running = sched.running_streams
    if running:
        # 3. batched fbank across every active stream with pending audio,
        #    on the dedicated _feat_stream so it overlaps the previous
        #    step's encoder forward
        needs_feat = [r for r in running if r.has_pending_audio]
        if needs_feat:
            self._input_processor.extract_streaming_batch(
                needs_feat, cuda_stream=self._feat_stream,
            )
            torch.cuda.current_stream().wait_stream(self._feat_stream)

        # 4. for every stream with a full encoder window, run forward.
        #    Streams with shared (offset) take the batched paged path;
        #    others fall back to per-stream forward.
        ready = [r for r in running if r.has_ready_encoder_chunk(window)]
        if ready:
            log_probs_map = self._model_runner.forward_streaming_step(ready)
            for req in ready:
                lp = log_probs_map.get(req.request_id)
                if lp is not None:
                    outputs.append(
                        self._output_processor.decode_streaming_chunk(req, lp)
                    )

        # 5. finalise streams whose audio is exhausted and feature
        #    buffer drained (may happen the same step as the last forward)
        for req in list(running):
            if (not req.has_pending_audio) \
                    and (not req.has_ready_encoder_chunk(window)):
                final = self._output_processor.finalize_streaming(req)
                req.output = final
                outputs.append(final)
                self._model_runner.free_stream(req)
                self._scheduler.finish_request(req.request_id)

    return outputs
```

## 5. Data Flow

```
audio bytes / file path / Tensor
        │
        ▼
InputProcessor.load_audio          ──▶ scale(audio_scale)
        │                                │
   (offline)                        (streaming)
        │                                │
        ▼                                ▼
prepare_offline                    prepare_streaming
  → Request.waveform                 → Request.audio_chunks
  → Request.num_frames               → Request.num_frames (exact estimate)
        │                                │
        └──────────────── Scheduler ─────┘
                              │
            ┌─────────────────┴──────────────────┐
            ▼                                    ▼
     OfflinePipeline                    streaming step:
       collate_gpu/cpu                    extract_streaming_batch  → _feat_stream
       _gpu_stage:                          (writes feature_buffer)
         forward_offline                  forward_streaming_step
         decode_offline                   _forward_batched_paged | _forward_single
                                          decode_streaming_chunk → CTC
            │                                    │
            ▼                                    ▼
                       RequestOutput
                       (text, tokens, scores, finished)
```

## 6. Configuration Options

`EngineConfig` aggregates every knob. Key groups:

### Model and runtime

| Field | Default | Description |
|-------|---------|-------------|
| `ckpt_dir` | `""` | WeNet checkpoint dir (`final.pt`, `train.yaml`, `global_cmvn`, optional `.model` and `units.txt`). |
| `checkpoint_name` | `"final.pt"` | Filename inside `ckpt_dir`. |
| `device` | `"cuda"` | Target device. |
| `dtype` | `torch.float16` | Model + cache precision. |
| `audio_scale` | `32768.0` | Multiplied into the float waveform to restore int16 scale used in WeNet training. |

### Streaming chunking

| Field | Default | Description |
|-------|---------|-------------|
| `chunk_size` | 16 | Encoder output frames per chunk. Must match training. |
| `num_left_chunks` | -1 | Past chunks kept in attention cache (-1 = unlimited). |

### Batching

| Field | Default | Description |
|-------|---------|-------------|
| `max_batch_size` | 32 | Concurrent streaming requests. |
| `max_offline_batch_size` | 1024 | Offline requests admitted per step. |
| `length_bucket_ratio` | 0.0 | Soft floor on `min_len/max_len` in offline batch. |
| `max_offline_pad_ratio` | 4.0 | Hard cap on padded/useful compute. |
| `max_wait_time` | 0.2 | Starvation bound (seconds). |
| `schedule_policy` | `"bucket"` | `fcfs` / `bucket` / `sjf`. |
| `streaming_cohort_admit` | `True` | Admit only when running pool offsets align — enables full `B` batched paged forward. |
| `offline_micro_batch_size` | 32 | GPU forward `B` for the offline pipeline. |
| `offline_pipeline_depth` | 3 | In-flight micro-batches (1 = sequential). |
| `offline_gpu_feature_extraction` | `True` | Batched GPU fbank vs CPU pool. |

### Paged KV cache

| Field | Default | Description |
|-------|---------|-------------|
| `max_num_blocks` | 2048 | Total physical blocks in the shared pool. |
| `block_size_frames` | 16 | Frames per block (= chunk_size by default). |
| `max_blocks_per_seq` | 512 | Block-table width. |
| `use_paged_cache` | `True` | False falls back to dense `forward_chunk`. |

### Feature extraction

| Field | Default | Description |
|-------|---------|-------------|
| `feature_config` | `FeatureConfig(dither=0.0)` | 80-dim log-mel FBANK at 16 kHz; deterministic. |
| `num_feature_workers` | 0 (auto) | CPU thread pool size for offline fbank. `0` → `min(4, nproc//4)`. |
| `cpu_intra_op_threads` | 0 (auto) | `torch.set_num_threads`; oversubscription protection. |

### Decoding and detokenization

| Field | Default | Description |
|-------|---------|-------------|
| `decoder_type` | `"ctc_prefix_beam"` | `ctc_greedy` / `ctc_prefix_beam` / `ctc_gpu` / `ctc_wfst`. |
| `gpu_decoder_config` | `GpuDecoderConfig()` | GPU CTC config (beam, blank ID, thresholds). |
| `cpu_decoder_config` | `DecoderConfig(search_type="prefix_beam")` | CPU decoder config. |
| `fst_path` | `None` | Required for `ctc_wfst`. |
| `sentencepiece_model` | auto-detected | `.model` in `ckpt_dir`. |
| `unit_table` | auto-detected | `units.txt` / `words.txt` fallback. |

### Derived properties

```python
subsampling_rate = 4              # Conv2dSubsampling
right_context    = 6
decoding_window  = (chunk_size - 1) * 4 + right_context + 1   # input frames per chunk
stride           = subsampling_rate * chunk_size              # frame advance
required_cache_size = chunk_size * num_left_chunks            # dense mode
```

`build_cache_config(model_config)` derives a `CacheConfig` from the
loaded encoder dimensions.

## 7. Usage Examples

### 7.1 Offline batch transcription

```python
from oasr.engine import OfflineEngine, EngineConfig

engine = OfflineEngine(EngineConfig(ckpt_dir="/path/to/ckpt"))

text  = engine.transcribe("audio.wav")
texts = engine.transcribe(["a.wav", "b.wav", "c.wav"])
```

### 7.2 Streaming, attached audio

```python
from oasr.engine import ASREngine, EngineConfig

engine = ASREngine(EngineConfig(ckpt_dir="/path/to/ckpt"))

# All audio handed in up-front; engine runs to completion.
texts = engine.transcribe(["a.wav", "b.wav", "c.wav"], streaming=True)
```

### 7.3 Streaming, chunk-by-chunk feed (real-time serving)

```python
rid = engine.add_streaming_request()
for chunk in mic_chunks(samples_per_chunk=4000):
    engine.feed_chunk(rid, chunk, is_last=False)
    outputs = engine.step()           # may return partial transcripts
    for o in outputs:
        if o.request_id == rid and not o.finished:
            print("partial:", o.text)

engine.feed_chunk(rid, last_chunk, is_last=True)
final = engine.run()                  # drain until finalised
```

### 7.4 Mixed offline + streaming on one engine

```python
engine = ASREngine(EngineConfig(ckpt_dir=..., max_batch_size=8,
                                max_offline_batch_size=64))
for path in offline_paths:
    engine.add_request(path, streaming=False)
for path in streaming_paths:
    engine.add_request(path, streaming=True)

results = engine.run()                # one engine handles both pools
```

### 7.5 Selecting GPU CTC

```python
engine = ASREngine(EngineConfig(
    ckpt_dir=...,
    decoder_type="ctc_gpu",
    gpu_decoder_config=GpuDecoderConfig(beam_size=10, blank_threshold=0.95),
))
```

## 8. Error Handling and Edge Cases

| Situation | Behaviour |
|-----------|-----------|
| `feed_chunk` for unknown / finalised id | `KeyError` from `Scheduler.find_request → None`. |
| Block pool exhaustion | `RuntimeError` from `BlockPool.allocate`. Currently fatal — size `max_num_blocks` for worst case. |
| Audio shorter than one window with `audio_final=True` | `_forward_single` flushes whatever frames remain (special `is_final_window` path). |
| `chunk.size(1) < context` (less than `right_context+1` input frames) and not final | Skipped — the engine waits for more audio. |
| First chunk of paged stream | `prepare_chunks_batched` lazily allocates `block_table` / `cache_seqlens` before writing the first physical block. |
| First-chunk CNN cache is `(0,0,0,0)` placeholder | Batched forward passes the placeholder; the model handles per-layer initialisation. |
| `ckpt_dir` missing required files | Fails fast inside `load_wenet_checkpoint` (out of engine scope). |
| Unsupported `decoder_type` | `OutputProcessor` raises during decode. |
| Aborting a running stream | `abort_request` removes from the scheduler and frees the cache via `ModelRunner.free_stream`. |
| Audio sample-rate mismatch | `InputProcessor.load_audio` resamples to `sample_rate`. |
| Force-flush firing inside an offline batch | All bucket guards skipped — batch may be highly padded. Acceptable to bound starvation. |
| Streaming admission gated by cohort | New requests wait until the running pool drains; visible as no `newly_admitted` even with `num_waiting_streaming > 0`. |

## 9. Performance Considerations

1. **Streaming throughput is dominated by `_forward_batched_paged`.**
   Keep `streaming_cohort_admit=True` and pick `max_batch_size` to fit
   the GPU. Larger batches amortise launch overhead but spread per-step
   latency.
2. **Two CUDA streams overlap fbank and forward.** `_feat_stream` runs
   the streaming fbank kernel; the default stream waits on a recorded
   event before reading `feature_buffer`. The offline pipeline does the
   same with its own feat stream.
3. **`offline_pipeline_depth=3` is a good default.** Depth 1 disables
   threading and overlap; depth 2 runs CPU prep one step ahead;
   depth 3 covers any bursty scheduling. Above 3 rarely helps because
   fbank is faster than encoder forward in steady state.
4. **GPU fbank vs CPU pool.** `offline_gpu_feature_extraction=True`
   bypasses the CPU pool's ~550 fbanks/s ceiling by running the whole
   micro-batch as one fused kernel sequence on the feat stream. Set
   `False` only when GPU is already saturated by the model forward.
5. **Length bucketing trade-off.** `length_bucket_ratio=0` (default)
   ships one big batch; `0.5` insists on ≥50 % length similarity. The
   default is faster on real datasets because splitting a batch
   multiplies CPU fbank cost; `max_offline_pad_ratio=4.0` is the safety
   net against pathological mixes.
6. **Avoid oversubscribing the host CPU.** The InputProcessor caps
   PyTorch intra-op threads (default `min(8, nproc / num_workers)`)
   because the PyTorch default oversubscribes for short fbank ops.
7. **NVTX profiling.** The step loop is annotated with
   `nvtx_push("engine.step")` → `schedule` / `allocate_stream` /
   `offline_batch` / `extract_fbank` / `forward_streaming` /
   `decode_streaming` / `finalize_streams`. Capture with
   `nsys profile`/`ncu` to get per-stage timing without touching code.
8. **Pool sizing.** The engine's most common production failure is
   `BlockPool` exhaustion. Size `max_num_blocks` for
   `max_batch_size × max_logical_blocks` plus headroom; trade off
   against GPU memory.

## 10. Extension Points

- **Custom feature extraction.** Override `FeatureConfig` and pass it
  through `EngineConfig.feature_config`. To plug in an entirely new
  backend, change `oasr.features.backends._extract` or hook in
  `InputProcessor.load_audio` / `extract_features_batch`.
- **Custom decoder.** Add a new branch in
  `OutputProcessor.decode_offline` / `decode_streaming_chunk` /
  `finalize_streaming` and a corresponding entry in
  `EngineConfig.decoder_type`.
- **Custom model.** `ModelRunner` is hard-wired to `ConformerModel`. To
  support a different architecture, mirror its `encoder` / `ctc` /
  `forward_chunk_paged` / `forward_chunk` interface, then swap the
  factory call in `ASREngine.__init__`.
- **Custom scheduler.** Subclass `Scheduler` and replace
  `self._scheduler` in `ASREngine.__init__`. The engine only depends on
  the `SchedulerOutput` shape and on `add_request`, `schedule`,
  `finish_request`, `abort_request`, `find_request`, `has_pending`,
  `num_running`, `num_waiting`.
- **Streaming preemption.** Currently absent. To add it, extend the
  scheduler to evict a low-priority running stream, then have
  `ASREngine` call `ModelRunner.free_stream` and re-queue the request.
  No public API changes required.
- **Engine subclasses.** `OfflineEngine` is a worked example: it
  overrides `transcribe` to default `streaming=False` and adds a
  legacy `transcribe_streaming` helper without touching the step loop.

## 11. Quick Reference

```text
# construction
engine = ASREngine(EngineConfig(ckpt_dir=..., **knobs))

# request submission
rid = engine.add_request(audio, streaming=True/False, priority=0)
rid = engine.add_streaming_request()       # then feed_chunk
engine.feed_chunk(rid, chunk, is_last=False)

# stepping
outputs = engine.step()                    # one tick, returns partials+finals
finals  = engine.run()                     # drain until empty

# convenience
texts = engine.transcribe(audio_or_list, streaming=...)

# mid-flight control
engine.abort_request(rid)

# observability
engine.num_running    # int
engine.num_waiting    # int
```

## 12. Related Documents

- `cache_manager.md` — paged KV cache, CNN cache, CTC state pool.
- `scheduler.md` — admission policies, length bucketing, cohort gate.
- `autotuning.md` — kernel-level tuning framework (orthogonal to engine).
- `CLAUDE.md` — top-level architecture and build/test commands.
