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
   executor or the streaming step path.
4. **Driving the step loop**: schedule → ingest fbank → forward → decode
   → finalise.
5. **Exposing a high-level API**:
   - `transcribe(audio | List[audio])` for one-shot streaming use.
   - `transcribe_offline(audio | List[audio])` for batch-only offline use.
   - `add_request` / `feed_chunk` / `step` / `run` for explicit control.

The engine no longer has a separate `OfflineEngine` subclass — pass
`streaming=False` to `add_request` / `transcribe`, or use the
`transcribe_offline` convenience helper, to take the offline path.

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
   │           │ OfflineExecutor       streaming step path    │
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
| `engine.py` | `ASREngine` | Top-level façade.  Owns one of each subsystem and the step loop.  Use `transcribe(...)` for streaming and `transcribe_offline(...)` for batch. |
| `config.py` | `EngineConfig` | Unified dataclass aggregating model / cache / feature / decoding / detokenization settings. Auto-detects SentencePiece model and `units.txt`. |
| `request.py` | `Request`, `RequestOutput`, `RequestState` | Single-request representation, output container, lifecycle enum (`WAITING → RUNNING → FINISHED`). |
| `scheduler.py` | `Scheduler`, `SchedulerOutput` | Dynamic-batching admission control, length bucketing, and offline micro-batch partition (`split_offline_batch`: count/preferred, padded-frame, or sequence-packed). See `scheduler.md`. |
| `input_processor.py` | `InputProcessor` | Audio loading, batched GPU fbank/MFCC, streaming chunk-split, CMVN-free (CMVN is in the model). |
| `model_runner.py` | `ModelRunner` | Wraps `ConformerModel`. Owns the cache managers; runs `forward_offline`, `forward_streaming_step`, and the batched paged path. |
| `executor/offline.py` | `OfflineExecutor` | Runs each scheduler-partitioned micro-batch (fbank → forward → decode → finalise) back-to-back; sequence-packed forward when `enable_sequence_packing` is set. |
| `executor/streaming.py` | `StreamingExecutor` | Chunk-by-chunk streaming with paged KV cache; partial outputs per tick, final on drain. |
| `output_processor.py` | `OutputProcessor` | CTC decode (GPU beam / k2 WFST) and SentencePiece-or-units detokenization. |

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
        outputs = OfflineExecutor.run(sched.offline_batch)
```

`OfflineExecutor.run` then:

1. Ask the scheduler to partition the batch into micro-batches
   (`Scheduler.split_offline_batch`: length-bucketed / padded-frame / packed).
2. Run batched GPU fbank over each micro-batch.
3. Run micro-batches back-to-back on the default stream
   (fbank → forward → decode → finalise); with `enable_sequence_packing`
   each micro-batch is one gapless varlen-attention packed row.
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

    # 2. offline batch (if any) — micro-batches run back-to-back
    if sched.offline_batch:
        outputs.extend(self._executor.run(sched.offline_batch))

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

The engine is **waveform-only** — file decoding happens at the entry point
(the serving front-end via `oasr-asr`, or the bench/test harness), never in
the engine. `audio_scale` is applied on the **GPU after padding** in
`collate` (offline).

```
Request.audio = waveform (Tensor / ndarray, at the model sample rate, or None)
        │
   (offline)                        (streaming)
        │                                │
        ▼                                ▼
prepare_offline                    prepare_streaming
  → canonicalises Request.audio      → Request.audio_chunks
    in place (1-D f32 CPU)           → Request.num_frames (exact estimate)
  → Request.num_frames
        │                                │
  collate: pad + scale          append_streaming_chunk: scale (CPU)
  on the GPU (after padding)         per chunk
        │                                │
        └──────────────── Scheduler ─────┘
                              │
            ┌─────────────────┴──────────────────┐
            ▼                                    ▼
     OfflineExecutor                    streaming step:
       collate/cpu                    extract_streaming_batch  → _feat_stream
       _run_stage:                          (writes feature_buffer)
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
| `max_batch_size` | 32 | Encoder forward `B`. In streaming mode caps the running pool; in offline mode is the GPU forward width. Offline admission per `step()` is one length-bucketed batch of up to `max_batch_size`, run as a single forward. |
| `preferred_batch_size` | `None` | When set, scheduler snaps streaming admission and offline micro-batches to one of these sizes; engine pre-warms the encoder CUDA-Graph cache at each value; defaults `feature_graph_batch_buckets`. `max_wait_time` is the escape valve. See [scheduler.md §4.6](scheduler.md). |
| `length_bucket_ratio` | 0.0 | Soft floor on `min_len/max_len` in offline batch. |
| `max_offline_pad_ratio` | 4.0 | Hard cap on padded/useful compute. |
| `max_wait_time` | 0.2 | Starvation bound (seconds). |
| `schedule_policy` | `"bucket"` | `fcfs` / `bucket` / `sjf`. |
| `streaming_cohort_admit` | `True` | Admit only when running pool offsets align — enables full `B` batched paged forward. |

### Paged KV cache

| Field | Default | Description |
|-------|---------|-------------|
| `max_num_blocks` | 2048 | Total physical blocks in the shared pool. `None` derives it from free VRAM — see [§6.1](#61-vram-aware-capacity-sizing). Inert in `service_mode="offline"`, which builds no pool. |
| `gpu_memory_utilization` | 0.90 | Share of the device the engine may occupy in total (weights + caches + activations) when it derives a capacity. Read only when something is left to derive. |
| `block_size_frames` | 16 | Frames per block (= chunk_size by default). |
| `max_blocks_per_seq` | 512 | Block-table width. With unlimited history this, times `max_batch_size`, is also the ceiling a derived pool is capped at — blocks past it cannot be addressed. |
| `use_paged_cache` | `True` | False falls back to dense `forward_chunk`. |

### 6.1 VRAM-aware capacity sizing

Two capacities are memory rather than compute, and both can be left to the
engine (`oasr/engine/memory.py`):

| Left unset | Derived as |
|---|---|
| `max_num_blocks=None` (streaming) | `available / bytes_per_block`, capped at `max_batch_size × blocks_per_seq` |
| `decode_kv_budget_gib=None` (AR families) | `available`, clamped up to one row so a tight card still admits work |

where

```
available = total × gpu_memory_utilization − resident − activation_reserve
```

`resident` is read from the driver (`torch.cuda.mem_get_info`) after the model
is on the device, so it covers the weights, the CUDA context **and** anything
another process holds — there is no separate "weights" term to get wrong.
`activation_reserve` is `1.5 ×` a *measured* probe forward at the widest shape
the engine will run (one chunk window at `max_batch_size` in streaming mode; the
frontend's fixed window, else 30 s, in offline mode), floored at 256 MiB. The
unspent `1 − utilization` is the headroom for what the probe cannot see: CUDA
graph capture pools, an AR family's prefill transient, allocator fragmentation.

The derivation is logged in full (`paged KV pool derived from VRAM: … | total=…
resident=… cap=… activation_reserve=… → available=…`) so the numbers are
auditable. Nothing is measured — and no probe forward runs — unless a capacity
was actually left unset.

Failure modes are explicit rather than silent: on a non-CUDA device
`max_num_blocks=None` raises (there is nothing to measure), and when not even
the minimum viable pool fits, construction fails with the arithmetic and the
levers (`max_batch_size`, `num_left_chunks`, `gpu_memory_utilization`, or an
explicit `max_num_blocks`) rather than deriving a pool that OOMs at allocation
or degrades every transcript. `decode_kv_budget_gib=0` turns the byte budget off
outright; `None` derives it.

Worth knowing: on a large card the pool is usually capped by
`max_blocks_per_seq × max_batch_size`, not by VRAM (a 32 GiB card had 27 GiB
available and took 3 GiB). The engine logs when that happens — the remaining
lever is then the *per-stream* ceiling, not the pool size.

### Feature extraction

| Field | Default | Description |
|-------|---------|-------------|
| `feature_config` | `FeatureConfig(dither=0.0)` | 80-dim log-mel FBANK at 16 kHz; deterministic. Extracted on the GPU via the batched fbank/mfcc kernels. |

### Decoding and detokenization

| Field | Default | Description |
|-------|---------|-------------|
| `decoder_type` | `"ctc_cuda"` | `ctc_cuda` (GPU CTC beam) / `ctc_wfst` (k2 WFST, GPU). |
| `ctc_decoder_config` | `GpuDecoderConfig()` | GPU CTC config (beam, blank ID, thresholds). |
| `wfst_decoder_config` | `DecoderConfig(search_type="wfst")` | k2 WFST decoder config. |
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
import torchaudio
from oasr.engine import ASREngine, EngineConfig

engine = ASREngine(EngineConfig(ckpt_dir="/path/to/ckpt", service_mode="offline"))

# The engine is waveform-only — decode files yourself (here in the harness;
# in serving, oasr-asr does this at the entry point).
def wav(p): return torchaudio.load(p)[0].squeeze(0).float()

text  = engine.transcribe_offline(wav("audio.wav"))
texts = engine.transcribe_offline([wav(p) for p in ("a.wav", "b.wav", "c.wav")])
```

### 7.2 Streaming, attached audio

```python
import torchaudio
from oasr.engine import ASREngine, EngineConfig

engine = ASREngine(EngineConfig(ckpt_dir="/path/to/ckpt"))

def wav(p): return torchaudio.load(p)[0].squeeze(0).float()

# All audio handed in up-front; engine splits it into chunks and runs to
# completion (real-time clients instead use add_streaming_request + feed_chunk).
texts = engine.transcribe([wav(p) for p in ("a.wav", "b.wav", "c.wav")], streaming=True)
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
engine = ASREngine(EngineConfig(ckpt_dir=..., max_batch_size=8))
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
    decoder_type="ctc_cuda",
    ctc_decoder_config=GpuDecoderConfig(beam_size=10, blank_threshold=0.95),
))
```

## 8. Error Handling and Edge Cases

| Situation | Behaviour |
|-----------|-----------|
| `feed_chunk` for unknown / finalised id | `KeyError` from `Scheduler.find_request → None`. |
| Block pool exhaustion | `RuntimeError` from `BlockPool.allocate`. Currently fatal — size `max_num_blocks` for worst case, or set it to `None` and let the engine derive it ([§6.1](#61-vram-aware-capacity-sizing)). With eviction on (`num_left_chunks >= 0`) the invariant is checked at construction instead. |
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
2. **A CUDA stream overlaps streaming fbank and forward.** `_feat_stream`
   runs the streaming fbank kernel; the default stream waits on a recorded
   event before reading `feature_buffer`. The offline executor runs its
   micro-batches sequentially (GPU fbank → forward → decode), so it needs
   no extra stream.
3. **Length bucketing trade-off.** `length_bucket_ratio=0` (default)
   ships one big batch; `0.5` insists on ≥50 % length similarity;
   `max_offline_pad_ratio=4.0` is the safety net against pathological
   mixes.
4. **NVTX profiling.** The step loop is annotated with
   `nvtx_push("engine.step")` → `schedule` / `allocate_stream` /
   `offline_batch` / `extract_fbank` / `forward_streaming` /
   `decode_streaming` / `finalize_streams`. Capture with
   `nsys profile`/`ncu` to get per-stage timing without touching code.
5. **Pool sizing.** The engine's most common production failure is
   `BlockPool` exhaustion. Size `max_num_blocks` for
   `max_batch_size × max_logical_blocks` plus headroom; trade off
   against GPU memory. Or hand it over: `max_num_blocks=None` derives the
   pool from free VRAM at construction ([§6.1](#61-vram-aware-capacity-sizing)),
   which is what makes one config portable across card sizes.

## 10. Extension Points

Every extension lands through a registry — subclass a base, register under a
name, and the engine picks it up by configuration. **Never edit the engine
core to add a variant.** `docs/architecture.md` is the authoritative map
(seams table + cookbook); the summary:

- **New model architecture** → model package under `oasr/models/<arch>/`
  (`BaseAsrModel` / `BaseEncoder`) + a `CheckpointConverter`, registered via
  `register_model`. The engine touches models only through the base-class
  surface (`forward_offline` / `encode_offline` / `forward_chunk_paged` /
  `cache_spec` / `capabilities` / `encoder.streaming_kind`). See
  `docs/checkpoints.md` for the converter contract and native format.
- **New decode family** → `DecodeStrategy` + `@register_decode_strategy`.
  Declare `consumes` (`"log_probs"` / `"hidden"` / `"both"`) and, for
  label-synchronous AR families, `incremental = True` with the bounded
  `begin_offline` / `advance(StepBudget)` / `has_pending` protocol. Selected
  per deployment by `EngineConfig.decode_method` (validated against
  `model.capabilities`); per-request knobs ride on
  `DecodingOptions` (`Request.decoding`).
- **New streaming runtime** → `StreamingEncoderBackend` +
  `@register_streaming_backend`, keyed by the encoder's `streaming_kind`.
  Implement `allocate` / `forward_step` / `free` + window geometry. Expose
  `stack_streaming_states` / `unstack_streaming_states` on the encoder to get
  batched (one `B = N` forward) stateful streaming for free.
- **New batching / partition policy** → `@register_batching_policy` /
  `@register_partition_policy`; select via `EngineConfig.schedule_policy` and
  the partition flags.
- **New tokenizer kind** → `Tokenizer` subclass + `register_tokenizer`,
  selected by the converter-emitted `TokenizerSpec.kind`. See
  `docs/tokenizers.md`.
- **Feature extraction** is checkpoint-derived: converters emit a
  `FeatureSpec` the engine materializes into `FeatureConfig`; an explicit
  `EngineConfig.feature_config` still overrides (with a loud mismatch
  warning).
- **Streaming preemption.** Currently absent. To add it, extend the
  scheduler to evict a low-priority running stream, then have
  `ASREngine` call `ModelRunner.free_stream` and re-queue the request.
  No public API changes required.

## 11. Quick Reference

```text
# construction
engine = ASREngine(EngineConfig(ckpt_dir=..., **knobs))

# request submission
rid = engine.add_request(audio, streaming=True/False, priority=0)
rid = engine.add_streaming_request()       # then feed_chunk
engine.feed_chunk(rid, chunk, is_last=False)

# per-request decoding options (n-best, AR generation cap, sampling, LLM prompt)
from oasr.engine import DecodingOptions
rid = engine.add_request(audio, streaming=False,
                         decoding=DecodingOptions(n_best=3, max_new_tokens=64))

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
