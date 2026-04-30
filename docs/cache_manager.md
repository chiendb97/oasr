# Cache Manager

The cache subsystem (`oasr/cache/`) manages all per-stream GPU state required
for chunk-by-chunk Conformer inference: paged attention KV cache, fixed-size
CNN left-context cache, and per-stream CTC decoder state. It is the
foundation that lets the engine run many concurrent streaming requests on a
single shared GPU memory pool without tearing down or re-allocating buffers
between chunks.

## 1. Purpose and Responsibilities

The cache manager exists to:

1. **Pre-allocate** all GPU memory needed for streaming KV-cache up front,
   avoiding per-chunk `cudaMalloc` calls.
2. **Page** the attention KV cache so that a fixed pool of physical blocks
   can be shared across many concurrent streams (vLLM-style).
3. **Track per-stream state** (logical block lists, frame counters,
   `block_table` / `cache_seqlens` tensors for paged attention) and keep it
   in sync with what the model writes into the pool.
4. **Evict** old context once a stream exceeds its left-context window.
5. **Pool** lightweight CTC decoder states so allocating a new stream does
   not require a fresh GPU buffer.
6. **Expose a single per-stream handle** (`StreamContext`) so engine code
   can interact with all three caches without juggling stream IDs.

## 2. High-Level Architecture

```
                         ┌────────────────────────────┐
                         │      CacheConfig            │
                         │  (one shared dataclass)     │
                         └──────────────┬─────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                │                       │                       │
                ▼                       ▼                       ▼
     ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
     │     BlockPool      │  │  CnnCacheManager   │  │ CtcStateCacheMgr   │
     │ (shared K/V pool)  │  │ (per-stream tensor)│  │ (1 decoder + pool) │
     └─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘
               │                       │                       │
               ▼                       │                       │
     ┌────────────────────┐            │                       │
     │AttentionCacheMgr   │            │                       │
     │ (per-stream block  │            │                       │
     │  table + seqlens)  │            │                       │
     └─────────┬──────────┘            │                       │
               │                       │                       │
               └───────────────────────┴────┬──────────────────┘
                                            ▼
                                  ┌──────────────────┐
                                  │  StreamContext   │
                                  │  (one per req)   │
                                  └──────────────────┘
```

`BlockPool` is the only object holding the large preallocated K and V
tensors. `AttentionCacheManager` consumes block IDs from it; `CnnCacheManager`
allocates a small per-stream tensor of its own; `CtcStateCacheManager`
delegates to a single shared `GpuStreamingDecoder` engine and pools per-stream
`StreamState` objects. `StreamContext` is a thin facade that bundles a
`stream_id` with all three managers.

## 3. Internal Structure and Key Classes

| File | Class / Type | Responsibility |
|------|--------------|----------------|
| `types.py` | `CacheConfig` | Master configuration dataclass (layers, heads, dims, chunk size, block size, pool capacity, dtype, device) and derived helpers (`max_logical_blocks`, `cnn_cache_frames`, `kv_last_dim`). |
| `block_pool.py` | `BlockPool` | Two large GPU tensors (`_k_pool`, `_v_pool`) of shape `(num_layers, max_num_blocks, block_size_frames, n_kv_head, head_dim)` plus a thread-safe free list of physical block IDs. |
| `attention_cache.py` | `AttentionCacheManager`, `_StreamKVState` | Per-stream logical-to-physical block mapping; supports both **dense** (`commit` + `get_stacked_cache`) and **paged** (`prepare_chunk` + `commit_chunk_paged` + `PagedKVCache`) modes; handles eviction. |
| `cnn_cache.py` | `CnnCacheManager` | Per-stream tensor of shape `(num_layers, 1, kernel_size - 1, hidden_dim)` overwritten in place each chunk (no paging needed — fixed size per stream). |
| `ctc_state.py` | `CtcStateCacheManager` | Single shared `GpuStreamingDecoder` plus a pool of `StreamState` objects; exposes a `StreamHandle` per call site so callers see the same API as a stand-alone decoder. |
| `stream.py` | `StreamContext` | Per-request facade that ties the three managers together using a single `stream_id`. |

### `CacheConfig` highlights

```python
@dataclass
class CacheConfig:
    num_layers: int = 12
    n_kv_head: int = 4
    head_dim: int = 64
    hidden_dim: int = 256
    kernel_size: int = 15           # depthwise conv kernel
    chunk_size: int = 16            # encoder output frames per chunk
    num_left_chunks: int = -1       # -1 = unlimited history
    block_size_frames: int = 16     # frames per physical block
    max_num_blocks: int = 1024      # total pool size
    max_blocks_per_seq: int = 512   # block_table capacity per stream
    device: torch.device = ...
    dtype:  torch.dtype  = torch.float16
```

Derived properties of interest:

- `kv_last_dim → head_dim * 2` — K and V are packed along the last dim
  in the dense API.
- `cnn_cache_frames → kernel_size - 1` — left-context frames stored by the
  causal depthwise conv per layer.
- `max_logical_blocks → ceil(chunk_size * num_left_chunks / block_size_frames)`
  or `None` when history is unlimited.

The pool must be sized so that `max_num_blocks ≥ max_batch_size *
max_logical_blocks` for the worst case, with extra headroom for short-lived
oversubscription during cohort transitions.

## 4. Core Algorithms and Workflows

### 4.1 Paged KV-cache lifecycle (per stream)

Paged mode is the production path used by `ASREngine`. For each new chunk:

```
prepare_chunk(stream_id):
    if stream is new:
        allocate per-stream block_table (zeros, int32, max_blocks_per_seq)
        allocate per-stream cache_seqlens (zeros, int32, shape (1,))
    block_id = pool.allocate(1)
    state.logical_blocks.append(block_id)
    state.block_table[0, len(logical_blocks) - 1] = block_id

# Model writes K/V directly into pool[block_id] via _paged_write_kv.

commit_chunk_paged(stream_id, chunk_frames):
    state.num_committed_frames += chunk_frames
    state.cache_seqlens[0] = state.num_committed_frames
    while len(state.logical_blocks) > max_logical_blocks:
        evicted = state.logical_blocks.pop(0)
        pool.free([evicted])
        # shift block_table left by one slot
        # adjust num_committed_frames + cache_seqlens
```

Key invariants:

- The model never sees a physical block ID — it walks the per-stream
  `block_table` indexed by logical block.
- `cache_seqlens[0]` is *the* source of truth for paged attention; the
  manager keeps it consistent across commit and eviction.
- Eviction is left-shift, preserving FIFO order. A stream's history is
  always the most recent `num_left_chunks` chunks.

### 4.2 Dense KV-cache mode (legacy / `forward_chunk`)

Dense mode is kept for the non-paged code path and for tests. `commit()`
allocates one block per chunk, splits packed `(K|V)` into K and V slabs,
writes them into the pool, then `get_stacked_cache()` gathers all blocks
back into a contiguous `(num_layers, n_kv_head, T, head_dim*2)` tensor —
the shape `ConformerEncoder.forward_chunk` expects. Eviction is identical
to paged mode.

### 4.3 Batched paged forward path

`AttentionCacheManager.prepare_chunks_batched(stream_ids)` performs a
single `BlockPool.allocate(B)` call, then writes the resulting block IDs
into each stream's own `block_table[0, idx]`. The engine
(`ModelRunner._forward_batched_paged`) then concatenates each stream's
`(block_table, cache_seqlens)` views (via `get_paged_state_views`) into
batched `(B, max_blocks_per_seq)` and `(B,)` tensors and constructs one
`PagedKVCache` per layer that all `B` streams share.

This is the most performance-critical path: it amortises ~`num_layers` ×
several kernel launches across all in-flight streams.

### 4.4 CNN cache update

CNN cache is fixed-size per stream and overwritten in place every chunk:

```
allocate_stream(sid):
    caches[sid] = zeros(num_layers, 1, cnn_cache_frames, hidden_dim)

update(sid, new_cnn_cache):
    caches[sid].copy_(new_cnn_cache)   # shape-checked; in-place
```

### 4.5 CTC state pooling

```
allocate_stream(sid, batch, vocab_size):
    if pool not empty:
        state = pool.pop()
        decoder.reset_state(state, batch, vocab_size)
    else:
        state = decoder.create_state(batch, vocab_size)
    states[sid] = state

free_stream(sid):
    pool.append(states.pop(sid))   # state's GPU buffer is NOT freed
```

`StreamHandle(decoder, state)` lets the caller treat the (decoder, state)
pair as a single object whose `decode_chunk` / `finalize_stream` /
`step` / `config` mirror the stand-alone `GpuStreamingDecoder` API.

## 5. Data Flow and Component Interaction

```
                ┌──────────── Caller (e.g. ASREngine.step) ────────────┐
                │                                                       │
                │    1. attention.prepare_chunk(sid)                    │
                │    2. att_caches  = attention.get_paged_caches(sid)   │
                │    3. cnn_cache   = cnn.get_cache(sid)                │
                │    4. logits, new_cnn = model.forward_chunk_paged(    │
                │            xs, offset, att_caches, cnn_cache, ...)    │
                │    5. attention.commit_chunk_paged(sid, frames)       │
                │    6. cnn.update(sid, new_cnn)                        │
                │    7. ctc.get_decoder(sid).decode_chunk(logits)       │
                │                                                       │
                └───────────────────────────────────────────────────────┘
                           ▲           ▲                ▲           ▲
                           │           │                │           │
                           │ allocate  │ get_kv_view    │ blocks    │ pool/restore
                           ▼           │                ▼           │
                     ┌──────────────────────────┐  ┌─────────────────┐
                     │       BlockPool          │  │ GpuStreaming    │
                     │  K/V tensors + freelist  │  │ Decoder + state │
                     └──────────────────────────┘  └─────────────────┘
```

`StreamContext` collapses steps 1–7 into a single `prepare_chunk()` /
`commit_chunk_paged()` / `get_decoder()` interface so the engine's hot
loop reads cleanly.

## 6. Configuration Options

`EngineConfig.build_cache_config(model_config)` derives `CacheConfig` from
the loaded model. The fields you typically tune:

| Field | Default | When to change |
|-------|---------|----------------|
| `chunk_size` | 16 | Must match training. Longer chunk = higher accuracy, higher per-chunk latency. |
| `num_left_chunks` | -1 | -1 is unlimited; set to a positive value (e.g. 4) to bound memory in long-running streams. |
| `block_size_frames` | 16 | Equal to `chunk_size` is simplest (one block per chunk). Smaller values reduce eviction granularity but increase block-table size. |
| `max_num_blocks` | 2048 | Pool capacity. Must cover all concurrent streams' worst-case logical block counts plus overhead. |
| `max_blocks_per_seq` | 512 | Width of the per-stream `block_table`. Must be ≥ `max_logical_blocks`. |
| `dtype` | `float16` | `bfloat16` works on Ampere+. The K/V pool is the dominant memory consumer. |

Pool memory in bytes:

```
2 (K and V) × num_layers × max_num_blocks × block_size_frames
            × n_kv_head × head_dim × bytes_per_element(dtype)
```

For a default Conformer (`num_layers=12, n_kv_head=4, head_dim=64,
block_size=16, max_num_blocks=2048`) with FP16 this is ~6 GiB.

## 7. Usage Examples

### 7.1 Full streaming loop (paged mode)

```python
from oasr import GpuDecoderConfig
from oasr.cache import (
    CacheConfig, BlockPool,
    AttentionCacheManager, CnnCacheManager, CtcStateCacheManager,
    StreamContext,
)

cfg = CacheConfig(num_layers=12, n_kv_head=4, head_dim=64, hidden_dim=256,
                  kernel_size=15, chunk_size=16, num_left_chunks=4,
                  block_size_frames=16, max_num_blocks=2048)

pool    = BlockPool(cfg)
att_mgr = AttentionCacheManager(pool, cfg)
cnn_mgr = CnnCacheManager(cfg)
ctc_mgr = CtcStateCacheManager(GpuDecoderConfig(beam_size=10))

sid = 42
att_mgr.allocate_stream(sid)
cnn_mgr.allocate_stream(sid)
ctc_mgr.allocate_stream(sid, batch=1, vocab_size=5000)
ctx = StreamContext(sid, att_mgr, cnn_mgr, ctc_mgr)

for chunk in audio_chunks:                    # produced by feature extractor
    ctx.prepare_chunk()
    att_caches = ctx.get_att_caches()         # List[PagedKVCache]
    cnn_cache  = ctx.get_cnn_cache()
    log_probs, new_cnn = model.forward_chunk_paged(
        chunk, offset, att_caches, cnn_cache, cache_t1=offset)
    ctx.commit_chunk_paged(log_probs.size(1), new_cnn)
    ctx.get_decoder().decode_chunk(log_probs)
    offset += log_probs.size(1)

result = ctx.get_decoder().finalize_stream()
ctx.free()
```

### 7.2 Dense mode (legacy)

```python
ctx = StreamContext(sid, att_mgr, cnn_mgr, ctc_mgr)

for chunk in audio_chunks:
    att_cache = ctx.get_att_cache()
    cnn_cache = ctx.get_cnn_cache()
    log_probs, new_att, new_cnn = model.forward_chunk(
        chunk, offset, required_cache_size, att_cache, cnn_cache)
    new_kv_chunk = new_att[:, :, -chunk_size:, :]
    ctx.commit_chunk(new_kv_chunk, new_cnn)
    ctx.get_decoder().decode_chunk(log_probs)
```

### 7.3 Batched paged forward (multi-stream, single encoder call)

```python
# All B streams share an offset and have a full window ready.
att_mgr.prepare_chunks_batched([req.stream_id for req in group])

block_tables, cache_seqlens = [], []
for req in group:
    bt, cs = req.stream_context.get_paged_state_views()
    block_tables.append(bt); cache_seqlens.append(cs)

batched_bt = torch.cat(block_tables, dim=0)
batched_cs = torch.cat(cache_seqlens, dim=0)

batched_caches = [PagedKVCache(
    k_cache=pool.get_kv_view(l)[0],
    v_cache=pool.get_kv_view(l)[1],
    block_table=batched_bt, cache_seqlens=batched_cs,
    block_size=cfg.block_size_frames, host_seqlen=offset)
    for l in range(cfg.num_layers)]

log_probs, new_cnn = model.forward_chunk_paged(
    xs_stacked, offset, batched_caches, cnn_stacked, cache_t1=offset)
```

## 8. Error Handling and Edge Cases

| Condition | Where | Behaviour |
|-----------|-------|-----------|
| Pool exhausted (`pool.allocate(n)` with `len(free) < n`) | `BlockPool.allocate` | Raises `RuntimeError` with the requested vs. free count. Caller (engine) must currently treat this as fatal — there is no built-in preemption. |
| Duplicate `allocate_stream(sid)` | All three managers | Raises `ValueError`. |
| Operating on a freed / unknown stream id | All three managers | `KeyError` from `_get_state` / `caches[sid]` / `states[sid]`. |
| `prepare_chunk` not called before `get_paged_caches` / `get_paged_state_views` | `AttentionCacheManager` | Raises `RuntimeError` — `block_table` is `None` until first prepare. |
| Shape mismatch in `commit` / `update` | `AttentionCacheManager.commit`, `CnnCacheManager.update` | Raises `ValueError` with both expected and actual shapes. |
| `chunk_frames > block_size_frames` in dense `commit` | `AttentionCacheManager.commit` | Raises `ValueError`. Dense mode requires one block per chunk. |
| First-chunk `(0, 0, 0, 0)` placeholder cnn_cache in batched forward | `ModelRunner._forward_batched_paged` | Skips the per-stream stack and passes the placeholder; the model handles it layer-wise. |
| Eviction inside batched forward | `commit_chunk_paged` | Per-stream eviction shifts each stream's `block_table` independently — no cross-stream interaction. |

`BlockPool.allocate` / `free` are guarded by a `threading.Lock`. The other
managers are **not** thread-safe internally. They are expected to be called
only from the engine's main step loop. `OfflinePipeline` runs CPU-side fbank
on a producer thread but never touches the cache managers; only the consumer
thread that performs the GPU forward calls into them.

## 9. Performance Considerations

1. **Pool over-provisioning.** Allocate enough `max_num_blocks` to cover
   peak concurrent demand. Falling back into a `RuntimeError` mid-stream
   is fatal; running out is the most common production failure mode.
2. **Avoid `cudaMalloc` in the hot loop.** Per-stream allocations
   (`block_table`, `cache_seqlens`) are lazy — they happen on first
   `prepare_chunk`. CTC states are pooled. The CNN cache tensor is
   allocated once at `allocate_stream`. After that, the streaming step
   does no allocator activity except the freelist popleft inside
   `BlockPool.allocate`.
3. **Use `prepare_chunks_batched`** when admitting or stepping a
   `B`-stream cohort. One pool lock + one CPU-side popleft loop is
   substantially cheaper than `B` independent prepares, even though the
   per-stream block-table writes are unavoidable.
4. **Cohort admission** (`EngineConfig.streaming_cohort_admit=True`)
   keeps `offset` aligned across the running pool so `_forward_batched_paged`
   can take the fast path. Without it, mismatched offsets fragment the
   batch into single-stream forwards.
5. **`gather_kv_blocks`** (dense mode) does an `index_select` per layer
   and is O(num_layers × num_blocks). Paged mode avoids this entirely —
   prefer it for long-context streams.
6. **`block_size_frames == chunk_size`** is the simplest invariant: each
   chunk is exactly one block, eviction matches chunk granularity, and
   the dense `commit` shape check trivially holds.

## 10. Extension Points

- **Custom stream identifiers.** Stream IDs are arbitrary integers — the
  engine assigns them monotonically, but external code can use any
  scheme as long as IDs are unique across active streams.
- **Alternative pool layouts.** `BlockPool` exposes `get_kv_view(layer)`
  and `get_kv_block_view(layer, block_id)` as the only tensor entry
  points. A subclass could change the underlying storage (e.g. swap
  K/V into separate per-layer tensors) without touching
  `AttentionCacheManager`.
- **Custom decoders.** `CtcStateCacheManager` is hard-wired to
  `GpuStreamingDecoder`. To plug in a different decoder you would mirror
  the `create_state` / `reset_state` / `decode_chunk` /
  `finalize_stream` interface and either subclass the manager or write a
  parallel one.
- **Per-stream metadata.** `_StreamKVState` is private. If you need to
  carry per-stream data (e.g. user-supplied tags, biasing graph
  pointers) prefer holding it on the engine `Request` object — keep the
  cache layer single-purpose.

## 11. Quick Reference

```text
allocate_stream(sid)         # all three managers
StreamContext(sid, ...)      # bundle them

per chunk:
  ctx.prepare_chunk()        # paged mode only
  ctx.get_att_cache[s]()     # dense | paged
  ctx.get_cnn_cache()
  ... model.forward_chunk[_paged](...) ...
  ctx.commit_chunk[_paged](...)
  ctx.get_decoder().decode_chunk(logits)

at end:
  ctx.get_decoder().finalize_stream()
  ctx.free()                 # releases blocks, CNN cache, pools CTC state
```
