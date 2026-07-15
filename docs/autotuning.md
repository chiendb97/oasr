# Autotuning

OASR provides FlashInfer-style runtime autotuning for CUTLASS kernels.
When enabled, the autotuner profiles multiple tile configurations for each
kernel call and caches the fastest one.  Subsequent calls with the same
shape, dtype, and GPU architecture reuse the cached result with zero
overhead.

## Quick start

```python
import oasr

with oasr.autotune():
    output = oasr.gemm(A, B)   # profiles tile variants on first call
    output = oasr.gemm(A, B)   # uses cached best — no profiling
```

## Persisting results to disk

Pass a `cache=` path to save and reload profiled configs across runs:

```python
# Run 1: profiles and saves
with oasr.autotune(cache="oasr_tune.json"):
    output = oasr.gemm(A, B)

# Run 2: loads cache, no profiling needed
with oasr.autotune(cache="oasr_tune.json"):
    output = oasr.gemm(A, B)
```

New results are merged with the existing file, so you can incrementally
tune different shapes across multiple runs.

## Cache-only mode (no profiling)

If you have a pre-built cache and want to skip profiling entirely:

```python
with oasr.autotune(False, cache="oasr_tune.json"):
    output = oasr.gemm(A, B)   # uses cached config or falls back to default
```

## Global toggle (no context manager)

For scripts that want to enable autotuning once at startup:

```python
oasr.enable_autotune(cache="oasr_tune.json")

output = oasr.gemm(A, B)       # auto-profiles on first call
output = oasr.conv2d(x, w, b, pad_h=1, pad_w=1)

oasr.disable_autotune()         # saves cache and disables
```

## Supported operations

Autotuning is available for operations backed by CUTLASS tile variants:

| Operation          | Notes                                          |
| ------------------ | ---------------------------------------------- |
| `gemm`             | Serial split-K (deep factors up to 16), parallel split-K, Stream-K, thin-N tiles, plus a torch/cuBLAS candidate |
| `gemm_activation`  | Fused GEMM + activation — serial split-K > 1 excluded (per-partition activation is invalid); parallel split-K variants apply the activation once post-reduction |
| `gemm_log_softmax` | The CTC head: every GEMM variant composed with the OASR online log_softmax kernel, the legacy single-call fused launcher (`Tactic("cutlass_fused")`, the fallback), and a torch candidate |
| `bmm`              | Batched GEMM (tile-only; no split-K / Stream-K) + torch candidate |
| `group_gemm`       | Grouped GEMM (tile-only)                       |
| `conv2d`           | NHWC implicit GEMM                             |
| `conv2d_activation` | Fused Conv2D + activation                     |

In addition to the CUTLASS tile variants, the GEMM ops register a **torch/cuBLAS**
candidate (`Tactic("torch")`) — cuBLAS wins on some thin shapes — and, on SM120,
two K-decompositions that fill the GPU on thin/deep-K GEMMs where the
data-parallel grid leaves SMs idle:

* **Stream-K** (`stream_k=1`; `GemmUniversal` + `ThreadblockSwizzleStreamK`) —
  balances the K-reduction across all SMs with an in-kernel fixup.  Set
  `OASR_GEMM_STREAMK=0` to exclude from the compile set.
* **Parallel split-K** (`parallel_split_k=1`; `GemmSplitKParallel`) — fp32
  partials + a reduction kernel that applies the (possibly activation-fused)
  epilogue exactly once.  Set `OASR_GEMM_SPLITK_PARALLEL=0` to exclude.

Serial split-K (`split_k > 1` on a plain config) runs as a **single kernel
launch**: the per-tile semaphores live in a persistent pre-zeroed workspace
(`include/oasr/common/workspace_cache.h`) and the kernel restores them itself,
so there is no per-launch allocation or memset (that ritual costs ~4 µs — the
historical CUTLASS-vs-cuBLAS gap on thin ASR GEMMs).  Set
`OASR_GEMM_WS_CACHE=0` to fall back to per-call workspaces.

Operations without tile variants (e.g., `depthwise_conv1d`, `layer_norm`)
are unaffected by autotuning and always use their default kernel.

## How it works

1. Each tunable kernel call checks a fast global flag (`is_tuning_enabled()`).
2. On cache hit: the cached tile config is used immediately.
3. On cache miss (with profiling enabled): all registered tile variants
   are JIT-compiled (if not already cached on disk) and benchmarked.
   The fastest is stored and used.
4. On cache miss (profiling disabled): the default config is used.

Cache keys include the operation name, shape signature, dtype, and GPU
SM version, so configs are portable across runs on the same GPU but
will be re-profiled if the hardware changes.

## Advanced parameters

The `autotune()` context manager and `enable_autotune()` accept optional
keyword arguments for fine-grained control:

```python
with oasr.autotune(
    cache="tune.json",
    warmup=50,          # warmup iterations before measurement (default: 25)
    rep=200,            # measurement iterations (default: 100)
    log_level="DEBUG",  # logging verbosity (default: "INFO")
):
    ...
```

## Benchmarking with autotuning

The GEMM benchmark script supports autotuning directly:

```bash
# Profile all tile variants and show comparison
python benchmarks/bench_gemm.py --autotune

# Save results to a cache file
python benchmarks/bench_gemm.py --autotune --cache gemm_tune.json

# Replay cached configs (no profiling overhead)
python benchmarks/bench_gemm.py --autotune --cache gemm_tune.json --no-tune
```

## Cache file format

The cache is a JSON file containing environment metadata and per-shape
entries:

```json
{
  "version": 1,
  "env": {
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "sm": 80,
    "cuda_version": "12.4",
    "oasr_version": "0.1.0"
  },
  "entries": {
    "gemm|gemm||(16000,256,2048)|float16|sm80": {
      "backend": "cutlass",
      "config": {
        "tile_m": 128, "tile_n": 128, "tile_k": 64,
        "warp_m": 64, "warp_n": 64, "warp_k": 64,
        "stages": 3
      },
      "median_ms": 0.432,
      "profiled_at": "2026-03-23T06:14:00+00:00"
    }
  }
}
```

## Design choices vs FlashInfer

| Aspect                | FlashInfer                    | OASR                                                  |
| --------------------- | ----------------------------- | ----------------------------------------------------- |
| Primary API           | `flashinfer.autotune()`       | `oasr.autotune()` (identical pattern)                 |
| Global toggle         | Context manager only          | Also `enable_autotune()` / `disable_autotune()`       |
| Profiling params      | Hidden                        | Optional keyword-only (`warmup`, `rep`)                |
| Cache mismatch        | Skips entire cache            | Warns but continues (less disruptive)                  |
| Tile variants         | Internal                      | Listed in `oasr.tune.kernel_configs`                   |
