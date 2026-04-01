# OASR Autotuning

Runtime kernel selection that benchmarks candidate backends and caches the fastest `(backend, tactic)` per operation and input signature. Follows the FlashInfer autotuner.py single-file pattern.

## Quick Start

```python
import oasr
from oasr.tune import autotune

# Profile and select the fastest backend on first encounter
with autotune(tune_mode=True, cache="oasr_tune.json"):
    output = oasr.conv2d(input, filter, pad_h=1, pad_w=1)

# Reuse cached results without profiling
with autotune(tune_mode=False, cache="oasr_tune.json"):
    output = oasr.conv2d(input, filter, pad_h=1, pad_w=1)
```

## How It Works

1. When `autotune()` is active, each operation checks the cache for a tuned config.
2. On **cache hit**, the cached backend/tactic is used directly.
3. On **cache miss** with `tune_mode=True`, all registered backends are benchmarked and the fastest is cached.
4. On **cache miss** with `tune_mode=False`, the fallback backend is used (CUTLASS default tile for most ops).
5. When the context manager exits, newly profiled configs are saved to the cache file.

## Tile-Variant Autotuning

CUTLASS kernels are compiled with specific tile sizes, warp layouts, and pipeline stages. The autotuner explores multiple tile configurations by JIT-compiling each as a separate shared library. Tile configs are defined in `kernel_configs.py`.

**Tunable parameters** (compile-time, one `.so` per unique combination):
- `tile_m`, `tile_n`, `tile_k` — Thread-block tile shape
- `warp_m`, `warp_n`, `warp_k` — Warp-level tile shape
- `stages` — Pipeline stages (async copy depth)

**Runtime parameters** (no recompilation needed):
- `split_k` — Split-K parallelism factor (GEMM only)

Configs that differ only in `split_k` share the same compiled module, reducing JIT compilation time.

### Kernel configurations

| Operation | Config list | Default | Count |
|-----------|------------|---------|-------|
| GEMM | `GEMM_TILE_CONFIGS` | 128x128x32, stages=2 | 11 (incl. split-K) |
| BMM | `BMM_TILE_CONFIGS` | Same as GEMM (no split-K) | 9 |
| Group GEMM | `GROUP_GEMM_TILE_CONFIGS` | Same as GEMM (no split-K) | 9 |
| Conv2D | `CONV2D_TILE_CONFIGS` | 128x128x64, stages=3 | 6 |

The C++ kernel headers use per-family config structs (e.g., `GemmConfig` in `include/oasr/gemm/cutlass_gemm_configs.h`, `Conv2dConfig` in `include/oasr/conv/cutlass_conv2d_configs.h`). When `-DOASR_GEMM_TILE_M=X` flags are passed during JIT compilation, the `JitGemmConfig` typedef overrides the default tile sizes. When no flags are set, `DefaultGemmConfig<SM>` provides per-SM defaults. For Jinja-based variants, tile configs are baked directly into generated source files.

## Cache Files

Cache files are JSON with environment metadata:

```json
{
  "_metadata": {
    "oasr_version": "0.1.0",
    "cuda_version": "12.4",
    "gpu": "NVIDIA A100-SXM4-80GB",
    "sm": "80"
  },
  "version": 1,
  "entries": {
    "conv|conv2d||(1,64,64,3,32,3,3,1,1,1,1,1,1)|float16|sm80": {
      "backend": "cudnn",
      "config": {},
      "median_ms": 0.042
    }
  }
}
```

- Configs are **merged incrementally**: multiple tuning sessions accumulate entries.
- Environment mismatches produce **warnings** (not errors).
- Files are written **atomically** (`tempfile` + `os.replace`).

## API Reference

```python
# Context manager (FlashInfer-style)
with autotune(tune_mode=True, cache="path.json", warmup=25, rep=100):
    ...

# Global toggle
enable_autotune(cache="path.json")
disable_autotune()

# Direct helpers
load_configs("path.json")    # Load cached configs
save_configs("path.json")    # Save current cache
clear_cache()                # Clear in-memory cache
get_selected_config(...)     # Query cached tactic for a profile

# Singleton access
tuner = AutoTuner.get()
tuner.stats                  # AutoTunerStatistics
tuner.reset_statistics()     # Reset stats counters
```

## Adding a New Backend

1. Create a file in `oasr/tune/backends/`, e.g. `tensorrtllm.py`.
2. Register entries using the global registry:

```python
from oasr.tune.autotuner import BackendEntry, _global_registry, OpKey, Tactic

def _trtllm_runner():
    # Return a callable with the same signature as the TVM-FFI module call
    ...

_global_registry.register(
    OpKey("gemm", "gemm"),
    BackendEntry(
        tactic=Tactic("tensorrtllm", (("algo", 3),)),
        is_available=lambda: _check_trtllm_available(),
        get_runner=_trtllm_runner,
        is_fallback=False,
    ),
)
```

3. Import the module in `backends/__init__.py`.

## Adding New Tile Configurations

Add entries to the config lists in `kernel_configs.py`:

```python
# In kernel_configs.py
GEMM_TILE_CONFIGS.append(
    TileConfig(tile_m=256, tile_n=256, tile_k=32,
               warp_m=64, warp_n=64, warp_k=32, stages=4)
)
```

Each `TileConfig` is automatically registered as a `BackendEntry` with a `Tactic("cutlass", config=...)` in the backend modules. The autotuner profiles them all and picks the fastest.

**Constraints** (CUTLASS 2.x TensorOp, SM80+, MMA 16x8x16):
- `tile_m % warp_m == 0` and `tile_n % warp_n == 0`
- `warp_m % 16 == 0` and `warp_n % 8 == 0`
- `tile_k % warp_k == 0` and `tile_k % 16 == 0`
- Total warps = `(tile_m/warp_m) x (tile_n/warp_n)`, typically 4 or 8

## Architecture

```
tune/
    __init__.py        # Re-exports from autotuner.py
    autotuner.py       # Single-file autotuner (FlashInfer-style)
                       #   Types: OpKey, ProfileKey, Tactic, TuneResult
                       #   Registry: BackendEntry, BackendRegistry
                       #   AutoTuner: singleton with cache, profiling, dispatch
                       #   Context manager: autotune()
                       #   Convenience API: enable/disable, load/save, etc.
    kernel_configs.py  # TileConfig dataclass + config lists
    backends/
        gemm.py        # CUTLASS tile variants (GEMM, BMM, Group GEMM)
        conv2d.py      # CUTLASS tile variants + cuDNN backends
```

Lookup priority: **in-memory cache -> user-loaded file configs -> bundled defaults -> fallback tactic**.
