# OASR Perf Benchmarking Framework -- `oasr_benchmark.py`

The aim of `oasr_benchmark.py` is to provide a single framework for benchmarking any OASR kernel and replace standalone benchmarking scripts.

## Overview

This framework provides tools to:
- Benchmark OASR's GEMM, Normalization, Convolution, Activation, and Composite kernel performance against PyTorch baselines
- Compare performance across different configurations and backends
- Batch performance test multiple test cases from testlist files
- Export results to CSV for analysis

Currently supports testing the following kernel families:
- GEMM:
    - `gemm` - General matrix multiply (CUTLASS). Supports autotuning.
    - `bmm` - Batched matrix multiply (CUTLASS).
    - `group_gemm` - Grouped GEMM (CUTLASS).
    - `gemm_activation` - Fused GEMM + activation (CUTLASS epilogue).
- Normalization:
    - `layer_norm` - Layer normalization.
    - `add_layer_norm` - Fused residual add + layer normalization.
    - `layer_norm_activation` - Fused layer normalization + activation.
    - `rms_norm` - Root mean square normalization.
    - `batch_norm` - Batch normalization (1D, inference mode).
    - `batch_norm_swish` - Fused batch normalization + Swish activation.
    - `batch_norm_activation` - Fused batch normalization + configurable activation.
    - `group_norm` - Group normalization.
- Convolution:
    - `depthwise_conv1d` - Depthwise 1D convolution (NLC layout).
    - `depthwise_conv1d_causal` - Causal depthwise 1D convolution.
    - `pointwise_conv1d` - Pointwise 1D convolution (equivalent to linear projection).
    - `pointwise_conv1d_activation` - Fused pointwise conv1d + activation.
    - `conv2d` - 2D convolution (NHWC layout via CUTLASS Implicit GEMM).
    - `conv2d_activation` - Fused 2D convolution + activation.
- Activation:
    - `glu` - Gated Linear Unit.
    - `swish` - Swish / SiLU activation.
- Composite:
    - `conv_block` - End-to-end Conformer convolution block (pointwise conv -> GLU -> depthwise conv -> Swish -> pointwise conv).

## Quick Start

### Single Test Run

A test case is generally invoked as `python benchmarks/oasr_benchmark.py --routine <routine> --subroutine <subroutine> <flags>`.

```bash
# GEMM
$ python benchmarks/oasr_benchmark.py --routine gemm --subroutine bmm \
    --backends cutlass torch --batch-count 256 --M 200 --N 200 --K 64 \
    --dtype float16 --refcheck -vv --generate_repro_command

[INFO] gemm/bmm
[VVERBOSE] gpu_name = 'NVIDIA_A100_80GB_PCIe'
[REPRO] python benchmarks/oasr_benchmark.py --routine gemm --subroutine bmm --backends cutlass torch --dtype float16 --num_iters 30 --dry_run_iters 5 --refcheck --batch-count 256 --M 200 --N 200 --K 64
[PERF] cutlass      :: median time 0.145 ms; std 0.002 ms; achieved tflops 125.3 TFLOPs/sec
[PERF] torch        :: median time 0.168 ms; std 0.003 ms; achieved tflops 108.1 TFLOPs/sec

# LayerNorm
$ python benchmarks/oasr_benchmark.py --routine norm --subroutine layer_norm \
    --backends cuda torch --batch 64 --seq 250 --hidden 512 --refcheck -vv

[INFO] norm/layer_norm
[PERF] cuda         :: median time 0.023 ms; std 0.001 ms; achieved tb_per_sec 2.87 TB/sec
[PERF] torch        :: median time 0.031 ms; std 0.001 ms; achieved tb_per_sec 2.13 TB/sec

# Depthwise Conv1D
$ python benchmarks/oasr_benchmark.py --routine conv --subroutine depthwise_conv1d \
    --backends cuda torch --batch 64 --seq 250 --channels 256 --kernel-size 15 \
    --refcheck -vv

[INFO] conv/depthwise_conv1d
[PERF] cuda         :: median time 0.018 ms; std 0.001 ms
[PERF] torch        :: median time 0.025 ms; std 0.001 ms

# End-to-end Conformer conv block
$ python benchmarks/oasr_benchmark.py --routine composite --subroutine conv_block \
    --backends cuda torch --batch 64 --seq 250 --d-model 256 --kernel-size 15 \
    --refcheck -vv

[INFO] composite/conv_block
[PERF] cuda         :: median time 0.089 ms; std 0.002 ms
[PERF] torch        :: median time 0.142 ms; std 0.003 ms
```

### List Available Routines

```bash
$ python benchmarks/oasr_benchmark.py --list

Available routines and subroutines:

  gemm:
    - gemm
    - bmm
    - group_gemm
    - gemm_activation
  norm:
    - layer_norm
    - add_layer_norm
    - layer_norm_activation
    - rms_norm
    - batch_norm
    - batch_norm_swish
    - batch_norm_activation
    - group_norm
  conv:
    - depthwise_conv1d
    - depthwise_conv1d_causal
    - pointwise_conv1d
    - pointwise_conv1d_activation
    - conv2d
    - conv2d_activation
  activation:
    - glu
    - swish
  composite:
    - conv_block
```

### Batch Testing

Run multiple tests from a file and save results:

```bash
python benchmarks/oasr_benchmark.py \
    --testlist benchmarks/testlists/conformer_base.txt \
    --output_path results.csv \
    --generate_repro_command \
    --refcheck
```

See `benchmarks/testlists/conformer_base.txt` and `benchmarks/testlists/all_kernels.txt` for example testlist files.

The output CSV contains:
- Routine, subroutine, backend, shape, dtype
- Median execution time and standard deviation
- TFLOPS/sec (compute-bound kernels) or TB/sec (memory-bound kernels)
- Device info, case tag, and reproducer commands

## Command Line Arguments

### General Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--routine` | Kernel routine family (gemm, norm, conv, activation, composite) | required |
| `--subroutine` | Specific kernel within the routine (e.g. bmm, layer_norm) | first available |
| `--backends` | Backend(s) to benchmark (space-separated). Per-family defaults: gemm/conv2d → `cutlass torch`; norm/conv1d/activation/composite → `cuda torch` | (auto) |
| `--dtype` | Data type (float16, bfloat16, float32) | float16 |
| `--num_iters` | Number of iterations for performance measurement | 30 |
| `--dry_run_iters` | Number of warmup iterations | 5 |
| `--refcheck` | Verify outputs match between backends | False |
| `--allow_output_mismatch` | Continue testing even if outputs don't pass refcheck | False |
| `--use_cuda_events` | Force CUDA events timing (skip Triton do_bench) | False |
| `--output_path` | Path to save CSV results | None |
| `--testlist` | Path to a file containing a list of test cases | None |
| `--profile` | Run in profiling mode (NVTX markers for Nsight Compute) | False |
| `-v`, `-vv` | Increase verbosity (can be used multiple times) | 0 |
| `--case_tag` | Optional tag for annotating results in the output CSV | None |
| `--generate_repro_command` | Print a reproducer command for each test case | False |
| `--list` | List all available routines and subroutines | False |

### GEMM Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--M` | M dimension | None (uses defaults) |
| `--N` | N dimension | None |
| `--K` | K dimension | None |
| `--batch-count` | Batch count for bmm / group_gemm | None |
| `--autotune` | Enable kernel autotuning (gemm subroutine only) | False |
| `--cache` | Path to autotune cache file | None |
| `--no-tune` | Use cached configs only, skip tuning | False |

### Normalization Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--batch` | Batch size | None (uses defaults) |
| `--seq` | Sequence length | None |
| `--hidden` | Hidden / channel dimension | None |
| `--num-groups` | Number of groups (group_norm) | 32 |
| `--activation` | Activation for fused variants (relu, gelu, swish) | swish |

### Convolution Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--batch` | Batch size | None (uses defaults) |
| `--seq` | Sequence length (1D conv) | None |
| `--channels` | Input channels | None |
| `--out-channels` | Output channels (pointwise conv1d) | None |
| `--kernel-size` | Kernel size (1D conv) | None |
| `--height` | Input height (2D conv) | None |
| `--width` | Input width (2D conv) | None |
| `--out-filters` | Output filters (2D conv) | None |
| `--filter-h` | Filter height (2D conv) | 3 |
| `--filter-w` | Filter width (2D conv) | 3 |
| `--pad` | Padding | 0 |
| `--stride` | Stride | 1 |
| `--activation` | Activation for fused variants (swish, relu, gelu) | None |

### Activation Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--batch` | Batch size | None (uses defaults) |
| `--seq` | Sequence length | None |
| `--channels` | Channel dimension | None |

### Composite Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--batch` | Batch size | None (uses defaults) |
| `--seq` | Sequence length | None |
| `--d-model` | Model dimension | None |
| `--kernel-size` | Depthwise conv kernel size | None |

## Timing

OASR uses two timing methods:

1. **Triton `do_bench` (preferred)** -- Uses Triton's built-in benchmarking utility for accurate median timing with automatic warmup. Requires `triton` to be installed.
2. **CUDA Events (fallback)** -- Standard CUDA event timing. Automatically used if Triton is not available, or when `--use_cuda_events` is specified.

The framework automatically selects the best available method.

## Routine & Backend Support Matrix

| Routine | Subroutines | Backends | Metric | SM 7.0+ | SM 8.0+ |
|---------|-------------|----------|--------|---------|---------|
| **gemm** | gemm, bmm, group_gemm, gemm_activation | `cutlass`, `torch` | TFLOPS | yes | yes |
| **norm** | layer_norm, add_layer_norm, layer_norm_activation, rms_norm, batch_norm, batch_norm_swish, batch_norm_activation, group_norm | `cuda`, `torch` | TB/sec | yes | yes |
| **conv** | depthwise_conv1d, depthwise_conv1d_causal | `cuda`, `torch` | time | yes | yes |
| **conv** | pointwise_conv1d, pointwise_conv1d_activation | `cuda`, `torch` | TFLOPS | yes | yes |
| **conv** | conv2d, conv2d_activation | `cutlass`, `torch` | TFLOPS | yes | yes |
| **activation** | glu, swish | `cuda`, `torch` | TB/sec | yes | yes |
| **composite** | conv_block | `cuda`, `torch` | time | yes | yes |

Backend legend:
- **cutlass**: CUTLASS-based OASR kernels (GEMM, Conv2D), JIT-compiled via TVM-FFI.
- **cuda**: Handwritten CUDA OASR kernels (Norm, Conv1D, Activation), JIT-compiled via TVM-FFI.
- **torch**: Standard PyTorch operations used as the reference baseline.

## Testlist Files

Testlist files contain one test case per line. Each line is a set of CLI arguments (parsed with `shlex.split`). Lines starting with `#` are comments. Empty lines are ignored.

### `testlists/conformer_base.txt`

Conformer-base workload (d_model=256, kernel_size=15) -- typical shapes from a WeNet Conformer-base ASR model:

```
# Normalization
--routine norm --subroutine layer_norm --batch 64 --seq 250 --hidden 256
--routine norm --subroutine add_layer_norm --batch 64 --seq 250 --hidden 256
--routine norm --subroutine batch_norm --batch 64 --seq 250 --hidden 256

# Convolution
--routine conv --subroutine depthwise_conv1d --batch 64 --seq 250 --channels 256 --kernel-size 15
--routine conv --subroutine pointwise_conv1d --batch 64 --seq 250 --channels 256 --out-channels 512

# Activation
--routine activation --subroutine glu --batch 64 --seq 250 --channels 256
--routine activation --subroutine swish --batch 64 --seq 250 --channels 256

# GEMM (linear projections)
--routine gemm --subroutine gemm --M 16000 --N 256 --K 256

# Composite
--routine composite --subroutine conv_block --batch 64 --seq 250 --d-model 256 --kernel-size 15
```

### `testlists/all_kernels.txt`

One representative configuration per subroutine across all kernel families.

## Profiling

Use `--profile` with `--dry_run_iters 0` to run a single kernel iteration wrapped in NVTX markers, then invoke `ncu` or `nsys` around the script:

### Nsight Compute (ncu)

```bash
# GEMM — generates gemm_profile.ncu-rep
ncu --set full -o gemm_profile \
    python benchmarks/oasr_benchmark.py \
    --routine gemm --subroutine gemm \
    --M 200 --N 200 --K 64 --backends cutlass \
    --profile --dry_run_iters 0

# LayerNorm
ncu --set full -o layer_norm_profile \
    python benchmarks/oasr_benchmark.py \
    --routine norm --subroutine layer_norm \
    --batch 64 --seq 250 --hidden 512 --backends cuda \
    --profile --dry_run_iters 0
```

Open the generated `.ncu-rep` file in the Nsight Compute UI to inspect kernel metrics.

### Nsight Systems (nsys)

```bash
nsys profile python benchmarks/oasr_benchmark.py \
    --routine conv --subroutine depthwise_conv1d \
    --batch 64 --seq 250 --channels 256 --kernel-size 15 \
    --profile --dry_run_iters 0
```

Profiling mode runs a single iteration with NVTX range markers (named `<backend>_<subroutine>`), making it easy to isolate the kernel of interest in the profiler output. Problem size is controlled by the same flags used for regular benchmarking (`--batch`, `--seq`, `--hidden`, `--M`, `--N`, `--K`, etc.).

## Legacy Standalone Scripts

Individual `bench_*.py` scripts are preserved as thin wrappers for backwards compatibility. They delegate to the same routine modules used by `oasr_benchmark.py`:

| Script | Routine | Subroutines |
|--------|---------|-------------|
| `bench_gemm.py` | gemm | gemm, gemm_activation |
| `bench_bmm.py` | gemm | bmm |
| `bench_group_gemm.py` | gemm | group_gemm |
| `bench_layer_norm.py` | norm | layer_norm, add_layer_norm, layer_norm_activation |
| `bench_rms_norm.py` | norm | rms_norm |
| `bench_batch_norm.py` | norm | batch_norm, batch_norm_swish, batch_norm_activation |
| `bench_group_norm.py` | norm | group_norm |
| `bench_depthwise_conv1d.py` | conv | depthwise_conv1d, depthwise_conv1d_causal |
| `bench_pointwise_conv1d.py` | conv | pointwise_conv1d, pointwise_conv1d_activation |
| `bench_conv2d.py` | conv | conv2d, conv2d_activation |
| `bench_glu.py` | activation | glu |
| `bench_swish.py` | activation | swish |
| `bench_conv_block.py` | composite | conv_block |

These support `--profile`, `--kernel`, `--target`, `--warmup`, and `--iters` flags via the legacy API.

## Directory Structure

```
benchmarks/
  oasr_benchmark.py              # Unified CLI entry point
  routines/
    __init__.py                   # Routine registry
    bench_utils.py                # Shared infrastructure (timing, metrics, output)
    gemm.py                       # GEMM family routines
    norm.py                       # Normalization family routines
    conv.py                       # Convolution family routines
    activation.py                 # Activation family routines
    composite.py                  # Composite / end-to-end routines
  testlists/
    conformer_base.txt            # Conformer-base workload
    all_kernels.txt               # All subroutines
  bench_*.py                      # Legacy standalone wrappers
  utils.py                        # Legacy utils (delegates to routines/bench_utils.py)
  __init__.py
```

## Prerequisites

- **Python >= 3.8** with PyTorch (CUDA-enabled)
- **OASR** installed in the environment (`pip install -e .` from the project root)
- **CUDA GPU** — all benchmarks run on GPU; tests are skipped if CUDA is not available
- **Triton** (optional, recommended) — provides more accurate timing via `triton.testing.do_bench`. Falls back to CUDA events if not installed.
- **Nsight Compute / Nsight Systems** (optional) — for `--profile` mode

## CSV Output Format

When `--output_path` is specified, results are written as CSV with the following columns:

| Column | Description |
|--------|-------------|
| `routine` | Kernel routine family (e.g. `gemm`, `norm`, `conv`) |
| `subroutine` | Specific kernel (e.g. `bmm`, `layer_norm`, `depthwise_conv1d`) |
| `backend` | Backend used (e.g. `cutlass`, `cuda`, `torch`) |
| `shape` | Shape string describing the test configuration (e.g. `B=256_M=200_N=200_K=64`) |
| `dtype` | Data type used (e.g. `float16`, `bfloat16`, `float32`) |
| `median_ms` | Median kernel execution time in milliseconds |
| `std_ms` | Standard deviation of execution time in milliseconds |
| `tflops` | Achieved TFLOPS for compute-bound kernels (GEMM, conv2d, pointwise conv); empty for memory-bound kernels |
| `bandwidth_tb_s` | Achieved memory bandwidth in TB/s for memory-bound kernels (norms, activations); empty for compute-bound kernels |
| `device` | GPU device name (e.g. `NVIDIA A100-SXM4-80GB`) |
| `case_tag` | User-provided tag from `--case_tag` (useful for grouping results) |
| `repro_command` | Full CLI command to reproduce this test case (if `--generate_repro_command` was set) |

Example analysis with pandas:

```python
import pandas as pd

df = pd.read_csv("results.csv")

# Compare OASR vs PyTorch for each kernel
pivot = df.pivot_table(
    index=["routine", "subroutine", "shape"],
    columns="backend",
    values="median_ms",
)
pivot["speedup"] = pivot["torch"] / pivot["cutlass"]  # or "cuda" for norm/conv1d
print(pivot.sort_values("speedup", ascending=False))
```

## Adding a New Benchmark Routine

To add a new kernel family or subroutine to the benchmark framework, follow this pattern (mirrors the FlashInfer convention):

### 1. Add a routine module

Create `benchmarks/routines/<family>.py` (or extend an existing one). Each module must expose:

```python
# List of subroutine names this module supports
SUBROUTINES: list[str] = ["my_kernel", "my_kernel_fused"]

# Optional: default configs for profiling mode
PROFILE_CONFIGS: dict[str, tuple] = {
    "my_kernel": (64, 250, 256),  # (batch, seq, hidden)
}

def parse_args(parser: argparse.ArgumentParser) -> None:
    """Add routine-specific CLI arguments."""
    parser.add_argument("--my-dim", type=int, default=None)

def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    """Run a single benchmark test for this routine."""
    ...

# Optional: setup functions for profiling mode
def setup_my_kernel(batch, seq, hidden):
    """Return (oasr_fn, pytorch_fn) callables for profiling."""
    ...
```

### 2. Register the routine

Add the module to the registry in `benchmarks/routines/__init__.py`:

```python
ROUTINE_REGISTRY: dict[str, str] = {
    ...
    "my_family": "benchmarks.routines.my_family",
}
```

### 3. Implement run_test

Inside `run_test`, use the shared infrastructure from `bench_utils.py`:

```python
from benchmarks.routines.bench_utils import (
    BenchResult, OutputWriter, bench_fn, check_close,
    compute_gemm_tflops, compute_bandwidth_tb_s, parse_dtype,
)

def run_test(args, output: OutputWriter):
    dtype = parse_dtype(args.dtype)
    # ... set up tensors and kernels ...
    fn_map = get_fn_map(args.subroutine, oasr_fn, pytorch_fn)
    backends = args.backends or list(fn_map.keys())

    for backend in backends:
        fn = fn_map[backend]
        median_ms, std_ms = bench_fn(
            fn,
            dry_run_iters=args.dry_run_iters,
            num_iters=args.num_iters,
            use_cuda_events=args.use_cuda_events,
        )
        result = BenchResult(
            routine="my_family",
            subroutine=args.subroutine,
            backend=backend,
            shape=f"B={batch}_S={seq}_H={hidden}",
            dtype=args.dtype,
            median_ms=median_ms,
            std_ms=std_ms,
            tflops=compute_gemm_tflops(M, N, K, median_ms),  # or None
            bandwidth_tb_s=compute_bandwidth_tb_s(nbytes, median_ms),  # or None
        )
        output.write_result(result)
```

### 4. Add a legacy wrapper (optional)

For backwards compatibility, add `benchmarks/bench_<name>.py` that delegates to the routine module:

```python
from benchmarks.routines.bench_utils import run_main
from benchmarks.routines.my_family import PROFILE_CONFIGS, setup_my_kernel, benchmark

setup_funcs = {"my_kernel": lambda: setup_my_kernel(*PROFILE_CONFIGS["my_kernel"])}

if __name__ == "__main__":
    run_main("My Family Kernels", PROFILE_CONFIGS, setup_funcs, benchmark)
```

### 5. Add test cases to testlists

Add representative configurations to `benchmarks/testlists/all_kernels.txt`:

```
--routine my_family --subroutine my_kernel --my-dim 256 --batch 64 --seq 250
```

## Troubleshooting

### JIT compilation errors

OASR kernels are JIT-compiled on first use. If you see compilation errors:

```bash
# Clear JIT cache and retry
rm -rf ~/.cache/oasr/jit/
python benchmarks/oasr_benchmark.py --routine gemm --subroutine gemm
```

### Triton not found

If Triton is not installed, the framework automatically falls back to CUDA events timing. To install Triton:

```bash
pip install triton
```

Or explicitly use CUDA events:

```bash
python benchmarks/oasr_benchmark.py --routine gemm --use_cuda_events
```

### Refcheck failures

Small numerical differences between OASR and PyTorch are expected (especially for FP16). Use `--allow_output_mismatch` to continue benchmarking despite refcheck failures:

```bash
python benchmarks/oasr_benchmark.py --routine gemm --refcheck --allow_output_mismatch
```

### CUDA out of memory

Reduce batch size or sequence length. For large GEMM shapes, try smaller dimensions:

```bash
python benchmarks/oasr_benchmark.py --routine gemm --M 4096 --N 4096 --K 256
```

### Noisy timing results

If `std_ms` is high relative to `median_ms`, increase measurement iterations:

```bash
python benchmarks/oasr_benchmark.py --routine norm --subroutine layer_norm \
    --num_iters 100 --dry_run_iters 20
```
