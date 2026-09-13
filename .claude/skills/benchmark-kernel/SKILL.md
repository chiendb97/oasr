---
name: benchmark-kernel
description: Guide for benchmarking and profiling OASR kernels for performance optimization
---

# Benchmarking and Profiling OASR Kernels

A practical guide to measuring kernel performance, identifying bottlenecks, and validating optimizations using OASR's benchmark and profiling tools.

## Setup

### Install timing backend (recommended)

```bash
pip install triton
```

OASR supports two timing methods, selected automatically:

| Aspect | Triton `do_bench` (`--timer triton`) | CUDA Events (default) |
|--------|-------------------------------|------------------------|
| **Accuracy** | Highest (CUPTI-based hardware timing) | Good (slight sync overhead) |
| **Installation** | `pip install triton` | Built-in with CUDA |
| **Best for** | Sub-millisecond kernels where overhead matters | Any kernel, any CUDA version |
| **Fallback** | N/A | Automatic if Triton unavailable |

**CUDA events are the default**, and every statistic in a row — median, mean, σ,
min, p99 — comes from that one timed loop. `--timer triton` delegates the loop to
`do_bench`, which reads `--iters` as a *millisecond budget* rather than a count.

`--iters` is a floor, not a cap: a 30 µs kernel measured for exactly 30
iterations spans 1 ms, which is not long enough for the part to leave its idle
clocks, and it measures ~1.4× slow. `--min-measure-ms` (default 20) raises the
count until the measurement spans enough wall time, and the row records how many
iterations actually ran.

### Verify benchmark tool

```bash
# List all available routines and subroutines
python benchmarks/run.py --list
```

## Quick Start: Measure a Kernel

### Method 1: Unified CLI (recommended for systematic benchmarking)

```bash
python benchmarks/run.py --family gemm --subroutine bmm \
    --backends cutlass torch \
    --batch-count 256 --M 200 --N 200 --K 64 \
    --dtype float16 --refcheck -vv
```

### Method 2: Python `bench_gpu_time()` (recommended for ad-hoc measurement)

```python
import torch
from oasr.testing import bench_gpu_time
import oasr

x = torch.randn(64, 250, 512, dtype=torch.float16, device="cuda")

# Triton preferred, CUDA events fallback -- automatic
median_s, std_s = bench_gpu_time(
    oasr.layer_norm,
    args=(x, torch.ones(512, device="cuda"), None, 1e-5),
    enable_cupti=True,       # prefer Triton/CUPTI, fallback to CUDA events
    dry_run_iters=10,        # warmup
    repeat_iters=100,        # measurement iterations
)
print(f"Median: {median_s*1e3:.3f} ms, Std: {std_s*1e3:.3f} ms")

# Force CUDA events (skip Triton even if installed)
median_s, std_s = bench_gpu_time(
    oasr.layer_norm,
    args=(x, torch.ones(512, device="cuda"), None, 1e-5),
    enable_cupti=False,      # explicitly use CUDA events
    repeat_iters=100,
)
```

**Note**: `bench_gpu_time` returns seconds. Multiply by 1e3 for milliseconds.

## Backend Names

Backend names differ by kernel family. Using the wrong backend name causes an error:

| Routine | Subroutines | Backends | Performance Metric |
|---------|-------------|----------|--------------------|
| **gemm** | gemm, bmm, bmm_strided, group_gemm, gemm_activation, gemm_log_softmax | `cutlass`, `torch` | TFLOPS |
| **conv** | conv2d, conv2d_activation | `cutlass`, `torch` | TFLOPS |
| **norm** | layer_norm, add_layer_norm, rms_norm, batch_norm, group_norm, and fused variants | `cuda`, `torch` | TB/sec |
| **conv** | depthwise_conv1d, depthwise_conv1d_causal, pointwise_conv1d, pointwise_conv1d_activation | `cuda`, `torch` | time / TFLOPS |
| **activation** | glu, swish | `cuda`, `torch` | TB/sec |
| **composite** | conv_block | `cuda`, `torch` | TFLOPS |
| **attention** | fmha_offline, fmha_bias, fmha_seqlens, fmha_bias_seqlens, fmha_paged, fmha_paged_bias | `cutlass`, `torch` | TFLOPS |
| **recurrent** | lstm, rnn_tanh, rnn_relu, lstm_slot_step, lstm_step_cute | `oasr`, `native`, `cutlass*`, `cudnn` | TFLOPS |
| **mlp** | gated_mlp | `cute`, `oasr`, `torch` | TFLOPS + TB/s |
| **feature** | fbank_preprocess, mel_log, dct_lifter, fbank_pipeline, mfcc_pipeline | `cuda`, `torch`, `torchaudio` | TB/s |

- **`cutlass`** -- CUTLASS-based OASR kernels, JIT-compiled via TVM-FFI (GEMM, Conv2D)
- **`cuda`** -- Handwritten CUDA OASR kernels, JIT-compiled via TVM-FFI (Norm, Conv1D, Activation)
- **`torch`** -- PyTorch reference baseline

## Running Benchmarks

### Single kernel

```bash
# GEMM
python benchmarks/run.py --family gemm --subroutine bmm \
    --backends cutlass torch \
    --batch-count 256 --M 200 --N 200 --K 64 \
    --dtype float16 --refcheck -vv

# LayerNorm
python benchmarks/run.py --family norm --subroutine layer_norm \
    --backends cuda torch \
    --batch 64 --seq 250 --hidden 512 \
    --refcheck -vv

# Depthwise Conv1D
python benchmarks/run.py --family conv --subroutine depthwise_conv1d \
    --backends cuda torch \
    --batch 64 --seq 250 --channels 256 --kernel-size 15 \
    --refcheck -vv

# Conv2D
python benchmarks/run.py --family conv --subroutine conv2d \
    --backends cutlass torch \
    --batch 16 --height 200 --width 80 \
    --channels 1 --out-filters 64 --filter-h 3 --filter-w 3 \
    --refcheck -vv

# Conformer conv block (end-to-end composite)
python benchmarks/run.py --family composite --subroutine conv_block \
    --backends cuda torch \
    --batch 64 --seq 250 --d-model 256 --kernel-size 15 \
    --refcheck -vv
```

When no shape flags are given, the routine runs its built-in default configs (representative ASR shapes).

### Batch benchmarking (testlist files)

Testlist files contain one CLI invocation per line (`#` lines are comments):

```bash
python benchmarks/run.py \
    --testlist benchmarks/testlists/conformer_base.txt \
    --output results.csv \
    --refcheck \
    -v
```

See `benchmarks/testlists/conformer_base.txt` (Conformer-base workload: d_model=256, kernel_size=15) and `all_kernels.txt` (one representative config per subroutine) for examples.

### Save results to CSV

Add `--output results.csv` to any run. The column list is declared once in
`benchmarks/core/schema.py`; print it with `python benchmarks/run.py --print-schema`
rather than transcribing it. Beside the CSV you also get `<out>.meta.json` (argv,
git commit, device, library versions, every `OASR_*` switch, and the JIT cache
hashes — so "was the JIT cache fresh?" is answerable after the fact).

Kernel rows carry `speedup_vs_ref` directly, so an OASR-vs-torch comparison no
longer needs a pandas pivot.

Use `--case-tag <label>` to annotate rows when comparing across kernel versions or experiments.

## Understanding the Output

```
[INFO] gemm/bmm
[VVERBOSE] gpu_name = 'NVIDIA_A100_80GB_PCIe'
[VVERBOSE] sm = 8.0, memory = 79.4 GB
[REPRO] python benchmarks/run.py --family gemm --subroutine bmm \
    --backends cutlass torch --dtype float16 --iters 30 --warmup-iters 5 \
    --refcheck --batch-count 256 --M 200 --N 200 --K 64
[PERF] cutlass       :: median time 0.145 ms; std 0.002 ms; achieved tflops 125.3 TFLOPs/sec
[PERF] torch         :: median time 0.168 ms; std 0.003 ms; achieved tflops 108.1 TFLOPs/sec
```

**Key metrics and what they tell you:**

| Metric | What it measures | How to interpret |
|--------|-----------------|------------------|
| `median time` | Median kernel execution time (ms) | Primary latency metric -- lower is better |
| `std` | Standard deviation across iterations | High std (> 5% of median) signals noisy GPU or insufficient warmup |
| `achieved tflops` | Compute throughput (TFLOPS/sec) | Compare against GPU peak TFLOPS to gauge compute utilization |
| `achieved tb_per_sec` | Memory bandwidth (TB/sec) | Compare against GPU peak HBM bandwidth to gauge memory utilization |

**Verbosity levels**: `-v` prints input shapes and dtypes. `-vv` additionally prints GPU name, SM version, and memory size.

## Performance Analysis Workflow

A systematic approach to kernel optimization:

### 1. Establish baseline

Run with `--refcheck -vv -v` to get a reproducible, correctness-verified baseline:

```bash
python benchmarks/run.py --family gemm --subroutine gemm \
    --backends cutlass torch --M 16000 --N 512 --K 2048 \
    --output baseline.csv --case-tag baseline \
    --refcheck -vv -v
```

### 2. Identify the bottleneck

The reported metric hints at the bottleneck type:

| Reported metric | Likely bottleneck | GPU peak reference (A100 SXM4) |
|-----------------|-------------------|-------------------------------|
| **TFLOPS** (GEMM, Conv2D, pointwise conv) | Compute-bound | ~312 TFLOPS (FP16 Tensor Core) |
| **TB/sec** (Norm, Activation, depthwise conv) | Memory-bound | ~2 TB/s HBM bandwidth |

**How to read the numbers:**

- **High TFLOPS utilization** (>70% of peak) -- kernel is compute-efficient; optimization headroom is small.
- **Low TFLOPS utilization** on a "compute-bound" kernel -- may actually be memory-bound at this problem size (small M/N/K, low arithmetic intensity). Profile with `ncu` to confirm.
- **High TB/sec** (>80% of peak HBM) -- kernel is bandwidth-efficient; optimization headroom is small.
- **Low TB/sec** on a "memory-bound" kernel -- check for excessive synchronization, padding overhead, or poor memory access patterns. Profile with `ncu` to investigate.

### 3. Profile with Nsight tools

Use `--profile` mode for a single kernel iteration wrapped in NVTX range markers, making it easy to isolate in the profiler.

**Nsight Compute (kernel-level metrics):**

```bash
ncu --set full -o gemm_profile \
    python benchmarks/run.py \
    --family gemm --subroutine gemm \
    --M 16000 --N 512 --K 2048 --backends cutlass \
    --profile --warmup-iters 0
```

Open the `.ncu-rep` file in Nsight Compute UI. Key sections to examine:

| Section | What to look for |
|---------|-----------------|
| **GPU Speed Of Light** | Overall compute and memory utilization as % of peak. Tells you which bound dominates. |
| **Compute Workload Analysis** | SM utilization and occupancy. Low occupancy may indicate register pressure or shared memory limits. |
| **Memory Workload Analysis** | L1/L2 hit rates, HBM throughput. Low hit rates suggest poor data reuse or access patterns. |
| **Warp State Statistics** | Stall reasons. `LG Throttle` / `Long Scoreboard` = memory latency. `Math Pipe Throttle` = compute-bound. `Barrier` = synchronization overhead. |
| **Occupancy** | Active warps vs max. Limited by registers, shared memory, or block size. |

**Nsight Systems (timeline and stream view):**

```bash
nsys profile python benchmarks/run.py \
    --family conv --subroutine depthwise_conv1d \
    --batch 64 --seq 250 --channels 256 --kernel-size 15 \
    --profile --warmup-iters 0
```

Use `nsys` to detect host/device synchronization gaps, stream overlap issues, and launch overhead. The NVTX markers appear in the timeline labeled `<backend>_<subroutine>`.

**Profiling tips:**

- Use `--warmup-iters 0` to avoid capturing warmup iterations in the profiler.
- Profile a single backend at a time (`--backends cutlass`) to isolate the kernel of interest.
- Use production-representative problem sizes -- profiling a toy problem may show different bottlenecks.
- For GEMM/Conv2D, autotune first so you profile the best tile configuration.

### 4. Optimize and validate

After making changes, re-run the same benchmark and compare:

```bash
python benchmarks/run.py --family gemm --subroutine gemm \
    --backends cutlass torch --M 16000 --N 512 --K 2048 \
    --output optimized.csv --case-tag optimized \
    --refcheck -vv -v
```

**Always use `--refcheck`** to confirm the optimization does not change numerical output.

### 5. Compare results

```python
import pandas as pd

base = pd.read_csv("baseline.csv")
opt  = pd.read_csv("optimized.csv")

merged = base.merge(opt, on=["routine", "subroutine", "params", "dtype", "backend"],
                    suffixes=("_base", "_opt"))
merged["speedup"] = merged["median_ms_base"] / merged["median_ms_opt"]
# (within a single run, the `speedup_vs_ref` column already gives OASR vs torch)
print(merged[["routine", "subroutine", "shape", "backend", "speedup"]]
      .sort_values("speedup", ascending=False))
```

## Autotuning GEMM Kernels

OASR's autotuner profiles CUTLASS tile configurations and caches the best one per shape/dtype/GPU:

```bash
# Profile and persist the best tile config
python benchmarks/run.py --family gemm --subroutine gemm \
    --backends cutlass torch \
    --M 16000 --N 512 --K 2048 \
    --autotune --cache oasr_tune.json \
    --refcheck -vv

# Use cached config only (no profiling overhead)
python benchmarks/run.py --family gemm --subroutine gemm \
    --backends cutlass torch \
    --M 16000 --N 512 --K 2048 \
    --autotune --cache oasr_tune.json --no-tune \
    --refcheck -vv
```

Autotune results accumulate across runs -- you can tune different shapes in separate runs and the cache merges them. Also accessible via Python:

```python
with oasr.autotune(cache="oasr_tune.json"):
    output = oasr.gemm(A, B)   # profiles on first call, reuses cache after
```

**Autotune before profiling** -- a default tile config may be far from optimal; autotuning first ensures you're profiling the best CUTLASS configuration.

## Reproducing and Sharing Results

The `-v` flag prints a self-contained CLI command for each test case:

```
[REPRO] python benchmarks/run.py --family gemm --subroutine bmm \
    --backends cutlass torch --dtype float16 --iters 30 --warmup-iters 5 \
    --refcheck --batch-count 256 --M 200 --N 200 --K 64
```

Copy this command to share exact configurations when reporting performance results or filing issues. For CSV output, reproducer commands are also stored in the `repro_command` column.

## Key CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--family` | Benchmark family; `--list` shows all of them (`--routine` still accepted) | required |
| `--subroutine` | Specific kernel (e.g. bmm, layer_norm, depthwise_conv1d) | first available |
| `--backends` | Space-separated backend names (see table above) | auto per family |
| `--dtype` | float16, bfloat16, float32 | float16 |
| `--iters` | Measurement iterations | 30 |
| `--warmup-iters` | Warmup iterations | 5 |
| `--refcheck` | Verify outputs agree between backends | False |
| `--allow-output-mismatch` | Continue benchmarking despite refcheck failure | False |
| `--timer cuda_events` | Force CUDA events timing, skip Triton do_bench | False |
| `--autotune` | Enable CUTLASS tile autotuning (gemm subroutine) | False |
| `--cache` | Path to autotune cache JSON | None |
| `--no-tune` | Load cache only, skip profiling | False |
| `--profile` | NVTX-wrapped single-iteration mode (for ncu/nsys) | False |
| `--output` | CSV file for results | None |
| `--testlist` | File with one test invocation per line | None |
| `-v` | Print reproducer command | False |
| `--case-tag` | Tag appended to CSV rows (use for A/B comparisons) | None |
| `-v` / `-vv` | Verbose / very verbose output | 0 |
| `--list` | List all available families and subroutines | False |
| `--print-schema` | Print both output schemas | False |
| `--min-measure-ms` | Wall-time floor per measurement; raises the iteration count for fast kernels | 20 |
| `--timer` | `cuda_events` (counts iterations) or `triton` (budgets milliseconds) | cuda_events |
| `--ref-backend` | Baseline for `--refcheck` and `speedup_vs_ref` | torch |

## Best Practices

1. **Always use `--refcheck`** -- catch correctness regressions before interpreting perf numbers. An incorrect kernel is irrelevant regardless of speed.
2. **Match production shapes** -- peak TFLOPS on large square matrices means nothing if your workload uses thin, tall shapes (M=16000, N=256).
3. **Run on an idle GPU** -- other processes inflate latency and variance. If `std` is > 5% of `median`, something is competing for the GPU.
4. **Use `-v`** -- share exact reproducer commands when reporting results or filing issues.
5. **Tag A/B comparisons** -- use `--case-tag baseline` and `--case-tag opt` with `--output` so results are easy to compare in CSV.
6. **Increase iterations for noisy results** -- `--iters 100 --warmup-iters 20` for more stable measurements on fast kernels (< 0.1 ms).
7. **Autotune before profiling** -- ensures you're profiling the best tile configuration, not a suboptimal default.
8. **Lock GPU clocks for reproducibility** -- `sudo nvidia-smi -lgc <base_clock>` eliminates frequency scaling noise in sensitive measurements.

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| JIT compilation errors | Stale or corrupted JIT cache | `rm -rf ~/.cache/oasr/jit/` and retry |
| `Triton not found` warning | Triton not installed; CUDA events used instead | `pip install triton` for more accurate timing, or ignore -- CUDA events work fine |
| Refcheck failure | FP16 reduction order differences between cutlass/cuda and torch | `--allow-output-mismatch` to continue; investigate with `-vv` if difference is unexpectedly large |
| High `std` / noisy results | GPU frequency scaling, thermal throttling, or other processes | Increase warmup (`--warmup-iters 20`), increase iterations (`--iters 100`), lock GPU clocks |
| CUDA OOM | Problem size too large for GPU memory | Reduce `--batch`, `--seq`, or matrix dimensions |
| Wrong backend name | Backend names differ by kernel family | Check the backend table above: `cutlass`/`torch` for GEMM/Conv2D, `cuda`/`torch` for Norm/Conv1D/Activation |

## Quick Reference: Common Benchmark Commands

```bash
# Full Conformer-base workload sweep
python benchmarks/run.py \
    --testlist benchmarks/testlists/conformer_base.txt \
    --output results.csv --refcheck -vv -v

# All kernel families, one config each
python benchmarks/run.py \
    --testlist benchmarks/testlists/all_kernels.txt \
    --output all_results.csv --refcheck

# Profile a single GEMM kernel for Nsight Compute
ncu --set full -o gemm_profile \
    python benchmarks/run.py \
    --family gemm --subroutine gemm \
    --M 16000 --N 512 --K 2048 --backends cutlass \
    --profile --warmup-iters 0

# Profile a LayerNorm kernel for Nsight Compute
ncu --set full -o layernorm_profile \
    python benchmarks/run.py \
    --family norm --subroutine layer_norm \
    --batch 64 --seq 250 --hidden 512 --backends cuda \
    --profile --warmup-iters 0

# Autotune GEMM then benchmark with best config
python benchmarks/run.py --family gemm --subroutine gemm \
    --backends cutlass torch --M 16000 --N 512 --K 2048 \
    --autotune --cache oasr_tune.json --refcheck -vv
```
