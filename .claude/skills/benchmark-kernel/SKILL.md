---
name: benchmark-kernel
description: Guide for benchmarking OASR kernels with the unified benchmark CLI
---

# Tutorial: Benchmarking OASR Kernels

This tutorial shows you how to benchmark OASR CUDA/CUTLASS kernels against PyTorch baselines.

## Goal

Measure the performance of OASR kernels:
- Get accurate GPU kernel execution time
- Compare OASR vs PyTorch backends
- Generate reproducible benchmark results
- Save results to CSV for analysis

## Timing Methods

OASR supports two timing methods:

1. **Triton do_bench (Preferred)**: Uses Triton's built-in benchmarking utility
   - Accurate median timing with automatic warmup
   - Requires `triton` to be installed

2. **CUDA Events (Fallback)**: Standard CUDA event timing
   - Automatically used if Triton is not available
   - Good accuracy, slight overhead from host synchronization

**The framework automatically uses Triton do_bench if available, otherwise falls back to CUDA events.**

## Method 1: Using oasr_benchmark.py (Recommended)

### Step 1: Choose Your Routine and Subroutine

Available routines and subroutines:

- **gemm**: `gemm`, `bmm`, `group_gemm`, `gemm_activation`
- **norm**: `layer_norm`, `add_layer_norm`, `layer_norm_activation`, `rms_norm`, `batch_norm`, `batch_norm_swish`, `batch_norm_activation`, `group_norm`
- **conv**: `depthwise_conv1d`, `depthwise_conv1d_causal`, `pointwise_conv1d`, `pointwise_conv1d_activation`, `conv2d`, `conv2d_activation`
- **activation**: `glu`, `swish`
- **composite**: `conv_block`

List all available routines:

```bash
python benchmarks/oasr_benchmark.py --list
```

### Step 2: Run a Single Benchmark

Example - Benchmark batched matrix multiply:

```bash
python benchmarks/oasr_benchmark.py \
    --routine gemm \
    --subroutine bmm \
    --backends oasr pytorch \
    --batch-count 256 \
    --M 200 --N 200 --K 64 \
    --dtype float16 \
    --num_iters 30 \
    --dry_run_iters 5 \
    --refcheck \
    -vv
```

Example - Benchmark LayerNorm:

```bash
python benchmarks/oasr_benchmark.py \
    --routine norm \
    --subroutine layer_norm \
    --backends oasr pytorch \
    --batch 64 --seq 250 --hidden 512 \
    --refcheck -vv
```

Example - Benchmark depthwise conv1d:

```bash
python benchmarks/oasr_benchmark.py \
    --routine conv \
    --subroutine depthwise_conv1d \
    --backends oasr pytorch \
    --batch 64 --seq 250 --channels 256 --kernel-size 15 \
    --refcheck -vv
```

**Timing behavior:**
- If Triton installed: Uses Triton do_bench (preferred)
- If Triton not installed: Automatically falls back to CUDA events
- To force CUDA events: Add `--use_cuda_events` flag

### Step 3: Understand the Output

```
[INFO] gemm/bmm
[VVERBOSE] gpu_name = 'NVIDIA_A100_80GB_PCIe'
[PERF] oasr         :: median time 0.145 ms; std 0.002 ms; achieved tflops 125.3 TFLOPs/sec
[PERF] pytorch      :: median time 0.168 ms; std 0.003 ms; achieved tflops 108.1 TFLOPs/sec
```

**Key metrics:**
- **median time**: Median kernel execution time (lower is better)
- **std**: Standard deviation (lower means more consistent)
- **achieved tflops**: Effective TFLOPS throughput (compute-bound kernels)
- **achieved tb_per_sec**: Memory bandwidth utilization (memory-bound kernels)

### Step 4: Run Batch Benchmarks

Create a test list file (e.g., `benchmarks/testlists/conformer_base.txt`):

```bash
--routine norm --subroutine layer_norm --batch 64 --seq 250 --hidden 256
--routine conv --subroutine depthwise_conv1d --batch 64 --seq 250 --channels 256 --kernel-size 15
--routine conv --subroutine pointwise_conv1d --batch 64 --seq 250 --channels 256 --out-channels 512
--routine activation --subroutine glu --batch 64 --seq 250 --channels 512
--routine gemm --subroutine gemm --M 16000 --N 256 --K 256
```

Run all tests:

```bash
python benchmarks/oasr_benchmark.py \
    --testlist benchmarks/testlists/conformer_base.txt \
    --output_path results.csv \
    --generate_repro_command \
    --refcheck
```

Results are saved to `results.csv` with all metrics and reproducer commands.

### Step 5: Common Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--routine` | Kernel family (gemm, norm, conv, activation, composite) | required |
| `--subroutine` | Specific kernel within routine | first available |
| `--backends` | Backend(s) to benchmark | oasr pytorch |
| `--dtype` | Data type (float16, bfloat16, float32) | float16 |
| `--num_iters` | Measurement iterations | 30 |
| `--dry_run_iters` | Warmup iterations | 5 |
| `--refcheck` | Verify output correctness | False |
| `--allow_output_mismatch` | Continue on mismatch | False |
| `--use_cuda_events` | Force CUDA events (skip Triton) | False |
| `--output_path` | Path for CSV output | None |
| `--testlist` | Path to testlist file | None |
| `--profile` | Run in profiling mode (Nsight Compute / NVTX) | False |
| `-v` / `-vv` | Verbose / very verbose output | - |
| `--generate_repro_command` | Print reproducer command | False |
| `--case_tag` | Tag for CSV output | None |
| `--list` | List all available routines | False |

## Method 2: Using bench_fn() in Python

For custom benchmarking in your own code:

```python
import torch
from benchmarks.routines.bench_utils import bench_fn, compute_gemm_tflops

# Setup your kernel
x = torch.randn(64, 250, 256, device="cuda", dtype=torch.float16)
weight = torch.randn(256, 256, device="cuda", dtype=torch.float16)

def my_kernel():
    return torch.mm(x.view(-1, 256), weight.t())

# Benchmark - Triton do_bench preferred, CUDA events fallback
median_ms, std_ms = bench_fn(
    my_kernel,
    dry_run_iters=5,
    num_iters=30,
    use_cuda_events=False,  # Set True to force CUDA events
)

print(f"Kernel time: {median_ms:.3f} ms +/- {std_ms:.3f} ms")

# Calculate TFLOPS
M, N, K = 64 * 250, 256, 256
tflops = compute_gemm_tflops(M, N, K, median_ms)
print(f"Achieved: {tflops:.2f} TFLOPS/sec")
```

## Troubleshooting

### Inconsistent Results

**Problem**: Large standard deviation or varying results

**Solutions**:
1. Increase warmup: `--dry_run_iters 10`
2. Increase measurement: `--num_iters 50`
3. Disable GPU boost: `sudo nvidia-smi -lgc <base_clock>`

### Reference Check Failures

**Error**: `[ERROR] Output mismatch for [shape] (max_diff=X.XXXXXX)`

**Solutions**:
1. Allow mismatch and continue: `--allow_output_mismatch`
2. Check numerical tolerance (FP16 vs FP32 precision differences)
3. Use very verbose mode for details: `-vv`

## Quick Examples

### GEMM with autotuning
```bash
python benchmarks/oasr_benchmark.py \
    --routine gemm --subroutine gemm \
    --backends oasr pytorch \
    --M 8000 --N 256 --K 256 \
    --autotune --refcheck -vv
```

### All norm variants
```bash
python benchmarks/oasr_benchmark.py \
    --routine norm --subroutine rms_norm \
    --backends oasr pytorch \
    --batch 64 --seq 250 --hidden 512 \
    --refcheck -vv
```

### Conv2D (NHWC CUTLASS)
```bash
python benchmarks/oasr_benchmark.py \
    --routine conv --subroutine conv2d \
    --backends oasr pytorch \
    --batch 16 --height 200 --width 80 \
    --channels 1 --out-filters 64 \
    --filter-h 3 --filter-w 3 --pad 0 --stride 2 \
    --refcheck -vv
```

### End-to-end Conformer conv block
```bash
python benchmarks/oasr_benchmark.py \
    --routine composite --subroutine conv_block \
    --backends oasr pytorch \
    --batch 64 --seq 250 --d-model 256 --kernel-size 15 \
    --refcheck -vv
```

### Profiling mode (for Nsight Compute)
```bash
ncu --set full -o gemm_profile python benchmarks/oasr_benchmark.py \
    --routine gemm --subroutine gemm \
    --profile --dry_run_iters 0
```

## Best Practices

1. **Use reference checking** to verify correctness: `--refcheck`
2. **Use verbose mode** to see input shapes and dtypes: `-vv`
3. **Generate reproducer commands** for sharing results: `--generate_repro_command`
4. **Run multiple iterations** for statistical significance: `--num_iters 30 --dry_run_iters 5`
5. **Save results to CSV** for later analysis: `--output_path results.csv`
6. **Compare backends** to measure OASR speedup: `--backends oasr pytorch`

## Legacy Standalone Scripts

Individual benchmark scripts are still available as thin wrappers:

```bash
python benchmarks/bench_gemm.py        # gemm routines
python benchmarks/bench_layer_norm.py   # norm routines
python benchmarks/bench_depthwise_conv1d.py  # conv routines
python benchmarks/bench_glu.py          # activation routines
python benchmarks/bench_conv_block.py   # composite routines
```

These delegate to the same routine modules used by `oasr_benchmark.py`.

## Related Documentation

- See `benchmarks/testlists/` for example testlist files
- See CLAUDE.md "Benchmarking" section for profiling details
