#!/usr/bin/env python3
"""
Performance benchmarks for normalization kernels.
Uses triton.testing.do_bench for accurate GPU timing.

Problem sizes align with WeNet (10 sec audio, batch up to 64):
  train_conformer.yaml, train_conformer_bidecoder_large.yaml

Profiling mode for NVIDIA Nsight Compute:
    ncu --set full -o profile_report python bench_norm_kernels.py \
    --profile --kernel layer_norm --target oasr \
    --warmup 0 --iters 1
"""

import sys
sys.path.insert(0, 'python')

import argparse
import torch

import oasr


# =============================================================================
# Profiling utilities
# =============================================================================

def profile_kernel(name: str, fn, warmup: int = 3, profile_iters: int = 1):
    """
    Run kernel for profiling with Nsight Compute.
    
    Args:
        name: Kernel name for NVTX range
        fn: Kernel function to profile
        warmup: Number of warmup iterations
        profile_iters: Number of iterations to profile
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    # Profile iterations with NVTX markers
    torch.cuda.nvtx.range_push(name)
    for _ in range(profile_iters):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


# =============================================================================
# Kernel setup functions (shared between benchmark and profile modes)
# =============================================================================

def setup_layer_norm(batch_size, seq_len, hidden_size, dtype=torch.float16):
    """Setup tensors for LayerNorm."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
    gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
    beta = torch.randn(hidden_size, device='cuda', dtype=dtype)
    output = torch.empty_like(x)
    
    dtype_map = {torch.float32: oasr.DataType.FP32, torch.float16: oasr.DataType.FP16}
    
    def oasr_fn():
        oasr.kernels.normalization.layer_norm(
            x.data_ptr(), output.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            batch_size, seq_len, hidden_size,
            eps, dtype_map[dtype]
        )
    
    layer_norm = torch.nn.LayerNorm(hidden_size, eps=eps, device='cuda', dtype=dtype)
    layer_norm.weight.data = gamma.clone()
    layer_norm.bias.data = beta.clone()
    
    def pytorch_fn():
        return layer_norm(x)
    
    return oasr_fn, pytorch_fn


def setup_rms_norm(batch_size, seq_len, hidden_size, dtype=torch.float16):
    """Setup tensors for RMSNorm."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
    gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
    output = torch.empty_like(x)
    
    dtype_map = {torch.float32: oasr.DataType.FP32, torch.float16: oasr.DataType.FP16}
    
    def oasr_fn():
        oasr.kernels.normalization.rms_norm(
            x.data_ptr(), output.data_ptr(),
            gamma.data_ptr(),
            batch_size, seq_len, hidden_size,
            eps, dtype_map[dtype]
        )
    
    def pytorch_fn():
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x / rms * gamma
    
    return oasr_fn, pytorch_fn


def setup_add_layer_norm(batch_size, seq_len, hidden_size, dtype=torch.float16):
    """Setup tensors for fused Add + LayerNorm."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
    residual = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
    gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
    beta = torch.randn(hidden_size, device='cuda', dtype=dtype)
    output = torch.empty_like(x)
    
    dtype_map = {torch.float32: oasr.DataType.FP32, torch.float16: oasr.DataType.FP16}
    
    def oasr_fn():
        oasr.kernels.normalization.add_layer_norm(
            x.data_ptr(), residual.data_ptr(), output.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            batch_size, seq_len, hidden_size,
            eps, dtype_map[dtype]
        )
    
    layer_norm = torch.nn.LayerNorm(hidden_size, eps=eps, device='cuda', dtype=dtype)
    layer_norm.weight.data = gamma.clone()
    layer_norm.bias.data = beta.clone()
    
    def pytorch_fn():
        return layer_norm(x + residual)
    
    return oasr_fn, pytorch_fn


def setup_group_norm(batch_size, seq_len, channels, num_groups, dtype=torch.float32):
    """Setup tensors for GroupNorm."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
    gamma = torch.randn(channels, device='cuda', dtype=dtype)
    beta = torch.randn(channels, device='cuda', dtype=dtype)
    output = torch.empty_like(x)
    
    dtype_map = {torch.float32: oasr.DataType.FP32, torch.float16: oasr.DataType.FP16}
    channels_per_group = channels // num_groups
    
    def oasr_fn():
        oasr.kernels.normalization.group_norm(
            x.data_ptr(), output.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            batch_size, seq_len, channels, num_groups,
            eps, dtype_map[dtype]
        )
    
    def pytorch_fn():
        x_r = x.view(batch_size, seq_len, num_groups, channels_per_group)
        mean = x_r.mean(dim=-1, keepdim=True)
        var = x_r.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_r - mean) / torch.sqrt(var + eps)
        x_norm = x_norm.view(batch_size, seq_len, channels)
        return x_norm * gamma + beta
    
    return oasr_fn, pytorch_fn


def setup_batch_norm(batch_size, seq_len, channels, dtype=torch.float32):
    """Setup tensors for BatchNorm1D."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
    gamma = torch.randn(channels, device='cuda', dtype=dtype)
    beta = torch.randn(channels, device='cuda', dtype=dtype)
    running_mean = torch.randn(channels, device='cuda', dtype=dtype)
    running_var = torch.abs(torch.randn(channels, device='cuda', dtype=dtype)) + 0.1
    output = torch.empty_like(x)
    
    dtype_map = {torch.float32: oasr.DataType.FP32, torch.float16: oasr.DataType.FP16}
    
    def oasr_fn():
        oasr.kernels.normalization.batch_norm_1d(
            x.data_ptr(), output.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            running_mean.data_ptr(), running_var.data_ptr(),
            batch_size, seq_len, channels,
            eps, dtype_map[dtype]
        )
    
    def pytorch_fn():
        return (x - running_mean) / torch.sqrt(running_var + eps) * gamma + beta
    
    return oasr_fn, pytorch_fn


# =============================================================================
# Benchmark functions
# =============================================================================


def benchmark_layer_norm():
    """Benchmark LayerNorm: OASR vs PyTorch. Sizes from WeNet Conformer-base/large (B,T,d_model)."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 256),
        (64, 500, 512),
        (32, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("LayerNorm Benchmark (Conformer encoder workload)")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, hidden_size in configs:
        oasr_fn, pytorch_fn = setup_layer_norm(batch_size, seq_len, hidden_size)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {hidden_size}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_rms_norm():
    """Benchmark RMSNorm: OASR vs manual PyTorch. Conformer (B, T_enc, d_model)."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("RMSNorm Benchmark (Conformer workload)")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, hidden_size in configs:
        oasr_fn, pytorch_fn = setup_rms_norm(batch_size, seq_len, hidden_size)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {hidden_size}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_add_layer_norm():
    """Benchmark fused Add + LayerNorm. Conformer (B, T_enc, d_model)."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("Add + LayerNorm (Fused) Benchmark (Conformer workload)")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, hidden_size in configs:
        oasr_fn, pytorch_fn = setup_add_layer_norm(batch_size, seq_len, hidden_size)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {hidden_size}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_group_norm():
    """Benchmark GroupNorm. Conformer (B, T_enc, channels, num_groups)."""
    import triton

    configs = [
        (32, 250, 256, 32),
        (64, 250, 256, 32),
        (64, 250, 512, 64),
        (64, 500, 512, 64),
    ]

    print("\n" + "=" * 70)
    print("GroupNorm Benchmark (Conformer workload)")
    print("=" * 70)
    print(f"\n{'Shape':<35} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 75)
    
    for batch_size, seq_len, channels, num_groups in configs:
        oasr_fn, pytorch_fn = setup_group_norm(batch_size, seq_len, channels, num_groups)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}] g={num_groups}"
        print(f"{shape_str:<35} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_batch_norm():
    """Benchmark BatchNorm1D (inference). Conformer (B, T_enc, channels)."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("BatchNorm1D (Inference) Benchmark (Conformer workload)")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, channels in configs:
        oasr_fn, pytorch_fn = setup_batch_norm(batch_size, seq_len, channels)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


# =============================================================================
# Profiling functions
# =============================================================================

# Default profiling configs (B=64, T=250)
PROFILE_CONFIGS = {
    'layer_norm': (64, 250, 256),
    'rms_norm': (64, 250, 256),
    'add_layer_norm': (64, 250, 512),
    'group_norm': (64, 250, 512, 64),
    'batch_norm': (64, 250, 512),
}


def profile_kernels(kernels: list, target: str = 'both', warmup: int = 3, iters: int = 1):
    """
    Profile specified kernels for Nsight Compute.
    
    Args:
        kernels: List of kernel names to profile
        target: Which implementation to profile ('oasr', 'pytorch', or 'both')
        warmup: Warmup iterations
        iters: Profile iterations
    """
    print("=" * 70)
    print("OASR Normalization Kernels - Profiling Mode")
    print("=" * 70)
    print(f"Target: {target}, Warmup: {warmup}, Profile iterations: {iters}")
    print("=" * 70)
    
    setup_funcs = {
        'layer_norm': lambda: setup_layer_norm(*PROFILE_CONFIGS['layer_norm']),
        'rms_norm': lambda: setup_rms_norm(*PROFILE_CONFIGS['rms_norm']),
        'add_layer_norm': lambda: setup_add_layer_norm(*PROFILE_CONFIGS['add_layer_norm']),
        'group_norm': lambda: setup_group_norm(*PROFILE_CONFIGS['group_norm']),
        'batch_norm': lambda: setup_batch_norm(*PROFILE_CONFIGS['batch_norm']),
    }
    
    for kernel_name in kernels:
        if kernel_name not in setup_funcs:
            print(f"Unknown kernel: {kernel_name}")
            continue
        
        config = PROFILE_CONFIGS[kernel_name]
        print(f"\nProfiling: {kernel_name}")
        print(f"  Config: {config}")
        
        oasr_fn, pytorch_fn = setup_funcs[kernel_name]()
        
        # Profile OASR kernel
        if target in ('oasr', 'both'):
            print(f"  Running OASR kernel...")
            profile_kernel(f"oasr_{kernel_name}", oasr_fn, warmup=warmup, profile_iters=iters)
        
        # Profile PyTorch reference
        if target in ('pytorch', 'both'):
            print(f"  Running PyTorch kernel...")
            profile_kernel(f"pytorch_{kernel_name}", pytorch_fn, warmup=warmup, profile_iters=iters)
        
        print(f"  Done.")
    
    print("\n" + "=" * 70)
    print("Profiling complete!")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OASR Normalization Kernels - Benchmark & Profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python bench_norm_kernels.py

  # Profile specific kernel with Nsight Compute (OASR only)
  ncu --set full -o layernorm_profile python bench_norm_kernels.py --profile --kernel layer_norm --target oasr

  # Profile PyTorch implementation only
  python bench_norm_kernels.py --profile --kernel layer_norm --target pytorch

  # Profile both implementations (default)
  python bench_norm_kernels.py --profile --kernel layer_norm --target both

  # Profile multiple kernels
  python bench_norm_kernels.py --profile --kernel layer_norm rms_norm

  # Profile all kernels
  python bench_norm_kernels.py --profile --kernel all

Available kernels:
  layer_norm, rms_norm, add_layer_norm, group_norm, batch_norm
        """
    )
    
    parser.add_argument(
        '--profile', action='store_true',
        help='Run in profiling mode (single iterations for ncu)'
    )
    parser.add_argument(
        '--kernel', nargs='+', default=['all'],
        help='Kernel(s) to profile (use "all" for all kernels)'
    )
    parser.add_argument(
        '--target', choices=['oasr', 'pytorch', 'both'], default='oasr',
        help='Which implementation to profile: oasr, pytorch, or both (default: oasr)'
    )
    parser.add_argument(
        '--warmup', type=int, default=3,
        help='Number of warmup iterations for profiling (default: 3)'
    )
    parser.add_argument(
        '--iters', type=int, default=1,
        help='Number of profile iterations (default: 1)'
    )
    
    args = parser.parse_args()
    
    if args.profile:
        # Profiling mode
        if 'all' in args.kernel:
            kernels = list(PROFILE_CONFIGS.keys())
        else:
            kernels = args.kernel
        
        profile_kernels(kernels, target=args.target, warmup=args.warmup, iters=args.iters)
    else:
        # Benchmark mode
        print("=" * 70)
        print("OASR Normalization Kernels - Performance Benchmarks")
        print("=" * 70)
        
        benchmark_layer_norm()
        benchmark_rms_norm()
        benchmark_add_layer_norm()
        benchmark_group_norm()
        benchmark_batch_norm()
        
        print("\n" + "=" * 70)
        print("Benchmarks complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
