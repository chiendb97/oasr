#!/usr/bin/env python3
"""
Performance benchmarks for convolution kernels.
Uses triton.testing.do_bench for accurate GPU timing.

Profiling mode for NVIDIA Nsight Compute:
    ncu --set full -o profile_report python bench_conv_kernels.py \
    --profile --kernel swish --target oasr \
    --warmup 0 --iters 1
"""

import sys
sys.path.insert(0, 'python')

import argparse
import torch
import torch.nn.functional as F

import oasr

try:
    from oasr import ConvType, ActivationType
except ImportError:
    from oasr._C import ConvType, ActivationType
    oasr.ConvType = ConvType
    oasr.ActivationType = ActivationType


# =============================================================================
# Profiling utilities
# =============================================================================

def profile_kernel(name: str, fn, warmup: int = 3, profile_iters: int = 1):
    """Run kernel for profiling with Nsight Compute."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    torch.cuda.nvtx.range_push(name)
    for _ in range(profile_iters):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


# =============================================================================
# Kernel setup functions (shared between benchmark and profile modes)
# =============================================================================

def setup_depthwise_conv1d(batch_size, seq_len, channels, kernel_size, dtype=torch.float16):
    """Setup tensors for depthwise conv1d."""
    padding = kernel_size // 2
    x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
    weight = torch.randn(channels, kernel_size, device='cuda', dtype=dtype)
    bias = torch.randn(channels, device='cuda', dtype=dtype)
    
    dtype_map = {torch.float32: oasr.DataType.FP32, torch.float16: oasr.DataType.FP16}
    
    def oasr_fn():
        return oasr.kernels.conv.depthwise_conv1d(
            x, weight, bias, padding,
            dtype_map[dtype]
        )
    
    x_nchw = x.permute(0, 2, 1).contiguous()
    weight_pt = weight.view(channels, 1, kernel_size)
    
    def pytorch_fn():
        return F.conv1d(x_nchw, weight_pt, bias, padding=padding, groups=channels)
    
    return oasr_fn, pytorch_fn


def setup_depthwise_conv1d_causal(batch_size, seq_len, channels, kernel_size, dtype=torch.float16):
    """Setup tensors for causal depthwise conv1d."""
    x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
    weight = torch.randn(channels, kernel_size, device='cuda', dtype=dtype)
    bias = torch.randn(channels, device='cuda', dtype=dtype)
    
    dtype_map = {torch.float32: oasr.DataType.FP32, torch.float16: oasr.DataType.FP16}
    
    def oasr_fn():
        return oasr.kernels.conv.depthwise_conv1d(
            x, weight, bias, 0,
            dtype_map[dtype]
        )
    
    x_nchw = x.permute(0, 2, 1).contiguous()
    weight_pt = weight.view(channels, 1, kernel_size)
    
    def pytorch_fn():
        x_padded = F.pad(x_nchw, (kernel_size - 1, 0))
        return F.conv1d(x_padded, weight_pt, bias, groups=channels)
    
    return oasr_fn, pytorch_fn


def setup_pointwise_conv1d(batch_size, seq_len, in_ch, out_ch, dtype=torch.float16):
    """Setup tensors for pointwise conv1d."""
    x = torch.randn(batch_size, seq_len, in_ch, device='cuda', dtype=dtype)
    weight = torch.randn(out_ch, in_ch, device='cuda', dtype=dtype)
    bias = torch.randn(out_ch, device='cuda', dtype=dtype)
    
    dtype_map = {torch.float32: oasr.DataType.FP32, torch.float16: oasr.DataType.FP16}
    
    def oasr_fn():
        return oasr.kernels.conv.pointwise_conv1d(
            x, weight, bias,
            oasr.ActivationType.SWISH, False, dtype_map[dtype]
        )
    
    def pytorch_fn():
        return F.linear(x, weight, bias)
    
    return oasr_fn, pytorch_fn


def setup_glu(batch_size, seq_len, channels, dtype=torch.float16):
    """Setup tensors for GLU."""
    x = torch.randn(batch_size, seq_len, 2 * channels, device='cuda', dtype=dtype)
    
    dtype_map = {torch.float32: oasr.DataType.FP32, torch.float16: oasr.DataType.FP16}
    
    def oasr_fn():
        return oasr.kernels.conv.glu(
            x,
            dtype_map[dtype]
        )
    
    def pytorch_fn():
        return F.glu(x, dim=-1)
    
    return oasr_fn, pytorch_fn


def setup_swish(batch_size, seq_len, channels, dtype=torch.float16):
    """Setup tensors for Swish."""
    x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
    
    dtype_map = {torch.float32: oasr.DataType.FP32, torch.float16: oasr.DataType.FP16}
    
    def oasr_fn():
        return oasr.kernels.conv.swish(
            x,
            dtype_map[dtype]
        )
    
    def pytorch_fn():
        return F.silu(x)
    
    return oasr_fn, pytorch_fn


def setup_batch_norm_swish(batch_size, seq_len, channels, dtype=torch.float16):
    """Setup tensors for fused BatchNorm + Swish."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
    gamma = torch.randn(channels, device='cuda', dtype=dtype)
    beta = torch.randn(channels, device='cuda', dtype=dtype)
    running_mean = torch.randn(channels, device='cuda', dtype=dtype)
    running_var = torch.abs(torch.randn(channels, device='cuda', dtype=dtype)) + 0.1
    
    dtype_map = {torch.float32: oasr.DataType.FP32, torch.float16: oasr.DataType.FP16}
    
    def oasr_fn():
        return oasr.kernels.conv.batch_norm_swish(
            x,
            gamma, beta,
            running_mean, running_var,
            eps, dtype_map[dtype]
        )
    
    def pytorch_fn():
        bn_out = (x - running_mean) / torch.sqrt(running_var + eps) * gamma + beta
        return F.silu(bn_out)
    
    return oasr_fn, pytorch_fn


def setup_conv_block(batch_size, seq_len, d_model, kernel_size, dtype=torch.float16):
    """Setup tensors for conv block (depthwise + pointwise + GLU + swish)."""
    x = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=dtype)
    
    pw1_weight = torch.randn(2 * d_model, d_model, device='cuda', dtype=dtype)
    pw1_bias = torch.randn(2 * d_model, device='cuda', dtype=dtype)
    dw_weight = torch.randn(d_model, kernel_size, device='cuda', dtype=dtype)
    dw_bias = torch.randn(d_model, device='cuda', dtype=dtype)
    pw2_weight = torch.randn(d_model, d_model, device='cuda', dtype=dtype)
    pw2_bias = torch.randn(d_model, device='cuda', dtype=dtype)
    
    dw_weight_pt = dw_weight.view(d_model, 1, kernel_size)
    dtype_map = {torch.float32: oasr.DataType.FP32, torch.float16: oasr.DataType.FP16}
    
    def oasr_fn():
        pw1_out = oasr.kernels.conv.pointwise_conv1d(
            x, pw1_weight, pw1_bias,
            oasr.ActivationType.SWISH, False, dtype_map[dtype]
        )
        glu_out = oasr.kernels.conv.glu(
            pw1_out,
            dtype_map[dtype]
        )
        dw_out = oasr.kernels.conv.depthwise_conv1d(
            glu_out, dw_weight, dw_bias, kernel_size // 2,
            dtype_map[dtype]
        )
        swish_out = oasr.kernels.conv.swish(
            dw_out,
            dtype_map[dtype]
        )
        return oasr.kernels.conv.pointwise_conv1d(
            swish_out, pw2_weight, pw2_bias,
            oasr.ActivationType.SWISH, False, dtype_map[dtype]
        )
    
    def pytorch_fn():
        pw1 = F.linear(x, pw1_weight, pw1_bias)
        glu = F.glu(pw1, dim=-1)
        glu_nchw = glu.permute(0, 2, 1)
        dw_nchw = F.conv1d(glu_nchw, dw_weight_pt, dw_bias, padding=kernel_size // 2, groups=d_model)
        dw = dw_nchw.permute(0, 2, 1)
        swish = F.silu(dw)
        return F.linear(swish, pw2_weight, pw2_bias)
    
    return oasr_fn, pytorch_fn


# =============================================================================
# Benchmark functions
# =============================================================================


def benchmark_depthwise_conv1d():
    """Benchmark DepthwiseConv1D: OASR vs PyTorch."""
    import triton

    configs = [
        (32, 250, 256, 15),
        (64, 250, 256, 15),
        (64, 250, 512, 31),
        (64, 500, 256, 15),
        (64, 500, 512, 31),
        (32, 125, 256, 15),
        (64, 125, 512, 31),
    ]

    print("\n" + "=" * 70)
    print("DepthwiseConv1D Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<35} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 75)
    
    for batch_size, seq_len, channels, kernel_size in configs:
        oasr_fn, pytorch_fn = setup_depthwise_conv1d(batch_size, seq_len, channels, kernel_size)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}] k={kernel_size}"
        print(f"{shape_str:<35} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_depthwise_conv1d_causal():
    """Benchmark causal DepthwiseConv1D."""
    import triton

    configs = [
        (32, 250, 256, 15),
        (64, 250, 256, 15),
        (64, 250, 512, 31),
        (64, 500, 512, 31),
    ]

    print("\n" + "=" * 70)
    print("DepthwiseConv1D (Causal) Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<35} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 75)
    
    for batch_size, seq_len, channels, kernel_size in configs:
        oasr_fn, pytorch_fn = setup_depthwise_conv1d_causal(batch_size, seq_len, channels, kernel_size)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}] k={kernel_size}"
        print(f"{shape_str:<35} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_pointwise_conv1d():
    """Benchmark PointwiseConv1D."""
    import triton

    configs = [
        (32, 250, 256, 512),
        (64, 250, 256, 512),
        (64, 250, 512, 1024),
        (64, 250, 256, 2048),
        (64, 250, 512, 2048),
        (64, 500, 256, 512),
        (64, 500, 512, 1024),
    ]

    print("\n" + "=" * 70)
    print("PointwiseConv1D Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<40} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 80)
    
    for batch_size, seq_len, in_ch, out_ch in configs:
        oasr_fn, pytorch_fn = setup_pointwise_conv1d(batch_size, seq_len, in_ch, out_ch)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {in_ch}] -> {out_ch}"
        print(f"{shape_str:<40} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_glu():
    """Benchmark GLU activation."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 256),
        (64, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("GLU Activation Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, channels in configs:
        oasr_fn, pytorch_fn = setup_glu(batch_size, seq_len, channels)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {2*channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_swish():
    """Benchmark Swish (SiLU) activation."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("Swish Activation Benchmark (Conformer workload)")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, channels in configs:
        oasr_fn, pytorch_fn = setup_swish(batch_size, seq_len, channels)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_batch_norm_swish():
    """Benchmark fused BatchNorm + Swish."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("BatchNorm + Swish (Fused) Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, channels in configs:
        oasr_fn, pytorch_fn = setup_batch_norm_swish(batch_size, seq_len, channels)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_conv_block():
    """Benchmark full conv block (depthwise + pointwise + GLU + swish)."""
    import triton

    configs = [
        (32, 250, 256, 15),
        (64, 250, 256, 15),
        (64, 250, 512, 31),
        (64, 500, 256, 15),
        (64, 500, 512, 31),
    ]

    print("\n" + "=" * 70)
    print("Conv Block (End-to-End) Benchmark")
    print("=" * 70)
    print(f"\n{'Config':<35} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 75)
    
    for batch_size, seq_len, d_model, kernel_size in configs:
        oasr_fn, pytorch_fn = setup_conv_block(batch_size, seq_len, d_model, kernel_size)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        config_str = f"[{batch_size}, {seq_len}, {d_model}] k={kernel_size}"
        print(f"{config_str:<35} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


# =============================================================================
# Profiling functions
# =============================================================================

PROFILE_CONFIGS = {
    'depthwise_conv1d': (64, 250, 512, 31),
    'depthwise_conv1d_causal': (64, 250, 512, 31),
    'pointwise_conv1d': (64, 250, 512, 1024),
    'glu': (64, 250, 512),
    'swish': (64, 250, 512),
    'batch_norm_swish': (64, 250, 512),
    'conv_block': (64, 250, 512, 31),
}


def profile_kernels(kernels: list, target: str = 'both', warmup: int = 3, iters: int = 1):
    """Profile specified kernels for Nsight Compute."""
    print("=" * 70)
    print("OASR Convolution Kernels - Profiling Mode")
    print("=" * 70)
    print(f"Target: {target}, Warmup: {warmup}, Profile iterations: {iters}")
    print("=" * 70)
    
    setup_funcs = {
        'depthwise_conv1d': lambda: setup_depthwise_conv1d(*PROFILE_CONFIGS['depthwise_conv1d']),
        'depthwise_conv1d_causal': lambda: setup_depthwise_conv1d_causal(*PROFILE_CONFIGS['depthwise_conv1d_causal']),
        'pointwise_conv1d': lambda: setup_pointwise_conv1d(*PROFILE_CONFIGS['pointwise_conv1d']),
        'glu': lambda: setup_glu(*PROFILE_CONFIGS['glu']),
        'swish': lambda: setup_swish(*PROFILE_CONFIGS['swish']),
        'batch_norm_swish': lambda: setup_batch_norm_swish(*PROFILE_CONFIGS['batch_norm_swish']),
        'conv_block': lambda: setup_conv_block(*PROFILE_CONFIGS['conv_block']),
    }
    
    for kernel_name in kernels:
        if kernel_name not in setup_funcs:
            print(f"Unknown kernel: {kernel_name}")
            continue
        
        config = PROFILE_CONFIGS[kernel_name]
        print(f"\nProfiling: {kernel_name}")
        print(f"  Config: {config}")
        
        oasr_fn, pytorch_fn = setup_funcs[kernel_name]()
        
        if target in ('oasr', 'both'):
            print(f"  Running OASR kernel...")
            profile_kernel(f"oasr_{kernel_name}", oasr_fn, warmup=warmup, profile_iters=iters)
        
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
        description="OASR Convolution Kernels - Benchmark & Profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python bench_conv_kernels.py

  # Profile specific kernel with Nsight Compute (OASR only)
  ncu --set full -o depthwise_profile python bench_conv_kernels.py --profile --kernel depthwise_conv1d --target oasr

  # Profile all kernels
  python bench_conv_kernels.py --profile --kernel all

Available kernels:
  depthwise_conv1d, depthwise_conv1d_causal, pointwise_conv1d,
  glu, swish, batch_norm_swish, conv_block
        """
    )
    
    parser.add_argument('--profile', action='store_true',
        help='Run in profiling mode (single iterations for ncu)')
    parser.add_argument('--kernel', nargs='+', default=['all'],
        help='Kernel(s) to profile (use "all" for all kernels)')
    parser.add_argument('--target', choices=['oasr', 'pytorch', 'both'], default='oasr',
        help='Which implementation to profile (default: oasr)')
    parser.add_argument('--warmup', type=int, default=3,
        help='Number of warmup iterations for profiling (default: 3)')
    parser.add_argument('--iters', type=int, default=1,
        help='Number of profile iterations (default: 1)')
    
    args = parser.parse_args()
    
    if args.profile:
        if 'all' in args.kernel:
            kernels = list(PROFILE_CONFIGS.keys())
        else:
            kernels = args.kernel
        profile_kernels(kernels, target=args.target, warmup=args.warmup, iters=args.iters)
    else:
        print("=" * 70)
        print("OASR Convolution Kernels - Performance Benchmarks")
        print("=" * 70)
        
        benchmark_depthwise_conv1d()
        benchmark_depthwise_conv1d_causal()
        benchmark_pointwise_conv1d()
        benchmark_glu()
        benchmark_swish()
        benchmark_batch_norm_swish()
        benchmark_conv_block()
        
        print("\n" + "=" * 70)
        print("Benchmarks complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
