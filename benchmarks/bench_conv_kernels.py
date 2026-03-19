#!/usr/bin/env python3
"""
Performance benchmarks for convolution kernels.
Uses triton.testing.do_bench for accurate GPU timing.

Profiling mode for NVIDIA Nsight Compute:
    ncu --set full -o profile_report python bench_conv_kernels.py \
    --profile --kernel swish --target oasr \
    --warmup 0 --iters 1
"""

import argparse
import torch
import torch.nn.functional as F

import oasr


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
    weight = torch.randn(kernel_size, channels, device='cuda', dtype=dtype)
    bias = torch.randn(channels, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.kernels.conv.depthwise_conv1d(
            x, weight, bias, padding,
        )

    x_nchw = x.permute(0, 2, 1).contiguous()
    weight_pt = weight.view(channels, 1, kernel_size)

    def pytorch_fn():
        return F.conv1d(x_nchw, weight_pt, bias, padding=padding, groups=channels)

    return oasr_fn, pytorch_fn


def setup_depthwise_conv1d_causal(batch_size, seq_len, channels, kernel_size, dtype=torch.float16):
    """Setup tensors for causal depthwise conv1d."""
    x = torch.randn(batch_size, seq_len + kernel_size - 1,
                    channels, device='cuda', dtype=dtype)
    weight = torch.randn(kernel_size, channels, device='cuda', dtype=dtype)
    bias = torch.randn(channels, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.kernels.conv.depthwise_conv1d(
            x, weight, bias, 0,
        )

    x_nchw = x.permute(0, 2, 1).contiguous()
    weight_pt = weight.view(channels, 1, kernel_size)

    def pytorch_fn():
        return F.conv1d(x_nchw, weight_pt, bias, padding=0, groups=channels)

    return oasr_fn, pytorch_fn


def setup_pointwise_conv1d(batch_size, seq_len, in_ch, out_ch, dtype=torch.float16):
    """Setup tensors for pointwise conv1d."""
    x = torch.randn(batch_size, seq_len, in_ch, device='cuda', dtype=dtype)
    weight = torch.randn(out_ch, in_ch, device='cuda', dtype=dtype)
    bias = torch.randn(out_ch, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.kernels.conv.pointwise_conv1d(
            x, weight, bias,
        )

    def pytorch_fn():
        return F.linear(x, weight, bias)

    return oasr_fn, pytorch_fn


def setup_pointwise_conv1d_activation(batch_size, seq_len, in_ch, out_ch, activation, dtype=torch.float16):
    """Setup tensors for pointwise conv1d with activation."""
    x = torch.randn(batch_size, seq_len, in_ch, device='cuda', dtype=dtype)
    weight = torch.randn(out_ch, in_ch, device='cuda', dtype=dtype)
    bias = torch.randn(out_ch, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.kernels.conv.pointwise_conv1d_activation(
            x, weight, bias,
            activation
        )

    def pytorch_fn():
        if activation == oasr.ActivationType.SWISH:
            return F.silu(F.linear(x, weight, bias))
        elif activation == oasr.ActivationType.RELU:
            return F.relu(F.linear(x, weight, bias))
        elif activation == oasr.ActivationType.GELU:
            return F.gelu(F.linear(x, weight, bias))
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

    return oasr_fn, pytorch_fn


def setup_conv_block(batch_size, seq_len, d_model, kernel_size, dtype=torch.float16):
    """Setup tensors for conv block (depthwise + pointwise + GLU + swish)."""
    x = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=dtype)

    pw1_weight = torch.randn(2 * d_model, d_model, device='cuda', dtype=dtype)
    pw1_bias = torch.randn(2 * d_model, device='cuda', dtype=dtype)
    dw_weight = torch.randn(kernel_size, d_model, device='cuda', dtype=dtype)
    dw_bias = torch.randn(d_model, device='cuda', dtype=dtype)
    pw2_weight = torch.randn(d_model, d_model, device='cuda', dtype=dtype)
    pw2_bias = torch.randn(d_model, device='cuda', dtype=dtype)

    dw_weight_pt = dw_weight.view(d_model, 1, kernel_size)

    def oasr_fn():
        pw1_out = oasr.kernels.conv.pointwise_conv1d(
            x, pw1_weight, pw1_bias
        )
        glu_out = oasr.kernels.activation.glu(
            pw1_out
        )
        dw_out = oasr.kernels.conv.depthwise_conv1d(
            glu_out, dw_weight, dw_bias, kernel_size // 2
        )
        swish_out = oasr.kernels.activation.swish(
            dw_out
        )
        return oasr.kernels.conv.pointwise_conv1d(
            swish_out, pw2_weight, pw2_bias
        )

    def pytorch_fn():
        pw1 = F.linear(x, pw1_weight, pw1_bias)
        glu = F.glu(pw1, dim=-1)
        glu_nchw = glu.permute(0, 2, 1)
        dw_nchw = F.conv1d(glu_nchw, dw_weight_pt, dw_bias,
                           padding=kernel_size // 2, groups=d_model)
        dw = dw_nchw.permute(0, 2, 1)
        swish = F.silu(dw)
        return F.linear(swish, pw2_weight, pw2_bias)

    return oasr_fn, pytorch_fn


def setup_conv2d(batch_size, H, W, IC, K, R, S, pad=0, stride=1, dtype=torch.float16):
    """Setup tensors for NHWC Conv2d.

    OASR kernel uses NHWC layout throughout.
    PyTorch baseline uses NCHW F.conv2d; input/weight are permuted before the
    call so the computation is equivalent.
    """
    # OASR: NHWC inputs
    x_nhwc = torch.randn(batch_size, H, W, IC, device='cuda', dtype=dtype)
    w_nhwc = torch.randn(K, R, S, IC, device='cuda', dtype=dtype)
    bias = torch.randn(K, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.kernels.conv.conv2d(
            x_nhwc, w_nhwc, bias, pad, pad, stride, stride)

    # PyTorch: NCHW inputs (permute once outside the timed loop)
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    w_nchw = w_nhwc.permute(0, 3, 1, 2).contiguous()  # [K, IC, R, S]

    def pytorch_fn():
        return F.conv2d(x_nchw, w_nchw, bias, stride=stride, padding=pad)

    return oasr_fn, pytorch_fn


def setup_conv2d_activation(
        batch_size, H, W, IC, K, R, S, pad=0, stride=1,
        activation=None, dtype=torch.float16):
    """Setup tensors for NHWC Conv2d with fused activation."""
    import oasr as _oasr
    if activation is None:
        activation = _oasr.ActivationType.SWISH

    x_nhwc = torch.randn(batch_size, H, W, IC, device='cuda', dtype=dtype)
    w_nhwc = torch.randn(K, R, S, IC, device='cuda', dtype=dtype)
    bias = torch.randn(K, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.kernels.conv.conv2d_activation(
            x_nhwc, w_nhwc, bias, activation, pad, pad, stride, stride)

    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    w_nchw = w_nhwc.permute(0, 3, 1, 2).contiguous()

    if activation == _oasr.ActivationType.RELU:
        act_fn = F.relu
    elif activation == _oasr.ActivationType.GELU:
        act_fn = F.gelu
    else:
        act_fn = F.silu  # SWISH

    def pytorch_fn():
        return act_fn(F.conv2d(x_nchw, w_nchw, bias, stride=stride, padding=pad))

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
        (32, 125, 256, 31),
        (16, 125, 256, 15),
        (64, 125, 512, 31),
    ]

    print("\n" + "=" * 70)
    print("DepthwiseConv1D Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<35} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 75)

    for batch_size, seq_len, channels, kernel_size in configs:
        oasr_fn, pytorch_fn = setup_depthwise_conv1d(
            batch_size, seq_len, channels, kernel_size)

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
        (64, 500, 256, 15),
        (64, 500, 512, 31),
        (32, 125, 256, 15),
        (32, 125, 256, 31),
        (16, 125, 256, 15),
        (64, 125, 512, 31),
    ]

    print("\n" + "=" * 70)
    print("DepthwiseConv1D (Causal) Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<35} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 75)

    for batch_size, seq_len, channels, kernel_size in configs:
        oasr_fn, pytorch_fn = setup_depthwise_conv1d_causal(
            batch_size, seq_len, channels, kernel_size)

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
        oasr_fn, pytorch_fn = setup_pointwise_conv1d(
            batch_size, seq_len, in_ch, out_ch)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {in_ch}] -> {out_ch}"
        print(f"{shape_str:<40} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_pointwise_conv1d_activation():
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
        oasr_fn, pytorch_fn = setup_pointwise_conv1d_activation(
            batch_size, seq_len, in_ch, out_ch, oasr.ActivationType.SWISH)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {in_ch}] -> {out_ch}"
        print(f"{shape_str:<40} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


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
        oasr_fn, pytorch_fn = setup_conv_block(
            batch_size, seq_len, d_model, kernel_size)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        config_str = f"[{batch_size}, {seq_len}, {d_model}] k={kernel_size}"
        print(
            f"{config_str:<35} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


# =============================================================================
# Profiling functions
# =============================================================================

def benchmark_conv2d():
    """Benchmark Conv2d NHWC: OASR CUTLASS vs PyTorch NCHW F.conv2d."""
    import triton

    # (batch, H, W, IC, K, R, S, pad, stride)
    # First two rows mirror Conv2dSubsampling (IC=1 then IC=odim);
    # remaining rows cover larger aligned shapes.
    configs = [
        # subsampling conv1 (IC=1, kAnalytic)
        (16, 200, 80,   1,  64, 3, 3, 0, 2),
        (16, 100, 40,  64,  64, 3, 3, 0, 2),  # subsampling conv2
        (32, 200, 80,   1,  64, 3, 3, 0, 2),
        (32, 100, 40,  64,  64, 3, 3, 0, 2),
        (8,  300, 80,   1, 256, 3, 3, 0, 2),
        (8,  150, 40, 256, 256, 3, 3, 0, 2),
        (16, 100, 40, 128, 128, 3, 3, 1, 1),  # same-size 128-ch conv
        (16, 100, 40, 256, 256, 3, 3, 1, 1),  # same-size 256-ch conv
    ]

    print("\n" + "=" * 75)
    print("Conv2d NHWC Benchmark (CUTLASS Ampere Tensor Core vs PyTorch NCHW)")
    print("=" * 75)
    print(
        f"\n{'Shape [N,H,W,IC->K]':<38} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 78)

    for N, H, W, IC, K, R, S, pad, stride in configs:
        oasr_fn, pytorch_fn = setup_conv2d(N, H, W, IC, K, R, S, pad, stride)
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{N},{H},{W},{IC}->{K}] {R}x{S} p={pad} s={stride}"
        print(f"{shape_str:<38} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_conv2d_activation():
    """Benchmark Conv2d + fused activation: OASR CUTLASS vs PyTorch."""
    import triton

    configs = [
        (16, 200, 80,   1,  64, 3, 3, 0, 2),
        (16, 100, 40,  64,  64, 3, 3, 0, 2),
        (32, 200, 80,   1,  64, 3, 3, 0, 2),
        (32, 100, 40,  64,  64, 3, 3, 0, 2),
        (8,  150, 40, 256, 256, 3, 3, 0, 2),
        (16, 100, 40, 256, 256, 3, 3, 1, 1),
    ]

    activations = [
        (oasr.ActivationType.RELU,  "relu"),
        (oasr.ActivationType.SWISH, "swish"),
        (oasr.ActivationType.GELU,  "gelu"),
    ]

    for act_type, act_name in activations:
        print("\n" + "=" * 78)
        print(
            f"Conv2d + {act_name.upper()} (fused, CUTLASS) vs PyTorch unfused")
        print("=" * 78)
        print(
            f"\n{'Shape [N,H,W,IC->K]':<38} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
        print("-" * 78)

        for N, H, W, IC, K, R, S, pad, stride in configs:
            oasr_fn, pytorch_fn = setup_conv2d_activation(
                N, H, W, IC, K, R, S, pad, stride, activation=act_type)
            oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
            pytorch_ms = triton.testing.do_bench(
                pytorch_fn, warmup=100, rep=500)
            speedup = pytorch_ms / oasr_ms
            shape_str = f"[{N},{H},{W},{IC}->{K}] {R}x{S} p={pad} s={stride}"
            print(
                f"{shape_str:<38} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


PROFILE_CONFIGS = {
    'depthwise_conv1d': (64, 250, 512, 31),
    'depthwise_conv1d_causal': (64, 250, 512, 31),
    'pointwise_conv1d': (64, 250, 512, 1024),
    'conv_block': (64, 250, 512, 31),
    # (batch, H, W, IC, K, R, S, pad, stride)
    'conv2d': (16, 100, 40, 256, 256, 3, 3, 1, 1),
    'conv2d_activation': (16, 100, 40, 256, 256, 3, 3, 1, 1),
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
        'conv_block': lambda: setup_conv_block(*PROFILE_CONFIGS['conv_block']),
        'conv2d': lambda: setup_conv2d(*PROFILE_CONFIGS['conv2d']),
        'conv2d_activation': lambda: setup_conv2d_activation(*PROFILE_CONFIGS['conv2d_activation']),
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
            profile_kernel(f"oasr_{kernel_name}", oasr_fn,
                           warmup=warmup, profile_iters=iters)

        if target in ('pytorch', 'both'):
            print(f"  Running PyTorch kernel...")
            profile_kernel(
                f"pytorch_{kernel_name}", pytorch_fn, warmup=warmup, profile_iters=iters)

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
  conv_block, conv2d, conv2d_activation
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
        profile_kernels(kernels, target=args.target,
                        warmup=args.warmup, iters=args.iters)
    else:
        print("=" * 70)
        print("OASR Convolution Kernels - Performance Benchmarks")
        print("=" * 70)

        benchmark_depthwise_conv1d()
        benchmark_depthwise_conv1d_causal()
        benchmark_pointwise_conv1d()
        benchmark_conv2d()
        benchmark_conv2d_activation()
        benchmark_conv_block()

        print("\n" + "=" * 70)
        print("Benchmarks complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
