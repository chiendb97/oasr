#!/usr/bin/env python3
"""Performance benchmarks for Depthwise Conv1D kernels (standard and causal)."""

import torch
import torch.nn.functional as F

import oasr
try:
    from .utils import run_main
except ImportError:
    from utils import run_main


def setup_depthwise_conv1d(batch_size, seq_len, channels, kernel_size, dtype=torch.float16):
    """Setup tensors for depthwise conv1d."""
    padding = kernel_size // 2
    x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
    weight = torch.randn(kernel_size, channels, device='cuda', dtype=dtype)
    bias = torch.randn(channels, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.depthwise_conv1d(
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
        return oasr.depthwise_conv1d(
            x, weight, bias, 0,
        )

    x_nchw = x.permute(0, 2, 1).contiguous()
    weight_pt = weight.view(channels, 1, kernel_size)

    def pytorch_fn():
        return F.conv1d(x_nchw, weight_pt, bias, padding=0, groups=channels)

    return oasr_fn, pytorch_fn


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


def benchmark_all():
    benchmark_depthwise_conv1d()
    benchmark_depthwise_conv1d_causal()


PROFILE_CONFIGS = {
    'depthwise_conv1d': (64, 250, 512, 31),
    'depthwise_conv1d_causal': (64, 250, 512, 31),
}


def main():
    setup_funcs = {
        'depthwise_conv1d': lambda: setup_depthwise_conv1d(*PROFILE_CONFIGS['depthwise_conv1d']),
        'depthwise_conv1d_causal': lambda: setup_depthwise_conv1d_causal(*PROFILE_CONFIGS['depthwise_conv1d_causal']),
    }
    run_main("DepthwiseConv1D Kernels", PROFILE_CONFIGS, setup_funcs, benchmark_all)


if __name__ == "__main__":
    main()
