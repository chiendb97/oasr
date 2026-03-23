#!/usr/bin/env python3
"""Performance benchmarks for Conv2D NHWC kernels (with and without fused activation)."""

import torch
import torch.nn.functional as F

import oasr
try:
    from .utils import run_main
except ImportError:
    from utils import run_main


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
        return oasr.conv2d(
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
    if activation is None:
        activation = oasr.ACTIVATION_SWISH

    x_nhwc = torch.randn(batch_size, H, W, IC, device='cuda', dtype=dtype)
    w_nhwc = torch.randn(K, R, S, IC, device='cuda', dtype=dtype)
    bias = torch.randn(K, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.conv2d_activation(
            x_nhwc, w_nhwc, bias, activation, pad, pad, stride, stride)

    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    w_nchw = w_nhwc.permute(0, 3, 1, 2).contiguous()

    if activation == oasr.ACTIVATION_RELU:
        act_fn = F.relu
    elif activation == oasr.ACTIVATION_GELU:
        act_fn = F.gelu
    else:
        act_fn = F.silu  # SWISH

    def pytorch_fn():
        return act_fn(F.conv2d(x_nchw, w_nchw, bias, stride=stride, padding=pad))

    return oasr_fn, pytorch_fn


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
        (oasr.ACTIVATION_RELU,  "relu"),
        (oasr.ACTIVATION_SWISH, "swish"),
        (oasr.ACTIVATION_GELU,  "gelu"),
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


def benchmark_all():
    benchmark_conv2d()
    benchmark_conv2d_activation()


PROFILE_CONFIGS = {
    'conv2d': (16, 100, 40, 256, 256, 3, 3, 1, 1),
    'conv2d_activation': (16, 100, 40, 256, 256, 3, 3, 1, 1),
}


def main():
    setup_funcs = {
        'conv2d': lambda: setup_conv2d(*PROFILE_CONFIGS['conv2d']),
        'conv2d_activation': lambda: setup_conv2d_activation(*PROFILE_CONFIGS['conv2d_activation']),
    }
    run_main("Conv2D NHWC Kernels", PROFILE_CONFIGS, setup_funcs, benchmark_all)


if __name__ == "__main__":
    main()
