#!/usr/bin/env python3
"""Performance benchmarks for the full Conformer conv block (end-to-end)."""

import torch
import torch.nn.functional as F

import oasr
try:
    from .utils import run_main
except ImportError:
    from utils import run_main


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
        pw1_out = oasr.pointwise_conv1d(
            x, pw1_weight, pw1_bias
        )
        glu_out = oasr.glu(
            pw1_out
        )
        dw_out = oasr.depthwise_conv1d(
            glu_out, dw_weight, dw_bias, kernel_size // 2
        )
        swish_out = oasr.swish(
            dw_out
        )
        return oasr.pointwise_conv1d(
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


PROFILE_CONFIGS = {
    'conv_block': (64, 250, 512, 31),
}


def main():
    setup_funcs = {
        'conv_block': lambda: setup_conv_block(*PROFILE_CONFIGS['conv_block']),
    }
    run_main("Conv Block Kernel", PROFILE_CONFIGS, setup_funcs, benchmark_conv_block)


if __name__ == "__main__":
    main()
