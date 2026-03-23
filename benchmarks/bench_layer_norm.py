#!/usr/bin/env python3
"""Performance benchmarks for LayerNorm kernels (standard, fused add, and fused activation)."""

import torch

import oasr
try:
    from .utils import run_main
except ImportError:
    from utils import run_main


def setup_layer_norm(batch_size, seq_len, hidden_size, dtype=torch.float16):
    """Setup tensors for LayerNorm."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, hidden_size,
                    device='cuda', dtype=dtype)
    gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
    beta = torch.randn(hidden_size, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.layer_norm(
            x, gamma, beta, eps
        )

    layer_norm = torch.nn.LayerNorm(
        hidden_size, eps=eps, device='cuda', dtype=dtype)
    layer_norm.weight.data = gamma.clone()
    layer_norm.bias.data = beta.clone()

    def pytorch_fn():
        return layer_norm(x)

    return oasr_fn, pytorch_fn


def setup_add_layer_norm(batch_size, seq_len, hidden_size, dtype=torch.float16):
    """Setup tensors for fused Add + LayerNorm."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, hidden_size,
                    device='cuda', dtype=dtype)
    residual = torch.randn(batch_size, seq_len,
                           hidden_size, device='cuda', dtype=dtype)
    gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
    beta = torch.randn(hidden_size, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.add_layer_norm(
            x, residual, gamma, beta, eps
        )

    layer_norm = torch.nn.LayerNorm(
        hidden_size, eps=eps, device='cuda', dtype=dtype)
    layer_norm.weight.data = gamma.clone()
    layer_norm.bias.data = beta.clone()

    def pytorch_fn():
        return layer_norm(x + residual)

    return oasr_fn, pytorch_fn


def setup_layer_norm_activation(batch_size, seq_len, hidden_size, activation_name="swish", dtype=torch.float16):
    """Setup tensors for fused LayerNorm + Activation."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, hidden_size,
                    device='cuda', dtype=dtype)
    gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
    beta = torch.randn(hidden_size, device='cuda', dtype=dtype)

    from oasr.utils.mappings import get_activation
    act_type_id = oasr.get_activation_type_id(activation_name)
    torch_act = get_activation(activation_name).cuda()

    def oasr_fn():
        return oasr.layer_norm_activation(
            x, gamma, beta, eps, act_type_id
        )

    layer_norm = torch.nn.LayerNorm(
        hidden_size, eps=eps, device='cuda', dtype=dtype)
    layer_norm.weight.data = gamma.clone()
    layer_norm.bias.data = beta.clone()

    def pytorch_fn():
        return torch_act(layer_norm(x))

    return oasr_fn, pytorch_fn


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
    print("LayerNorm Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)

    for batch_size, seq_len, hidden_size in configs:
        oasr_fn, pytorch_fn = setup_layer_norm(
            batch_size, seq_len, hidden_size)

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
        oasr_fn, pytorch_fn = setup_add_layer_norm(
            batch_size, seq_len, hidden_size)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {hidden_size}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_layer_norm_activation():
    """Benchmark fused LayerNorm + Activation: OASR fused vs PyTorch separate."""
    import triton

    configs = [
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 512),
    ]
    activations = ["relu", "gelu", "swish"]

    print("\n" + "=" * 80)
    print("LayerNorm + Activation (Fused) Benchmark")
    print("=" * 80)
    print(
        f"\n{'Shape':<25} {'Act':<8} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 80)

    for batch_size, seq_len, hidden_size in configs:
        for act_name in activations:
            oasr_fn, pytorch_fn = setup_layer_norm_activation(
                batch_size, seq_len, hidden_size, act_name)

            oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
            pytorch_ms = triton.testing.do_bench(
                pytorch_fn, warmup=100, rep=500)

            speedup = pytorch_ms / oasr_ms
            shape_str = f"[{batch_size}, {seq_len}, {hidden_size}]"
            print(
                f"{shape_str:<25} {act_name:<8} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_all():
    benchmark_layer_norm()
    benchmark_add_layer_norm()
    benchmark_layer_norm_activation()


PROFILE_CONFIGS = {
    'layer_norm': (64, 250, 256),
    'add_layer_norm': (64, 250, 512),
    'layer_norm_activation': (64, 250, 512),
}


def main():
    setup_funcs = {
        'layer_norm': lambda: setup_layer_norm(*PROFILE_CONFIGS['layer_norm']),
        'add_layer_norm': lambda: setup_add_layer_norm(*PROFILE_CONFIGS['add_layer_norm']),
        'layer_norm_activation': lambda: setup_layer_norm_activation(*PROFILE_CONFIGS['layer_norm_activation']),
    }
    run_main("LayerNorm Kernels", PROFILE_CONFIGS, setup_funcs, benchmark_all)


if __name__ == "__main__":
    main()
