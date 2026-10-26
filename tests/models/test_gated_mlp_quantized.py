# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr.layers.GatedMLP`` quantized: gate, up and down as int8 GEMMs.

The fused ``oasr.gated_mlp`` kernel reads the gate and up weights raw, so a
quantized projection must never reach it.  What has to hold:

* **unquantized, nothing moved** -- the forward is bitwise what it was, on the
  fused route and on the two-GEMM route;
* **all three are quantizable**, and once gate or up carries a method the fused
  kernel is never called -- the block runs ``down(act(gate(x)) * up(x))`` with
  each projection on its own scaled GEMM (the gate's SiLU in its epilogue);
* **kernel == reference** on CUDA in the served dtypes, per projection and for
  the block;
* a partly quantized block (only ``down``) keeps the fused route for gate/up.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

import oasr
from oasr.layers import GatedMLP, layers_backend_override
from oasr.layers._backend import use_gated_mlp_kernel
from oasr.layers.mlp import _UNFUSED_ACTIVATION
from oasr.quantization import (
    INT8_W8A8,
    QuantGroup,
    QuantizationSpec,
    apply_quantization,
    bind_quantization,
    quantizable_layers,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

D, H = 64, 128


def _mlp(seed: int = 0) -> GatedMLP:
    torch.manual_seed(seed)
    return GatedMLP(D, H, activation="silu", bias=False).eval()


def _legacy_forward(mlp: GatedMLP, x: torch.Tensor) -> torch.Tensor:
    """The forward as it was before the quantized route existed."""
    gate_proj, up_proj, down_proj = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    if mlp.fuse_gate_up and use_gated_mlp_kernel(
        x, gate_proj.weight, activation=mlp.activation, has_bias=gate_proj.bias is not None
    ):
        h = oasr.gated_mlp(
            x,
            gate_proj.weight,
            up_proj.weight,
            gate_proj.bias,
            up_proj.bias,
            activation=mlp.activation,
        )
    else:
        gate = gate_proj(x)
        if not mlp.fused:
            gate = _UNFUSED_ACTIVATION[mlp.activation](gate)
        h = gate * up_proj(x)
    return down_proj(h)


def _quantize(mlp: GatedMLP, targets=("gate_proj", "up_proj", "down_proj")) -> GatedMLP:
    holder = torch.nn.Module()
    holder.mlp = mlp
    spec = QuantizationSpec((QuantGroup(INT8_W8A8, tuple(f"mlp.{t}" for t in targets)),))
    apply_quantization(holder, spec, quantize=True)
    return mlp


@pytest.fixture
def fused_calls(monkeypatch) -> List[int]:
    """Count calls of the fused kernel (still running it)."""
    calls: List[int] = []
    original = oasr.gated_mlp

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(oasr, "gated_mlp", spy)
    return calls


class TestUnquantizedIsUnchanged:
    def test_cpu(self):
        mlp = _mlp()
        x = torch.randn(3, 5, D)
        with torch.no_grad():
            assert torch.equal(mlp(x), _legacy_forward(mlp, x))

    @cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("rows", [4, 600])
    def test_cuda_both_routes(self, dtype, rows):
        """4 rows sits in the fused kernel's band, 600 outside it."""
        mlp = _mlp().cuda().to(dtype)
        x = torch.randn(rows, D, device="cuda", dtype=dtype)
        with torch.no_grad():
            assert torch.equal(mlp(x), _legacy_forward(mlp, x))


class TestObserved:
    @cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_a_hooked_projection_is_called(self, fused_calls, dtype):
        """Calibration observes a projection's input with a forward hook, and the
        fused kernel never calls the projection: while a hook watches gate or up,
        the block runs the two GEMMs, so every row it serves is seen -- a reduced-
        precision calibration twin on CUDA used to skip the fused kernel's rows."""
        mlp = _mlp().cuda().to(dtype)
        x = torch.randn(4, D, device="cuda", dtype=dtype)  # inside the fused band
        seen: List[int] = []
        with torch.no_grad():
            mlp(x)
            assert fused_calls, "4 rows should take the fused kernel when unobserved"
            fused_calls.clear()
            handle = mlp.up_proj.register_forward_pre_hook(
                lambda m, args: seen.append(args[0].shape[0])
            )
            try:
                got = mlp(x)
            finally:
                handle.remove()
            assert seen == [4] and not fused_calls
            # Same function either way: two GEMMs instead of the dual-B kernel.
            torch.testing.assert_close(got, _legacy_forward(mlp, x), rtol=2e-2, atol=2e-2)
            mlp(x)
        assert fused_calls, "the hook's removal returns the block to the fused kernel"


class TestQuantized:
    def test_all_three_projections_are_quantizable(self):
        holder = torch.nn.Module()
        holder.mlp = _mlp()
        assert set(quantizable_layers(holder)) == {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}

    def test_runs_on_cpu_through_the_reference(self):
        mlp = _mlp()
        x = torch.randn(2, 7, D)
        with torch.no_grad():
            want = mlp(x)
            _quantize(mlp)
            bind_quantization(mlp)
            got = mlp(x)
        assert got.shape == want.shape
        cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
        assert float(cos) > 0.99

    @cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("rows", [1, 4, 600])
    def test_kernel_agrees_with_the_reference_and_never_fuses(self, fused_calls, dtype, rows):
        mlp = _quantize(_mlp()).cuda().to(dtype)
        bind_quantization(mlp)
        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            assert proj.quant_method.lane is not None, proj.quant_method.lane_note
            assert proj.weight.dtype == torch.int8
        x = torch.randn(rows, D, device="cuda", dtype=dtype)
        with torch.no_grad():
            got = mlp(x)
            with layers_backend_override("torch"):
                ref = mlp(x)
        assert fused_calls == [], "the fused kernel was handed int8 weights"
        rel = 2.0**-8 if dtype == torch.float16 else 2.0**-5
        err = (got.float() - ref.float()).abs()
        assert bool((err <= ref.float().abs() * rel + 2e-3).all()), float(err.max())

    @cuda
    def test_the_fused_route_is_taken_when_unquantized(self, fused_calls):
        """The control for the test above: the same shape does reach the fused
        kernel when nothing is quantized, so "never called" is not vacuous."""
        mlp = _mlp().cuda().half()
        x = torch.randn(4, D, device="cuda", dtype=torch.float16)
        if not use_gated_mlp_kernel(x, mlp.gate_proj.weight, activation="silu"):
            pytest.skip("no fused gated-MLP kernel serves this GPU")
        with torch.no_grad():
            mlp(x)
        assert fused_calls == [1]

    @cuda
    def test_only_down_quantized_keeps_the_fused_route(self, fused_calls):
        mlp = _quantize(_mlp(), targets=("down_proj",)).cuda().half()
        bind_quantization(mlp)
        x = torch.randn(4, D, device="cuda", dtype=torch.float16)
        fusable = use_gated_mlp_kernel(x, mlp.gate_proj.weight, activation="silu")
        with torch.no_grad():
            got = mlp(x)
            with layers_backend_override("torch"):
                ref = mlp(x)
        assert fused_calls == ([1] if fusable else [])
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    @cuda
    @pytest.mark.parametrize("quantized", ["gate_proj", "up_proj"])
    def test_one_of_gate_or_up_quantized_leaves_the_fused_kernel(self, fused_calls, quantized):
        mlp = _quantize(_mlp(), targets=(quantized,)).cuda().half()
        bind_quantization(mlp)
        x = torch.randn(4, D, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            got = mlp(x)
            with layers_backend_override("torch"):
                ref = mlp(x)
        assert fused_calls == []
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
