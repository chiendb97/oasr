#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``Linear.batch_invariant`` / ``LinearActivation.batch_invariant``: the layer pin.

The kernel-level guarantee is tested next door
(``tests/kernels/test_gemm_batch_invariant.py``).  What the layer adds, and what
is tested here, is routing: a pinned projection takes the pinned CUTLASS path
at *every* M -- below ``GEMM_MIN_ROWS`` too, where the unpinned layer uses
``F.linear`` (cuBLAS choosing its kernel from M) -- and nothing changes for an
unpinned layer or off the CUDA kernel path.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import oasr
import oasr.functionals.gemm as fgemm
from oasr.layers import Linear, LinearActivation
from oasr.layers._backend import GEMM_MIN_ROWS, layers_backend_override

#: Conformer's ``encoder.embed.out.0``: the projection the pin was built for.
IN, OUT = 4864, 256
_SMALL_M = 16  # below the row floor
_LARGE_M = 3000


def _layer(cls=Linear, dtype=torch.float16, device="cuda", **kw):
    torch.manual_seed(0)
    return cls(IN, OUT, device=device, dtype=dtype, **kw)


def _x(M, dtype=torch.float16, device="cuda"):
    g = torch.Generator(device=device).manual_seed(1)
    return torch.randn(M, IN, device=device, dtype=dtype, generator=g)


class _Spy:
    """Record calls to a callable while still forwarding them."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.fn(*args, **kwargs)


def test_small_m_is_below_the_row_floor():
    assert _SMALL_M < GEMM_MIN_ROWS


@pytest.mark.cuda
class TestPinnedLinear:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_rows_do_not_depend_on_m(self, dtype):
        layer = _layer(dtype=dtype)
        layer.batch_invariant = True
        x = _x(_LARGE_M, dtype)
        with torch.no_grad():
            ref = layer(x)
            for M in (1, 7, _SMALL_M, GEMM_MIN_ROWS, 1000):
                got = layer(x[:M])
                assert torch.equal(got, ref[:M]), f"rows differ at M={M}"

    def test_below_the_row_floor_takes_the_cutlass_kernel(self, monkeypatch):
        layer = _layer()
        layer.batch_invariant = True

        def refuse(*a, **k):
            raise AssertionError("a pinned projection fell through to F.linear")

        gemm_spy = _Spy(oasr.gemm)
        plan_spy = _Spy(fgemm._plan_pinned)
        plans = []

        def recording_plan(*args, **kwargs):
            plan = plan_spy(*args, **kwargs)
            plans.append(plan)
            return plan

        monkeypatch.setattr(F, "linear", refuse)
        monkeypatch.setattr(oasr, "gemm", gemm_spy)
        monkeypatch.setattr(fgemm, "_plan_pinned", recording_plan)
        with torch.no_grad():
            layer(_x(_SMALL_M))
        assert gemm_spy.calls == [{"batch_invariant": True}]
        assert plans and all(p[2] == fgemm._PLAN_CUTLASS for p in plans)

    def test_unpinned_behaviour_is_unchanged(self, monkeypatch):
        layer = _layer()
        assert layer.batch_invariant is False
        linear_spy = _Spy(F.linear)
        gemm_spy = _Spy(oasr.gemm)
        monkeypatch.setattr(F, "linear", linear_spy)
        monkeypatch.setattr(oasr, "gemm", gemm_spy)
        with torch.no_grad():
            small = layer(_x(_SMALL_M))
            layer(_x(GEMM_MIN_ROWS))
        assert len(linear_spy.calls) == 1, "below the floor an unpinned layer uses F.linear"
        assert gemm_spy.calls == [{}], "at the floor it uses oasr.gemm, unpinned"
        monkeypatch.undo()
        assert torch.equal(small, F.linear(_x(_SMALL_M), layer.weight, layer.bias))

    def test_pin_is_per_instance_and_not_state(self):
        a, b = _layer(), _layer()
        a.batch_invariant = True
        assert b.batch_invariant is False and Linear.batch_invariant is False
        assert set(a.state_dict()) == set(b.state_dict()) == {"weight", "bias"}
        assert "batch_invariant=True" in repr(a) and "batch_invariant" not in repr(b)

    def test_quant_method_runs_first(self):
        layer = _layer()
        layer.batch_invariant = True
        sentinel = torch.zeros(1)

        class _Fake:
            def apply(self, module, x):
                return sentinel

        layer.quant_method = _Fake()
        assert layer(_x(_SMALL_M)) is sentinel

    @pytest.mark.parametrize("activation", ["relu", "swish", "gelu"])
    def test_linear_activation(self, activation, monkeypatch):
        layer = _layer(LinearActivation, activation_type=activation)
        layer.batch_invariant = True
        x = _x(_LARGE_M)
        with torch.no_grad():
            ref = layer(x)
            for M in (1, _SMALL_M, 1000):
                assert torch.equal(layer(x[:M]), ref[:M]), f"rows differ at M={M}"

        def refuse(*a, **k):
            raise AssertionError("a pinned projection fell through to F.linear")

        spy = _Spy(oasr.gemm_activation)
        monkeypatch.setattr(F, "linear", refuse)
        monkeypatch.setattr(oasr, "gemm_activation", spy)
        with torch.no_grad():
            layer(x[:_SMALL_M])
        assert spy.calls == [{"batch_invariant": True}]

    def test_torch_backend_keeps_its_route(self, monkeypatch):
        layer = _layer()
        layer.batch_invariant = True

        def refuse(*a, **k):
            raise AssertionError("OASR_LAYERS_BACKEND=torch reached oasr.gemm")

        monkeypatch.setattr(oasr, "gemm", refuse)
        with layers_backend_override("torch"), torch.no_grad():
            got = layer(_x(_SMALL_M))
        assert got.shape == (_SMALL_M, OUT)


class TestOffTheKernelPath:
    """CPU and fp32 are out of the kernel's scope: the pin changes nothing there."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    def test_cpu(self, dtype):
        layer = _layer(dtype=dtype, device="cpu")
        layer.batch_invariant = True
        x = _x(_SMALL_M, dtype=dtype, device="cpu")
        with torch.no_grad():
            assert torch.equal(layer(x), F.linear(x, layer.weight, layer.bias))

    @pytest.mark.cuda
    def test_cuda_fp32(self):
        layer = _layer(dtype=torch.float32)
        layer.batch_invariant = True
        x = _x(_SMALL_M, dtype=torch.float32)
        with torch.no_grad():
            assert torch.equal(layer(x), F.linear(x, layer.weight, layer.bias))
