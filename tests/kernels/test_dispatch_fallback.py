# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Which launch failures the GEMM / BMM / Conv1D dispatch answers with the default.

One kind only: a variant that refuses the problem before anything launches
(``VariantUnsupportedError`` -- CUTLASS's ``can_implement``, or no path for the
arguments).  The dispatch remembers it and says so once.  Every other failure --
a CUDA error, a failed shared-memory opt-in, a cuBLAS error -- propagates.  The
dispatch used to catch every exception and relaunch the default, which turned a
sticky CUDA error into one that surfaced calls later, and a variant that could
never launch (two modules sharing a CUTLASS kernel's one-time init) into a
silent slowdown.

The CPU half drives the real planner and dispatch functions with fake
launchers; the CUDA half shows a real launcher raising the distinct error and
the dispatch falling back from it.
"""

from __future__ import annotations

import logging
import types

import pytest
import torch

import oasr.functionals.conv as fc
import oasr.functionals.gemm as fg
import oasr.jit.gemm as jg
from oasr.jit.errors import VariantUnsupportedError

_REFUSED = VariantUnsupportedError("gemm_fake cannot implement this problem (NOT_SUPPORTED)")
_FAILED = RuntimeError("gemm_fake failed: CUTLASS_ERROR -- an illegal memory access")


class _Kernel:
    """A compiled-variant stand-in: raises *exc* when given, counts its calls."""

    def __init__(self, exc=None):
        self.exc = exc
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        if self.exc is not None:
            raise self.exc


_CHOICE = types.SimpleNamespace(compile_name="fake_tile", split_k=1)


@pytest.fixture
def fresh_plans(monkeypatch):
    """An empty plan cache and warning set, and a fixed arch."""
    monkeypatch.setattr(fg, "_PLANS", {})
    monkeypatch.setattr(fg, "_PLAN_WARNED", set())
    monkeypatch.setattr(fg, "_target_sm", lambda: 120)


@pytest.fixture
def fresh_conv(monkeypatch):
    """An empty Conv1D warning set and a fixed arch."""
    monkeypatch.setattr(fc, "_CONV1D_WARNED", set())
    monkeypatch.setattr(fc, "_target_sm", lambda: 120)


def _route(monkeypatch, op, variant, default):
    """Plan every *op* shape onto *variant*; return ``(call, default_counter)``.

    ``call`` runs the op's real dispatch function once on throwaway tensors.
    """
    monkeypatch.setattr(jg, "select_default_config", lambda *a, **k: _CHOICE)
    monkeypatch.setattr(jg, "select_batch_invariant_config", lambda *a, **k: _CHOICE)
    monkeypatch.setattr(jg, "is_batch_invariant", lambda choice: True)
    monkeypatch.setattr(fg, "_gemm_fn", lambda name, activation: variant)
    monkeypatch.setattr(fg, "_bmm_fn", lambda name: variant)
    monkeypatch.setattr(fg, "_log_softmax_inplace", lambda out: None)
    monkeypatch.setattr(fg, "_default_gemm_fn", lambda: default)
    monkeypatch.setattr(fg, "_default_gemm_activation_fn", lambda: default)
    monkeypatch.setattr(fg, "_default_bmm_fn", lambda: default)
    monkeypatch.setattr(
        fg, "_get_gemm_log_softmax_module", lambda: types.SimpleNamespace(gemm_log_softmax=default)
    )
    M, N, K = 16, 32, 64
    A, B = torch.zeros(M, K, dtype=torch.float16), torch.zeros(N, K, dtype=torch.float16)
    out = torch.zeros(M, N, dtype=torch.float16)
    calls = {
        "gemm": lambda: fg._dispatch_gemm(out, A, B, None, N, K, M),
        "gemm_activation": lambda: fg._dispatch_gemm_activation(out, A, B, None, 2, N, K, M),
        "gemm_log_softmax": lambda: fg._dispatch_gemm_log_softmax(out, A, B, None, N, K, M),
        "bmm": lambda: fg._dispatch_bmm(out.view(1, M, N), A.view(1, M, K), B.view(1, N, K), N, K),
        "gemm_pinned": lambda: fg._dispatch_gemm_pinned(out, A, B, None, N, K, M),
        "gemm_activation_pinned": lambda: fg._dispatch_gemm_activation_pinned(
            out, A, B, None, 2, N, K, M
        ),
    }
    return calls[op]


_OPS = [
    "gemm",
    "gemm_activation",
    "gemm_log_softmax",
    "bmm",
    "gemm_pinned",
    "gemm_activation_pinned",
]


@pytest.mark.usefixtures("fresh_plans")
class TestGemmFamilyDispatch:
    @pytest.mark.parametrize("op", _OPS)
    def test_a_failed_launch_propagates_and_the_default_never_runs(self, op, monkeypatch):
        default = _Kernel()
        call = _route(monkeypatch, op, _Kernel(_FAILED), default)
        with pytest.raises(RuntimeError, match="CUTLASS_ERROR"):
            call()
        assert default.calls == 0

    @pytest.mark.parametrize("op", _OPS)
    def test_a_refusing_variant_falls_back_once_and_is_remembered(self, op, monkeypatch, caplog):
        variant, default = _Kernel(_REFUSED), _Kernel()
        call = _route(monkeypatch, op, variant, default)
        with caplog.at_level(logging.WARNING, logger="oasr.functionals.gemm"):
            for _ in range(3):
                call()
        assert default.calls == 3
        assert variant.calls == 1, "the plan must remember the refusal, not retry it per call"
        warned = [r for r in caplog.records if "cannot implement" in r.getMessage()]
        assert len(warned) == 1, [r.getMessage() for r in caplog.records]

    def test_the_cublas_route_refuses_an_out_it_would_have_written_a_copy_of(self, monkeypatch):
        """An N-D ``out`` whose leading dims cannot merge into M: ``reshape`` made
        it a copy, so the product went into a tensor the caller never sees.  The
        CUTLASS launchers already refuse such an ``out``; now both routes do."""
        monkeypatch.setattr(jg, "select_default_config", lambda *a, **k: "torch")
        T, Bt, N, K = 3, 2, 8, 16
        A, B = torch.randn(Bt, T, K), torch.randn(N, K)
        out = torch.zeros(T, Bt, N).transpose(0, 1)  # (B, T, N) over a (T, B, N) buffer
        with pytest.raises(RuntimeError, match="view"):
            fg._dispatch_gemm(out, A, B, None, N, K, Bt * T)


@pytest.mark.usefixtures("fresh_conv")
class TestConv1dDispatch:
    def _route(self, monkeypatch, variant, default, *, compiled=True):
        choice = types.SimpleNamespace(compile_name="fake_tile")
        monkeypatch.setattr(fc._jit_conv, "select_default_conv1d_config", lambda *a: choice)
        monkeypatch.setattr(
            fc._jit_conv, "select_default_conv1d_activation_config", lambda *a: choice
        )

        def resolve(name):
            if not compiled:
                raise AttributeError(f"no conv1d_{name}")
            return variant

        monkeypatch.setattr(fc, "_conv1d_fn", resolve)
        monkeypatch.setattr(fc, "_conv1d_activation_fn", resolve)
        monkeypatch.setattr(fc, "_default_conv1d_fn", lambda: default)
        monkeypatch.setattr(fc, "_default_conv1d_activation_fn", lambda: default)
        x, w = torch.zeros(2, 10, 8), torch.zeros(16, 3, 8)
        out = torch.zeros(2, 8, 16)
        return (
            lambda: fc._dispatch_conv1d(out, x, w, None, 0, 1, 1),
            lambda: fc._dispatch_conv1d_activation(out, x, w, None, 2, 0, 1, 1),
        )

    def test_a_failed_launch_propagates(self, monkeypatch):
        default = _Kernel()
        for call in self._route(monkeypatch, _Kernel(_FAILED), default):
            with pytest.raises(RuntimeError, match="CUTLASS_ERROR"):
                call()
        assert default.calls == 0

    @pytest.mark.parametrize("compiled", [True, False], ids=["refused", "not-compiled"])
    def test_a_tile_that_cannot_serve_falls_back_with_one_warning(
        self, compiled, monkeypatch, caplog
    ):
        default = _Kernel()
        calls = self._route(monkeypatch, _Kernel(_REFUSED), default, compiled=compiled)
        with caplog.at_level(logging.WARNING, logger="oasr.functionals.conv"):
            for call in calls:
                call()
                call()
        assert default.calls == 4
        assert len(caplog.records) == 2  # once per op, not once per call


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestARealLauncher:
    """Serial split-K applies the epilogue per K partition, so the CUTLASS 2.x
    template refuses it with a fused activation -- a real ``NOT_SUPPORTED``."""

    def _operands(self):
        torch.manual_seed(0)
        A = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        B = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        return A, B, torch.empty(64, 64, device="cuda", dtype=torch.float16)

    def test_raises_variant_unsupported_and_stays_a_runtime_error(self):
        if isinstance(jg.GEMM_DEFAULT, jg.CutlassGemmConfigSm90):
            pytest.skip("the default is a CUTLASS 3.x config, which takes no split-K")
        A, B, out = self._operands()
        with pytest.raises(VariantUnsupportedError, match="NOT_SUPPORTED") as info:
            fg._default_gemm_activation_fn()(out, A, B, None, 2, 2)  # swish, split_k=2
        assert isinstance(info.value, RuntimeError)
        assert jg.gemm_activation_func_name(jg.GEMM_DEFAULT) in str(info.value)
        torch.cuda.synchronize()  # nothing was launched, so nothing is pending or poisoned

    def test_the_dispatch_falls_back_to_a_correct_result(self, monkeypatch, caplog):
        if isinstance(jg.GEMM_DEFAULT, jg.CutlassGemmConfigSm90):
            pytest.skip("the default is a CUTLASS 3.x config, which takes no split-K")
        import dataclasses

        monkeypatch.setattr(fg, "_PLANS", {})
        monkeypatch.setattr(fg, "_PLAN_WARNED", set())
        split = dataclasses.replace(jg.GEMM_DEFAULT, split_k=2)
        monkeypatch.setattr(jg, "select_default_config", lambda *a, **k: split)
        A, B, out = self._operands()
        with caplog.at_level(logging.WARNING, logger="oasr.functionals.gemm"):
            fg._dispatch_gemm_activation(out, A, B, None, 2, 64, 128, 64)
        ref = torch.nn.functional.silu(A.float() @ B.float().t())
        torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
        assert any("cannot implement" in r.getMessage() for r in caplog.records)
