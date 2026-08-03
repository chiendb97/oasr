#!/usr/bin/env python3
"""Tests for the shape-aware GEMM selection heuristic + torch/cuBLAS backend.

Covers:
  * the torch backend runners (oasr.gemm_torch) vs a torch reference;
  * select_default_config() routing, fallback, and *actionability* (every CUTLASS
    config it returns must correspond to a compiled kernel);
  * the production (non-autotuned) gemm / gemm_activation path numerics for the
    real Conformer-CTC FF / conv shapes at small (streaming) and large (offline) M.
"""

import pytest
import torch
import torch.nn.functional as F

import oasr
from oasr.gemm_torch import torch_bmm, torch_gemm, torch_gemm_activation
from oasr.jit.core import _get_target_sm
from oasr.jit.gemm import (
    GEMM_DEFAULT,
    CutlassGemmConfig,
    get_unique_compile_configs,
    select_default_config,
)

# Every test in this module allocates directly on ``device="cuda"`` and calls a
# JIT-compiled kernel, so the whole file is CUDA-only.  Declaring that here is
# what lets the CPU CI job run `pytest tests/` and get a green, meaningful run
# instead of a wall of `RuntimeError: No CUDA GPUs are available`.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="OASR kernels require CUDA")


_SM = _get_target_sm()

# (N, K) pairs that actually hit the OASR GEMM path for Conformer-CTC base.
_FF_CONV_SHAPES = [(256, 2048), (256, 4864), (256, 256), (512, 256)]


class TestTorchBackend:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_torch_gemm(self, dtype):
        M, N, K = 48, 256, 2048
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        C = torch.randn(N, device="cuda", dtype=dtype)
        out = torch.empty(M, N, device="cuda", dtype=dtype)
        torch_gemm(out, A, B, C)
        torch.testing.assert_close(out, F.linear(A, B, C), rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("act,ref", [(0, F.relu), (1, F.gelu), (2, F.silu)])
    def test_torch_gemm_activation(self, dtype, act, ref):
        M, N, K = 48, 2048, 256
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        C = torch.randn(N, device="cuda", dtype=dtype)
        out = torch.empty(M, N, device="cuda", dtype=dtype)
        torch_gemm_activation(out, A, B, C, act)
        torch.testing.assert_close(out, ref(F.linear(A, B, C)), rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_torch_bmm(self, dtype):
        Bc, M, N, K = 8, 64, 128, 256
        A = torch.randn(Bc, M, K, device="cuda", dtype=dtype)
        B = torch.randn(Bc, N, K, device="cuda", dtype=dtype)
        out = torch.empty(Bc, M, N, device="cuda", dtype=dtype)
        torch_bmm(out, A, B)
        torch.testing.assert_close(out, torch.bmm(A, B.transpose(-1, -2)), rtol=2e-2, atol=2e-2)


class TestSelectDefaultConfig:
    def test_fallback_unknown_shape(self):
        assert select_default_config("gemm", 64, 999, 999, torch.bfloat16, _SM) is GEMM_DEFAULT

    def test_fallback_fp32(self):
        # fp32 is gated off (small tiles assume 2-byte operands).
        assert select_default_config("gemm", 64, 256, 2048, torch.float32, _SM) is GEMM_DEFAULT

    def test_fallback_other_arch(self):
        # Rules are SM120-specific; any other arch falls back to the default.
        assert select_default_config("gemm", 64, 256, 2048, torch.bfloat16, 80) is GEMM_DEFAULT

    @pytest.mark.skipif(_SM != 120, reason="heuristic rules are SM120-specific")
    def test_thin_contract_routes_to_torch(self):
        # Thin FF/subsampling contract GEMMs at small M go to cuBLAS.
        assert select_default_config("gemm", 64, 256, 2048, torch.bfloat16, 120) == "torch"
        assert select_default_config("gemm", 64, 256, 4864, torch.bfloat16, 120) == "torch"

    @pytest.mark.skipif(_SM != 120, reason="heuristic rules are SM120-specific")
    def test_small_m_avoids_large_default(self):
        # At small M the selector never wastes the 128-row default: with the
        # thin-N tiles in the candidate space, (256,256) goes to a small-tile
        # CUTLASS config across the small/mid band — never 128x128.
        for m in (64, 848):
            cfg = select_default_config("gemm", m, 256, 256, torch.bfloat16, 120)
            assert isinstance(cfg, CutlassGemmConfig)
            assert cfg.block_m < 128  # a tall-thin tile, not the 128-row default

    @pytest.mark.skipif(_SM != 120, reason="heuristic rules are SM120-specific")
    def test_large_m_contract_avoids_default(self):
        # The deep-K thin contract GEMM (FF-down, N=256 K=2048) at large
        # offline M: the expanded candidate space (thin-N tiles + working
        # split-K) beats both the 128x128 default and cuBLAS — the winner is a
        # measured CUTLASS variant or torch, never the fixed default.
        choice = select_default_config("gemm", 16000, 256, 2048, torch.bfloat16, 120)
        assert choice == "torch" or (
            isinstance(choice, CutlassGemmConfig)
            and choice.compile_name != GEMM_DEFAULT.compile_name
        )

    @pytest.mark.skipif(_SM != 120, reason="heuristic rules are SM120-specific")
    @pytest.mark.parametrize(
        "op,N,K", [("gemm", n, k) for (n, k) in _FF_CONV_SHAPES] + [("gemm_activation", 2048, 256)]
    )
    @pytest.mark.parametrize("M", [16, 64, 256, 720, 2048, 16000])
    def test_actionable_configs(self, op, N, K, M):
        """Every CUTLASS config the heuristic returns must be a compiled kernel."""
        cfg = select_default_config(op, M, N, K, torch.bfloat16, 120)
        if cfg == "torch":
            return
        compiled = get_unique_compile_configs(120)
        assert cfg.compile_name in compiled, f"{cfg.compile_name} is not compiled"


class TestProductionDispatch:
    """End-to-end numerics through the non-autotuned production path."""

    @pytest.mark.parametrize("N,K", _FF_CONV_SHAPES)
    @pytest.mark.parametrize("M", [16, 64, 720, 9472])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gemm(self, N, K, M, dtype):
        """Whatever backend the rules route to must stay within the same
        low-precision error envelope as torch/cuBLAS itself (vs an fp32
        reference) — an exact match against F.linear would only test
        bit-identity with cuBLAS, which CUTLASS backends legitimately differ
        from in accumulation order."""
        torch.manual_seed(0)
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        bias = torch.randn(N, device="cuda", dtype=dtype)
        out = oasr.gemm(A, B, bias)
        ref32 = torch.addmm(bias.float(), A.float(), B.float().t())
        torch_err = (F.linear(A, B, bias).float() - ref32).abs().max().item()
        our_err = (out.float() - ref32).abs().max().item()
        floor = 1e-2 * (K / 256) ** 0.5 * (4.0 if dtype == torch.bfloat16 else 1.0)
        assert our_err <= max(4.0 * torch_err, floor), f"error {our_err} vs torch's own {torch_err}"

    @pytest.mark.parametrize("M", [16, 64, 720, 9472])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gemm_activation_swish(self, M, dtype):
        N, K = 2048, 256
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        bias = torch.randn(N, device="cuda", dtype=dtype)
        out = oasr.gemm_activation(A, B, bias, oasr.get_activation_type_id("swish"))
        torch.testing.assert_close(out, F.silu(F.linear(A, B, bias)), rtol=2e-2, atol=2e-2)


class TestGemmLogSoftmaxDispatch:
    """Routing + numerics of the shape-aware CTC-head dispatch."""

    def _mk(self, M=64, N=5008, K=256, dtype=torch.bfloat16):
        torch.manual_seed(0)
        A = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
        B = torch.randn(N, K, device="cuda", dtype=dtype) * 0.1
        C = torch.randn(N, device="cuda", dtype=dtype) * 0.1
        ref = F.log_softmax(F.linear(A.float(), B.float(), C.float()), dim=-1)
        return A, B, C, ref

    def _check(self, out, ref):
        torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)

    def test_unaligned_vocab_raises_and_names_the_fix(self):
        """``N % 8 != 0`` (unpadded vocab) raises, with the remedy in the message.

        This assertion is the *reverse* of what it used to be, so the history
        matters.  It originally required a silent reroute to cuBLAS, because the
        behaviour before that was a bare ``GEMM kernel failed`` crash — a fair
        complaint about an unhelpful error, answered by removing the error
        entirely.  That left ``gemm_log_softmax`` as the one member of the GEMM
        family with its own contract: ``oasr.gemm`` still failed on the very same
        input.

        The resolution keeps the honest half of both: the precondition is
        enforced uniformly (``CHECK_GEMM_ALIGNMENT`` in every launcher) and the
        message names the fix.  Padding an output projection is cheap and is
        what every in-tree caller already does — Conformer and Zipformer pad the
        CTC vocab in their converters (5002 → 5008, 500 → 504), Paraformer and
        the transducer pad theirs on load.  A quiet reroute would instead leave
        such a model permanently off the kernel path with nothing to notice.
        """
        A, B, C, _ref = self._mk(N=5002)
        with pytest.raises(Exception) as exc:
            oasr.gemm_log_softmax(A, B, C)
        text = str(exc.value)
        assert "8-aligned" in text and "N=5002" in text, text
        assert "align_out_features" in text, "the error must say what to do"

    def test_choice_torch(self, monkeypatch):
        import oasr.gemm_torch as gt
        import oasr.jit.gemm as jg

        calls = []
        orig = gt.torch_gemm_log_softmax
        monkeypatch.setattr(
            gt, "torch_gemm_log_softmax", lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
        )
        monkeypatch.setattr(jg, "select_default_config", lambda *a, **k: "torch")
        A, B, C, ref = self._mk()
        out = oasr.gemm_log_softmax(A, B, C)
        assert calls
        self._check(out, ref)

    def test_choice_fused(self, monkeypatch):
        import oasr.jit.gemm as jg

        monkeypatch.setattr(jg, "select_default_config", lambda *a, **k: "fused")
        A, B, C, ref = self._mk()
        self._check(oasr.gemm_log_softmax(A, B, C), ref)

    def test_choice_cutlass_composed(self, monkeypatch):
        """A rule that names a CUTLASS variant runs GEMM-variant + the OASR
        log_softmax kernel — verify that exact path executes."""
        import importlib

        import oasr.jit.gemm as jg

        og = importlib.import_module("oasr.gemm")
        cfg = next(
            c
            for c in get_unique_compile_configs(_SM).values()
            if isinstance(c, CutlassGemmConfig)
            and not getattr(c, "stream_k", False)
            and not getattr(c, "parallel_split_k", False)
        )
        monkeypatch.setattr(jg, "select_default_config", lambda *a, **k: cfg)

        calls = []
        orig = og._log_softmax_inplace
        monkeypatch.setattr(
            og, "_log_softmax_inplace", lambda out2d: (calls.append(1), orig(out2d))[1]
        )
        A, B, C, ref = self._mk()
        out = oasr.gemm_log_softmax(A, B, C)
        assert calls, "expected the composed cutlass + log_softmax path"
        self._check(out, ref)


class TestBmmDispatch:
    """Routing + numerics of the shape-aware bmm dispatch."""

    def _mk(self, batch=8, M=64, N=128, K=256, dtype=torch.bfloat16):
        torch.manual_seed(0)
        A = torch.randn(batch, M, K, device="cuda", dtype=dtype)
        B = torch.randn(batch, N, K, device="cuda", dtype=dtype)
        ref = torch.matmul(A.float(), B.float().transpose(-1, -2))
        return A, B, ref

    def test_choice_torch(self, monkeypatch):
        import oasr.gemm_torch as gt
        import oasr.jit.gemm as jg

        calls = []
        orig = gt.torch_bmm
        monkeypatch.setattr(gt, "torch_bmm", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
        monkeypatch.setattr(jg, "select_default_config", lambda *a, **k: "torch")
        A, B, ref = self._mk()
        out = oasr.bmm(A, B)
        assert calls
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-1)

    def test_choice_cutlass_variant(self, monkeypatch):
        import oasr.jit.gemm as jg

        cfg = next(
            c
            for c in get_unique_compile_configs(_SM).values()
            if isinstance(c, CutlassGemmConfig)
            and not getattr(c, "stream_k", False)
            and not getattr(c, "parallel_split_k", False)
            and c.compile_name != GEMM_DEFAULT.compile_name
        )
        monkeypatch.setattr(jg, "select_default_config", lambda *a, **k: cfg)
        A, B, ref = self._mk()
        out = oasr.bmm(A, B)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-1)

    def test_default_fallback(self):
        # No rules for this (N, K) → GEMM_DEFAULT variant, same as before.
        A, B, ref = self._mk(N=120, K=40)
        out = oasr.bmm(A, B)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-1)
