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
    CutlassGemmConfig,
    GEMM_DEFAULT,
    get_unique_compile_configs,
    select_default_config,
)

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
        torch.testing.assert_close(out, torch.bmm(A, B.transpose(-1, -2)),
                                   rtol=2e-2, atol=2e-2)


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
    def test_small_m_picks_small_tile(self):
        cfg = select_default_config("gemm", 64, 256, 256, torch.bfloat16, 120)
        assert isinstance(cfg, CutlassGemmConfig)
        assert cfg.block_m < 128  # a tall-thin tile, not the 128-row default

    @pytest.mark.skipif(_SM != 120, reason="heuristic rules are SM120-specific")
    def test_large_m_uses_large_tile(self):
        # At large offline M a full 128x128 tile wins (the sweep may pick a
        # marginally-better warp shape than GEMM_DEFAULT, but never a small tile).
        cfg = select_default_config("gemm", 16000, 256, 2048, torch.bfloat16, 120)
        assert isinstance(cfg, CutlassGemmConfig)
        assert (cfg.block_m, cfg.block_n) == (128, 128)

    @pytest.mark.skipif(_SM != 120, reason="heuristic rules are SM120-specific")
    @pytest.mark.parametrize("op,N,K", [("gemm", n, k) for (n, k) in _FF_CONV_SHAPES]
                             + [("gemm_activation", 2048, 256)])
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
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        bias = torch.randn(N, device="cuda", dtype=dtype)
        out = oasr.gemm(A, B, bias)
        torch.testing.assert_close(out, F.linear(A, B, bias), rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("M", [16, 64, 720, 9472])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gemm_activation_swish(self, M, dtype):
        N, K = 2048, 256
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        bias = torch.randn(N, device="cuda", dtype=dtype)
        out = oasr.gemm_activation(A, B, bias, oasr.get_activation_type_id("swish"))
        torch.testing.assert_close(out, F.silu(F.linear(A, B, bias)), rtol=2e-2, atol=2e-2)
