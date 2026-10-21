#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr.gemm(..., batch_invariant=True)`` and the selector behind it.

The pin's whole contract is that an output row is **bitwise** independent of M
-- of how many other rows share the launch -- so every numerical assertion
here is ``torch.equal``, never a tolerance.  A tolerance would pass the
unpinned path too, whose cuBLAS / split-K / Stream-K choices move a row by an
ulp with M; that ulp is what an int8 quantizer downstream turns into a flipped
word.

Three surfaces:

* the rows themselves, across M, through ``gemm`` and ``gemm_activation``;
* the class (``is_batch_invariant``): every member the production module
  compiles returns the same bits as every other, which is the property that
  lets the selector pick its tile from M at all;
* the selector and the dispatch: never a sentinel, a K decomposition or the
  tuner, and no silent fallback out of the class.
"""

from __future__ import annotations

import pytest
import torch
from helpers import assert_graph_replay

import oasr
import oasr.functionals.gemm as fgemm
import oasr.jit.gemm as jg
from oasr.jit.core import _get_target_sm

pytestmark = pytest.mark.cuda

_SM = _get_target_sm()

#: ``(N, K)``: Conformer's ``encoder.embed.out.0`` (the projection that made a
#: quantized model batch-dependent), an attention-width and a feed-forward
#: shape, and one whose N and K are multiples of no tile (a partial column tile
#: and a K residue, which the mainloop handles in its first iteration).
_SHAPES = [(256, 4864), (768, 256), (256, 2048), (264, 328)]
_MS = [1, 7, 16, 128, 1000, 3000]
_DTYPES = [torch.float16, torch.bfloat16]
_ACTIVATIONS = [0, 2, 4]  # relu, swish, exact-erf gelu


def _operands(M, N, K, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.randn(M, K, device="cuda", dtype=dtype, generator=g)
    # Scaled so deep-K fp16 sums stay finite and every rounding is a real one.
    B = torch.randn(N, K, device="cuda", dtype=dtype, generator=g) * K**-0.5
    C = torch.randn(N, device="cuda", dtype=dtype, generator=g)
    return A, B, C


def _first_mismatch(got, ref):
    # Equal shapes first: ``!=`` broadcasts, and a one-row reference would
    # compare every row against row 0.
    assert got.shape == ref.shape, (tuple(got.shape), tuple(ref.shape))
    rows = (got != ref).any(dim=-1).nonzero().flatten()
    return int(rows[0]) if rows.numel() else None


# ---------------------------------------------------------------------------
# Rows across M
# ---------------------------------------------------------------------------


class TestRowsAreIndependentOfM:
    """The contract: row *i* of a pinned call is the same bits at every M."""

    @pytest.mark.parametrize("dtype", _DTYPES, ids=["fp16", "bf16"])
    @pytest.mark.parametrize("N,K", _SHAPES, ids=[f"N{n}_K{k}" for n, k in _SHAPES])
    def test_gemm(self, N, K, dtype):
        A, B, C = _operands(max(_MS), N, K, dtype)
        ref = oasr.gemm(A, B, C, batch_invariant=True)
        assert torch.isfinite(ref).all()
        for M in _MS:
            got = oasr.gemm(A[:M], B, C, batch_invariant=True)
            bad = _first_mismatch(got, ref[:M])
            assert bad is None, (
                f"N={N} K={K} {dtype}: row {bad} at M={M} differs from the same row at "
                f"M={max(_MS)} (picks {self._pick(M, N, K, dtype)} vs "
                f"{self._pick(max(_MS), N, K, dtype)})"
            )

    @pytest.mark.parametrize("dtype", _DTYPES, ids=["fp16", "bf16"])
    def test_gemm_without_bias(self, dtype):
        N, K = 256, 4864
        A, B, _ = _operands(max(_MS), N, K, dtype)
        ref = oasr.gemm(A, B, None, batch_invariant=True)
        for M in _MS:
            assert torch.equal(oasr.gemm(A[:M], B, None, batch_invariant=True), ref[:M]), M

    @pytest.mark.parametrize("activation", _ACTIVATIONS)
    @pytest.mark.parametrize("dtype", _DTYPES, ids=["fp16", "bf16"])
    def test_gemm_activation(self, activation, dtype):
        N, K = 256, 4864
        A, B, C = _operands(max(_MS), N, K, dtype)
        ref = oasr.gemm_activation(A, B, C, activation, batch_invariant=True)
        for M in _MS:
            got = oasr.gemm_activation(A[:M], B, C, activation, batch_invariant=True)
            bad = _first_mismatch(got, ref[:M])
            assert bad is None, f"activation {activation} {dtype}: row {bad} differs at M={M}"

    def test_leading_dims_are_rows_too(self):
        """A ``(B, T, K)`` input is ``B*T`` rows: one utterance alone == in a batch."""
        N, K, T = 256, 4864, 375
        A, B, C = _operands(4 * T, N, K, torch.float16)
        batch = oasr.gemm(A.view(4, T, K), B, C, batch_invariant=True)
        for b in range(4):
            alone = oasr.gemm(A.view(4, T, K)[b : b + 1], B, C, batch_invariant=True)
            assert torch.equal(alone[0], batch[b]), b

    def test_matches_a_reference(self):
        """Pinned is a GEMM first: the rows it pins are the right rows."""
        N, K, M = 256, 4864, 1000
        A, B, C = _operands(M, N, K, torch.float16)
        got = oasr.gemm(A, B, C, batch_invariant=True).float()
        ref = torch.addmm(C.float(), A.float(), B.float().t())
        assert (got - ref).abs().max().item() < 2e-2

    def test_empty_input(self):
        A, B, C = _operands(1, 256, 4864, torch.float16)
        assert oasr.gemm(A[:0], B, C, batch_invariant=True).shape == (0, 256)
        assert oasr.gemm_activation(A[:0], B, C, 2, batch_invariant=True).shape == (0, 256)

    def test_survives_cuda_graph_capture_and_replay(self):
        """Captured and eager agree bit for bit (AGENTS.md rule 11)."""
        N, K, M = 256, 4864, 64
        A, B, C = _operands(M, N, K, torch.float16)
        out = torch.empty(M, N, device="cuda", dtype=torch.float16)
        fresh = _operands(M, N, K, torch.float16, seed=1)[0]
        assert_graph_replay(
            lambda: oasr.gemm(A, B, C, out=out, batch_invariant=True),
            out=out,
            mutate=lambda: A.copy_(fresh),
            expected=lambda: oasr.gemm(fresh, B, C, batch_invariant=True),
            atol=0.0,
            rtol=0.0,
        )

    @staticmethod
    def _pick(M, N, K, dtype):
        return jg.select_batch_invariant_config("gemm", M, N, K, dtype, _SM).compile_name


# ---------------------------------------------------------------------------
# The class
# ---------------------------------------------------------------------------

_CLASS = jg.batch_invariant_configs(_SM)


class TestTheClass:
    def test_this_arch_has_members(self):
        assert _CLASS, f"no batch-invariant GEMM variant is compiled for sm_{_SM}"

    @pytest.mark.parametrize("dtype", _DTYPES, ids=["fp16", "bf16"])
    @pytest.mark.parametrize("N,K", [(256, 4864), (264, 328)], ids=["embed", "residue"])
    def test_every_member_computes_the_same_bits(self, N, K, dtype):
        """Why the selector may change tile with M: any member, any M, same row.

        Run directly, not through the selector, so a member the selector does
        not reach on today's shapes is still held to the property.
        """
        A, B, C = _operands(1000, N, K, dtype)
        ref_name = _CLASS[0].compile_name
        ref = torch.empty(1000, N, device="cuda", dtype=dtype)
        fgemm._gemm_fn(ref_name, False)(ref, A, B, C, 1)
        assert torch.isfinite(ref).all()
        for cfg in _CLASS:
            fn = fgemm._gemm_fn(cfg.compile_name, False)
            for M in (1, 77, 1000):
                out = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
                fn(out, A[:M].contiguous(), B, C, 1)
                bad = _first_mismatch(out, ref[:M])
                assert (
                    bad is None
                ), f"{cfg.compile_name} at M={M} disagrees with {ref_name} on row {bad}"

    def test_members_are_production_variants_the_epilogue_can_address(self):
        """The thin-N epilogue trap (AGENTS.md): only tiles the space already admits."""
        production = jg.get_production_configs(_SM)
        for cfg in _CLASS:
            assert cfg.compile_name in production, cfg.compile_name
            tile = jg.TileShape(
                cfg.block_m, cfg.block_n, cfg.block_k, cfg.warp_m, cfg.warp_n, cfg.warp_k
            )
            assert jg._epilogue_covers_warp(tile), cfg.compile_name
            assert jg.is_batch_invariant(cfg)

    def test_every_data_parallel_production_tile_shares_the_k_tile(self):
        """The class fixes ``block_k``; today that excludes nothing, and should not."""
        dp = [
            c
            for c in jg.get_production_configs(_SM).values()
            if isinstance(c, jg.CutlassGemmConfig)
            and not (c.stream_k or c.parallel_split_k)
            and c.warp_k == c.block_k
        ]
        assert {c.compile_name for c in dp} == {c.compile_name for c in _CLASS}

    def test_predicate(self):
        dp = jg.CutlassGemmConfig(64, 64, 64, 32, 32, 64, 3, _SM)
        assert jg.is_batch_invariant(dp)
        assert not jg.is_batch_invariant("torch")
        assert not jg.is_batch_invariant("fused")
        assert not jg.is_batch_invariant(jg.CutlassGemmConfig(64, 64, 64, 32, 32, 64, 3, _SM, 4))
        assert not jg.is_batch_invariant(
            jg.CutlassGemmConfig(64, 128, 64, 32, 64, 64, 3, _SM, stream_k=True)
        )
        assert not jg.is_batch_invariant(
            jg.CutlassGemmConfig(64, 64, 64, 32, 32, 64, 3, _SM, 2, parallel_split_k=True)
        )
        # Sliced-K: warps split the K tile and reduce in the epilogue.
        assert not jg.is_batch_invariant(jg.CutlassGemmConfig(64, 64, 64, 32, 32, 32, 3, _SM))
        assert not jg.is_batch_invariant(jg.default_config_for_sm(90))


# ---------------------------------------------------------------------------
# The selector
# ---------------------------------------------------------------------------

_SWEEP_MS = sorted({*range(1, 600, 37), *(2**i for i in range(14)), 1360, 2720, 5440, 20000})


class TestSelector:
    @pytest.mark.parametrize("op", ["gemm", "gemm_activation"])
    @pytest.mark.parametrize("dtype", _DTYPES, ids=["fp16", "bf16"])
    @pytest.mark.parametrize("N,K", _SHAPES, ids=[f"N{n}_K{k}" for n, k in _SHAPES])
    def test_only_ever_returns_a_compiled_member(self, op, N, K, dtype):
        names = {c.compile_name for c in _CLASS}
        for M in _SWEEP_MS:
            choice = jg.select_batch_invariant_config(op, M, N, K, dtype, _SM)
            assert isinstance(choice, jg.CutlassGemmConfig), (M, choice)
            assert choice.split_k == 1 and not choice.stream_k and not choice.parallel_split_k
            assert choice.warp_k == choice.block_k, (M, choice.compile_name)
            assert jg.is_batch_invariant(choice)
            assert choice.compile_name in names, (M, choice.compile_name)

    def test_keeps_a_measured_choice_that_is_already_pinned(self):
        """Where the tuned table already names a member, the pin costs nothing."""
        rules = jg._GEMM_HEURISTIC_RULES.get(_SM, {})
        if not rules:
            pytest.skip(f"no tuned rule table for sm_{_SM}")
        kept = 0
        for (op, N, K), regions in rules.items():
            if op not in ("gemm", "gemm_activation"):
                continue
            for m_max, _ in regions:
                M = m_max or 20000
                tuned = jg.select_default_config(op, M, N, K, torch.bfloat16, _SM)
                pinned = jg.select_batch_invariant_config(op, M, N, K, torch.bfloat16, _SM)
                assert jg.is_batch_invariant(pinned)
                if tuned is not jg.GEMM_DEFAULT and jg.is_batch_invariant(tuned):
                    assert pinned == tuned, ((op, M, N, K), tuned, pinned)
                    kept += 1
        assert kept, "no tuned region names a batch-invariant config; the test proves nothing"

    def test_is_a_function_of_shape_and_device(self):
        a = jg.select_batch_invariant_config("gemm", 777, 256, 4864, torch.float16, _SM)
        b = jg.select_batch_invariant_config("gemm", 777, 256, 4864, torch.float16, _SM)
        assert a == b
        # The device enters only through its SM count.
        narrow = jg.select_batch_invariant_config(
            "gemm", 1024, 256, 4864, torch.float16, _SM, num_sms=16
        )
        assert jg.is_batch_invariant(narrow)

    def test_occupancy_rule(self):
        """One wave: the smallest warp tile; past it, the least operand traffic."""
        pick = jg._pinned_occupancy_pick
        small = pick(_CLASS, 64, 256, 170)
        assert small.warp_m * small.warp_n == min(c.warp_m * c.warp_n for c in _CLASS)
        huge = pick(_CLASS, 1 << 20, 256, 170)
        assert huge.block_m * huge.block_n == max(c.block_m * c.block_n for c in _CLASS)

    def test_refuses_an_arch_with_no_member(self, monkeypatch):
        monkeypatch.setattr(jg, "batch_invariant_configs", lambda sm: ())
        with pytest.raises(RuntimeError, match="no batch-invariant GEMM variant"):
            jg.select_batch_invariant_config("gemm", 64, 256, 4864, torch.float16, _SM)

    def test_refuses_an_op_it_does_not_pin(self):
        with pytest.raises(ValueError, match="only 'gemm' and 'gemm_activation'"):
            jg.select_batch_invariant_config("bmm", 64, 256, 4864, torch.float16, _SM)


# ---------------------------------------------------------------------------
# The dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_never_cublas(self, monkeypatch):
        """M=64 on the embed shape is a cuBLAS rule unpinned; pinned, it is CUTLASS."""
        import oasr.functionals.gemm_torch as gt

        def refuse(*a, **k):
            raise AssertionError("a pinned GEMM reached cuBLAS")

        monkeypatch.setattr(gt, "torch_gemm", refuse)
        monkeypatch.setattr(gt, "torch_gemm_activation", refuse)
        A, B, C = _operands(64, 256, 4864, torch.float16)
        oasr.gemm(A, B, C, batch_invariant=True)
        oasr.gemm_activation(A, B, C, 2, batch_invariant=True)
        _, _, kind, fn = fgemm._plan_pinned("gemm", False, 64, 256, 4864, torch.float16)
        choice = jg.select_batch_invariant_config("gemm", 64, 256, 4864, torch.float16, _SM)
        assert kind == fgemm._PLAN_CUTLASS
        assert fn is fgemm._gemm_fn(choice.compile_name, False)

    def test_never_the_tuner(self, monkeypatch):
        import oasr.tune

        def refuse(*a, **k):
            raise AssertionError("a pinned GEMM consulted the autotuner")

        monkeypatch.setattr(fgemm, "is_tuning_enabled", lambda: True)
        monkeypatch.setattr(oasr.tune, "get_tuner", refuse)
        A, B, C = _operands(128, 256, 4864, torch.float16)
        oasr.gemm(A, B, C, batch_invariant=True)
        oasr.gemm_activation(A, B, C, 2, batch_invariant=True)

    def test_a_resolver_outside_the_class_is_not_launched(self, monkeypatch):
        """A non-member falls back only to a default that is itself a member."""
        sk = jg.CutlassGemmConfig(64, 128, 64, 32, 64, 64, 3, _SM, stream_k=True)
        monkeypatch.setattr(jg, "select_batch_invariant_config", lambda *a, **k: sk)
        plan = fgemm._plan_pinned("gemm", False, 4096, 256, 4864, torch.float16)
        if jg.is_batch_invariant(jg.GEMM_DEFAULT):
            assert plan[2] == fgemm._PLAN_DEFAULT
        # And where the default is not a member either, it raises.
        monkeypatch.setattr(jg, "GEMM_DEFAULT", sk)
        with pytest.raises(RuntimeError, match="not batch-invariant"):
            fgemm._plan_pinned("gemm", False, 4095, 256, 4864, torch.float16)

    def test_positional_behaviour_is_unchanged(self):
        A, B, C = _operands(256, 256, 2048, torch.float16)
        out = torch.empty(256, 256, device="cuda", dtype=torch.float16)
        assert oasr.gemm(A, B, C, out) is out
        out2 = torch.empty_like(out)
        assert oasr.gemm_activation(A, B, C, 2, out2) is out2
