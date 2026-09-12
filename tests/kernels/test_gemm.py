#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Everything ``oasr/functionals/gemm.py`` exposes, in one file.

Four sections, because they are four surfaces of the same launcher and a
change to one routinely breaks another:

* the plain GEMM / group-GEMM / BMM lanes and the 4-D permuted layouts
  Zipformer feeds them;
* the input-layout and 8-alignment contracts;
* the fused ``log_softmax`` epilogue -- the CTC head;
* the split-K decompositions and the persistent workspace cache.

Kernel *selection* (the SM rule table, ``select_default_config``, the torch
fallback) lives next door in ``test_gemm_heuristic.py``: this file is about
what the kernels compute, that one about which kernel gets picked.
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_dest_passing, assert_graph_replay

import oasr
from oasr.functionals.gemm import _bmm_fn, _gemm_fn, _get_gemm_module
from oasr.jit.core import _get_target_sm
from oasr.jit.gemm import CutlassGemmConfig, get_unique_compile_configs

pytestmark = pytest.mark.cuda

_SM = _get_target_sm()


class TestGemm:
    """Tests for oasr.gemm() functional API."""

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (64, 128, 256),
            (32, 32, 32),
            (256, 32, 128),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gemm(self, M, N, K, dtype):
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)

        D = oasr.gemm(A, B)

        expected = torch.matmul(A, B.T)
        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gemm_destination_passing(self, dtype):
        """Test GEMM with pre-allocated output."""
        M, N, K = 64, 128, 256
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        out = torch.empty(M, N, device="cuda", dtype=dtype)

        result = oasr.gemm(A, B, out=out)

        assert result.data_ptr() == out.data_ptr()
        expected = torch.matmul(A, B.T)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


class TestGroupGemm:
    """Tests for oasr.group_gemm() functional API.

    The grouped path had no functional coverage at all, which is how its SM90
    kernel shipped never having compiled: ``offset`` is the cumulative *end* row
    of each group, so group ``i`` is ``A[offset[i-1]:offset[i]] @ B[i].T``.
    """

    @staticmethod
    def _reference(A, B, rows):
        start = 0
        parts = []
        for group, count in enumerate(rows):
            parts.append(A[start : start + count] @ B[group].transpose(0, 1))
            start += count
        return torch.cat(parts, dim=0)

    @pytest.mark.parametrize("rows", [[32, 16, 48], [64, 64], [8]])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_group_gemm(self, rows, dtype):
        torch.manual_seed(0)
        N, K = 128, 64
        A = torch.randn(sum(rows), K, device="cuda", dtype=dtype)
        B = torch.randn(len(rows), N, K, device="cuda", dtype=dtype)
        offset = torch.tensor(list(itertools.accumulate(rows)), device="cuda", dtype=torch.int32)

        D = oasr.group_gemm(A, B, offset)

        torch.testing.assert_close(D, self._reference(A, B, rows), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_group_gemm_destination_passing(self, dtype):
        torch.manual_seed(1)
        rows, N, K = [32, 32], 128, 64
        A = torch.randn(sum(rows), K, device="cuda", dtype=dtype)
        B = torch.randn(len(rows), N, K, device="cuda", dtype=dtype)
        offset = torch.tensor(list(itertools.accumulate(rows)), device="cuda", dtype=torch.int32)
        out = torch.empty(sum(rows), N, device="cuda", dtype=dtype)

        result = oasr.group_gemm(A, B, offset, out=out)

        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(result, self._reference(A, B, rows), rtol=1e-2, atol=1e-2)


class TestBmm:
    """Tests for oasr.bmm() functional API."""

    @pytest.mark.parametrize(
        "batch_size,M,N,K",
        [
            (4, 64, 128, 256),
            (3, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_bmm(self, batch_size, M, N, K, dtype):
        A = torch.randn(batch_size, M, K, device="cuda", dtype=dtype)
        B = torch.randn(batch_size, N, K, device="cuda", dtype=dtype)

        D = oasr.bmm(A, B)

        expected = torch.bmm(A, B.permute(0, 2, 1))
        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_bmm_destination_passing(self, dtype):
        """Test BMM with pre-allocated output."""
        batch_size, M, N, K = 4, 64, 128, 256
        A = torch.randn(batch_size, M, K, device="cuda", dtype=dtype)
        B = torch.randn(batch_size, N, K, device="cuda", dtype=dtype)
        out = torch.empty(batch_size, M, N, device="cuda", dtype=dtype)

        result = oasr.bmm(A, B, out=out)

        assert result.data_ptr() == out.data_ptr()
        expected = torch.bmm(A, B.permute(0, 2, 1))
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestBmmGeneralLane:
    """The shapes the tuned alignment-8 tile variants refuse (KG5).

    Every case here is a Zipformer attention product with its real strides: the
    operands are 4-D permuted views of a ``(time, batch, head, dim)`` activation,
    one of them broadcasts over the request batch, and N or K is small, odd, or
    both.  These used to be ``torch.matmul`` in the model file, which is why the
    ratchet in ``test_layer_waist.py`` exists alongside them.
    """

    @staticmethod
    def _check(got, A, B, expected_shape):
        assert tuple(got.shape) == expected_shape, (tuple(got.shape), expected_shape)
        # fp32 reference: at K = 496 an fp16 reference accumulates its own error.
        expected = torch.matmul(A.float(), B.float().transpose(-1, -2))
        torch.testing.assert_close(got.float(), expected, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "T,heads,batch",
        [(496, 4, 1), (124, 4, 3), (62, 8, 1)],
        ids=["stack0-b1", "stack2-b3", "stack3-b1"],
    )
    def test_score_product_permuted_4d(self, T, heads, batch, dtype):
        """``q @ k``: K = 32 aligned, N = T arbitrary, both operands permuted."""
        x = torch.randn(T, batch, heads, 96, device="cuda", dtype=dtype)
        q = x[..., 0:32].permute(2, 1, 0, 3)
        k = x[..., 32:64].permute(2, 1, 0, 3)
        assert not q.is_contiguous() and not k.is_contiguous()
        self._check(oasr.bmm(q, k), q, k, (heads, batch, T, T))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("T,heads,batch", [(496, 4, 1), (124, 4, 3)])
    def test_relative_position_product_broadcast_batch(self, T, heads, batch, dtype):
        """``p @ pos_emb``: K = 4, N = 2T-1 (always odd), B shared over the batch.

        The broadcast axis is the one no single batch stride can express, so this
        is also the case that exercises the launcher's per-outer-index loop.
        """
        p = torch.randn(T, batch, heads, 4, device="cuda", dtype=dtype).permute(2, 1, 0, 3)
        pos = torch.randn(1, 2 * T - 1, heads, 4, device="cuda", dtype=dtype).permute(2, 0, 1, 3)
        self._check(oasr.bmm(p, pos), p, pos, (heads, batch, T, 2 * T - 1))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("T,heads,batch", [(496, 4, 1), (62, 8, 3)])
    def test_value_product_row_major_b(self, T, heads, batch, dtype):
        """``probs @ v``: N = 12, K = T, and B contiguous along N, not K.

        This is the layout that would otherwise cost a ``.contiguous()`` — one
        extra copy *and* one extra launch — at every value product.
        """
        w = torch.randn(heads, batch, T, T, device="cuda", dtype=dtype)
        v = torch.randn(T, batch, heads, 12, device="cuda", dtype=dtype).permute(2, 1, 3, 0)
        assert v.stride(-1) != 1 and v.stride(-2) == 1
        self._check(oasr.bmm(w, v), w, v, (heads, batch, T, 12))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("value_dim", [4, 8, 12, 16])
    def test_value_product_every_addressable_n(self, value_dim, dtype):
        """The same product across the value dims the thin-N tile serves.

        ``N <= 16`` with B row-major selects ``ThinNTile`` (64x16), whose
        epilogue folds a warp into a ``kAccessRows x kAccessWidth`` grid derived
        from the tile *and* the runtime N alignment.  At ``kElementsPerAccess =
        8`` that grid is 8x2 = 16 lanes and the kernel returns wrong numbers at
        full speed — so ``value_dim`` 8 and 16 were wrong while the shipped 12
        (alignment 4, grid 8x4) was right, and no test asked about any dim but
        12.  ``epilogueAlignment`` in ``include/oasr/gemm/bmm.cuh`` caps the
        store; the ``cp.async`` load keeps its 8 elements.
        """
        T, heads, batch = 200, 4, 1
        w = torch.randn(heads, batch, T, T, device="cuda", dtype=dtype)
        v = torch.randn(T, batch, heads, value_dim, device="cuda", dtype=dtype).permute(2, 1, 3, 0)
        assert v.stride(-2) == 1, "must be the row-major-B layout that picks ThinNTile"
        self._check(oasr.bmm(w, v), w, v, (heads, batch, T, value_dim))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_value_product_single_head(self, dtype):
        """NonlinAttention shares one head's weights across a wide value dim."""
        T, hidden, batch = 248, 192, 2
        w = torch.randn(1, batch, T, T, device="cuda", dtype=dtype)
        v = torch.randn(T, batch, 1, hidden, device="cuda", dtype=dtype).permute(2, 1, 3, 0)
        self._check(oasr.bmm(w, v), w, v, (1, batch, T, hidden))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("M,N,K", [(7, 11, 5), (33, 1, 16), (33, 17, 1), (9, 17, 4)])
    def test_unaligned_3d(self, M, N, K, dtype):
        """The historical ``[batch, N, K]`` contract, now with arbitrary N and K.

        ``K = 5`` is odd, so it has no tensor-op instantiation at all and proves
        the SIMT lane is reachable rather than a dead branch.
        """
        A = torch.randn(2, M, K, device="cuda", dtype=dtype)
        B = torch.randn(2, N, K, device="cuda", dtype=dtype)
        self._check(oasr.bmm(A, B), A, B, (2, M, N))

    @pytest.mark.parametrize("a_batch,b_batch", [(4, 1), (1, 4), (1, 1)])
    def test_broadcast_within_the_3d_fast_path(self, a_batch, b_batch):
        """3-D broadcasting has its own branch, because 3-D has its own fast path.

        The pre-KG5 shape (two 3-D operands) bypasses the general broadcast
        machinery to keep the entry point cheap — measured, that routing was
        worth 2.0 us a call — so its broadcast rule is separate code and needs
        its own coverage.
        """
        A = torch.randn(a_batch, 9, 16, device="cuda", dtype=torch.float16)
        B = torch.randn(b_batch, 5, 16, device="cuda", dtype=torch.float16)
        self._check(oasr.bmm(A, B), A, B, (max(a_batch, b_batch), 9, 5))

    def test_non_broadcastable_3d_batch_is_refused(self):
        A = torch.randn(4, 9, 16, device="cuda", dtype=torch.float16)
        B = torch.randn(3, 5, 16, device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match="not broadcastable"):
            oasr.bmm(A, B)

    def test_broadcast_a_3d_operand_against_a_4d_one(self):
        A = torch.randn(5, 13, 24, device="cuda", dtype=torch.float16)
        B = torch.randn(2, 5, 9, 24, device="cuda", dtype=torch.float16)
        self._check(oasr.bmm(A, B), A, B, (2, 5, 13, 9))

    def test_aligned_contiguous_keeps_the_tuned_lane(self, monkeypatch):
        """KG5 must not move the established alignment-8 lane onto the general one.

        The tuned lane carries the shape heuristic and its measured tile choices;
        a contiguous 3-D alignment-8 call reaching the general dispatcher would
        be a silent performance regression, not a wrong answer, so nothing else
        would catch it.
        """
        import oasr.functionals.gemm as gemm_module

        monkeypatch.setattr(
            gemm_module,
            "_general_bmm_fn",
            lambda: pytest.fail("aligned contiguous BMM reached the general lane"),
        )
        A = torch.randn(4, 32, 64, device="cuda", dtype=torch.float16)
        B = torch.randn(4, 24, 64, device="cuda", dtype=torch.float16)
        self._check(oasr.bmm(A, B), A, B, (4, 32, 24))

    def test_b_contiguous_along_neither_axis_is_refused(self):
        """A layout with no CUTLASS expression must say so, not be reinterpreted."""
        base = torch.randn(2, 8, 16, 4, device="cuda", dtype=torch.float16)
        A = torch.randn(2, 6, 16, device="cuda", dtype=torch.float16)
        B = base[..., 0]  # (2, 8, 16) -- strides (512, 64, 4): neither axis is 1
        assert B.stride(-1) != 1 and B.stride(-2) != 1
        with pytest.raises(Exception, match="contiguous along one of its two trailing axes"):
            oasr.bmm(A, B)

    def test_survives_cuda_graph_capture_and_replay(self):
        """The streaming encoder is graph-captured, so this lane has to be.

        Two ways it could fail and neither shows up in an eager test: an
        allocation inside the launcher (``GemmBatched`` needs no workspace, which
        is why there is none), and the per-outer-index host loop, whose trip
        count must come from shapes rather than from anything that changes
        between capture and replay.  The broadcast case is the one that loops.
        """
        T, heads, batch = 124, 4, 2
        p = torch.randn(T, batch, heads, 4, device="cuda", dtype=torch.float16).permute(2, 1, 0, 3)
        pos = torch.randn(1, 2 * T - 1, heads, 4, device="cuda", dtype=torch.float16)
        pos_v = pos.permute(2, 0, 1, 3)
        out = torch.empty(heads, batch, T, 2 * T - 1, device="cuda", dtype=torch.float16)

        def _mutate():
            # New operand values through the captured buffers: a replay must
            # recompute, not reproduce the captured result.
            p.copy_(torch.randn_like(p))
            pos.copy_(torch.randn_like(pos))

        assert_graph_replay(
            lambda: oasr.bmm(p, pos_v, out=out),
            out=out,
            mutate=_mutate,
            expected=lambda: torch.matmul(p.float(), pos_v.float().transpose(-1, -2)).to(out.dtype),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_out_shape_mismatch_is_refused(self):
        A = torch.randn(2, 7, 5, device="cuda", dtype=torch.float16)
        B = torch.randn(2, 11, 5, device="cuda", dtype=torch.float16)
        out = torch.empty(2, 7, 12, device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match="expected"):
            oasr.bmm(A, B, out=out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestGemmInputLayout:
    """The launchers flatten N-D inputs themselves and require contiguity.

    Both properties are load-bearing.  Flattening in C++ (``FLATTENED_ROWS``)
    is what removed the two per-call ``reshape(-1, K)`` calls that made
    ``oasr.gemm`` lose to ``F.linear`` on small shapes.  The contiguity check is
    a correctness fix that came with it.
    """

    @staticmethod
    def _weight(N, K):
        return torch.randn(N, K, device="cuda", dtype=torch.float16)

    @pytest.mark.parametrize("shape", [(64,), (8, 64), (4, 6, 64), (2, 3, 5, 64)])
    def test_leading_dims_are_flattened(self, shape):
        """1-D through 4-D activations all work; the launcher computes M."""
        A = torch.randn(*shape, device="cuda", dtype=torch.float16)
        B = self._weight(32, 64)
        got = oasr.gemm(A, B)
        assert tuple(got.shape) == tuple(shape[:-1]) + (32,)
        torch.testing.assert_close(got, torch.nn.functional.linear(A, B), rtol=2e-2, atol=2e-2)

    def test_row_strided_2d_input_is_correct(self):
        """Regression: ``x[:, -1]`` of a ``(B, T, D)`` tensor.

        ``reshape(-1, K)`` is a *no-op* on an already-2-D tensor, so it never
        delivered the row-major layout the kernel assumes — the launcher only
        checked that the tensor was on CUDA, and the kernel indexes rows as
        ``A + row * K``.  This shape used to come back with a max error of ~30
        against ``F.linear``, silently.  The N-D path was accidentally safe
        because flattening a strided N-D tensor cannot be a view.
        """
        x = torch.randn(4, 6, 64, device="cuda", dtype=torch.float16)
        A = x[:, -1]
        assert not A.is_contiguous(), "the test input must actually be strided"
        B = self._weight(64, 64)
        torch.testing.assert_close(
            oasr.gemm(A, B), torch.nn.functional.linear(A, B), rtol=2e-2, atol=2e-2
        )

    def test_noncontiguous_out_is_rejected(self):
        """A strided ``out`` would be written at the wrong rows; say so."""
        A = torch.randn(8, 64, device="cuda", dtype=torch.float16)
        B = self._weight(64, 64)
        bad = torch.empty(8, 2, 64, device="cuda", dtype=torch.float16)[:, 0]
        with pytest.raises(Exception, match="[Cc]ontiguous"):
            oasr.gemm(A, B, None, out=bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestGemmAlignmentContract:
    """Projection GEMMs reject unaligned shapes; BMM has a declared general lane.

    The projections used to disagree with each other.  ``gemm`` let CUTLASS fail
    and surfaced "GEMM kernel failed", which tells the caller nothing actionable;
    ``gemm_log_softmax`` silently rerouted the same input to cuBLAS, which tells
    them nothing at all and quietly leaves the model off the kernel path forever.
    They keep one precondition and one actionable answer, because every unaligned
    projection in this repo is an output head and padding it at the model layer
    is the established fix (``align_out_features`` / ``pad_output_projection``).

    BMM is the one member that cannot take that answer: Zipformer's attention
    needs arbitrary N and K on a *contraction* it does not own, so KG5 gave it a
    general lane instead of an error message.
    """

    MSG = "8-aligned"

    @staticmethod
    def _ab(M, N, K, dtype=torch.float16):
        return (
            torch.randn(M, K, device="cuda", dtype=dtype),
            torch.randn(N, K, device="cuda", dtype=dtype),
        )

    @pytest.mark.parametrize("N,K", [(500, 64), (64, 60), (30, 24)])
    def test_gemm_rejects_unaligned(self, N, K):
        A, B = self._ab(16, N, K)
        with pytest.raises(Exception, match=self.MSG):
            oasr.gemm(A, B)

    @pytest.mark.parametrize("N,K", [(500, 64), (64, 60)])
    def test_gemm_activation_rejects_unaligned(self, N, K):
        A, B = self._ab(16, N, K)
        with pytest.raises(Exception, match=self.MSG):
            oasr.gemm_activation(A, B, None, oasr.ACTIVATION_RELU)

    @pytest.mark.parametrize("N,K", [(500, 64), (64, 60)])
    def test_gemm_log_softmax_rejects_unaligned(self, N, K):
        """Regression: this one used to succeed via a silent cuBLAS reroute."""
        A, B = self._ab(16, N, K)
        with pytest.raises(Exception, match=self.MSG):
            oasr.gemm_log_softmax(A, B)

    def test_bmm_accepts_unaligned(self):
        """The one deliberate exception to the family's alignment contract."""
        A = torch.randn(2, 16, 60, device="cuda", dtype=torch.float16)
        B = torch.randn(2, 64, 60, device="cuda", dtype=torch.float16)
        torch.testing.assert_close(
            oasr.bmm(A, B).float(),
            torch.matmul(A.float(), B.float().transpose(-1, -2)),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_the_message_names_the_fix(self):
        """An error a caller cannot act on is barely better than a silent one."""
        A, B = self._ab(16, 500, 64)
        with pytest.raises(Exception) as exc:
            oasr.gemm(A, B)
        text = str(exc.value)
        assert "N=500" in text and "K=64" in text, text
        assert "align_out_features" in text, text

    @pytest.mark.parametrize("N,K", [(504, 64), (64, 64), (8, 8)])
    def test_aligned_shapes_still_work(self, N, K):
        A, B = self._ab(16, N, K)
        torch.testing.assert_close(
            oasr.gemm(A, B), torch.nn.functional.linear(A, B), rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Fused log-softmax epilogue -- the CTC head
# ---------------------------------------------------------------------------


def _log_softmax_reference(
    A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """Reference: F.log_softmax(F.linear(A, B, bias), dim=-1) in fp32."""
    out = torch.matmul(A.float(), B.float().T)
    if bias is not None:
        out = out + bias.float()
    return F.log_softmax(out, dim=-1).to(A.dtype)


class TestGemmLogSoftmax:
    """The fused CTC head: GEMM with a log-softmax epilogue."""

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (32, 1024, 128),  # single-tile row
            (16, 2048, 256),  # vocab > tile_N, so multi-tile rows
            (4, 40, 64),  # small N -- CUTLASS still requires 8-alignment
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_torch(self, M, N, K, dtype):
        torch.manual_seed(0)
        A = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
        B = torch.randn(N, K, device="cuda", dtype=dtype) * 0.1
        bias = torch.randn(N, device="cuda", dtype=dtype) * 0.1

        out = oasr.gemm_log_softmax(A, B, bias)
        expected = _log_softmax_reference(A, B, bias)

        # log_softmax has tighter requirements than softmax; allow slightly looser
        # tolerance for FP16/BF16 accumulation.
        rtol = 5e-2 if dtype == torch.bfloat16 else 5e-3
        atol = 5e-2 if dtype == torch.bfloat16 else 1e-2
        torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_no_bias(self, dtype):
        torch.manual_seed(1)
        A = torch.randn(16, 128, device="cuda", dtype=dtype) * 0.1
        B = torch.randn(256, 128, device="cuda", dtype=dtype) * 0.1

        out = oasr.gemm_log_softmax(A, B, None)
        expected = _log_softmax_reference(A, B, None)

        torch.testing.assert_close(out, expected, rtol=5e-3, atol=1e-2)

    def test_3d_leading_dims(self):
        """A may have arbitrary leading dims (B, T, K) — same as oasr.gemm."""
        torch.manual_seed(2)
        B_dim, T, K, V = 2, 50, 256, 1024
        A = torch.randn(B_dim, T, K, device="cuda", dtype=torch.float16) * 0.1
        W = torch.randn(V, K, device="cuda", dtype=torch.float16) * 0.1
        bias = torch.randn(V, device="cuda", dtype=torch.float16) * 0.1

        out = oasr.gemm_log_softmax(A, W, bias)
        assert out.shape == (B_dim, T, V)

        expected = F.log_softmax(F.linear(A.float(), W.float(), bias.float()), dim=-1).half()
        torch.testing.assert_close(out, expected, rtol=5e-3, atol=1e-2)

    def test_destination_passing(self):
        torch.manual_seed(4)
        A = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        B = torch.randn(512, 128, device="cuda", dtype=torch.float16)
        out = torch.empty(32, 512, device="cuda", dtype=torch.float16)

        assert_dest_passing(oasr.gemm_log_softmax, A, B, None, out=out)
        torch.testing.assert_close(out, _log_softmax_reference(A, B, None), rtol=5e-3, atol=1e-2)


# ---------------------------------------------------------------------------
# Split-K decompositions and the persistent workspace cache
# ---------------------------------------------------------------------------


def _tol(dtype, K, split_k=1, serial=False):
    """Accumulation-order tolerance: bf16/f16 outputs at deep K; serial split-K
    round-trips partials through the output dtype, one rounding per slice."""
    base = 4e-2 if dtype == torch.float16 else 2.5e-1
    scale = (K / 256) ** 0.5
    if serial:
        scale *= max(1, split_k) ** 0.5
    return base * scale


def _find_cfg(*, parallel_split_k=False, block_m=None):
    for cfg in get_unique_compile_configs(_SM).values():
        if not isinstance(cfg, CutlassGemmConfig):
            continue
        if getattr(cfg, "stream_k", False):
            continue
        if bool(getattr(cfg, "parallel_split_k", False)) != parallel_split_k:
            continue
        if block_m is not None and cfg.block_m != block_m:
            continue
        return cfg
    return None


_SERIAL_CFG = _find_cfg(parallel_split_k=False, block_m=16)
_PK_CFG = _find_cfg(parallel_split_k=True, block_m=16)


@pytest.mark.skipif(_SERIAL_CFG is None, reason="no serial CUTLASS 2.x config for this arch")
class TestSerialSplitK:
    @pytest.mark.parametrize("split_k", [2, 16])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_numerics_repeated(self, dtype, split_k):
        """Deep serial split-K stays correct over REPEATED launches — this is
        what proves the pre-zeroed cached semaphores are restored each run."""
        torch.manual_seed(0)
        M, N, K = 64, 256, 2048
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        C = torch.randn(N, device="cuda", dtype=dtype)
        out = torch.empty(M, N, device="cuda", dtype=dtype)
        ref = torch.addmm(C.float(), A.float(), B.float().t())

        fn = getattr(_get_gemm_module(), f"gemm_{_SERIAL_CFG.compile_name}")
        for _ in range(5):
            out.zero_()
            fn(out, A, B, C, split_k)
            torch.cuda.synchronize()
            err = (out.float() - ref).abs().max().item()
            assert err < _tol(dtype, K, split_k, serial=True), f"split_k={split_k} err={err}"

    def test_activation_rejected(self):
        """Serial split-K + fused activation must fail loudly, not silently
        produce nested-activation garbage."""
        M, N, K = 64, 2048, 256
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        C = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        fn = getattr(_get_gemm_module(), f"gemm_{_SERIAL_CFG.compile_name}_activation")
        with pytest.raises(RuntimeError):
            fn(out, A, B, C, 2, 4)

    def test_cuda_graph_capture_replay(self):
        """Warm the workspace cache, capture the split-K launch in a CUDA
        graph, replay several times with mutated inputs — every replay must be
        correct (the captured kernel must restore the semaphores itself)."""
        dtype = torch.bfloat16
        M, N, K = 64, 256, 2048
        split_k = 8
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        C = torch.randn(N, device="cuda", dtype=dtype)
        out = torch.empty(M, N, device="cuda", dtype=dtype)
        fn = getattr(_get_gemm_module(), f"gemm_{_SERIAL_CFG.compile_name}")

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fn(out, A, B, C, split_k)  # warm-up: allocates + zeroes the cached workspace
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn(out, A, B, C, split_k)

        for i in range(3):
            A.normal_(generator=None)
            graph.replay()
            torch.cuda.synchronize()
            ref = torch.addmm(C.float(), A.float(), B.float().t())
            err = (out.float() - ref).abs().max().item()
            assert err < _tol(dtype, K, split_k, serial=True), f"replay {i}: err={err}"


@pytest.mark.skipif(_PK_CFG is None, reason="parallel split-K configs not built for this arch")
class TestParallelSplitK:
    @pytest.mark.parametrize("split_k", [2, 16])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_numerics(self, dtype, split_k):
        torch.manual_seed(0)
        M, N, K = 64, 256, 4864
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        C = torch.randn(N, device="cuda", dtype=dtype)
        out = torch.empty(M, N, device="cuda", dtype=dtype)
        ref = torch.addmm(C.float(), A.float(), B.float().t())

        fn = getattr(_get_gemm_module(), f"gemm_{_PK_CFG.compile_name}")
        fn(out, A, B, C, split_k)
        torch.cuda.synchronize()
        err = (out.float() - ref).abs().max().item()
        # fp32 partials: error must NOT grow with split depth
        assert err < _tol(dtype, K), f"split_k={split_k} err={err}"

    def test_no_bias(self):
        M, N, K = 32, 256, 2048
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        fn = getattr(_get_gemm_module(), f"gemm_{_PK_CFG.compile_name}")
        fn(out, A, B, None, 8)
        torch.cuda.synchronize()
        ref = A.float() @ B.float().t()
        assert (out.float() - ref).abs().max().item() < _tol(torch.bfloat16, K)

    @pytest.mark.parametrize("act,ref_fn", [(0, F.relu), (2, F.silu), (4, F.gelu)])
    def test_activation_applied_once(self, act, ref_fn):
        """The activation epilogue runs in the reduction kernel — exactly once
        over the FULL sum (the property serial split-K cannot provide)."""
        torch.manual_seed(1)
        M, N, K = 64, 128, 2048
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        C = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        cfg = _find_cfg(parallel_split_k=True, block_m=16)
        fn = getattr(_get_gemm_module(), f"gemm_{cfg.compile_name}_activation")
        fn(out, A, B, C, act, 8)
        torch.cuda.synchronize()
        ref = ref_fn(torch.addmm(C.float(), A.float(), B.float().t()))
        err = (out.float() - ref).abs().max().item()
        assert err < _tol(torch.bfloat16, K), f"err={err}"

    def test_slices_one_rejected(self):
        """parallel split-K variants require split_k_slices > 1."""
        M, N, K = 32, 256, 2048
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        fn = getattr(_get_gemm_module(), f"gemm_{_PK_CFG.compile_name}")
        with pytest.raises(RuntimeError):
            fn(out, A, B, None, 1)


def _split_k_variant(kind: str):
    """A registered variant of one kind, or ``None``.

    ``pk`` deliberately for the size tests: a parallel split-K workspace is
    ``M*N*4*split`` bytes of fp32 partials, where a *serial* split-K semaphore is
    one int per output tile.  With the small one the bound is untestable — 32
    keys of 4 KiB is 128 KiB whether it is enforced or not.
    """
    for c in get_unique_compile_configs(_SM).values():
        if kind == "pk" and getattr(c, "parallel_split_k", False):
            return c
        if kind == "sk" and getattr(c, "stream_k", False):
            return c
        if kind == "any" and (getattr(c, "split_k", 1) > 1 or getattr(c, "stream_k", False)):
            return c
    return None


def _run_on_streams(fn, split_k, M, N, K, device, n_streams):
    """Run one GEMM per fresh ``torch.cuda.Stream()``; return the relative error.

    ``torch.cuda.Stream()`` does not create a stream — it takes the next of a
    POOL of 32 per device and cycles.  That is the whole reason the first version
    of this test was blind: the cache is keyed on the stream handle, so a caller
    like this saturates at 32 keys after the first 32 iterations, and a test that
    warmed up and then measured a *second* burst measured zero growth whether the
    cache was bounded or not.  What grows is not the key count but the bytes
    behind each key.
    """
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    C = torch.randn(N, device=device, dtype=torch.bfloat16)
    out = torch.empty(M, N, device=device, dtype=torch.bfloat16)
    ref = A.float() @ B.float().t() + C.float()
    for _ in range(n_streams):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(out, A, B, C, split_k)
        torch.cuda.current_stream().wait_stream(s)
        del s
    torch.cuda.synchronize()
    return (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


@pytest.mark.slow
class TestWorkspaceCacheBytesBound:
    """The split-K / Stream-K workspace cache must bound the BYTES it holds.

    ``include/oasr/common/workspace_cache.h`` keeps one grow-only buffer per
    ``(device, stream, pool)`` and never frees — a retired buffer's address may
    be baked into a captured graph.  So every byte it hands out is held until the
    process exits, which makes it a cache that has to be bounded in bytes rather
    than an allocator.

    It was not.  A parallel split-K workspace is ``M*N*4*split`` bytes, so a
    single 4096x5008 shape is 328 MiB per key; run across the 32-handle stream
    pool that is 10,016 MiB held forever, measured, with the next ladder step
    failing the kernel outright.  ``scripts/tune_asr_gemm.py`` hit exactly this
    over a 121-shape sweep and died on a 66 MiB allocation while PyTorch's own
    allocator held 1 GiB.

    Serving never sees it — every architecture in the tree asks for 152-296 bytes
    of semaphore and at most 1 MiB of scratch — which is exactly why it needs a
    test rather than a comment.

    These assert on ``ws_cache_bytes()`` rather than on ``cudaMemGetInfo``,
    because ``cudaMallocAsync`` recycles: "cached once" and "allocated and freed
    every call" both read as flat free memory, and a bound whose test cannot see
    it is how the unbounded version shipped.
    """

    @pytest.mark.cuda
    def test_a_large_workspace_is_not_cached(self, device):
        cfg = _split_k_variant("pk") or _split_k_variant("any")
        if cfg is None:
            pytest.skip("no split-K / Stream-K variant registered for this arch")
        mod = _get_gemm_module()
        fn = _gemm_fn(cfg.compile_name, False)
        split_k = max(2, int(getattr(cfg, "split_k", 1)))
        # 4096 x 5008 fp32 partials x split -> >= 156 MiB, far over the per-key
        # ceiling.  96 iterations to walk the whole 32-handle stream pool three
        # times over.
        before = mod.ws_cache_bytes()
        err = _run_on_streams(fn, split_k, 4096, 5008, 256, device, 96)
        grew = (mod.ws_cache_bytes() - before) / 2**20
        assert grew < 64, (
            f"a 156+ MiB workspace across 96 stream-per-call GEMMs added {grew:.1f} MiB "
            "to a cache that never frees — the per-key size ceiling is not holding"
        )
        # And it still computed the right answer on the per-call path.
        assert err < 1e-2, f"declining to cache changed the result (rel err {err:.2e})"

    @pytest.mark.cuda
    def test_a_small_workspace_is_still_cached(self, device):
        """The guard in the other direction.

        A "fix" that simply stopped caching would satisfy every bound here, and
        would silently give back the win the cache exists for (a serial split-K
        semaphore that stays zeroed lets the kernel skip a per-launch memset —
        one whole kernel launch, ~5 us on this box, against a kernel of the same
        order).  So: a small workspace must still be cached, and re-used.
        """
        cfg = _split_k_variant("pk") or _split_k_variant("any")
        if cfg is None:
            pytest.skip("no split-K / Stream-K variant registered for this arch")
        mod = _get_gemm_module()
        fn = _gemm_fn(cfg.compile_name, False)
        split_k = max(2, int(getattr(cfg, "split_k", 1)))
        M, N, K = 128, 256, 512  # partials ~ 256 KiB * split
        # One stream, so one key.  Enough calls that a per-call allocation would
        # be unmistakable if it were counted.
        stream = torch.cuda.Stream()
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        C = torch.randn(N, device=device, dtype=torch.bfloat16)
        out = torch.empty(M, N, device=device, dtype=torch.bfloat16)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(out, A, B, C, split_k)  # first call: allocates
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        keys_after_first = mod.ws_cache_keys()
        bytes_after_first = mod.ws_cache_bytes()
        assert keys_after_first >= 1, (
            "a 256 KiB workspace was not cached at all — the cache is disabled or "
            "its ceiling is below what production actually asks for"
        )
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(64):
                fn(out, A, B, C, split_k)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        assert mod.ws_cache_bytes() == bytes_after_first, (
            "64 further calls on the same stream allocated again — the cached "
            "buffer is not being re-used"
        )
        assert mod.ws_cache_keys() == keys_after_first

    @pytest.mark.cuda
    def test_the_total_stays_bounded_across_shapes_and_streams(self, device):
        """Many shapes x the whole stream pool: the total is what must not grow."""
        cfg = _split_k_variant("pk") or _split_k_variant("any")
        if cfg is None:
            pytest.skip("no split-K / Stream-K variant registered for this arch")
        mod = _get_gemm_module()
        fn = _gemm_fn(cfg.compile_name, False)
        split_k = max(2, int(getattr(cfg, "split_k", 1)))
        for M, N in ((128, 256), (512, 1024), (1024, 2048), (2048, 5008), (4096, 5008)):
            err = _run_on_streams(fn, split_k, M, N, 256, device, 40)
            held = mod.ws_cache_bytes() / 2**20
            assert held <= 64, (
                f"cache holds {held:.1f} MiB after M={M} N={N} — over its documented "
                "64 MiB total ceiling"
            )
            assert err < 1e-2, f"M={M} N={N}: rel err {err:.2e}"

    @pytest.mark.cuda
    def test_results_stay_correct_when_the_cache_declines(self, device):
        """Past a ceiling the cache returns nullptr and the caller allocates a
        per-call workspace — correct, just without the saved ritual.  Prove the
        numerics do not change, on the same shape, either side of the ceiling."""
        cfg = _split_k_variant("any")
        if cfg is None:
            pytest.skip("no split-K / Stream-K variant registered for this arch")
        fn = _gemm_fn(cfg.compile_name, False)
        M, N, K = 128, 256, 512
        torch.manual_seed(3)
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        C = torch.randn(N, device=device, dtype=torch.bfloat16)
        ref = A.float() @ B.float().t() + C.float()
        split_k = max(2, int(getattr(cfg, "split_k", 1)))
        out = torch.empty(M, N, device=device, dtype=torch.bfloat16)
        for _ in range(140):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                out.zero_()
                fn(out, A, B, C, split_k)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-1)
            del s


# ---------------------------------------------------------------------------
# Every compiled variant, against an fp32 oracle
# ---------------------------------------------------------------------------

_ALL_VARIANTS = sorted(
    {
        cfg.compile_name: cfg
        for cfg in get_unique_compile_configs(_SM).values()
        if isinstance(cfg, CutlassGemmConfig)
    }.items()
)

#: Tiles without a K-decomposition — the subset every family exports.
_PLAIN_VARIANTS = [
    (name, cfg) for name, cfg in _ALL_VARIANTS if not (cfg.stream_k or cfg.parallel_split_k)
]


class TestEveryCompiledVariantIsCorrect:
    """The oracle the tuned rule table was missing.

    Selection was covered ("is the chosen config compiled?") and speed was
    covered (the tuner), and between them sat ``b128x16x64_w32x16x64``: a
    variant that compiled, launched, timed well and returned values 100-170x off
    the reference, because CUTLASS's epilogue cannot address a 16-wide tile at
    8 elements per access.  It was in the candidate space for the GEMM, BMM,
    grouped-GEMM and Conv2D families and in one production rule, where it
    emptied the transcript of any 1.1-2.2 s utterance a Zipformer CTC engine
    decoded on its own.

    So: sweep the whole compiled set, not the configs a rule happens to name
    today.  The shapes are one production shape (Zipformer's ConvNeXt pointwise
    contraction, at the M the broken rule covered) and one whose M, N and K are
    all indivisible by every tile, so partial tiles and predication are exercised
    too.
    """

    SHAPES = [(1710, 128, 384), (200, 264, 392)]

    @pytest.mark.parametrize("name,cfg", _ALL_VARIANTS, ids=[n for n, _ in _ALL_VARIANTS])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_an_fp32_reference(self, name, cfg, dtype, device):
        # Parallel split-K is a decomposition, not a tile: its reduction kernel
        # requires more than one K slice and refuses split_k == 1 by design.
        split_k = max(2, int(getattr(cfg, "split_k", 1))) if cfg.parallel_split_k else 1
        for M, N, K in self.SHAPES:
            torch.manual_seed(0)
            A = torch.randn(M, K, device=device, dtype=dtype)
            B = torch.randn(N, K, device=device, dtype=dtype)
            C = torch.randn(N, device=device, dtype=dtype)
            # A sentinel, not ``empty``: an epilogue that skips part of its tile
            # otherwise returns whatever the caching allocator last held there,
            # which is often plausible enough to pass a loose tolerance.
            out = torch.full((M, N), float("nan"), device=device, dtype=dtype)
            _gemm_fn(cfg.compile_name, False)(out, A, B, C, split_k)
            ref = torch.addmm(C.float(), A.float(), B.float().t())
            assert torch.isfinite(out).all(), f"{name} {dtype} {M}x{N}x{K}: left NaN behind"
            err = (out.float() - ref).abs().max().item()
            assert err < _tol(dtype, K, split_k, serial=False), (
                f"{name} {dtype} M={M} N={N} K={K}: max abs err {err:.4g} "
                f"(reference max {ref.abs().max().item():.4g})"
            )

    @pytest.mark.parametrize("name,cfg", _PLAIN_VARIANTS, ids=[n for n, _ in _PLAIN_VARIANTS])
    def test_bmm_variant_matches_an_fp32_reference(self, name, cfg, device):
        """Same sweep through the BMM launcher, which renders the same tiles.

        Only the plain tiles: Stream-K and parallel split-K are confined to the
        GEMM family, so ``bmm_*_sk`` / ``bmm_*_pk`` are never exported.
        """
        batch, M, N, K = 4, 200, 264, 392
        torch.manual_seed(0)
        A = torch.randn(batch, M, K, device=device, dtype=torch.float16)
        B = torch.randn(batch, N, K, device=device, dtype=torch.float16)
        out = torch.full((batch, M, N), float("nan"), device=device, dtype=torch.float16)
        _bmm_fn(cfg.compile_name)(out, A, B)
        ref = torch.matmul(A.float(), B.float().transpose(-1, -2))
        assert torch.isfinite(out).all(), f"{name}: left NaN behind"
        err = (out.float() - ref).abs().max().item()
        assert err < _tol(torch.float16, K), f"{name}: max abs err {err:.4g}"
