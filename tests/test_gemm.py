#!/usr/bin/env python3
"""
Unit tests for functional GEMM API (TVM-FFI JIT path).
"""

import itertools

import pytest
import torch

import oasr

# Every test in this module allocates directly on ``device="cuda"`` and calls a
# JIT-compiled kernel, so the whole file is CUDA-only.  Declaring that here is
# what lets the CPU CI job run `pytest tests/` and get a green, meaningful run
# instead of a wall of `RuntimeError: No CUDA GPUs are available`.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="OASR kernels require CUDA")


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
    """Every GEMM-family entry point answers the alignment question the same way.

    They used to disagree.  ``gemm`` let CUTLASS fail and surfaced "GEMM kernel
    failed", which tells the caller nothing actionable; ``gemm_log_softmax``
    silently rerouted the same input to cuBLAS, which tells them nothing at all
    and quietly leaves the model off the kernel path forever.  One precondition,
    one answer, and the message names the fix — every unaligned case in this repo
    is an output projection, and padding it at the model layer is the established
    pattern (``align_out_features`` / ``pad_output_projection``).
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

    def test_bmm_rejects_unaligned(self):
        A = torch.randn(2, 16, 60, device="cuda", dtype=torch.float16)
        B = torch.randn(2, 64, 60, device="cuda", dtype=torch.float16)
        with pytest.raises(Exception, match=self.MSG):
            oasr.bmm(A, B)

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
