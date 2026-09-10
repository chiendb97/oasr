#!/usr/bin/env python3
"""Unit tests for oasr.softmax() functional API."""

import pytest
import torch
import torch.nn.functional as F

import oasr

# Every test in this module allocates directly on ``device="cuda"`` and calls a
# JIT-compiled kernel, so the whole file is CUDA-only.  Declaring that here is
# what lets the CPU CI job run `pytest tests/` and get a green, meaningful run
# instead of a wall of `RuntimeError: No CUDA GPUs are available`.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="OASR kernels require CUDA")


class TestSoftmax:
    """Tests for oasr.softmax() functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,channels",
        [
            (1, 64, 128),
            (2, 128, 256),
            (4, 256, 512),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_softmax_correctness(self, batch_size, seq_len, channels, dtype):
        x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)

        output = oasr.softmax(x)
        expected = F.softmax(x.float(), dim=-1).to(dtype)

        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_softmax_sums_to_one(self, dtype):
        x = torch.randn(4, 128, 256, device="cuda", dtype=dtype)
        output = oasr.softmax(x)

        row_sums = output.float().sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_softmax_destination_passing(self, dtype):
        x = torch.randn(2, 64, 128, device="cuda", dtype=dtype)
        out = torch.empty_like(x)

        result = oasr.softmax(x, out=out)

        assert result.data_ptr() == out.data_ptr()
        expected = F.softmax(x.float(), dim=-1).to(dtype)
        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def test_softmax_numerical_stability(self):
        """Large inputs should not produce NaN or Inf."""
        x = torch.full((2, 64, 128), 1e4, device="cuda", dtype=torch.float32)
        output = oasr.softmax(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        # uniform distribution expected
        torch.testing.assert_close(output, torch.full_like(output, 1.0 / 128), rtol=1e-4, atol=1e-4)

    def test_softmax_2d_input(self):
        """Softmax should work on 2D inputs."""
        x = torch.randn(32, 256, device="cuda", dtype=torch.float32)
        output = oasr.softmax(x)
        expected = F.softmax(x, dim=-1)
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_softmax_cpu_error(self):
        """CPU tensors should raise an error."""
        x = torch.randn(2, 64, 128, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="CUDA tensor"):
            oasr.softmax(x)


def _prepared(scores, bias=None, mask=None, mask2=None, mask_value=-1000.0):
    """The op sequence ``oasr.masked_softmax`` fuses, up to (not including) the softmax."""
    out = scores
    if bias is not None:
        out = (out + bias).to(scores.dtype)
    if mask is not None:
        out = out.masked_fill(mask, mask_value)
    if mask2 is not None:
        out = out.masked_fill(mask2, mask_value)
    return out.contiguous()


def _relative_shift_view(pos, seq_len, key_len):
    """Zipformer's shifted window over a ``(H, B, T, 2T-1+L)`` position product."""
    num_heads, batch_size = pos.shape[0], pos.shape[1]
    return pos.as_strided(
        (num_heads, batch_size, seq_len, key_len),
        (pos.stride(0), pos.stride(1), pos.stride(2) - pos.stride(3), pos.stride(3)),
        storage_offset=pos.stride(3) * (seq_len - 1),
    )


class TestMaskedSoftmax:
    """KG6: the fused bias + mask + softmax that Zipformer's attention runs.

    The primary oracle is the **same kernel** fed the materialized sequence.
    That isolates what the fusion changed: folding the bias, the two masks and
    their strides into the loop must be numerically free, so the check is
    ``torch.equal`` and any drift is a real defect rather than fp32 rounding.

    Against ``oasr.softmax`` on the same materialized tensor the two agree to
    fp16 rounding, not bit for bit, and the reason is *not* the fusion: this
    kernel walks a 8/4/2/1 vector ladder where ``oasr.softmax`` only tries the
    widest width and then falls to scalar, so at a row length like 500 (which
    is 4- but not 8-divisible) they group the online reduction differently.
    Measured at that length: one element in 3000 moves by one fp16 ulp, and
    both are the same distance from the fp32 reference.
    """

    @staticmethod
    def _zipformer_operands(dtype, num_heads=4, batch=3, seq_len=37, seed=0):
        """The four tensors ``RelPositionMultiheadAttentionWeights`` hands over."""
        torch.manual_seed(seed)
        scores = torch.randn(num_heads, batch, seq_len, seq_len, device="cuda", dtype=dtype)
        pos = torch.randn(num_heads, batch, seq_len, 2 * seq_len - 1, device="cuda", dtype=dtype)
        bias = _relative_shift_view(pos, seq_len, seq_len)
        attn_mask = torch.rand(seq_len, seq_len, device="cuda") < 0.2
        key_padding = torch.rand(batch, seq_len, device="cuda") < 0.2
        key_padding[:, 0] = False  # no row is masked everywhere
        return scores, bias, attn_mask, key_padding.unsqueeze(1)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_folding_the_operands_in_is_numerically_free(self, dtype):
        """Reading the bias and masks through their strides must change nothing."""
        scores, bias, attn_mask, key_padding = self._zipformer_operands(dtype)

        got = oasr.masked_softmax(scores, bias=bias, mask=attn_mask, mask2=key_padding)
        expected = oasr.masked_softmax(_prepared(scores, bias, attn_mask, key_padding))
        assert torch.equal(got, expected)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_matches_the_unfused_sequence(self, dtype):
        scores, bias, attn_mask, key_padding = self._zipformer_operands(dtype)

        got = oasr.masked_softmax(scores, bias=bias, mask=attn_mask, mask2=key_padding)
        expected = oasr.softmax(_prepared(scores, bias, attn_mask, key_padding))
        rtol, atol = (1e-6, 1e-7) if dtype == torch.float32 else (1e-3, 1e-6)
        torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_no_operands_is_plain_softmax(self, dtype):
        x = torch.randn(2, 64, 128, device="cuda", dtype=dtype)
        assert torch.equal(oasr.masked_softmax(x), oasr.softmax(x))

    @pytest.mark.parametrize("num_cols", [1, 2, 3, 5, 7, 63, 64, 65, 500, 501, 4096])
    def test_vector_ladder_covers_odd_row_lengths(self, num_cols):
        """A row length is an attention extent, so it is whatever the audio made it."""
        torch.manual_seed(num_cols)
        scores = torch.randn(6, num_cols, device="cuda", dtype=torch.float16)
        bias = torch.randn(1, num_cols, device="cuda", dtype=torch.float16)
        got = oasr.masked_softmax(scores, bias=bias)
        assert torch.equal(got, oasr.masked_softmax(_prepared(scores, bias)))

    def test_row_wider_than_the_shared_memory_cache(self):
        """Past the cache the kernel re-derives the row; both phases must agree."""
        torch.manual_seed(7)
        scores = torch.randn(4, 20000, device="cuda", dtype=torch.float16)
        bias = torch.randn(4, 20000, device="cuda", dtype=torch.float16)
        got = oasr.masked_softmax(scores, bias=bias)
        assert torch.equal(got, oasr.masked_softmax(_prepared(scores, bias)))

    def test_strided_and_broadcast_operands_are_not_materialized(self):
        """A ``[..., ::ds]`` key-padding slice is what a downsampled stack passes."""
        torch.manual_seed(3)
        num_heads, batch, seq_len, ds = 4, 3, 37, 2
        scores = torch.randn(num_heads, batch, seq_len, seq_len, device="cuda", dtype=torch.float16)
        wide = torch.rand(batch, seq_len * ds, device="cuda") < 0.3
        key_padding = wide[..., ::ds]
        key_padding[:, 0] = False
        assert not key_padding.is_contiguous()

        got = oasr.masked_softmax(scores, mask=key_padding.unsqueeze(1))
        expected = oasr.masked_softmax(_prepared(scores, mask=key_padding.unsqueeze(1)))
        assert torch.equal(got, expected)

    def test_fully_masked_row_is_uniform_like_masked_fill(self):
        """A finite floor everywhere leaves a uniform row -- what torch does."""
        scores = torch.randn(2, 16, device="cuda", dtype=torch.float16)
        mask = torch.ones(2, 16, dtype=torch.bool, device="cuda")
        got = oasr.masked_softmax(scores, mask=mask)
        assert not torch.isnan(got).any()
        torch.testing.assert_close(got, torch.full_like(got, 1.0 / 16), rtol=1e-3, atol=1e-3)

    def test_mask_value_is_honored(self):
        scores = torch.zeros(1, 4, device="cuda", dtype=torch.float16)
        mask = torch.tensor([[False, True, False, False]], device="cuda")
        got = oasr.masked_softmax(scores, mask=mask, mask_value=-10.0)
        expected = oasr.masked_softmax(_prepared(scores, mask=mask, mask_value=-10.0))
        assert torch.equal(got, expected)

    def test_destination_passing(self):
        scores = torch.randn(2, 8, 32, device="cuda", dtype=torch.float16)
        bias = torch.randn(1, 1, 32, device="cuda", dtype=torch.float16)
        out = torch.empty_like(scores)
        result = oasr.masked_softmax(scores, bias=bias, out=out)
        assert result.data_ptr() == out.data_ptr()
        assert torch.equal(result, oasr.masked_softmax(_prepared(scores, bias)))

    def test_in_place_output(self):
        """The row is cached in shared memory, so phase 2 never re-reads input."""
        scores = torch.randn(3, 64, device="cuda", dtype=torch.float16)
        bias = torch.randn(3, 64, device="cuda", dtype=torch.float16)
        expected = oasr.masked_softmax(scores, bias=bias)
        result = oasr.masked_softmax(scores, bias=bias, out=scores)
        assert result.data_ptr() == scores.data_ptr()
        assert torch.equal(result, expected)

    def test_empty_input(self):
        scores = torch.empty(0, 8, device="cuda", dtype=torch.float16)
        assert oasr.masked_softmax(scores).shape == scores.shape

    def test_non_broadcastable_operand_raises(self):
        scores = torch.randn(2, 8, 32, device="cuda", dtype=torch.float16)
        bias = torch.randn(3, 32, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="broadcast"):
            oasr.masked_softmax(scores, bias=bias)

    def test_bias_dtype_mismatch_raises(self):
        scores = torch.randn(2, 8, 32, device="cuda", dtype=torch.float16)
        bias = torch.randn(2, 8, 32, device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="dtype"):
            oasr.masked_softmax(scores, bias=bias)

    def test_rank_above_four_with_an_operand_raises(self):
        """Three grid axes carry three leading axes; a 5-D score tensor needs a reshape."""
        scores = torch.randn(2, 2, 2, 4, 8, device="cuda", dtype=torch.float16)
        bias = torch.randn(2, 2, 2, 4, 8, device="cuda", dtype=torch.float16)
        assert oasr.masked_softmax(scores).shape == scores.shape  # no operand: any rank
        with pytest.raises(RuntimeError, match="leading axes"):
            oasr.masked_softmax(scores, bias=bias)

    def test_non_contiguous_scores_raise(self):
        """The broadcast views index by *logical* row, which a permuted view breaks."""
        scores = torch.randn(2, 32, 8, device="cuda", dtype=torch.float16).transpose(1, 2)
        with pytest.raises(RuntimeError, match="contiguous"):
            oasr.masked_softmax(scores)
