#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Normalization kernels (``oasr/functionals/norm.py``) against torch.

Every kernel here reduces over the last dim, so the grid axes that reach a
different code path are the **row width** (vec4 vs the scalar tail) and the
**row layout**; batch and sequence are grid parallelism.  ``alpha`` and
``has_bias`` are a scalar multiply and a vector add in the epilogue, and
neither can interact with the width or the dtype -- so they are swept as a
pair of representative combinations rather than as a full product.

:class:`TestRowLayoutPrecondition` is the load-bearing part of the file: the
launchers used to accept a padded row stride and return a plausible wrong
answer.
"""

import pytest
import torch
from helpers import assert_dest_passing, tol

import oasr

pytestmark = pytest.mark.cuda


class TestLayerNorm:
    """Tests for oasr.layer_norm() functional API."""

    #: One vec4-aligned row width and one that is not: the two launcher paths.
    @pytest.mark.parametrize("hidden_size", [256, 130])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_layer_norm(self, hidden_size, dtype, has_bias):
        eps = 1e-5
        x = torch.randn(2, 128, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
        bias = torch.randn(hidden_size, device="cuda", dtype=dtype) if has_bias else None

        output = oasr.layer_norm(x, weight, bias, eps)
        expected = torch.nn.functional.layer_norm(x, (hidden_size,), weight, bias, eps)

        torch.testing.assert_close(output, expected, **tol(dtype))

    def test_destination_passing(self):
        x = torch.randn(2, 128, 256, device="cuda", dtype=torch.float32)
        weight = torch.randn(256, device="cuda", dtype=torch.float32)
        bias = torch.randn(256, device="cuda", dtype=torch.float32)
        assert_dest_passing(
            oasr.layer_norm,
            x,
            weight,
            bias,
            1e-5,
            out=torch.empty_like(x),
            expected=torch.nn.functional.layer_norm(x, (256,), weight, bias, 1e-5),
        )


class TestRMSNorm:
    """Tests for oasr.rms_norm() functional API."""

    @pytest.mark.parametrize("hidden_size", [256, 130])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_rms_norm(self, hidden_size, dtype):
        eps = 1e-6
        x = torch.randn(2, 128, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        output = oasr.rms_norm(x, weight, eps=eps)

        # RMSNorm reference: y = x * weight / sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        expected = (x.float() / rms * weight.float()).to(dtype)

        torch.testing.assert_close(output, expected, **tol(dtype))


class TestAddRMSNorm:
    """Tests for fused residual add + RMSNorm, with and without passthrough."""

    @staticmethod
    def _reference(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
        alpha: float,
    ):
        summed_f = residual.float() + alpha * x.float()
        normalized = (
            summed_f * torch.rsqrt(summed_f.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
        )
        if bias is not None:
            normalized = normalized + bias.float()
        return normalized.to(x.dtype), summed_f.to(x.dtype)

    #: ``alpha`` is a scalar on the add and ``has_bias`` a vector in the
    #: epilogue; neither can interact with the row width or the dtype, so they
    #: are swept as two representative pairs rather than as a 2x2 product.
    @pytest.mark.parametrize("alpha,has_bias", [(1.0, True), (0.5, False)])
    @pytest.mark.parametrize("hidden_size", [256, 257])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_add_rms_norm(self, hidden_size, dtype, alpha, has_bias):
        eps = 1e-6
        x = torch.randn(2, 17, hidden_size, device="cuda", dtype=dtype)
        residual = torch.randn_like(x)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
        bias = torch.randn(hidden_size, device="cuda", dtype=dtype) if has_bias else None

        output = oasr.add_rms_norm(x, residual, weight, bias, eps, alpha)
        expected, _ = self._reference(x, residual, weight, bias, eps, alpha)

        torch.testing.assert_close(output, expected, **tol(dtype, half=(2e-2, 2e-2)))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("alpha", [1.0, 0.5])
    def test_add_rms_norm_residual(self, dtype, alpha):
        eps = 1e-6
        x = torch.randn(2, 11, 384, device="cuda", dtype=dtype)
        residual = torch.randn_like(x)
        weight = torch.randn(384, device="cuda", dtype=dtype)
        expected, summed = self._reference(x, residual, weight, None, eps, alpha)

        output, residual_out = oasr.add_rms_norm_residual(x, residual, weight, None, eps, alpha)

        torch.testing.assert_close(residual_out, summed, rtol=0, atol=0)
        torch.testing.assert_close(output, expected, **tol(dtype, half=(2e-2, 2e-2)))

    def test_destination_passing(self):
        x = torch.randn(2, 8, 256, device="cuda", dtype=torch.float16)
        residual = torch.randn_like(x)
        weight = torch.randn(256, device="cuda", dtype=torch.float16)
        expected, summed = self._reference(x, residual, weight, None, 1e-6, 1.0)

        assert_dest_passing(
            oasr.add_rms_norm, x, residual, weight, out=torch.empty_like(x), expected=expected
        )

        out, residual_out = torch.empty_like(x), torch.empty_like(x)
        assert_dest_passing(
            oasr.add_rms_norm_residual,
            x,
            residual,
            weight,
            out=out,
            outs=[residual_out],
            residual_out=residual_out,
        )
        torch.testing.assert_close(residual_out, summed, rtol=0, atol=0)


class TestBatchNorm1D:
    """Tests for oasr.batch_norm_1d() functional API."""

    @pytest.mark.parametrize("channels", [64, 130])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_batch_norm_1d(self, channels, dtype):
        """The affine parameters are deliberately not the identity.

        ``weight=1, bias=0, mean=0, var=1`` makes the whole transform an
        identity, so the test would pass on a kernel that ignored all four.
        """
        eps = 1e-5
        x = torch.randn(2, 128, channels, device="cuda", dtype=dtype)
        weight = torch.randn(channels, device="cuda", dtype=dtype)
        bias = torch.randn(channels, device="cuda", dtype=dtype)
        running_mean = torch.randn(channels, device="cuda", dtype=dtype)
        running_var = torch.rand(channels, device="cuda", dtype=dtype) + 0.5

        output = oasr.batch_norm_1d(x, weight, bias, running_mean, running_var, eps)

        # Reference: (x - mean) / sqrt(var + eps) * weight + bias
        expected = (
            (x.float() - running_mean.float())
            / torch.sqrt(running_var.float() + eps)
            * weight.float()
            + bias.float()
        ).to(dtype)

        torch.testing.assert_close(output, expected, **tol(dtype))


class TestGroupNorm:
    """Tests for oasr.group_norm() functional API."""

    @pytest.mark.parametrize("channels,num_groups", [(64, 4), (128, 8)])
    def test_group_norm(self, channels, num_groups):
        """Against ``F.group_norm`` over the flattened rows.

        This used to assert ``output.shape == x.shape`` and nothing else, so it
        passed on any kernel that allocated the right buffer -- including one
        that never wrote to it.
        """
        eps = 1e-5
        B, T = 2, 128
        x = torch.randn(B, T, channels, device="cuda", dtype=torch.float32)
        weight = torch.randn(channels, device="cuda", dtype=torch.float32)
        bias = torch.randn(channels, device="cuda", dtype=torch.float32)

        output = oasr.group_norm(x, weight, bias, num_groups, eps)

        # The kernel normalizes per (row, group), i.e. each (b, t) row is its
        # own "batch" of `channels` values split into `num_groups`.
        expected = torch.nn.functional.group_norm(
            x.reshape(B * T, channels), num_groups, weight, bias, eps
        ).reshape(B, T, channels)
        torch.testing.assert_close(output, expected, **tol(torch.float32))


class TestAddLayerNorm:
    """Tests for oasr.add_layer_norm() functional API."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("alpha", [1.0, 0.5])
    def test_add_layer_norm(self, dtype, alpha):
        batch_size, seq_len, hidden_size = 2, 128, 256
        eps = 1e-5
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        residual = torch.randn_like(x)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
        bias = torch.randn(hidden_size, device="cuda", dtype=dtype)

        output = oasr.add_layer_norm(x, residual, weight, bias, eps, alpha=alpha)

        # The fused add stays in fp32 through normalization and rounds only
        # when the normalized output is stored.
        combined = residual.float() + alpha * x.float()
        expected = torch.nn.functional.layer_norm(
            combined, (hidden_size,), weight.float(), bias.float(), eps
        ).to(dtype)

        torch.testing.assert_close(output, expected, **tol(dtype))


class TestAddLayerNormResidual:
    """Tests for oasr.add_layer_norm_residual() (fused add+LN + residual sum)."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("alpha,has_bias", [(1.0, True), (0.5, False)])
    def test_add_layer_norm_residual(self, dtype, alpha, has_bias):
        batch_size, seq_len, hidden_size = 2, 128, 256
        eps = 1e-5
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        residual = torch.randn_like(x)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
        bias = torch.randn(hidden_size, device="cuda", dtype=dtype) if has_bias else None

        out, res_out = oasr.add_layer_norm_residual(x, residual, weight, bias, eps, alpha)

        # Normalization consumes the sum directly in fp32. The carried residual
        # is the only copy rounded to the served dtype.
        s_float = residual.float() + alpha * x.float()
        s_ref = s_float.to(dtype)
        bias_float = None if bias is None else bias.float()
        expected = torch.nn.functional.layer_norm(
            s_float, (hidden_size,), weight.float(), bias_float, eps
        ).to(dtype)

        # The carried residual is the fp32 sum rounded once on output.
        torch.testing.assert_close(res_out, s_ref, rtol=0, atol=0)
        torch.testing.assert_close(out, expected, **tol(dtype))


def _ref_bias_norm(x: torch.Tensor, bias: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """icefall BiasNorm reference (channel_dim == -1), computed in fp32."""
    xf = x.float()
    b = bias.float()
    scales = torch.mean((xf - b) ** 2, dim=-1, keepdim=True) ** -0.5 * log_scale.float().exp()
    return (xf * scales).to(x.dtype)


class TestBiasNorm:
    """Tests for oasr.bias_norm() functional API (Zipformer BiasNorm)."""

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 250, 384),  # a real Zipformer width
            (2, 8, 50, 64),  # 4-D
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_bias_norm(self, shape, dtype):
        hidden_size = shape[-1]
        x = torch.randn(*shape, device="cuda", dtype=dtype)
        bias = torch.randn(hidden_size, device="cuda", dtype=dtype) * 0.1
        log_scale = torch.tensor(0.7, device="cuda", dtype=dtype)

        output = oasr.bias_norm(x, bias, log_scale)
        expected = _ref_bias_norm(x, bias, log_scale)

        torch.testing.assert_close(
            output.float(), expected.float(), **tol(dtype, half=(2e-2, 2e-2))
        )

    def test_bias_norm_non_vec_aligned(self):
        """Hidden size not divisible by 4 must fall back to the scalar path."""
        x = torch.randn(2, 32, 130, device="cuda", dtype=torch.float32)
        bias = torch.randn(130, device="cuda", dtype=torch.float32) * 0.1
        log_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        output = oasr.bias_norm(x, bias, log_scale)
        torch.testing.assert_close(output, _ref_bias_norm(x, bias, log_scale), rtol=1e-4, atol=1e-4)

    def test_bias_norm_destination_passing(self):
        x = torch.randn(2, 128, 256, device="cuda", dtype=torch.float32)
        bias = torch.randn(256, device="cuda", dtype=torch.float32) * 0.1
        log_scale = torch.tensor(0.3, device="cuda", dtype=torch.float32)
        assert_dest_passing(
            oasr.bias_norm,
            x,
            bias,
            log_scale,
            out=torch.empty_like(x),
            expected=_ref_bias_norm(x, bias, log_scale),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestRowLayoutPrecondition:
    """What the norm kernels actually require of their input layout.

    Every norm kernel walks rows as ``base + row * hidden_size``.  The launchers
    used to check only ``stride(-1) == 1``, which is both too weak and too
    strong:

    * **too weak** — a padded row stride (``x[..., :H]`` of a wider buffer, or
      ``x[:, -1]`` of a ``(B, T, D)`` tensor) satisfies it, and the kernel then
      reads the wrong memory and returns a plausible wrong answer *silently*;
    * **too strong** in the sense that the fix "require ``is_contiguous()``"
      needlessly refuses a *permuted* dense view.  Zipformer works in
      ``(T, B, C)`` — a transpose of a contiguous ``(B, T, C)`` — whose rows
      still tile memory exactly, so processing them in memory order is
      identical (normalization is per-row and independent, and
      ``torch.empty_like`` preserves the strides).

    The precondition is therefore "rows tile memory exactly", which the
    launchers now check via ``IsRowDense``.
    """

    HIDDEN = 64

    def _params(self, dtype=torch.float16):
        H = self.HIDDEN
        return (
            torch.randn(H, device="cuda", dtype=dtype),
            torch.randn(H, device="cuda", dtype=dtype),
        )

    def test_permuted_dense_view_matches_torch(self):
        """The Zipformer layout: accepted, and right."""
        H = self.HIDDEN
        torch.manual_seed(0)
        x = torch.randn(2, 96, H, device="cuda", dtype=torch.float16).transpose(0, 1)
        assert not x.is_contiguous(), "the test input must actually be permuted"
        w, b = self._params()
        got = oasr.layer_norm(x, w, b, 1e-5)
        ref = torch.nn.functional.layer_norm(x, (H,), w, b, 1e-5)
        assert got.stride() == x.stride(), "output must keep the input's layout"
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)

    def test_permuted_dense_view_rms_and_bias_norm(self):
        H = self.HIDDEN
        torch.manual_seed(1)
        x = torch.randn(2, 96, H, device="cuda", dtype=torch.float16).transpose(0, 1)
        w, b = self._params()
        xf = x.float()
        torch.testing.assert_close(
            oasr.rms_norm(x, w, None, 1e-6),
            (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).half(),
            rtol=1e-2,
            atol=1e-2,
        )
        log_scale = torch.tensor(1.0, device="cuda", dtype=torch.float16)
        scales = (torch.mean((x - b) ** 2, dim=-1, keepdim=True) ** -0.5) * log_scale.exp()
        torch.testing.assert_close(
            oasr.bias_norm(x, b, log_scale), x * scales, rtol=1e-2, atol=1e-2
        )

    @pytest.mark.parametrize("kind", ["trailing_slice", "row_slice"])
    def test_padded_row_stride_is_rejected(self, kind):
        """Regression: these used to return silently wrong data."""
        H = self.HIDDEN
        if kind == "trailing_slice":
            x = torch.randn(4, 6, 2 * H, device="cuda", dtype=torch.float16)[:, :, :H]
        else:
            x = torch.randn(4, 6, H, device="cuda", dtype=torch.float16)[:, -1]
        assert x.stride(-1) == 1, "premise: the weak check would have passed this"
        w, b = self._params()
        with pytest.raises(Exception, match="tile memory exactly"):
            oasr.layer_norm(x, w, b, 1e-5)

    def test_waist_predicate_agrees_with_the_launcher(self):
        """``is_row_dense`` mirrors ``IsRowDense``; they must not drift apart, or
        the waist would route something the launcher rejects."""
        from oasr.layers._backend import is_row_dense

        H = self.HIDDEN
        cases = {
            "contiguous": (torch.randn(8, H), True),
            "permuted dense": (torch.randn(2, 96, H).transpose(0, 1), True),
            "trailing slice": (torch.randn(4, 6, 2 * H)[:, :, :H], False),
            "row slice": (torch.randn(4, 6, H)[:, -1], False),
            "expanded (aliasing)": (torch.randn(1, H).expand(8, H), False),
        }
        for name, (t, expected) in cases.items():
            assert is_row_dense(t) is expected, name
