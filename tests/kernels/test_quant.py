#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr/functionals/quant.py``: per-token int8 activation quantization.

The kernel's numerics are pinned (``include/oasr/quant/quantize.cuh``) and the
torch twin ``reference_quantize_per_token`` reproduces them, so every check here
is **bit-exact** — ``torch.equal``, not a tolerance.  A one-ulp difference in a
scale is a different quantized activation, which is the class of difference that
has changed decoded tokens before (AGENTS.md rule 11).
"""

from __future__ import annotations

import pytest
import torch
from helpers import assert_dest_passing, assert_graph_replay

import oasr
from oasr.functionals.quant import INT8_QMAX, QUANT_EPS, reference_quantize_per_token

pytestmark = pytest.mark.cuda

_DTYPES = (torch.float16, torch.bfloat16)


def _activations(m: int, k: int, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """Rows with per-channel outliers (what SmoothQuant exists for) and one zero row."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda")
    x = x * torch.linspace(0.05, 40.0, k, device="cuda")
    x[0] = 0.0
    return x.to(dtype)


class TestBitExact:
    # K covers the warp-per-row lane (<= 1024), the CTA-per-row lane, the
    # two-pass fallback (> 16384) and the scalar path (not 8-aligned).
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("k", [8, 256, 264, 1024, 1032, 2048, 4096, 16384, 20000])
    @pytest.mark.parametrize("m", [1, 7, 333])
    def test_matches_reference(self, dtype, k, m):
        x = _activations(m, k, dtype, seed=k + m)
        q, scale = oasr.quantize_per_token(x)
        q_ref, scale_ref = reference_quantize_per_token(x)
        assert torch.equal(q, q_ref)
        assert torch.equal(scale, scale_ref)

    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("k", [256, 2048, 264])
    def test_smoother_is_applied_before_the_amax(self, dtype, k):
        x = _activations(64, k, dtype)
        smooth = torch.rand(k, device="cuda") + 0.25
        q, scale = oasr.quantize_per_token(x, smooth)
        q_ref, scale_ref = reference_quantize_per_token(x, smooth)
        assert torch.equal(q, q_ref) and torch.equal(scale, scale_ref)
        unsmoothed, _ = oasr.quantize_per_token(x)
        assert not torch.equal(q, unsmoothed), "the smoother changed nothing"

    def test_reference_is_device_independent(self):
        x = _activations(50, 512, torch.float16)
        q, scale = reference_quantize_per_token(x)
        q_cpu, scale_cpu = reference_quantize_per_token(x.cpu())
        assert torch.equal(q.cpu(), q_cpu) and torch.equal(scale.cpu(), scale_cpu)


class TestNumerics:
    def test_zero_row_is_zero_with_a_finite_scale(self):
        x = torch.zeros(4, 256, dtype=torch.float16, device="cuda")
        q, scale = oasr.quantize_per_token(x)
        assert not q.any()
        assert torch.isfinite(scale).all() and (scale > 0).all()
        assert torch.allclose(scale, torch.full_like(scale, QUANT_EPS / INT8_QMAX))

    def test_symmetric_range_and_saturation(self):
        x = _activations(128, 512, torch.float16)
        q, _ = oasr.quantize_per_token(x)
        assert int(q.min()) >= -INT8_QMAX and int(q.max()) <= INT8_QMAX
        # The amax element of every nonzero row lands exactly on +-127.
        assert torch.equal(q[1:].abs().amax(dim=1), torch.full((127,), 127, device="cuda").to(q))

    def test_dequantization_error_is_half_a_step(self):
        x = _activations(64, 1024, torch.float16)
        q, scale = oasr.quantize_per_token(x)
        err = (q.float() * scale[:, None] - x.float()).abs()
        # Half a step, plus the fp32 rounding of ``x * (127 / amax)`` (~1e-5 step).
        assert bool((err <= scale[:, None] * (0.5 + 1e-4)).all())

    @pytest.mark.parametrize("k", [256, 2048])
    def test_rows_are_independent_of_the_batch(self, k):
        """A row quantizes the same alone and among 1000 others (batch invariance)."""
        x = _activations(1000, k, torch.float16)
        q_all, s_all = oasr.quantize_per_token(x)
        for r in (1, 500, 999):
            q_one, s_one = oasr.quantize_per_token(x[r : r + 1].contiguous())
            assert torch.equal(q_one[0], q_all[r]) and torch.equal(s_one[0], s_all[r])

    def test_leading_dims_are_flattened(self):
        x = _activations(6 * 40, 256, torch.bfloat16).reshape(6, 40, 256)
        q, scale = oasr.quantize_per_token(x)
        assert q.shape == x.shape and scale.shape == (6, 40)
        q_ref, s_ref = reference_quantize_per_token(x)
        assert torch.equal(q, q_ref) and torch.equal(scale, s_ref)


class TestContract:
    def test_destination_passing(self):
        x = _activations(32, 256, torch.float16)
        out = torch.empty(32, 256, dtype=torch.int8, device="cuda")
        scale = torch.empty(32, device="cuda")
        q_ref, s_ref = reference_quantize_per_token(x)
        ret = assert_dest_passing(oasr.quantize_per_token, x, out=out, scale=scale)
        assert ret[1].data_ptr() == scale.data_ptr()
        assert torch.equal(out, q_ref) and torch.equal(scale, s_ref)

    def test_rejects_a_wrong_scale_shape(self):
        x = _activations(32, 256, torch.float16)
        with pytest.raises(RuntimeError, match="one float per row"):
            oasr.quantize_per_token(x, scale=torch.empty(31, device="cuda"))

    def test_rejects_fp32_input(self):
        with pytest.raises(RuntimeError):
            oasr.quantize_per_token(torch.randn(4, 256, device="cuda"))

    def test_graph_replay_recomputes(self):
        x = _activations(64, 512, torch.float16)
        out = torch.empty(64, 512, dtype=torch.int8, device="cuda")
        scale = torch.empty(64, device="cuda")

        def launch():
            return oasr.quantize_per_token(x, out=out, scale=scale)

        def mutate():
            x.mul_(-2.0)

        assert_graph_replay(
            launch,
            out=out,
            mutate=mutate,
            expected=lambda: reference_quantize_per_token(x)[0],
            rtol=0,
            atol=0,
        )
