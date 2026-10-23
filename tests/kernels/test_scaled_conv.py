#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr/functionals/scaled_conv.py``: the int8 implicit-GEMM convolution.

* every compiled tile against the fp64 reference, over geometries no tile
  divides (padding, stride, dilation, a height-one conv1d) and every fused
  activation -- the GEMM families' lesson: a tile whose epilogue cannot address
  its output still runs and returns wrong values;
* the int32 accumulator is exact: with unit scales the output is the integer
  convolution rounded once, and a sample computed alone is bitwise the one
  computed among others (batch invariance by construction);
* the launcher contracts: destination passing, graph replay, alignment.
"""

from __future__ import annotations

import pytest
import torch
from helpers import assert_dest_passing, assert_graph_replay

from oasr.functionals.scaled_conv import (
    _get_scaled_conv_module,
    reference_scaled_conv1d,
    reference_scaled_conv2d,
    scaled_conv1d,
    scaled_conv2d,
)
from oasr.jit.scaled_conv import get_production_configs, scaled_conv2d_func_name, select_config

pytestmark = pytest.mark.cuda

_DTYPES = (torch.float16, torch.bfloat16)
_ACTIVATIONS = (None, "relu", "swish", "gelu_tanh", "gelu")


def _operands(n, h, w, c, k, r, s, dtype, seed=0, lo=-127, hi=128):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randint(lo, hi, (n, h, w, c), generator=g, dtype=torch.int8, device="cuda")
    wt = torch.randint(lo, hi, (k, r, s, c), generator=g, dtype=torch.int8, device="cuda")
    scale = torch.rand(k, generator=g, device="cuda") * 1e-4 + 1e-6
    bias = torch.randn(k, generator=g, device="cuda").to(dtype)
    return x, wt, scale, bias


def _assert_within_an_ulp(got: torch.Tensor, ref: torch.Tensor) -> None:
    assert not torch.isnan(got).any()
    rel = 2.0**-9 if got.dtype == torch.float16 else 2.0**-6
    diff = (got.float() - ref.float()).abs()
    assert bool((diff <= ref.float().abs() * rel + 1e-3).all()), f"max diff {float(diff.max())}"


#: ``(N, H, W, C, K, R, S, padding, stride, dilation)`` -- none a multiple of a tile.
_GEOMETRIES = [
    (2, 17, 11, 32, 64, 3, 3, (1, 1), (2, 2), (1, 1)),
    (3, 9, 13, 16, 24, 3, 3, (0, 0), (1, 1), (1, 1)),
    (1, 23, 7, 48, 136, 5, 3, (2, 1), (1, 2), (1, 1)),
    (2, 12, 12, 32, 40, 3, 3, (2, 2), (1, 1), (2, 2)),
    (5, 6, 5, 64, 72, 1, 1, (0, 0), (1, 1), (1, 1)),
]


class TestEveryCompiledVariantIsCorrect:
    @pytest.mark.parametrize("name", sorted(get_production_configs()))
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("geometry", _GEOMETRIES)
    def test_variant(self, name, dtype, geometry):
        n, h, w, c, k, r, s, pad, stride, dil = geometry
        fn = getattr(
            _get_scaled_conv_module(), scaled_conv2d_func_name(get_production_configs()[name])
        )
        x, wt, scale, bias = _operands(n, h, w, c, k, r, s, dtype, seed=n + c + k)
        p = (h + 2 * pad[0] - dil[0] * (r - 1) - 1) // stride[0] + 1
        q = (w + 2 * pad[1] - dil[1] * (s - 1) - 1) // stride[1] + 1
        for act_id, act in zip((3, 0, 2, 1, 4), _ACTIVATIONS):
            for b in (bias, None):
                out = torch.full((n, p, q, k), float("nan"), dtype=dtype, device="cuda")
                fn(out, x, wt, scale, b, act_id, *pad, *stride, *dil)
                ref = reference_scaled_conv2d(x, wt, scale, b, act, pad, stride, dil, dtype)
                _assert_within_an_ulp(out, ref)


class TestFunctional:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_matches_reference(self, dtype):
        x, wt, scale, bias = _operands(4, 33, 19, 256, 256, 3, 3, dtype)
        got = scaled_conv2d(x, wt, scale, bias, "relu", 0, 2, out_dtype=dtype)
        _assert_within_an_ulp(
            got, reference_scaled_conv2d(x, wt, scale, bias, "relu", 0, 2, out_dtype=dtype)
        )

    def test_accumulator_is_exact(self):
        """Unit scales, no bias, identity: the integer convolution, rounded once."""
        x, wt, _, _ = _operands(2, 15, 9, 64, 32, 3, 3, torch.float16, lo=-3, hi=4)
        ones = torch.ones(32, device="cuda")
        got = scaled_conv2d(x, wt, ones, None, None, 1, 1)
        exact = torch.nn.functional.conv2d(
            x.permute(0, 3, 1, 2).double().cpu(), wt.permute(0, 3, 1, 2).double().cpu(), padding=1
        ).permute(0, 2, 3, 1)
        assert torch.equal(got.cpu(), exact.float().half())

    def test_a_sample_is_bitwise_independent_of_the_batch(self):
        """M picks the tile; the int32 sum makes the sample identical anyway."""
        x, wt, scale, bias = _operands(64, 33, 19, 256, 256, 3, 3, torch.float16)
        m_full, m_one = 64 * 16 * 9, 16 * 9
        assert select_config(m_full, 256, 2304) != select_config(m_one, 256, 2304)
        full = scaled_conv2d(x, wt, scale, bias, "swish", 0, 2)
        for i in (0, 37, 63):
            one = scaled_conv2d(x[i : i + 1].contiguous(), wt, scale, bias, "swish", 0, 2)
            assert torch.equal(one[0], full[i])

    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("padding,stride,dilation", [(1, 1, 1), (1, 2, 1), (2, 1, 2)])
    def test_conv1d_is_the_height_one_problem(self, dtype, padding, stride, dilation):
        g = torch.Generator(device="cuda").manual_seed(5)
        x = torch.randint(-127, 128, (3, 101, 80), generator=g, dtype=torch.int8, device="cuda")
        wt = torch.randint(-127, 128, (64, 3, 80), generator=g, dtype=torch.int8, device="cuda")
        scale = torch.rand(64, generator=g, device="cuda") * 1e-4
        bias = torch.randn(64, generator=g, device="cuda").to(dtype)
        got = scaled_conv1d(x, wt, scale, bias, "gelu", padding, stride, dilation, out_dtype=dtype)
        ref = reference_scaled_conv1d(x, wt, scale, bias, "gelu", padding, stride, dilation, dtype)
        assert got.shape == ref.shape
        _assert_within_an_ulp(got, ref)

    def test_destination_passing(self):
        x, wt, scale, bias = _operands(2, 9, 9, 32, 64, 3, 3, torch.float16)
        out = torch.empty(2, 9, 9, 64, dtype=torch.float16, device="cuda")
        assert_dest_passing(
            scaled_conv2d,
            x,
            wt,
            scale,
            bias,
            "relu",
            1,
            1,
            out=out,
            expected=reference_scaled_conv2d(x, wt, scale, bias, "relu", 1, 1),
        )

    def test_graph_replay_recomputes(self):
        x, wt, scale, bias = _operands(2, 9, 9, 32, 64, 3, 3, torch.float16)
        out = torch.empty(2, 9, 9, 64, dtype=torch.float16, device="cuda")
        assert_graph_replay(
            lambda: scaled_conv2d(x, wt, scale, bias, None, 1, 1, out=out),
            out=out,
            mutate=lambda: x.copy_(torch.randint_like(x, -127, 128)),
            expected=lambda: reference_scaled_conv2d(x, wt, scale, bias, None, 1, 1),
            atol=1e-3,
            rtol=2.0**-9,
        )

    def test_unaligned_channels_are_refused(self):
        x, wt, scale, _ = _operands(1, 5, 5, 24, 16, 3, 3, torch.float16)
        with pytest.raises(Exception, match="C % 16"):
            scaled_conv2d(x, wt, scale)

    def test_every_tile_is_buildable_and_the_heuristic_picks_one(self):
        prod = get_production_configs()
        assert prod
        for m in (1, 100, 5000, 200000):
            for n, k in ((256, 2304), (384, 240), (1280, 3840)):
                assert select_config(m, n, k).compile_name in prod
