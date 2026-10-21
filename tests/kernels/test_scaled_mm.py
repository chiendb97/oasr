#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr/functionals/scaled_mm.py``: the int8 W8A8 GEMM with its dequant epilogue.

Three layers of checking:

* every compiled tile variant against the reference, on shapes no tile divides
  (``TestEveryCompiledVariantIsCorrect`` — the GEMM family's lesson: a tile whose
  epilogue cannot address its output still runs and returns wrong values);
* the int32 accumulator is exact, so the kernel's output is **bitwise** the same
  for a row computed alone or among 1000 others — batch invariance by
  construction, which the CI's bit-identical-transcripts property relies on;
* the launcher contracts: destination passing, alignment, graph replay.

The epilogue fuses ``a_scale * t + bias`` into an FMA, so against the fp32 torch
reference the output agrees to within an output ulp, not bitwise.
"""

from __future__ import annotations

import pytest
import torch
from helpers import assert_dest_passing, assert_graph_replay

import oasr
from oasr.functionals.scaled_mm import _get_scaled_mm_module, activation_id, reference_scaled_mm
from oasr.jit.gemm import _epilogue_covers_warp
from oasr.jit.scaled_mm import (
    _EPILOGUE_ELEMENT_BITS,
    get_production_configs,
    scaled_mm_func_name,
    select_config,
)

pytestmark = pytest.mark.cuda

_DTYPES = (torch.float16, torch.bfloat16)
_ACTIVATIONS = (None, "relu", "swish", "gelu", "gelu_tanh")


def _operands(m: int, n: int, k: int, dtype: torch.dtype, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randint(-127, 128, (m, k), generator=g, dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8, device="cuda")
    a_scale = torch.rand(m, generator=g, device="cuda") * 0.01 + 1e-3
    b_scale = torch.rand(n, generator=g, device="cuda") * 0.01 + 1e-3
    bias = torch.randn(n, generator=g, device="cuda").to(dtype)
    return a, b, a_scale, b_scale, bias


def _assert_within_an_ulp(got: torch.Tensor, ref: torch.Tensor) -> None:
    assert not torch.isnan(got).any()
    # fp16 has 10 mantissa bits, bf16 7: one ulp of the larger operand, plus the
    # activation's own fast-math approximation near zero.
    rel = 2.0**-9 if got.dtype == torch.float16 else 2.0**-6
    tol = ref.float().abs() * rel + 1e-3
    diff = (got.float() - ref.float()).abs()
    assert bool((diff <= tol).all()), f"max diff {float(diff.max())}"


class TestEveryCompiledVariantIsCorrect:
    @pytest.mark.parametrize("name", sorted(get_production_configs()))
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("m,n,k", [(1710, 128, 384), (200, 264, 384), (1, 256, 2048)])
    def test_variant(self, name, dtype, m, n, k):
        cfg = get_production_configs()[name]
        fn = getattr(_get_scaled_mm_module(), scaled_mm_func_name(cfg))
        a, b, a_scale, b_scale, bias = _operands(m, n, k, dtype, seed=m + n)
        for act in _ACTIVATIONS:
            out = torch.full((m, n), float("nan"), dtype=dtype, device="cuda")
            fn(out, a, b, a_scale, b_scale, bias, activation_id(act))
            _assert_within_an_ulp(
                out, reference_scaled_mm(a, b, a_scale, b_scale, bias, act, dtype)
            )

    def test_every_variant_covers_a_warp(self):
        """The thin-N epilogue trap is a config-space property, checked statically."""
        for cfg in get_production_configs().values():
            assert _epilogue_covers_warp(cfg.tile, element_bits=_EPILOGUE_ELEMENT_BITS), cfg


class TestFunctional:
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("m", [1, 16, 17, 1000])
    def test_matches_reference_with_and_without_bias(self, dtype, m):
        a, b, a_scale, b_scale, bias = _operands(m, 768, 256, dtype, seed=m)
        for bias_or_none in (bias, None):
            got = oasr.scaled_mm(a, b, a_scale, b_scale, bias_or_none, "swish", out_dtype=dtype)
            ref = reference_scaled_mm(a, b, a_scale, b_scale, bias_or_none, "swish", dtype)
            _assert_within_an_ulp(got, ref)

    def test_accumulator_is_exact(self):
        """Unit scales, no bias, identity: the output is the int32 product, rounded once."""
        a, b, _, _, _ = _operands(64, 128, 2048, torch.float16)
        ones_m = torch.ones(64, device="cuda")
        ones_n = torch.ones(128, device="cuda")
        got = oasr.scaled_mm(a, b, ones_m, ones_n, None, None, out_dtype=torch.float16)
        exact = torch._int_mm(a.cpu(), b.cpu().t()).float().half()
        assert torch.equal(got.cpu(), exact)

    @pytest.mark.parametrize("n,k", [(768, 256), (256, 2048)])
    def test_rows_are_bitwise_independent_of_the_batch(self, n, k):
        """M=1 and M=1000 pick different tiles; int32 sums make the row identical anyway."""
        a, b, a_scale, b_scale, bias = _operands(1000, n, k, torch.float16)
        assert select_config(1, n, k) != select_config(1000, n, k)
        full = oasr.scaled_mm(a, b, a_scale, b_scale, bias, "swish")
        for r in (0, 517, 999):
            one = oasr.scaled_mm(
                a[r : r + 1].contiguous(),
                b,
                a_scale[r : r + 1].contiguous(),
                b_scale,
                bias,
                "swish",
            )
            assert torch.equal(one[0], full[r])

    def test_leading_dims_are_flattened(self):
        a, b, a_scale, b_scale, bias = _operands(4 * 30, 256, 256, torch.bfloat16)
        got = oasr.scaled_mm(
            a.reshape(4, 30, 256), b, a_scale, b_scale, bias, None, out_dtype=torch.bfloat16
        )
        assert got.shape == (4, 30, 256)
        _assert_within_an_ulp(
            got.reshape(120, 256),
            reference_scaled_mm(a, b, a_scale, b_scale, bias, None, torch.bfloat16),
        )

    def test_unknown_activation_is_refused(self):
        with pytest.raises(ValueError, match="cannot fuse"):
            activation_id("tanh")


class TestContract:
    def test_destination_passing(self):
        a, b, a_scale, b_scale, bias = _operands(64, 256, 256, torch.float16)
        out = torch.empty(64, 256, dtype=torch.float16, device="cuda")
        assert_dest_passing(oasr.scaled_mm, a, b, a_scale, b_scale, bias, None, out=out)
        _assert_within_an_ulp(
            out, reference_scaled_mm(a, b, a_scale, b_scale, bias, None, torch.float16)
        )

    def test_alignment_contract_names_the_shape(self):
        a, b, a_scale, b_scale, _ = _operands(8, 256, 264, torch.float16)
        with pytest.raises(RuntimeError, match="K % 16 == 0"):
            oasr.scaled_mm(a, b, a_scale, b_scale)

    def test_scale_shapes_are_checked(self):
        a, b, a_scale, b_scale, _ = _operands(8, 256, 256, torch.float16)
        with pytest.raises(RuntimeError, match="a_scale needs one float per row"):
            oasr.scaled_mm(a, b, a_scale[:7].contiguous(), b_scale)

    def test_graph_replay_recomputes(self):
        a, b, a_scale, b_scale, bias = _operands(256, 512, 256, torch.float16)
        out = torch.empty(256, 512, dtype=torch.float16, device="cuda")

        def launch():
            return oasr.scaled_mm(a, b, a_scale, b_scale, bias, "relu", out=out)

        def mutate():
            a.neg_()
            a_scale.mul_(2.0)

        assert_graph_replay(
            launch,
            out=out,
            mutate=mutate,
            expected=lambda: reference_scaled_mm(a, b, a_scale, b_scale, bias, "relu"),
            rtol=2.0**-9,
            atol=1e-3,
        )
