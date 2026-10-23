#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr/functionals/scaled_mm.py``: the 8-bit W8A8 GEMM with its dequant epilogue.

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

The fp8 e4m3 lane (``TestFp8``) accumulates in fp32, so it is not exact -- but
every one of its tiles sums each output in the same order (fast accumulation, no
K sliced across warps, no split), so all of them compute the same bits, and the
row-alone-vs-in-a-batch check is bitwise there too.
"""

from __future__ import annotations

import pytest
import torch
from helpers import assert_dest_passing, assert_graph_replay

import oasr
from oasr.functionals.quant import static_multiplier
from oasr.functionals.scaled_mm import (
    _get_scaled_mm_module,
    activation_id,
    reference_scaled_mm,
    reference_scaled_mm_quant,
)
from oasr.jit.gemm import _epilogue_covers_warp
from oasr.jit.scaled_mm import (
    _EPILOGUE_ELEMENT_BITS,
    get_production_configs,
    lane_available,
    scaled_mm_func_name,
    scaled_mm_quant_func_name,
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


_FP8 = torch.float8_e4m3fn


def _fp8_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return lane_available(major * 10 + minor, "fp8")


def _fp8_operands(m: int, n: int, k: int, dtype: torch.dtype, seed: int = 0):
    """Quantized-like e4m3 operands: heavy-tailed values over the whole range."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(m, k, generator=g, device="cuda") * 64.0).clamp(-448, 448).to(_FP8)
    b = (torch.randn(n, k, generator=g, device="cuda") * 64.0).clamp(-448, 448).to(_FP8)
    a_scale = torch.rand(m, generator=g, device="cuda") * 1e-3 + 1e-4
    b_scale = torch.rand(n, generator=g, device="cuda") * 1e-3 + 1e-4
    bias = torch.randn(n, generator=g, device="cuda").to(dtype)
    return a, b, a_scale, b_scale, bias


def _assert_fp8_close(got: torch.Tensor, ref: torch.Tensor) -> None:
    """An output ulp, plus the fp32 accumulator's rounding (the reference sums exactly)."""
    assert not torch.isnan(got).any()
    rel = 2.0**-8 if got.dtype == torch.float16 else 2.0**-6
    tol = ref.float().abs() * rel + 2e-3
    diff = (got.float() - ref.float()).abs()
    assert bool((diff <= tol).all()), f"max diff {float(diff.max())}"


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


@pytest.mark.skipif(not _fp8_supported(), reason="fp8 tensor cores need sm89+")
class TestFp8:
    @pytest.mark.parametrize("name", sorted(get_production_configs(fmt="fp8")))
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("m,n,k", [(1710, 128, 384), (200, 264, 384), (1, 256, 2048)])
    def test_variant(self, name, dtype, m, n, k):
        cfg = get_production_configs(fmt="fp8")[name]
        fn = getattr(_get_scaled_mm_module("fp8"), scaled_mm_func_name(cfg))
        a, b, a_scale, b_scale, bias = _fp8_operands(m, n, k, dtype, seed=m + n)
        for act in _ACTIVATIONS:
            out = torch.full((m, n), float("nan"), dtype=dtype, device="cuda")
            fn(out, a, b, a_scale, b_scale, bias, activation_id(act))
            _assert_fp8_close(out, reference_scaled_mm(a, b, a_scale, b_scale, bias, act, dtype))

    def test_config_space_never_splits_k(self):
        """The summation order is a property of the config space: no sliced K, no split."""
        cfgs = get_production_configs(fmt="fp8").values()
        assert cfgs
        for cfg in cfgs:
            assert cfg.fp8 and cfg.block_k == cfg.warp_k and not cfg.stream_k, cfg
            assert _epilogue_covers_warp(cfg.tile, element_bits=_EPILOGUE_ELEMENT_BITS), cfg

    @pytest.mark.parametrize("m,n,k", [(1000, 768, 2048), (300, 896, 18944)])
    def test_every_tile_computes_the_same_bits(self, m, n, k):
        a, b, a_scale, b_scale, bias = _fp8_operands(m, n, k, torch.float16)
        mod = _get_scaled_mm_module("fp8")
        outs = {}
        for name, cfg in get_production_configs(fmt="fp8").items():
            out = torch.empty(m, n, dtype=torch.float16, device="cuda")
            getattr(mod, scaled_mm_func_name(cfg))(out, a, b, a_scale, b_scale, bias, 3)
            outs[name] = out
        first = next(iter(outs.values()))
        assert all(torch.equal(o, first) for o in outs.values()), "a tile summed in another order"

    @pytest.mark.parametrize("n,k", [(768, 256), (256, 2048), (1024, 4096)])
    def test_rows_are_bitwise_independent_of_the_batch(self, n, k):
        """Every row computed in batches of 1, 16, 64 and 919 -- different tiles -- matches.

        fp16 and all 1000 rows: an order difference in the fp32 sums flips an fp16
        rounding in only ~1 output in 4000, so a few sampled rows would miss it.
        """
        a, b, a_scale, b_scale, bias = _fp8_operands(1000, n, k, torch.float16)
        cuts = (0, 1, 17, 81, 1000)
        assert len({select_config(hi - lo, n, k, fmt="fp8") for lo, hi in zip(cuts, cuts[1:])}) > 1
        full = oasr.scaled_mm(a, b, a_scale, b_scale, bias, "swish")
        parts = [
            oasr.scaled_mm(
                a[lo:hi].contiguous(), b, a_scale[lo:hi].contiguous(), b_scale, bias, "swish"
            )
            for lo, hi in zip(cuts, cuts[1:])
        ]
        assert torch.equal(torch.cat(parts), full)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_quantizing_epilogue_is_the_unfused_pair(self, dtype):
        """Every tile's fp8-output epilogue == ``quantize_static(scaled_mm(...))``, bit for bit."""
        m, n, k = 333, 1024, 512
        a, b, a_scale, b_scale, bias = _fp8_operands(m, n, k, dtype)
        out_mul = static_multiplier(torch.tensor([0.05], device="cuda"), None, n)
        code = 0 if dtype == torch.float16 else 1
        mod = _get_scaled_mm_module("fp8")
        y = oasr.scaled_mm(a, b, a_scale, b_scale, bias, "swish", out_dtype=dtype)
        unfused = oasr.quantize_static(y, out_mul, dtype=_FP8)
        for cfg in get_production_configs(fmt="fp8").values():
            q = torch.empty(m, n, dtype=_FP8, device="cuda")
            getattr(mod, scaled_mm_quant_func_name(cfg))(
                q, a, b, a_scale, b_scale, bias, out_mul, activation_id("swish"), code
            )
            assert torch.equal(_bytes(q), _bytes(unfused)), cfg.compile_name
        ref = reference_scaled_mm_quant(a, b, a_scale, b_scale, bias, "swish", out_mul, dtype)
        # The reference GEMM is exact, the kernel's fp32: within one e4m3 step.
        diff = (q.float() - ref.float()).abs()
        assert bool((diff <= ref.float().abs() * 2.0**-3 + 2.0**-9).all())

    def test_formats_cannot_be_mixed(self):
        a, b, a_scale, b_scale, _ = _fp8_operands(8, 256, 256, torch.float16)
        b_int8 = torch.zeros(256, 256, dtype=torch.int8, device="cuda")
        with pytest.raises(RuntimeError, match="A and B must be float8_e4m3fn"):
            oasr.scaled_mm(a, b_int8, a_scale, b_scale)

    def test_graph_replay_recomputes(self):
        a, b, a_scale, b_scale, bias = _fp8_operands(256, 512, 256, torch.float16)
        out = torch.empty(256, 512, dtype=torch.float16, device="cuda")

        def launch():
            return oasr.scaled_mm(a, b, a_scale, b_scale, bias, "relu", out=out)

        def mutate():
            a.copy_((-a.float()).to(_FP8))
            a_scale.mul_(2.0)

        assert_graph_replay(
            launch,
            out=out,
            mutate=mutate,
            expected=lambda: reference_scaled_mm(a, b, a_scale, b_scale, bias, "relu"),
            rtol=2.0**-8,
            atol=2e-3,
        )
