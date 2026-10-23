#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""LayerNorm with a per-token 8-bit output (``oasr/functionals/norm.py``).

The contract is **bit-exactness with the unfused pair** -- ``torch.equal``, not
a tolerance::

    layer_norm_quant(x, ...)               == quantize_per_token(layer_norm(x, ...))
    add_layer_norm_residual_quant(x, r, ...) == (quantize_per_token(y), s)
        where (y, s) = add_layer_norm_residual(x, r, ...)

A one-ulp difference in a normalized value is a different int8 code for the
rows where it straddles a rounding boundary, and a different scale when it is
the row's amax -- the class of difference that has changed decoded tokens before
(AGENTS.md rule 11).  The first version of this kernel differed by exactly that:
nvcc contracted the norm's ``n * w + b`` into an FMA in the plain kernel and not
in the fused one, which a few-per-million element mismatch at rows=1000 caught.

The torch twins (``reference_*``) share the quantize numerics but not the
kernel's LayerNorm reduction, so they are held to one int8 step.

Grid: the widths reach the launcher's paths -- 8-aligned widths take the
vectorized kernel (one vector per thread, held in registers, up to 8192), ``100``
the scalar one, and an unaligned *pointer* the scalar one at an 8-aligned width.
Past 1024 vectors a row the extra vectors are staged in shared memory: ``12288``
(vector) and ``4100`` (scalar) stage some, ``32768`` stages more than 40 KiB and
takes the shared-memory opt-in.
"""

from __future__ import annotations

import pytest
import torch
from helpers import assert_dest_passing, assert_graph_replay

import oasr
from oasr.functionals.norm import (
    add_layer_norm_residual_quant,
    layer_norm_quant,
    reference_add_layer_norm_residual_quant,
    reference_layer_norm_quant,
)
from oasr.functionals.quant import INT8_QMAX, QUANT_EPS, quantize_per_token

pytestmark = pytest.mark.cuda

_DTYPES = (torch.float16, torch.bfloat16)
_HIDDEN = (256, 512, 1024, 2048, 100, 4096)
_ROWS = (1, 3, 1000)

#: Relative spacing of the served dtypes: a scale is ``amax / 127`` and ``amax``
#: is a value of T, so a twin's scale can sit one T ulp away.
_ULP = {torch.float16: 2.0**-10, torch.bfloat16: 2.0**-7}


def _inputs(rows: int, hidden: int, dtype: torch.dtype, *, seed: int, has_bias: bool = True):
    """Rows of mixed magnitude, a constant row, and a non-trivial affine.

    Row 0 is scaled by 300 (the variance pass in fp32 at large values); the last
    row is constant, so its normalized value is exactly 0 and its output is the
    bias alone -- with no bias, a zero row whose amax is the ``qeps`` floor.  The
    weight carries per-channel outliers, as trained LayerNorm weights do.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, hidden, generator=g, device="cuda") * 3.0 + 1.0
    x[0] *= 300.0
    if rows > 1:
        x[-1] = 2.5
    weight = torch.randn(hidden, generator=g, device="cuda")
    weight = weight * torch.linspace(0.1, 8.0, hidden, device="cuda")
    bias = torch.randn(hidden, generator=g, device="cuda") if has_bias else None
    residual = torch.randn(rows, hidden, generator=g, device="cuda")
    if rows > 1:
        # The add variant's constant row: residual + alpha * x == 2.5 + alpha * 2.5.
        residual[-1] = 2.5
    cast = lambda t: None if t is None else t.to(dtype)  # noqa: E731
    return cast(x), cast(residual), cast(weight), cast(bias)


def _ieee_div(a: float, b: float) -> torch.Tensor:
    """fp32 ``a / b`` correctly rounded, as the kernel's ``__fdiv_rn``.

    Tensor / tensor on purpose: torch lowers ``tensor / python_scalar`` to a
    multiply by the reciprocal, one ulp away.
    """
    return torch.tensor(a, device="cuda") / torch.tensor(float(b), device="cuda")


def _unfused(x, w, b, eps=1e-5, qeps=QUANT_EPS):
    return quantize_per_token(oasr.layer_norm(x, w, b, eps), None, qeps)


def _unfused_add(x, r, w, b, eps=1e-5, alpha=1.0, qeps=QUANT_EPS):
    y, s = oasr.add_layer_norm_residual(x, r, w, b, eps, alpha)
    q, scale = quantize_per_token(y, None, qeps)
    return q, scale, s


def _assert_equal(got, want, what: str) -> None:
    for name, a, b in zip(("q", "scale", "residual_out"), got, want):
        assert a.dtype == b.dtype and a.shape == b.shape, f"{what}: {name} {a.shape} vs {b.shape}"
        if not torch.equal(a, b):
            n = int((a != b).sum())
            raise AssertionError(f"{what}: {name} differs from the unfused pair in {n} elements")


class TestBitExactVsUnfused:
    @pytest.mark.parametrize("has_bias", [True, False])
    @pytest.mark.parametrize("rows", _ROWS)
    @pytest.mark.parametrize("hidden", _HIDDEN)
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_layer_norm_quant(self, dtype, hidden, rows, has_bias):
        x, _, w, b = _inputs(rows, hidden, dtype, seed=hidden * 7 + rows, has_bias=has_bias)
        got = layer_norm_quant(x, w, b, 1e-5)
        _assert_equal(got, _unfused(x, w, b), "layer_norm_quant")
        if rows > 1 and not has_bias:
            # The constant row normalizes to zeros: its amax is the qeps floor.
            assert not got[0][-1].any()
            assert got[1][-1] == _ieee_div(QUANT_EPS, INT8_QMAX)

    #: ``alpha`` scales the add and ``has_bias`` the affine's spelling; both
    #: reach the epilogue every width shares, so they are swept as two pairs.
    @pytest.mark.parametrize("alpha,has_bias", [(1.0, True), (0.5, False), (0.5, True)])
    @pytest.mark.parametrize("rows", _ROWS)
    @pytest.mark.parametrize("hidden", _HIDDEN)
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_add_layer_norm_residual_quant(self, dtype, hidden, rows, alpha, has_bias):
        x, r, w, b = _inputs(rows, hidden, dtype, seed=hidden * 11 + rows, has_bias=has_bias)
        got = add_layer_norm_residual_quant(x, r, w, b, 1e-5, alpha)
        _assert_equal(got, _unfused_add(x, r, w, b, 1e-5, alpha), "add_layer_norm_residual_quant")

    @pytest.mark.parametrize("op", ["layer_norm", "add_layer_norm_residual"])
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_volume_catches_a_one_ulp_drift(self, dtype, op):
        """A one-fp32-ulp drift in the normalized value (an FMA contracted in one
        instantiation and not the other) reaches the int8 output only where it
        flips a T rounding that sits on a quantization boundary: a few elements
        per million.  The grid above sees it in about one config in seventy;
        these 32M elements see it in both dtypes (seeded, so deterministically)."""
        x, r, w, b = _inputs(32768, 1024, dtype, seed=13)
        if op == "layer_norm":
            _assert_equal(layer_norm_quant(x, w, b), _unfused(x, w, b), "volume layer_norm_quant")
        else:
            _assert_equal(
                add_layer_norm_residual_quant(x, r, w, b, alpha=0.5),
                _unfused_add(x, r, w, b, alpha=0.5),
                "volume add_layer_norm_residual_quant",
            )

    #: Rows past one vector per thread: 12288 (vector path) and 4100 (scalar)
    #: stage the overflow in shared memory; 32768 stages 48 KiB, past the
    #: default dynamic limit, and takes the opt-in.
    @pytest.mark.parametrize("hidden", [12288, 4100, 32768])
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_wide_rows_stage_the_overflow(self, dtype, hidden):
        x, r, w, b = _inputs(3, hidden, dtype, seed=5)
        _assert_equal(layer_norm_quant(x, w, b), _unfused(x, w, b), "wide layer_norm_quant")
        _assert_equal(
            add_layer_norm_residual_quant(x, r, w, b, alpha=0.5),
            _unfused_add(x, r, w, b, alpha=0.5),
            "wide add_layer_norm_residual_quant",
        )

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_unaligned_input_takes_the_scalar_path(self, dtype):
        """An 8-aligned width whose *pointer* is not 16-byte aligned: LayerNorm's
        launcher falls back to the scalar kernel, and the fused one must too --
        the vector/scalar choice fixes the accumulation order of the mean."""
        rows, hidden = 64, 512
        x0, r0, w, b = _inputs(rows, hidden, dtype, seed=9)
        buf = torch.empty(rows * hidden + 1, device="cuda", dtype=dtype)
        x = buf[1:].view(rows, hidden)
        x.copy_(x0)
        assert x.is_contiguous() and x.data_ptr() % 16 != 0
        _assert_equal(layer_norm_quant(x, w, b), _unfused(x, w, b), "unaligned layer_norm_quant")
        _assert_equal(
            add_layer_norm_residual_quant(x, r0, w, b),
            _unfused_add(x, r0, w, b),
            "unaligned add_layer_norm_residual_quant",
        )

    def test_unaligned_int8_destination(self):
        """The int8 store is 8 bytes wide only when ``out`` allows it."""
        rows, hidden = 16, 256
        x, _, w, b = _inputs(rows, hidden, torch.float16, seed=3)
        buf = torch.empty(rows * hidden + 1, device="cuda", dtype=torch.int8)
        out = buf[1:].view(rows, hidden)
        q, scale = layer_norm_quant(x, w, b, out=out)
        assert q.data_ptr() == out.data_ptr()
        _assert_equal((q, scale), _unfused(x, w, b), "unaligned int8 out")

    def test_leading_dims_and_scale_shape(self):
        x, _, w, b = _inputs(6 * 40, 256, torch.bfloat16, seed=4)
        x = x.view(6, 40, 256)
        q, scale = layer_norm_quant(x, w, b)
        assert q.shape == x.shape and q.dtype == torch.int8
        assert scale.shape == (6, 40) and scale.dtype == torch.float32
        _assert_equal((q, scale), _unfused(x, w, b), "3-D layer_norm_quant")

    def test_no_rows(self):
        x = torch.empty(0, 256, device="cuda", dtype=torch.float16)
        w = torch.ones(256, device="cuda", dtype=torch.float16)
        q, scale = layer_norm_quant(x, w)
        assert q.shape == (0, 256) and scale.shape == (0,)
        q, scale, s = add_layer_norm_residual_quant(x, x, w)
        assert q.shape == (0, 256) and scale.shape == (0,) and s.shape == (0, 256)

    def test_qeps_is_the_amax_floor(self):
        x = torch.full((4, 256), 1.5, device="cuda", dtype=torch.float16)
        w = torch.ones(256, device="cuda", dtype=torch.float16)
        q, scale = layer_norm_quant(x, w, None, qeps=1e-3)
        assert not q.any()
        assert torch.equal(scale, torch.full_like(scale, _ieee_div(1e-3, INT8_QMAX)))


_FP8 = torch.float8_e4m3fn


class TestFp8Output:
    """The e4m3 format: the same kernels, only the final store differs.

    Bit-exact with ``quantize_per_token(..., dtype=float8_e4m3fn)`` of the plain
    norm, compared on the bytes (fp8 has no CUDA ``torch.equal``).  The widths
    reach the vector, scalar and shared-memory-staged paths.
    """

    @pytest.mark.parametrize("rows", (3, 1000))
    @pytest.mark.parametrize("hidden", (256, 100, 2048, 12288, 4100))
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_layer_norm_quant(self, dtype, hidden, rows):
        x, _, w, b = _inputs(rows, hidden, dtype, seed=hidden * 3 + rows)
        q, scale = layer_norm_quant(x, w, b, 1e-5, dtype=_FP8)
        q_ref, scale_ref = quantize_per_token(oasr.layer_norm(x, w, b, 1e-5), None, dtype=_FP8)
        assert q.dtype == _FP8
        assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
        assert torch.equal(scale, scale_ref)

    @pytest.mark.parametrize("hidden", (256, 100, 12288))
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_add_layer_norm_residual_quant(self, dtype, hidden):
        x, r, w, b = _inputs(1000, hidden, dtype, seed=hidden * 5)
        q, scale, s = add_layer_norm_residual_quant(x, r, w, b, 1e-5, 0.5, dtype=_FP8)
        y, s_ref = oasr.add_layer_norm_residual(x, r, w, b, 1e-5, 0.5)
        q_ref, scale_ref = quantize_per_token(y, None, dtype=_FP8)
        assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
        assert torch.equal(scale, scale_ref) and torch.equal(s, s_ref)

    def test_the_out_tensor_names_the_format(self):
        x, _, w, b = _inputs(8, 256, torch.float16, seed=1)
        out = torch.empty(8, 256, dtype=_FP8, device="cuda")
        q, _ = layer_norm_quant(x, w, b, 1e-5, out=out)
        assert q.data_ptr() == out.data_ptr()
        ref, _ = quantize_per_token(oasr.layer_norm(x, w, b, 1e-5), None, dtype=_FP8)
        assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))


class TestReferenceTwins:
    """The torch twins: same quantize, torch's LayerNorm reduction."""

    @pytest.mark.parametrize("hidden", _HIDDEN)
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_within_one_step(self, dtype, hidden):
        x, r, w, b = _inputs(1000, hidden, dtype, seed=hidden)
        for (q, s, *res), (qr, sr, *res_ref) in (
            (layer_norm_quant(x, w, b), reference_layer_norm_quant(x, w, b)),
            (
                add_layer_norm_residual_quant(x, r, w, b, alpha=0.5),
                reference_add_layer_norm_residual_quant(x, r, w, b, alpha=0.5),
            ),
        ):
            dq = (q.int() - qr.int()).abs()
            assert int(dq.max()) <= 1, "a normalized value one T ulp apart moves q by one step"
            assert float((dq > 0).float().mean()) < 1e-3
            assert bool(((s - sr).abs() <= sr * _ULP[dtype]).all())
            for a, e in zip(res, res_ref):
                # The carried residual is one rounding of the fp32 sum on both sides.
                assert torch.equal(a, e)

    def test_quantize_half_is_bit_exact(self):
        """Given the kernel's normalized values, the twin's quantize is exact."""
        from oasr.functionals.quant import reference_quantize_per_token

        x, _, w, b = _inputs(300, 1024, torch.bfloat16, seed=1)
        got = layer_norm_quant(x, w, b)
        _assert_equal(got, reference_quantize_per_token(oasr.layer_norm(x, w, b)), "twin quantize")

    def test_twins_run_on_cpu(self):
        x, r, w, b = _inputs(8, 256, torch.float16, seed=2)
        q, s = reference_layer_norm_quant(x.cpu(), w.cpu(), b.cpu())
        q_gpu, s_gpu = layer_norm_quant(x, w, b)
        assert q.device.type == "cpu" and q.dtype == torch.int8 and s.shape == (8,)
        assert int((q.int() - q_gpu.cpu().int()).abs().max()) <= 1
        q, s, res = reference_add_layer_norm_residual_quant(x.cpu(), r.cpu(), w.cpu(), b.cpu())
        assert res.dtype == torch.float16 and q.shape == x.shape


class TestContract:
    def test_destination_passing(self):
        x, r, w, b = _inputs(32, 256, torch.float16, seed=6)
        out = torch.empty(32, 256, dtype=torch.int8, device="cuda")
        scale = torch.empty(32, device="cuda")
        ret = assert_dest_passing(layer_norm_quant, x, w, b, out=out, outs=[scale], scale=scale)
        _assert_equal(ret, _unfused(x, w, b), "dest-passing layer_norm_quant")

        out = torch.empty(32, 256, dtype=torch.int8, device="cuda")
        scale = torch.empty(32, device="cuda")
        residual_out = torch.empty_like(x)
        ret = assert_dest_passing(
            add_layer_norm_residual_quant,
            x,
            r,
            w,
            b,
            out=out,
            outs=[scale, residual_out],
            scale=scale,
            residual_out=residual_out,
        )
        _assert_equal(ret, _unfused_add(x, r, w, b), "dest-passing add_layer_norm_residual_quant")

    def test_graph_replay_recomputes(self):
        x, r, w, b = _inputs(64, 512, torch.bfloat16, seed=7)
        out = torch.empty(64, 512, dtype=torch.int8, device="cuda")
        scale = torch.empty(64, device="cuda")

        def mutate():
            x.copy_(torch.randn_like(x) * 5.0 - 2.0)

        assert_graph_replay(
            lambda: layer_norm_quant(x, w, b, out=out, scale=scale),
            out=out,
            mutate=mutate,
            expected=lambda: _unfused(x, w, b)[0],
            rtol=0,
            atol=0,
        )
        assert torch.equal(scale, _unfused(x, w, b)[1])

    def test_add_graph_replay_recomputes(self):
        x, r, w, b = _inputs(64, 512, torch.float16, seed=8)
        out = torch.empty(64, 512, dtype=torch.int8, device="cuda")
        scale = torch.empty(64, device="cuda")
        residual_out = torch.empty_like(x)

        def mutate():
            x.copy_(torch.randn_like(x))
            r.copy_(torch.randn_like(r) * 4.0)

        assert_graph_replay(
            lambda: add_layer_norm_residual_quant(
                x, r, w, b, alpha=0.5, out=out, scale=scale, residual_out=residual_out
            ),
            out=out,
            mutate=mutate,
            expected=lambda: _unfused_add(x, r, w, b, alpha=0.5)[0],
            rtol=0,
            atol=0,
        )
        _, want_scale, want_res = _unfused_add(x, r, w, b, alpha=0.5)
        assert torch.equal(scale, want_scale) and torch.equal(residual_out, want_res)


class TestLauncherValidation:
    """What the launchers refuse, by message."""

    def _args(self, dtype=torch.float16):
        x, r, w, b = _inputs(4, 256, dtype, seed=0)
        return x, r, w, b

    def test_scale_needs_one_float_per_row(self):
        x, r, w, b = self._args()
        with pytest.raises(RuntimeError, match="scale needs one float per row"):
            layer_norm_quant(x, w, b, scale=torch.empty(3, device="cuda"))
        with pytest.raises(RuntimeError, match="scale needs one float per row"):
            add_layer_norm_residual_quant(x, r, w, b, scale=torch.empty(5, device="cuda"))

    def test_scale_must_be_float32(self):
        x, _, w, b = self._args()
        with pytest.raises(RuntimeError, match="scale must be float32"):
            layer_norm_quant(x, w, b, scale=torch.empty(4, device="cuda", dtype=torch.float16))

    def test_out_must_be_int8_shaped_like_input(self):
        x, _, w, b = self._args()
        with pytest.raises(RuntimeError, match="out must be int8"):
            layer_norm_quant(x, w, b, out=torch.empty(4, 256, device="cuda", dtype=torch.uint8))
        with pytest.raises(RuntimeError, match="out must be shaped like input"):
            layer_norm_quant(x, w, b, out=torch.empty(4, 128, device="cuda", dtype=torch.int8))

    def test_input_must_be_half_precision(self):
        x, _, w, b = self._args(torch.float32)
        with pytest.raises(RuntimeError, match="input must be float16 or bfloat16"):
            layer_norm_quant(x, w, b)

    def test_input_must_be_contiguous(self):
        """Even a permuted *dense* view: the int8 rows would land out of order."""
        x = torch.randn(2, 8, 256, device="cuda", dtype=torch.float16).transpose(0, 1)
        w = torch.ones(256, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="contiguous"):
            layer_norm_quant(x, w)

    def test_weight_and_bias_shape_and_dtype(self):
        x, _, w, b = self._args()
        with pytest.raises(RuntimeError, match="weight must be 1-D"):
            layer_norm_quant(x, w[:128], b)
        with pytest.raises(RuntimeError, match="weight must have the input dtype"):
            layer_norm_quant(x, w.float(), b)
        with pytest.raises(RuntimeError, match="bias must have the input dtype"):
            layer_norm_quant(x, w, b.to(torch.bfloat16))

    def test_residual_must_match_the_input(self):
        x, r, w, b = self._args()
        with pytest.raises(RuntimeError, match="residual and residual_out must have the input"):
            add_layer_norm_residual_quant(x, r.to(torch.bfloat16), w, b)
        with pytest.raises(RuntimeError):
            add_layer_norm_residual_quant(x, r[:, :128].contiguous(), w, b)
