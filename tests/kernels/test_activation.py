#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Pointwise activations (``oasr/functionals/activation.py``) against torch.

Every op here is elementwise, which decides the grid: the axes that reach a
different code path are the **last-dim alignment** (vec4 vs the scalar tail)
and the **layout** the wrapper is handed -- not the batch or sequence extents.
So one parametrization walks the ops and each test covers both alignments in
its body, rather than multiplying the node count by shapes that share a branch.

Two groups, because they do not accept the same inputs. The generic ops take
any rank and any layout; ``swoosh_l`` / ``swoosh_r`` want rank >= 2 and have no
unaligned-pointer path, so they get their own tests instead of being folded
into a grid that would assert a surface they never claimed.

``oasr.gelu`` is the exact-erf one. The *fused* GELU epilogues elsewhere are
the tanh approximation, and :class:`TestGeluIsExactErf` is what keeps the two
from being quietly swapped.
"""

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_dest_passing

import oasr

pytestmark = pytest.mark.cuda


def _ref_swoosh_l(x: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.logaddexp(zero, x - 4.0) - 0.08 * x - 0.035


def _ref_swoosh_r(x: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687


#: Ops sharing one signature and the generic launcher.
OPS = {
    "gelu": (oasr.gelu, F.gelu),
    "sigmoid": (oasr.sigmoid, torch.sigmoid),
    "tanh": (oasr.tanh, torch.tanh),
    "relu": (oasr.relu, torch.relu),
    "swish": (oasr.swish, F.silu),
}
NAMES = list(OPS)
#: The subset whose wrapper has the awkward-layout branches.
LAYOUT_OPS = ["gelu", "sigmoid", "tanh", "relu"]
SWOOSH = {"swoosh_l": (oasr.swoosh_l, _ref_swoosh_l), "swoosh_r": (oasr.swoosh_r, _ref_swoosh_r)}

#: One shape whose last dim is vec4-aligned and one whose is not: the two
#: launcher paths. Anything else is the same kernel over more elements.
_ALIGNED = (2, 128, 256)
_SCALAR_TAIL = (2, 17, 63)


def _atol(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return 2e-6
    return 2e-2 if dtype == torch.bfloat16 else 2e-3


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_matches_reference(name, dtype):
    fn, ref = OPS[name]
    for shape in (_ALIGNED, _SCALAR_TAIL):
        x = torch.randn(*shape, device="cuda", dtype=dtype)
        torch.testing.assert_close(fn(x), ref(x), rtol=1e-5, atol=_atol(dtype))


@pytest.mark.parametrize("name", NAMES)
def test_destination_passing(name):
    """Rule 4, for every op that takes an ``out=``."""
    fn, ref = OPS[name]
    x = torch.randn(2, 31, 65, device="cuda", dtype=torch.float16)
    assert_dest_passing(fn, x, out=torch.empty_like(x))
    torch.testing.assert_close(fn(x), ref(x), rtol=1e-5, atol=2e-3)


@pytest.mark.parametrize("name", LAYOUT_OPS)
@pytest.mark.parametrize("layout", ["transposed", "unaligned_offset", "row_strided_chunk"])
def test_awkward_layouts(name, layout):
    """The three layouts the generic wrapper has a branch for."""
    fn, ref = OPS[name]
    if layout == "transposed":
        x = torch.randn(2, 31, 65, device="cuda", dtype=torch.float16).transpose(1, 2)
        assert not x.is_contiguous()
    elif layout == "unaligned_offset":
        base = torch.randn(4097, device="cuda", dtype=torch.float16)
        x = base[1:]
        assert x.is_contiguous() and x.data_ptr() % 16 != 0
    else:
        base = torch.randn(17, 3, 3 * 64, device="cuda", dtype=torch.float16)
        x = base.chunk(3, dim=-1)[1]
        assert x.stride() == (576, 192, 1) and not x.is_contiguous()
        assert fn(x).is_contiguous()
    torch.testing.assert_close(fn(x), ref(x), rtol=1e-5, atol=2e-3)


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "relu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_special_values(name, dtype):
    """NaN / +-inf / signed zero must survive with torch's own semantics."""
    fn, ref = OPS[name]
    x = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0, 0.0], device="cuda", dtype=dtype
    )
    got, expected = fn(x), ref(x)
    torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(got), torch.signbit(expected))


def test_saturation_is_stable_over_a_wide_range():
    """The sigmoids must not wrap at the ends of the representable range."""
    x = torch.linspace(-80.0, 80.0, 4096, device="cuda")
    torch.testing.assert_close(oasr.sigmoid(x), torch.sigmoid(x), rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(oasr.tanh(x), torch.tanh(x), rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "relu"])
def test_empty_input_is_a_shape_preserving_noop(name):
    """Only the unary table claims a zero-element input; the others are not asked."""
    x = torch.empty(0, 8, device="cuda", dtype=torch.float16)
    assert OPS[name][0](x).shape == x.shape


class TestGeluIsExactErf:
    """``oasr.gelu`` is erf, not the tanh approximation.

    Fusing the tanh approximation under the exact-erf name is a silent
    accuracy change, so the difference is asserted rather than assumed: over a
    dense sweep the two must actually disagree somewhere.
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_reaches_the_served_dtype_launcher_and_uses_erf(self, dtype):
        x = torch.linspace(-5.0, 5.0, 4096, device="cuda", dtype=dtype)
        output = oasr.gelu(x)
        torch.testing.assert_close(output, F.gelu(x), rtol=0, atol=2e-3)
        assert torch.count_nonzero(output != F.gelu(x, approximate="tanh")).item() > 0


class TestSwoosh:
    """Zipformer's two softplus activations. Rank >= 2, no unaligned path."""

    @pytest.mark.parametrize("name", list(SWOOSH))
    @pytest.mark.parametrize(
        "shape",
        [
            (2, 128, 256),  # vec4-aligned last dim
            (2, 128, 255),  # non-vec-aligned last dim -> scalar path
            (2, 8, 50, 19),  # 4-D, conv-output-like
        ],
    )
    def test_fp32(self, name, shape):
        fn, ref = SWOOSH[name]
        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(fn(x), ref(x), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("name", list(SWOOSH))
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half(self, name, dtype):
        fn, ref = SWOOSH[name]
        x = torch.randn(4, 200, 512, device="cuda", dtype=dtype)
        torch.testing.assert_close(fn(x), ref(x), rtol=2e-2, atol=2e-2)

    def test_large_magnitude(self):
        """Softplus must not overflow over a wide input range."""
        x = torch.linspace(-60.0, 60.0, 4096, device="cuda", dtype=torch.float32).reshape(1, 64, 64)
        torch.testing.assert_close(oasr.swoosh_l(x), _ref_swoosh_l(x), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(oasr.swoosh_r(x), _ref_swoosh_r(x), rtol=1e-5, atol=1e-5)

    def test_noncontiguous(self):
        """A transposed input is handled by an internal ``.contiguous()``."""
        x = torch.randn(2, 256, 128, device="cuda", dtype=torch.float32).transpose(1, 2)
        assert not x.is_contiguous()
        torch.testing.assert_close(oasr.swoosh_l(x), _ref_swoosh_l(x), rtol=1e-5, atol=1e-5)

    def test_destination_passing(self):
        x = torch.randn(2, 128, 256, device="cuda", dtype=torch.float32)
        assert_dest_passing(oasr.swoosh_r, x, out=torch.empty_like(x), expected=_ref_swoosh_r(x))


class TestGlu:
    """``oasr.glu`` halves the last dim, so it does not share the OPS signature."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_torch(self, dtype):
        x = torch.randn(2, 128, 512, device="cuda", dtype=dtype)
        torch.testing.assert_close(oasr.glu(x), F.glu(x, dim=-1).to(dtype), rtol=1e-2, atol=1e-2)

    def test_destination_passing(self):
        x = torch.randn(2, 128, 512, device="cuda", dtype=torch.float16)
        out = torch.empty(2, 128, 256, device="cuda", dtype=torch.float16)
        assert_dest_passing(oasr.glu, x, out=out, expected=F.glu(x, dim=-1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
