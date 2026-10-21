# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr.layers.LayerNorm.quantize_output``: a norm that hands its int8 GEMM
consumer a pre-quantized activation.

Armed, ``forward`` / ``forward_add_residual`` return a
:class:`~oasr.layers.QuantizedActivation` whose ``data`` / ``scale`` are
**bit-identical** to ``quantize_per_token`` of the plain output -- the consumer
cannot tell the fused launch from the pair it replaces.  Everywhere the fused
kernel does not apply (CPU, fp32, the torch backend, a non-contiguous input) the
layer returns exactly what it returns unarmed, which every such consumer accepts.
"""

from __future__ import annotations

import pytest
import torch

from oasr.functionals.quant import quantize_per_token
from oasr.layers import LayerNorm, QuantizedActivation, layers_backend_override

HIDDEN = 256

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _pair(device: str, dtype: torch.dtype, seed: int = 0):
    """An armed and an unarmed LayerNorm with the same non-trivial affine."""
    torch.manual_seed(seed)
    plain = LayerNorm(HIDDEN, eps=1e-5).to(device=device, dtype=dtype)
    with torch.no_grad():
        plain.weight.copy_(torch.randn(HIDDEN) * torch.linspace(0.1, 6.0, HIDDEN))
        plain.bias.copy_(torch.randn(HIDDEN))
    armed = LayerNorm(HIDDEN, eps=1e-5).to(device=device, dtype=dtype)
    armed.load_state_dict(plain.state_dict())
    armed.quantize_output = True
    return plain, armed


def _x(device: str, dtype: torch.dtype, shape=(4, 37, HIDDEN), seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 3.0 + 0.5).to(device=device, dtype=dtype)


def test_flag_is_off_by_default_and_per_instance():
    assert LayerNorm.quantize_output is False
    a, b = LayerNorm(HIDDEN), LayerNorm(HIDDEN)
    a.quantize_output = True
    assert b.quantize_output is False and LayerNorm.quantize_output is False


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestArmedOnCuda:
    def test_forward_returns_the_quantized_plain_output(self, dtype):
        plain, armed = _pair("cuda", dtype)
        x = _x("cuda", dtype)
        with torch.no_grad():
            got = armed(x)
            want_q, want_scale = quantize_per_token(plain(x))
        assert isinstance(got, QuantizedActivation)
        assert got.dtype == dtype and got.shape == x.shape and got.per_token
        assert got.data.dtype == torch.int8 and got.scale.shape == x.shape[:-1]
        assert torch.equal(got.data, want_q) and torch.equal(got.scale, want_scale)

    @pytest.mark.parametrize("alpha", [1.0, 0.5])
    def test_forward_add_residual(self, dtype, alpha):
        plain, armed = _pair("cuda", dtype)
        x, residual = _x("cuda", dtype, seed=2), _x("cuda", dtype, seed=3)
        with torch.no_grad():
            got, summed = armed.forward_add_residual(x, residual, alpha)
            want, want_summed = plain.forward_add_residual(x, residual, alpha)
            want_q, want_scale = quantize_per_token(want)
        assert isinstance(got, QuantizedActivation) and got.dtype == dtype
        assert torch.equal(got.data, want_q) and torch.equal(got.scale, want_scale)
        assert isinstance(summed, torch.Tensor) and torch.equal(summed, want_summed)

    def test_forward_add_is_untouched(self, dtype):
        """Only ``forward`` and ``forward_add_residual`` honour the flag."""
        plain, armed = _pair("cuda", dtype)
        x, residual = _x("cuda", dtype, seed=4), _x("cuda", dtype, seed=5)
        with torch.no_grad():
            got = armed.forward_add(x, residual, 0.5)
            assert isinstance(got, torch.Tensor)
            assert torch.equal(got, plain.forward_add(x, residual, 0.5))

    def test_non_contiguous_input_returns_the_plain_tensor(self, dtype):
        """A permuted dense view reaches the norm kernel but not the fused one."""
        plain, armed = _pair("cuda", dtype)
        x = _x("cuda", dtype, shape=(37, 4, HIDDEN)).transpose(0, 1)
        residual = _x("cuda", dtype, shape=(37, 4, HIDDEN), seed=6).transpose(0, 1)
        assert not x.is_contiguous()
        with torch.no_grad():
            got = armed(x)
            assert isinstance(got, torch.Tensor) and torch.equal(got, plain(x))
            got, summed = armed.forward_add_residual(x, residual)
            want, want_summed = plain.forward_add_residual(x, residual)
        assert isinstance(got, torch.Tensor)
        assert torch.equal(got, want) and torch.equal(summed, want_summed)

    def test_residual_in_another_layout_is_refused_as_unarmed(self, dtype):
        """A residual laid out unlike the input skips the fused kernel and meets
        the plain launcher's own refusal -- the armed layer adds no new outcome."""
        plain, armed = _pair("cuda", dtype)
        x = _x("cuda", dtype, shape=(4, 37, HIDDEN), seed=7)
        residual = _x("cuda", dtype, shape=(37, 4, HIDDEN), seed=8).transpose(0, 1)
        for norm in (plain, armed):
            with torch.no_grad(), pytest.raises(RuntimeError, match="shape and strides"):
                norm.forward_add_residual(x, residual)

    def test_torch_backend_returns_the_plain_tensor(self, dtype):
        plain, armed = _pair("cuda", dtype)
        x, residual = _x("cuda", dtype, seed=9), _x("cuda", dtype, seed=10)
        with torch.no_grad(), layers_backend_override("torch"):
            got = armed(x)
            assert isinstance(got, torch.Tensor) and torch.equal(got, plain(x))
            got, summed = armed.forward_add_residual(x, residual)
            want, want_summed = plain.forward_add_residual(x, residual)
        assert isinstance(got, torch.Tensor)
        assert torch.equal(got, want) and torch.equal(summed, want_summed)


@needs_cuda
def test_fp32_on_cuda_returns_the_plain_tensor():
    plain, armed = _pair("cuda", torch.float32)
    x, residual = _x("cuda", torch.float32), _x("cuda", torch.float32, seed=11)
    with torch.no_grad():
        got = armed(x)
        assert isinstance(got, torch.Tensor) and torch.equal(got, plain(x))
        got, summed = armed.forward_add_residual(x, residual)
        want, want_summed = plain.forward_add_residual(x, residual)
    assert isinstance(got, torch.Tensor)
    assert torch.equal(got, want) and torch.equal(summed, want_summed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpu_returns_the_plain_tensor(dtype):
    plain, armed = _pair("cpu", dtype)
    x, residual = _x("cpu", dtype, seed=12), _x("cpu", dtype, seed=13)
    with torch.no_grad():
        got = armed(x)
        assert isinstance(got, torch.Tensor) and torch.equal(got, plain(x))
        got, summed = armed.forward_add_residual(x, residual, 0.5)
        want, want_summed = plain.forward_add_residual(x, residual, 0.5)
    assert isinstance(got, torch.Tensor)
    assert torch.equal(got, want) and torch.equal(summed, want_summed)
