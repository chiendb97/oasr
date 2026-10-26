# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The waist's quantization seam: ``QuantizableGemm`` and an installed method.

What a quantized layer must keep doing that a full-precision one does:

* expose a pinned state dict (``weight`` int8 in the layer's own shape,
  ``weight_scale`` fp32, ``smooth_inv`` only when runtime-smoothed);
* survive ``.to(dtype)`` / ``.half()`` / ``.cuda()`` — moved by device, never cast;
* run on CPU (the reference path) and agree with the kernel on CUDA in the
  served dtypes;
* take its own branch before the fp row floor (``GEMM_MIN_ROWS``): there is no fp
  weight to fall back to.

And per architecture: the declared layout resolves, and quantizing every
quantizable layer still builds — the layers a fused kernel reads raw are exactly
the ones marked ``quantizable = False``.
"""

from __future__ import annotations

import fnmatch

import pytest
import torch
from test_layer_waist import _build, _tiny_configs

from oasr.layers import (
    Linear,
    LinearActivation,
    PointwiseConv1d,
    QuantizableConv,
    QuantizableGemm,
)
from oasr.models.registry import list_models
from oasr.quantization import (
    INT8_W8A8,
    INT8_W8A8_STATIC,
    QuantGroup,
    QuantizationSpec,
    apply_quantization,
    bind_quantization,
    quantizable_layers,
)


def _quantize(layer: torch.nn.Module, *, smoothed: bool = False) -> torch.nn.Module:
    """Wrap *layer* in a one-layer model and quantize it online."""
    holder = torch.nn.Module()
    holder.layer = layer
    spec = QuantizationSpec(
        (QuantGroup(INT8_W8A8, ("layer",)),),
        runtime_smoothed=("layer",) if smoothed else (),
    )
    apply_quantization(holder, spec, quantize=True)
    bind_quantization(holder)
    return layer


def _layers():
    torch.manual_seed(0)
    return {
        "linear": Linear(64, 96),
        "linear_activation": LinearActivation(64, 96, activation_type="swish"),
        "pointwise_conv": PointwiseConv1d(64, 96),
    }


def _everything(model):
    """A spec quantizing every quantizable layer an 8-bit kernel serves: the GEMMs
    per token, the convolutions with a static input scale (one, uncalibrated).
    """
    layers = quantizable_layers(model)
    gemms = tuple(
        p
        for p, m in layers.items()
        if isinstance(m, QuantizableGemm) and m.quant_unsupported() is None
    )
    convs = tuple(
        p
        for p, m in layers.items()
        if isinstance(m, QuantizableConv) and m.quant_unsupported() is None
    )
    groups = [QuantGroup(INT8_W8A8, gemms)]
    if convs:
        groups.insert(0, QuantGroup(INT8_W8A8_STATIC, convs))
    return QuantizationSpec(tuple(groups)), len(gemms) + len(convs)


class TestStorage:
    @pytest.mark.parametrize("kind", ["linear", "linear_activation", "pointwise_conv"])
    def test_state_dict_is_pinned(self, kind):
        layer = _layers()[kind]
        shape = tuple(layer.weight.shape)
        _quantize(layer)
        sd = layer.state_dict()
        assert set(sd) == {"weight", "bias", "weight_scale"}
        assert sd["weight"].dtype == torch.int8 and tuple(sd["weight"].shape) == shape
        assert sd["weight_scale"].dtype == torch.float32 and sd["weight_scale"].shape == (96,)
        assert sd["bias"].dtype == torch.float32

    def test_runtime_smoother_is_a_buffer(self):
        layer = _quantize(_layers()["linear"], smoothed=True)
        assert layer.state_dict()["smooth_inv"].dtype == torch.float32
        assert layer.smooth_inv.shape == (64,)

    def test_weight_quantization_round_trips(self):
        layer = _layers()["linear"]
        w = layer.weight.detach().clone()
        _quantize(layer)
        deq = layer.weight.float() * layer.weight_scale[:, None]
        assert torch.allclose(deq, w, atol=float(layer.weight_scale.max()) / 2 + 1e-7)

    def test_the_method_is_not_a_module(self):
        layer = _quantize(_layers()["linear"])
        assert [name for name, _ in layer.named_modules()] == [""]

    @pytest.mark.parametrize("cast", ["to_bf16", "half", "double"])
    def test_dtype_casts_leave_quantized_storage_alone(self, cast):
        layer = _quantize(_layers()["pointwise_conv"], smoothed=True)
        before = layer.weight.detach().clone(), layer.weight_scale.clone()
        {"to_bf16": lambda: layer.to(torch.bfloat16), "half": layer.half, "double": layer.double}[
            cast
        ]()
        assert layer.weight.dtype == torch.int8 and torch.equal(layer.weight, before[0])
        assert layer.weight_scale.dtype == torch.float32
        assert torch.equal(layer.weight_scale, before[1])
        assert layer.smooth_inv.dtype == torch.float32
        assert layer.bias.dtype != torch.float32  # the fp parameters did cast

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_device_moves_carry_quantized_storage(self):
        layer = _quantize(_layers()["linear"]).cuda().half()
        assert layer.weight.device.type == "cuda" and layer.weight.dtype == torch.int8
        assert layer.weight_scale.device.type == "cuda"
        assert layer.weight_scale.dtype == torch.float32


class TestForward:
    @pytest.mark.parametrize("kind", ["linear", "linear_activation", "pointwise_conv"])
    def test_runs_on_cpu_through_the_reference(self, kind):
        layer = _layers()[kind]
        x = torch.randn(3, 7, 64)
        with torch.no_grad():
            want = layer(x)
            _quantize(layer)
            got = layer(x)
        assert got.shape == want.shape and got.dtype == x.dtype
        assert layer.quant_method.lane is None
        assert torch.allclose(got, want, atol=0.05, rtol=0.05)

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("kind", ["linear", "linear_activation", "pointwise_conv"])
    @pytest.mark.parametrize("rows", [1, 300])
    def test_kernel_agrees_with_the_reference(self, kind, dtype, rows):
        layer = _layers()[kind]
        _quantize(layer, smoothed=True)
        layer.smooth_inv.uniform_(0.5, 2.0)
        layer = layer.cuda().to(dtype)
        bind_quantization(layer)
        assert layer.quant_method.lane is not None, layer.quant_method.lane_note
        x = torch.randn(rows, 64, device="cuda").to(dtype)
        with torch.no_grad():
            got = layer(x)
            ref = layer.quant_method.apply_reference(layer, x)
        rel = 2.0**-9 if dtype == torch.float16 else 2.0**-6
        assert bool(((got.float() - ref.float()).abs() <= ref.float().abs() * rel + 1e-3).all())

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_a_single_row_takes_the_kernel_not_the_row_floor(self, monkeypatch):
        """M=1 is below GEMM_MIN_ROWS; an fp layer would take F.linear there."""
        layer = _quantize(_layers()["linear"]).cuda().half()
        bind_quantization(layer)
        calls = []
        lane = layer.quant_method.lane
        original = type(lane).run
        monkeypatch.setattr(
            type(lane), "run", lambda self, *a: calls.append(1) or original(self, *a)
        )
        with torch.no_grad():
            layer(torch.randn(1, 64, device="cuda").half())
        assert calls == [1]


class TestArchitectures:
    @pytest.mark.parametrize("arch", list_models())
    def test_declared_layout_resolves(self, arch):
        model = _build(arch)
        modules = dict(model.named_modules())
        layout = model.quantization_layout()
        for group in layout.groups:
            for path in group.consumers:
                assert isinstance(modules.get(path), QuantizableGemm), f"{arch}: {path}"
            if group.producer is not None:
                assert group.producer in modules, f"{arch}: {group.producer}"
        for chain in layout.chains:
            for path in (chain.producer, chain.consumer):
                assert isinstance(modules.get(path), QuantizableGemm), f"{arch}: {path}"
        for glob in layout.batch_invariant:
            assert any(fnmatch.fnmatchcase(p, glob) for p in modules), f"{arch}: {glob}"

    @pytest.mark.parametrize("arch", list_models())
    def test_everything_quantizable_quantizes(self, arch):
        model = _build(arch)
        spec, n = _everything(model)
        assert n, f"{arch} has no quantizable layer"
        assert len(apply_quantization(model, spec, quantize=True)) == n

    @pytest.mark.parametrize("arch", ["conformer", "whisper", "paraformer"])
    def test_a_fully_quantized_encoder_runs_on_cpu(self, arch):
        model = _build(arch).eval()
        apply_quantization(model, _everything(model)[0], quantize=True)
        bind_quantization(model)
        with torch.no_grad():
            hidden, _ = model.encode_offline(torch.randn(1, 64, 80), torch.tensor([64]))
        assert torch.isfinite(hidden).all()

    def test_raw_weight_readers_are_not_quantizable(self):
        """The layers a fused kernel reads raw are exactly the ones withheld.

        The gated MLP's gate/up are *not* among them: a quantized projection
        takes the block off the fused kernel (``GatedMLP._forward_unfused``).
        """
        llm = quantizable_layers(_build("speech_llm"))
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert any(p.endswith(name) for p in llm), name
        joiner = {p for p in quantizable_layers(_build("transducer")) if p.startswith("joiner.")}
        assert joiner == {"joiner.encoder_proj"}

    def test_a_spec_selecting_a_raw_read_layer_is_refused(self):
        model = _build("transducer")
        spec = QuantizationSpec((QuantGroup(INT8_W8A8, ("joiner.output_linear",)),))
        with pytest.raises(ValueError, match="read raw"):
            apply_quantization(model, spec, quantize=True)

    def test_conformer_keeps_its_declared_layers_full_precision(self):
        model = _build("conformer")
        spec = QuantizationSpec.for_model(model, INT8_W8A8)
        targets = set(spec.groups[0].targets)
        assert not any(t.startswith(("encoder.embed.", "ctc.")) for t in targets)
        assert not any(t.endswith(("linear_pos", "output_layer")) for t in targets)
        cfg = _tiny_configs()["conformer"]
        dec_layers = cfg.decoder.num_blocks + cfg.decoder.r_num_blocks
        # Encoder: qkv, out, 2x(w_1, w_2), pw1, pw2.  U2++ decoder: 2x(q, k, v,
        # out) + w_1, w_2 per layer, both directions; the vocab heads stay fp.
        assert len(targets) == 8 * cfg.encoder.num_blocks + 10 * dec_layers
