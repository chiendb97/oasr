# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Quantized convolutions and the ``module_types`` selection.

A dense ``Conv1d`` / ``Conv2d`` is an implicit GEMM, and quantizes like one --
int8 weights per output channel -- except for its input scale: an output pixel
sums over a receptive field spanning several input pixels, so a per-token scale
would not factor out of the sum.  Its scale is static and calibrated, which is
why a convolution is selected only by name (``module_types``), only offline
(``oasr quantize``), and why the defaults quantize exactly the dense GEMMs they
always did.  Checked here:

* the kinds: every quantizable layer declares one, aliases expand, unknown
  names and an online request for a convolution are refused by name;
* the selection: default kinds leave convolutions alone, ``conv2d`` adds the
  dense int8-aligned ones with the static scheme and records why the others
  (a one-channel stem, a depthwise conv) were skipped;
* the method: storage keeps the layer's KRSC / KSC shape, survives dtype casts
  and a native round trip; the CPU reference tracks the fp32 convolution; on
  CUDA the kernel agrees with the reference inside the model.
"""

from __future__ import annotations

import pytest
import torch
from helpers import tiny_conformer, tiny_conformer_config

from oasr.engine.config import QuantizationConfig
from oasr.layers import (
    QUANT_KINDS,
    Conv1d,
    Conv1dActivation,
    Conv2d,
    Conv2dActivation,
    Linear,
    LinearActivation,
    PointwiseConv1d,
    QuantizableConv,
    QuantizableLayer,
)
from oasr.quantization import QuantizationSpec, apply_quantization, bind_quantization
from oasr.quantization.export import quantize_model
from oasr.quantization.methods.base import method_for
from oasr.quantization.methods.w8a8 import W8A8ConvMethod, W8A8Method
from oasr.quantization.recipes import get_recipe
from oasr.quantization.spec import (
    DEFAULT_MODULE_TYPES,
    FP8_W8A8,
    INT8_W8A8,
    INT8_W8A8_STATIC,
    resolve_module_types,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the int8 kernels are CUDA")

CONV = "encoder.embed.convs.1"  # the tiny Conformer's dense subsampling conv (32 -> 32)
STEM = "encoder.embed.convs.0"  # one input channel: no int8 kernel


def _feats():
    g = torch.Generator().manual_seed(3)
    return torch.randn(2, 60, 80, generator=g), torch.tensor([60, 33])


def _loop(model):
    with torch.no_grad():
        model.encode_offline(*_feats())


def _calibrated(layer: QuantizableLayer, x: torch.Tensor, **kwargs) -> QuantizableLayer:
    """*layer* quantized with a static input scale measured on *x*."""
    amax = x.detach().abs().amax().clamp_min(1e-6).float()
    layer.register_buffer("input_scale", (amax / 127.0).reshape(1))
    spec = QuantizationSpec.for_model(_Wrap(layer), INT8_W8A8, module_types=("conv",))
    apply_quantization(_Wrap(layer), spec, quantize=True)
    return layer


class _Wrap(torch.nn.Module):
    """A one-layer model, so a spec can name it."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer


# -----------------------------------------------------------------------------
# Kinds and module_types
# -----------------------------------------------------------------------------


class TestKinds:
    def test_every_quantizable_layer_declares_a_known_kind(self):
        layers = {
            Linear(32, 32): "linear",
            LinearActivation(32, 32): "linear",
            PointwiseConv1d(32, 32): "pointwise_conv",
            Conv1d(32, 32, 3): "conv1d",
            Conv1dActivation(32, 32, 3): "conv1d",
            Conv2d(32, 32, 3): "conv2d",
            Conv2dActivation(32, 32, 3, activation_type="relu"): "conv2d",
        }
        for layer, kind in layers.items():
            assert isinstance(layer, QuantizableLayer) and layer.quant_kind == kind
            assert kind in QUANT_KINDS

    def test_a_convolution_says_why_it_has_no_int8_kernel(self):
        assert Conv2d(32, 32, 3).quant_unsupported() is None
        assert Conv1d(80, 384, 3).quant_unsupported() is None
        assert "groups=32" in Conv2d(32, 32, 3, groups=32).quant_unsupported()
        assert "in_channels=1" in Conv2d(1, 32, 3).quant_unsupported()
        assert "out_channels=12" in Conv2d(32, 12, 3).quant_unsupported()

    def test_module_types_resolve(self):
        assert resolve_module_types(None) == DEFAULT_MODULE_TYPES == ("linear", "pointwise_conv")
        assert resolve_module_types("conv2d,linear") == ("linear", "conv2d")
        assert resolve_module_types(["conv"]) == ("pointwise_conv", "conv1d", "conv2d")
        assert resolve_module_types(["all"]) == QUANT_KINDS
        with pytest.raises(ValueError, match="unknown module type 'conv3d'"):
            resolve_module_types(["conv3d"])
        with pytest.raises(ValueError, match="selects no layer kind"):
            resolve_module_types([" "])

    def test_the_config_takes_module_types_as_the_server_sends_them(self):
        cfg = QuantizationConfig.coerce(
            {"scheme": "int8_w8a8", "module_types": "linear, pointwise_conv"}
        )
        assert cfg.module_types == ("linear", "pointwise_conv")
        with pytest.raises(ValueError, match="unknown module type"):
            QuantizationConfig(scheme="int8_w8a8", module_types=("dense",))
        with pytest.raises(ValueError, match="needs a scheme"):
            QuantizationConfig(module_types=("linear",))

    def test_online_quantization_refuses_a_convolution(self):
        from oasr.models.registry import _online_quantization_spec

        cfg = QuantizationConfig(scheme="int8_w8a8", module_types=("linear", "conv2d"))
        with pytest.raises(ValueError, match="oasr quantize --module-types linear,conv2d"):
            _online_quantization_spec(tiny_conformer(), cfg)
        # A pointwise conv is a GEMM: per-token, online, fine.
        pw = QuantizationConfig(scheme="int8_w8a8", module_types=("pointwise_conv",))
        spec = _online_quantization_spec(tiny_conformer(), pw)
        assert all(
            t.endswith(("pointwise_conv1", "pointwise_conv2")) for t in spec.groups[0].targets
        )


# -----------------------------------------------------------------------------
# Selection
# -----------------------------------------------------------------------------


class TestSelection:
    def test_the_default_leaves_every_convolution_alone(self):
        spec = QuantizationSpec.for_model(tiny_conformer(), INT8_W8A8)
        targets = [t for g in spec.groups for t in g.targets]
        assert CONV not in targets and "module_types" not in spec.producer
        assert any(t.endswith("pointwise_conv1") for t in targets)

    def test_conv2d_adds_the_dense_convolution_with_the_static_scheme(self):
        spec = QuantizationSpec.for_model(
            tiny_conformer(), INT8_W8A8, module_types=("linear", "pointwise_conv", "conv2d")
        )
        assert spec.resolve(CONV) == INT8_W8A8_STATIC
        assert spec.resolve(STEM) is None
        assert "in_channels=1" in spec.producer["skipped"][STEM]
        assert spec.producer["module_types"] == ["linear", "pointwise_conv", "conv2d"]
        # The layout's own full-precision layers stay so.
        assert spec.resolve("encoder.embed.out.0") is None

    def test_a_kind_list_without_linear_quantizes_no_linear(self):
        spec = QuantizationSpec.for_model(tiny_conformer(), INT8_W8A8, module_types=("conv2d",))
        assert [t for g in spec.groups for t in g.targets] == [CONV]

    def test_fp8_has_no_convolution_kernel(self):
        with pytest.raises(ValueError, match="only int8 has a quantized convolution"):
            QuantizationSpec.for_model(tiny_conformer(), FP8_W8A8, module_types=("conv",))

    def test_an_explicit_target_naming_an_unsupported_conv_is_refused(self):
        from oasr.quantization.spec import QuantGroup

        spec = QuantizationSpec(groups=(QuantGroup(INT8_W8A8_STATIC, (STEM,)),))
        with pytest.raises(ValueError, match="cannot be quantized: in_channels=1"):
            spec.check_against(tiny_conformer())

    def test_the_method_follows_the_layer(self):
        assert isinstance(method_for(INT8_W8A8_STATIC, Conv2d(32, 32, 3)), W8A8ConvMethod)
        assert type(method_for(INT8_W8A8_STATIC, Linear(32, 32))) is W8A8Method
        with pytest.raises(NotImplementedError, match="calibrated static input scale"):
            method_for(INT8_W8A8, Conv2d(32, 32, 3))


# -----------------------------------------------------------------------------
# Storage and numerics
# -----------------------------------------------------------------------------


def _conv_layers():
    torch.manual_seed(0)
    return {
        "conv2d": (Conv2d(32, 48, 3, padding=1, stride=2), torch.randn(2, 11, 9, 32)),
        "conv2d_relu": (
            Conv2dActivation(32, 48, 3, stride=2, activation_type="relu"),
            torch.randn(2, 11, 9, 32),
        ),
        "conv1d": (Conv1d(16, 32, 3, padding=1, stride=2), torch.randn(2, 30, 16)),
        "conv1d_swish": (Conv1dActivation(16, 32, 3, padding=1), torch.randn(2, 30, 16)),
    }


class TestStorage:
    @pytest.mark.parametrize("name", sorted(_conv_layers()))
    def test_slots_keep_the_layer_shape_and_survive_a_cast(self, name):
        layer, x = _conv_layers()[name]
        shape = tuple(layer.weight.shape)
        _calibrated(layer, x)
        assert isinstance(layer.quant_method, W8A8ConvMethod)
        assert layer.weight.dtype == torch.int8 and tuple(layer.weight.shape) == shape
        assert layer.weight_scale.shape == (layer.out_channels,)
        assert layer.input_scale.shape == (1,)
        layer.half()
        assert layer.weight.dtype == torch.int8 and layer.weight_scale.dtype == torch.float32
        assert layer.bias.dtype == torch.float16

    @pytest.mark.parametrize("name", sorted(_conv_layers()))
    def test_the_reference_tracks_the_fp32_convolution(self, name):
        layer, x = _conv_layers()[name]
        with torch.no_grad():
            fp = layer(x)
            _calibrated(layer, x)
            bind_quantization(_Wrap(layer))
            q = layer(x)
        assert q.shape == fp.shape and q.dtype == fp.dtype
        assert float((q - fp).norm() / fp.norm()) < 0.02


class TestCalibrationAndExport:
    def test_quantize_model_calibrates_the_convolution(self):
        model = tiny_conformer()
        spec = quantize_model(
            model, get_recipe("int8_w8a8_rtn"), _loop, module_types=("linear", "conv2d")
        )
        conv = dict(model.named_modules())[CONV]
        assert isinstance(conv.quant_method, W8A8ConvMethod)
        assert float(conv.input_scale) > 0
        assert spec.producer["static_conv_inputs"]["layers"] == 1

    def test_a_convolution_needs_calibration_audio(self):
        with pytest.raises(ValueError, match="need calibration audio"):
            quantize_model(
                tiny_conformer(), get_recipe("int8_w8a8_rtn"), None, module_types=("conv2d",)
            )

    def test_native_round_trip(self, tmp_path):
        pytest.importorskip("safetensors")
        from oasr.checkpoints.native import save_native
        from oasr.models.registry import load_checkpoint_bundle, materialize_model

        model = tiny_conformer()
        spec = quantize_model(
            model, get_recipe("smoothquant_w8a8_fused"), _loop, module_types=("all",)
        )
        dst = save_native(
            tmp_path / "q",
            architecture="conformer",
            model=model,
            model_config=tiny_conformer_config(),
            aux={"global_cmvn": None},
            quantization=spec,
        )
        arch, bundle = load_checkpoint_bundle(dst)
        loaded, _, _ = materialize_model(arch, bundle)
        before, after = model.state_dict(), loaded.state_dict()
        assert set(before) == set(after)
        for key in before:
            assert before[key].dtype == after[key].dtype and torch.equal(before[key], after[key])
        assert isinstance(dict(loaded.named_modules())[CONV].quant_method, W8A8ConvMethod)
        with torch.no_grad():
            assert torch.equal(
                model.encode_offline(*_feats())[0], loaded.encode_offline(*_feats())[0]
            )


@cuda
class TestKernelInTheModel:
    @pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
    @pytest.mark.parametrize("name", sorted(_conv_layers()))
    def test_kernel_matches_reference(self, name, dtype):
        layer, x = _conv_layers()[name]
        _calibrated(layer, x)
        layer = layer.cuda().to(dtype)
        bind_quantization(_Wrap(layer))
        assert layer.quant_method.lane is not None
        x = x.cuda().to(dtype)
        with torch.no_grad():
            got = layer(x)
            ref = layer.quant_method.apply_reference(layer, x)
        rel = 2.0**-9 if dtype == torch.float16 else 2.0**-6
        diff = (got.float() - ref.float()).abs()
        assert bool((diff <= ref.float().abs() * rel + 1e-3).all()), float(diff.max())

    def test_the_quantized_conformer_runs_on_the_kernel(self):
        model = tiny_conformer()
        quantize_model(model, get_recipe("int8_w8a8_rtn"), _loop, module_types=("all",))
        model = model.cuda().half()
        bind_quantization(model)
        conv = dict(model.named_modules())[CONV]
        assert conv.quant_method.lane is not None
        feats, lens = _feats()
        with torch.no_grad():
            out = model.encode_offline(feats.cuda().half(), lens.cuda())[0]
        assert torch.isfinite(out).all()
        assert isinstance(conv, QuantizableConv)
