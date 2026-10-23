# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The fp8 e4m3 W8A8 format through the quantization plane.

One method (``W8A8Method``) serves int8 and fp8; what differs is the storage
dtype, the range (``Q = 448``), the GEMM lane (``cutlass_sm89_fp8``, sm89+) and
the clip search's error model.  So these tests follow the int8 ones' structure
and check what fp8 adds:

* storage: ``float8_e4m3fn`` weights that survive a dtype cast and a native
  checkpoint round trip;
* the reference path on CPU, and the kernel path agreeing with it on CUDA;
* the fused producers in fp8 -- a norm emitting e4m3, a GEMM chain exchanging
  e4m3 at a calibrated static scale -- bit-identical to the unfused model, the
  same ``torch.equal`` oracle as ``test_quant_fusion.py``;
* formats never mix across a link: a producer only emits its consumers' format.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from helpers import tiny_conformer, tiny_conformer_config

from oasr.layers import QuantizedActivation
from oasr.layers._backend import layers_backend_override
from oasr.quantization import (
    FP8_W8A8,
    INT8_W8A8,
    QuantGroup,
    QuantizationSpec,
    apply_quantization,
    bind_quantization,
    get_scheme,
    method_for,
    quantized_layers,
)
from oasr.quantization.calibration import clip_mse
from oasr.quantization.export import quantize_model
from oasr.quantization.methods.w8a8 import quantize_weight_per_channel
from oasr.quantization.recipes import get_recipe

_FP8 = torch.float8_e4m3fn


def _fp8_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89


fp8_gpu = pytest.mark.skipif(not _fp8_gpu(), reason="fp8 tensor cores need sm89+")


def _loop(model: torch.nn.Module) -> None:
    """A calibration loop: random features through the encoder."""
    param = next(p for p in model.parameters() if p.is_floating_point())
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for t in (40, 64):
            feats = (torch.randn(1, t, 80, generator=g) * 2).to(param.device, param.dtype)
            model.encode_offline(feats, torch.tensor([t], device=param.device))


def _feats(device="cpu", dtype=torch.float32):
    g = torch.Generator().manual_seed(2)
    feats = torch.randn(3, 70, 80, generator=g).to(device, dtype)
    return feats, torch.tensor([70, 47, 12], device=device)


def _encode(model, device="cpu", dtype=torch.float32):
    with torch.no_grad():
        out, _ = model.encode_offline(*_feats(device, dtype))
    return out


def _served(model, dtype):
    model = model.to(device="cuda", dtype=dtype)
    bind_quantization(model)
    return model


# -----------------------------------------------------------------------------
# Scheme, storage, weights
# -----------------------------------------------------------------------------


class TestScheme:
    def test_one_method_serves_both_formats(self):
        fp8, int8 = method_for(FP8_W8A8), method_for(INT8_W8A8)
        assert type(fp8) is type(int8)
        assert (fp8.fmt, fp8.quant_dtype) == ("fp8", _FP8)
        assert (int8.fmt, int8.quant_dtype) == ("int8", torch.int8)

    def test_static_preset_has_per_channel_weights(self):
        """The GEMM's weight scale is per channel; the static fp8 preset says so too."""
        scheme = get_scheme("fp8_w8a8_static")
        assert scheme.weights.granularity == "channel"
        assert method_for(scheme).static

    def test_weight_quantization_is_per_channel_e4m3(self):
        g = torch.Generator().manual_seed(0)
        w = torch.randn(64, 128, generator=g) * torch.linspace(0.01, 4.0, 64)[:, None]
        w[3] = 0.0  # a padded vocabulary row
        q, scale = quantize_weight_per_channel(w, dtype=_FP8)
        assert q.dtype == _FP8 and scale.dtype == torch.float32
        assert torch.equal(
            q.float().abs().amax(dim=1)[torch.arange(64) != 3], torch.full((63,), 448.0)
        )
        assert not q[3].float().any() and torch.isfinite(scale).all()
        rel = (q.float() * scale[:, None] - w).abs() / w.abs().amax(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        assert (
            float(rel.max()) <= 2.0**-4
        )  # half an e4m3 step at the top binade, relative to the max


class TestStorage:
    def test_slots_are_fp8_and_survive_a_cast(self):
        model = tiny_conformer()
        spec = QuantizationSpec.for_model(model, FP8_W8A8)
        apply_quantization(model, spec, quantize=True)
        layers = quantized_layers(model)
        assert layers
        model.to(torch.bfloat16)
        for layer in layers.values():
            assert layer.weight.dtype == _FP8
            assert layer.weight_scale.dtype == torch.float32

    def test_runs_on_cpu_through_the_reference(self):
        model = tiny_conformer()
        ref = _encode(model)
        quantize_model(model, get_recipe("fp8_w8a8_rtn"))
        out = _encode(model)
        assert torch.isfinite(out).all()
        # fp8 W and A: a few percent of the activations' scale, not garbage.
        assert float((out - ref).abs().mean()) < 0.1 * float(ref.abs().mean())

    def test_native_round_trip(self, tmp_path: Path):
        pytest.importorskip("safetensors")
        from oasr.checkpoints.native import read_native_config, save_native
        from oasr.models.registry import load_checkpoint_bundle, materialize_model

        model = tiny_conformer()
        spec = quantize_model(model, get_recipe("smoothquant_fp8_w8a8"), _loop)
        dst = save_native(
            tmp_path / "q",
            architecture="conformer",
            model=model,
            model_config=tiny_conformer_config(),
            aux={"global_cmvn": None},
            quantization=spec,
        )
        assert [g["scheme"] for g in read_native_config(dst)["quantization"]["groups"]] == [
            "fp8_w8a8"
        ]
        arch, bundle = load_checkpoint_bundle(dst)
        loaded, _, _ = materialize_model(arch, bundle)
        want, got = model.state_dict(), loaded.state_dict()
        assert set(want) == set(got)
        for key, tensor in want.items():
            assert got[key].dtype == tensor.dtype, key
            if tensor.dtype == _FP8:
                assert torch.equal(got[key].view(torch.uint8), tensor.view(torch.uint8)), key
            else:
                assert torch.equal(got[key], tensor), key
        assert torch.equal(_encode(loaded), _encode(model))


# -----------------------------------------------------------------------------
# Kernels vs reference
# -----------------------------------------------------------------------------


@fp8_gpu
class TestKernelPath:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_lane_and_reference_agree(self, dtype):
        model = tiny_conformer()
        quantize_model(model, get_recipe("smoothquant_fp8_w8a8"), _loop)
        model = _served(model, dtype)
        for layer in quantized_layers(model).values():
            assert layer.quant_method.lane is not None
            assert layer.quant_method.lane.name == "cutlass_sm89_fp8"
        got = _encode(model, "cuda", dtype)
        with layers_backend_override("torch"):
            ref = _encode(model, "cuda", dtype)
        scale = float(ref.float().abs().mean())
        assert float((got.float() - ref.float()).abs().mean()) < 0.05 * scale

    def test_torch_lane_agrees_with_cutlass(self):
        from oasr.quantization.kernels import CutlassSm89Fp8Lane, TorchScaledMMLane

        g = torch.Generator(device="cuda").manual_seed(0)
        a = (torch.randn(77, 256, generator=g, device="cuda") * 64).to(_FP8)
        b = (torch.randn(384, 256, generator=g, device="cuda") * 64).to(_FP8)
        a_s = torch.rand(77, generator=g, device="cuda") * 1e-3
        b_s = torch.rand(384, generator=g, device="cuda") * 1e-3
        bias = torch.randn(384, generator=g, device="cuda").half()
        cut = CutlassSm89Fp8Lane().run(a, b, a_s, b_s, bias, "swish", torch.float16)
        tor = TorchScaledMMLane().run(a, b, a_s, b_s, bias, "swish", torch.float16)
        diff = (cut.float() - tor.float()).abs()
        assert bool((diff <= tor.float().abs() * 2.0**-8 + 2e-3).all())


# -----------------------------------------------------------------------------
# Fused producers in fp8
# -----------------------------------------------------------------------------


def _disarm(model) -> None:
    for layer in quantized_layers(model).values():
        layer.quant_method.emit = None
        layer.quant_method.input_source = None
    for module in model.modules():
        for knob in ("quantize_output", "quantize_dtype"):
            if knob in vars(module):
                delattr(module, knob)


@fp8_gpu
class TestFusedFp8:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_armed_equals_disarmed(self, dtype):
        model = tiny_conformer()
        spec = quantize_model(model, get_recipe("smoothquant_fp8_w8a8_fused"), _loop)
        assert spec.groups[0].scheme.name == "fp8_w8a8_static"
        model = _served(model, dtype)
        norms = [m for m in model.modules() if vars(m).get("quantize_output")]
        assert norms and all(m.quantize_dtype == _FP8 for m in norms)
        producers = [
            p for p, layer in quantized_layers(model).items() if layer.quant_method.emit is not None
        ]
        assert producers, "no GEMM chain was armed"
        armed = _encode(model, "cuda", dtype)
        twin = copy.deepcopy(model)
        _disarm(twin)
        assert torch.equal(armed, _encode(twin, "cuda", dtype))

    def test_a_norm_emits_its_consumers_format(self):
        model = tiny_conformer()
        apply_quantization(model, QuantizationSpec.for_model(model, FP8_W8A8), quantize=True)
        model = _served(model, torch.float16)
        norm = model.encoder.encoders[0].norm_ff
        assert norm.quantize_output and norm.quantize_dtype == _FP8
        x = torch.randn(1, 9, norm.normalized_shape, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            q = norm(x)
        assert isinstance(q, QuantizedActivation) and q.data.dtype == _FP8

    def test_each_norm_emits_its_consumers_format(self):
        """One block's FFN quantized int8, the rest fp8: every norm emits what its GEMM takes."""
        paths = QuantizationSpec.for_model(tiny_conformer(), FP8_W8A8).groups[0].targets
        w_1 = "encoder.encoders.0.feed_forward.w_1"
        spec = QuantizationSpec(
            (
                QuantGroup(INT8_W8A8, (w_1,)),
                QuantGroup(FP8_W8A8, tuple(p for p in paths if p != w_1)),
            )
        )
        model = tiny_conformer()
        apply_quantization(model, spec, quantize=True)
        model = _served(model, torch.float16)
        block = model.encoder.encoders[0]
        assert block.norm_ff.quantize_output and block.norm_ff.quantize_dtype == torch.int8
        assert block.norm_mha.quantize_output and block.norm_mha.quantize_dtype == _FP8
        assert torch.isfinite(_encode(model, "cuda", torch.float16)).all()

    def test_a_chain_never_mixes_formats(self):
        """A producer whose format is not its static consumer's is left unfused."""
        model = tiny_conformer()
        quantize_model(model, get_recipe("smoothquant_fp8_w8a8_fused"), _loop)
        model = _served(model, torch.float16)
        w_1 = model.encoder.encoders[0].feed_forward.w_1
        assert w_1.quant_method.emit is not None
        w_1.quant_method.quant_dtype = torch.int8  # as if it were an int8 layer
        bind_quantization(model)
        assert w_1.quant_method.emit is None
        others = model.encoder.encoders[1].feed_forward.w_1
        assert others.quant_method.emit is not None

    def test_a_wrong_format_is_refused_at_the_consumer(self):
        model = tiny_conformer()
        apply_quantization(model, QuantizationSpec.for_model(model, FP8_W8A8), quantize=True)
        model = _served(model, torch.float16)
        layer = model.encoder.encoders[0].feed_forward.w_1
        bogus = QuantizedActivation(
            data=torch.zeros(4, layer.in_features, dtype=torch.int8, device="cuda"),
            scale=torch.ones(4, device="cuda"),
            dtype=torch.float16,
        )
        with pytest.raises(RuntimeError, match="must emit torch.float8_e4m3fn"):
            layer(bogus)


# -----------------------------------------------------------------------------
# The clip search's error model
# -----------------------------------------------------------------------------


class TestFp8Clip:
    def test_fp8_keeps_more_of_the_range_than_int8(self):
        """Relative precision: a lone outlier costs fp8's small values little."""
        counts = torch.zeros(4096, dtype=torch.float64)
        counts[:400] = 1000.0  # the bulk under 10 % of the range
        counts[-1] = 3.0  # a few values at the maximum
        int8 = clip_mse(counts, 1.0, 127.0)
        fp8 = clip_mse(counts, 1.0, 448.0, "fp8")
        # Measured 0.59 (int8) vs 0.85 (fp8): fp8's step follows the value, so the
        # bulk's error barely depends on the clip, and the outliers win the trade.
        assert fp8 > int8 + 0.15

    def test_unknown_format_is_refused(self):
        with pytest.raises(ValueError, match="unknown format"):
            clip_mse(torch.ones(16), 1.0, 448.0, "fp4")
