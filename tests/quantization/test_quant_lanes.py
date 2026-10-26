# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr/quantization/kernels/lanes.py`` and the lane choice at bind.

What a quantized layer runs on is decided once, at bind, and must be what the
report says it is (AGENTS.md rule 3):

* **the device is keyed by its kernel family**, as the JIT builds it: an sm_121
  device runs the sm_120 kernels, an sm_87 one the sm_86 kernels.  Keyed by the
  raw capability, every lane table missed them -- the GEMMs fell to a torch lane
  and a quantized convolution raised;
* **``auto`` never falls back to a torch lane**.  The torch lanes are baselines
  and A/B arms, reached by naming ``torch``; what the CUTLASS lanes cannot serve
  is a declared gap (sm75 runs the counted torch reference) or a refusal, and a
  GEMM outside their operand contract is not selected at all;
* **a bad ``OASR_SCALED_MM_BACKEND`` is refused where it takes effect**, the first
  bind -- not at import, which every model file reaches.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

import oasr.quantization.kernels.lanes as lanes
from oasr.jit.core import sm_family
from oasr.layers import Conv2d, Linear
from oasr.layers._backend import gap_hits
from oasr.quantization import QuantizationSpec, apply_quantization, bind_quantization
from oasr.quantization.kernels import CutlassSm80Int8Lane, TorchIntMMLane, TorchScaledMMLane
from oasr.quantization.kernels.lanes import CutlassSm80Int8ConvLane
from oasr.quantization.spec import FP8_W8A8, INT8_W8A8, QuantGroup

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="bind picks a lane on CUDA")


class _Wrap(torch.nn.Module):
    """A one-layer model, so a spec can name it."""

    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer


def _quantized_linear(in_features: int = 64, out_features: int = 96) -> Linear:
    torch.manual_seed(0)
    layer = Linear(in_features, out_features)
    holder = _Wrap(layer)
    apply_quantization(
        holder, QuantizationSpec((QuantGroup(INT8_W8A8, ("layer",)),)), quantize=True
    )
    return layer


def _quantized_conv() -> Conv2d:
    torch.manual_seed(0)
    layer = Conv2d(16, 32, 3, stride=2)
    layer.register_buffer("input_scale", torch.full((1,), 0.02))
    spec = QuantizationSpec.for_model(_Wrap(layer), INT8_W8A8, module_types=("conv",))
    apply_quantization(_Wrap(layer), spec, quantize=True)
    return layer


def _as_device(monkeypatch, major: int, minor: int) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (major, minor))


class TestSmFamily:
    @pytest.mark.parametrize(
        "cap, family",
        [((8, 0), 80), ((8, 7), 86), ((8, 8), 86), ((10, 3), 100), ((12, 1), 120), ((12, 0), 120)],
    )
    def test_a_device_runs_its_family(self, cap, family):
        assert sm_family(*cap) == family

    def test_an_unbuilt_capability_is_itself(self):
        """Not some other architecture's entry: a lane table then refuses it by name."""
        assert sm_family(11, 0) == 110
        assert not CutlassSm80Int8Lane.is_supported(sm_family(11, 0))[0]

    @cuda
    @pytest.mark.parametrize("cap", [(12, 1), (8, 7), (10, 3)])
    def test_bind_keys_the_gemm_lane_by_family(self, monkeypatch, cap):
        layer = _quantized_linear().cuda().half()
        _as_device(monkeypatch, *cap)
        bind_quantization(_Wrap(layer))
        assert isinstance(layer.quant_method.lane, CutlassSm80Int8Lane)
        native = sm_family(*cap) in (80, 86, 89, 120)
        assert layer.quant_method.lane_note == (
            "native" if native else f"non-native lane on sm{sm_family(*cap)}"
        )

    @cuda
    def test_bind_keys_the_conv_lane_by_family(self, monkeypatch):
        layer = _quantized_conv().cuda().half()
        _as_device(monkeypatch, 12, 1)
        bind_quantization(_Wrap(layer))
        assert isinstance(layer.quant_method.lane, CutlassSm80Int8ConvLane)
        assert layer.quant_method.lane_note == "native"


class TestAutoNeverFallsBackToTorch:
    def test_the_torch_lanes_are_not_auto_candidates(self):
        assert not TorchIntMMLane.in_auto and not TorchScaledMMLane.in_auto
        assert CutlassSm80Int8Lane.in_auto

    def test_sm75_has_no_auto_lane(self, monkeypatch):
        monkeypatch.setattr(lanes, "_backend", lambda: "auto")
        lane, why = lanes.choose_lane(INT8_W8A8, 96, 64, torch.float16, 75)
        assert lane is None and "cutlass_sm80_int8" in why and "torch" not in why

    def test_an_unaligned_gemm_has_no_auto_lane(self, monkeypatch):
        monkeypatch.setattr(lanes, "_backend", lambda: "auto")
        assert lanes.choose_lane(INT8_W8A8, 96, 24, torch.float16, 120)[0] is None
        assert lanes.choose_lane(FP8_W8A8, 96, 24, torch.float16, 120)[0] is None

    def test_naming_torch_reaches_the_torch_lanes(self, monkeypatch):
        monkeypatch.setattr(lanes, "_backend", lambda: "torch")
        assert isinstance(
            lanes.choose_lane(INT8_W8A8, 96, 64, torch.float16, 75)[0], TorchIntMMLane
        )
        lane, _ = lanes.choose_lane(FP8_W8A8, 96, 64, torch.float16, 120)
        assert isinstance(lane, TorchScaledMMLane)

    @cuda
    def test_sm75_takes_the_declared_gap(self, monkeypatch):
        """The gap the report counts, running the torch reference -- it used to be
        unreachable, because ``auto`` handed sm75 to ``torch._int_mm`` as 'native'."""
        monkeypatch.setattr(lanes, "_backend", lambda: "auto")
        layer = _quantized_linear().cuda().half()
        _as_device(monkeypatch, 7, 5)
        before = gap_hits().get("scaled-mm-int8-sm75", 0)
        bind_quantization(_Wrap(layer))
        assert layer.quant_method.lane is None
        assert layer.quant_method.lane_note == "reference (declared gap scaled-mm-int8-sm75)"
        assert gap_hits()["scaled-mm-int8-sm75"] == before + 1


class TestGemmOperandContract:
    @pytest.mark.parametrize(
        "shape, needle", [((24, 32), "in_features=24"), ((64, 36), "out_features=36")]
    )
    def test_an_unaligned_gemm_says_why(self, shape, needle):
        assert needle in Linear(*shape).quant_unsupported()
        assert Linear(64, 96).quant_unsupported() is None

    def test_selection_skips_it_and_records_why(self):
        model = torch.nn.Module()
        model.ok = Linear(64, 96)
        model.odd = Linear(24, 32)
        spec = QuantizationSpec.for_model(model, INT8_W8A8)
        assert spec.resolve("ok") is not None and spec.resolve("odd") is None
        assert "in_features=24" in spec.producer["skipped"]["odd"]

    def test_naming_it_is_refused(self):
        model = torch.nn.Module()
        model.odd = Linear(24, 32)
        spec = QuantizationSpec((QuantGroup(INT8_W8A8, ("odd",)),))
        with pytest.raises(ValueError, match="in_features=24"):
            apply_quantization(model, spec, quantize=True)


class TestBackendSwitch:
    def _run(self, code: str) -> subprocess.CompletedProcess:
        env = {**os.environ, "OASR_SCALED_MM_BACKEND": "cultass"}
        return subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300
        )

    def test_a_bad_value_does_not_stop_a_model_import(self):
        result = self._run("import oasr.models.conformer; print('imported')")
        assert result.returncode == 0, result.stderr
        assert "imported" in result.stdout

    def test_a_bad_value_is_refused_at_the_first_lane_choice(self):
        result = self._run(
            "import torch\n"
            "from oasr.quantization.kernels.lanes import choose_lane\n"
            "from oasr.quantization.spec import INT8_W8A8\n"
            "choose_lane(INT8_W8A8, 96, 64, torch.float16, 120)\n"
        )
        assert result.returncode != 0
        assert "OASR_SCALED_MM_BACKEND='cultass'" in result.stderr
