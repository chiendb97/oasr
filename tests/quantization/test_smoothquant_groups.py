# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SmoothQuant group handling beyond the Conformer: geometry, observation, twins.

* **geometry is validated** -- a ``v_to_o`` whose V rows do not match the
  attention output (or the grouped-KV heads), a ``producer_rows`` producer of
  the wrong width, a norm that cannot absorb the scale: each refused by name
  before any statistic is taken, since every one of them folds silently wrong;
* **every consumer is observed** -- a group's input is shared, so a consumer the
  loop does not reach is covered by one it does, and a group no consumer of
  which is reached raises instead of folding a scale computed from nothing;
* **a reduced-precision calibration twin** (``oasr quantize --calib-dtype``):
  the loop runs on a copy of the model while the algorithms hook and fold the
  fp32 master.  An fp32 twin must reproduce direct calibration bit for bit --
  smoothing *and* the static input scales measured after it, which need the
  folds mirrored into the twin -- and a bf16 twin on CUDA must land close.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch
from helpers import tiny_conformer

import oasr.quantization.algorithms.smoothquant as sq
from oasr.layers import Linear
from oasr.layers.norm import LayerNorm
from oasr.quantization import INT8_W8A8, QuantizationSpec, QuantLayout, SmoothGroup
from oasr.quantization.algorithms import build_algorithm
from oasr.quantization.export import mirrored_forward_loop, quantize_model
from oasr.quantization.recipes import get_recipe


class _Block(torch.nn.Module):
    """``norm -> (a, b)``; ``v -> o``; only ``b`` is called when ``skip_a``."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.norm = LayerNorm(32)
        self.a = Linear(32, 32)
        self.b = Linear(32, 32)
        self.v = Linear(32, 16)
        self.o = Linear(32, 32)
        self.skip_a = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out = self.b(h)
        if not self.skip_a:
            out = out + self.a(h)
        return out


def _spec(model):
    return QuantizationSpec.for_model(model, INT8_W8A8)


def _run(model, layout, loop):
    return build_algorithm("smoothquant").apply(model, layout, _spec(model), loop)


def _loop(model):
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        model(torch.randn(4, 32, generator=g) * 3)


class TestGeometry:
    @pytest.mark.parametrize(
        "group, match",
        [
            (SmoothGroup("v", ("o",), "v_to_o", v_rows=(0, 8)), "is not the 32 V channels"),
            (SmoothGroup("v", ("o",), "v_to_o", v_rows=(0, 40)), "outside"),
            # 4 query heads x 8 = 32 channels, 2 KV heads x 8 = 16 V rows: valid
            # geometry, so only a wrong one is refused.
            (SmoothGroup("v", ("o",), "v_to_o", v_rows=(0, 16), heads=(4, 3, 8)), "heads"),
            (SmoothGroup("v", ("o",), "v_to_o", v_rows=(0, 16), heads=(2, 2, 8)), "heads"),
            (SmoothGroup("v", ("o",), "producer_rows"), "16 output rows"),
        ],
    )
    def test_wrong_geometry_is_refused(self, group, match):
        model = _Block()
        with pytest.raises(ValueError, match=match):
            _run(model, QuantLayout(groups=(group,)), _loop)

    def test_grouped_kv_geometry_is_accepted(self):
        model = _Block()
        model.forward = lambda x: model.o(torch.cat([model.v(x)] * 2, dim=-1))  # GQA-ish
        group = SmoothGroup("v", ("o",), "v_to_o", v_rows=(0, 16), heads=(4, 2, 8))
        _run(model, QuantLayout(groups=(group,)), _loop)


class TestUnobservedChannels:
    def test_a_channel_calibration_never_saw_active_is_left_alone(self):
        """A ReLU channel that stayed off on the calibration audio would get ``s`` at
        its floor, so the first input that fires it is amplified ~1e5 and owns the
        token's quantization scale (Paraformer: every FFN ``w_2`` went to noise)."""
        from oasr.quantization.algorithms import SmoothQuantOptions

        model = _Block()
        group = SmoothGroup("norm", ("a", "b"), "norm_affine")
        act = torch.full((32,), 4.0)
        act[[3, 17]] = 0.0
        s = sq._smoothing_vector(group, act, dict(model.named_modules()), SmoothQuantOptions())
        assert float(s[3]) == 1.0 and float(s[17]) == 1.0
        assert not torch.allclose(s[act > 0], torch.ones(30))


class TestObservation:
    def test_a_consumer_the_loop_skips_is_covered_by_its_siblings(self):
        """``a`` (the first consumer) never runs; ``b`` reads the same input."""
        model = _Block()
        model.skip_a = True
        group = SmoothGroup("norm", ("a", "b"), "norm_affine")
        before = model.a.weight.detach().clone()
        _run(model, QuantLayout(groups=(group,)), _loop)
        assert not torch.equal(model.a.weight, before), "a's columns took no scale"

    def test_a_group_no_consumer_of_which_runs_raises(self):
        model = _Block()
        model.skip_a = True
        group = SmoothGroup("norm", ("a",), "norm_affine")
        with pytest.raises(RuntimeError, match="never reached"):
            _run(model, QuantLayout(groups=(group,)), _loop)


class _Ffn(torch.nn.Module):
    """``b(relu(a(h))) + c(h)``, ``h = norm(x)``: ``a`` is a consumer of the norm
    *and* b's producer."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.norm = LayerNorm(32)
        self.a = Linear(32, 32)
        self.b = Linear(32, 32)
        self.c = Linear(32, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.b(torch.relu(self.a(h))) + self.c(h)


class TestStoredRange:
    """Only what is stored in the served dtype must stay inside fp16's range."""

    LAYOUT = QuantLayout(
        groups=(
            SmoothGroup("norm", ("a", "c"), "norm_affine"),
            SmoothGroup("a", ("b",), "producer_rows"),
        )
    )

    @pytest.fixture
    def extreme_scales(self, monkeypatch):
        """``a``'s columns x 1e4 (norm group), its rows / 1e-2 (producer_rows):
        ``W * s1 / s2`` lands ~1e5 -- past fp16, harmless for an 8-bit weight."""

        def fake(group, act_amax, modules, opts):
            value = 1e4 if group.fold == "norm_affine" else 1e-2
            return torch.full((act_amax.numel(),), value)

        monkeypatch.setattr(sq, "_smoothing_vector", fake)

    def test_a_quantized_producers_weight_is_not_range_checked(self, extreme_scales):
        model = _Ffn()
        _run(model, self.LAYOUT, _loop)
        assert float(model.a.weight.abs().max()) > 65504.0  # 8-bit storage takes it
        assert float(model.a.bias.abs().max()) < 0.5 * 65504.0  # the served bias is bounded

    def test_a_full_precision_producers_weight_is(self, extreme_scales):
        model = _Ffn()
        spec = QuantizationSpec.for_model(model, INT8_W8A8, ignore=("a",))
        with pytest.raises(RuntimeError, match="past fp16's range"):
            build_algorithm("smoothquant").apply(model, self.LAYOUT, spec, _loop)


def _conformer_loop(model):
    param = next(p for p in model.parameters() if p.is_floating_point())
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for t in (40, 64):
            feats = (torch.randn(1, t, 80, generator=g) * 2).to(param.device, param.dtype)
            model.calibration_forward(feats, torch.tensor([t], device=param.device))


def _state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


class TestCalibrationTwin:
    @pytest.mark.parametrize("recipe", ["smoothquant_w8a8", "smoothquant_w8a8_fused"])
    def test_an_fp32_twin_reproduces_direct_calibration_exactly(self, recipe):
        direct = tiny_conformer()
        master = copy.deepcopy(direct)
        twin = copy.deepcopy(direct)
        spec_direct = quantize_model(direct, get_recipe(recipe), _conformer_loop)
        spec_twin = quantize_model(
            master, get_recipe(recipe), mirrored_forward_loop(_conformer_loop, twin)
        )
        assert spec_direct.to_dict()["groups"] == spec_twin.to_dict()["groups"]
        a, b = _state(direct), _state(master)
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]), key

    def test_the_twin_sees_the_masters_folds(self):
        """``static_inputs`` measures after ``smoothquant``: if the twin kept its
        unfolded weights, the static scales would be measured on the wrong
        activations.  Copying the master into the twin each pass is what makes
        them match -- break it and the fused recipe's scales move."""
        import oasr.quantization.export as export

        direct = tiny_conformer()
        master = copy.deepcopy(direct)
        twin = copy.deepcopy(direct)
        quantize_model(direct, get_recipe("smoothquant_w8a8_fused"), _conformer_loop)
        original = export._copy_state
        export._copy_state = lambda m, t: None
        try:
            quantize_model(
                master,
                get_recipe("smoothquant_w8a8_fused"),
                mirrored_forward_loop(_conformer_loop, twin),
            )
        finally:
            export._copy_state = original
        scales = [k for k in _state(direct) if k.endswith("input_scale")]
        assert scales
        assert any(not torch.equal(_state(direct)[k], _state(master)[k]) for k in scales)

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="a bf16 twin on the device")
    def test_a_bf16_device_twin_lands_close(self):
        direct = tiny_conformer()
        master = copy.deepcopy(direct)
        twin = copy.deepcopy(direct).to(device="cuda", dtype=torch.bfloat16)
        quantize_model(direct, get_recipe("smoothquant_w8a8"), _conformer_loop)
        quantize_model(
            master, get_recipe("smoothquant_w8a8"), mirrored_forward_loop(_conformer_loop, twin)
        )
        assert all(p.device.type == "cpu" for p in master.parameters())
        for key, want in _state(direct).items():
            got = _state(master)[key]
            if want.is_floating_point():
                torch.testing.assert_close(got, want, rtol=5e-2, atol=5e-2, msg=key)
            else:  # int8 weights: a bf16 statistic may move a value by one step
                assert int((got.int() - want.int()).abs().max()) <= 2, key


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="calibrates on the device")
@pytest.mark.requires_assets("SPEECH_LLM_TINY", "LIBRISPEECH_DEV_DIR")
def test_quantize_checkpoint_with_a_bf16_calibration_twin(tmp_path):
    """``oasr quantize --calib-dtype bfloat16`` end to end on a real speech-LLM
    checkpoint: the LM is reached through its text prompt (the checkpoint's
    tokenizer), and the result serves."""
    from oasr.models.registry import load_checkpoint_bundle, materialize_model
    from oasr.quantization.export import REPORT_NAME, quantize_checkpoint

    src = os.environ["SPEECH_LLM_TINY"]
    manifest = Path(os.environ["LIBRISPEECH_DEV_DIR"]) / "manifest.jsonl"
    out = quantize_checkpoint(
        src,
        str(tmp_path / "q"),
        recipe="smoothquant_w8a8",
        calib_manifest=str(manifest),
        calib_utterances=4,
        calibration_dtype="bfloat16",
    )
    report = json.loads((tmp_path / "q" / REPORT_NAME).read_text())
    producer = report["spec"]["producer"]
    assert producer["calibration"]["dtype"] == "bfloat16"
    assert any(p.startswith("language_model.") for p in report["layers"])
    assert any(p.endswith("mlp.gate_proj") for p in report["layers"])
    arch, bundle = load_checkpoint_bundle(Path(out))
    model, _, _ = materialize_model(arch, bundle, device="cuda", dtype=torch.bfloat16)
    feats = torch.randn(1, 2 * model.config.audio_max_source_positions, 128, device="cuda")
    with torch.no_grad():
        audio, _ = model.encode_offline(feats.bfloat16(), torch.tensor([feats.size(1)]).cuda())
    assert torch.isfinite(audio).all()
