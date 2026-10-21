# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr/quantization/export.py`` and the native format v2 it writes.

A quantized checkpoint is the one artifact the offline and runtime planes share,
so its round trip is checked tensor by tensor, and the load order that makes it
sound is checked by what it refuses: an fp checkpoint's tensors cannot be loaded
into int8 slots, a v1 checkpoint cannot carry a quantization section, and an
online request cannot re-quantize a checkpoint as a different scheme.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from helpers import tiny_conformer, tiny_conformer_config

from oasr.checkpoints.native import NATIVE_CONFIG_NAME, read_native_config, save_native
from oasr.models.registry import load_checkpoint_bundle, materialize_model
from oasr.quantization import quantized_layers
from oasr.quantization.export import quantize_model
from oasr.quantization.recipes import get_recipe

pytest.importorskip("safetensors")


def _save(model, dst: Path, quantization=None) -> Path:
    return save_native(
        dst,
        architecture="conformer",
        model=model,
        model_config=tiny_conformer_config(),
        aux={"global_cmvn": None},
        quantization=quantization,
    )


@pytest.fixture
def fp_model():
    return tiny_conformer()


def _feats():
    g = torch.Generator().manual_seed(3)
    return torch.randn(2, 60, 80, generator=g), torch.tensor([60, 33])


class TestRoundTrip:
    @pytest.mark.parametrize(
        "recipe", ["int8_w8a8_rtn", "smoothquant_w8a8", "smoothquant_w8a8_fused"]
    )
    def test_export_then_load_is_identical(self, fp_model, tmp_path, recipe):
        def loop(model):
            with torch.no_grad():
                model.encode_offline(*_feats())

        spec = quantize_model(fp_model, get_recipe(recipe), loop)
        dst = _save(fp_model, tmp_path / "q", quantization=spec)

        cfg = read_native_config(dst)
        assert cfg["format_version"] == 2
        schemes = [g["scheme"] for g in cfg["quantization"]["groups"]]
        if recipe == "smoothquant_w8a8_fused":
            # The chain consumers' static group first, so it wins resolution.
            assert schemes == ["int8_w8a8_static", "int8_w8a8"]
        else:
            assert schemes == ["int8_w8a8"]
        if recipe.startswith("smoothquant"):
            assert cfg["quantization"]["runtime_smoothed"]

        arch, bundle = load_checkpoint_bundle(dst)
        loaded, _, _ = materialize_model(arch, bundle)
        assert loaded.quantization_spec == spec
        before, after = fp_model.state_dict(), loaded.state_dict()
        assert set(before) == set(after)
        for key in before:
            assert before[key].dtype == after[key].dtype, key
            assert torch.equal(before[key], after[key]), key
        with torch.no_grad():
            torch.testing.assert_close(
                loaded.forward_offline(*_feats())[0], fp_model.forward_offline(*_feats())[0]
            )

    def test_a_full_precision_export_stays_version_1(self, fp_model, tmp_path):
        dst = _save(fp_model, tmp_path / "fp")
        cfg = json.loads((dst / NATIVE_CONFIG_NAME).read_text())
        assert cfg["format_version"] == 1 and "quantization" not in cfg
        arch, bundle = load_checkpoint_bundle(dst)
        loaded, _, _ = materialize_model(arch, bundle)
        assert loaded.quantization_spec is None and not quantized_layers(loaded)

    def test_a_dtype_cast_after_load_keeps_the_quantized_storage(self, fp_model, tmp_path):
        spec = quantize_model(fp_model, get_recipe("int8_w8a8_rtn"))
        dst = _save(fp_model, tmp_path / "q", quantization=spec)
        arch, bundle = load_checkpoint_bundle(dst)
        loaded, _, _ = materialize_model(arch, bundle, dtype=torch.bfloat16)
        layer = loaded.encoder.encoders[0].feed_forward.w_1
        assert layer.weight.dtype == torch.int8 and layer.weight_scale.dtype == torch.float32
        assert layer.bias.dtype == torch.bfloat16


class TestRefusals:
    def test_fp_tensors_cannot_fill_int8_slots(self, fp_model, tmp_path):
        spec = quantize_model(copy.deepcopy(fp_model), get_recipe("int8_w8a8_rtn"))
        dst = _save(fp_model, tmp_path / "forged", quantization=spec)  # fp weights + spec
        arch, bundle = load_checkpoint_bundle(dst)
        with pytest.raises(ValueError, match="do not have their slots' dtypes"):
            materialize_model(arch, bundle)

    def test_a_version_1_config_cannot_carry_quantization(self, fp_model, tmp_path):
        dst = _save(fp_model, tmp_path / "fp")
        cfg = json.loads((dst / NATIVE_CONFIG_NAME).read_text())
        cfg["quantization"] = {"format": 1, "groups": []}
        (dst / NATIVE_CONFIG_NAME).write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match="needs format_version 2"):
            read_native_config(dst)

    def test_online_requests_follow_the_checkpoint(self, fp_model, tmp_path):
        spec = quantize_model(fp_model, get_recipe("int8_w8a8_rtn"))
        dst = _save(fp_model, tmp_path / "q", quantization=spec)
        arch, bundle = load_checkpoint_bundle(dst)
        same, _, _ = materialize_model(
            arch, bundle, quantization=SimpleNamespace(scheme="int8_w8a8", ignore=())
        )
        assert same.quantization_spec == spec
        arch, bundle = load_checkpoint_bundle(dst)
        with pytest.raises(ValueError, match="already quantized"):
            materialize_model(arch, bundle, quantization=SimpleNamespace(scheme="fp8_w8a8"))

    def test_online_static_activation_schemes_need_calibration(self, fp_model, tmp_path):
        dst = _save(fp_model, tmp_path / "fp")
        arch, bundle = load_checkpoint_bundle(dst)
        with pytest.raises(ValueError, match="oasr quantize"):
            materialize_model(arch, bundle, quantization=SimpleNamespace(scheme="int8_w8a8_static"))

    def test_quantizing_twice_is_refused(self, fp_model):
        quantize_model(fp_model, get_recipe("int8_w8a8_rtn"))
        with pytest.raises(ValueError, match="already quantized"):
            quantize_model(fp_model, get_recipe("int8_w8a8_rtn"))
