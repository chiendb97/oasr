# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr/quantization/spec.py``: formats, presets and the checkpoint's spec.

CPU-only and pure data.  The spec is what a quantized checkpoint carries, so the
JSON round trip and the refusals are its contract: a spec that does not fit a
model must say so at load, not quantize the wrong layers.
"""

from __future__ import annotations

import json

import pytest
import torch
from helpers import tiny_conformer

from oasr.quantization.spec import (
    INT8_W8A8,
    PRESETS,
    QuantArgs,
    QuantGroup,
    QuantizationSpec,
    QuantScheme,
    get_scheme,
)


class TestFormats:
    def test_int8_storage_and_range(self):
        assert INT8_W8A8.weights.storage_dtype == torch.int8
        assert INT8_W8A8.weights.qmax == 127.0
        assert INT8_W8A8.input_activations.dynamic
        assert INT8_W8A8.input_activations.granularity == "token"

    def test_fp8_and_fp4_formats_are_describable(self):
        assert get_scheme("fp8_w8a8").weights.storage_dtype == torch.float8_e4m3fn
        assert get_scheme("fp8_w8a8").weights.qmax == 448.0
        nvfp4 = get_scheme("nvfp4")
        assert nvfp4.weights.group_size == 16 and nvfp4.weights.qmax == 6.0
        assert get_scheme("nvfp4_w4a16").input_activations is None

    def test_group_granularity_needs_a_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            QuantArgs(4, "float", "group")

    def test_unknown_scheme_names_the_presets(self):
        with pytest.raises(ValueError, match="int8_w8a8"):
            get_scheme("int4_magic")

    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_every_preset_round_trips(self, name):
        scheme = PRESETS[name]
        assert QuantScheme.from_dict(json.loads(json.dumps(scheme.to_dict()))) == scheme


class TestSpec:
    def _spec(self, **kw):
        return QuantizationSpec(
            (QuantGroup(INT8_W8A8, ("encoder.*.w_1", "encoder.*.w_2")),),
            ignore=("encoder.encoders.1.*",),
            **kw,
        )

    def test_ignore_wins_over_targets(self):
        spec = self._spec()
        assert spec.resolve("encoder.encoders.0.feed_forward.w_1") is INT8_W8A8
        assert spec.resolve("encoder.encoders.1.feed_forward.w_1") is None
        assert spec.resolve("ctc.ctc_lo") is None

    def test_json_round_trip(self):
        spec = self._spec(
            runtime_smoothed=("encoder.encoders.0.feed_forward.w_2",),
            producer={"algorithm": "smoothquant", "alpha": 0.5},
        )
        back = QuantizationSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert back == spec
        assert back.producer == spec.producer

    def test_unknown_section_format_is_refused(self):
        d = self._spec().to_dict()
        d["format"] = 99
        with pytest.raises(ValueError, match="format"):
            QuantizationSpec.from_dict(d)

    def test_per_tensor_dynamic_activations_are_refused(self):
        """A batch-wide dynamic scale makes a row depend on the rest of the batch."""
        bad = QuantScheme(
            QuantArgs(8, "int", "channel"), QuantArgs(8, "int", "tensor", dynamic=True)
        )
        with pytest.raises(ValueError, match="rest of the batch"):
            QuantizationSpec((QuantGroup(bad, ("*",)),))


class TestAgainstAModel:
    def test_a_target_matching_nothing_is_refused(self):
        model = tiny_conformer()
        spec = QuantizationSpec((QuantGroup(INT8_W8A8, ("encoder.no_such_layer",)),))
        with pytest.raises(ValueError, match="matches no quantizable layer"):
            spec.check_against(model)

    def test_a_non_gemm_target_is_not_quantizable(self):
        """The CTC head reads its weight raw through the fused log-softmax."""
        model = tiny_conformer()
        spec = QuantizationSpec((QuantGroup(INT8_W8A8, ("ctc.ctc_lo",)),))
        with pytest.raises(ValueError, match="matches no quantizable layer"):
            spec.check_against(model)

    def test_runtime_smoothed_must_be_quantized(self):
        model = tiny_conformer()
        spec = QuantizationSpec(
            (QuantGroup(INT8_W8A8, ("encoder.encoders.0.feed_forward.w_2",)),),
            runtime_smoothed=("encoder.encoders.0.feed_forward.w_1",),
        )
        with pytest.raises(ValueError, match="not quantized"):
            spec.check_against(model)

    def test_for_model_honours_the_layout_and_extra_ignores(self):
        model = tiny_conformer()
        spec = QuantizationSpec.for_model(model, INT8_W8A8, ignore=("*.pointwise_conv2",))
        targets = spec.groups[0].targets
        assert targets and all("pointwise_conv2" not in t for t in targets)
        assert spec.ignore == ("*.pointwise_conv2",)
        spec.check_against(model)
