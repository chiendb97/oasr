# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""A quantized model inside the engine: graphs, batching, streaming, config.

What the engine promises an fp model it must promise a quantized one:

* a captured offline forward replays exactly what eager computes;
* the quantized layers add no batch dependence (per-token scales are row-local,
  the int32 accumulator is exact) — the fp16 GEMMs still have theirs, see
  ``TestBatchInvariance``;
* a ``oasr quantize`` checkpoint is served quantized with no flag, offline and
  streaming;
* ``EngineConfig.quantization`` takes what the Rust front-end sends.
"""

from __future__ import annotations

from types import SimpleNamespace

import assets
import pytest
import torch
from helpers import make_engine, tiny_conformer

from oasr.engine import ASREngine
from oasr.engine.config import EngineConfig, QuantizationConfig
from oasr.layers import QuantizedActivation
from oasr.quantization import (
    FP8_W8A8,
    INT8_W8A8,
    QuantizationSpec,
    apply_quantization,
    bind_quantization,
    quantized_layers,
)


def _dense(x) -> torch.Tensor:
    """A layer input or output as a tensor: a pre-quantized activation dequantized.

    A fused norm or a chained producer hands a quantized layer an int8
    ``QuantizedActivation`` instead of a tensor; its rows are equal exactly when
    their data and scales are, which the dequantized product shows bit for bit.
    """
    if isinstance(x, QuantizedActivation):
        scale = x.scale.unsqueeze(-1) if x.per_token else x.scale
        return x.data.float() * scale
    return x.detach()


def _layout_count(ckpt: str, scheme=INT8_W8A8) -> int:
    """How many layers *ckpt*'s own ``quantization_layout()`` lets *scheme* quantize.

    Derived, not hard-coded: the count moves whenever a layout does (the U2++
    attention decoder joining the Conformer's took it from 96 to 156).  Built on
    the meta device -- only the module tree is needed.
    """
    from pathlib import Path

    from oasr.models.registry import get_model_entry, load_checkpoint_bundle

    arch, bundle = load_checkpoint_bundle(Path(ckpt))
    with torch.device("meta"):
        model = get_model_entry(arch).model_cls.from_config(bundle.model_config, **bundle.aux)
    return len(QuantizationSpec.for_model(model, scheme).groups[0].targets)


def _waves(n: int):
    import soundfile as sf

    return [torch.from_numpy(sf.read(path, dtype="float32")[0]) for path in assets.require_wavs(n)]


class TestConfig:
    def test_mapping_from_the_front_end(self):
        cfg = EngineConfig(ckpt_dir="/x", quantization={"scheme": "int8_w8a8", "ignore": "a.*, b"})
        assert cfg.quantization == QuantizationConfig("int8_w8a8", ("a.*", "b"))

    def test_scheme_name_shorthand_and_default(self):
        assert EngineConfig(ckpt_dir="/x", quantization="int8_w8a8").quantization.scheme == (
            "int8_w8a8"
        )
        assert EngineConfig(ckpt_dir="/x").quantization is None

    def test_unknown_scheme_or_option_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="presets"):
            EngineConfig(ckpt_dir="/x", quantization={"scheme": "int3"})
        with pytest.raises(ValueError, match="unknown quantization option"):
            EngineConfig(ckpt_dir="/x", quantization={"scheme": "int8_w8a8", "alpha": "0.5"})

    @pytest.mark.parametrize("key, value", [("ignore", "ctc.*"), ("module_types", "linear")])
    def test_a_layer_selection_needs_a_scheme(self, key, value):
        """``ignore`` / ``module_types`` only mean something to online quantization:
        without a scheme there is nothing for them to select, and dropping them
        silently is what ``ignore`` used to do."""
        with pytest.raises(ValueError, match="needs a scheme"):
            EngineConfig(ckpt_dir="/x", quantization={key: value})
        # The KV cache format stands alone (fp weights over an fp8 cache).
        assert EngineConfig(
            ckpt_dir="/x", quantization={"kv_cache": "fp8_e4m3"}
        ).quantization == QuantizationConfig(kv_cache="fp8_e4m3")

    def test_the_served_identity_includes_the_kv_cache(self):
        """An fp8 KV cache over full-precision weights is a configuration of its
        own: ``quantization_info`` names it, ``quantization`` (the weight schemes
        ``/v1/models`` reports) stays ``None``."""
        engine = SimpleNamespace(
            _model=torch.nn.Linear(4, 4),
            _config=SimpleNamespace(quantization=QuantizationConfig(kv_cache="fp8_e4m3")),
        )
        info = ASREngine.quantization_info(engine)
        assert info == {"schemes": [], "layers": 0, "lanes": [], "kv_cache": "fp8_e4m3"}
        engine.quantization_info = lambda: info
        assert ASREngine.quantization.fget(engine) is None
        engine._config.quantization = None
        assert ASREngine.quantization_info(engine) is None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the int8 lane is a CUDA kernel")
class TestGraphs:
    @pytest.mark.parametrize("module_types", [None, ("all",)], ids=["gemms", "all"])
    def test_captured_quantized_forward_matches_eager(self, module_types):
        """Capture changes launch count, never numerics — quantize kernel included.

        ``all`` adds the int8 subsampling convolution (static scale, uncalibrated).
        """
        from oasr.engine.offline_graph import FUSED, GraphedOfflineForward

        model = tiny_conformer(output_size=256, attention_heads=4, linear_units=256)
        spec = QuantizationSpec.for_model(model, INT8_W8A8, module_types=module_types)
        apply_quantization(model, spec, quantize=True)
        model = model.to(device="cuda", dtype=torch.float16)
        bind_quantization(model)
        assert all(
            layer.quant_method.lane is not None
            for layer in (model.encoder.encoders[0].feed_forward.w_1,)
        )
        cache = GraphedOfflineForward(device=torch.device("cuda"), batch_buckets=[4])
        g = torch.Generator().manual_seed(0)
        feats = torch.randn(4, 160, 80, generator=g).cuda().half()
        lens = torch.tensor([160, 150, 101, 64], device="cuda", dtype=torch.int32)
        with torch.no_grad():
            padded, plens = cache.pad_time(feats, lens)
            want, want_len = model.forward_offline(padded, plens)
            got = cache.run(FUSED, model.forward_offline, feats, lens)
        assert got is not None, "the quantized forward did not capture"
        assert torch.equal(got[0], want) and torch.equal(got[1], want_len)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestBatchInvariance:
    """The quantized path adds no batch dependence of its own.

    It cannot remove the fp16 path's: a full-precision GEMM's rule-table config
    (tile, split-K) is a function of M, so its rounding depends on how many rows
    share it.  On the U2++ checkpoint the subsampling projection (K=4864) differs
    by 0.03 between a batch of 8 and one utterance; fp16 transcripts absorb that,
    int8 rounding can amplify it into a flipped borderline word.  What int8 owns
    is the per-layer property tested here: given bitwise-equal input rows, a
    quantized layer returns bitwise-equal output rows, however the batch is made
    up.  (Per-token scales are row-local; the int32 accumulator is exact, and the
    fp8 lane sums every output in one order whatever the tile.)
    """

    @pytest.mark.parametrize(
        "scheme,module_types",
        [(INT8_W8A8, None), (FP8_W8A8, None), (INT8_W8A8, ("all",))],
        ids=["int8_w8a8", "fp8_w8a8", "int8_w8a8+conv"],
    )
    def test_equal_input_rows_give_equal_output_rows(self, scheme, module_types):
        if scheme is FP8_W8A8 and torch.cuda.get_device_capability() < (8, 9):
            pytest.skip("fp8 tensor cores need sm89+")
        model = tiny_conformer(output_size=256, attention_heads=4, linear_units=256)
        spec = QuantizationSpec.for_model(model, scheme, module_types=module_types)
        apply_quantization(model, spec, quantize=True)
        model = model.to(device="cuda", dtype=torch.float16)
        bind_quantization(model)
        seen = {}

        def record(name):
            def hook(module, args, out):
                seen.setdefault(name, []).append((_dense(args[0]), _dense(out)))

            return hook

        for name, layer in quantized_layers(model).items():
            layer.register_forward_hook(record(name))

        g = torch.Generator().manual_seed(9)
        lens = torch.tensor([300, 211, 160, 97], device="cuda")
        feats = torch.zeros(4, 300, 80, device="cuda", dtype=torch.float16)
        for i, n in enumerate(lens.tolist()):
            feats[i, :n] = torch.randn(n, 80, generator=g).cuda().half()
        with torch.no_grad():
            _, out_lens = model.encode_offline(feats, lens)
            model.encode_offline(feats[1:2, :211].contiguous(), lens[1:2])

        compared = 0
        valid = int(out_lens[1])
        for name, ((x_b, y_b), (x_1, y_1)) in seen.items():
            if torch.equal(x_b[1, :valid], x_1[0, :valid]):
                compared += 1
                assert torch.equal(y_b[1, :valid], y_1[0, :valid]), name
        assert compared, "no quantized layer saw equal inputs; the check is vacuous"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestRealCheckpoint:
    def test_a_smoothquant_checkpoint_serves_quantized_with_no_flag(self, tmp_path):
        """``oasr quantize`` output: loaded quantized, offline and streaming."""
        from oasr.quantization.export import quantize_checkpoint

        ckpt = assets.require("CKPT_DIR")
        dev = assets.require("LIBRISPEECH_DEV_DIR")
        dst = tmp_path / "sq"
        quantize_checkpoint(
            ckpt,
            str(dst),
            recipe="smoothquant_w8a8",
            calib_manifest=f"{dev}/manifest.jsonl",
            calib_utterances=8,
        )
        waves = _waves(4)
        expected = _layout_count(ckpt)
        for mode in ("offline", "streaming"):
            engine = make_engine(str(dst), service_mode=mode)
            try:
                info = engine.quantization_info()
                texts = engine.transcribe(waves, streaming=mode == "streaming")
                engine.reload_tuning()  # rebinds lanes; must not unquantize anything
                assert engine.quantization_info() == info
            finally:
                del engine
                torch.cuda.empty_cache()
            assert info["schemes"] == ["int8_w8a8"] and info["layers"] == expected
            assert all(texts), mode

    def test_the_tuning_report_counts_the_quantized_layers_by_tier(self):
        """Every quantized layer is accounted for: a tuning entry, or the shape heuristic."""
        ckpt = assets.require("CKPT_DIR")
        reports = {}
        for scheme in (None, "int8_w8a8"):
            engine = make_engine(ckpt, quantization=scheme)
            try:
                reports[scheme] = engine.tuning_report()
            finally:
                del engine
                torch.cuda.empty_cache()
        expected = _layout_count(ckpt)
        line = next(
            row
            for row in reports["int8_w8a8"].splitlines()
            if row.startswith(f"scaled_mm: {expected} quantized layer(s)")
        )
        assert "with a tuning entry" in line and "on the shape heuristic" in line
        assert "scaled_mm" not in reports[None]

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
        reason="fp8 tensor cores need sm89+",
    )
    def test_online_fp8_serves_offline_and_streaming(self):
        ckpt = assets.require("CKPT_DIR")
        waves = _waves(4)
        for mode in ("offline", "streaming"):
            engine = make_engine(ckpt, service_mode=mode, quantization="fp8_w8a8")
            try:
                info = engine.quantization_info()
                texts = engine.transcribe(waves, streaming=mode == "streaming")
            finally:
                del engine
                torch.cuda.empty_cache()
            assert info["schemes"] == ["fp8_w8a8"] and info["lanes"] == ["cutlass_sm89_fp8"]
            assert all(texts), mode
