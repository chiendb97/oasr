"""``ASREngine``'s tuning hooks over the int8 ``scaled_mm`` family.

``_apply_tuning`` (``tuning.strict`` / ``tuning.mode="prewarm"``), ``tuning_report``
and the quantized-layer coverage line, driven on a stand-in that carries only
what those methods read -- a census, a device, a model -- so the int8 coverage
logic is tested without loading a checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import oasr.jit.scaled_mm as jsm
from oasr.engine.config import TuningConfig
from oasr.engine.engine import ASREngine
from oasr.tune import database as db
from oasr.tune.census import ShapePoint, ShapeSet

#: A width no shipped file covers.
N, K, M = 1024, 512, 64


@pytest.fixture
def user_db(tmp_path, monkeypatch):
    monkeypatch.setenv("OASR_TUNE_USER_DB", str(tmp_path))
    db.reload()
    yield tmp_path
    monkeypatch.setenv("OASR_TUNE_USER_DB", "off")
    db.reload()


def _sm() -> int:
    from oasr.jit.core import _get_target_sm

    return _get_target_sm()


class _Stub:
    """What ``_apply_tuning`` / ``tuning_report`` read off an engine."""

    _apply_tuning = ASREngine._apply_tuning
    tuning_report = ASREngine.tuning_report
    _quantized_tuning_note = ASREngine._quantized_tuning_note

    def __init__(self, census=None, model=None):
        self._device = torch.device("cuda")
        self._census = census
        self._model = model if model is not None else torch.nn.Linear(2, 2)

    def _tuning_census(self, config):
        return self._census


def _config(**tuning):
    return SimpleNamespace(tuning=TuningConfig(**tuning), service_mode="offline")


def _int8_census(op="scaled_mm"):
    dtype = "float8_e4m3fn" if jsm.op_format(op) == "fp8" else "int8"
    return ShapeSet([ShapePoint(op, N, K, dtype, 1, M, 1.0, 1.0, True)], N * K)


def _cover(root, op="scaled_mm", n=N, k=K, cid=jsm.HEURISTIC_ID):
    """A user-tier entry for each op, in the file of the op's format."""
    sm = _sm()
    ops = [(op, n, k)] if isinstance(op, str) else op
    fmt = jsm.op_format(ops[0][0])
    family = jsm.TUNING_FAMILY[fmt]
    tf = db.new_file(family, sm, jsm.LANE_ID[fmt])
    for o, nn, kk in ops:
        tf.entries[jsm.scaled_mm_sig_key(o, nn, kk)] = db.Entry(regions=[(None, cid)])
    tf.configs[cid] = (
        {"kind": cid}
        if cid == jsm.HEURISTIC_ID
        else jsm.config_to_params(jsm.get_production_configs(sm, fmt)[cid])
    )
    db.save_file(tf, root / f"sm{sm}" / f"{family}.json")
    db.reload()


class TestStrict:
    @pytest.mark.parametrize("op", jsm.SCALED_MM_OPS)
    def test_an_uncovered_int8_must_point_raises(self, user_db, op):
        with pytest.raises(RuntimeError, match=rf"tuning.strict: 1 must-tune.*{op}\(N={N},K={K}"):
            _Stub(_int8_census(op))._apply_tuning(_config(strict=True))

    @pytest.mark.parametrize("op", jsm.SCALED_MM_OPS)
    def test_a_covered_int8_must_point_passes(self, user_db, op):
        _cover(user_db, op)
        _Stub(_int8_census(op))._apply_tuning(_config(strict=True))

    def test_an_entry_for_the_other_op_does_not_cover_it(self, user_db):
        _cover(user_db, "scaled_mm")
        with pytest.raises(RuntimeError, match="scaled_mm_quant"):
            _Stub(_int8_census("scaled_mm_quant"))._apply_tuning(_config(strict=True))

    def test_a_point_that_is_not_must_tune_is_not_required(self, user_db):
        ss = _int8_census()
        ss.points[0].must = False
        _Stub(ss)._apply_tuning(_config(strict=True))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="measures the int8 kernels")
class TestPrewarm:
    def test_prewarm_measures_the_uncovered_int8_signature_into_the_user_tier(self, user_db):
        _Stub(_int8_census())._apply_tuning(_config(mode="prewarm", strict=True, budget_s=120))
        path = db.user_path(_sm(), "scaled_mm")
        tf = db.load_file(path, "scaled_mm", _sm())
        assert tf is not None and jsm.scaled_mm_sig_key("scaled_mm", N, K) in tf.entries
        assert jsm.covering_tier("scaled_mm", M, N, K) == "user"


def _quantized_tiny_model():
    from helpers import tiny_conformer

    from oasr.quantization import INT8_W8A8, QuantizationSpec, apply_quantization
    from oasr.quantization.apply import bind_quantization

    model = tiny_conformer(output_size=128, attention_heads=4, linear_units=256)
    apply_quantization(model, QuantizationSpec.for_model(model, INT8_W8A8), quantize=True)
    model = model.to(device="cuda", dtype=torch.float16)
    bind_quantization(model)
    return model


def _signatures(model):
    from oasr.quantization import quantized_layers

    out = set()
    for layer in quantized_layers(model).values():
        op = "scaled_mm_quant" if layer.quant_method.emit is not None else "scaled_mm"
        out.add((op, layer.out_features, layer.in_features))
    return out


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="binds the int8 lane")
class TestReport:
    def test_a_full_precision_report_leaves_the_int8_family_out(self, user_db):
        jsm.select_config(M, N, K)  # some other engine in the process resolved one
        report = _Stub().tuning_report()
        assert "scaled_mm" not in report
        assert _Stub()._quantized_tuning_note() is None

    def test_the_coverage_line_counts_layers_by_tier(self, user_db):
        from oasr.quantization import quantized_layers

        model = _quantized_tiny_model()
        layers = len(quantized_layers(model))
        sigs = _signatures(model)
        stub = _Stub(model=model)
        line = stub._quantized_tuning_note()
        assert line == (
            f"scaled_mm: {layers} quantized layer(s) over {len(sigs)} signature(s): "
            f"0 with a tuning entry, {layers} on the shape heuristic"
        )
        _cover(user_db, sorted(sigs))
        line = stub._quantized_tuning_note()
        assert f"{layers} with a tuning entry (user={layers}), 0 on the shape heuristic" in line
        report = stub.tuning_report()
        assert report.endswith(line)
