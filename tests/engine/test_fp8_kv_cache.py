# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``EngineConfig.quantization.kv_cache = "fp8_e4m3"``: config, refusals, budget, decode.

The kernel half lives in ``tests/kernels/test_fmha_kv8.py`` and the cache
bookkeeping in ``tests/engine/test_decoder_kv.py``.  This file is the part that
only exists once a strategy drives a decoder over the cache:

* the setting is **refused by name** wherever it cannot be served -- a family
  with no AR cache, a decoder that does not declare the format, a device /
  dtype / attention mode the fp8 lane cannot read -- never accepted and ignored;
* the byte accounting admits more rows, because that is the point;
* a quantized decode is the same whichever storage and capture path runs it:
  paged eager == paged replayed **bit for bit** (rule 11), and dense agrees;
* turning it off is the code path that existed before (``kv_cache="auto"`` is
  ``None``), and an engine with it on decodes real speech like fp16 does.
"""

from __future__ import annotations

from dataclasses import replace

import assets
import pytest
import torch
from helpers import device_sm, tiny_conformer, wav_paths

from oasr.engine.config import EngineConfig, QuantizationConfig

WHISPER_CKPT = assets.declared("WHISPER_CKPT")
SPEECH_LLM_TINY = assets.declared("SPEECH_LLM_TINY")

_SUPPORTED = (80, 86, 89, 120)


def _requires_lane():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if device_sm() not in _SUPPORTED:
        pytest.skip(f"the C++ FMHA lane is not compiled for sm_{device_sm()}")


def _whisper_config(**overrides):
    from oasr.models.whisper import WhisperModelConfig

    base = {
        "vocab_size": 64,
        "d_model": 128,
        "encoder_layers": 1,
        "decoder_layers": 2,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "encoder_ffn_dim": 64,
        "decoder_ffn_dim": 64,
        "num_mel_bins": 80,
        "max_source_positions": 50,
        "max_target_positions": 64,
        "decoder_start_token_id": 60,
        "eos_token_id": 61,
        "forced_decoder_ids": [(1, 62)],
        "suppress_tokens": [0, 1],
        "begin_suppress_tokens": [2],
    }
    base.update(overrides)
    return WhisperModelConfig(**base)


def _aed(kv_cache="fp8_e4m3", *, cuda=False, dtype=torch.float16, cfg=None, **options):
    from oasr.engine.decode.aed import AedDecodeStrategy
    from oasr.engine.decode.detokenize import Detokenizer
    from oasr.models.whisper import WhisperModel

    torch.manual_seed(0)
    model = WhisperModel(cfg or _whisper_config()).eval()
    if cuda:
        model = model.cuda().to(dtype)
    config = EngineConfig(
        ckpt_dir="x",
        dtype=dtype,
        quantization=None if kv_cache is None else {"kv_cache": kv_cache},
        decode_options=options or None,
    )
    return AedDecodeStrategy(config, Detokenizer(None, None), model)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_the_rust_front_end_mapping_coerces(self):
        """``oasr-server --quantization-option kv_cache=fp8_e4m3`` sends exactly this."""
        q = QuantizationConfig.coerce({"kv_cache": "fp8_e4m3"})
        assert q.kv_cache == "fp8_e4m3" and q.scheme is None
        assert EngineConfig(ckpt_dir="x", quantization={"kv_cache": "fp8_e4m3"}).quantization == q

    def test_auto_is_the_model_dtype(self):
        assert QuantizationConfig.coerce({"kv_cache": "auto"}).kv_cache is None
        assert QuantizationConfig.coerce({"kv_cache": ""}).kv_cache is None
        assert QuantizationConfig.coerce("int8_w8a8").kv_cache is None

    def test_an_unknown_format_names_the_known_ones(self):
        with pytest.raises(ValueError, match="fp8_e4m3"):
            QuantizationConfig.coerce({"kv_cache": "fp4"})

    def test_an_unknown_key_names_kv_cache_among_the_valid_ones(self):
        with pytest.raises(ValueError, match="kv_cache"):
            QuantizationConfig.coerce({"kv": "fp8_e4m3"})


# ---------------------------------------------------------------------------
# Refusals at construction
# ---------------------------------------------------------------------------


class TestRefusedByName:
    def test_a_family_without_an_ar_cache(self):
        from oasr.engine.decode.base import get_decode_strategy_class
        from oasr.engine.decode.detokenize import Detokenizer

        config = EngineConfig(ckpt_dir="x", quantization={"kv_cache": "fp8_e4m3"})
        cls = get_decode_strategy_class("ctc", config)
        with pytest.raises(ValueError, match="keeps no autoregressive decoder KV cache"):
            cls(config, Detokenizer(None, None), tiny_conformer())

    def test_a_decoder_that_does_not_declare_the_format(self, monkeypatch):
        from oasr.models.whisper.model import WhisperDecoder

        monkeypatch.setattr(WhisperDecoder, "kv_cache_dtypes", ())
        with pytest.raises(ValueError, match="does not store that KV-cache format"):
            _aed()

    def test_the_shipped_decoders_declare_it(self):
        from oasr.models.speech_llm.llm import Qwen2Lm
        from oasr.models.whisper.model import WhisperDecoder

        assert "fp8_e4m3" in WhisperDecoder.kv_cache_dtypes
        assert "fp8_e4m3" in Qwen2Lm.kv_cache_dtypes

    @pytest.mark.cuda
    def test_a_float32_model_on_cuda(self):
        _requires_lane()
        with pytest.raises(ValueError, match="fp16/bf16"):
            _aed(cuda=True, dtype=torch.float32)

    @pytest.mark.cuda
    def test_a_head_dim_the_lane_cannot_tile(self):
        _requires_lane()
        # d_model 160 over 2 heads: head_dim 80, not a whole number of fp8 loads.
        with pytest.raises(ValueError, match="cannot read this decoder's cache"):
            _aed(cuda=True, cfg=_whisper_config(d_model=160))

    @pytest.mark.cuda
    @pytest.mark.parametrize("mode,needle", [("sdpa", "forces the reference"), ("cute", "CuTeDSL")])
    def test_an_attention_mode_that_cannot_read_fp8(self, monkeypatch, mode, needle):
        _requires_lane()
        import oasr.jit.attention as arbiter

        monkeypatch.setattr(arbiter, "get_backend_mode", lambda: mode)
        with pytest.raises(ValueError, match=needle):
            _aed(cuda=True)

    def test_off_cuda_the_reference_serves_it(self):
        """CPU is out of scope for serving, not an error: the parity oracle runs."""
        assert _aed().kv_cache_format == "fp8_e4m3"


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class TestBudget:
    def test_a_row_costs_fp8_plus_scales(self):
        fp16, fp8 = _aed(kv_cache=None), _aed()
        spec = fp8._model.decoder_cache_spec
        positions = fp8._position_budget() + spec.cross_attention_len
        assert fp16.kv_bytes_per_row() == positions * 2 * spec.num_layers * spec.n_kv_head * (
            spec.head_dim * 2
        )
        assert fp8.kv_bytes_per_row() == positions * 2 * spec.num_layers * spec.n_kv_head * (
            spec.head_dim + 4
        )

    def test_the_vram_budget_admits_more_rows(self):
        from oasr.engine.memory import MemoryProfile, derive_decode_kv_budget

        profile = MemoryProfile(
            total_bytes=32 << 30, free_bytes=24 << 30, activation_bytes=1 << 30, utilization=0.9
        )
        fp16, fp8 = _aed(kv_cache=None), _aed()
        rows16 = derive_decode_kv_budget(profile, bytes_per_row=fp16.kv_bytes_per_row()).rows
        rows8 = derive_decode_kv_budget(profile, bytes_per_row=fp8.kv_bytes_per_row()).rows
        head_dim = fp8._model.decoder_cache_spec.head_dim
        assert rows8 / rows16 == pytest.approx(2 * head_dim / (head_dim + 4), rel=1e-3)

    def test_the_paged_pool_is_fp8_and_holds_more_blocks_for_the_same_bytes(self):
        def pool(kv_cache):
            strat = _aed(kv_cache=kv_cache, kv_storage="paged")
            strat._config.decode_kv_budget_gib = 0.002
            strat._config.max_decode_slots = 64
            return strat.kv_manager().pool

        p16, p8 = pool(None), pool("fp8_e4m3")
        assert p8.config.dtype == torch.float8_e4m3fn and p8.quantized
        assert not p16.quantized
        assert p8.num_blocks / p16.num_blocks == pytest.approx(128 / 68, rel=0.01)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def _decode(strat, rows=3, seed=1):
    from oasr.engine.generation import StepBudget
    from oasr.engine.request import Request

    cfg = strat._mcfg
    torch.manual_seed(seed)
    requests = [Request(audio=torch.zeros(16000), streaming=False) for _ in range(rows)]
    enc = torch.randn(rows, cfg.max_source_positions, cfg.d_model, device="cuda").half()
    strat.begin_offline(requests, enc, torch.tensor([enc.size(1)] * rows))
    outputs = {}
    for _ in range(500):
        for out in strat.advance(StepBudget(max_steps=8)):
            outputs[out.request_id] = out.tokens[0]
        if not strat.has_pending():
            break
    assert not strat.has_pending()
    return [outputs[r.request_id] for r in requests]


@pytest.mark.cuda
class TestQuantizedDecode:
    """The AED strategy end to end over a tiny random Whisper, on CUDA."""

    def test_storage_and_capture_paths_agree(self):
        _requires_lane()
        paged_eager = _aed(cuda=True, kv_storage="paged", step_graphs=False)
        paged_graphs = _aed(cuda=True, kv_storage="paged", step_graphs=True)
        dense = _aed(cuda=True, kv_storage="dense", step_graphs=False)
        a, b, c = _decode(paged_eager), _decode(paged_graphs), _decode(dense)
        assert paged_graphs._step_graphs() is not None
        assert paged_graphs._step_graphs().num_captured > 0, "the fp8 step was not captured"
        assert a == b, "a replayed fp8 step must be the eager step (rule 11)"
        assert a == c
        for strat in (paged_eager, paged_graphs):
            mgr = strat.kv_manager()
            assert mgr.quantized
            assert mgr.pool.num_free_blocks == mgr.pool.num_blocks, "pages leaked"

    def test_auto_and_absent_are_the_same_engine_path(self):
        _requires_lane()
        off = _aed(kv_cache=None, cuda=True)
        auto = _aed(kv_cache="auto", cuda=True)
        assert auto.kv_cache_format is None and auto._kv_cache_kwargs() == {}
        assert _decode(off) == _decode(auto)
        assert off.kv_manager().pool.config.dtype == torch.float16


@pytest.mark.cuda
@pytest.mark.requires_assets("WHISPER_CKPT", "WAV_DIR")
class TestRealWhisper:
    """whisper-tiny on real speech: fp8 KV transcribes like fp16, graphs or not."""

    def _engine(self, **kw):
        from helpers import make_engine

        return make_engine(WHISPER_CKPT, **kw)

    def test_fp8_transcripts_and_capture_paths(self):
        _requires_lane()
        import soundfile as sf

        audios = [
            torch.from_numpy(sf.read(p, dtype="float32")[0])
            for p in wav_paths(assets.declared("WAV_DIR"))[:6]
        ]

        def run(**kw):
            engine = self._engine(**kw)
            outs = engine.transcribe_outputs(audios, streaming=False)
            return engine, [o.tokens[0] for o in outs], [o.text for o in outs]

        _, fp16_tokens, fp16_text = run()
        fp8_q = {"kv_cache": "fp8_e4m3"}
        engine, graphs_tokens, graphs_text = run(quantization=fp8_q)
        strat = engine._output_processor.strategy
        assert strat.kv_manager() is not None and strat.kv_manager().quantized
        _, eager_tokens, _ = run(
            quantization=fp8_q, decode_options={"step_graphs": False, "kv_storage": "paged"}
        )
        _, dense_tokens, _ = run(quantization=fp8_q, decode_options={"kv_storage": "dense"})
        assert graphs_tokens == eager_tokens
        assert graphs_tokens == dense_tokens
        # The format's cost, measured on the 200-utterance gate as 3.70 vs 3.67 %
        # WER; on six utterances it changes no word.
        assert graphs_text == fp16_text


@pytest.mark.cuda
@pytest.mark.requires_assets("SPEECH_LLM_TINY")
class TestQwen2LmFp8:
    """The speech-LLM's LM over fp8 KV: left-padded prefill (attended fresh), then
    steps over the fp8 cache -- dense, paged and replayed must agree."""

    def test_storage_and_capture_paths_agree(self):
        _requires_lane()
        from oasr.cache.decoder_kv import DecoderKVCacheManager
        from oasr.engine.decoder_graph import DecoderStepGraphCache
        from oasr.models.registry import build_model_from_checkpoint

        model, _ = build_model_from_checkpoint(SPEECH_LLM_TINY, device="cuda", dtype=torch.bfloat16)
        lm, cfg, spec = model.decoder, model.config, model.decoder_cache_spec
        torch.manual_seed(1)
        B, P, steps = 3, 40, 24
        emb = torch.randn(B, P, cfg.text_hidden_size, device="cuda", dtype=torch.bfloat16)
        valid = torch.ones(B, P, dtype=torch.bool, device="cuda")
        valid[1, :7] = False
        valid[2, :19] = False
        feed = torch.randint(10, 100, (B, steps), device="cuda")

        def run(storage):
            mgr = None
            if storage != "dense":
                mgr = DecoderKVCacheManager(
                    DecoderKVCacheManager.build_pool(
                        num_layers=spec.num_layers,
                        n_kv_head=spec.n_kv_head,
                        head_dim=spec.head_dim,
                        block_tokens=16,
                        max_num_blocks=64,
                        device=torch.device("cuda"),
                        dtype=torch.float8_e4m3fn,
                    )
                )
            graphs = DecoderStepGraphCache(lm, mgr, width_pages=1) if storage == "graphs" else None
            out = []
            with torch.no_grad():
                logits, st = lm.prefill(
                    emb, valid, capacity=P + steps + 1, kv_manager=mgr, kv_cache_dtype="fp8_e4m3"
                )
                out.append(logits.float().cpu())
                for t in range(steps):
                    if graphs is None:
                        logits, st = lm.step(feed[:, t], st)
                    else:
                        logits = graphs.step(feed[:, t], st)
                        assert logits is not None
                        st["kv"].commit(1)
                    out.append(logits.float().cpu())
            return torch.stack(out)

        dense, paged, graphs = run("dense"), run("paged"), run("graphs")
        assert torch.equal(paged, graphs)
        assert torch.equal(dense, paged)


def test_strategy_kwargs_are_empty_without_the_setting():
    """A full-precision engine calls ``prefill`` exactly as before (no new kwarg)."""
    assert _aed(kv_cache=None)._kv_cache_kwargs() == {}
    strat = _aed()
    assert strat._kv_cache_kwargs() == {"kv_cache_dtype": "fp8_e4m3"}
    assert replace(strat.options).kv_storage == "auto"
