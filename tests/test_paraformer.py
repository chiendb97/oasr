# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paraformer tests: LFR transform, CIF semantics, FunASR char tokenizer,
converter/native round-trip, FunASR reference parity, and engine e2e.

The parity oracle (``oasr_ref/ref.pt`` under the checkpoint dir) was captured
from FunASR 1.3.14 on CPU float32 with dither forced to 0 — it holds the
frontend features, encoder output, CIF outputs, decoder logits, and the final
transcript for ``example/asr_example.wav``.
"""

import os

import assets
import numpy as np
import pytest
import torch

# Declared once in tests/assets.py so --strict-assets can make a missing
# checkpoint fatal instead of silently green.
PARA_CKPT = assets.declared("OASR_PARAFORMER_CKPT")
REF_PT = os.path.join(assets.declared("OASR_PARAFORMER_REF"), "ref.pt")

needs_ckpt = pytest.mark.requires_assets("OASR_PARAFORMER_CKPT")
needs_ref = pytest.mark.requires_assets("OASR_PARAFORMER_CKPT", "OASR_PARAFORMER_REF")


# ---------------------------------------------------------------------------
# LFR feature stacking
# ---------------------------------------------------------------------------


def _ref_apply_lfr(inputs: torch.Tensor, lfr_m: int, lfr_n: int) -> torch.Tensor:
    """FunASR ``apply_lfr`` reference loop (single utterance)."""
    lfr = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    for i in range(T_lfr):
        if lfr_m <= T - i * lfr_n:
            lfr.append(inputs[i * lfr_n : i * lfr_n + lfr_m].reshape(1, -1))
        else:
            num_padding = lfr_m - (T - i * lfr_n)
            frame = inputs[i * lfr_n :].reshape(-1)
            for _ in range(num_padding):
                frame = torch.hstack((frame, inputs[-1]))
            lfr.append(frame.reshape(1, -1))
    return torch.vstack(lfr)


class TestLFR:
    @pytest.mark.parametrize("lfr_m,lfr_n", [(7, 6), (5, 4), (3, 1), (1, 2)])
    def test_matches_reference_loop(self, lfr_m, lfr_n):
        from oasr.features.lfr import apply_lfr_batch

        torch.manual_seed(0)
        lens = torch.tensor([218, 1, 13, 100])
        feats = torch.randn(4, 218, 8)
        for b in range(4):
            feats[b, lens[b] :] = 0
        out, out_lens = apply_lfr_batch(feats, lens, lfr_m, lfr_n)
        assert out.shape[-1] == 8 * lfr_m
        for b in range(4):
            ref = _ref_apply_lfr(feats[b, : lens[b]], lfr_m, lfr_n)
            assert out_lens[b].item() == ref.size(0)
            assert torch.equal(out[b, : out_lens[b]], ref)
            assert out[b, out_lens[b] :].abs().sum() == 0  # padding stays zero

    def test_disabled_is_identity(self):
        from oasr.features.lfr import apply_lfr_batch

        feats = torch.randn(2, 10, 4)
        lens = torch.tensor([10, 7])
        out, out_lens = apply_lfr_batch(feats, lens, 1, 1)
        assert out is feats and out_lens is lens

    def test_feature_config_output_dim_folds_lfr(self):
        from oasr.features import FeatureConfig

        cfg = FeatureConfig(num_mel_bins=80, lfr_m=7, lfr_n=6, window_type="hamming")
        assert cfg.output_dim == 560 and cfg.lfr_enabled

    def test_streaming_rejects_lfr(self):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor
        from oasr.engine.request import Request
        from oasr.features import FeatureConfig

        cfg = EngineConfig(
            feature_config=FeatureConfig(lfr_m=7, lfr_n=6, window_type="hamming"),
            use_cuda_graphs=False,
        )
        proc = InputProcessor(cfg, torch.device("cpu"))
        with pytest.raises(NotImplementedError, match="offline-only"):
            proc.prepare_streaming(Request(streaming=True))


# ---------------------------------------------------------------------------
# CIF integrate-and-fire
# ---------------------------------------------------------------------------


def _naive_cif(hidden: torch.Tensor, alphas: torch.Tensor, threshold: float = 1.0):
    """Scalar integrate-and-fire loop (the semantic definition)."""
    frames = []
    integral = 0.0
    acc = torch.zeros(hidden.size(-1))
    for t in range(alphas.numel()):
        a = float(alphas[t])
        if integral + a < threshold:
            integral += a
            acc = acc + a * hidden[t]
        else:
            spill = integral + a - threshold
            acc = acc + (threshold - integral) * hidden[t]
            frames.append(acc)
            integral = spill
            acc = spill * hidden[t]
    return torch.stack(frames) if frames else torch.zeros(0, hidden.size(-1))


class TestCif:
    def test_cif_v1_matches_naive_loop(self):
        from oasr.models.paraformer.predictor import cif_v1

        torch.manual_seed(1)
        alphas = torch.tensor([[0.4, 0.7, 0.2, 0.9, 0.3, 0.6, 1.1, 0.45]])
        hidden = torch.randn(1, 8, 4)
        frames, fires = cif_v1(hidden, alphas, 1.0)
        naive = _naive_cif(hidden[0], alphas[0])
        assert frames.size(1) >= naive.size(0)
        torch.testing.assert_close(frames[0, : naive.size(0)], naive, rtol=1e-5, atol=1e-5)
        # fires >= 1 exactly where the integral crossed the threshold
        assert (fires[0] >= 1.0).sum().item() == naive.size(0)

    def test_cif_v1_no_fires(self):
        from oasr.models.paraformer.predictor import cif_v1

        hidden = torch.randn(2, 4, 3)
        alphas = torch.full((2, 4), 0.1)
        frames, fires = cif_v1(hidden, alphas, 1.0)
        assert frames.size(1) == 0 and (fires < 1).all()

    def test_predictor_tail_fires_last_token(self):
        from oasr.models.paraformer.config import ParaformerModelConfig
        from oasr.models.paraformer.predictor import CifPredictor

        cfg = ParaformerModelConfig(predictor_idim=8)
        pred = CifPredictor(cfg)
        torch.manual_seed(2)
        hidden = torch.randn(2, 12, 8)
        mask = torch.ones(2, 12)
        mask[1, 5:] = 0  # row 1: only 5 valid frames
        embeds, token_num, alphas, fires = pred(hidden, mask)
        # tail column appended → T+1; the padded row's tail lands at its own
        # first pad position, not the end of the padded buffer
        assert alphas.shape == (2, 13) and fires.shape == (2, 13)
        assert alphas[1, 5] >= 0.0 and alphas[1, 6:].sum() == 0
        assert embeds.size(1) == int(token_num.max().item())
        assert token_num.min() >= 0


# ---------------------------------------------------------------------------
# sentence_postprocess + tokenizer
# ---------------------------------------------------------------------------

# Expected strings generated with funasr 1.3.14's
# ``postprocess_utils.sentence_postprocess`` (see docstring at top).
_POSTPROCESS_CASES = [
    (["正", "是", "因", "为"], "正是因为"),
    (["hel@@", "lo", "wor@@", "ld"], "hello world"),
    (["we", "are", "b", "b", "c", "news"], "we are BBC news"),
    (["我", "用", "i@@", "pho@@", "ne", "打", "电", "话"], "我用iphone打电话"),
    (["price", "is", "十", "元"], "price is十元"),
    (["<s>", "你", "好", "</s>", "<unk>"], "你好"),
    (["a"], "a"),
    (["数", "字", "2", "0", "2", "4"], "数字2024"),
    (["it's", "fine"], "it's fine"),
    (["tail@@"], ""),
    (["b", "b", "c"], "BBC"),
    (["你", "好", "b", "b", "c", "新", "闻"], "你好BBC新闻"),
]


class TestSentencePostprocess:
    @pytest.mark.parametrize("tokens,expected", _POSTPROCESS_CASES)
    def test_matches_funasr(self, tokens, expected):
        from oasr.tokenizers.funasr_char import sentence_postprocess

        assert sentence_postprocess(tokens) == expected


@needs_ckpt
class TestFunASRCharTokenizer:
    @pytest.fixture(scope="class")
    def tok(self):
        from oasr.tokenizers import TokenizerSpec, build_tokenizer

        spec = TokenizerSpec(
            kind="funasr_char",
            files={"tokens": os.path.join(PARA_CKPT, "tokens.json")},
            options={"special_ids": [0, 1, 2]},
        )
        return build_tokenizer(spec)

    def test_vocab_and_specials(self, tok):
        assert tok.vocab_size == 8404
        assert tok.special_ids == frozenset({0, 1, 2})

    def test_decode_strips_specials(self, tok):
        ids = tok.encode("正义")
        assert ids and all(i > 2 for i in ids)
        assert tok.decode([1] + ids + [2, 0]) == "正义"

    def test_encode_falls_back_per_char(self, tok):
        ids = tok.encode("正义")
        assert len(ids) == 2
        assert tok.decode(ids) == "正义"


# ---------------------------------------------------------------------------
# Converter + detection + native round-trip
# ---------------------------------------------------------------------------


@needs_ckpt
class TestConverter:
    def test_detect_and_precedence(self):
        from pathlib import Path

        from oasr.models.paraformer.convert import FunASRParaformerConverter
        from oasr.models.registry import resolve_architecture

        assert FunASRParaformerConverter().detect(Path(PARA_CKPT))
        assert resolve_architecture(Path(PARA_CKPT)) == "paraformer"

    def test_bundle(self):
        from oasr.models.registry import load_checkpoint_bundle

        arch, bundle = load_checkpoint_bundle(PARA_CKPT)
        assert arch == "paraformer"
        assert bundle.source_format == "funasr"
        cfg = bundle.model_config
        assert (cfg.vocab_size, cfg.input_size, cfg.encoder_num_blocks) == (8404, 560, 50)
        f = bundle.features
        assert (f.kind, f.window_type, f.lfr_m, f.lfr_n) == ("kaldi_fbank", "hamming", 7, 6)
        assert f.audio_scale == 32768.0
        assert bundle.tokenizer.kind == "funasr_char"
        assert bundle.decoding.default_decode_type == "paraformer"
        # am.mvn CMVN rides the state dict
        assert "encoder.cmvn_shift" in bundle.state_dict
        assert bundle.state_dict["encoder.cmvn_shift"].shape == (560,)

    def test_load_report_clean(self):
        from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

        arch, bundle = load_checkpoint_bundle(PARA_CKPT)
        model, cfg, report = instantiate_from_bundle(arch, bundle)
        assert not report.missing
        assert not report.dropped
        assert sorted(model.capabilities) == ["paraformer"]
        assert model.default_decode_type == "paraformer"
        assert model.streaming_kind == "none"

    @pytest.mark.slow
    def test_native_round_trip(self, tmp_path):
        pytest.importorskip("safetensors")
        from oasr.checkpoints.convert import convert_to_native
        from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

        out = tmp_path / "native"
        convert_to_native(PARA_CKPT, str(out))
        arch, bundle = load_checkpoint_bundle(out)
        assert (arch, bundle.source_format) == ("paraformer", "native")
        assert bundle.tokenizer.kind == "funasr_char"
        assert (bundle.features.lfr_m, bundle.features.window_type) == (7, "hamming")
        m2, _, _ = instantiate_from_bundle(arch, bundle)

        arch1, b1 = load_checkpoint_bundle(PARA_CKPT)
        m1, _, _ = instantiate_from_bundle(arch1, b1)
        sd1, sd2 = m1.state_dict(), m2.state_dict()
        assert set(sd1) == set(sd2)
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k]), k


# ---------------------------------------------------------------------------
# FunASR reference parity (CPU float32)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@needs_ckpt
@needs_ref
class TestFunASRParity:
    @pytest.fixture(scope="class")
    def ref(self):
        return torch.load(REF_PT, map_location="cpu", weights_only=False)

    @pytest.fixture(scope="class")
    def model(self):
        from pathlib import Path

        from oasr.models.paraformer.convert import FunASRParaformerConverter
        from oasr.models.paraformer.model import ParaformerModel

        conv = FunASRParaformerConverter()
        model = ParaformerModel.from_config(conv.build_config(Path(PARA_CKPT)))
        report = model.load_weights(conv.load_state_dict(Path(PARA_CKPT), "model.pt", "cpu"))
        assert not report.dropped and not report.missing
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def features(self, model):
        from pathlib import Path

        import torchaudio

        from oasr.features.batched import batched_fbank
        from oasr.features.lfr import apply_lfr_batch
        from oasr.models.paraformer.convert import FunASRParaformerConverter

        wav, sr = torchaudio.load(os.path.join(PARA_CKPT, "example", "asr_example.wav"))
        assert sr == 16000
        fcfg = FunASRParaformerConverter().build_feature_spec(Path(PARA_CKPT)).to_feature_config()
        wav_scaled = (wav.reshape(-1) * 32768.0).unsqueeze(0)
        feats80, flens = batched_fbank(wav_scaled, torch.tensor([wav_scaled.size(1)]), fcfg)
        return apply_lfr_batch(feats80, flens, fcfg.lfr_m, fcfg.lfr_n)

    def test_frontend_parity(self, ref, model, features):
        feats, _ = features
        with_cmvn = (feats + model.encoder.cmvn_shift) * model.encoder.cmvn_scale
        ref_feats = ref["encoder_inp"][0]
        assert with_cmvn.shape == ref_feats.shape
        assert (with_cmvn - ref_feats).abs().max() < 5e-4

    def test_full_pipeline_parity(self, ref, model, features):
        feats, flens = features
        with torch.no_grad():
            hidden, out_lens = model.encode_offline(feats, flens)
            assert (hidden - ref["encoder_out"][0]).abs().max() < 1e-3

            embeds, token_lens, fires = model.predict(hidden, out_lens)
            assert token_lens.tolist() == [int(ref["predictor_out"][1][0].item())]
            assert (embeds - ref["predictor_out"][0]).abs().max() < 1e-3
            assert torch.equal(fires >= 1.0, ref["predictor_out"][3] >= 1.0)

            log_probs = model.nar_decode(hidden, out_lens, embeds, token_lens)
            ids = log_probs[0, : token_lens[0]].argmax(-1)
            ref_ids = ref["decoder_out"][0][0].argmax(-1)
            assert torch.equal(ids, ref_ids)

    def test_transcript_parity(self, ref, model, features):
        from oasr.tokenizers import TokenizerSpec, build_tokenizer

        feats, flens = features
        with torch.no_grad():
            hidden, out_lens = model.encode_offline(feats, flens)
            embeds, token_lens, _ = model.predict(hidden, out_lens)
            log_probs = model.nar_decode(hidden, out_lens, embeds, token_lens)
        ids = log_probs[0, : token_lens[0]].argmax(-1).tolist()
        tok = build_tokenizer(
            TokenizerSpec(
                kind="funasr_char", files={"tokens": os.path.join(PARA_CKPT, "tokens.json")}
            )
        )
        assert tok.decode(ids) == ref["result"][0]["text"]


# ---------------------------------------------------------------------------
# Decode-strategy seams (no checkpoint needed)
# ---------------------------------------------------------------------------


class TestStrategySeams:
    def test_registered_and_consumes_hidden(self):
        from oasr.engine.decode.base import _REGISTRY
        from oasr.engine.decode.paraformer import ParaformerDecodeStrategy

        assert _REGISTRY["paraformer"] is ParaformerDecodeStrategy
        assert ParaformerDecodeStrategy.consumes == "hidden"
        assert ParaformerDecodeStrategy.incremental is False

    def test_needs_capable_model(self):
        from oasr.engine.config import EngineConfig
        from oasr.engine.decode import Detokenizer, build_decode_strategy

        cfg = EngineConfig(use_cuda_graphs=False)
        with pytest.raises(ValueError, match="predict"):
            build_decode_strategy("paraformer", cfg, Detokenizer(None, None), model=None)


# ---------------------------------------------------------------------------
# Engine end-to-end (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="engine requires CUDA")
@needs_ckpt
@needs_ref
class TestEngineE2E:
    @pytest.fixture(scope="class")
    def engine(self):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(ckpt_dir=PARA_CKPT, service_mode="offline", max_batch_size=4)
        eng = ASREngine(cfg)
        yield eng
        del eng
        torch.cuda.empty_cache()

    def test_transcribe_matches_reference(self, engine):
        import torchaudio

        ref = torch.load(REF_PT, map_location="cpu", weights_only=False)
        wav = torchaudio.load(os.path.join(PARA_CKPT, "example", "asr_example.wav"))[0].squeeze(0)
        texts = engine.transcribe_offline([wav])
        text = texts[0].text if hasattr(texts[0], "text") else texts[0]
        assert text == ref["result"][0]["text"]

    def test_timestamps_monotonic(self, engine):
        import torchaudio

        wav = torchaudio.load(os.path.join(PARA_CKPT, "example", "asr_example.wav"))[0].squeeze(0)
        engine.add_request(wav, request_id="ts", streaming=False)
        outs = []
        while not outs:
            outs = engine.step()
        o = outs[0]
        assert o.finished and o.timestamps is not None
        assert len(o.timestamps) == len(o.tokens[0])
        duration = wav.numel() / 16000.0
        prev_end = 0.0
        for start, end in o.timestamps:
            assert start >= prev_end - 1e-6 and end >= start
            prev_end = end
        assert prev_end <= duration + 0.5

    def test_streaming_rejected(self):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(ckpt_dir=PARA_CKPT, service_mode="streaming")
        with pytest.raises(ValueError, match="offline-only"):
            ASREngine(cfg)

    def test_wrong_decode_method_rejected(self):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(ckpt_dir=PARA_CKPT, service_mode="offline", decode_method="ctc")
        with pytest.raises(ValueError, match="not a capability"):
            ASREngine(cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPositionEncodingCache:
    """N1: the PE is a pure function — build it once, in fp32.

    It was rebuilt on every encoder forward (arange, exp, outer product, two
    trig passes, a cat) *in the compute dtype*, which under fp16 makes the
    integer position ladder exact only to 2048 — past that consecutive
    positions collide and every derived value is wrong.  FunASR computes in
    fp32, so this is a parity requirement too.
    """

    def _pe(self, length, depth=512, dtype=torch.float32):
        from oasr.models.paraformer.modules import sinusoidal_position_encoding

        return sinusoidal_position_encoding(length, depth, torch.device("cpu"), dtype)

    def test_shape_and_dtype(self):
        out = self._pe(37, depth=64, dtype=torch.float16)
        assert out.shape == (1, 37, 64)
        assert out.dtype == torch.float16

    def test_repeated_calls_agree(self):
        a = self._pe(64, depth=64)
        b = self._pe(64, depth=64)
        assert torch.equal(a, b)

    def test_a_longer_request_extends_the_table_consistently(self):
        """A cached prefix must not change when the table grows."""
        short = self._pe(16, depth=64).clone()
        long = self._pe(4000, depth=64)
        torch.testing.assert_close(long[:, :16], short)

    def test_fp16_positions_stay_exact_past_2048(self):
        """The reason for building in fp32.

        In fp16 ``arange(1, L+1)`` stops being exact at 2048, so positions 2049
        and 2050 both round to 2050 and produce *identical* encodings.  Built in
        fp32 and cast afterwards, they stay distinct.
        """
        pe = self._pe(3000, depth=64, dtype=torch.float16)
        assert not torch.equal(pe[0, 2048], pe[0, 2049]), "positions collapsed in fp16"

    def test_matches_a_direct_fp32_reference(self):
        import math

        depth, length = 64, 40
        positions = torch.arange(1, length + 1, dtype=torch.float32)
        inv = torch.exp(
            torch.arange(depth // 2, dtype=torch.float32) * -(math.log(10000.0) / (depth / 2 - 1))
        )
        st = positions.unsqueeze(1) * inv.unsqueeze(0)
        want = torch.cat([torch.sin(st), torch.cos(st)], dim=1).unsqueeze(0)
        torch.testing.assert_close(self._pe(length, depth=depth), want)
