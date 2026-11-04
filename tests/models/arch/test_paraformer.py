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
import random
from types import SimpleNamespace

import assets
import numpy as np
import pytest
import torch
from conftest import assert_native_weights_round_trip

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

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_cuda_gather_matches_the_reference_path(self, dtype, monkeypatch):
        from oasr.features.lfr import apply_lfr_batch

        monkeypatch.delenv("OASR_FEATURE_BACKEND", raising=False)
        torch.manual_seed(8)
        lens = torch.tensor([218, 1, 13, 100])
        feats = torch.randn(4, 218, 8, dtype=dtype)
        for b, length in enumerate(lens.tolist()):
            feats[b, length:] = 0
        expected, expected_lens = apply_lfr_batch(feats, lens, 7, 6, max_length=218)
        got, got_lens = apply_lfr_batch(feats.cuda(), lens.cuda(), 7, 6, max_length=218)
        assert torch.equal(got_lens.cpu(), expected_lens)
        assert torch.equal(got.cpu(), expected)

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_gather_at_the_served_shape(self):
        """80 mel / LFR 7/6 / 30 s — the shape Paraformer actually runs.

        The 8-dim fixture above is below one 16-byte transfer per feature row,
        so it exercises neither the vectorized path's inner loop nor a feature
        row that spans several of them.
        """
        from oasr.features.lfr import apply_lfr_batch

        torch.manual_seed(12)
        T = 3000
        lens = torch.tensor([T, 2999, 7, 1801], dtype=torch.int32)
        feats = torch.randn(4, T, 80)
        for b, length in enumerate(lens.tolist()):
            feats[b, length:] = 0
        expected, expected_lens = apply_lfr_batch(feats, lens, 7, 6, max_length=T)
        got, got_lens = apply_lfr_batch(feats.cuda(), lens.cuda(), 7, 6, max_length=T)
        assert got.shape == (4, 500, 560)
        assert torch.equal(got_lens.cpu(), expected_lens)
        assert torch.equal(got.cpu(), expected)

    # A feature row is copied 16 bytes at a time when the row divides into whole
    # vectors *and* both tensors are 16-byte aligned; otherwise element by
    # element.  Neither condition is a property of the config, so both branches
    # need a case -- and every in-tree feature dim (80, and the 8 of the fixture
    # above) satisfies both, which is exactly why they are easy to leave untested.
    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("feature_dim", [5, 8, 17, 80])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_cuda_gather_scalar_and_vector_paths_agree(self, feature_dim, dtype):
        from oasr.features.lfr import apply_lfr_batch

        torch.manual_seed(feature_dim)
        lens = torch.tensor([53, 1, 20, 53], dtype=torch.int32)
        feats = torch.randn(4, 53, feature_dim, dtype=dtype)
        for b, length in enumerate(lens.tolist()):
            feats[b, length:] = 0
        expected, _ = apply_lfr_batch(feats, lens, 7, 6, max_length=53)
        got, _ = apply_lfr_batch(feats.cuda(), lens.cuda(), 7, 6, max_length=53)
        assert torch.equal(got.cpu(), expected)

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_gather_handles_an_unaligned_input_view(self):
        """A contiguous view whose storage offset is not a 16-byte multiple.

        ``.contiguous()`` does not copy it — it is already contiguous — so the
        vectorized path would reinterpret an unaligned pointer as ``int4``.
        """
        from oasr.features.lfr import apply_lfr_batch

        torch.manual_seed(31)
        B, T, F = 2, 40, 80
        flat = torch.randn(1 + B * T * F, device="cuda")
        feats = flat[1:].view(B, T, F)  # storage offset 1 float = 4 bytes
        assert feats.is_contiguous() and feats.data_ptr() % 16 != 0
        lens = torch.full((B,), T, dtype=torch.int32, device="cuda")

        got, _ = apply_lfr_batch(feats, lens, 7, 6, max_length=T)
        expected, _ = apply_lfr_batch(feats.cpu(), lens.cpu(), 7, 6, max_length=T)
        assert torch.equal(got.cpu(), expected)

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


def _clean_by_replace(word):
    """FunASR's ``_clean``, as ported: five ``replace`` calls, in this order."""
    for junk in (" ", "</s>", "<s>", "<unk>", "<OOV>"):
        word = word.replace(junk, "")
    return word


def _as_ported(monkeypatch):
    """Put ``funasr_char`` back on FunASR's code path: the five-replace ``_clean``,
    and the two predicates, uncached, wherever the decode now calls a cached one."""
    import oasr.tokenizers.funasr_char as fc

    monkeypatch.setattr(fc, "_clean", _clean_by_replace)
    monkeypatch.setattr(fc, "_tokens_all_chinese", fc._is_all_chinese)
    monkeypatch.setattr(fc, "_tokens_all_alpha", fc._is_all_alpha)
    monkeypatch.setattr(fc, "_token_all_chinese", fc._is_all_chinese)
    monkeypatch.setattr(fc, "_token_all_alpha", fc._is_all_alpha)


class TestCleanFastPath:
    """``_clean`` returns a word with neither ``<`` nor a space untouched (audit L7).

    Exact only because every junk string but the space contains ``<``. Nested
    junk, where one removal exposes another, still has to come out as the
    sequential replaces leave it.
    """

    def test_it_matches_the_five_replaces(self):
        from oasr.tokenizers.funasr_char import _clean

        cases = ["", " ", "a", "<", ">", "<s>", "</s>", "<unk>", "<OOV>", "a b", "<<s>s>"]
        cases += ["</</s>s>", "x<unk>y", "你好", "@@", "it's", "<s", "s>", " <s> ", "<un k>"]
        alphabet = ["<", ">", "/", "s", "u", "n", "k", "O", "V", " ", "a", "好", "<s>", "</s>"]
        rng = random.Random(3)
        for _ in range(20000):
            cases.append("".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 10))))
        for word in cases:
            assert _clean(word) == _clean_by_replace(word), word

    @pytest.mark.parametrize("tokens,expected", _POSTPROCESS_CASES)
    def test_the_ported_path_still_matches_funasr(self, tokens, expected, monkeypatch):
        """The reference the cached decode is compared against is itself right."""
        from oasr.tokenizers.funasr_char import sentence_postprocess

        _as_ported(monkeypatch)
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

    def test_pieces_give_an_abbreviation_letter_by_letter(self, tok):
        """Word timings are cut from these; see ``test_tokenizers.py``."""
        ids = [tok._ids[t] for t in ("we", "are", "b", "b", "c")]
        assert tok.token_pieces(ids) == ["we", " are", " B", "B", "C"]
        assert tok.decode(ids) == "we are BBC"

    def test_clean_matches_over_the_real_vocabulary(self, tok):
        """Every token, and every character of every token, as decode cleans them."""
        from oasr.tokenizers.funasr_char import _clean

        words = list(tok._tokens) + sorted({ch for t in tok._tokens for ch in t})
        assert all(_clean(w) == _clean_by_replace(w) for w in words)

    def test_the_cached_decode_matches_the_ported_one(self, tok, monkeypatch):
        """Audit L7 caches the per-token predicates and short-circuits ``_clean``.
        The text and the pieces must be what FunASR's path, ported as is, gives,
        on sequences weighted toward the mixed Chinese and English branch, where
        both are used most."""
        english = [i for i, t in enumerate(tok._tokens) if t.isascii() and t.isalpha()]
        pieces_at = [i for i, t in enumerate(tok._tokens) if t.endswith("@@")]
        pools = (english, pieces_at, range(tok.vocab_size))
        rng = random.Random(7)
        seqs = [
            [rng.choice(rng.choice(pools)) for _ in range(rng.randrange(0, 40))]
            for _ in range(3000)
        ]
        fast = [(tok.decode(ids), tok.token_pieces(ids)) for ids in seqs]
        _as_ported(monkeypatch)
        ported = [(tok.decode(ids), tok.token_pieces(ids)) for ids in seqs]
        assert fast == ported

    def test_pieces_join_to_decode_over_the_real_vocabulary(self, tok):
        """Random ids weighted toward the English pieces and single letters,
        which is where the merges happen."""
        english = [i for i, t in enumerate(tok._tokens) if t.isascii() and t.isalpha()]
        pieces_at = [i for i, t in enumerate(tok._tokens) if t.endswith("@@")]
        pools = (english, pieces_at, range(tok.vocab_size))
        rng = random.Random(14)
        for _ in range(5000):
            ids = [rng.choice(rng.choice(pools)) for _ in range(rng.randrange(0, 24))]
            assert "".join(tok.token_pieces(ids)) == tok.decode(ids), ids


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
        """The weight comparison lives in ``conftest``; what is FunASR's is here."""
        bundle, _converted = assert_native_weights_round_trip(PARA_CKPT, tmp_path, "paraformer")
        assert bundle.tokenizer.kind == "funasr_char"
        assert (bundle.features.lfr_m, bundle.features.window_type) == (7, "hamming")


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
# Post-processing: batched list work, exactly the old per-token loop (audit L7)
# ---------------------------------------------------------------------------

#: blank, sos, eos of the stub model; every one of them is filtered out of a row.
_FILTERED = (0, 1, 2)


class _StubParaformer:
    """``predict`` / ``nar_decode`` returning fixed tensors, so the strategy's
    post-processing can be driven over any shape of batch without a model."""

    def __init__(self, log_probs, token_lens, fires):
        self.config = SimpleNamespace(blank_id=0, sos_id=1, eos_id=2)
        self._log_probs, self._token_lens, self._fires = log_probs, token_lens, fires

    def predict(self, enc_out, enc_lens, return_alphas=False):
        B, U = self._log_probs.shape[:2]
        return torch.zeros(B, U, 4), self._token_lens, self._fires

    def nar_decode(self, enc_out, enc_lens, acoustic_embeds, token_lens):
        return self._log_probs


def _strategy(model, detok):
    from oasr.engine.decode.alignment import FrameClock
    from oasr.engine.decode.paraformer import ParaformerDecodeStrategy

    strategy = ParaformerDecodeStrategy.__new__(ParaformerDecodeStrategy)
    strategy._model, strategy._detok = model, detok
    strategy._clock = FrameClock(0.06)
    strategy._filtered_ids = set(_FILTERED)
    return strategy


def _reference_rows(strategy, log_probs, token_lens, fires, requests):
    """The per-row, per-token post-processing as it was before audit L7, verbatim
    apart from inlining ``_token_spans``: the oracle the batched code must equal."""
    from oasr.engine.decode.alignment import TokenAlignment, wants_word_timings
    from oasr.engine.request import RequestOutput

    best_lp, best_ids = log_probs.float().max(dim=-1)
    ids_cpu, lp_cpu, lens_cpu, fires_cpu = best_ids, best_lp, token_lens, fires
    outputs = []
    for b in range(log_probs.size(0)):
        n = int(lens_cpu[b].item())
        if n <= 0:
            outputs.append(strategy._empty_output())
            continue
        row_ids = ids_cpu[b, :n].tolist()
        score = float(lp_cpu[b, :n].sum().item())
        fire_idx = (fires_cpu[b] >= 1.0).nonzero(as_tuple=False).reshape(-1).tolist()
        spans, prev = [], 0.0
        for k in range(n):
            end = float(fire_idx[k] if k < len(fire_idx) else (fire_idx[-1] if fire_idx else 0))
            end = max(end, prev)
            spans.append((prev, end))
            prev = end
        probs = lp_cpu[b, :n].exp().tolist()
        kept = []
        for k, tok in enumerate(row_ids):
            if tok in strategy._filtered_ids:
                continue
            start, end = spans[k]
            kept.append(TokenAlignment(int(tok), start, end, float(min(max(probs[k], 0.0), 1.0))))
        out = RequestOutput(
            request_id="",
            text=strategy._detok.detokenize([a.token for a in kept]),
            tokens=[[a.token for a in kept]],
            scores=[score],
            finished=True,
        )
        strategy.attach_alignment(
            out, kept, words=requests is not None and wants_word_timings(requests[b])
        )
        outputs.append(out)
    return outputs


def _published(out):
    return (
        out.text,
        out.tokens,
        out.scores,
        out.finished,
        out.timestamps,
        out.confidence,
        out.words,
    )


@pytest.mark.requires_assets("OASR_C")  # the span pass is C++ only
class TestPostProcessing:
    def _batch(self, rng, B, U, T, V=12):
        """Log-probs that put the filtered ids on top now and then, lengths from
        empty to full, and fire counts below, at and above each row's length."""
        logits = torch.randn(B, U, V, generator=rng) * 3
        logits[..., :3] += torch.randn(B, U, 3, generator=rng).clamp(min=0) * 4
        log_probs = torch.log_softmax(logits, dim=-1)
        # A max log-softmax can round just above 0, which is what the
        # confidence clamp is for, so plant a few.
        hot = torch.rand(B, U, generator=rng) < 0.1
        top = log_probs.argmax(dim=-1, keepdim=True)
        log_probs.scatter_(
            -1, top, torch.where(hot, 3e-7, log_probs.gather(-1, top)[..., 0])[..., None]
        )
        token_lens = torch.randint(0, U + 1, (B,), generator=rng, dtype=torch.int32)
        fires = torch.rand(B, T + 1, generator=rng) * 0.99
        for b in range(B):
            k = int(torch.randint(0, T + 1, (1,), generator=rng))
            frames = torch.randperm(T + 1, generator=rng)[:k]
            fires[b, frames] += 1.0
        return log_probs, token_lens, fires

    @pytest.mark.parametrize("words", [False, True], ids=["timings", "words"])
    def test_every_published_field_matches_the_per_token_loop(self, words):
        from oasr.engine.decode import Detokenizer
        from oasr.engine.request import DecodingOptions, Request

        rng = torch.Generator().manual_seed(21)
        detok = Detokenizer(None, None)
        for trial in range(60):
            B, U, T = 1 + trial % 9, 1 + trial % 13, 2 + trial % 17
            log_probs, token_lens, fires = self._batch(rng, B, U, T)
            if trial % 10 == 0:
                token_lens[0] = 0  # an empty row beside full ones
            requests = [
                Request(audio=None, decoding=DecodingOptions(word_timestamps=words))
                for _ in range(B)
            ]
            strategy = _strategy(_StubParaformer(log_probs, token_lens, fires), detok)
            enc = torch.zeros(B, T, 4)
            got = strategy.decode_offline(enc, torch.full((B,), T), requests)
            expected = _reference_rows(strategy, log_probs, token_lens, fires, requests)
            assert [_published(o) for o in got] == [_published(o) for o in expected], trial

    def test_spans_reuse_the_last_fire_when_tokens_outnumber_fires(self):
        from oasr.engine.decode.paraformer import ParaformerDecodeStrategy as S

        assert S._token_spans([3, 7], 4) == ([0, 3, 7, 7], [3, 7, 7, 7])
        assert S._token_spans([3, 7, 9], 2) == ([0, 3], [3, 7])
        assert S._token_spans([], 3) == ([0, 0, 0], [0, 0, 0])

    def test_fire_frames_come_back_ascending_per_row(self):
        from oasr.engine.decode.paraformer import ParaformerDecodeStrategy as S

        fires = torch.tensor([[0.2, 1.1, 0.5, 1.0], [0.0, 0.3, 0.9, 0.1], [1.4, 0.0, 2.0, 1.0]])
        assert S._fire_frames(fires, 3) == [[1, 3], [], [0, 2, 3]]


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


class TestFsmnDepthwiseFusion:
    """KG2: asymmetric padding and both mask multiplies stay behind the waist."""

    @pytest.mark.parametrize("sanm_shift", [0, 2, 5])
    def test_matches_the_unfused_funasr_expression(self, sanm_shift):
        import torch.nn.functional as F

        from oasr.models.paraformer.modules import FsmnBlock

        torch.manual_seed(0)
        block = FsmnBlock(16, kernel_size=11, sanm_shift=sanm_shift).eval()
        x = torch.randn(2, 23, 16)
        mask = (torch.rand(2, 23, 1) > 0.2).to(x.dtype)

        got = block(x, mask)
        masked = x * mask
        left = 5 + sanm_shift
        right = 10 - left
        conv = F.conv1d(
            F.pad(masked.transpose(1, 2), (left, right)),
            block.fsmn_block.weight.permute(2, 1, 0),
            groups=16,
        ).transpose(1, 2)
        torch.testing.assert_close(got, (conv + masked) * mask)

    def test_sanm_shift_is_owned_by_the_depthwise_layer(self):
        from oasr.models.paraformer.modules import FsmnBlock, SanmSelfAttention

        assert FsmnBlock(16, 11, sanm_shift=2).fsmn_block.padding == (7, 3)
        assert SanmSelfAttention(2, 16, 16, 11, sanm_shift=3).fsmn_block.padding == (8, 2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel path requires CUDA")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sanm_self_attention_runs_on_the_kernel_path(self, dtype):
        """The encoder block, in a served dtype, on the real strided ``v``.

        ``test_matches_the_unfused_funasr_expression`` above runs on CPU in fp32
        and exercises ``FsmnBlock``, whose input is already contiguous — so it
        reaches neither the CUDA kernel nor the call site that broke.  This one
        drives ``SanmSelfAttention``, where ``v`` is a last-dim slice of the
        fused QKV projection with a row stride of ``3 * n_feat``, in fp16 and
        bf16 on CUDA, which is the only configuration the defect appeared in:
        every Paraformer request returned an empty transcript because the
        depthwise launcher rejected that view.

        ``_forward_fsmn`` owns the ``.contiguous()``; the layer asserts.  Drop
        that call and this fails with the assertion, in both dtypes.
        """
        from oasr.layers import layers_backend_override
        from oasr.models.paraformer.modules import SanmSelfAttention

        torch.manual_seed(0)
        n_head, n_feat, t = 4, 64, 23
        block = SanmSelfAttention(n_head, n_feat, n_feat, kernel_size=11).cuda().to(dtype).eval()
        x = torch.randn(2, t, n_feat, device="cuda", dtype=dtype)
        mask = torch.ones(2, 1, t, device="cuda", dtype=torch.bool)
        mask[1, :, t - 5 :] = False  # right padding, as the encoder supplies

        with torch.inference_mode():
            got = block(x, mask)
            with layers_backend_override("torch"):
                ref = block(x, mask)

        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPositionEncoding:
    """N1: the PE is a pure function of the length — computed in fp32.

    It was computed *in the compute dtype*, which under fp16 makes the integer
    position ladder exact only to 2048 — past that consecutive positions collide
    and every derived value is wrong.  FunASR computes in fp32, so this is a
    parity requirement too.  It is computed per call rather than cached: a cache
    that grew freed the table captured graphs read (see
    ``tests/models/test_positional_encoding.py``).
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

    def test_a_longer_request_agrees_on_the_shared_prefix(self):
        """A position's encoding does not depend on how long the request is."""
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


# ---------------------------------------------------------------------------
# Cross-attention key padding
# ---------------------------------------------------------------------------


class TestCrossAttentionPadding:
    """The NAR decoder must not attend to encoder padding.

    The memory mask used to reach SDPA as a ``(B, 1, T)`` tensor cast to the
    activation dtype.  SDPA reads a non-bool ``attn_mask`` as **additive**, so
    that ``{0.0, 1.0}`` tensor added ``+1.0`` to valid logits and left padded
    keys fully attendable.  A constant shift cancels in softmax, so it was
    invisible at ``B=1`` and with equal-length rows — which is every shape the
    FunASR oracle exercises — and only corrupted mixed-length batches.

    The gate is batch invariance: decoding an utterance alone and inside a
    padded batch must agree.  It fails on the pre-fix module.
    """

    def _decoder(self, seed=0):
        from oasr.models.paraformer.config import ParaformerModelConfig
        from oasr.models.paraformer.decoder import ParaformerSANMDecoder

        torch.manual_seed(seed)
        cfg = ParaformerModelConfig(
            vocab_size=32,
            encoder_output_size=32,
            decoder_attention_heads=2,
            decoder_linear_units=64,
            decoder_att_layer_num=2,
            decoder_num_blocks=2,
        )
        return ParaformerSANMDecoder(cfg).eval()

    def test_padded_batch_matches_single(self):
        dec = self._decoder()
        torch.manual_seed(1)
        short_t, long_t, u = 5, 12, 3
        memory = torch.randn(2, long_t, 32)
        memory[0, short_t:] = 0.0  # row 0 is padded past `short_t`
        embeds = torch.randn(2, u, 32)
        lens = torch.tensor([short_t, long_t])
        toks = torch.tensor([u, u])

        with torch.no_grad():
            batched, _ = dec(memory, lens, embeds, toks)
            solo, _ = dec(memory[:1, :short_t], lens[:1], embeds[:1], toks[:1])
        torch.testing.assert_close(batched[:1], solo, atol=1e-5, rtol=1e-5)

    def test_padding_content_is_ignored(self):
        """Changing what sits in the padded region must not move the output."""
        dec = self._decoder(seed=2)
        torch.manual_seed(3)
        short_t, long_t, u = 4, 10, 2
        memory = torch.randn(2, long_t, 32)
        embeds = torch.randn(2, u, 32)
        lens = torch.tensor([short_t, long_t])
        toks = torch.tensor([u, u])

        other = memory.clone()
        other[0, short_t:] = torch.randn(long_t - short_t, 32) * 50.0
        with torch.no_grad():
            a, _ = dec(memory, lens, embeds, toks)
            b, _ = dec(other, lens, embeds, toks)
        torch.testing.assert_close(a[:1], b[:1], atol=1e-5, rtol=1e-5)
