#!/usr/bin/env python3
"""Tests for CTC + attention-decoder rescoring (``decode_method="ctc_aed_rescoring"``).

Three tiers:

* CPU fusion-math tests — the vectorized batched scoring must match a naive
  per-hypothesis WeNet ``attention_rescoring`` loop over the *same* decoder.
* Engine seam tests — capability validation, streaming rejection.
* GPU end-to-end on the real U2++ checkpoint (skipped when absent).
"""

from dataclasses import dataclass
from typing import List, Optional

import assets
import pytest
import torch

from oasr.models.decoders import (
    BiTransformerDecoder,
    TransformerDecoderConfig,
    add_sos_eos,
    reverse_pad_list,
)

# Declared in tests/assets.py so --strict-assets can make these fatal.
CKPT_DIR = assets.declared("CKPT_DIR")
WAV_DIR = assets.declared("WAV_DIR")


# ---------------------------------------------------------------------------
# CPU: vectorized fusion vs naive WeNet loop
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    tokens: List[List[List[int]]]
    scores: torch.Tensor
    lengths: Optional[torch.Tensor] = None


class _FakeModelConfig:
    def __init__(self, decoder_cfg):
        self.decoder = decoder_cfg


class _FakeModel:
    """The full ``ctc_aed_rescoring`` surface, with only the decoder half real.

    These tests drive the fusion arithmetic directly off a precomputed ``hidden``,
    so ``head`` / ``encode_offline`` are never called — but they are part of the
    capability contract (:data:`oasr.models.interfaces.CAPABILITIES`), which the
    strategy constructor validates, so a fake that omitted them would be claiming
    a capability it cannot serve.  Stub them rather than weaken the check.
    """

    def __init__(self, decoder, decoder_cfg):
        self.decoder = decoder
        self.config = _FakeModelConfig(decoder_cfg)
        self.head = object()

    def encode_offline(self, *args, **kwargs):  # pragma: no cover - contract only
        raise AssertionError("fusion-math tests pass `hidden` in directly")


def _tiny_decoder(seed=0, r_num_blocks=2):
    cfg = TransformerDecoderConfig(
        vocab_size=40,
        encoder_output_size=16,
        attention_heads=2,
        linear_units=32,
        num_blocks=2,
        r_num_blocks=r_num_blocks,
        sos_id=39,
        eos_id=39,
        reverse_weight=0.3,
    )
    torch.manual_seed(seed)
    return BiTransformerDecoder(cfg).eval(), cfg


def _naive_rescoring(decoder, cfg, hidden, enc_lens, result, ctc_weight, reverse_weight):
    """WeNet attention_rescoring per-hypothesis reference loop (B=1 forwards)."""
    B = hidden.size(0)
    beam = result.scores.size(1)
    picks, all_scores = [], []
    for b in range(B):
        T = int(enc_lens[b])
        memory = hidden[b : b + 1, :T]
        best_score, best_k, scores = -float("inf"), 0, []
        for k in range(beam):
            hyp = result.tokens[b][k]
            n = len(hyp)
            ys = torch.tensor([hyp], dtype=torch.long)
            ys_in, ys_out = add_sos_eos(
                ys if n else torch.full((1, 1), -1, dtype=torch.long), cfg.sos_id, cfg.eos_id, -1
            )
            r_ys_in, r_ys_out = add_sos_eos(
                reverse_pad_list(
                    ys if n else torch.full((1, 1), -1, dtype=torch.long),
                    torch.tensor([n]),
                    -1,
                ),
                cfg.sos_id,
                cfg.eos_id,
                -1,
            )
            with torch.no_grad():
                l_x, r_x = decoder(
                    memory,
                    torch.tensor([T]),
                    ys_in,
                    torch.tensor([n + 1]),
                    r_ys_in if reverse_weight > 0 else None,
                )
            lp = torch.log_softmax(l_x.float(), -1)[0]
            score = sum(lp[j, w].item() for j, w in enumerate(hyp)) + lp[n, cfg.eos_id].item()
            if reverse_weight > 0 and r_x is not None:
                r_lp = torch.log_softmax(r_x.float(), -1)[0]
                r_score = (
                    sum(r_lp[n - j - 1, w].item() for j, w in enumerate(hyp))
                    + r_lp[n, cfg.eos_id].item()
                )
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            score += ctc_weight * result.scores[b, k].item()
            scores.append(score)
            if score > best_score:
                best_score, best_k = score, k
        picks.append(best_k)
        all_scores.append(scores)
    return picks, all_scores


def _build_strategy(monkeypatch, tmp_path, model, result, **cfg_overrides):
    from oasr.engine.config import EngineConfig
    from oasr.engine.decode import rescoring as rescoring_mod
    from oasr.engine.decode.detokenize import Detokenizer

    monkeypatch.setattr(rescoring_mod, "ctc_beam_search_decode", lambda *a, **k: result)
    from oasr.functionals.ctc_decode import GpuDecoderConfig

    cfg = EngineConfig(
        ckpt_dir=str(tmp_path),
        service_mode="offline",
        ctc_decoder_config=GpuDecoderConfig(beam_size=result.scores.size(1)),
        **cfg_overrides,
    )
    return rescoring_mod.CtcAedRescoringStrategy(cfg, Detokenizer(None, None), model)


class TestScoreGathering:
    """H12: reading n*L scalars must not materialise an (n, L, V) log_softmax."""

    def _reference(self, logits, ys_out):
        """The previous implementation, verbatim."""
        from oasr.engine.decode.rescoring import _IGNORE_ID

        lp = torch.log_softmax(logits.float(), dim=-1)
        mask = ys_out != _IGNORE_ID
        idx = ys_out.masked_fill(~mask, 0).unsqueeze(-1)
        return (lp.gather(2, idx).squeeze(-1) * mask).sum(dim=1)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("n,L,V", [(1, 3, 17), (7, 5, 64), (70, 4, 33)])
    def test_matches_the_log_softmax_form(self, dtype, n, L, V):
        from oasr.engine.decode.rescoring import _IGNORE_ID, CtcAedRescoringStrategy

        torch.manual_seed(n * 100 + L)
        logits = torch.randn(n, L, V, dtype=dtype)
        ys_out = torch.randint(0, V, (n, L))
        # Ragged: pad the tail of every other row.
        ys_out[::2, -1] = _IGNORE_ID
        got = CtcAedRescoringStrategy._gather_scores(logits, ys_out)
        want = self._reference(logits, ys_out)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def test_row_count_beyond_one_chunk_is_covered(self):
        """``n`` larger than the chunk size must still cover every row."""
        from oasr.engine.decode.rescoring import _SCORE_CHUNK_ROWS, CtcAedRescoringStrategy

        n = _SCORE_CHUNK_ROWS * 2 + 3
        torch.manual_seed(1)
        logits = torch.randn(n, 3, 11)
        ys_out = torch.randint(0, 11, (n, 3))
        got = CtcAedRescoringStrategy._gather_scores(logits, ys_out)
        torch.testing.assert_close(got, self._reference(logits, ys_out), rtol=1e-5, atol=1e-5)
        assert got.shape == (n,)


class TestFusionMath:
    @pytest.mark.parametrize("reverse_weight", [0.0, 0.3])
    def test_vectorized_matches_naive_loop(self, monkeypatch, tmp_path, reverse_weight):
        decoder, cfg = _tiny_decoder()
        model = _FakeModel(decoder, cfg)
        torch.manual_seed(7)
        B, T, beam = 2, 9, 3
        hidden = torch.randn(B, T, cfg.encoder_output_size)
        enc_lens = torch.tensor([T, T - 3], dtype=torch.int32)
        tokens = [
            [[5, 6, 7], [5, 6], [5, 6, 7, 8]],
            [[10, 11], [], [12]],  # includes an empty hypothesis
        ]
        result = _FakeResult(
            tokens=tokens, scores=torch.tensor([[-1.0, -1.5, -2.0], [-0.5, -3.0, -1.2]])
        )

        strat = _build_strategy(
            monkeypatch,
            tmp_path,
            model,
            result,
            rescoring_ctc_weight=0.5,
            rescoring_reverse_weight=reverse_weight,
        )
        from oasr.engine.decode.base import EncodeOutput

        outs = strat.decode_offline(
            EncodeOutput(hidden=hidden, log_probs=torch.zeros(B, T, 40)), enc_lens
        )

        picks, ref_scores = _naive_rescoring(
            decoder, cfg, hidden, enc_lens, result, ctc_weight=0.5, reverse_weight=reverse_weight
        )
        for b in range(B):
            assert outs[b].tokens[0] == tokens[b][picks[b]], f"row {b}: pick mismatch"
            ours = {tuple(t): s for t, s in zip(outs[b].tokens, outs[b].scores)}
            for k in range(beam):
                assert ours[tuple(tokens[b][k])] == pytest.approx(ref_scores[b][k], abs=1e-4)
            # n-best must be ordered by fused score, descending
            assert outs[b].scores == sorted(outs[b].scores, reverse=True)

    def test_trained_reverse_weight_default(self, monkeypatch, tmp_path):
        """rescoring_reverse_weight=None must fall back to the checkpoint's
        trained reverse_weight (0.3 here), i.e. reproduce the explicit run."""
        decoder, cfg = _tiny_decoder()
        model = _FakeModel(decoder, cfg)
        torch.manual_seed(8)
        hidden = torch.randn(1, 7, cfg.encoder_output_size)
        enc_lens = torch.tensor([7], dtype=torch.int32)
        result = _FakeResult(tokens=[[[4, 5], [4]]], scores=torch.tensor([[-1.0, -1.1]]))

        from oasr.engine.decode.base import EncodeOutput

        enc = EncodeOutput(hidden=hidden, log_probs=torch.zeros(1, 7, 40))
        s_default = _build_strategy(monkeypatch, tmp_path, model, result)
        s_explicit = _build_strategy(
            monkeypatch, tmp_path, model, result, rescoring_reverse_weight=0.3
        )
        assert s_default.decode_offline(enc, enc_lens)[0].scores == pytest.approx(
            s_explicit.decode_offline(enc, enc_lens)[0].scores
        )

    def test_out_of_vocab_hypothesis_disqualified(self, monkeypatch, tmp_path):
        """Ids past the decoder vocab (CTC-head padding) must never win."""
        decoder, cfg = _tiny_decoder()
        model = _FakeModel(decoder, cfg)
        hidden = torch.randn(1, 5, cfg.encoder_output_size)
        enc_lens = torch.tensor([5], dtype=torch.int32)
        # First beam contains an id >= vocab_size(40) with a huge CTC score.
        result = _FakeResult(tokens=[[[45], [4, 5]]], scores=torch.tensor([[100.0, -2.0]]))

        strat = _build_strategy(monkeypatch, tmp_path, model, result)
        from oasr.engine.decode.base import EncodeOutput

        out = strat.decode_offline(
            EncodeOutput(hidden=hidden, log_probs=torch.zeros(1, 5, 40)), enc_lens
        )[0]
        assert out.tokens[0] == [4, 5]
        assert out.scores[-1] == float("-inf")

    def test_model_without_decoder_rejected(self, tmp_path):
        from oasr.engine.config import EngineConfig
        from oasr.engine.decode.detokenize import Detokenizer
        from oasr.engine.decode.rescoring import CtcAedRescoringStrategy

        cfg = EngineConfig(ckpt_dir=str(tmp_path), service_mode="offline")

        class _NoDecoder:
            config = _FakeModelConfig(None)
            head = object()

            def encode_offline(self, *a, **k):
                raise AssertionError("unreachable")

        with pytest.raises(ValueError, match=r"ctc_aed_rescoring.*is missing:.*\bdecoder\b"):
            CtcAedRescoringStrategy(cfg, Detokenizer(None, None), _NoDecoder())


# ---------------------------------------------------------------------------
# Engine seams
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="engine requires CUDA")
class TestEngineSeams:
    @pytest.fixture(scope="class")
    def ckpt(self):
        return assets.require("CKPT_DIR")

    def test_unknown_decode_method_rejected(self, ckpt):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(ckpt_dir=ckpt, service_mode="offline", decode_method="bogus")
        with pytest.raises(ValueError, match="not a capability"):
            ASREngine(cfg)

    def test_streaming_rescoring_rejected(self, ckpt):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(
            ckpt_dir=ckpt, service_mode="streaming", decode_method="ctc_aed_rescoring"
        )
        with pytest.raises(ValueError, match="offline-only"):
            ASREngine(cfg)


# ---------------------------------------------------------------------------
# GPU end-to-end on the real checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="engine requires CUDA")
class TestEngineRescoringE2E:
    @pytest.fixture(scope="class")
    def engine(self):
        assets.require("CKPT_DIR")
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(
            ckpt_dir=CKPT_DIR,
            service_mode="offline",
            decode_method="ctc_aed_rescoring",
            max_batch_size=4,
        )
        eng = ASREngine(cfg)
        yield eng
        del eng
        torch.cuda.empty_cache()

    @pytest.fixture(scope="class")
    def audios(self):
        wavs = assets.require_wavs(12)
        import torchaudio

        loaded = [torchaudio.load(w)[0].squeeze(0) for w in wavs]
        # Batch comparable lengths: the CuteDSL FMHA has a pre-existing NaN
        # bug on heavily key-padded rows (skewed-length batches silently
        # decode to empty) — an encoder kernel issue, not a rescoring one.
        loaded.sort(key=lambda a: a.numel())
        return loaded[-4:]

    def test_transcribe_offline_rescored(self, engine, audios):
        texts = engine.transcribe_offline(audios)
        assert len(texts) == len(audios)
        for t in texts:
            text = t.text if hasattr(t, "text") else t
            assert isinstance(text, str) and len(text) > 0
