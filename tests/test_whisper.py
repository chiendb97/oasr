#!/usr/bin/env python3
"""Tests for the Whisper package: frontend, model, converter, aed strategy.

CPU tests use a tiny random Whisper (1 s window); the end-to-end engine tests
use the real ``openai/whisper-tiny`` snapshot at ``WHISPER_CKPT`` and skip
when it is absent.
"""

import assets
import pytest
import torch

from oasr.features import FeatureConfig
from oasr.features.whisper import batched_whisper_logmel
from oasr.models.whisper import WhisperModel, WhisperModelConfig

# Gated inputs are declared once in tests/assets.py; `declared()` gives the
# path the suite would use and the `requires_assets` marker does the gating,
# so a missing snapshot can be made fatal with --strict-assets.
WHISPER_CKPT = assets.declared("WHISPER_CKPT")
WAV_DIR = assets.declared("WAV_DIR")


def _tiny_config(**overrides):
    base = {
        "vocab_size": 64,
        "d_model": 32,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "encoder_ffn_dim": 64,
        "decoder_ffn_dim": 64,
        "num_mel_bins": 80,
        "max_source_positions": 50,  # 1 s window → 100 frames → 50 positions
        "max_target_positions": 32,
        "decoder_start_token_id": 60,
        "eos_token_id": 61,
        "forced_decoder_ids": [(1, 62)],
        "suppress_tokens": [0, 1],
        "begin_suppress_tokens": [2],
    }
    base.update(overrides)
    return WhisperModelConfig(**base)


def _multilingual_config(**overrides):
    """A tiny config shaped like a *multilingual* Whisper prompt.

    ``[SOT, <|lang|>, <|task|>, <|notimestamps|>]`` — the four-token sequence
    whose language and task slots a per-request option substitutes.
    """
    base = {
        "forced_decoder_ids": [(1, 50), (2, 55), (3, 59)],
        "task_token_ids": {"transcribe": 55, "translate": 56},
        "language_token_ids": {"en": 50, "fr": 51, "de": 52},
    }
    base.update(overrides)
    return _tiny_config(**base)


# ---------------------------------------------------------------------------
# Log-mel frontend
# ---------------------------------------------------------------------------


class TestWhisperLogmel:
    CFG = FeatureConfig(feature_type="whisper_logmel", whisper_chunk_seconds=1.0)

    def test_shapes_and_true_lengths(self):
        wav = torch.randn(3, 12000) * 0.1
        feats, lens = batched_whisper_logmel(wav, torch.tensor([12000, 8000, 4000]), self.CFG)
        # Features always span the padded window; lengths carry the real
        # ceil(len/hop) frame counts (HF attention-mask semantics — the
        # Whisper encoder ignores them, the Qwen2-Audio tower masks by them).
        assert feats.shape == (3, 100, 80)
        assert lens.tolist() == [75, 50, 25]

    def test_normalization_range(self):
        wav = torch.randn(1, 16000) * 0.5
        feats, _ = batched_whisper_logmel(wav, torch.tensor([16000]), self.CFG)
        # (log10 clamped to [max-8, max] + 4) / 4 → span <= 2
        assert feats.max() - feats.min() <= 2.0 + 1e-5

    def test_padding_invariance(self):
        """Extra zero padding past the valid length must not change features."""
        torch.manual_seed(0)
        wav = torch.randn(4000) * 0.3
        a, _ = batched_whisper_logmel(wav.unsqueeze(0), torch.tensor([4000]), self.CFG)
        padded = torch.cat([wav, torch.randn(8000)]).unsqueeze(0)  # garbage tail
        b, _ = batched_whisper_logmel(padded, torch.tensor([4000]), self.CFG)
        assert torch.allclose(a, b, atol=1e-6)

    def test_batch_rows_independent(self):
        torch.manual_seed(1)
        w1, w2 = torch.randn(5000) * 0.2, torch.randn(7000) * 0.9
        both, _ = batched_whisper_logmel(
            torch.stack([torch.nn.functional.pad(w1, (0, 2000)), w2]),
            torch.tensor([5000, 7000]),
            self.CFG,
        )
        solo, _ = batched_whisper_logmel(w1.unsqueeze(0), torch.tensor([5000]), self.CFG)
        assert torch.allclose(both[0], solo[0], atol=1e-5)

    @pytest.mark.requires_assets("WHISPER_CKPT")
    def test_matches_transformers_feature_extractor(self):
        transformers = pytest.importorskip("transformers")
        fe = transformers.WhisperFeatureExtractor()
        torch.manual_seed(2)
        wav = (torch.randn(24000) * 0.1).numpy()
        ref = torch.tensor(fe(wav, sampling_rate=16000, return_tensors="np").input_features[0])
        ours, _ = batched_whisper_logmel(
            torch.tensor(wav).unsqueeze(0),
            torch.tensor([len(wav)]),
            FeatureConfig(feature_type="whisper_logmel"),
        )
        assert torch.allclose(ours[0].t(), ref, atol=1e-4)


# ---------------------------------------------------------------------------
# Decoder incremental surface
# ---------------------------------------------------------------------------


class TestDecoderIncremental:
    def test_step_matches_prefill(self):
        """KV-cached step-by-step logits == one big teacher-forced prefill."""
        cfg = _tiny_config()
        torch.manual_seed(3)
        model = WhisperModel(cfg).eval()
        B, T_enc = 2, cfg.max_source_positions
        enc = torch.randn(B, T_enc, cfg.d_model)
        seq = torch.randint(3, 59, (B, 8))

        with torch.no_grad():
            # Reference: prefill over the whole sequence at once.
            ref_logits, _ = model.decoder.prefill(enc, seq)
            # Incremental: prefill the first 4, then step the remaining 4.
            logits, state = model.decoder.prefill(enc, seq[:, :4])
            for t in range(4, 8):
                logits, state = model.decoder.step(seq[:, t], state)
        assert torch.allclose(logits, ref_logits, atol=1e-4)

    def test_select_drops_rows(self):
        cfg = _tiny_config()
        torch.manual_seed(4)
        model = WhisperModel(cfg).eval()
        enc = torch.randn(3, cfg.max_source_positions, cfg.d_model)
        seq = torch.randint(3, 59, (3, 5))
        with torch.no_grad():
            _, state = model.decoder.prefill(enc, seq)
            state1 = model.decoder.select(state, torch.tensor([2]))
            logits1, _ = model.decoder.step(seq[2, -1:].expand(1), state1)
            _, solo = model.decoder.prefill(enc[2:3], seq[2:3])
            logits_solo, _ = model.decoder.step(seq[2, -1:], solo)
        assert torch.allclose(logits1, logits_solo, atol=1e-4)


# ---------------------------------------------------------------------------
# Per-request task / language (H5)
# ---------------------------------------------------------------------------


class TestTaskAndLanguagePrompt:
    """The SOT slots that used to be frozen at conversion time.

    ``forced_decoder_ids`` pinned language and task when the checkpoint was
    converted, which is why ``POST /v1/audio/translations`` could not be served
    at all: there was no request-level way to say "translate".
    """

    def test_no_override_is_the_checkpoints_own_prompt(self):
        cfg = _multilingual_config()
        assert cfg.sot_sequence() == [60, 50, 55, 59]

    def test_task_and_language_substitute_their_own_slot(self):
        cfg = _multilingual_config()
        assert cfg.sot_sequence(task="translate") == [60, 50, 56, 59]
        assert cfg.sot_sequence(language="fr") == [60, 51, 55, 59]
        assert cfg.sot_sequence(task="translate", language="de") == [60, 52, 56, 59]

    def test_substitution_preserves_the_prompt_length(self):
        """A batch mixing tasks prefills as one rectangular tensor only because
        substitution never changes the length; the strategy asserts it, and this
        pins the property the assert relies on."""
        cfg = _multilingual_config()
        n = len(cfg.sot_sequence())
        for task in (None, "transcribe", "translate"):
            for lang in (None, "en", "fr", "de"):
                assert len(cfg.sot_sequence(task=task, language=lang)) == n

    def test_unknown_language_names_the_known_ones(self):
        cfg = _multilingual_config()
        with pytest.raises(ValueError, match="unknown language"):
            cfg.sot_sequence(language="xx")
        with pytest.raises(ValueError, match="unknown task"):
            cfg.sot_sequence(task="summarize")

    def test_a_checkpoint_without_the_slot_says_so(self):
        """An English-only Whisper has no language token at all.  Quietly
        decoding under the checkpoint's own language instead would return a
        confident transcript in the wrong one."""
        cfg = _multilingual_config(
            forced_decoder_ids=[(1, 55), (2, 59)],  # task + notimestamps, no language
        )
        assert cfg.sot_sequence(task="translate") == [60, 56, 59]
        with pytest.raises(ValueError, match="no language slot"):
            cfg.sot_sequence(language="fr")

    def test_a_checkpoint_converted_before_the_tables_existed(self):
        """Old native checkpoints carry empty tables; the message has to say
        what to do rather than report an unknown language."""
        cfg = _tiny_config()
        with pytest.raises(ValueError, match="no task token table"):
            cfg.sot_sequence(task="translate")
        with pytest.raises(ValueError, match="no language token table"):
            cfg.sot_sequence(language="fr")

    # -- strategy-level validation + per-row prefill ------------------------

    @staticmethod
    def _strategy(cfg=None):
        from oasr.engine.config import EngineConfig
        from oasr.engine.decode.aed import AedDecodeStrategy
        from oasr.engine.decode.detokenize import Detokenizer

        model = WhisperModel(cfg or _multilingual_config()).eval()
        return AedDecodeStrategy(EngineConfig(ckpt_dir="x"), Detokenizer(None, None), model)

    def test_validate_options_resolves_against_the_checkpoint(self):
        from oasr.engine.request import DecodingOptions

        strat = self._strategy()
        strat.validate_options(DecodingOptions(task="translate", language="fr"))
        strat.validate_options(None)
        strat.validate_options(DecodingOptions())
        with pytest.raises(ValueError, match="unknown language"):
            strat.validate_options(DecodingOptions(language="xx"))

    def test_a_family_without_the_control_rejects_the_option(self):
        """Rule: an option that changes *what is decoded* must never be
        accepted and ignored.  CTC has no task or language token, so it says so
        instead of transcribing as if nothing was asked."""
        from oasr.engine.decode.base import DecodeStrategy
        from oasr.engine.request import DecodingOptions

        class _NoControl:
            decode_type = "ctc"
            selective_options = ()
            # No alignment either, so the whole selective-option table is
            # exercised rather than just the two prompt slots.
            word_timing_modes = ()
            _clock = None
            _SELECTIVE_UNSET = DecodeStrategy._SELECTIVE_UNSET
            validate_options = DecodeStrategy.validate_options
            _require_word_timings = DecodeStrategy._require_word_timings

        for opts in (DecodingOptions(task="translate"), DecodingOptions(language="fr")):
            with pytest.raises(ValueError, match="cannot honour"):
                _NoControl().validate_options(opts)
        with pytest.raises(ValueError, match="cannot produce word timestamps"):
            _NoControl().validate_options(DecodingOptions(word_timestamps=True))
        # Everything else stays accepted: a sampling knob a family ignores
        # returns the same transcript, which is a performance surprise at worst.
        _NoControl().validate_options(DecodingOptions(temperature=0.7, n_best=3))

    def test_prefill_builds_one_prompt_row_per_request(self):
        from oasr.engine.request import DecodingOptions, Request

        strat = self._strategy()
        default = Request(audio=torch.zeros(16000), streaming=False)
        translate = Request(
            audio=torch.zeros(16000),
            streaming=False,
            decoding=DecodingOptions(task="translate", language="fr"),
        )
        assert strat._prompt_for(default) == [60, 50, 55, 59]
        assert strat._prompt_for(translate) == [60, 51, 56, 59]
        # Cached by (task, language), not rebuilt per request.
        assert strat._prompt_for(translate) is strat._prompt_for(translate)

    def test_prefill_mixes_tasks_in_one_batch(self):
        from oasr.engine.request import DecodingOptions, Request

        strat = self._strategy()
        cfg = strat._mcfg
        requests = [
            Request(audio=torch.zeros(16000), streaming=False),
            Request(
                audio=torch.zeros(16000),
                streaming=False,
                decoding=DecodingOptions(task="translate"),
            ),
        ]
        enc = torch.randn(2, cfg.max_source_positions, cfg.d_model)
        prefill = strat._prefill(requests, enc, torch.tensor([enc.size(1)] * 2))
        # Two rows prefilled together despite different prompts — the point of
        # keeping the substitution length-preserving.
        assert prefill.logits.size(0) == 2


# ---------------------------------------------------------------------------
# Converter + registry + native round trip (real snapshot)
# ---------------------------------------------------------------------------


class TestControlTokenTables:
    """Whisper's task / language ids are read from the tokenizer at conversion.

    Hardcoding them would be wrong per release: large-v3 added a language and
    shifted every id after it, so a table pinned to v2's numbering would select
    the *neighbouring* language.
    """

    @staticmethod
    def _tokenizer_json(tmp_path, added):
        import json

        p = tmp_path / "tokenizer.json"
        p.write_text(json.dumps({"added_tokens": added}), encoding="utf-8")
        return p

    def test_tables_are_read_by_name_not_by_id(self, tmp_path):
        from oasr.models.whisper.convert import _control_token_tables

        # Ids deliberately unlike any real release.
        added = [
            {"id": 900, "content": "<|endoftext|>"},
            {"id": 901, "content": "<|startoftranscript|>"},
            {"id": 902, "content": "<|en|>"},
            {"id": 903, "content": "<|fr|>"},
            {"id": 904, "content": "<|yue|>"},
            {"id": 950, "content": "<|translate|>"},
            {"id": 951, "content": "<|transcribe|>"},
            {"id": 952, "content": "<|notimestamps|>"},
            {"id": 953, "content": "<|startoflm|>"},
            {"id": 954, "content": "<|0.00|>"},
        ]
        tasks, languages = _control_token_tables(self._tokenizer_json(tmp_path, added))
        assert tasks == {"translate": 950, "transcribe": 951}
        # Only the language tags — every other control token is longer than
        # three letters or is not letters at all.
        assert languages == {"en": 902, "fr": 903, "yue": 904}

    def test_a_missing_tokenizer_yields_empty_tables(self, tmp_path):
        from oasr.models.whisper.convert import _control_token_tables

        tasks, languages = _control_token_tables(tmp_path / "absent.json")
        assert tasks == {} and languages == {}


# ---------------------------------------------------------------------------


@pytest.mark.requires_assets("WHISPER_CKPT")
class TestConverter:
    def test_detect_and_bundle(self):
        from oasr.models.registry import load_checkpoint_bundle

        arch, bundle = load_checkpoint_bundle(WHISPER_CKPT)
        assert arch == "whisper"
        assert bundle.source_format == "huggingface"
        assert bundle.tokenizer.kind == "whisper"
        f = bundle.features
        assert (f.kind, f.feature_dim, f.audio_scale) == ("whisper_logmel", 80, 1.0)
        assert bundle.decoding.default_decode_type == "aed"
        assert bundle.model_config.sot_sequence()[0] == 50258

    def test_load_report_clean(self):
        from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

        arch, bundle = load_checkpoint_bundle(WHISPER_CKPT)
        model, cfg, report = instantiate_from_bundle(arch, bundle)
        assert not report.missing
        assert not [k for k in report.dropped if not k.startswith("proj_out.")]
        assert sorted(model.capabilities) == ["aed"]

    def test_native_round_trip(self, tmp_path):
        pytest.importorskip("safetensors")
        from oasr.checkpoints.convert import convert_to_native
        from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

        out = tmp_path / "native"
        convert_to_native(WHISPER_CKPT, str(out))
        arch, bundle = load_checkpoint_bundle(out)
        assert (arch, bundle.source_format) == ("whisper", "native")
        assert bundle.tokenizer.kind == "whisper"
        m2, cfg2, _ = instantiate_from_bundle(arch, bundle)
        assert cfg2.sot_sequence() == [50258, 50259, 50359, 50363]

        arch1, b1 = load_checkpoint_bundle(WHISPER_CKPT)
        m1, _, _ = instantiate_from_bundle(arch1, b1)
        sd1, sd2 = m1.state_dict(), m2.state_dict()
        assert set(sd1) == set(sd2)
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k]), k


# ---------------------------------------------------------------------------
# Engine end-to-end (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="engine requires CUDA")
@pytest.mark.requires_assets("WHISPER_CKPT")
class TestEngineWhisperE2E:
    @pytest.fixture(scope="class")
    def engine(self):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(
            ckpt_dir=WHISPER_CKPT,
            service_mode="offline",
            max_batch_size=4,
            decode_steps_per_tick=64,
        )
        eng = ASREngine(cfg)
        yield eng
        del eng
        torch.cuda.empty_cache()

    def test_transcribe_offline(self, engine):
        wavs = assets.require_wavs(2)
        import torchaudio

        audios = [torchaudio.load(w)[0].squeeze(0) for w in wavs]
        texts = engine.transcribe_offline(audios)
        texts = [t.text if hasattr(t, "text") else t for t in texts]
        assert len(texts) == 2
        # LJ001-0001/0002 ground truth openers.
        assert "printing" in texts[0].lower()
        assert "modern" in texts[1].lower()

    def test_streaming_mode_rejected(self):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(ckpt_dir=WHISPER_CKPT, service_mode="streaming")
        with pytest.raises(ValueError, match="offline-only"):
            ASREngine(cfg)

    def test_wrong_decode_method_rejected(self):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(ckpt_dir=WHISPER_CKPT, service_mode="offline", decode_method="ctc")
        with pytest.raises(ValueError, match="not a capability"):
            ASREngine(cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="engine requires CUDA")
@pytest.mark.requires_assets("WHISPER_CKPT")
class TestAedBeamSearch:
    """Beam search over the incremental AR protocol (P4).

    The decoder surface needs no beam-specific method: ``select`` is an
    ``index_select``, which permits repeated indices, so one call both expands
    ``B`` prefilled rows into ``B * k`` and reorders the grid onto each new slot's
    parent.
    """

    def _audios(self, n=2):
        import torchaudio

        wavs = assets.require_wavs(n)
        return [torchaudio.load(w)[0].squeeze(0) for w in wavs]

    def _run(self, audios, *, beam, n_best=1, force_beam_path=False):
        from oasr.engine import ASREngine, DecodingOptions, EngineConfig

        cfg = EngineConfig(
            ckpt_dir=WHISPER_CKPT,
            service_mode="offline",
            max_batch_size=4,
            decode_steps_per_tick=64,
            decode_options={"beam_size": beam},
        )
        eng = ASREngine(cfg)
        if force_beam_path:
            # ``beam_size=1`` takes the greedy branch by construction; to compare
            # the two algorithms we need width 1 *through the beam code*.
            self._force_beam_group_at_width_one(eng)
        try:
            ids = [
                eng.add_request(audio=a, streaming=False, decoding=DecodingOptions(n_best=n_best))
                for a in audios
            ]
            got = {}
            for _ in range(20000):
                for out in eng.step():
                    if out.finished:
                        got[out.request_id] = out
                if not (eng.num_running or eng.num_waiting):
                    break
            assert len(got) == len(ids), "not every request finished"
            return [got[i] for i in ids]
        finally:
            del eng
            torch.cuda.empty_cache()

    @staticmethod
    def _force_beam_group_at_width_one(eng):
        """Route prefill down the beam branch while keeping the width at 1.

        In production ``begin_offline`` picks greedy for ``beam_size == 1``, which
        is right — greedy is cheaper and identical.  To *compare* the two
        implementations the test needs width 1 through the beam code, so it
        replaces the dispatch rather than lying about the configured width.
        """
        strat = eng._output_processor.strategy  # noqa: SLF001

        def begin_offline(requests, enc_out, enc_lengths):
            plan = strat._prefill(requests, enc_out, enc_lengths)  # noqa: SLF001
            opts = [getattr(r, "decoding", None) for r in requests]
            strat._groups.append(  # noqa: SLF001
                strat._make_beam_group(requests, plan, opts)  # noqa: SLF001
            )

        strat.begin_offline = begin_offline

    def test_beam_width_one_reproduces_greedy(self):
        """Beam search at width 1 *is* greedy — same argmax, same sequence.

        The exactness gate: an error in the expand, the parent reorder, the EOS
        bookkeeping or the token fed back into ``step`` shows up here as a
        wrong transcript rather than as a slightly different WER.  (It caught a
        real one: without feeding the chosen token back through ``step``, every
        slot re-picked its first token and the transcript became that token
        repeated to the generation cap.)
        """
        audios = self._audios(2)
        greedy = self._run(audios, beam=1)
        beam1 = self._run(audios, beam=1, force_beam_path=True)
        for i, (g, b) in enumerate(zip(greedy, beam1)):
            assert b.tokens[0] == g.tokens[0], f"utt {i}: {b.text!r} != {g.text!r}"

    @pytest.mark.parametrize("beam", [2, 4])
    def test_wider_beams_keep_the_transcript(self, beam):
        """A broken beam shows up as truncation or repetition, not a subtle WER."""
        outs = self._run(self._audios(2), beam=beam)
        assert "printing" in outs[0].text.lower()
        assert "modern" in outs[1].text.lower()
        assert all(o.finish_reason == "stop" for o in outs)

    def test_nbest_is_populated_ordered_and_capped(self):
        """T5's point: ``n_best`` finally means something for an AR family."""
        outs = self._run(self._audios(2), beam=4, n_best=4)
        for out in outs:
            assert out.scores is not None
            assert out.scores == sorted(out.scores, reverse=True)
            assert 1 < len(out.tokens) <= 4
            assert out.nbest_texts is not None
            assert out.nbest_texts[0] == out.text
            assert len(out.nbest_texts) == len(out.tokens)

    def test_length_penalty_changes_ranking_not_correctness(self):
        """``length_penalty=0`` is the raw log-prob sum — still a valid transcript.

        Raw sums favour short hypotheses, which is why the GNMT normalisation is
        the default; both must still transcribe the audio.
        """
        from oasr.engine import ASREngine, EngineConfig

        for penalty in (0.0, 1.0):
            cfg = EngineConfig(
                ckpt_dir=WHISPER_CKPT,
                service_mode="offline",
                max_batch_size=2,
                decode_steps_per_tick=64,
                decode_options={"beam_size": 4, "length_penalty": penalty},
            )
            eng = ASREngine(cfg)
            try:
                text = eng.transcribe_offline(self._audios(1))[0]
                text = text.text if hasattr(text, "text") else text
                assert "printing" in text.lower(), (penalty, text)
            finally:
                del eng
                torch.cuda.empty_cache()


class TestLongFormMerge:
    """The stitching primitives, tested without a model (pure functions)."""

    def test_split_covers_the_whole_waveform(self):
        from oasr.engine.longform import split_windows

        audio = torch.arange(1000, dtype=torch.float32)
        wins = split_windows(audio, window_samples=300, overlap_samples=0)
        assert torch.equal(torch.cat(wins), audio)
        assert [w.numel() for w in wins] == [300, 300, 300, 100]

    def test_overlapped_split_shares_audio(self):
        from oasr.engine.longform import split_windows

        audio = torch.arange(1000, dtype=torch.float32)
        wins = split_windows(audio, window_samples=300, overlap_samples=100)
        # Stride 200: starts at 0, 200, 400, ...
        assert torch.equal(wins[0][200:], wins[1][:100])
        assert torch.cat([w[:200] for w in wins[:-1]] + [wins[-1]]).numel() >= 1000

    def test_short_audio_is_one_window(self):
        from oasr.engine.longform import split_windows

        audio = torch.zeros(50)
        assert len(split_windows(audio, 300, 100)) == 1

    @pytest.mark.parametrize(
        "pieces,expected",
        [
            (["a b c", "c d e"], "a b c d e"),
            (["a b c", "b c d"], "a b c d"),
            (["a b", "c d"], "a b c d"),  # no overlap at all
            (["hello world", ""], "hello world"),
            (["", "hello"], "hello"),
            (["a b c d", "A B c d e"], "a b c d e"),  # case-insensitive match
        ],
    )
    def test_merge_drops_duplicated_overlap(self, pieces, expected):
        from oasr.engine.longform import merge_texts

        assert merge_texts(pieces) == expected

    def test_merge_prefers_the_longest_overlap(self):
        """A short spurious match must not win over the real one."""
        from oasr.engine.longform import merge_texts

        assert merge_texts(["x a b c", "a b c y"]) == "x a b c y"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="engine requires CUDA")
@pytest.mark.requires_assets("WHISPER_CKPT")
class TestLongFormEngine:
    """Long-form decoding end to end (P4 — the real fix behind C5).

    C5 turned a silent 30 s truncation into a clean rejection.  This turns the
    rejection into a transcript: the request is fanned out into windows, decoded
    through the normal batched path, and fanned back into one output, so the
    caller sees one request id and the serving layer needs no change.
    """

    def _audio(self, n_clips=8):
        import torchaudio

        wavs = assets.require_wavs(n_clips)
        return torch.cat([torchaudio.load(w)[0].squeeze(0) for w in wavs])

    def _engine(self, long_form, overlap=1.0):
        from oasr.engine import ASREngine, EngineConfig

        return ASREngine(
            EngineConfig(
                ckpt_dir=WHISPER_CKPT,
                service_mode="offline",
                max_batch_size=8,
                decode_steps_per_tick=64,
                long_form=long_form,
                long_form_overlap_seconds=overlap,
            )
        )

    def _drain(self, eng, audio, rid="long"):
        eng.add_request(audio=audio, request_id=rid, streaming=False)
        got = {}
        for _ in range(50000):
            for out in eng.step():
                if out.finished:
                    got[out.request_id] = out
            if not (eng.num_running or eng.num_waiting):
                break
        return got

    def test_over_window_audio_is_still_rejected_when_disabled(self):
        """C5's guard must survive: opting out means rejection, not truncation."""
        eng = self._engine(False)
        try:
            with pytest.raises(ValueError, match="fixed to a 30s window"):
                eng.add_request(audio=self._audio(), streaming=False)
        finally:
            del eng
            torch.cuda.empty_cache()

    def test_whole_file_is_transcribed_as_one_output(self):
        audio = self._audio()
        assert audio.numel() / 16000 > 30, "fixture must exceed one window"
        eng = self._engine(True)
        try:
            got = self._drain(eng, audio)
            # One id in, one id out — the windows are an implementation detail.
            assert set(got) == {"long"}
            text = got["long"].text.lower()
            # Content from the first *and* the last window must be present; the
            # bug this fixes is a transcript of the first 30 s only.
            assert "printing" in text
            assert sum(k in text for k in ("modern", "chinese", "exhibition")) >= 2
            assert got["long"].finish_reason == "stop"
        finally:
            del eng
            torch.cuda.empty_cache()

    def test_short_audio_takes_the_normal_path(self):
        """Enabling long-form must not perturb requests that fit a window."""
        import torchaudio

        wav = assets.require_wavs(1)[0]
        audio = torchaudio.load(wav)[0].squeeze(0)
        eng = self._engine(True)
        try:
            got = self._drain(eng, audio, rid="short")
            assert set(got) == {"short"}
            assert "printing" in got["short"].text.lower()
        finally:
            del eng
            torch.cuda.empty_cache()

    def test_zero_overlap_also_works(self):
        eng = self._engine(True, overlap=0.0)
        try:
            got = self._drain(eng, self._audio())
            assert "printing" in got["long"].text.lower()
        finally:
            del eng
            torch.cuda.empty_cache()

    def test_aborting_a_parent_aborts_its_windows(self):
        """The executor knows the windows, not the parent id.

        Without the fan-out-aware abort the cancelled file would keep decoding
        and its outputs would accumulate in the tracker forever.
        """
        eng = self._engine(True)
        try:
            eng.add_request(audio=self._audio(), request_id="doomed", streaming=False)
            assert eng.num_waiting + eng.num_running > 1, "expected several windows"
            eng.abort_request("doomed")
            assert eng.num_waiting + eng.num_running == 0
            assert not eng._longform  # noqa: SLF001  tracker drained
        finally:
            del eng
            torch.cuda.empty_cache()
