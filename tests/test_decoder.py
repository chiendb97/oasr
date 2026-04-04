#!/usr/bin/env python3
"""
Unit tests for decoder wrappers: CTC greedy search and CTC prefix beam search.
"""

from pathlib import Path

import pytest
import torch

from oasr.decoder import (
    ContextGraph,
    CtcGreedySearch,
    CtcGreedySearchOptions,
    CtcPrefixBeamSearch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logp(T: int, V: int, best_path: list[int]) -> torch.Tensor:
    """
    Build a [T, V] log-prob tensor where at each timestep t the token
    best_path[t] has log-prob 0.0 (probability 1) and all others have -1e9.
    """
    assert len(best_path) == T
    logp = torch.full((T, V), -1e9)
    for t, tok in enumerate(best_path):
        logp[t, tok] = 0.0
    return logp


# ---------------------------------------------------------------------------
# CTC Greedy Search tests (synthetic, no checkpoint required)
# ---------------------------------------------------------------------------

class TestCtcGreedySearch:
    """Synthetic tests for CtcGreedySearch that run without any external data."""

    def test_basic_sequence(self):
        """Greedy decode of a deterministic sequence matches expected output."""
        # path: [1, 2, 3] with blank=0 interspersed
        #   frame: 0->1, 1->0(blank), 2->2, 3->0(blank), 4->3
        # expected tokens: [1, 2, 3]
        V = 5
        logp = _make_logp(5, V, [1, 0, 2, 0, 3])
        decoder = CtcGreedySearch(blank=0)
        decoder.search(logp)
        decoder.finalize_search()
        assert decoder.outputs[0] == [1, 2, 3]
        assert decoder.times[0] == [0, 2, 4]

    def test_blank_collapse(self):
        """Consecutive blanks and repeated tokens are collapsed correctly."""
        # path: [1, 1, 0, 1, 0, 2, 2]
        # CTC collapse: [1, 1] -> [1]; blank-separated [1] -> emitted again -> [1, 1]; [2,2]->[2]
        # But blank is between the two 1s, so: 1 (t=0), 1(t=1 duplicate skip),
        # blank(t=2), 1(t=3 new because prev was blank), blank(t=4), 2(t=5), 2(t=6 dup skip)
        # => output [1, 1, 2]
        V = 4
        logp = _make_logp(7, V, [1, 1, 0, 1, 0, 2, 2])
        decoder = CtcGreedySearch(blank=0)
        decoder.search(logp)
        assert decoder.outputs[0] == [1, 1, 2]
        assert decoder.times[0] == [0, 3, 5]

    def test_all_blank(self):
        """All-blank sequence yields empty output."""
        V = 4
        logp = _make_logp(5, V, [0, 0, 0, 0, 0])
        decoder = CtcGreedySearch(blank=0)
        decoder.search(logp)
        assert decoder.outputs[0] == []
        assert decoder.times[0] == []

    def test_streaming_matches_offline(self):
        """Chunked (streaming) decoding matches single-pass offline decoding."""
        V = 5
        full_path = [1, 0, 2, 2, 0, 3, 0, 1]
        logp_full = _make_logp(len(full_path), V, full_path)

        # Offline
        offline = CtcGreedySearch(blank=0)
        offline.search(logp_full)
        offline.finalize_search()

        # Streaming: two chunks
        stream = CtcGreedySearch(blank=0)
        stream.search(logp_full[:4])
        stream.search(logp_full[4:])
        stream.finalize_search()

        assert stream.outputs[0] == offline.outputs[0]
        assert stream.times[0] == offline.times[0]

    def test_streaming_cross_chunk_duplicate_collapse(self):
        """Repeated token spanning a chunk boundary is collapsed correctly."""
        V = 3
        # Chunk 1 ends with token 1; chunk 2 starts with token 1 (consecutive dup)
        # Then blank, then 2.
        # Expected: [1, 2]  (the second 1 at t=2 is a dup of prev_token_=1)
        chunk1 = _make_logp(2, V, [1, 1])  # t=0->1, t=1->1
        chunk2 = _make_logp(3, V, [1, 0, 2])  # t=2->1(dup), t=3->blank, t=4->2

        stream = CtcGreedySearch(blank=0)
        stream.search(chunk1)
        stream.search(chunk2)
        stream.finalize_search()
        assert stream.outputs[0] == [1, 2]
        assert stream.times[0] == [0, 4]

    def test_reset_clears_state(self):
        """Reset clears all state; re-decoding the same input gives the same result."""
        V = 4
        logp = _make_logp(4, V, [1, 0, 2, 0])
        decoder = CtcGreedySearch(blank=0)
        decoder.search(logp)
        first = list(decoder.outputs[0])

        decoder.reset()
        assert decoder.outputs[0] == []
        assert decoder.likelihood[0] == 0.0

        decoder.search(logp)
        assert decoder.outputs[0] == first

    def test_options_constructor(self):
        """CtcGreedySearchOptions constructor works and sets blank correctly."""
        opts = CtcGreedySearchOptions()
        opts.blank = 2
        decoder = CtcGreedySearch(opts)
        # Token id=2 is now blank; ids 0, 1, 3 are non-blank
        V = 4
        logp = _make_logp(3, V, [0, 2, 1])  # blank=2, so only t=0->0 and t=2->1 are kept
        decoder.search(logp)
        assert decoder.outputs[0] == [0, 1]

    def test_likelihood_accumulates(self):
        """Likelihood is the sum of argmax log-probs across all frames."""
        V = 3
        # Set specific log-probs: t=0 best=-0.5, t=1 best=-0.3
        logp = torch.full((2, V), -1e9)
        logp[0, 1] = -0.5
        logp[1, 0] = -0.3  # blank -> not emitted as token, but still contributes to likelihood
        decoder = CtcGreedySearch(blank=0)
        decoder.search(logp)
        assert abs(decoder.likelihood[0] - (-0.5 + -0.3)) < 1e-5

    def test_outputs_interface_is_size_one_nbest(self):
        """The N-best interface returns size-1 lists (greedy has one hypothesis)."""
        V = 4
        logp = _make_logp(3, V, [1, 0, 2])
        decoder = CtcGreedySearch(blank=0)
        decoder.search(logp)
        assert len(decoder.outputs) == 1
        assert len(decoder.likelihood) == 1
        assert len(decoder.times) == 1


# ---------------------------------------------------------------------------
# Context Graph tests
# ---------------------------------------------------------------------------

class TestContextGraph:
    """Tests for the ContextGraph phrase-boosting trie."""

    def test_exact_phrase_match(self):
        """Traversing a complete phrase reaches an end state with positive score."""
        # Phrase: [1, 2, 3]  context_score=1.0 => bonus of 3.0 total
        ctx = ContextGraph(phrases=[[1, 2, 3]], context_score=1.0)
        state = ctx.start_state
        total_delta = 0.0
        for tok in [1, 2, 3]:
            state, delta = ctx.get_next_state(state, tok)
            total_delta += delta
        assert abs(total_delta - 3.0) < 1e-5, f"Expected 3.0, got {total_delta}"

    def test_no_match_returns_to_root(self):
        """An unrelated token stays at / returns to start state with zero score."""
        ctx = ContextGraph(phrases=[[1, 2]], context_score=2.0)
        state = ctx.start_state
        state2, delta = ctx.get_next_state(state, 99)
        assert state2 == ctx.start_state
        assert abs(delta) < 1e-5

    def test_partial_match_backoff(self):
        """Partial phrase match is backed off at finalization."""
        # Phrase [1, 2, 3], context_score=1.0; advance 2 steps (bonus=2.0)
        ctx = ContextGraph(phrases=[[1, 2, 3]], context_score=1.0)
        state = ctx.start_state
        for tok in [1, 2]:
            state, _ = ctx.get_next_state(state, tok)
        # Backoff should equal accumulated partial score
        backoff = ctx.get_backoff_score(state)
        assert abs(backoff - 2.0) < 1e-5, f"Expected 2.0, got {backoff}"

    def test_start_state_backoff_is_zero(self):
        """No partial match at start state => zero backoff."""
        ctx = ContextGraph(phrases=[[1, 2]], context_score=2.0)
        assert ctx.get_backoff_score(ctx.start_state) == 0.0

    def test_multiple_phrases(self):
        """Both phrases can be traversed independently."""
        ctx = ContextGraph(phrases=[[1, 2], [3, 4, 5]], context_score=1.0)

        s = ctx.start_state
        score = 0.0
        for tok in [1, 2]:
            s, d = ctx.get_next_state(s, tok)
            score += d
        assert abs(score - 2.0) < 1e-5

        s = ctx.start_state
        score = 0.0
        for tok in [3, 4, 5]:
            s, d = ctx.get_next_state(s, tok)
            score += d
        assert abs(score - 3.0) < 1e-5


class TestBeamSearchWithContext:
    """Beam search + phrase boosting integration tests."""

    def test_context_tips_beam_ranking(self):
        """Phrase boosting promotes a slightly worse acoustic path to first place."""
        V = 5
        blank = 0
        # Two competing paths: [1] and [2].
        # t=0: token 1 log-prob=-0.1 (better), token 2 log-prob=-0.5 (worse)
        # t=1: blank dominates (both paths end here)
        # Without context: [1] wins acoustically.
        # With context [2] and bonus=5.0: [2] total score = -0.5 + 5.0 = 4.5 > [1]'s -0.1
        T = 2
        logp = torch.full((T, V), -100.0)
        logp[0, 1] = -0.1   # path [1]: acoustic score ≈ -0.1
        logp[0, 2] = -0.5   # path [2]: acoustic score ≈ -0.5
        logp[1, blank] = -0.01  # blank at t=1 (paths don't change)

        # Without context: top path should be [1]
        decoder_no_ctx = CtcPrefixBeamSearch(blank=blank, first_beam_size=5, second_beam_size=5)
        decoder_no_ctx.search(logp)
        decoder_no_ctx.finalize_search()
        top_no_ctx = decoder_no_ctx.outputs[0]

        # With context [2]: +5.0 bonus lifts [2] above [1]
        ctx = ContextGraph(phrases=[[2]], context_score=5.0)
        decoder_ctx = CtcPrefixBeamSearch(blank=blank, first_beam_size=5, second_beam_size=5)
        decoder_ctx.set_context_graph(ctx)
        decoder_ctx.search(logp)
        decoder_ctx.finalize_search()
        top_ctx = decoder_ctx.outputs[0]

        assert top_no_ctx == [1], f"Expected [1] without context, got {top_no_ctx}"
        assert top_ctx == [2], f"Expected [2] with context, got {top_ctx}"

    def test_no_context_regression(self):
        """Without a context graph, beam search behavior is unchanged."""
        V = 4
        logp = _make_logp(4, V, [1, 0, 2, 0])
        decoder = CtcPrefixBeamSearch(blank=0, first_beam_size=4, second_beam_size=4)
        decoder.search(logp)
        decoder.finalize_search()
        assert decoder.outputs[0] == [1, 2]

    def test_context_streaming_state_preserved(self):
        """Context state persists correctly across streaming chunks."""
        V = 5
        blank = 0
        ctx = ContextGraph(phrases=[[1, 2, 3]], context_score=2.0)

        # Offline: single call
        logp = _make_logp(5, V, [1, 0, 2, 0, 3])
        d_offline = CtcPrefixBeamSearch(blank=blank, first_beam_size=5, second_beam_size=5)
        d_offline.set_context_graph(ctx)
        d_offline.search(logp)
        d_offline.finalize_search()

        # Streaming: two chunks
        d_stream = CtcPrefixBeamSearch(blank=blank, first_beam_size=5, second_beam_size=5)
        d_stream.set_context_graph(ctx)
        d_stream.search(logp[:3])
        d_stream.search(logp[3:])
        d_stream.finalize_search()

        assert d_stream.outputs[0] == d_offline.outputs[0]
        assert abs(d_stream.likelihood[0] - d_offline.likelihood[0]) < 1e-4


class TestGreedySearchWithContext:
    """Greedy search + phrase boosting integration tests."""

    def test_context_boosts_likelihood(self):
        """Context bonus increases reported likelihood for a matching path."""
        V = 5
        blank = 0
        ctx = ContextGraph(phrases=[[1, 2]], context_score=3.0)

        logp = _make_logp(4, V, [1, 0, 2, 0])

        d_no_ctx = CtcGreedySearch(blank=blank)
        d_no_ctx.search(logp)
        no_ctx_likelihood = d_no_ctx.likelihood[0]

        d_ctx = CtcGreedySearch(blank=blank)
        d_ctx.set_context_graph(ctx)
        d_ctx.search(logp)
        ctx_likelihood = d_ctx.likelihood[0]

        # Context version should have higher total likelihood (+6.0 for 2 tokens)
        assert ctx_likelihood > no_ctx_likelihood
        assert abs((ctx_likelihood - no_ctx_likelihood) - 6.0) < 1e-5

    def test_context_streaming_state_preserved(self):
        """Context state is preserved across greedy streaming chunks."""
        V = 4
        blank = 0
        ctx = ContextGraph(phrases=[[1, 2]], context_score=1.0)
        logp = _make_logp(4, V, [1, 0, 2, 0])

        d_offline = CtcGreedySearch(blank=blank)
        d_offline.set_context_graph(ctx)
        d_offline.search(logp)

        d_stream = CtcGreedySearch(blank=blank)
        d_stream.set_context_graph(ctx)
        d_stream.search(logp[:2])
        d_stream.search(logp[2:])

        assert d_stream.outputs[0] == d_offline.outputs[0]
        assert abs(d_stream.likelihood[0] - d_offline.likelihood[0]) < 1e-5


# ---------------------------------------------------------------------------
# K2 WFST Beam Search tests (skipped when K2 is not available)
# ---------------------------------------------------------------------------

import oasr.decoder as _decoder_mod

_K2_AVAILABLE = getattr(_decoder_mod, "k2_available", False)


@pytest.mark.skipif(not _K2_AVAILABLE, reason="K2 not available (build with OASR_USE_K2=1)")
class TestCtcWfstBeamSearch:
    """Tests for CtcWfstBeamSearch (K2 backend). Only run when K2 is compiled in."""

    @staticmethod
    def _save_fsa(fsa, tmp_path, name: str) -> str:
        """Save a k2 Fsa as a dict to a temp file and return the path."""
        import torch

        path = str(tmp_path / name)
        torch.save(fsa.as_dict(), path, _use_new_zipfile_serialization=True)
        return path

    def test_wfst_basic(self, tmp_path):
        """Decode a single token through a CTC topo decoding graph."""
        import k2

        # k2.ctc_topo creates a proper CTC topology graph with aux_labels,
        # which is required by k2.GetLattice + k2.BestPath.
        # max_token=1: blank=0, token=1
        graph = k2.ctc_topo(max_token=1)
        graph_path = self._save_fsa(graph, tmp_path, "graph_basic.pt")

        from oasr.decoder import CtcWfstBeamSearch, CtcWfstBeamSearchOptions

        opts = CtcWfstBeamSearchOptions()
        opts.blank_skip_thresh = 0.0  # disable blank-skipping for deterministic test
        decoder = CtcWfstBeamSearch.from_file(graph_path, opts)

        # Two frames: token 1 then blank (0). CTC should decode to [1].
        V = 2  # vocab: blank=0, token=1
        logp = _make_logp(2, V, [1, 0])
        decoder.search(logp)
        decoder.finalize_search()

        assert len(decoder.outputs) >= 1
        assert decoder.outputs[0] == [1]

    def test_wfst_streaming(self, tmp_path):
        """Chunked WFST decoding matches offline decoding."""
        import k2

        # max_token=2: blank=0, tokens 1 and 2
        graph = k2.ctc_topo(max_token=2)
        graph_path = self._save_fsa(graph, tmp_path, "graph_stream.pt")

        from oasr.decoder import CtcWfstBeamSearch, CtcWfstBeamSearchOptions

        opts = CtcWfstBeamSearchOptions()
        opts.blank_skip_thresh = 0.0  # disable blank skipping for this test
        V = 3  # vocab: blank=0, token 1, token 2
        full_path = [1, 0, 2, 0]  # token1, blank, token2, blank → expected [1, 2]
        logp = _make_logp(len(full_path), V, full_path)

        # Offline
        d_offline = CtcWfstBeamSearch.from_file(graph_path, opts)
        d_offline.search(logp)
        d_offline.finalize_search()

        # Streaming (same frames split in two chunks)
        d_stream = CtcWfstBeamSearch.from_file(graph_path, opts)
        d_stream.search(logp[:2])
        d_stream.search(logp[2:])
        d_stream.finalize_search()

        assert d_stream.outputs[0] == d_offline.outputs[0]


class TestCtcPrefixBeamSearch:
    """Tests for the CtcPrefixBeamSearch Python wrapper."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_decode(self, ckpt_dir: str, audio_path: str, dtype: torch.dtype):
        """Decoder should match WeNet for the same checkpoint."""
        import torchaudio
        from oasr.models.conformer import load_wenet_checkpoint

        if not ckpt_dir or not Path(ckpt_dir).exists():
            pytest.skip(
                "WeNet checkpoint dir not set or not found; set WENET_CKPT_DIR env var or --ckpt-dir"
            )
        if not audio_path or not Path(audio_path).exists():
            pytest.skip(
                "Audio path not set or not found; set WENET_AUDIO_PATH env var or --audio-path"
            )

        def read_audio(path: str):
            """Read audio file."""
            audio, sr = torchaudio.load(path)
            audio = audio * (1 << 15)
            return audio, sr

        def extract_features(audio: torch.Tensor, sr: int):
            # extract feature using torchaudio.compliance.kaldi as kaldi
            feats = torchaudio.compliance.kaldi.fbank(
                audio,
                sr,
                num_mel_bins=80,
                frame_shift=10,
                frame_length=25,
                dither=1.0,
            )
            return feats

        def build_dictionary(ckpt_dir: str):
            """Build dictionary from words.txt."""
            id2word = {}
            with open(Path(ckpt_dir) / "words.txt", "r") as f:
                for line in f.readlines():
                    word, idx = line.strip().split()
                    id2word[int(idx)] = word
            return id2word

        def decode(probs: torch.Tensor, id2word: dict[int, str]):
            """Detokenize outputs."""
            decoder = CtcPrefixBeamSearch(
                blank=0, first_beam_size=10, second_beam_size=10)

            probs = probs.squeeze(0).to(device="cpu", dtype=torch.float32)

            decoder.reset()
            decoder.search(probs)
            decoder.finalize_search()

            outputs = decoder.outputs
            likelihood = decoder.likelihood

            text = "".join([id2word[idx]
                           for idx in outputs[0]]).replace("▁", " ").strip()
            return text, likelihood

        # Run the encoder on CUDA using the requested half/bfloat16 dtype.
        device = "cuda"
        oasr_model, _ = load_wenet_checkpoint(
            str(ckpt_dir), device=device, dtype=dtype)

        decoder = CtcPrefixBeamSearch(
            blank=0, first_beam_size=10, second_beam_size=10)

        torch.manual_seed(42)

        audio, sr = read_audio(audio_path)
        feats = extract_features(audio, sr)  # [T, F]
        feats = feats.unsqueeze(0)  # [1, T, F]
        feats = feats.to(device=device, dtype=dtype)

        lengths = torch.full(
            (1,), feats.size(1), dtype=torch.long, device=device
        )

        with torch.no_grad():
            probs = oasr_model(feats, lengths)
            chunk_by_chunk_probs = oasr_model.forward_chunk_by_chunk(
                feats, decoding_chunk_size=4, num_decoding_left_chunks=2)

        # CtcPrefixBeamSearch expects log-probs of shape [T, V] on CPU in
        # float32; collapse the batch dimension and move to CPU.
        probs = probs.squeeze(0).to(device="cpu", dtype=dtype)
        chunk_by_chunk_probs = chunk_by_chunk_probs.squeeze(
            0).to(device="cpu", dtype=dtype)

        id2word = build_dictionary(ckpt_dir)

        text, _ = decode(probs, id2word)
        chunk_by_chunk_text, _ = decode(chunk_by_chunk_probs, id2word)
        print(f"text: {text}")
        print(f"chunk_by_chunk_text: {chunk_by_chunk_text}")

        assert len(text.split()) > 0
        assert len(chunk_by_chunk_text.split()) > 0


# ---------------------------------------------------------------------------
# High-level Python Decoder API tests
# ---------------------------------------------------------------------------

class TestDecoderAPI:
    """Integration tests for the high-level oasr.decode.Decoder class."""

    def test_greedy_offline(self):
        """Decoder.decode() with greedy search returns correct tokens."""
        from oasr.decode import Decoder, DecoderConfig

        V = 5
        logp = _make_logp(4, V, [1, 0, 2, 0])
        config = DecoderConfig(search_type="greedy", blank=0)
        decoder = Decoder(config)
        result = decoder.decode(logp)

        assert len(result.tokens) == 1
        assert result.tokens[0] == [1, 2]
        assert len(result.scores) == 1
        assert len(result.times) == 1
        assert result.times[0] == [0, 2]

    def test_beam_offline(self):
        """Decoder.decode() with prefix_beam returns a non-empty N-best list."""
        from oasr.decode import Decoder, DecoderConfig

        V = 5
        logp = _make_logp(5, V, [1, 0, 2, 0, 3])
        config = DecoderConfig(search_type="prefix_beam", blank=0, beam_size=5)
        decoder = Decoder(config)
        result = decoder.decode(logp)

        assert len(result.tokens) >= 1
        assert result.tokens[0] == [1, 2, 3]

    def test_streaming_greedy(self):
        """Streaming greedy via init_stream/decode_chunk/finalize_stream matches offline."""
        from oasr.decode import Decoder, DecoderConfig

        V = 5
        full_path = [1, 0, 2, 0, 3, 0]
        logp = _make_logp(len(full_path), V, full_path)

        config = DecoderConfig(search_type="greedy", blank=0)
        d_offline = Decoder(config)
        offline_result = d_offline.decode(logp)

        d_stream = Decoder(config)
        d_stream.init_stream()
        d_stream.decode_chunk(logp[:3])
        d_stream.decode_chunk(logp[3:])
        final = d_stream.finalize_stream()

        assert final.tokens[0] == offline_result.tokens[0]

    def test_streaming_beam(self):
        """Streaming beam search via init/chunk/finalize matches single-pass offline."""
        from oasr.decode import Decoder, DecoderConfig

        V = 5
        full_path = [1, 0, 2, 0, 3]
        logp = _make_logp(len(full_path), V, full_path)

        config = DecoderConfig(search_type="prefix_beam", blank=0, beam_size=5)
        d_offline = Decoder(config)
        offline_result = d_offline.decode(logp)

        d_stream = Decoder(config)
        d_stream.init_stream()
        for chunk in [logp[:2], logp[2:]]:
            d_stream.decode_chunk(chunk)
        final = d_stream.finalize_stream()

        assert final.tokens[0] == offline_result.tokens[0]

    def test_context_boosting_via_api(self):
        """Phrase boosting via DecoderConfig shifts the top hypothesis."""
        from oasr.decode import Decoder, DecoderConfig

        V = 5
        blank = 0
        T = 2
        logp = torch.full((T, V), -100.0)
        logp[0, 1] = -0.1   # path [1]: better acoustics
        logp[0, 2] = -0.5   # path [2]: worse acoustics, but boosted
        logp[1, blank] = -0.01

        # Without context
        d1 = Decoder(DecoderConfig(search_type="prefix_beam", blank=blank, beam_size=5))
        r1 = d1.decode(logp)
        assert r1.tokens[0] == [1]

        # With context boosting [2]
        d2 = Decoder(
            DecoderConfig(
                search_type="prefix_beam",
                blank=blank,
                beam_size=5,
                context_phrases=[[2]],
                context_score=5.0,
            )
        )
        r2 = d2.decode(logp)
        assert r2.tokens[0] == [2]

    def test_search_type_switch(self):
        """Same log-probs decoded with different search_types all return valid results."""
        from oasr.decode import Decoder, DecoderConfig

        V = 5
        logp = _make_logp(4, V, [1, 0, 2, 0])

        for stype in ("greedy", "prefix_beam"):
            config = DecoderConfig(search_type=stype, blank=0)
            decoder = Decoder(config)
            result = decoder.decode(logp)
            assert len(result.tokens) >= 1
            assert len(result.tokens[0]) > 0, f"Empty decode for search_type={stype!r}"

    def test_unknown_search_type_raises(self):
        """DecoderConfig with unknown search_type raises ValueError."""
        from oasr.decode import Decoder, DecoderConfig
        import pytest as _pytest

        config = DecoderConfig(search_type="unknown")
        with _pytest.raises(ValueError, match="Unknown search_type"):
            Decoder(config)

    def test_wfst_no_k2_raises(self):
        """Requesting WFST without K2 raises RuntimeError with a helpful message."""
        from oasr.decode import Decoder, DecoderConfig
        import oasr.decoder as _d
        import pytest as _pytest

        if _d.k2_available:
            pytest.skip("K2 is available; skipping 'no-K2' error test.")

        config = DecoderConfig(search_type="wfst")
        with _pytest.raises(RuntimeError, match="K2 WFST decoder is not available"):
            Decoder(config)

    def test_decoder_config_property(self):
        """Decoder.config returns the original DecoderConfig."""
        from oasr.decode import Decoder, DecoderConfig

        config = DecoderConfig(search_type="greedy", beam_size=7)
        decoder = Decoder(config)
        assert decoder.config is config

    def test_result_dataclass_fields(self):
        """DecoderResult has tokens, scores, and times fields."""
        from oasr.decode import DecoderResult

        r = DecoderResult(tokens=[[1, 2]], scores=[-0.5], times=[[0, 2]])
        assert r.tokens == [[1, 2]]
        assert r.scores == [-0.5]
        assert r.times == [[0, 2]]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
