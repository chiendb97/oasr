# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the tokenizer axis (registry, specs, implementations) — no GPU."""

from __future__ import annotations

import pytest

from oasr.tokenizers import (
    DEFAULT_SPECIAL_IDS,
    SymbolTableTokenizer,
    Tokenizer,
    TokenizerSpec,
    build_tokenizer,
    list_tokenizers,
    register_tokenizer,
)

UNITS = "<blank> 0\n<unk> 1\n▁he 2\n▁hel 3\nlo 4\n▁wor 5\nld 6\n<sos/eos> 29\n"


@pytest.fixture
def units_file(tmp_path):
    p = tmp_path / "units.txt"
    p.write_text(UNITS, encoding="utf-8")
    return p


class TestRegistry:
    def test_builtins_registered(self):
        kinds = list_tokenizers()
        assert {"symbol_table", "sentencepiece", "huggingface"} <= set(kinds)

    def test_unknown_kind_raises(self):
        with pytest.raises(KeyError, match="Unknown tokenizer kind"):
            build_tokenizer(TokenizerSpec(kind="does-not-exist"))

    def test_custom_registration(self, units_file):
        class Fixed(SymbolTableTokenizer):
            pass

        register_tokenizer("fixed-test", lambda spec: Fixed(spec.files["table"]))
        try:
            tok = build_tokenizer(
                TokenizerSpec(kind="fixed-test", files={"table": str(units_file)})
            )
            assert isinstance(tok, Fixed)
        finally:
            from oasr.tokenizers import registry

            del registry._REGISTRY["fixed-test"]


class TestTokenizerSpec:
    def test_dict_round_trip(self):
        spec = TokenizerSpec(
            kind="symbol_table",
            files={"table": "/x/units.txt"},
            options={"special_ids": [0, 1]},
        )
        assert TokenizerSpec.from_dict(spec.to_dict()) == spec

    def test_special_ids_default_and_override(self):
        assert TokenizerSpec(kind="symbol_table").special_ids() == DEFAULT_SPECIAL_IDS
        spec = TokenizerSpec(kind="symbol_table", options={"special_ids": [0, 7]})
        assert spec.special_ids() == frozenset({0, 7})


class TestSymbolTable:
    def test_decode_matches_legacy_detokenizer(self, units_file):
        """Bit-compatibility with the historical Detokenizer behavior."""
        from oasr.engine.decode.detokenize import Detokenizer

        tok = build_tokenizer(TokenizerSpec(kind="symbol_table", files={"table": str(units_file)}))
        detok = Detokenizer(None, str(units_file))
        for ids in ([0, 2, 4, 1, 5, 6], [3, 4], [0, 1], [], [2, 999, 4]):
            assert tok.decode(ids) == detok.detokenize(ids)
        assert tok.decode([0, 2, 4, 1, 5, 6]) == "lo world"  # 2 stripped as special

    def test_special_ids_override(self, units_file):
        spec = TokenizerSpec(
            kind="symbol_table",
            files={"table": str(units_file)},
            options={"special_ids": [0, 1]},
        )
        tok = build_tokenizer(spec)
        # id 2 is a real token under the overridden special set.
        assert tok.decode([0, 2, 4]) == "helo"

    def test_vocab_size_and_encode(self, units_file):
        tok = SymbolTableTokenizer(str(units_file))
        assert tok.vocab_size == 30  # max id 29 + 1
        with pytest.raises(NotImplementedError):
            tok.encode("hello")

    def test_detokenizer_fallback_without_table(self):
        from oasr.engine.decode.detokenize import Detokenizer

        detok = Detokenizer(None, None)
        assert detok.detokenize([0, 5, 1, 7]) == "5 7"
        assert detok.tokenizer is None


class TestSentencePiece:
    @pytest.fixture(scope="class")
    def sp_model(self, tmp_path_factory):
        spm = pytest.importorskip("sentencepiece")
        d = tmp_path_factory.mktemp("spm")
        corpus = d / "corpus.txt"
        corpus.write_text("hello world\nhello there\nworld peace\n" * 20)
        spm.SentencePieceTrainer.Train(
            input=str(corpus),
            model_prefix=str(d / "bpe"),
            vocab_size=16,  # tiny corpus caps the trainable vocab
            model_type="unigram",
        )
        return d / "bpe.model"

    def test_encode_decode_round_trip(self, sp_model):
        tok = build_tokenizer(
            TokenizerSpec(
                kind="sentencepiece",
                files={"model": str(sp_model)},
                options={"special_ids": [0, 1, 2]},
            )
        )
        ids = tok.encode("hello world")
        assert ids and all(isinstance(i, int) for i in ids)
        assert tok.decode(ids) == "hello world"
        assert tok.vocab_size == 16
        assert tok.decode([]) == ""

    def test_decode_drops_ids_past_the_piece_count(self, sp_model):
        """A CTC head padded wider than the tokenizer must not crash decode.

        icefall's zipformer-large ships ``ctc_lo`` with 504 rows against a
        500-piece ``bpe.model``, so the beam search can legitimately emit a
        class that is not a piece. SentencePiece answers those with
        ``Out of range: piece id is out of range``; letting that propagate
        turns one junk frame into a failed request.
        """
        tok = build_tokenizer(
            TokenizerSpec(
                kind="sentencepiece",
                files={"model": str(sp_model)},
                options={"special_ids": [0, 1, 2]},
            )
        )
        ids = tok.encode("hello world")
        pad_id = tok.vocab_size + 3  # GEMM-alignment padding class
        assert tok.decode(ids + [pad_id]) == tok.decode(ids)
        assert tok.decode([pad_id]) == ""
        assert tok.decode([-1]) == ""


class TestHuggingFace:
    @pytest.fixture(scope="class")
    def hf_tokenizer_json(self, tmp_path_factory):
        tokenizers = pytest.importorskip("tokenizers")
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
        tok = tokenizers.Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        path = tmp_path_factory.mktemp("hf") / "tokenizer.json"
        tok.save(str(path))
        return path

    def test_encode_decode(self, hf_tokenizer_json):
        tok = build_tokenizer(
            TokenizerSpec(kind="huggingface", files={"tokenizer": str(hf_tokenizer_json)})
        )
        assert tok.encode("hello world") == [2, 3]
        assert tok.decode([2, 3]) == "hello world"
        assert tok.vocab_size == 4


# ---------------------------------------------------------------------------
# Registry-wide contract
# ---------------------------------------------------------------------------


class TestTokenizerContract:
    """Properties every registered kind must hold, checked over the registry.

    Pure metadata + a symbol-table instance: no optional deps, no checkpoints, so
    this runs everywhere.  Instance-level behaviour for the dependency-bearing
    kinds is covered by the per-kind tests above.
    """

    def test_registry_has_the_documented_kinds(self):
        from oasr.tokenizers import list_tokenizers

        assert set(list_tokenizers()) == {
            "symbol_table",
            "sentencepiece",
            "huggingface",
            "whisper",
            "funasr_char",
        }

    def test_supports_encode_is_declared_per_kind(self):
        """``hasattr(tok, "encode")`` cannot answer this — ``encode`` is abstract
        on the ABC, so it is always present even where it raises."""
        from oasr.tokenizers import SymbolTableTokenizer
        from oasr.tokenizers.base import Tokenizer

        assert Tokenizer.supports_encode is True  # opt-out, not opt-in
        assert SymbolTableTokenizer.supports_encode is False
        assert hasattr(SymbolTableTokenizer, "encode"), "the misleading attribute is still there"

    def test_decode_only_kind_raises_from_encode(self, tmp_path):
        table = tmp_path / "units.txt"
        table.write_text("<blank> 0\n<unk> 1\n<sos/eos> 2\nhi 3\n")
        tok = SymbolTableTokenizer(str(table))
        assert not tok.supports_encode
        with pytest.raises(NotImplementedError):
            tok.encode("hi")

    def test_special_ids_are_what_decode_strips(self, tmp_path):
        """The contract: filtering by ``special_ids`` yields what decode keeps."""
        table = tmp_path / "units.txt"
        table.write_text("<blank> 0\n<unk> 1\n<sos/eos> 2\n▁he 3\nllo 4\n")
        tok = SymbolTableTokenizer(str(table))
        ids = [0, 3, 4, 2, 1]
        assert tok.decode(ids) == tok.decode([i for i in ids if i not in tok.special_ids])

    def test_vocab_size_is_stable_and_cheap(self, tmp_path):
        table = tmp_path / "units.txt"
        table.write_text("\n".join(f"p{i} {i}" for i in range(64)))
        tok = SymbolTableTokenizer(str(table))
        assert tok.vocab_size == 64 == tok.vocab_size  # repeat reads agree


class TestIncrementalDecode:
    """T3: a partial must not re-decode the whole prefix every time.

    The contract is that incremental decoding is *indistinguishable* from full
    decoding — concatenating every delta equals ``decode(all_ids)``, and
    ``state["text"]`` holds the full transcript at every step.  Any override
    that trades correctness for speed fails here.
    """

    def _drive(self, tok, ids, chunk):
        """Feed ``ids`` through ``decode_incremental`` in chunks of ``chunk``."""
        state = tok.new_decode_state()
        deltas = []
        for i in range(0, len(ids), chunk):
            deltas.append(tok.decode_incremental(ids[i : i + chunk], state))
        return deltas, state

    @pytest.mark.parametrize("chunk", [1, 2, 3, 100])
    def test_symbol_table_matches_full_decode(self, units_file, chunk):
        tok = build_tokenizer(TokenizerSpec(kind="symbol_table", files={"table": str(units_file)}))
        ids = [2, 4, 5, 6, 3, 4]
        deltas, state = self._drive(tok, ids, chunk)
        assert state["text"] == tok.decode(ids)
        assert "".join(deltas) == tok.decode(ids)

    def test_symbol_table_skips_special_ids_incrementally(self, units_file):
        tok = build_tokenizer(TokenizerSpec(kind="symbol_table", files={"table": str(units_file)}))
        ids = [0, 2, 4, 1, 5, 6, 0]
        _, state = self._drive(tok, ids, 2)
        assert state["text"] == tok.decode(ids)

    def test_symbol_table_is_actually_incremental(self, units_file):
        """Not just correct — it must stop re-decoding the prefix.

        ``symbol_table`` rendering is piece-local, so the override exists; the
        guard is that it never calls the O(prefix) ``decode``.
        """
        tok = build_tokenizer(TokenizerSpec(kind="symbol_table", files={"table": str(units_file)}))
        calls = []
        original = tok.decode
        tok.decode = lambda ids: (calls.append(len(list(ids))), original(ids))[1]
        self._drive(tok, [2, 4, 5, 6], 1)
        assert not calls, "the fast path fell back to a full decode"

    @pytest.mark.parametrize("chunk", [1, 3])
    def test_huggingface_matches_full_decode(self, chunk):
        tokenizers = pytest.importorskip("tokenizers")
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        vocab = {"[UNK]": 0, "hello": 1, "world": 2, "again": 3, "and": 4}
        t = tokenizers.Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        t.pre_tokenizer = Whitespace()
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp())
        t.save(str(d / "tokenizer.json"))
        tok = build_tokenizer(
            TokenizerSpec(kind="huggingface", files={"tokenizer": str(d / "tokenizer.json")})
        )
        ids = [1, 2, 4, 3, 1, 2, 4, 3, 1, 2] * 6  # long enough to force re-anchoring
        deltas, state = self._drive(tok, ids, chunk)
        assert "".join(deltas) == tok.decode(ids)
        assert state["text"] == tok.decode(ids)

    def test_default_fallback_is_correct_for_every_kind(self, units_file):
        """A kind with no override still returns deltas that sum to the whole."""

        class NoOverride(SymbolTableTokenizer):
            # Force the base-class fallback path.
            decode_incremental = Tokenizer.decode_incremental
            new_decode_state = Tokenizer.new_decode_state

        tok = NoOverride(str(units_file))
        ids = [2, 4, 5, 6]
        deltas, state = self._drive(tok, ids, 2)
        assert "".join(deltas) == tok.decode(ids)
        assert state["text"] == tok.decode(ids)
