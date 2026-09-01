# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Incremental detokenization must be O(new tokens), not O(transcript).

``decode_incremental`` runs once per active stream per partial, and at the
default ``partial_decode_interval=1`` that is every tick.  Both shipped
implementations claimed constant per-partial cost and both were linear in the
whole transcript:

* ``SymbolTableTokenizer`` finished with ``raw.replace("▁", " ").strip()`` plus a
  ``startswith`` against the previous text — three full walks of the prefix.
  Measured 3.4 us at 60 characters and **44.5 us at 20 400**.
* ``HuggingFaceTokenizer`` decoded a bounded *window* (constant) but then did
  ``state["text"] = state.get("text", "") + delta``, and ``a + b`` copies both
  operands.

The contract these tests pin, from ``Tokenizer.decode_incremental``: the
concatenated deltas equal ``decode(all_ids)``, and ``state["text"]`` is the full
text at every point.
"""

from __future__ import annotations

import random
import time

import pytest

from oasr.tokenizers.symbol_table import SymbolTableTokenizer

PIECES = [
    "▁the",
    "▁quick",
    "▁brown",
    "▁fox",
    "s",
    "ing",
    "ed",
    "▁a",
    "▁of",
    "▁print",
    "er",
    "▁and",
    "▁to",
    "'",
    "▁",
    "▁in",
    "▁i",
]
SPECIAL = 99


def _tok() -> SymbolTableTokenizer:
    tok = SymbolTableTokenizer.__new__(SymbolTableTokenizer)
    tok._table = dict(enumerate(PIECES))
    tok._special_ids = {SPECIAL}
    return tok


def _feed(tok, ids, step):
    """Drive decode_incremental in ``step``-sized chunks -> (deltas, state)."""
    state, out = {}, ""
    for i in range(0, len(ids), step):
        out += tok.decode_incremental(ids[i : i + step], state)
    return out, state


class TestMatchesWholeSequenceDecode:
    @pytest.mark.parametrize("step", [1, 2, 3, 7, 64])
    def test_deltas_concatenate_to_decode(self, step):
        tok = _tok()
        rng = random.Random(step)
        for _ in range(120):
            n = rng.randint(1, 60)
            ids = [rng.randrange(len(PIECES)) for _ in range(n)]
            if rng.random() < 0.3:
                ids.insert(rng.randrange(len(ids) + 1), SPECIAL)
            deltas, state = _feed(tok, ids, step)
            want = tok.decode(ids)
            assert deltas == want, f"deltas != decode for {ids}"
            assert state["text"] == want, f"state['text'] != decode for {ids}"

    @pytest.mark.parametrize(
        "ids,want",
        [
            ([], ""),
            ([SPECIAL], ""),
            ([14], ""),  # "▁" alone renders to whitespace only -> stripped away
            ([14, 14, 0], "the"),  # leading whitespace dropped once
            ([0, 14], "the"),  # trailing whitespace never emitted
            # Interior whitespace is preserved verbatim, doubled marker and all
            # — the incremental path must not "tidy" what decode() would keep.
            ([0, 14, 1], "the  quick"),
        ],
    )
    def test_whitespace_edges(self, ids, want):
        tok = _tok()
        state = tok.new_decode_state()
        deltas = "".join(tok.decode_incremental([t], state) for t in ids)
        assert tok.decode(ids) == want, "the expectation itself is wrong"
        assert state["text"] == want
        assert deltas == want

    def test_state_text_is_correct_after_every_chunk(self):
        """Not just at the end: a partial is read at each step."""
        tok = _tok()
        rng = random.Random(7)
        ids = [rng.randrange(len(PIECES)) for _ in range(80)]
        state = {}
        for i, t in enumerate(ids):
            tok.decode_incremental([t], state)
            assert state["text"] == tok.decode(ids[: i + 1])


class TestCostIsFlatInTranscriptLength:
    def test_per_partial_cost_does_not_grow_with_the_transcript(self):
        """The regression this file exists for.

        Times a fixed 4-token append against a short prefix and a long one. The
        old implementation was ~13x slower on the long prefix; a correct
        incremental one is flat. The threshold is loose because this is a timing
        test — it is there to catch a return to *linear*, not to police jitter.
        """
        tok = _tok()
        rng = random.Random(0)

        def per_append_us(prefix_tokens: int) -> float:
            best = None
            for _ in range(5):
                state = {}
                tok.decode_incremental(
                    [rng.randrange(len(PIECES)) for _ in range(prefix_tokens)], state
                )
                add = [rng.randrange(len(PIECES)) for _ in range(4)]
                t0 = time.perf_counter()
                for _ in range(300):
                    tok.decode_incremental(add, state)
                us = (time.perf_counter() - t0) / 300 * 1e6
                best = us if best is None else min(best, us)
            return best

        short = per_append_us(20)
        long = per_append_us(6400)
        assert long < short * 4, (
            f"per-partial cost grew {long / short:.1f}x from a 20-token prefix "
            f"({short:.2f} us) to a 6400-token one ({long:.2f} us) — "
            f"decode_incremental is linear in the transcript again"
        )


class TestStateIsSelfContained:
    def test_a_resumed_state_keeps_rendering_correctly(self):
        """The engine carries this dict across ticks; nothing may live outside it."""
        tok = _tok()
        rng = random.Random(3)
        ids = [rng.randrange(len(PIECES)) for _ in range(50)]
        state = {}
        for i in range(0, len(ids), 5):
            state = dict(state)  # as if it round-tripped through the request
            tok.decode_incremental(ids[i : i + 5], state)
        assert state["text"] == tok.decode(ids)
