# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Seam tests for per-request DecodingOptions (no GPU, no checkpoints).

Covers the option dataclass itself (validation + PyO3-dict coercion), the
shared next-token selection helper the AR strategies use, and the executor's
n-best detokenization fill.  Strategy-level behaviour (per-request prompt /
cap / sampling inside the llm strategy) lives in ``test_speech_llm.py``.
"""

import pytest
import torch

from oasr.engine.decode.detokenize import Detokenizer
from oasr.engine.generation import select_next_tokens
from oasr.engine.output_processor import OutputProcessor
from oasr.engine.request import (
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    DecodingOptions,
    Request,
    RequestOutput,
)


class TestDecodingOptionsDataclass:
    def test_defaults_are_inert(self):
        opts = DecodingOptions()
        assert opts.n_best == 1
        assert opts.max_new_tokens is None
        assert not opts.sampling
        assert opts.prompt is None

    def test_validation(self):
        with pytest.raises(ValueError, match="n_best"):
            DecodingOptions(n_best=0)
        with pytest.raises(ValueError, match="max_new_tokens"):
            DecodingOptions(max_new_tokens=0)
        with pytest.raises(ValueError, match="temperature"):
            DecodingOptions(temperature=-0.1)
        with pytest.raises(ValueError, match="top_k"):
            DecodingOptions(top_k=-1)
        with pytest.raises(ValueError, match="top_p"):
            DecodingOptions(top_p=0.0)
        with pytest.raises(ValueError, match="top_p"):
            DecodingOptions(top_p=1.5)

    def test_temperature_range(self):
        """A near-zero temperature overflows ``logits / temperature`` to inf and
        makes ``torch.multinomial`` raise inside the batched decoder step."""
        with pytest.raises(ValueError, match="temperature"):
            DecodingOptions(temperature=1e-30)
        with pytest.raises(ValueError, match="temperature"):
            DecodingOptions(temperature=1e6)
        # Bounds themselves are accepted, and 0 stays the greedy sentinel.
        assert DecodingOptions(temperature=MIN_TEMPERATURE).sampling
        assert DecodingOptions(temperature=MAX_TEMPERATURE).sampling
        assert not DecodingOptions(temperature=0.0).sampling

    def test_sampling_flag(self):
        assert not DecodingOptions(temperature=0.0).sampling
        assert DecodingOptions(temperature=0.7).sampling

    def test_coerce(self):
        assert DecodingOptions.coerce(None) is None
        opts = DecodingOptions(n_best=3)
        assert DecodingOptions.coerce(opts) is opts
        # Plain dict (the PyO3 dispatcher shape); None values mean "default".
        got = DecodingOptions.coerce(
            {"n_best": 2, "max_new_tokens": None, "temperature": 0.5, "prompt": "hi"}
        )
        assert got == DecodingOptions(n_best=2, temperature=0.5, prompt="hi")
        with pytest.raises(ValueError):
            DecodingOptions.coerce({"n_best": 0})

    def test_request_carries_options(self):
        req = Request(audio=None, streaming=True, decoding=DecodingOptions(n_best=4))
        assert req.decoding.n_best == 4
        assert Request(audio=None).decoding is None


class TestSelectNextTokens:
    def _logits(self, B=4, V=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(B, V, generator=g)

    def test_all_greedy_is_argmax(self):
        logits = self._logits()
        tokens = select_next_tokens(logits, [None] * logits.size(0))
        assert torch.equal(tokens, logits.argmax(dim=-1))

    def test_default_options_stay_greedy(self):
        logits = self._logits()
        opts = [DecodingOptions() for _ in range(logits.size(0))]
        tokens = select_next_tokens(logits, opts)
        assert torch.equal(tokens, logits.argmax(dim=-1))

    def test_topk1_sampling_is_argmax(self):
        logits = self._logits()
        opts = [DecodingOptions(temperature=5.0, top_k=1) for _ in range(logits.size(0))]
        tokens = select_next_tokens(logits, opts)
        assert torch.equal(tokens, logits.argmax(dim=-1))

    def test_tiny_top_p_keeps_the_peak(self):
        # With a strongly peaked row, a tiny nucleus keeps only the argmax.
        logits = torch.full((2, 16), -10.0)
        logits[:, 5] = 10.0
        opts = [DecodingOptions(temperature=1.0, top_p=0.5)] * 2
        tokens = select_next_tokens(logits, opts)
        assert tokens.tolist() == [5, 5]

    def test_mixed_rows(self):
        logits = self._logits(B=3)
        opts = [None, DecodingOptions(temperature=3.0, top_k=1), DecodingOptions()]
        tokens = select_next_tokens(logits, opts)
        assert torch.equal(tokens, logits.argmax(dim=-1))

    def test_topk_restricts_support(self):
        # temperature high enough to flatten, top_k=2: every draw must come
        # from the two highest-logit tokens.
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]]).repeat(64, 1)
        opts = [DecodingOptions(temperature=10.0, top_k=2)] * 64
        torch.manual_seed(0)
        tokens = select_next_tokens(logits, opts)
        assert set(tokens.tolist()) <= {2, 3}
        # Flat-ish at temperature 10 → both survivors should actually appear.
        assert len(set(tokens.tolist())) == 2


class TestNbestTexts:
    def _op(self):
        # Bare facade with only the detokenizer (id-join fallback) — enough
        # for fill_nbest_texts, which never touches config or strategy.
        op = OutputProcessor.__new__(OutputProcessor)
        op._detok = Detokenizer()
        return op

    def _output(self, rows):
        return RequestOutput(
            request_id="r0",
            text=" ".join(str(t) for t in rows[0]),
            tokens=[list(r) for r in rows],
            finished=True,
        )

    def test_default_n_best_leaves_none(self):
        req = Request(audio=None)
        out = self._output([[5, 6], [7]])
        self._op().fill_nbest_texts(req, out)
        assert out.nbest_texts is None

    def test_n_best_fills_texts(self):
        req = Request(audio=None, decoding=DecodingOptions(n_best=2))
        out = self._output([[5, 6], [7, 8], [9]])
        self._op().fill_nbest_texts(req, out)
        assert out.nbest_texts == ["5 6", "7 8"]
        assert out.nbest_texts[0] == out.text

    def test_n_best_beyond_available_rows(self):
        req = Request(audio=None, decoding=DecodingOptions(n_best=10))
        out = self._output([[5], [6]])
        self._op().fill_nbest_texts(req, out)
        assert out.nbest_texts == ["5", "6"]

    def test_single_hypothesis_untouched(self):
        req = Request(audio=None, decoding=DecodingOptions(n_best=4))
        out = self._output([[5, 6]])
        self._op().fill_nbest_texts(req, out)
        assert out.nbest_texts is None

    def test_rows_are_truncated_before_crossing_pyo3(self):
        """CTC ships its whole beam regardless of n_best; Rust discards the rest.

        Marshalling 10-16 rows to deliver 2 is pure cost on the GIL-holding
        dispatcher thread, so the engine trims to what was asked for.
        """
        req = Request(audio=None, decoding=DecodingOptions(n_best=2))
        out = self._output([[5, 6], [7, 8], [9], [10], [11]])
        out.scores = [-1.0, -2.0, -3.0, -4.0, -5.0]
        self._op().fill_nbest_texts(req, out)
        assert len(out.tokens) == 2
        assert out.scores == [-1.0, -2.0]
        assert out.nbest_texts == ["5 6", "7 8"]

    def test_a_single_hypothesis_is_left_alone(self):
        """Greedy families must not have their one row trimmed away."""
        req = Request(audio=None, decoding=DecodingOptions(n_best=5))
        out = self._output([[5, 6]])
        self._op().fill_nbest_texts(req, out)
        assert out.tokens == [[5, 6]]
        assert out.nbest_texts is None
