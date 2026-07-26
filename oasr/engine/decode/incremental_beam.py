# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Beam search for the incremental (label-synchronous) AR families.

Greedy AR decode keeps one hypothesis per request and retires a row when it hits
EOS.  Beam search keeps ``k`` per request, so the decoder grid is ``(B, k)``
flattened to ``B * k`` rows, and a *request* retires only once its whole beam has
finished.  Everything else — the tick budget, the round-robin over groups, the
partial/final emission — is unchanged, which is why this lives beside
:class:`~oasr.engine.decode.incremental.ArGroup` rather than replacing it.

The enabling observation is that the AR decoder surface already has everything
beam search needs.  ``select(state, keep)`` is implemented with
``index_select``, and ``index_select`` permits **repeated** indices — so the same
one call both *expands* B prefilled rows into ``B * k`` (indices
``[0,0,..,1,1,..]``) and *reorders* the grid to follow each new beam slot's
parent.  No new model-side method, and every model that supports greedy AR
decode supports beam search for free.

Two things are deliberately different from the transducer beam
(:mod:`oasr.engine.decode.transducer_beam`):

* **Hypotheses are host-side lists here.** The transducer runs one beam step per
  encoder *frame* with a trivial predictor, so per-frame Python work dominates
  and the tokens have to live on the device.  An AR step is a full decoder
  forward — milliseconds — so ``k`` list copies per step are free by comparison,
  and lists keep the EOS bookkeeping readable.
* **Finished hypotheses are set aside, not dropped.** A beam slot that emits EOS
  is complete: its score is final and it must stop competing, but it is still a
  candidate for the answer.  Greedy has no analogue because its single
  hypothesis finishing *is* the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..request import DecodingOptions, Request

#: Score for a beam slot that is no longer live (finished or never started).
#: Finite rather than ``-inf`` for the same reason as the transducer beam: every
#: score stays a real number, so no downstream arithmetic can make a NaN.
DEAD_SCORE = -1.0e30


@dataclass
class ArBeamGroup:
    """One encoded micro-batch generating with ``beam`` hypotheses per request.

    Row ``b * beam + j`` of the decoder state is utterance ``b``'s beam slot
    ``j``; that layout is what lets one ``select`` call do the reorder.
    """

    requests: List[Request]
    #: Decoder state for ``len(requests) * beam`` rows.
    state: Dict[str, Any]
    #: ``(B * beam, V)`` pending selection input.
    last_logits: torch.Tensor
    beam: int
    #: Per-**request** generation cap (all its slots share it).
    max_new: List[int]
    opts: List[Optional[DecodingOptions]]
    #: ``(B, beam)`` accumulated log-probabilities.
    scores: torch.Tensor
    #: ``B x beam`` live hypotheses (EOS is not appended).
    tokens: List[List[List[int]]] = field(default_factory=list)
    #: ``B`` lists of ``(score, tokens)`` completed hypotheses.
    finished: List[List[Tuple[float, List[int]]]] = field(default_factory=list)
    #: Generated positions so far (every row advances in lockstep).
    steps: int = 0

    def __post_init__(self) -> None:
        B = len(self.requests)
        if not self.tokens:
            self.tokens = [[[] for _ in range(self.beam)] for _ in range(B)]
        if not self.finished:
            self.finished = [[] for _ in range(B)]

    @property
    def batch(self) -> int:
        return len(self.requests)

    @property
    def first_generation_step(self) -> bool:
        """Whether no token has been generated yet.

        Read by the families' logit processors (Whisper's
        ``begin_suppress_tokens`` applies only here).  A beam group cannot answer
        this from ``tokens``: that is ``B x k`` *nested* lists, so
        ``not tokens[0]`` is ``False`` from the very first step because
        ``[[], [], ...]`` is a non-empty list.  ``steps`` is unambiguous.
        """
        return self.steps == 0

    def best(self, index: int, length_penalty: float) -> Tuple[List[int], float, bool]:
        """Best hypothesis for request ``index``: ``(tokens, score, finished)``.

        Prefers a completed hypothesis; falls back to the best live one when the
        request ran out of generation budget before any slot emitted EOS — which
        is exactly the case ``finish_reason="length"`` reports.
        """
        candidates = [
            (self._normalized(sc, len(tk), length_penalty), tk, True)
            for sc, tk in self.finished[index]
        ]
        live = self.scores[index].tolist()
        candidates += [
            (
                self._normalized(live[j], len(self.tokens[index][j]), length_penalty),
                self.tokens[index][j],
                False,
            )
            for j in range(self.beam)
            if live[j] > DEAD_SCORE / 2
        ]
        if not candidates:
            return [], 0.0, False
        score, tokens, done = max(candidates, key=lambda c: c[0])
        return list(tokens), float(score), done

    def ranked(self, index: int, length_penalty: float) -> Tuple[List[List[int]], List[float]]:
        """All candidates for request ``index``, best first — this is the n-best.

        Completed and live hypotheses are ranked together: a request that hit its
        generation cap has no completed hypothesis at all, and returning nothing
        would be worse than returning the truncated ones.
        """
        pool = [
            (self._normalized(sc, len(tk), length_penalty), list(tk))
            for sc, tk in self.finished[index]
        ]
        live = self.scores[index].tolist()
        pool += [
            (
                self._normalized(live[j], len(self.tokens[index][j]), length_penalty),
                list(self.tokens[index][j]),
            )
            for j in range(self.beam)
            if live[j] > DEAD_SCORE / 2
        ]
        pool.sort(key=lambda c: c[0], reverse=True)
        # Drop duplicate sequences, keeping the best-scoring occurrence, and cap
        # at the beam width.  The pool can exceed ``beam`` — completed *and* live
        # candidates rank together — but "``beam_size`` hypotheses" is the
        # contract callers expect, and rows past it would only cross the PyO3
        # boundary for the serving layer to discard.
        seen, rows, scores = set(), [], []
        for score, toks in pool:
            key = tuple(toks)
            if key in seen:
                continue
            seen.add(key)
            rows.append(toks)
            scores.append(score)
            if len(rows) >= self.beam:
                break
        return rows, scores

    @staticmethod
    def _normalized(score: float, length: int, penalty: float) -> float:
        """Length-normalised score (GNMT form) — ``penalty=0`` disables it.

        Raw log-probability sums favour short hypotheses, which for ASR means
        systematically truncated transcripts.  The GNMT denominator
        ``((5 + n) / 6) ** penalty`` is what Whisper's own beam search uses.
        """
        if penalty == 0.0:
            return score
        return score / (((5.0 + max(length, 1)) / 6.0) ** penalty)


def expand_indices(batch: int, beam: int, device: torch.device) -> torch.Tensor:
    """``[0]*k + [1]*k + ...`` — turn ``B`` prefilled rows into ``B * k``."""
    return torch.arange(batch, device=device).repeat_interleave(beam)


def initial_scores(batch: int, beam: int, device: torch.device) -> torch.Tensor:
    """Only slot 0 of each request is live; the rest are dead until first split.

    Without this every slot would start identical and the first ``topk`` would
    return the same token ``k`` times, collapsing the beam to width 1 forever.
    """
    scores = torch.full((batch, beam), DEAD_SCORE, dtype=torch.float32, device=device)
    scores[:, 0] = 0.0
    return scores


def topk_step(
    log_probs: torch.Tensor,
    scores: torch.Tensor,
    beam: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One beam expansion.

    ``log_probs`` is ``(B * k, V)``; ``scores`` is ``(B, k)``.  Returns
    ``(new_scores (B,k), parent (B,k) local slot, token (B,k))``.
    """
    B, k = scores.shape
    vocab = int(log_probs.size(-1))
    total = scores.reshape(B * k, 1) + log_probs
    top_scores, top_idx = total.view(B, k * vocab).topk(beam, dim=-1)
    parent = torch.div(top_idx, vocab, rounding_mode="floor")
    token = top_idx - parent * vocab
    return top_scores, parent, token


def global_parent_rows(parent: torch.Tensor, beam: int) -> torch.Tensor:
    """Local slot indices → flat decoder rows, for ``select``."""
    B = int(parent.size(0))
    offset = (torch.arange(B, device=parent.device) * beam).unsqueeze(1)
    return (parent + offset).reshape(-1)


def is_finite_score(x: float) -> bool:
    """Whether a slot is still live (not the dead sentinel, not NaN)."""
    return not math.isnan(x) and x > DEAD_SCORE / 2


__all__ = [
    "ArBeamGroup",
    "DEAD_SCORE",
    "expand_indices",
    "global_parent_rows",
    "initial_scores",
    "is_finite_score",
    "topk_step",
]
