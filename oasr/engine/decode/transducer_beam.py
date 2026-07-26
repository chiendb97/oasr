# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Batched modified beam search for the transducer decode family.

This is icefall's ``modified_beam_search`` — the variant that emits **at most one
symbol per encoder frame** — batched over utterances *and* beam width so a whole
micro-batch advances in one set of kernels per frame.  It is the practical RNNT
beam: full ALSD's variable emissions-per-frame make the per-frame hypothesis set
ragged, which is exactly what stops it from vectorizing, while the modified
variant keeps a fixed ``(B, k)`` grid throughout.

Per frame, for every one of the ``B * k`` live hypotheses:

1. run the stateless predictor over its label window and join with the frame,
2. add ``log_softmax`` over the vocabulary to the hypothesis score,
3. take the top ``k`` over the flattened ``(k, V)`` grid **per utterance**,
4. blank keeps the parent's label window; a real token shifts it and appends.

Two design points worth keeping:

**Hypothesis tokens live on the device.** The obvious implementation keeps a
Python ``list`` per hypothesis and reorders them by the parent indices each
frame, which is ``B * k`` list copies per frame — Θ(T²) over an utterance.  A
padded ``(B, k, cap)`` int64 tensor plus a length makes the reorder one
``gather`` and the append one ``scatter_``.

**Blank writes and then doesn't advance.** Appending is unconditional: a blank
transition writes its token at ``tok_len`` but leaves ``tok_len`` alone, so the
next real token overwrites it.  That removes a mask from the hot path, and the
index is always in bounds because at most one symbol is emitted per frame, so
``tok_len <= t`` when frame ``t`` writes.

Relationship to greedy: with ``max_sym_per_frame = 1``, greedy decoding *is*
beam search at ``k = 1`` — both take the argmax and advance.  That identity is
the exactness gate in ``tests/test_transducer.py``; there is no reference
implementation to diff against otherwise, and a beam search that silently
disagrees with greedy at ``k=1`` is broken in a way WER on random weights would
never show.

Hypothesis merging (log-adding the scores of two beam entries that spell the same
token sequence) is **not** implemented; see :func:`beam_search_step`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

#: Score assigned to the ``k - 1`` initially-dead beam slots.  A large finite
#: negative rather than ``-inf``: the slots are added to log-probs, and while
#: ``-inf + finite`` is well defined, keeping every score finite means an
#: accidental ``-inf - -inf`` anywhere downstream cannot produce a NaN that
#: silently poisons a whole utterance's beam.
_DEAD_SCORE = -1.0e30

#: Growth granularity for the device-side token buffer, in tokens.  Streaming
#: cannot know the final length up front, so the buffer grows geometrically with
#: this floor rather than once per chunk.
_TOKEN_CAP_GROWTH = 128


@dataclass
class BeamState:
    """``(B, k)`` live hypotheses, all device-side.

    Threaded through :func:`beam_search_chunk` so offline (one call over the
    whole utterance) and streaming (one call per chunk) share the same core —
    the same arrangement the greedy path uses for its label window.
    """

    #: ``(B, k, context_size)`` int64 predictor label windows.
    context: torch.Tensor
    #: ``(B, k)`` float32 accumulated log-probabilities.
    scores: torch.Tensor
    #: ``(B, k, cap)`` int64 emitted tokens, left-aligned.
    tokens: torch.Tensor
    #: ``(B, k)`` int64 count of valid entries in ``tokens``.
    tok_len: torch.Tensor

    @property
    def batch(self) -> int:
        return int(self.context.size(0))

    @property
    def beam(self) -> int:
        return int(self.context.size(1))

    def ensure_capacity(self, extra: int) -> None:
        """Grow ``tokens`` so ``extra`` more emissions per hypothesis fit."""
        need = int(self.tok_len.max().item()) + int(extra)
        cap = int(self.tokens.size(2))
        if need <= cap:
            return
        grow = max(need, cap * 2, _TOKEN_CAP_GROWTH)
        pad = torch.zeros(
            self.batch,
            self.beam,
            grow - cap,
            dtype=self.tokens.dtype,
            device=self.tokens.device,
        )
        self.tokens = torch.cat([self.tokens, pad], dim=2)

    def hypotheses(self) -> Tuple[List[List[List[int]]], List[List[float]]]:
        """Read back per-utterance hypotheses, best first.

        Returns ``(tokens[B][k][*], scores[B][k])``.  One host sync for the whole
        beam — the same discipline as the greedy loop's single readback.
        """
        order = self.scores.argsort(dim=1, descending=True)  # (B, k)
        toks = self.tokens.gather(1, order.unsqueeze(-1).expand(-1, -1, self.tokens.size(2)))
        lens = self.tok_len.gather(1, order)
        scores = self.scores.gather(1, order)
        toks_h = toks.tolist()
        lens_h = lens.tolist()
        out = [[toks_h[b][j][: lens_h[b][j]] for j in range(self.beam)] for b in range(self.batch)]
        return out, scores.tolist()


def init_beam_state(
    decoder,
    batch: int,
    beam: int,
    device: torch.device,
    capacity: int = 0,
) -> BeamState:
    """One live hypothesis (the empty one) per utterance, the rest dead."""
    context = decoder.init_state(batch * beam, device).view(batch, beam, -1)
    scores = torch.full((batch, beam), _DEAD_SCORE, dtype=torch.float32, device=device)
    scores[:, 0] = 0.0
    cap = max(int(capacity), _TOKEN_CAP_GROWTH)
    return BeamState(
        context=context.contiguous(),
        scores=scores,
        tokens=torch.zeros(batch, beam, cap, dtype=torch.long, device=device),
        tok_len=torch.zeros(batch, beam, dtype=torch.long, device=device),
    )


def beam_search_step(
    model,
    enc_proj_t: torch.Tensor,
    state: BeamState,
    active: torch.Tensor,
) -> BeamState:
    """Advance every live hypothesis by one encoder frame.

    Parameters
    ----------
    enc_proj_t : Tensor
        ``(B, J)`` joiner-projected encoder frame.
    active : Tensor
        ``(B,)`` bool — utterances whose frame ``t`` is within their length.
        Inactive rows are left completely untouched (score, window and tokens),
        so a short utterance in a mixed batch is not penalised by the padding
        frames the batch forced it to carry.

    Hypothesis merging is deliberately absent.  Two beam entries can spell the
    same sequence — a parent taking blank keeps sequence ``A`` while a shorter
    parent ``B`` extended by ``y`` also spells ``A`` when ``A == B + [y]`` —
    and icefall log-adds those scores.  Merging needs a per-frame sequence
    comparison across the beam, which is precisely the Θ(T²) host-side work the
    device-side token buffer exists to avoid.  The cost of skipping it is a beam
    slot occasionally spent on a duplicate, i.e. an effectively narrower beam,
    never a wrong hypothesis.  Revisit with a rolling sequence hash if a real
    checkpoint shows a WER gap.
    """
    joiner = model.joiner
    decoder = model.decoder
    blank = int(model.blank_id)

    B, k = state.batch, state.beam
    ctx = int(state.context.size(2))

    dec_out = decoder(state.context.reshape(B * k, ctx))
    dec_proj = joiner.decoder_proj(dec_out)  # (B*k, J)
    enc_rep = enc_proj_t.unsqueeze(1).expand(B, k, enc_proj_t.size(-1)).reshape(B * k, -1)
    logits = joiner(enc_rep, dec_proj, project_input=False)  # (B*k, V)
    vocab = int(logits.size(-1))
    log_probs = torch.log_softmax(logits.float(), dim=-1).view(B, k, vocab)

    total = state.scores.unsqueeze(-1) + log_probs  # (B, k, V)
    top_scores, top_idx = total.view(B, k * vocab).topk(k, dim=-1)  # (B, k)
    parent = torch.div(top_idx, vocab, rounding_mode="floor")  # (B, k)
    token = top_idx - parent * vocab  # (B, k)

    # Reorder the parents' state into the new beam.
    new_context = state.context.gather(1, parent.unsqueeze(-1).expand(B, k, ctx))
    cap = int(state.tokens.size(2))
    new_tokens = state.tokens.gather(1, parent.unsqueeze(-1).expand(B, k, cap))
    new_len = state.tok_len.gather(1, parent)

    # Blank keeps the parent's label window; a real token shifts it in.
    is_blank = token == blank
    shifted = torch.cat([new_context[:, :, 1:], token.unsqueeze(-1)], dim=2)
    new_context = torch.where(is_blank.unsqueeze(-1), new_context, shifted)

    # Unconditional append (see the module docstring): a blank writes at
    # ``new_len`` and does not advance it, so the slot is reused.
    new_tokens = new_tokens.scatter(
        2, new_len.clamp(max=cap - 1).unsqueeze(-1), token.unsqueeze(-1)
    )
    new_len = new_len + (~is_blank).long()

    keep = active.view(B, 1)
    return BeamState(
        context=torch.where(keep.unsqueeze(-1), new_context, state.context),
        scores=torch.where(keep, top_scores, state.scores),
        tokens=torch.where(keep.unsqueeze(-1), new_tokens, state.tokens),
        tok_len=torch.where(keep, new_len, state.tok_len),
    )


@torch.no_grad()
def beam_search_chunk(
    model,
    enc_out: torch.Tensor,
    lengths: torch.Tensor,
    state: BeamState,
) -> BeamState:
    """Advance ``state`` over every frame of ``enc_out``.

    ``lengths`` is per-row valid frames *within this chunk*, so the same call
    serves an offline utterance and one streaming chunk.
    """
    device = enc_out.device
    T = int(enc_out.size(1))
    if T == 0:
        return state
    lengths = lengths.to(device=device, dtype=torch.long)
    # At most one emission per frame, so this chunk can add at most T tokens.
    state.ensure_capacity(T)
    enc_proj = model.joiner.encoder_proj(enc_out)  # (B, T, J)
    for t in range(T):
        state = beam_search_step(model, enc_proj[:, t], state, active=t < lengths)
    return state


def select_rows(state: BeamState, rows: torch.Tensor) -> BeamState:
    """Keep only ``rows`` of the batch (streaming regroups per tick)."""
    return BeamState(
        context=state.context.index_select(0, rows),
        scores=state.scores.index_select(0, rows),
        tokens=state.tokens.index_select(0, rows),
        tok_len=state.tok_len.index_select(0, rows),
    )


def stack_states(states: List[BeamState]) -> Optional[BeamState]:
    """Stack per-stream ``(1, k, ...)`` states into one batched state.

    Streaming groups streams by chunk length per tick, so the group's membership
    changes every tick; the token buffers are padded to the widest before the
    concatenation.
    """
    if not states:
        return None
    cap = max(int(s.tokens.size(2)) for s in states)
    toks = []
    for s in states:
        c = int(s.tokens.size(2))
        if c < cap:
            pad = torch.zeros(
                s.tokens.size(0),
                s.tokens.size(1),
                cap - c,
                dtype=s.tokens.dtype,
                device=s.tokens.device,
            )
            toks.append(torch.cat([s.tokens, pad], dim=2))
        else:
            toks.append(s.tokens)
    return BeamState(
        context=torch.cat([s.context for s in states], dim=0),
        scores=torch.cat([s.scores for s in states], dim=0),
        tokens=torch.cat(toks, dim=0),
        tok_len=torch.cat([s.tok_len for s in states], dim=0),
    )


__all__ = [
    "BeamState",
    "beam_search_chunk",
    "beam_search_step",
    "init_beam_state",
    "select_rows",
    "stack_states",
]
