# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Batched modified beam search with at most one symbol per frame.

A fixed ``(B, beam)`` grid keeps scoring and reordering on the device.  Each frame
records, per slot, the slot it was extended from and the label it took (blank when
it emitted nothing); the token sequences are recovered once per chunk by walking
those back-pointers from the final slots.  This replaced a ``(B, beam, cap)``
token buffer that every frame gathered onto the new parents, scattered into and
masked -- a cost that grew with the utterance and kept the buffer's growth on the
host.  Recording is the same handful of ``(B, beam)`` writes at any length, which
is also what lets a whole block of frames replay from one CUDA graph
(``oasr/engine/beam_graph.py``).

Streaming keeps every live stream's beam in one :class:`BeamSlotPool`: a tick
gathers its cohort's rows, runs the chunk at a bucketed width
(:func:`beam_width_bucket`) and scatters them back.

Beam size one must exactly match one-symbol greedy decoding.  Equivalent
hypotheses are not merged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

from oasr.functionals.transducer import transducer_beam_topk, transducer_beam_topk_supports
from oasr.layers import _backend

#: Score assigned to the ``k - 1`` initially-dead beam slots.  A large finite
#: negative rather than ``-inf``: the slots are added to log-probs, and while
#: ``-inf + finite`` is well defined, keeping every score finite means an
#: accidental ``-inf - -inf`` anywhere downstream cannot produce a NaN that
#: silently poisons a whole utterance's beam.
_DEAD_SCORE = -1.0e30


@dataclass
class BeamState:
    """``(B, k)`` live hypotheses between chunks.

    Threaded through :func:`beam_search_chunk` so offline (one call over the
    whole utterance) and streaming (one call per chunk) share the same core --
    the same arrangement the greedy path uses for its label window.
    """

    #: ``(B, k, context_size)`` int64 predictor label windows (device).
    context: torch.Tensor
    #: ``(B, k)`` float32 accumulated log-probabilities (device).
    scores: torch.Tensor
    #: ``[B][k]`` tokens each slot emitted before the current chunk (host).  A
    #: chunk's back-pointers are folded into these when it ends, so the device
    #: state stays the same size however long a stream runs.
    prefixes: List[List[List[int]]]

    @property
    def batch(self) -> int:
        return int(self.context.size(0))

    @property
    def beam(self) -> int:
        return int(self.context.size(1))

    def hypotheses(self) -> Tuple[List[List[List[int]]], List[List[float]]]:
        """Per-utterance hypotheses, best first: ``(tokens[B][k][*], scores[B][k])``."""
        order = self.scores.argsort(dim=1, descending=True).tolist()
        scores = self.scores.tolist()
        rows = [[list(self.prefixes[b][j]) for j in order[b]] for b in range(self.batch)]
        return rows, [[scores[b][j] for j in order[b]] for b in range(self.batch)]


def init_beam_state(decoder, batch: int, beam: int, device: torch.device) -> BeamState:
    """One live hypothesis (the empty one) per utterance, the rest dead."""
    context = decoder.init_state(batch * beam, device).view(batch, beam, -1)
    scores = torch.full((batch, beam), _DEAD_SCORE, dtype=torch.float32, device=device)
    scores[:, 0] = 0.0
    return BeamState(
        context=context.contiguous(),
        scores=scores,
        prefixes=[[[] for _ in range(beam)] for _ in range(batch)],
    )


def _fused_selection(logits: torch.Tensor, beam: int) -> bool:
    """Whether a frame's selection runs as the one fused kernel.

    CPU and fp32 are out of scope -- the parity oracles' dtypes -- and an
    fp16/bf16 frame the kernel cannot take is a declared gap, never a silent
    reroute (see ``oasr/layers/_backend.py``).
    """
    if _backend.layers_backend() == "torch":
        return False
    if not logits.is_cuda or logits.dtype not in _backend.SERVED_DTYPES:
        return _backend.out_of_scope("transducer beam selection on CPU / fp32")
    if not transducer_beam_topk_supports(logits, beam):
        return _backend.take_gap(
            "transducer-beam-topk", f"vocabulary {int(logits.size(-1))}, beam {beam}"
        )
    return True


def beam_search_step(
    model,
    enc_proj_t: torch.Tensor,
    context: torch.Tensor,
    scores: torch.Tensor,
    active: torch.Tensor,
    stay: torch.Tensor,
    blank_label: torch.Tensor,
    fused: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance every live hypothesis by one encoder frame.

    Parameters
    ----------
    enc_proj_t : Tensor
        ``(B, J)`` joiner-projected encoder frame.
    context, scores : Tensor
        The beam before the frame: ``(B, k, ctx)`` label windows, ``(B, k)`` scores.
    active : Tensor
        ``(B,)`` bool -- utterances whose frame ``t`` is within their length.
        Inactive rows are left completely untouched and record "every slot is
        its own parent, emitting blank", so a short utterance in a mixed batch is
        not penalised by the padding frames the batch forced it to carry, and
        walking back through those frames is a no-op.
    stay, blank_label : Tensor
        ``(B, k)`` constants an inactive row records: ``arange(k)`` and ``blank``.
        Passed in, not built here, so a captured graph reads buffers that outlive
        the capture.
    fused : bool
        Run everything after the joiner -- log-softmax, score add, top-k, the
        ``(parent, label)`` split, the masks and the window reorder -- as one
        kernel (:func:`oasr.transducer_beam_topk`) instead of about fourteen
        torch launches.  Same scores bit for bit; a tie inside the selected
        ``k`` may take a different slot order (see the kernel's header).

    Returns ``(context, scores, parent, label)``: the beam after the frame and,
    for each new slot, the slot it extended and the label it took.  The new beam
    is **best first** -- both arms select with a sorted top-k -- and an inactive
    row keeps its order, so slot ``j`` always holds the ``j``-th best
    hypothesis and nothing downstream needs to sort the scores.

    Hypothesis merging is deliberately absent.  Two beam entries can spell the
    same sequence -- a parent taking blank keeps sequence ``A`` while a shorter
    parent ``B`` extended by ``y`` also spells ``A`` when ``A == B + [y]`` --
    and icefall log-adds those scores.  Merging needs a per-frame sequence
    comparison across the beam, which the device-side grid exists to avoid.  The
    cost of skipping it is a beam slot occasionally spent on a duplicate, i.e. an
    effectively narrower beam, never a wrong hypothesis.  Revisit with a rolling
    sequence hash if a real checkpoint shows a WER gap.
    """
    joiner = model.joiner
    decoder = model.decoder
    blank = int(model.blank_id)

    B, k, ctx = (int(d) for d in context.shape)

    dec_out = decoder(context.reshape(B * k, ctx))
    dec_proj = joiner.decoder_proj(dec_out)  # (B*k, J)
    enc_rep = enc_proj_t.unsqueeze(1).expand(B, k, enc_proj_t.size(-1)).reshape(B * k, -1)
    logits = joiner(enc_rep, dec_proj, project_input=False)  # (B*k, V)
    if fused and _fused_selection(logits, k):
        selected: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        selected = transducer_beam_topk(logits, scores, context, active, blank)
        return selected
    vocab = int(logits.size(-1))
    log_probs = torch.log_softmax(logits.float(), dim=-1).view(B, k, vocab)

    total = scores.unsqueeze(-1) + log_probs  # (B, k, V)
    top_scores, top_idx = total.view(B, k * vocab).topk(k, dim=-1)  # (B, k)
    parent = torch.div(top_idx, vocab, rounding_mode="floor")  # (B, k)
    label = top_idx - parent * vocab  # (B, k)

    # Reorder the parents' windows into the new beam; blank keeps the parent's
    # window, a real token shifts it in.
    new_context = context.gather(1, parent.unsqueeze(-1).expand(B, k, ctx))
    is_blank = label == blank
    shifted = torch.cat([new_context[:, :, 1:], label.unsqueeze(-1)], dim=2)
    new_context = torch.where(is_blank.unsqueeze(-1), new_context, shifted)

    keep = active.view(B, 1)
    return (
        torch.where(keep.unsqueeze(-1), new_context, context),
        torch.where(keep, top_scores, scores),
        torch.where(keep, parent, stay),
        torch.where(keep, label, blank_label),
    )


def step_constants(batch: int, beam: int, blank: int, device: torch.device):
    """``(stay, blank_label)`` for :func:`beam_search_step`."""
    stay = torch.arange(beam, device=device).expand(batch, beam).contiguous()
    blank_label = torch.full((batch, beam), int(blank), dtype=torch.long, device=device)
    return stay, blank_label


@torch.no_grad()
def beam_search_frames(
    model,
    enc_proj: torch.Tensor,
    lengths: torch.Tensor,
    state: BeamState,
    fused: bool = True,
) -> BeamState:
    """Advance ``state`` over every frame of a joiner-projected chunk, eagerly."""
    if int(enc_proj.size(1)) == 0:
        return state
    context, scores, parents, labels = beam_search_history(
        model, enc_proj, lengths, state.context, state.scores, fused
    )
    return fold_chunk(context, scores, state.prefixes, parents, labels, int(model.blank_id))


@torch.no_grad()
def beam_search_history(
    model,
    enc_proj: torch.Tensor,
    lengths: torch.Tensor,
    context: torch.Tensor,
    scores: torch.Tensor,
    fused: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The eager frame loop, unfolded: ``(context, scores, parents, labels)``.

    The beam after the chunk and its frame-major ``(T, B, k)`` back-pointers and
    labels -- what :class:`oasr.engine.beam_graph.BeamLoopGraphCache` returns, so
    a caller folds either the same way.
    """
    device = enc_proj.device
    B, T = int(enc_proj.size(0)), int(enc_proj.size(1))
    k = int(context.size(1))
    if T == 0:
        empty = torch.empty(0, B, k, dtype=torch.long, device=device)
        return context, scores, empty, empty
    lengths = lengths.to(device=device, dtype=torch.long)
    stay, blank_label = step_constants(B, k, int(model.blank_id), device)
    parents: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for t in range(T):
        context, scores, parent, label = beam_search_step(
            model, enc_proj[:, t], context, scores, t < lengths, stay, blank_label, fused
        )
        parents.append(parent)
        labels.append(label)
    return context, scores, torch.stack(parents, dim=0), torch.stack(labels, dim=0)


@torch.no_grad()
def beam_search_chunk(
    model,
    enc_out: torch.Tensor,
    lengths: torch.Tensor,
    state: BeamState,
    fused: bool = True,
) -> BeamState:
    """Advance ``state`` over every frame of ``enc_out``.

    ``lengths`` is per-row valid frames *within this chunk*, so the same call
    serves an offline utterance and one streaming chunk.
    """
    if int(enc_out.size(1)) == 0:
        return state
    enc_proj = model.joiner.encoder_proj(enc_out)  # (B, T, J)
    return beam_search_frames(model, enc_proj, lengths, state, fused)


@dataclass
class ChunkWalk:
    """A chunk's back-pointers walked from every final slot, per hypothesis ``h = b * k + j``.

    ``root[h]`` is the slot the hypothesis held when the chunk began -- whose
    tokens it extends -- and :meth:`tokens` the labels it emitted in the chunk.
    """

    beam: int
    root: List[int]
    flat: List[int]
    starts: List[int]
    ends: List[int]

    def tokens(self, h: int) -> List[int]:
        return self.flat[self.starts[h] : self.ends[h]]


def walk_chunk(parents: torch.Tensor, labels: torch.Tensor, blank: int) -> ChunkWalk:
    """Walk a chunk's frame-major ``(n, B, k)`` back-pointers once.

    At frame ``t`` new slot ``j`` extended slot ``parents[t, b, j]`` with
    ``labels[t, b, j]``.  Walking from each final slot to frame 0 gives the
    chunk's labels in order and the slot the hypothesis occupied when the chunk
    began.  One device->host copy, and a host walk vectorized over the ``B x k``
    grid -- ``n`` steps of array indexing, never per token.
    """
    n, B, k = (int(d) for d in parents.shape)
    hyps = B * k
    if n == 0:
        return ChunkWalk(k, [h % k for h in range(hyps)], [], [0] * hyps, [0] * hyps)
    # Back-pointers as absolute indices into the flattened (B * k) grid, so each
    # frame of the walk is two 1-D gathers rather than a 2-D fancy index.  The
    # offset is added where the history lives -- one kernel on the device, not a
    # pass over the copy on the host -- and both planes cross in one transfer.
    row_base = torch.arange(0, hyps, k, device=parents.device).view(1, B, 1)
    history = torch.stack((parents + row_base, labels)).cpu().numpy().reshape(2, n, hyps)
    parent, lab = history[0], history[1]
    base = np.repeat(np.arange(0, hyps, k), k)
    slot = np.arange(hyps)
    path = np.empty((n, hyps), dtype=lab.dtype)
    for t in range(n - 1, -1, -1):
        path[t] = lab[t, slot]
        slot = parent[t, slot]
    # Every hypothesis's emitted labels in one compaction and one ``tolist``,
    # then cut by count: per hypothesis this is a list slice, not an array op.
    tokens = path.T  # (B * k, n)
    emitted = tokens != blank
    ends = np.cumsum(emitted.sum(axis=1)).tolist()
    return ChunkWalk(
        beam=k,
        root=(slot - base).tolist(),
        flat=tokens[emitted].tolist(),
        starts=[0] + ends[:-1],
        ends=ends,
    )


def read_walk(packed: torch.Tensor, beam: int, frames: int, rows: int) -> ChunkWalk:
    """The :class:`ChunkWalk` the fused beam kernel computed on the device.

    ``packed`` is its walk buffer (:func:`oasr.functionals.transducer.beam_walk_buffer`):
    roots, counts and tokens for every hypothesis of the launch, of which the
    first ``rows`` utterances are read.  One device-to-host copy, and the
    tokens kept by count in one compaction -- the end of :func:`walk_chunk`,
    without the walk.
    """
    hyps = packed.numel() // (2 + frames)
    host = packed.cpu().numpy()
    keep = rows * beam
    counts = host[hyps : hyps + keep]
    tokens = host[2 * hyps :].reshape(hyps, frames)[:keep]
    emitted = np.arange(frames)[None, :] < counts[:, None]
    ends = np.cumsum(counts).tolist()
    return ChunkWalk(
        beam=beam,
        root=host[:keep].tolist(),
        flat=tokens[emitted].tolist(),
        starts=[0] + ends[:-1],
        ends=ends,
    )


def fold_chunk(
    context: torch.Tensor,
    scores: torch.Tensor,
    prefixes: Sequence[Sequence[List[int]]],
    parents: torch.Tensor,
    labels: torch.Tensor,
    blank: int,
) -> BeamState:
    """Close a chunk: each new slot's tokens are its root slot's prefix plus what
    it emitted in the chunk (see :func:`walk_chunk`)."""
    walk = walk_chunk(parents, labels, blank)
    k = walk.beam
    folded = [
        [prefixes[b][walk.root[h]] + walk.tokens(h) for h in range(b * k, (b + 1) * k)]
        for b in range(len(prefixes))
    ]
    return BeamState(context=context, scores=scores, prefixes=folded)


def beam_width_bucket(n: int) -> int:
    """The width a chunk of ``n`` utterances runs at: the next power of two.

    Graph captures key on the exact width (``oasr/engine/beam_graph.py``), and a
    cohort takes every width up to ``max_batch_size`` -- more than the capture
    budget holds, so the widths past it ran the eager loop.  Callers pad to this
    ladder *before* the graph/eager branch, so both see the same shapes.
    """
    width = 1
    while width < n:
        width *= 2
    return width


class BeamSlotPool:
    """Every live stream's beam, as one row of a device-resident pool.

    A stream holds a slot from its first chunk until it is released.  A tick
    gathers its cohort's rows with one index -- padding rows may point at any
    row, they are inactive and never written back -- runs the chunk, and
    scatters the new rows home.  This replaces a ``(1, k, ...)`` state per stream
    that every tick concatenated into a batch and split back, plus a freshly
    built state for every arriving stream.

    Tokens stay on the host, split at the beam's common prefix: ``committed``
    holds what every hypothesis agrees on -- final, since every future
    hypothesis extends one of today's -- and each slot keeps only the suffix it
    adds.  A chunk therefore costs O(k x suffix) per stream rather than
    O(k x stream length), where the per-stream lists used to copy every
    hypothesis's whole prefix on every tick.

    Hypothesis ``j`` of a slot is its ``j``-th best (see
    :func:`beam_search_step`), so the best transcript is ``tokens(slot, 0)``
    without reading a score back.
    """

    def __init__(self, decoder, beam: int, device: torch.device, capacity: int = 16) -> None:
        self.beam = int(beam)
        first = init_beam_state(decoder, 1, self.beam, device)
        self._init_context = first.context[0].clone()  # (k, ctx)
        self._init_scores = first.scores[0].clone()  # (k,)
        capacity = max(1, int(capacity))
        self.context = self._init_context.unsqueeze(0).repeat(capacity, 1, 1)
        self.scores = self._init_scores.unsqueeze(0).repeat(capacity, 1)
        self._free: List[int] = list(range(capacity - 1, -1, -1))
        self._committed: Dict[int, List[int]] = {}
        self._suffixes: Dict[int, List[List[int]]] = {}

    @property
    def capacity(self) -> int:
        return int(self.context.size(0))

    @property
    def live(self) -> int:
        return len(self._committed)

    def allocate(self) -> int:
        """A slot holding the empty hypothesis (one live entry, the rest dead)."""
        if not self._free:
            self._grow()
        slot = self._free.pop()
        self.context[slot].copy_(self._init_context)
        self.scores[slot].copy_(self._init_scores)
        self._committed[slot] = []
        self._suffixes[slot] = [[] for _ in range(self.beam)]
        return slot

    def release(self, slot: int) -> None:
        if self._committed.pop(slot, None) is not None:
            self._suffixes.pop(slot, None)
            self._free.append(slot)

    def _grow(self) -> None:
        old = self.capacity
        self.context = torch.cat(
            [self.context, self._init_context.unsqueeze(0).expand(old, -1, -1)], dim=0
        )
        self.scores = torch.cat(
            [self.scores, self._init_scores.unsqueeze(0).expand(old, -1)], dim=0
        )
        self._free.extend(range(2 * old - 1, old - 1, -1))

    def gather(self, index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(context, scores)`` for the rows ``index`` names, as owned copies."""
        return self.context.index_select(0, index), self.scores.index_select(0, index)

    def scatter(self, index: torch.Tensor, context: torch.Tensor, scores: torch.Tensor) -> None:
        """Write a chunk's new rows back to the slots ``index`` names."""
        self.context.index_copy_(0, index, context)
        self.scores.index_copy_(0, index, scores)

    def fold(self, slots: Sequence[int], walk: ChunkWalk) -> None:
        """Extend each slot's hypotheses by the chunk, then commit their common prefix."""
        k = self.beam
        for b, slot in enumerate(slots):
            old = self._suffixes[slot]
            new = [old[walk.root[h]] + walk.tokens(h) for h in range(b * k, (b + 1) * k)]
            shared = len(os.path.commonprefix(new))
            if shared:
                self._committed[slot].extend(new[0][:shared])
                new = [suffix[shared:] for suffix in new]
            self._suffixes[slot] = new

    def tokens(self, slot: int, j: int) -> List[int]:
        """Hypothesis ``j`` of ``slot`` in full: the committed prefix plus its suffix."""
        return self._committed[slot] + self._suffixes[slot][j]


def select_rows(state: BeamState, rows: Union[torch.Tensor, Sequence[int]]) -> BeamState:
    """Keep only ``rows`` of the batch (copies)."""
    if isinstance(rows, torch.Tensor):
        index = rows.to(device=state.context.device, dtype=torch.long)
        keep = index.tolist()
    else:
        keep = [int(r) for r in rows]
        index = torch.tensor(keep, dtype=torch.long, device=state.context.device)
    return BeamState(
        context=state.context.index_select(0, index),
        scores=state.scores.index_select(0, index),
        prefixes=[list(map(list, state.prefixes[r])) for r in keep],
    )


def stack_states(states: List[BeamState]) -> BeamState:
    """Stack per-stream ``(1, k, ...)`` states into one batched state.

    For a caller that keeps one state per stream (the strategy keeps a
    :class:`BeamSlotPool` instead).  The device half is two small
    concatenations; the token prefixes stay on the host.
    """
    if not states:
        raise ValueError("stack_states needs at least one state")
    return BeamState(
        context=torch.cat([s.context for s in states], dim=0),
        scores=torch.cat([s.scores for s in states], dim=0),
        prefixes=[row for s in states for row in s.prefixes],
    )


__all__ = [
    "BeamSlotPool",
    "BeamState",
    "ChunkWalk",
    "beam_search_chunk",
    "beam_search_frames",
    "beam_search_history",
    "beam_search_step",
    "beam_width_bucket",
    "fold_chunk",
    "init_beam_state",
    "read_walk",
    "select_rows",
    "stack_states",
    "step_constants",
    "walk_chunk",
]
