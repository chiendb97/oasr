# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared machinery for label-synchronous (autoregressive) decode strategies.

The AED and speech-LLM strategies differ in four places — how a micro-batch is
prefilled, how logits are filtered before selection, what counts as EOS, and
whether partial outputs are emitted — and agreed on everything else: group
bookkeeping, the budget loop, per-row finalisation, abort, and the four one-shot /
streaming surfaces they cannot serve.  That "everything else" is here, so a third
AR family is a handful of hooks rather than another 250-line copy.

**Batching model, and why groups are not merged.** Each encoded micro-batch
becomes one :class:`ArGroup` that generates together, and
:meth:`IncrementalArStrategy.advance` round-robins one batched decoder step across
groups until the tick budget is spent.

Groups are **never merged**, and that costs real throughput: an AR decoder step is
weight-read bound, so its cost barely depends on how many rows it carries.  Total
decoder forwards is the *sum over groups* of each group's step count, so two
groups cost roughly twice one group with the same total rows.  Measured on
Qwen2-Audio-7B (4 utterances, 124 tokens): 922 ms when they arrive together
(one group) vs 1614 ms one-per-tick (two groups) — identical work.

Merging is not currently expressible.  Both decoder surfaces track the generation
offset as a **shared scalar** — ``WhisperDecoder`` keeps ``state["pos"]`` as an
``int`` used for both the position embedding and the KV write offset, and
``Qwen2Lm`` writes new KV at ``state["len"]`` into a shared-``cap`` buffer — so
rows sitting at different offsets cannot share a forward.  Merging needs per-row
KV offsets plus padding to a common width, which is the same prerequisite paged
decoder KV needs (and that is blocked on the CuteDSL FMHA masked-tile fix).

Until then the lever is admission, not merging: ``EngineConfig.decode_admit_window_ms``
holds a thin waiting queue briefly so near-simultaneous arrivals prefill as one
wide group instead of several narrow ones (see
:meth:`oasr.engine.executor.offline.OfflineExecutor._batch_wide_enough`).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

import torch

from ..generation import select_next_tokens
from ..request import DecodingOptions, Request, RequestOutput
from .alignment import TokenAlignment, wants_word_timings
from .base import DecodeStrategy
from .incremental_beam import (
    DEAD_SCORE,
    ArBeamGroup,
    expand_indices,
    global_parent_rows,
    initial_scores,
    is_finite_score,
    topk_step,
)
from .options import option

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from ..generation import StepBudget
    from .detokenize import Detokenizer

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class ArGroup:
    """One encoded micro-batch generating together.

    Rows leave as they finish; ``state`` / ``last_logits`` are compacted to match
    via the decoder's ``select``, so row *i* of every field refers to the same
    request throughout.

    ``eq=False``: a generated ``__eq__`` would compare ``last_logits`` elementwise
    and raise on ``bool()`` of a multi-element tensor, so groups compare by
    identity — which is what every call site wants anyway.
    """

    requests: List[Request]
    #: Opaque per-family decoder state (KV cache, positions, ...).
    state: Dict[str, Any]
    #: ``(B_active, V)`` — the pending selection input for the next step.
    last_logits: torch.Tensor
    #: Per-row generation cap.
    max_new: List[int]
    #: Per-row :class:`DecodingOptions` (``None`` = engine defaults).
    opts: List[Optional[DecodingOptions]]
    #: Tokens generated so far, per row.
    tokens: List[List[int]] = field(default_factory=list)
    #: Per-row incremental-detokenization state (T3).  AR generation only ever
    #: appends, so a partial can decode just the new ids instead of the whole
    #: prefix — at 32 tokens/tick over a 448-token run that is ~3.1k
    #: token-decodes replaced by 448.
    detok_state: List[Dict[str, Any]] = field(default_factory=list)
    #: Per-row ``(1, T_enc, D)`` encoder output, retained **only** for the rows
    #: that asked for word timings; ``None`` for every other row.  The
    #: alignment pass is a second forward against the encoder states, and by
    #: the time a row finishes the micro-batch's tensor is long gone —
    #: retaining per row rather than per group is what keeps a single timed
    #: request from pinning the whole batch's encoder output.
    align_enc: List[Optional[torch.Tensor]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = [[] for _ in self.requests]
        if not self.detok_state:
            self.detok_state = [{} for _ in self.requests]
        if not self.align_enc:
            self.align_enc = [None] * len(self.requests)

    @property
    def first_generation_step(self) -> bool:
        """Whether no token has been generated yet (rows advance in lockstep)."""
        return not any(self.tokens)

    def keep_rows(self, keep: List[int]) -> None:
        """Drop every row not in ``keep`` from the host-side bookkeeping."""
        self.requests = [self.requests[r] for r in keep]
        self.tokens = [self.tokens[r] for r in keep]
        self.max_new = [self.max_new[r] for r in keep]
        self.opts = [self.opts[r] for r in keep]
        self.detok_state = [self.detok_state[r] for r in keep]
        self.align_enc = [self.align_enc[r] for r in keep]


@dataclass
class Prefill:
    """What a family's prefill produces for a fresh micro-batch."""

    state: Dict[str, Any]
    #: ``(B, V)`` logits for the first token selection.
    logits: torch.Tensor
    #: Per-row generation cap, already clamped to the model's position capacity.
    max_new: List[int]


@dataclass(frozen=True)
class ArOptions:
    """Options shared by every incremental autoregressive family.

    Subclass to add family-specific knobs (see ``LlmOptions``); the base's
    ``max_new_tokens`` stays a single declaration so AED and LLM cannot drift.
    """

    max_new_tokens: int = option(
        448,
        legacy="max_new_tokens",
        doc="Default generation cap; DecodingOptions.max_new_tokens overrides per request.",
    )
    beam_size: int = option(
        1,
        doc=(
            "1 (default) = greedy.  >1 runs beam search over the same incremental "
            "protocol, which is also what makes DecodingOptions.n_best return "
            "real alternatives for the AR families."
        ),
    )
    length_penalty: float = option(
        1.0,
        doc=(
            "GNMT length normalisation exponent applied when ranking beam "
            "hypotheses; 0 disables it (raw log-prob sums, which favour short "
            "transcripts).  Ignored by greedy."
        ),
    )

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens!r}")
        if self.beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {self.beam_size!r}")
        if self.length_penalty < 0.0:
            raise ValueError(f"length_penalty must be >= 0, got {self.length_penalty!r}")


class IncrementalArStrategy(DecodeStrategy):
    """Base for ``incremental = True`` strategies (AED / speech-LLM).

    Subclasses implement :meth:`_prefill` and :meth:`_is_eos`, optionally
    :meth:`_process_logits`, and set :attr:`emit_partials`.  Everything below the
    hooks is shared and should not be overridden.
    """

    consumes: ClassVar[str] = "hidden"
    incremental: ClassVar[bool] = True
    options_cls: ClassVar[type] = ArOptions
    #: Emit a ``finished=False`` output per advanced request per tick.  Greedy AED
    #: reports finals only; the speech-LLM path streams tokens.
    emit_partials: ClassVar[bool] = False

    def __init__(
        self,
        config: "EngineConfig",
        detok: "Detokenizer",
        model: "BaseAsrModel" = None,
    ) -> None:
        super().__init__(config, detok, model)
        self._groups: List[ArGroup] = []

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _prefill(
        self,
        requests: List[Request],
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
    ) -> Prefill:
        """Run the decoder's prefill for a freshly-encoded micro-batch."""
        raise NotImplementedError

    @abstractmethod
    def _is_eos(self, token: int) -> bool:
        """Whether ``token`` ends generation for a row."""
        raise NotImplementedError

    def _process_logits(self, logits: torch.Tensor, group: ArGroup) -> torch.Tensor:
        """Filter ``(B_active, V)`` logits before selection.  Default: no-op.

        Must **not** mutate ``logits`` in place — it may alias
        ``group.last_logits`` (``Tensor.float()`` returns ``self`` when the tensor
        is already fp32), which :meth:`free_session` reads afterwards.  Return a
        new tensor instead (``index_fill`` / ``masked_fill``).
        """
        return logits

    def _decoder(self):
        """The batched incremental decoder surface (``prefill``/``step``/``select``)."""
        return getattr(self._model, "decoder", None)

    # ------------------------------------------------------------------
    # Admission budgeting (C3)
    # ------------------------------------------------------------------

    def kv_bytes_per_row(self) -> Optional[int]:
        """Worst-case decoder-KV bytes for one row at its generation cap.

        ``2 (K and V) * layers * kv_heads * head_dim * itemsize`` per token,
        times the row's position budget.  The position budget is
        ``prompt + max_new_tokens``, and the prompt length is knowable ahead of
        the encode **because these families run a fixed-window frontend** — a
        speech-LLM's audio prompt is the same length for a 2 s clip and a 30 s
        one.  That is what makes a pre-admission byte estimate meaningful here
        and not for a variable-length frontend.

        ``None`` when the model does not declare ``decoder_cache_spec``; the
        budget then stays off rather than guessing a footprint.
        """
        spec = getattr(self._model, "decoder_cache_spec", None)
        if spec is None:
            return None
        itemsize = torch.empty((), dtype=self._config.dtype).element_size()
        per_token = 2 * spec.num_layers * spec.n_kv_head * spec.head_dim * itemsize
        return int(per_token) * int(self._position_budget())

    def _position_budget(self) -> int:
        """Positions one row can occupy: prompt estimate + generation cap."""
        return int(self._prompt_len_estimate()) + int(self.options.max_new_tokens)

    def _prompt_len_estimate(self) -> int:
        """Upper bound on the prefill prompt length, in decoder positions.

        Overridden by families whose prompt is more than a few control tokens
        (the speech-LLM splices ~750 audio embeddings into it).
        """
        return 8

    # ------------------------------------------------------------------
    # Per-request generation cap
    # ------------------------------------------------------------------

    def _row_cap(self, request: Request, capacity: int) -> int:
        """Per-row generation cap: request override else engine default, clamped.

        ``capacity`` is the row's remaining position budget in the decoder.
        """
        opts = getattr(request, "decoding", None)
        requested = (
            opts.max_new_tokens
            if opts is not None and opts.max_new_tokens is not None
            else int(self.options.max_new_tokens)
        )
        return max(1, min(int(requested), capacity))

    # ------------------------------------------------------------------
    # Incremental protocol (shared)
    # ------------------------------------------------------------------

    @property
    def _beam(self) -> int:
        """Configured beam width (1 = greedy)."""
        return int(getattr(self.options, "beam_size", 1))

    @property
    def _length_penalty(self) -> float:
        return float(getattr(self.options, "length_penalty", 1.0))

    def begin_offline(
        self,
        requests: List[Request],
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
    ) -> None:
        plan = self._prefill(requests, enc_out, enc_lengths)
        opts = [getattr(r, "decoding", None) for r in requests]
        if self._beam > 1:
            self._groups.append(self._make_beam_group(requests, plan, opts))
            return
        # Retain one encoder row per request that asked to be timed.  ``clone``
        # rather than a view: the micro-batch tensor is far larger than one row
        # and a view would keep all of it alive until the last row retires.
        # Greedy only: the beam path returned above, and its group has no slot
        # for a retained encoder row.  ``word_timing_modes`` says so, so this is
        # a belt-and-braces guard rather than the check that matters.
        align_enc: List[Optional[torch.Tensor]] = [None] * len(requests)
        if self.word_timing_modes:
            for row, req in enumerate(requests):
                if wants_word_timings(req):
                    align_enc[row] = enc_out[row : row + 1].clone()
        self._groups.append(
            ArGroup(
                requests=list(requests),
                state=plan.state,
                last_logits=plan.logits,
                max_new=list(plan.max_new),
                opts=opts,
                align_enc=align_enc,
            )
        )

    def _make_beam_group(self, requests, plan, opts) -> ArBeamGroup:
        """Widen a prefilled ``B``-row state into a ``B * k`` beam grid.

        The expansion is one ``select`` with repeated indices — the decoder
        surface needs no beam-specific method, so any model that supports greedy
        AR decode supports beam search unchanged.
        """
        k = self._beam
        B = len(requests)
        device = plan.logits.device
        idx = expand_indices(B, k, device)
        state = self._decoder().select(plan.state, idx)
        logits = plan.logits.index_select(0, idx)
        return ArBeamGroup(
            requests=list(requests),
            state=state,
            last_logits=logits,
            beam=k,
            max_new=list(plan.max_new),
            opts=opts,
            scores=initial_scores(B, k, device),
        )

    def has_pending(self) -> bool:
        return bool(self._groups)

    def advance(self, budget: "StepBudget") -> List[RequestOutput]:
        outputs: List[RequestOutput] = []
        advanced: List[ArGroup] = []
        while self._groups and budget.take():
            # Round-robin: pop the front group, advance it one batched step,
            # re-queue it at the back if it still has active rows.
            group = self._groups.pop(0)
            if isinstance(group, ArBeamGroup):
                outputs.extend(self._advance_beam_group(group))
            else:
                outputs.extend(self._advance_group(group))
            if group.requests:
                self._groups.append(group)
                if not any(g is group for g in advanced):
                    advanced.append(group)
        if self.emit_partials and advanced:
            beams = [g for g in advanced if isinstance(g, ArBeamGroup)]
            greedy = [g for g in advanced if not isinstance(g, ArBeamGroup)]
            outputs.extend(self._partials(greedy))
            outputs.extend(self._beam_partials(beams))
        return outputs

    def _advance_group(self, group: ArGroup) -> List[RequestOutput]:
        """Select one token per active row, retire finished rows, take one step."""
        logits = self._process_logits(group.last_logits.float(), group)
        next_tokens = select_next_tokens(logits, group.opts)  # (B_active,)

        toks = next_tokens.cpu().tolist()
        finished_rows: List[int] = []
        reasons: Dict[int, str] = {}
        for row, tok in enumerate(toks):
            if self._is_eos(int(tok)):
                finished_rows.append(row)
                reasons[row] = "stop"
            else:
                group.tokens[row].append(int(tok))
                if len(group.tokens[row]) >= group.max_new[row]:
                    finished_rows.append(row)
                    reasons[row] = "length"

        outputs = [self._finalize_row(group, row, reasons[row]) for row in finished_rows]
        if finished_rows:
            keep = [r for r in range(len(group.requests)) if r not in finished_rows]
            group.keep_rows(keep)
            if keep:
                keep_idx = torch.tensor(keep, dtype=torch.long, device=next_tokens.device)
                group.state = self._decoder().select(group.state, keep_idx)
                next_tokens = next_tokens.index_select(0, keep_idx)

        if group.requests:
            with torch.no_grad():
                group.last_logits, group.state = self._decoder().step(next_tokens, group.state)
        return outputs

    # ------------------------------------------------------------------
    # Beam advance
    # ------------------------------------------------------------------

    def _advance_beam_group(self, group: ArBeamGroup) -> List[RequestOutput]:
        """One batched beam step over every live slot of every live request.

        Structure mirrors :meth:`_advance_group` — score, reorder, retire, step —
        with three differences: selection is a per-request top-``k`` over the
        flattened ``(k, V)`` grid rather than a per-row argmax; a slot emitting
        EOS is *set aside* as a completed candidate instead of retiring its
        request; and a request retires only once its whole beam is done.

        The ``step`` at the end is the part that is easy to leave out and
        impossible to miss once it is: without feeding the chosen token back into
        the decoder, ``last_logits`` never changes and every slot re-picks its
        first token forever.
        """
        logits = self._process_logits(group.last_logits.float(), group)
        log_probs = torch.log_softmax(logits, dim=-1)
        new_scores, parent, token = topk_step(log_probs, group.scores, group.beam)

        # Reorder the decoder state so flat row ``b * k + j`` carries slot j's
        # parent state.  One ``select`` with (possibly repeated) parent indices.
        group.state = self._decoder().select(group.state, global_parent_rows(parent, group.beam))
        chosen = token.reshape(-1)  # (B * k,) token to feed each new slot

        # Host-side bookkeeping: extend each new slot from its parent, and move
        # EOS slots into the completed pool.  ``k`` list copies per step is
        # nothing next to a decoder forward (see incremental_beam's docstring).
        parent_h = parent.tolist()
        token_h = token.tolist()
        scores_h = new_scores.tolist()
        group.steps += 1
        for b in range(group.batch):
            src = group.tokens[b]
            fresh: List[List[int]] = []
            for j in range(group.beam):
                p, tok, sc = parent_h[b][j], int(token_h[b][j]), scores_h[b][j]
                if not is_finite_score(sc):
                    # Every live slot was already expanded, so this entry came
                    # from a dead parent — leave it dead, never resurrect it.
                    fresh.append([])
                    new_scores[b, j] = DEAD_SCORE
                elif self._is_eos(tok):
                    # Complete: banked as a candidate, and out of the running so
                    # it stops consuming a slot.  EOS itself is not part of the
                    # hypothesis.
                    group.finished[b].append((sc, list(src[p])))
                    fresh.append([])
                    new_scores[b, j] = DEAD_SCORE
                else:
                    fresh.append(src[p] + [tok])
            group.tokens[b] = fresh
        group.scores = new_scores

        outputs, keep = self._retire_finished_requests(group)
        if group.requests:
            # Restrict the fed tokens to the surviving slots, in the same flat
            # order ``_retire_finished_requests`` left the state in.
            if keep is not None:
                chosen = chosen.index_select(0, keep)
            with torch.no_grad():
                group.last_logits, group.state = self._decoder().step(chosen, group.state)
        return outputs

    def _retire_finished_requests(
        self, group: ArBeamGroup
    ) -> Tuple[List[RequestOutput], Optional[torch.Tensor]]:
        """Emit + drop requests whose beam is exhausted or capped.

        Returns the outputs plus, when anything was dropped, the flat slot rows
        that survived — the caller needs them to line its per-slot tensors up
        with the shrunken state.
        """
        done: List[int] = []
        reasons: Dict[int, str] = {}
        live_h = group.scores.tolist()
        for b in range(group.batch):
            live = sum(1 for sc in live_h[b] if is_finite_score(sc))
            if len(group.finished[b]) >= group.beam or live == 0:
                done.append(b)
                reasons[b] = "stop"
            elif group.steps >= group.max_new[b]:
                done.append(b)
                # Only "length" when nothing completed — a request whose best
                # candidate ended on EOS did stop, it just also ran long.
                reasons[b] = "stop" if group.finished[b] else "length"

        outputs = [self._finalize_beam_request(group, b, reasons[b]) for b in done]
        if not done:
            return outputs, None

        keep = [b for b in range(group.batch) if b not in set(done)]
        slot_rows: Optional[torch.Tensor] = None
        if keep:
            device = group.scores.device
            slot_rows = torch.tensor(
                [b * group.beam + j for b in keep for j in range(group.beam)],
                dtype=torch.long,
                device=device,
            )
            group.state = self._decoder().select(group.state, slot_rows)
            group.scores = group.scores.index_select(
                0, torch.tensor(keep, dtype=torch.long, device=device)
            )
        group.requests = [group.requests[b] for b in keep]
        group.tokens = [group.tokens[b] for b in keep]
        group.finished = [group.finished[b] for b in keep]
        group.max_new = [group.max_new[b] for b in keep]
        group.opts = [group.opts[b] for b in keep]
        return outputs, slot_rows

    def _finalize_beam_request(self, group: ArBeamGroup, index: int, reason: str) -> RequestOutput:
        rows, scores = group.ranked(index, self._length_penalty)
        best = rows[0] if rows else []
        return RequestOutput(
            request_id=group.requests[index].request_id,
            text=self._detok.detokenize(best),
            tokens=rows or [[]],
            scores=scores or None,
            finished=True,
            finish_reason=reason,
        )

    def _beam_partials(self, advanced: List[ArBeamGroup]) -> List[RequestOutput]:
        """Interim outputs from each request's current best hypothesis.

        The best hypothesis can be *revised* between steps (a different slot
        wins), so unlike greedy this is not an append-only stream — the full
        transcript is re-decoded rather than extended, and the client replaces
        rather than appends (which the wire contract already requires).
        """
        outputs: List[RequestOutput] = []
        for group in advanced:
            for b, req in enumerate(group.requests):
                best, _score, _done = group.best(b, self._length_penalty)
                if best:
                    outputs.append(
                        RequestOutput(
                            request_id=req.request_id,
                            text=self._detok.detokenize(best),
                            tokens=[list(best)],
                            finished=False,
                        )
                    )
        return outputs

    def _row_text(self, group: ArGroup, row: int) -> str:
        """Full transcript for a row, decoding only the ids added since last call.

        Keeps ``RequestOutput.text`` the complete transcript (the wire contract
        is unchanged — clients replace, not append) while the *work* is
        incremental.  ``state["text"]`` is what the tokenizer axis maintains.
        """
        state = group.detok_state[row]
        tokens = group.tokens[row]
        seen = len(state.get("ids", ()))
        if seen < len(tokens):
            self._detok.detokenize_incremental(tokens[seen:], state)
        return state.get("text", "")

    def _partials(self, advanced: List[ArGroup]) -> List[RequestOutput]:
        """One ``finished=False`` output per still-active request that moved."""
        outputs: List[RequestOutput] = []
        for group in advanced:
            for row, req in enumerate(group.requests):
                tokens = group.tokens[row]
                if tokens:
                    outputs.append(
                        RequestOutput(
                            request_id=req.request_id,
                            text=self._row_text(group, row),
                            tokens=[list(tokens)],
                            finished=False,
                        )
                    )
        return outputs

    def _finalize_row(self, group: ArGroup, row: int, reason: str) -> RequestOutput:
        tokens = group.tokens[row]
        out = RequestOutput(
            request_id=group.requests[row].request_id,
            text=self._row_text(group, row),
            tokens=[tokens],
            finished=True,
            finish_reason=reason,
        )
        enc = group.align_enc[row] if row < len(group.align_enc) else None
        if enc is not None and tokens:
            # The row's decoder state is about to be dropped, so this is the
            # last moment its encoder output is still around to align against.
            align = self._align_row(group.requests[row], enc, list(tokens))
            if align:
                self.attach_alignment(out, align)
        return out

    def _align_row(
        self, request: Request, enc_out: torch.Tensor, tokens: List[int]
    ) -> Optional[List["TokenAlignment"]]:
        """Per-token spans for one finished row, or ``None``.

        Default: nothing.  A family that populates ``ArGroup.align_enc`` — i.e.
        one that declared ``word_timing_modes`` — implements this.
        """
        del request, enc_out, tokens
        return None

    # ------------------------------------------------------------------
    # Session cleanup (abort path)
    # ------------------------------------------------------------------

    def free_session(self, request: Request) -> None:
        for group in self._groups:
            if request not in group.requests:
                continue
            row = group.requests.index(request)
            keep = [r for r in range(len(group.requests)) if r != row]
            if isinstance(group, ArBeamGroup):
                # A beam request owns ``beam`` consecutive decoder rows.
                device = group.last_logits.device
                if keep:
                    slot_rows = torch.tensor(
                        [b * group.beam + j for b in keep for j in range(group.beam)],
                        dtype=torch.long,
                        device=device,
                    )
                    group.state = self._decoder().select(group.state, slot_rows)
                    group.last_logits = group.last_logits.index_select(0, slot_rows)
                    group.scores = group.scores.index_select(
                        0, torch.tensor(keep, dtype=torch.long, device=device)
                    )
                group.requests = [group.requests[b] for b in keep]
                group.tokens = [group.tokens[b] for b in keep]
                group.finished = [group.finished[b] for b in keep]
                group.max_new = [group.max_new[b] for b in keep]
                group.opts = [group.opts[b] for b in keep]
                break
            group.keep_rows(keep)
            if keep:
                keep_idx = torch.tensor(keep, dtype=torch.long, device=group.last_logits.device)
                group.state = self._decoder().select(group.state, keep_idx)
                group.last_logits = group.last_logits.index_select(0, keep_idx)
            break
        self._groups = [g for g in self._groups if g.requests]

    # ------------------------------------------------------------------
    # One-shot / streaming surfaces (not applicable to AR families)
    # ------------------------------------------------------------------

    def decode_offline(
        self,
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
        requests: Optional[List[Request]] = None,
    ) -> List[RequestOutput]:
        raise NotImplementedError(
            f"{self.decode_type} is an incremental strategy; the executor drives it "
            "via begin_offline/advance, not one-shot decode_offline"
        )

    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        raise NotImplementedError(
            f"{self.decode_type} decoding is offline-only (not genuinely streamable)"
        )

    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        raise NotImplementedError(
            f"{self.decode_type} decoding is offline-only (not genuinely streamable)"
        )

    def finalize(self, request: Request) -> RequestOutput:
        raise NotImplementedError(
            f"{self.decode_type} decoding is offline-only (not genuinely streamable)"
        )
