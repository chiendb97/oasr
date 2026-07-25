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
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

import torch

from ..generation import select_next_tokens
from ..request import DecodingOptions, Request, RequestOutput
from .base import DecodeStrategy

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

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = [[] for _ in self.requests]

    def keep_rows(self, keep: List[int]) -> None:
        """Drop every row not in ``keep`` from the host-side bookkeeping."""
        self.requests = [self.requests[r] for r in keep]
        self.tokens = [self.tokens[r] for r in keep]
        self.max_new = [self.max_new[r] for r in keep]
        self.opts = [self.opts[r] for r in keep]


@dataclass
class Prefill:
    """What a family's prefill produces for a fresh micro-batch."""

    state: Dict[str, Any]
    #: ``(B, V)`` logits for the first token selection.
    logits: torch.Tensor
    #: Per-row generation cap, already clamped to the model's position capacity.
    max_new: List[int]


class IncrementalArStrategy(DecodeStrategy):
    """Base for ``incremental = True`` strategies (AED / speech-LLM).

    Subclasses implement :meth:`_prefill` and :meth:`_is_eos`, optionally
    :meth:`_process_logits`, and set :attr:`emit_partials`.  Everything below the
    hooks is shared and should not be overridden.
    """

    consumes: ClassVar[str] = "hidden"
    incremental: ClassVar[bool] = True
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
            else int(self._config.max_new_tokens)
        )
        return max(1, min(int(requested), capacity))

    # ------------------------------------------------------------------
    # Incremental protocol (shared)
    # ------------------------------------------------------------------

    def begin_offline(
        self,
        requests: List[Request],
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
    ) -> None:
        plan = self._prefill(requests, enc_out, enc_lengths)
        self._groups.append(
            ArGroup(
                requests=list(requests),
                state=plan.state,
                last_logits=plan.logits,
                max_new=list(plan.max_new),
                opts=[getattr(r, "decoding", None) for r in requests],
            )
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
            outputs.extend(self._advance_group(group))
            if group.requests:
                self._groups.append(group)
                if not any(g is group for g in advanced):
                    advanced.append(group)
        if self.emit_partials:
            outputs.extend(self._partials(advanced))
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
                            text=self._detok.detokenize(tokens),
                            tokens=[list(tokens)],
                            finished=False,
                        )
                    )
        return outputs

    def _finalize_row(self, group: ArGroup, row: int, reason: str) -> RequestOutput:
        tokens = group.tokens[row]
        return RequestOutput(
            request_id=group.requests[row].request_id,
            text=self._detok.detokenize(tokens),
            tokens=[tokens],
            finished=True,
            finish_reason=reason,
        )

    # ------------------------------------------------------------------
    # Session cleanup (abort path)
    # ------------------------------------------------------------------

    def free_session(self, request: Request) -> None:
        for group in self._groups:
            if request in group.requests:
                row = group.requests.index(request)
                keep = [r for r in range(len(group.requests)) if r != row]
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
        self, enc_out: torch.Tensor, enc_lengths: torch.Tensor
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
