# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer (RNNT) frame-synchronous greedy decode strategy.

Consumes raw encoder hidden states (``consumes="hidden"``) and drives the
model's label predictor (``model.decoder``) + ``model.joiner`` directly. For
each encoder frame the joiner combines the frame with the current prediction;
``argmax`` either emits a label (fold it into the predictor state, stay on the
frame, bounded by ``max_sym_per_frame``) or is blank (advance to the next frame).
The predictor projection is recomputed only on steps where at least one row
emitted; the encoder is projected once up front (the icefall greedy fast path).

**The predictor state is opaque here.**  The loop calls
``decoder.predict`` / ``advance`` / ``stack_states`` / ``unstack_states``
(:class:`~oasr.models.decoders.base.TransducerPredictor`) rather than shifting a
label window itself, which is what lets one loop serve both a stateless
convolutional predictor (icefall: state == the last ``k`` labels, recomputable)
and a recurrent one (NeMo's 2-layer LSTM: state == ``(output, h, c)``, *not*
recomputable from a bounded window).  Inlining the shift, as this file used to,
made the second impossible to express.

One vectorized greedy core (:meth:`_greedy_loop`) serves both paths:

* **offline** — fresh predictor state per micro-batch row, loop to the row's
  encoder length;
* **streaming** — per-request :class:`_Session` (predictor state + its
  projection + accumulated hypothesis) threaded across chunks; each tick decodes
  the new chunk's frames in a batch grouped by chunk length.

The per-emit row loop is fully vectorized: the predictor folds the batch's
emitted labels in under a row mask, and emitted tokens are collected as per-step
snapshots read back in one sync at loop end.  Loop *control* costs one host sync
per iteration (the predictor-recompute gate) plus one per
``_TERMINATION_CHECK_STRIDE`` iterations; see that constant for why the second
one is amortized and the first is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Sequence, Tuple, cast

import torch

from ..request import Request, RequestOutput
from .alignment import wants_word_timings
from .base import DecodeStrategy, register_decode_strategy, wants_speech_activity
from .options import option
from .transducer_beam import (
    BeamState,
    beam_search_chunk,
    init_beam_state,
    select_rows,
    stack_states,
)

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel
    from oasr.models.decoders.base import Joiner, TransducerPredictor

    from ..config import EngineConfig
    from ..predictor_graph import PredictorStepGraphCache
    from .detokenize import Detokenizer


#: Greedy iterations between termination checks. Batching checks reduces host
#: synchronization; 16 bounds inert overshoot. Keep the emission check because
#: it avoids unnecessary predictor forwards on blank-dominated inputs.
_TERMINATION_CHECK_STRIDE = 16


def _unzip_marks(marks: Sequence[Tuple[int, float]]) -> Tuple[List[int], List[float]]:
    """Split ``(frame, posterior)`` pairs for the shared alignment pass."""
    return [f for f, _ in marks], [p for _, p in marks]


@dataclass
class _Session:
    """Per-stream decode state carried across chunks.

    Greedy uses ``state`` / ``dec_proj`` / ``hyp``; beam search uses ``beam``
    (a ``(1, k, ...)`` :class:`BeamState`) and refreshes ``hyp`` from its best
    hypothesis after each chunk, so the partial/final emission path and the
    incremental detokenizer are shared between the two.
    """

    #: Opaque per-stream predictor state (``B == 1``): a label window for the
    #: stateless predictor, an ``(output, h, c)`` tuple for a recurrent one.
    state: Any
    dec_proj: torch.Tensor  # (1, J) predictor projection for that state
    hyp: List[int] = field(default_factory=list)
    #: Beam-search state, ``None`` for greedy.
    beam: Optional["BeamState"] = None
    #: Per-hypothesis token lists + scores from the last beam chunk (n-best).
    nbest: Optional[Tuple[List[List[int]], List[float]]] = None
    steps: int = 0  # decoded chunks (drives the partial-emit cadence)
    #: Encoder frames consumed by previous chunks, so a chunk-local emission
    #: frame becomes an utterance-absolute one.
    frames: int = 0
    #: Accumulated ``(frame, posterior)`` emission marks, in ``hyp`` order.
    #: Only populated for a stream that asked for word timings — the greedy
    #: loop's tracking is opt-in per launch.
    marks: List[Tuple[int, float]] = field(default_factory=list)
    #: Incremental-detokenization state.  Greedy transducer decode only
    #: appends to ``hyp``, so a partial decodes just the new ids rather than
    #: re-rendering the whole transcript every chunk.
    detok: Dict[str, Any] = field(default_factory=dict)

    def text(self, detok) -> str:
        """Full transcript, decoding only what was appended since last call.

        Incremental decoding assumes the hypothesis only grows.  Greedy
        guarantees that; **beam search does not** — a later frame can promote a
        different beam entry, rewriting the prefix.  So verify the recorded ids
        are still a prefix of ``hyp`` and re-decode from scratch when they are
        not.  Silently feeding ``hyp[seen:]`` after a revision would splice the
        tail of the new hypothesis onto the text of the old one.
        """
        seen_ids = self.detok.get("ids", [])
        seen = len(seen_ids)
        if seen > len(self.hyp) or seen_ids != self.hyp[:seen]:
            self.detok.clear()
            seen = 0
        if seen < len(self.hyp):
            detok.detokenize_incremental(self.hyp[seen:], self.detok)
        return self.detok.get("text", "")


@dataclass(frozen=True)
class TransducerOptions:
    """Options for the frame-synchronous transducer greedy decode."""

    max_sym_per_frame: int = option(
        10,
        legacy="transducer_max_sym_per_frame",
        doc="Cap on tokens emitted at one encoder frame before advancing.",
    )
    beam_size: int = option(
        1,
        doc=(
            "1 (default) = greedy.  >1 runs icefall-style modified beam search "
            "(at most one symbol per frame), which is also what makes "
            "DecodingOptions.n_best return real alternatives for this family."
        ),
    )

    def __post_init__(self) -> None:
        if self.max_sym_per_frame < 1:
            raise ValueError(f"max_sym_per_frame must be >= 1, got {self.max_sym_per_frame!r}")
        if self.beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {self.beam_size!r}")


@register_decode_strategy("transducer")
class TransducerDecodeStrategy(DecodeStrategy):
    """Greedy RNNT decoding over encoder hidden states (offline + streaming)."""

    decode_type: ClassVar[str] = "transducer"
    consumes: ClassVar[str] = "hidden"
    options_cls: ClassVar[type] = TransducerOptions

    speech_activity_kind: ClassVar[str] = "transducer_blank"

    @property
    def asr_speech_activity_modes(self) -> Tuple[str, ...]:
        """Greedy only, for the same reason word timings are.

        The signal *is* the emission trace, and beam search's device-side
        hypothesis buffer carries labels rather than frames — so under beam there
        is nothing to read, and saying so is better than reporting boundaries
        derived from something else.
        """
        return () if self._beam > 1 else ("offline", "streaming")

    @property
    def word_timing_modes(self) -> Tuple[str, ...]:
        """Both modes under greedy; **neither** under beam search.

        A transducer's emissions are frame-indexed by construction, so the
        greedy loop already knows *when* at the moment it decides *what*.  The
        beam keeps its hypotheses in a device-side ``(B, k, cap)`` buffer that
        carries labels and no frames, and a later frame can promote a different
        entry — so the emission marks the greedy loop records have no
        counterpart there.  Declaring from what *this configuration* can do,
        rather than from what the class implements, is what keeps an engine from
        admitting the request and then answering without the field.
        """
        return () if self._beam > 1 else ("offline", "streaming")

    def __init__(
        self,
        config: "EngineConfig",
        detok: "Detokenizer",
        model: "BaseAsrModel" = None,
    ) -> None:
        super().__init__(config, detok, model)
        # Cap on non-blank emissions per frame (safety against degenerate loops;
        # the same cap is applied uniformly so results are deterministic).
        self._max_sym = int(self.options.max_sym_per_frame)
        #: >1 selects modified beam search over the greedy loop.
        self._beam = int(self.options.beam_size)
        # Interim-partial cadence (shared engine knob): emit a partial every
        # N-th chunk; <= 0 disables partials (final transcript only).
        self._partial_interval = int(getattr(config, "partial_decode_interval", 1))
        # ``None`` marks a created-but-uninitialized session (state materializes
        # on the first chunk, when the encoder output's device is known).
        self._sessions: Dict[str, Optional[_Session]] = {}
        # The predictor step is nine launches for ~12-39 us of GPU work, and a
        # real Nemotron decode spends 34% of this loop on it.  Capturing it turns
        # a host-bound step into a graph replay; see oasr/engine/predictor_graph.py
        # for what that costs and the three hazards it owns.  Lazily built so a
        # CPU-only or eager-only run never touches CUDA graphs.
        self._pred_graphs: Optional["PredictorStepGraphCache"] = None
        self._pred_graphs_enabled = bool(
            getattr(config, "use_cuda_graphs", False)
            and getattr(config, "use_transducer_cuda_graphs", False)
            and self._beam <= 1
        )
        if self._beam > 1 and model is not None:
            # Beam search keeps every hypothesis's state in one ``(B, k, ctx)``
            # buffer and reorders it onto the new parents with a ``gather``
            # (``transducer_beam.py``), which only expresses a label window.  A
            # recurrent predictor would need the same reordering over its hidden
            # and cell tensors — real work, not a wiring change — so refuse at
            # engine construction rather than at the first decode.
            if not getattr(model.decoder, "label_window_state", False):
                raise ValueError(
                    f"beam_size={self._beam} is not supported for "
                    f"{type(model.decoder).__name__}: modified beam search reorders a "
                    "label-window state across the beam, and this predictor carries "
                    "recurrent state instead. Use beam_size=1 (greedy)."
                )

    # ------------------------------------------------------------------
    # Vectorized greedy core (shared by offline + streaming)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _greedy_loop(
        self,
        enc_out: torch.Tensor,  # (B, T, D) encoder hidden
        lengths: torch.Tensor,  # (B,) valid frames per row
        state: Any,  # opaque batched predictor state (B rows)
        dec_proj: torch.Tensor,  # (B, J) predictor projections for that state
        track: bool = False,  # also record each emission's frame + posterior
    ) -> Tuple[List[List[int]], List[List[Tuple[int, float]]], Any, torch.Tensor]:
        """Run batched greedy over ``enc_out``.

        Returns the newly emitted tokens per row, the matching
        ``(frame, posterior)`` pairs when ``track`` is set (``[]`` otherwise),
        and the updated ``(state, dec_proj)`` predictor state.

        Tracking is opt-in because it costs a ``logsumexp`` over the vocabulary
        and two extra ``(B,)`` snapshots per emitting step.  The *frames* are
        free — this loop already knows ``t`` at the moment it emits, which is
        the whole reason a transducer can time its output in either mode while
        a CTC beam needs a second pass.
        """
        joiner, decoder = self._surface()
        blank = int(cast(int, self._model.blank_id))
        max_sym = self._max_sym

        device = enc_out.device
        B, T, _ = enc_out.shape
        lengths = lengths.to(device=device, dtype=torch.long)

        # Project the encoder output once; per step only the predictor is re-run.
        enc_proj = joiner.encoder_proj(enc_out)  # (B, T, J)

        t = torch.zeros(B, dtype=torch.long, device=device)
        sym = torch.zeros(B, dtype=torch.long, device=device)
        rows = torch.arange(B, device=device)
        no_emit = torch.full((B,), -1, dtype=torch.long, device=device)
        zero_sym = torch.zeros_like(sym)
        emitted: List[torch.Tensor] = []  # per-step (B,) token snapshots, -1 = no emit
        emit_frame: List[torch.Tensor] = []  # per-step (B,) frame index at emission
        emit_prob: List[torch.Tensor] = []  # per-step (B,) posterior of that token

        graphs = self._predictor_graphs()
        graphed = False

        max_steps = int(T) * (max_sym + 1) + B + 1  # termination safety bound
        done = 0
        while done < max_steps:
            # Termination is checked once per block rather than per iteration (see
            # _TERMINATION_CHECK_STRIDE).  Overshooting is inert: once every row
            # has t >= its length, ``active`` is all-false, so ``emit`` and
            # ``advance`` are too and nothing mutates — the extra iterations
            # cost one joiner call each and change no state.
            for _ in range(min(_TERMINATION_CHECK_STRIDE, max_steps - done)):
                done += 1
                active = t < lengths

                enc_t = enc_proj[rows, t.clamp(max=T - 1)]  # (B, J)
                logits = joiner(enc_t, dec_proj, project_input=False)  # (B, V)
                tok = logits.argmax(dim=-1)  # (B,)

                is_blank = (tok == blank) | (sym >= max_sym)
                emit = active & ~is_blank
                advance = active & is_blank

                # The host sync is worth paying only when word timings are
                # wanted.  Dropping it lets the loop run entirely async, and the
                # predictor step is masked by ``emit`` either way, so a
                # no-emission iteration mutates nothing -- it just costs one more
                # ``(B,)`` snapshot in ``emitted``, ~1.1 us per entry at the one
                # readback after the loop, against a sync that makes the host wait
                # out whatever the iteration queued.  Measured on nemotron
                # (transcripts bit-identical in every case):
                #
                #     no word timings   offline b8 1.044x  b32 1.055x  streaming 1.111x
                #                       132 s utterances:  offline 1.057x  streaming 1.168x
                #     word timings      offline b8 0.995x  b32 1.028x  streaming 1.085x
                #
                # The word-timing snapshot is what turns it: it adds a ``t.clone()``,
                # a gather, a vocab-wide ``logsumexp`` and an ``exp`` -- four eager
                # launches -- to *every* iteration rather than to the emitting ones,
                # and at batch 8 that is more than the sync it saves.  So keep the
                # branch exactly there, and only there.  ``emitted``, ``emit_frame``
                # and ``emit_prob`` are read back index-aligned, which is also why
                # this has to be one branch over all three rather than a cheap
                # unconditional token append plus a guarded timing append.
                if not track or bool(emit.any()):
                    # Fold the emitted label into each emitting row's state; rows
                    # that didn't emit keep theirs, so the batched projection that
                    # follows reproduces their previous value exactly.
                    stepped = graphs.step(state, tok, emit) if graphs is not None else None
                    if stepped is not None:
                        # Both are graph memory, live until the next replay.  The
                        # loop only reads them before then; the one copy happens
                        # after the loop, where the state escapes.
                        state, dec_proj = stepped
                        graphed = True
                    else:
                        state = decoder.advance(state, tok, emit)
                        dec_proj = joiner.decoder_proj(decoder.predict(state))
                    emitted.append(torch.where(emit, tok, no_emit))
                    if track:
                        emit_frame.append(t.clone())
                        # ``logit - logsumexp`` rather than a full softmax: only
                        # the chosen token's posterior is ever read.
                        chosen = logits[rows, tok] - logits.logsumexp(dim=-1)
                        emit_prob.append(chosen.exp())
                    sym = sym + emit.long()

                t = t + advance.long()
                sym = torch.where(advance, zero_sym, sym)

            if not bool((t < lengths).any()):
                break

        marks: List[List[Tuple[int, float]]] = []
        if emitted:
            # One host readback for the whole loop.
            snap = torch.stack(emitted, dim=1).tolist()  # B × S
            hyps = [[tk for tk in row if tk >= 0] for row in snap]
            if track:
                frames = torch.stack(emit_frame, dim=1).tolist()
                probs = torch.stack(emit_prob, dim=1).tolist()
                marks = [
                    [(int(f), float(p)) for tk, f, p in zip(row, fr, pr) if tk >= 0]
                    for row, fr, pr in zip(snap, frames, probs)
                ]
        else:
            hyps = [[] for _ in range(B)]
            if track:
                marks = [[] for _ in range(B)]
        if graphed:
            # ``state`` and ``dec_proj`` are the graph's own buffers and the next
            # replay overwrites them.  The streaming path keeps per-session state
            # across ticks and slices it with ``unstack_states``, which for a
            # recurrent predictor hands back *views* — so without this copy a
            # session would read some later tick's state.  One copy per loop, not
            # per step.
            state = cast("PredictorStepGraphCache", graphs).detach(state)
            dec_proj = dec_proj.clone()
        return hyps, marks, state, dec_proj

    def _predictor_graphs(self) -> Optional["PredictorStepGraphCache"]:
        """The predictor-step graph cache, built on first use.  ``None`` if off."""
        if not self._pred_graphs_enabled:
            return None
        if self._pred_graphs is None:
            from oasr.engine.predictor_graph import PredictorStepGraphCache

            joiner, decoder = self._surface()
            self._pred_graphs = PredictorStepGraphCache(decoder, joiner)
        return self._pred_graphs

    def _surface(self) -> Tuple["Joiner", "TransducerPredictor"]:
        """``(joiner, predictor)`` with their real types.

        ``nn.Module.__getattr__`` types every submodule as ``Tensor | Module``, so
        without this every call through them is an error the type checker cannot
        see past.  The members themselves are guaranteed by
        ``CAPABILITIES["transducer"]``, which the base ``DecodeStrategy``
        constructor already validated.
        """
        return (
            cast("Joiner", self._model.joiner),
            cast("TransducerPredictor", self._model.decoder),
        )

    def _init_state(self, batch_size: int, device: torch.device) -> Tuple[Any, torch.Tensor]:
        joiner, decoder = self._surface()
        state = decoder.init_state(batch_size, device)
        dec_proj = joiner.decoder_proj(decoder.predict(state))  # (B, J)
        return state, dec_proj

    # ------------------------------------------------------------------
    # Offline greedy
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_offline(
        self,
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
        requests: Optional[List[Request]] = None,
    ) -> List[RequestOutput]:
        if self._beam > 1:
            return self._decode_offline_beam(enc_out, enc_lengths)
        B = enc_out.size(0)
        state, dec_proj = self._init_state(B, enc_out.device)
        # Timing is decided for the whole micro-batch, not per row: the greedy
        # core is one batched loop, so tracking is either on for the launch or
        # off.  The facade only passes ``requests`` when some row asked.
        track = requests is not None
        hyps, marks, _, _ = self._greedy_loop(enc_out, enc_lengths, state, dec_proj, track=track)
        outputs = [
            RequestOutput(
                request_id="",
                text=self._detok.detokenize(hyps[b]),
                tokens=[hyps[b]],
                finished=True,
            )
            for b in range(B)
        ]
        if track:
            for b, (req, out) in enumerate(zip(requests or [], outputs)):
                if wants_word_timings(req):
                    frames, probs = _unzip_marks(marks[b])
                    self.attach_emission_alignment(out, hyps[b], frames, probs)
            self._attach_emission_activity(outputs, marks, enc_out, enc_lengths, requests)
        return outputs

    def _attach_emission_activity(
        self,
        outputs: List[RequestOutput],
        marks: List[List[Tuple[int, float]]],
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
        requests: Optional[List[Request]],
    ) -> None:
        """Speech activity from the frames the greedy loop already recorded.

        The transducer has no per-frame posterior to read the way CTC does — the
        joiner is only evaluated at the steps the loop takes — so the signal is
        the emission trace, which is recorded for word timings anyway.  Building
        the dense indicator on the host is per-token Python, which is exactly the
        cost this codebase avoids on the decode path; it is acceptable *here*
        only because it runs once per finished request and only for the rows that
        asked, never per step.
        """
        if requests is None or not any(wants_speech_activity(r) for r in requests):
            return
        detector = self._speech_detector()
        if detector is None:
            return
        rows = len(outputs)
        frames = int(enc_out.size(1))
        indicator = torch.zeros(rows, frames, dtype=torch.float32)
        for b in range(min(rows, len(marks))):
            if requests[b] is not None and not wants_speech_activity(requests[b]):
                continue
            for frame, _posterior in marks[b]:
                if 0 <= frame < frames:
                    indicator[b, frame] = 1.0
        self.attach_asr_speech_activity(outputs, indicator, enc_lengths.cpu(), requests)

    @torch.no_grad()
    def _decode_offline_beam(
        self, enc_out: torch.Tensor, enc_lengths: torch.Tensor
    ) -> List[RequestOutput]:
        """Modified beam search over the whole utterance.

        Emits **all** ``beam_size`` hypotheses in ``tokens`` / ``scores``, best
        first, so ``DecodingOptions.n_best`` finally means something for this
        family (``OutputProcessor.fill_nbest_texts`` then detokenizes and trims
        to what the request asked for).
        """
        B, T = enc_out.size(0), enc_out.size(1)
        state = init_beam_state(self._model.decoder, B, self._beam, enc_out.device, capacity=T)
        state = beam_search_chunk(self._model, enc_out, enc_lengths, state)
        rows, scores = state.hypotheses()
        return [
            RequestOutput(
                request_id="",
                text=self._detok.detokenize(rows[b][0]),
                tokens=rows[b],
                scores=scores[b],
                finished=True,
            )
            for b in range(B)
        ]

    # ------------------------------------------------------------------
    # Streaming greedy (per-request predictor state across chunks)
    # ------------------------------------------------------------------

    def create_session(self, request: Request) -> None:
        """Register the stream; predictor state initializes lazily on the first
        chunk (the device/dtype come from the encoder output)."""
        self._sessions.setdefault(request.request_id, None)  # type: ignore[arg-type]

    def free_session(self, request: Request) -> None:
        self._sessions.pop(request.request_id, None)

    def _session(self, request_id: str, device: torch.device) -> _Session:
        s = self._sessions.get(request_id)
        if s is None:
            state, dec_proj = self._init_state(1, device)
            s = _Session(state=state, dec_proj=dec_proj)
            self._sessions[request_id] = s
        return s

    @torch.no_grad()
    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        ready = [r for r in requests if r.request_id in enc_out_map]
        if not ready:
            return []

        # Group by chunk length so each group runs one batched greedy loop.
        groups: Dict[int, List[Request]] = {}
        for req in ready:
            groups.setdefault(int(enc_out_map[req.request_id].size(1)), []).append(req)

        outputs: List[RequestOutput] = []
        for T_chunk, group in groups.items():
            enc = torch.cat([enc_out_map[r.request_id] for r in group], dim=0)  # (B, T, D)
            device = enc.device
            sessions = [self._session(r.request_id, device) for r in group]
            lengths = torch.full((len(group),), T_chunk, dtype=torch.long, device=device)

            if self._beam > 1:
                self._advance_beam(group, sessions, enc, lengths)
            else:
                # One launch serves the group, so tracking is on whenever any
                # member asked; the sessions that did not simply discard it.
                self._advance_greedy(
                    group,
                    sessions,
                    enc,
                    lengths,
                    track=any(wants_word_timings(r) or wants_speech_activity(r) for r in group),
                )

            for req, s in zip(group, sessions):
                s.steps += 1
                if self._partial_interval > 0 and s.steps % self._partial_interval == 0:
                    outputs.append(
                        RequestOutput(
                            request_id=req.request_id,
                            text=s.text(self._detok),
                            tokens=[list(s.hyp)],
                            finished=False,
                        )
                    )
        return outputs

    def _advance_greedy(self, group, sessions, enc, lengths, track: bool = False) -> None:
        """One batched greedy loop over the group's chunk; append per session.

        The cohort of ready streams changes every tick, so the per-stream states
        are stacked here and split back afterwards — through the predictor, which
        is the only thing that knows the state's shape.

        Emission frames are chunk-local, so each session rebases them by the
        frames its earlier chunks consumed before appending.
        """
        _joiner, decoder = self._surface()
        state = decoder.stack_states([s.state for s in sessions])
        dec_proj = torch.cat([s.dec_proj for s in sessions], dim=0)
        new_hyps, marks, state, dec_proj = self._greedy_loop(
            enc, lengths, state, dec_proj, track=track
        )
        chunk_frames = int(enc.size(1))
        for b, (s, row_state) in enumerate(zip(sessions, decoder.unstack_states(state))):
            s.state = row_state
            s.dec_proj = dec_proj[b : b + 1]
            s.hyp.extend(new_hyps[b])
            if track:
                s.marks.extend((f + s.frames, p) for f, p in marks[b])
            s.frames += chunk_frames

    def _advance_beam(self, group, sessions, enc, lengths) -> None:
        """One batched beam-search pass over the group's chunk.

        The group's membership changes every tick (streams are grouped by chunk
        length), so the per-stream ``(1, k, ...)`` states are stacked here and
        split back afterwards — the same regrouping the greedy path does for its
        label window, just over four tensors instead of two.

        ``hyp`` is **replaced**, not extended: the beam's best entry can change
        as later frames arrive, and appending would splice a revised hypothesis
        onto the stale prefix.  ``_Session.text`` detects that and re-decodes.
        """
        state = stack_states(
            [
                (
                    s.beam
                    if s.beam is not None
                    else init_beam_state(self._model.decoder, 1, self._beam, enc.device)
                )
                for s in sessions
            ]
        )
        state = beam_search_chunk(self._model, enc, lengths, state)
        rows, scores = state.hypotheses()
        for b, s in enumerate(sessions):
            s.beam = select_rows(state, torch.tensor([b], device=enc.device))
            s.nbest = (rows[b], scores[b])
            s.hyp = list(rows[b][0])

    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        outs = self.decode_streaming_batch([request], {request.request_id: enc_out})
        if outs:
            return outs[0]
        # Partials disabled (partial_decode_interval <= 0): state advanced, no emit.
        s = self._sessions.get(request.request_id)
        hyp = list(s.hyp) if s is not None else []
        return RequestOutput(
            request_id=request.request_id,
            text=s.text(self._detok) if s is not None else "",
            tokens=[hyp],
            finished=False,
        )

    def finalize(self, request: Request) -> RequestOutput:
        """Final transcript from the accumulated session hypothesis.

        The session itself is released by :meth:`free_session` (the executor
        calls it right after finalize).
        """
        s: Optional[_Session] = self._sessions.get(request.request_id)
        hyp = list(s.hyp) if s is not None else []
        # Beam search carries real alternatives; greedy has exactly one row.
        if s is not None and s.nbest is not None:
            rows, scores = s.nbest
            return RequestOutput(
                request_id=request.request_id,
                text=s.text(self._detok),
                tokens=[list(r) for r in rows],
                scores=list(scores),
                finished=True,
            )
        out = RequestOutput(
            request_id=request.request_id,
            text=s.text(self._detok) if s is not None else "",
            tokens=[hyp],
            finished=True,
        )
        if s is not None and s.marks and wants_word_timings(request):
            frames, probs = _unzip_marks(s.marks)
            self.attach_emission_alignment(out, hyp, frames, probs)
        return out
