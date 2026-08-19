# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Attention encoder-decoder (AED) decode strategy — incremental greedy.

Everything about driving a label-synchronous decoder within the engine's tick
budget lives in :class:`~oasr.engine.decode.incremental.IncrementalArStrategy`;
this module is only what makes an AED decoder different from the speech-LLM one:

* the prompt is the checkpoint's fixed SOT sequence, identical for every row, so
  the prefill is a plain expand (no per-row padding);
* the checkpoint's suppression lists are applied before selection;
* EOS is a single token id;
* greedy emits finals only — token-streaming partials land with the LLM family.

The decoder must expose ``prefill`` / ``step`` / ``select`` and the model config
must provide generation-control ids.  Per-request options carry a generation cap
and sampling controls.  Free-text ``prompt`` is ignored because it has no SOT
slot; ``task`` and ``language`` replace their corresponding SOT ids.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

import torch

from ..request import DecodingOptions, Request
from .alignment import TokenAlignment
from .attention_align import resolve_alignment_heads, token_frame_spans
from .base import register_decode_strategy
from .incremental import ArGroup, IncrementalArStrategy, Prefill

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from .detokenize import Detokenizer

logger = logging.getLogger(__name__)


@register_decode_strategy("aed")
class AedDecodeStrategy(IncrementalArStrategy):
    """Incremental greedy AED decoding."""

    decode_type: ClassVar[str] = "aed"
    emit_partials: ClassVar[bool] = False
    #: Both slots exist in the SOT sequence, so both can be set per request.
    selective_options: ClassVar[Tuple[str, ...]] = ("task", "language")

    def __init__(
        self,
        config: "EngineConfig",
        detok: "Detokenizer",
        model: "BaseAsrModel" = None,
    ) -> None:
        super().__init__(config, detok, model)
        # Surface validation lives in ``build_decode_strategy`` via
        # ``oasr.models.interfaces.CAPABILITIES["aed"]`` — one table, one message.
        # Typed ``Any`` deliberately: the members below are a *duck-typed* AED
        # config surface, declared in that table and checked once at
        # construction, so naming a concrete class here would unnecessarily tie
        # the family to one architecture.
        mcfg: Any = model.config
        self._mcfg = mcfg
        self._prompt = list(mcfg.sot_sequence())
        # ``(task, language) -> prompt ids``.  Substituting a slot never changes
        # the prompt's length, so a batch mixing tasks still prefills as one
        # rectangular tensor — the whole reason this is a substitution rather
        # than a re-templating.
        self._prompt_cache: Dict[Tuple[Optional[str], Optional[str]], List[int]] = {
            (None, None): self._prompt
        }
        self._eos = int(mcfg.eos_token_id)
        self._suppress = sorted({int(t) for t in mcfg.suppress_tokens})
        self._begin_suppress = sorted({int(t) for t in mcfg.begin_suppress_tokens})
        # Remaining decoder positions once the prompt is in place.
        self._cap = int(mcfg.max_target_positions) - len(self._prompt) - 1
        # Suppress-id index tensors, materialised per device on first use: an
        # ``index_fill`` with a cached tensor beats re-building an index from a
        # Python list on every step.
        self._suppress_idx: Dict[torch.device, torch.Tensor] = {}
        self._begin_suppress_idx: Dict[torch.device, torch.Tensor] = {}

        # -- word timings ------------------------------------------------
        # Declared from what *this* decoder can do: the alignment is a DTW over
        # cross-attention, so a decoder surface without ``cross_attention`` has
        # no way to produce one, and claiming otherwise would admit the request
        # and then answer it with the field missing.
        decoder = getattr(model, "decoder", None) if model is not None else None
        self._can_align = callable(getattr(decoder, "cross_attention", None))
        self._align_heads: Optional[List[Tuple[int, int]]] = None

    # ------------------------------------------------------------------
    # IncrementalArStrategy hooks
    # ------------------------------------------------------------------

    def _prefill(
        self,
        requests: List[Request],
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
    ) -> Prefill:
        del enc_lengths  # Whisper: fixed-window encoder output, no padding mask
        device = enc_out.device
        rows = [self._prompt_for(r) for r in requests]
        if all(row is self._prompt for row in rows):
            # The common case — no request overrode task or language — stays a
            # single expand rather than a per-row stack.
            prompt = torch.tensor(self._prompt, dtype=torch.long, device=device)
            prompt_ids = prompt.unsqueeze(0).expand(enc_out.size(0), -1).contiguous()
        else:
            prompt_ids = torch.tensor(rows, dtype=torch.long, device=device)
        max_new = [self._row_cap(r, self._cap) for r in requests]
        with torch.no_grad():
            # capacity → the decoder preallocates its KV buffers and every step
            # writes its own row's slot in place.  It is also what makes the
            # state mergeable: a cat-grown cache has no room to hold rows at
            # different offsets.
            logits, state = self._decoder().prefill(
                enc_out,
                prompt_ids,
                capacity=prompt_ids.size(1) + max(max_new) + 1,
                kv_manager=self.kv_manager(),
            )
        return Prefill(state=state, logits=logits, max_new=max_new)

    def _prompt_for(self, request: Request) -> List[int]:
        """This request's SOT sequence, with its task / language slots applied.

        Validated at admission (:meth:`validate_options`), so a lookup failure
        here would be an engine bug rather than a client error.
        """
        opts = request.decoding
        key = (
            getattr(opts, "task", None) if opts else None,
            getattr(opts, "language", None) if opts else None,
        )
        if key == (None, None):
            return self._prompt
        cached = self._prompt_cache.get(key)
        if cached is None:
            cached = list(self._mcfg.sot_sequence(task=key[0], language=key[1]))
            if len(cached) != len(self._prompt):
                # Substitution must be length-preserving; anything else would
                # need per-row prefill padding, which this path does not do.
                raise ValueError(
                    f"decoder prompt for task={key[0]!r} language={key[1]!r} has "
                    f"{len(cached)} tokens, expected {len(self._prompt)}"
                )
            self._prompt_cache[key] = cached
        return cached

    @property
    def word_timing_modes(self) -> Tuple[str, ...]:
        """Offline only — so is the family — and only for greedy decoding with a
        decoder that can report its cross-attention.

        Beam search runs in an :class:`ArBeamGroup`, which retains no encoder
        output to re-forward against and finalizes through its own path; the
        alignment would silently produce nothing.  Declaring it here is what
        turns that into a refusal at admission.
        """
        return ("offline",) if self._can_align and self._beam <= 1 else ()

    def validate_options(
        self, options: Optional[DecodingOptions], *, streaming: bool = False
    ) -> None:
        """Resolve ``task`` / ``language`` against *this checkpoint* now.

        The base class only asks whether the family understands the option at
        all.  Whether this Whisper snapshot knows ``<|yue|>`` is a checkpoint
        question, and answering it at admission is what turns "unknown
        language" into a 400 for that request instead of a raise from inside a
        prefill shared with unrelated requests.
        """
        super().validate_options(options, streaming=streaming)
        if options is None or (options.task is None and options.language is None):
            return
        self._mcfg.sot_sequence(task=options.task, language=options.language)

    def _is_eos(self, token: int) -> bool:
        return token == self._eos

    def _process_logits(self, logits: torch.Tensor, group: ArGroup) -> torch.Tensor:
        """Apply Whisper's suppress lists.

        ``begin_suppress_tokens`` applies only to a row's *first* generated
        position, which the group answers per row (``fresh_rows``) rather than
        per group: a merged group holds rows prefilled this tick next to rows
        forty steps in, and a beam group's ``tokens`` is ``B x k`` *nested* lists
        where ``not tokens[0]`` would be ``False`` from step one.

        Out-of-place (``index_fill`` / ``index_put``) because ``logits`` may alias
        ``group.last_logits``.
        """
        if self._suppress:
            logits = logits.index_fill(
                1, self._ids(self._suppress_idx, self._suppress, logits), -torch.inf
            )
        if not self._begin_suppress:
            return logits
        rows = group.fresh_rows()
        if not rows:
            return logits
        ids = self._ids(self._begin_suppress_idx, self._begin_suppress, logits)
        if len(rows) == logits.size(0):
            return logits.index_fill(1, ids, -torch.inf)
        row_idx = torch.tensor(rows, dtype=torch.long, device=logits.device)
        out = logits.clone()
        out[row_idx.unsqueeze(1), ids.unsqueeze(0)] = -torch.inf
        return out

    @staticmethod
    def _ids(
        cache: Dict[torch.device, torch.Tensor], ids: List[int], like: torch.Tensor
    ) -> torch.Tensor:
        idx = cache.get(like.device)
        if idx is None:
            idx = torch.tensor(ids, dtype=torch.long, device=like.device)
            cache[like.device] = idx
        return idx

    # ------------------------------------------------------------------
    # Word timings (cross-attention DTW)
    # ------------------------------------------------------------------

    def _heads(self) -> List[Tuple[int, int]]:
        """The ``(layer, head)`` set to average, resolved once and cached."""
        if self._align_heads is None:
            heads, declared = resolve_alignment_heads(
                getattr(self._mcfg, "alignment_heads", None),
                int(self._mcfg.decoder_layers),
                int(self._mcfg.decoder_attention_heads),
            )
            if not declared:
                logger.warning(
                    "this checkpoint declares no alignment_heads "
                    "(generation_config.json); word timestamps will average all "
                    "%d heads of the upper %d decoder layers, which is noisier "
                    "and needs more transient memory than the published set",
                    len(heads),
                    int(self._mcfg.decoder_layers) - int(self._mcfg.decoder_layers) // 2,
                )
            self._align_heads = heads
        return self._align_heads

    @torch.no_grad()
    def _align_row(
        self, request: Request, enc_out: torch.Tensor, tokens: List[int]
    ) -> Optional[List[TokenAlignment]]:
        """One teacher-forced pass → per-token spans and posteriors.

        The pass covers ``prompt + tokens`` because the decoder is causal and
        the transcript's attention depends on the prompt preceding it; only the
        transcript's rows are then aligned — the SOT tokens are control, not
        speech, and leaving them in drags the first word's span back to frame 0.
        """
        prompt = self._prompt_for(request)
        ids = torch.tensor([prompt + tokens], dtype=torch.long, device=enc_out.device)
        heads = self._heads()
        weights, logits = self._decoder().cross_attention(
            enc_out, ids, heads, max_frames=self._real_frames(request, enc_out)
        )
        # Rows ``len(prompt) - 1 .. -2`` of the logits predict ``tokens``: step
        # j's distribution sits at the position of the token *before* it.
        lp = torch.log_softmax(logits[0, len(prompt) - 1 : -1].float(), dim=-1)
        want = torch.tensor(tokens, dtype=torch.long, device=lp.device)
        probs = lp.gather(1, want.unsqueeze(1)).squeeze(1).exp().clamp_(0.0, 1.0).tolist()

        spans = token_frame_spans(
            weights[0, :, len(prompt) :].float().cpu().numpy(),
            num_frames=int(weights.size(-1)),
        )
        if spans is None or len(spans) != len(tokens):
            return None
        return [
            TokenAlignment(
                token=int(tok),
                start_frame=float(start),
                end_frame=float(max(end, start + 1)),
                confidence=float(prob),
            )
            for tok, (start, end), prob in zip(tokens, spans, probs)
        ]

    def _real_frames(self, request: Request, enc_out: torch.Tensor) -> int:
        """Encoder frames that are audio rather than the padded 30 s window.

        Whisper's encoder always emits ``max_source_positions`` frames, so a 4 s
        clip is followed by 26 s of silence the DTW would otherwise walk into.
        The count comes from the request's own feature-frame estimate divided by
        the encoder's subsampling — accurate to a frame, which at 20 ms is far
        finer than the boundary it is protecting.
        """
        total = int(enc_out.size(1))
        frames = int(getattr(request, "num_frames", 0) or 0)
        if frames <= 0:
            return total
        rate = int(getattr(self._model.encoder, "subsampling_rate", 1) or 1)
        return max(1, min(total, -(-frames // rate)))
