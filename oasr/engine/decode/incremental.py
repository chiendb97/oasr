# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared machinery for label-synchronous (autoregressive) decode strategies.

The AED and speech-LLM strategies differ in four places — how a micro-batch is
prefilled, how logits are filtered before selection, what counts as EOS, and
whether partial outputs are emitted — and agreed on everything else: group
bookkeeping, the budget loop, per-row finalisation, abort, and the four one-shot /
streaming surfaces they cannot serve.  That "everything else" is here, so a third
AR family is a handful of hooks rather than another 250-line copy.

Each encoded micro-batch becomes an :class:`ArGroup`, and
:meth:`IncrementalArStrategy.advance` round-robins decoder steps across groups
until the tick budget is spent.  Fresh groups are absorbed into an active group
when the decoder can merge per-row state, avoiding duplicate weight-bound
forwards.  Beam groups and decoders without a safe state merge remain separate.

``EngineConfig.decode_admit_window_ms`` can still avoid a merge copy and an
extra encoder/prefill pass, at the cost of first-token latency.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

import torch

from oasr.cache import DecoderKvExhausted, PagedDecoderKv
from oasr.models.base import first_compute_parameter
from oasr.utils.staging import to_device

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
    from oasr.cache.decoder_kv import DecoderKVCacheManager
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from ..generation import StepBudget
    from .detokenize import Detokenizer

logger = logging.getLogger(__name__)

#: Sentinel for "the KV pool has not been decided yet" — ``None`` is a real
#: answer there (dense storage), so it cannot double as "unresolved".
_UNSET = object()


#: Single steps timed per batch-width bucket before deciding whether looping pays.
_TIMED_STEPS = 8

#: Loop only where a single step's host time is at least this fraction of its
#: GPU time (see :meth:`IncrementalArStrategy._loop_pays`).  Measured medians:
#: Whisper tiny 0.43-0.62, Whisper large-v3 0.03-0.14, Qwen2-Audio-7B 0.02.
_HOST_TO_STEP = 0.25


def _rows_bucket(rows: int) -> int:
    """Power-of-two bucket of a batch width: a step's GPU time grows with it."""
    bucket = 1
    while bucket < rows:
        bucket *= 2
    return bucket


def _identity_filter(logits: torch.Tensor, fresh: Optional[torch.Tensor]) -> torch.Tensor:
    del fresh
    return logits


def _held_bytes(obj: Any, seen: Optional[set] = None, depth: int = 0) -> int:
    """Device bytes held by the tensors in a decoder state, each storage once.

    Walks dicts, sequences and dataclass fields only -- the shapes a decoder
    state is built from -- so it cannot wander into a module it merely points at.
    """
    seen = set() if seen is None else seen
    if obj is None or depth > 6:
        return 0
    if isinstance(obj, torch.Tensor):
        storage = obj.untyped_storage()
        if storage.data_ptr() in seen:
            return 0
        seen.add(storage.data_ptr())
        return int(storage.nbytes())
    if isinstance(obj, dict):
        return sum(_held_bytes(v, seen, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_held_bytes(v, seen, depth + 1) for v in obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return sum(_held_bytes(getattr(obj, f.name, None), seen, depth + 1) for f in fields(obj))
    return 0


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
    #: Per-row incremental-detokenization state.  AR generation only ever
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
    #: Per-row ``P(no speech)``, read off the prefill logits for the families
    #: that have such a token.  Captured at prefill because that is the *only*
    #: position it exists at — Whisper predicts it once, before the first
    #: generated token — so a row that finishes forty steps later would have
    #: nothing left to read.
    no_speech: List[Optional[float]] = field(default_factory=list)
    #: The tokens of the loop launched last, still on their way to the host
    #: (``loop_readahead``): read when the next loop is launched, or drained
    #: first by anything that changes the rows.
    pending_read: Optional["_LoopRead"] = None
    #: ``(rows, started, ended, host seconds)`` of the last single step, timed
    #: toward the loop verdict for its width; read once the step has finished.
    timed_step: Optional[Tuple[int, Any, Any, float]] = None

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = [[] for _ in self.requests]
        if not self.detok_state:
            self.detok_state = [{} for _ in self.requests]
        if not self.align_enc:
            self.align_enc = [None] * len(self.requests)
        if not self.no_speech:
            self.no_speech = [None] * len(self.requests)

    def fresh_rows(self) -> List[int]:
        """Rows that have not generated a token yet.

        Read by the families' logit processors (Whisper's
        ``begin_suppress_tokens`` applies only at a row's first generated
        position).  Per row rather than per group because a merged group holds
        rows at different offsets: rows that have been generating for 40 steps
        sit next to rows that were prefilled this tick, and a group-wide
        "is this the first step" answer is wrong for one half either way.
        """
        return [row for row, toks in enumerate(self.tokens) if not toks]

    def keep_rows(self, keep: List[int]) -> None:
        """Drop every row not in ``keep`` from the host-side bookkeeping."""
        self.requests = [self.requests[r] for r in keep]
        self.tokens = [self.tokens[r] for r in keep]
        self.max_new = [self.max_new[r] for r in keep]
        self.opts = [self.opts[r] for r in keep]
        self.detok_state = [self.detok_state[r] for r in keep]
        self.align_enc = [self.align_enc[r] for r in keep]
        self.no_speech = [self.no_speech[r] for r in keep]

    def extend(self, other: "ArGroup", state: Dict[str, Any], logits: torch.Tensor) -> None:
        """Absorb ``other``'s rows, given the already-joined decoder state.

        The decoder owns the join (its state is opaque here); this is the
        host-side half — every per-row list grows by ``other``'s rows, in the same
        order the decoder concatenated them.
        """
        self.requests = self.requests + other.requests
        self.tokens = self.tokens + other.tokens
        self.max_new = self.max_new + other.max_new
        self.opts = self.opts + other.opts
        self.detok_state = self.detok_state + other.detok_state
        self.align_enc = self.align_enc + other.align_enc
        self.no_speech = self.no_speech + other.no_speech
        self.state = state
        self.last_logits = logits


@dataclass(eq=False)
class _LoopRead:
    """One launched loop's tokens on their way to the host."""

    #: The group's rows at launch, in order: the rows of ``tokens``.
    requests: List[Request]
    #: ``(B, steps)`` int64 in pinned memory, valid once ``landed`` has passed.
    tokens: torch.Tensor
    landed: Optional["torch.cuda.Event"]
    steps: int


@dataclass
class Prefill:
    """What a family's prefill produces for a fresh micro-batch."""

    state: Dict[str, Any]
    #: ``(B, V)`` logits for the first token selection.
    logits: torch.Tensor
    #: Per-row generation cap, already clamped to the model's position capacity.
    max_new: List[int]
    #: ``(B, V)`` logits a no-speech probability is read from, when a row asked
    #: for one and the family has the signal; see :meth:`prefill_no_speech`.
    no_speech_logits: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ArOptions:
    """Options shared by every incremental autoregressive family.

    Subclass to add family-specific knobs (see ``LlmOptions``); the base's
    ``max_new_tokens`` stays a single declaration so AED and LLM cannot drift.
    """

    max_new_tokens: int = option(
        448,
        legacy="max_new_tokens",
        doc=(
            "Generation cap, and the ceiling for DecodingOptions.max_new_tokens: a "
            "request may ask for fewer tokens, not more.  The decoder-KV pool and the "
            "admission budget are sized from it, so a row past it would claim KV "
            "nobody budgeted for.  A row that reaches it stops with "
            "finish_reason='length'."
        ),
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
    merge_groups: bool = option(
        True,
        doc=(
            "Absorb a freshly prefilled micro-batch into a group that is already "
            "generating, so late arrivals do not each cost their own step "
            "sequence (an AR step is weight-read bound: N groups cost ~N times "
            "one group of the same total rows).  0 keeps one group per encoded "
            "micro-batch, which is the A/B for attributing a regression to the "
            "merge, and the escape hatch if its transient copy is unaffordable."
        ),
    )
    kv_storage: str = option(
        "auto",
        doc=(
            "Where the decoder's self-attention KV lives.  'dense' gives each "
            "decode group its own capacity buffer; 'paged' allocates one "
            "persistent block pool and gives each row a block table, so a merge "
            "moves no K/V, a prefill allocates nothing, and decoder-step CUDA "
            "graphs become expressible (they need addresses that outlive a "
            "batch).  'auto' (default) pages exactly where that last point pays "
            "for the block-table indirection: a decoder that declares "
            "supports_step_graphs, with step_graphs on.  Paged alone costs a few "
            "percent for the block-table indirection and buys no extra admission "
            "capacity (each row's ceiling is reserved either way), so 'auto' does "
            "not choose it for a family that cannot capture — even though it does "
            "flatten the peak, since a dense select index_selects the whole "
            "capacity buffer.  Beam search is dense regardless: "
            "a paged row's pages are its own, and expanding a beam grid would "
            "alias two slots onto one."
        ),
    )
    kv_block_tokens: int = option(
        16,
        doc="Tokens per page in the paged decoder-KV pool.  Inert when kv_storage='dense'.",
    )
    step_graphs: bool = option(
        True,
        doc=(
            "Replay decoder steps from CUDA graphs, captured per (rows, "
            "block-table width bucket).  Needs paged decoder KV (a graph records "
            "addresses, and only a pool's are stable), which kv_storage='auto' "
            "selects for you, and a decoder that declares supports_step_graphs; "
            "a family without it runs eager and is not an error.  What a graph "
            "removes is the launch half of a weight-read-bound step, so the win "
            "is a small-batch one: 0.65x at one row, and a wash (0.99-1.06x, "
            "paying for the paging rather than the graphs) on a synchronous "
            "burst of eight.  Every batch *ends* small as rows finish, and on "
            "the 200-utterance corpus at max_batch_size 16 the pair is 0.899x "
            "wall with p99 down 47% and transcripts byte-identical.  0 for the "
            "A/B, or if your load really is wide synchronous bursts."
        ),
    )
    step_graph_width_pages: int = option(
        8,
        doc=(
            "Block-table bucket granularity for step graphs, in pages.  Larger "
            "buckets capture fewer shapes and mask more columns per step."
        ),
    )
    step_graph_max_captures: int = option(
        64,
        doc=(
            "Ceiling on distinct (rows, width bucket) shapes captured.  Past it a "
            "step runs eager rather than growing graph memory without bound, "
            "which matters because rows are an *exact* key: a pool whose width "
            "changes every tick can reach many shapes.  Lower it to bound graph "
            "memory without giving up graphs entirely."
        ),
    )

    loop_steps: int = option(
        8,
        doc=(
            "Greedy steps per decoder graph launch where looping pays.  A looping "
            "group runs on the device: the logits filter, the argmax and the "
            "token feedback are in the graph, and the host reads tokens back once "
            "per launch instead of once per token.  It pays where a step is "
            "host-bound (Whisper tiny: +31-37 % backlog, -29 % single-request "
            "latency) and not where it is GPU-bound (Whisper large-v3, "
            "Qwen2-Audio-7B), so each batch width starts on single steps, times "
            "8 of them, and loops from then on only if the host's time per step "
            "is at least a quarter of the GPU's.  A row that ends mid-launch "
            "computes to the launch's end and those tokens are dropped.  0 never "
            "loops (the A/B).  Sampled rows and beam search always step singly."
        ),
    )

    loop_readahead: bool = option(
        True,
        doc=(
            "When looping, launch a group's next loop before reading the last "
            "one's tokens, so the host's work per loop (finishing rows, mapping "
            "pages, detokenizing) runs while the GPU decodes.  A row that ends in "
            "one loop then also runs the next, whose tokens are dropped.  0 reads "
            "every loop's tokens before launching the next (the A/B)."
        ),
    )

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens!r}")
        if self.beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {self.beam_size!r}")
        if self.length_penalty < 0.0:
            raise ValueError(f"length_penalty must be >= 0, got {self.length_penalty!r}")
        if self.kv_storage not in ("auto", "paged", "dense"):
            raise ValueError(
                f"kv_storage must be 'auto', 'paged' or 'dense', got {self.kv_storage!r}"
            )
        if self.kv_block_tokens < 1:
            raise ValueError(f"kv_block_tokens must be >= 1, got {self.kv_block_tokens!r}")
        if self.step_graph_width_pages < 1:
            raise ValueError(
                f"step_graph_width_pages must be >= 1, got {self.step_graph_width_pages!r}"
            )
        if self.loop_steps < 0:
            raise ValueError(f"loop_steps must be >= 0, got {self.loop_steps!r}")
        if self.step_graph_max_captures < 1:
            raise ValueError(
                f"step_graph_max_captures must be >= 1, got {self.step_graph_max_captures!r}"
            )


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
        #: Error finals for groups that failed outside :meth:`advance` (a merge
        #: during :meth:`begin_offline`), handed out by the next ``advance``.
        self._failed: List[RequestOutput] = []
        #: Finals produced outside :meth:`advance` -- rows a merge or an abort
        #: found finished while draining a loop's tokens -- handed out by the
        #: next ``advance``.
        self._ready: List[RequestOutput] = []
        #: Paged decoder-KV pool, or ``None`` for dense storage; ``_UNSET`` until
        #: the first prefill decides (see :meth:`kv_manager`).
        self._kv_manager: Any = _UNSET
        #: Decoder-step graph cache, or ``None`` when steps run eager; ``_UNSET``
        #: until the first step decides (see :meth:`_step_graphs`).
        self._graphs: Any = _UNSET
        #: Multi-step greedy loop graph cache, or ``None``; ``_UNSET`` until the
        #: first greedy step decides (see :meth:`_loop_cache`).
        self._loop_graphs: Any = _UNSET
        #: Prefill prompt-forward graph cache, or ``None``; ``_UNSET`` until the
        #: first prefill decides (see :meth:`_prefill_cache`).
        self._prefill_graphs: Any = _UNSET
        #: Per rows bucket: the host-to-GPU time ratio of each single step timed,
        #: and once decided, whether looping pays (see :meth:`_loop_pays`).
        self._step_ratios: Dict[int, List[float]] = {}
        self._loop_verdict: Dict[int, bool] = {}

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

    def _loop_filter(
        self,
    ) -> Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]]:
        """:meth:`_process_logits` as a capturable function, or ``None``.

        ``(logits (B, V), fresh (B,) bool or None) -> logits``, out of place, in
        the logits' dtype (fp32 on the eager path, the decoder's on the loop's)
        and free of host reads, where ``fresh`` marks the rows choosing their first
        token (``None`` when none can).  The multi-step loop runs it inside a CUDA
        graph, so it may not consult the group.  The default is the identity, and
        only for a family that kept the default :meth:`_process_logits`: a family
        that filters must say how to do it on the device, or it keeps stepping one
        token at a time rather than having its filter skipped.
        """
        if type(self)._process_logits is IncrementalArStrategy._process_logits:
            return _identity_filter
        return None

    def _decoder(self):
        """The batched incremental decoder surface (``prefill``/``step``/``select``)."""
        return getattr(self._model, "decoder", None)

    # ------------------------------------------------------------------
    # Decoder-KV storage
    # ------------------------------------------------------------------

    @property
    def kv_cache_format(self) -> Optional[str]:
        """``EngineConfig.quantization.kv_cache`` (``"fp8_e4m3"``), or ``None`` for the model dtype."""
        return getattr(getattr(self._config, "quantization", None), "kv_cache", None)

    def _kv_cache_kwargs(self) -> Dict[str, Any]:
        """What a family's ``prefill`` call adds for a quantized cache (nothing otherwise).

        Empty by default so a full-precision engine calls ``prefill`` exactly as
        it always has.
        """
        fmt = self.kv_cache_format
        return {} if fmt is None else {"kv_cache_dtype": fmt}

    def _require_kv_cache(self, kv_cache: str) -> None:
        """Refuse a KV-cache format this decoder, device or attention lane cannot serve.

        Each refusal names its reason, at engine construction.  Accepting the
        setting and then storing (or reading) the cache some other way would
        size the pool and admission for one format while serving another, which
        is exactly the silent mismatch ``kv_cache_dtypes`` is declared to prevent.

        * the decoder must declare the format (``BaseDecoder.kv_cache_dtypes``);
        * the model must declare its decoder geometry (``decoder_cache_spec``),
          which is what sizes the pool and is checked against the kernel;
        * on CUDA, the decoder must run fp16/bf16 and the fp8 attention lane must
          serve its head dim on this device, in this process's attention mode
          (``OASR_ATTN_BACKEND=sdpa`` / ``cute`` name lanes that cannot read fp8,
          and ``OASR_LAYERS_BACKEND=torch`` forces the dequantizing reference).
          Off CUDA the cache runs on the dequantizing reference -- the parity
          oracle, which OASR does not target for serving.
        """
        decoder = self._decoder()
        declared = tuple(getattr(decoder, "kv_cache_dtypes", ()) or ())
        name = type(decoder).__name__ if decoder is not None else "None"
        if kv_cache not in declared:
            raise ValueError(
                f"quantization.kv_cache={kv_cache!r}: the {self.decode_type!r} decoder {name} "
                f"does not store that KV-cache format (it declares kv_cache_dtypes={declared})"
            )
        spec = getattr(self._model, "decoder_cache_spec", None)
        if spec is None:
            raise ValueError(
                f"quantization.kv_cache={kv_cache!r}: the model declares no decoder_cache_spec, "
                "so there is no decoder KV geometry to size or check the cache against"
            )
        try:
            param = first_compute_parameter(decoder)
        except (StopIteration, TypeError):  # pragma: no cover - a decoder with no weights
            return
        if param.device.type != "cuda":
            return
        if param.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"quantization.kv_cache={kv_cache!r}: the fp8 KV attention lane reads fp16/bf16 "
                f"queries, and this decoder runs {param.dtype}; serve the model in float16 or "
                "bfloat16"
            )
        from oasr.jit.attention import fmha_kv8_backend, fmha_kv8_refusal
        from oasr.layers._backend import use_fmha_kernel

        try:
            lane = fmha_kv8_backend()
        except NotImplementedError as exc:
            raise ValueError(f"quantization.kv_cache={kv_cache!r}: {exc}") from None
        if lane != "cxx":
            raise ValueError(
                f"quantization.kv_cache={kv_cache!r}: OASR_ATTN_BACKEND=sdpa forces the "
                "reference attention, which would dequantize the whole cache on every step; "
                "the fp8 cache is read by the C++ lane (OASR_ATTN_BACKEND=auto or cxx)"
            )
        if not use_fmha_kernel():
            raise ValueError(
                f"quantization.kv_cache={kv_cache!r}: OASR_LAYERS_BACKEND=torch forces the "
                "reference attention, which would dequantize the whole cache on every step; "
                "the fp8 cache is read by the C++ lane (OASR_LAYERS_BACKEND=oasr)"
            )
        dtype_str = "float16" if param.dtype is torch.float16 else "bfloat16"
        for paged in (False, True):
            why = fmha_kv8_refusal(head_dim=int(spec.head_dim), dtype_str=dtype_str, paged=paged)
            if why is not None:
                raise ValueError(
                    f"quantization.kv_cache={kv_cache!r}: the fp8 KV attention lane cannot "
                    f"read this decoder's cache ({why})"
                )

    def kv_manager(self) -> Optional["DecoderKVCacheManager"]:
        """The paged decoder-KV pool, built on first use, or ``None`` for dense.

        Built lazily rather than in ``__init__`` because its size comes from
        ``EngineConfig.decode_kv_budget_gib``, which the engine derives from free
        VRAM *after* the strategy exists (it needs ``kv_bytes_per_row``).

        ``None`` — dense storage — whenever the pool cannot serve the request
        correctly rather than merely less well: beam search (a paged row's pages
        are its own, so the repeated-index ``select`` that reorders a beam grid
        would alias two slots onto one page), a decoder that has not declared
        ``supports_paged_kv``, a model with no ``decoder_cache_spec`` to shape the
        pool from, or the operator asking for ``kv_storage='dense'``.
        """
        if self._kv_manager is not _UNSET:
            return self._kv_manager
        self._kv_manager = self._build_kv_manager()
        return self._kv_manager

    def _resolve_storage(self, decoder: Any) -> str:
        """``kv_storage``, with ``'auto'`` answered for this decoder.

        Paging is not free — the block-table indirection costs a few percent —
        and on its own it buys no capacity while admission reserves each row's
        ceiling.  What it buys is *step graphs*, which need addresses that
        outlive a batch.  So ``auto`` pages exactly where that trade closes: a
        decoder that will capture its steps — an AED included, because paged
        storage pages its cross-attention K/V too.  A family that cannot
        capture keeps the dense buffer it was already paying nothing for.
        """
        storage = str(getattr(self.options, "kv_storage", "auto"))
        if storage != "auto":
            return storage
        if not bool(getattr(self.options, "step_graphs", False)):
            return "dense"
        if not getattr(decoder, "supports_step_graphs", False):
            return "dense"
        # A graph is a CUDA object.  Off the device there is nothing to capture,
        # so paging would be pure indirection — and it would put every CPU test
        # of an AR family on a path production never runs.
        try:
            param = next(decoder.parameters())
        except (AttributeError, StopIteration):
            return "dense"
        return "paged" if param.device.type == "cuda" else "dense"

    def _pages_kv(self) -> bool:
        """Whether the decoder KV will be paged, answered without building the pool.

        The pool is sized from ``decode_kv_budget_gib``, which the engine derives
        *from* :meth:`kv_bytes_per_row`.  So that method has to know the storage
        before any pool exists.  These are the same conditions
        :meth:`_build_kv_manager` applies.
        """
        if self._kv_manager is not _UNSET:
            return self._kv_manager is not None
        decoder = self._decoder()
        if self._resolve_storage(decoder) != "paged" or self._beam > 1:
            return False
        spec = getattr(self._model, "decoder_cache_spec", None)
        return spec is not None and bool(getattr(decoder, "supports_paged_kv", False))

    def _kv_block_tokens(self) -> int:
        return int(getattr(self.options, "kv_block_tokens", 16))

    def _pages_per_row(self, spec: Any) -> int:
        """Pages one row reserves at admission: its ceiling, rounded per slot.

        An AED row holds two slots, its self-attention and its fixed
        cross-attention window (``decoder_cache_spec.cross_attention_len``), and
        each rounds up to whole pages on its own.
        """
        block_tokens = self._kv_block_tokens()
        cross_len = int(getattr(spec, "cross_attention_len", 0) or 0)
        pages = -(-self._position_budget() // block_tokens)
        if cross_len:
            pages += -(-cross_len // block_tokens)
        return pages

    def _build_kv_manager(self) -> Optional["DecoderKVCacheManager"]:
        decoder = self._decoder()
        if self._resolve_storage(decoder) != "paged" or self._beam > 1:
            return None
        spec = getattr(self._model, "decoder_cache_spec", None)
        if spec is None or not getattr(decoder, "supports_paged_kv", False):
            logger.info(
                "%s: decoder KV stays dense (%s)",
                self.decode_type,
                "no decoder_cache_spec" if spec is None else "decoder has no paged surface",
            )
            return None

        from oasr.cache.decoder_kv import DecoderKVCacheManager

        # The pool's dtype and device come from the decoder's own weights, not
        # from ``EngineConfig``: the K/V written into it are that forward's
        # output, an ``index_put_`` does not cast, and a model placed anywhere
        # other than the configured device (every CPU test) would otherwise get a
        # pool it cannot write to.
        try:
            param = first_compute_parameter(decoder)
        except StopIteration:  # pragma: no cover - a decoder with no weights
            return None
        block_tokens = self._kv_block_tokens()
        rows = int(self._config.max_decode_slots or self._config.max_batch_size)
        # Two ceilings, and the tighter one wins.  The first is what admission
        # would ever hand the pool: ``max_decode_slots`` rows at their whole
        # position budget — sizing past it would reserve VRAM no admission path
        # can use.  The second is the byte budget derived from remaining device
        # memory, which binds when the admission ceiling itself would not fit.
        #
        # Counted in pages per row, the unit admission reserves in: each slot
        # rounds its ceiling up to whole pages, and an AED row holds two slots —
        # its self-attention and its fixed cross-attention window (declared as
        # ``decoder_cache_spec.cross_attention_len``).  A token total would
        # under-count both roundings, and a pool one page short per row refuses
        # the last rows admission already promised.  :meth:`kv_bytes_per_row`
        # counts the same pages, so the executor's byte budget and this pool
        # agree on how many rows fit.
        cross_len = int(getattr(spec, "cross_attention_len", 0) or 0)
        blocks = rows * self._pages_per_row(spec)
        budget_gib = float(self._config.decode_kv_budget_gib or 0.0)
        per_token = self._kv_bytes_per_token(spec)
        if budget_gib > 0:
            if per_token:
                blocks = min(blocks, int(budget_gib * (1024**3) / (per_token * block_tokens)))
        blocks = max(1, int(blocks))
        # A quantized cache is an fp8 pool (with per-(slot, head) scales beside
        # it); otherwise the K/V the forward writes are the weights' own dtype.
        from oasr.functionals.kv_quant import kv_cache_dtype

        pool_dtype = kv_cache_dtype(self.kv_cache_format) or param.dtype
        pool = DecoderKVCacheManager.build_pool(
            num_layers=int(spec.num_layers),
            n_kv_head=int(spec.n_kv_head),
            head_dim=int(spec.head_dim),
            block_tokens=block_tokens,
            max_num_blocks=blocks,
            max_batch_size=rows,
            device=param.device,
            dtype=pool_dtype,
        )
        gib = (per_token * blocks * block_tokens) / float(1024**3)
        logger.info(
            "decoder-KV pool: %d blocks x %d tokens (%.2f GiB, %s) for up to %d rows "
            "of %d positions%s",
            blocks,
            block_tokens,
            gib,
            self.kv_cache_format or str(pool_dtype).replace("torch.", ""),
            rows,
            self._position_budget(),
            f" + a {cross_len}-frame cross-attention window" if cross_len else "",
        )
        return DecoderKVCacheManager(pool)

    # ------------------------------------------------------------------
    # Decoder-step CUDA graphs
    # ------------------------------------------------------------------

    def _step_graphs(self):
        """The decoder-step graph cache, or ``None`` when steps run eager."""
        if self._graphs is not _UNSET:
            return self._graphs
        self._graphs = None
        manager = self.kv_manager()
        decoder = self._decoder()
        if (
            bool(getattr(self.options, "step_graphs", False))
            and manager is not None
            and bool(getattr(decoder, "supports_step_graphs", False))
            and manager.pool.config.device.type == "cuda"
        ):
            from ..decoder_graph import DecoderStepGraphCache

            self._graphs = DecoderStepGraphCache(
                decoder,
                manager,
                width_pages=int(getattr(self.options, "step_graph_width_pages", 8)),
                max_captures=int(getattr(self.options, "step_graph_max_captures", 64)),
            )
        elif bool(getattr(self.options, "step_graphs", False)):
            # A family that cannot capture is the normal case, not a mistake, so
            # this is only worth an INFO when somebody actually asked for graphs
            # — otherwise every AED engine logs a refusal of a default it never
            # set.  ``decode_options`` carries exactly what was passed in.
            asked = "step_graphs" in (getattr(self._config, "decode_options", None) or {})
            (logger.info if asked else logger.debug)(
                "%s: step_graphs not available (%s); steps run eager",
                self.decode_type,
                "decoder KV is not paged" if manager is None else "decoder does not declare it",
            )
        return self._graphs

    def _loop_cache(self):
        """The multi-step greedy loop graph cache, or ``None`` when steps run singly.

        Exactly where single steps capture (paged KV, a decoder that declares
        ``supports_step_graphs``, CUDA, ``step_graphs`` on), with ``loop_steps``
        at least 1 and a family whose logits filter runs on the device
        (:meth:`_loop_filter`).
        """
        if self._loop_graphs is not _UNSET:
            return self._loop_graphs
        self._loop_graphs = None
        steps = int(getattr(self.options, "loop_steps", 0))
        logits_filter = self._loop_filter()
        if steps >= 1 and logits_filter is not None and self._step_graphs() is not None:
            from ..decoder_graph import DecoderLoopGraphCache

            self._loop_graphs = DecoderLoopGraphCache(
                self._decoder(),
                self.kv_manager(),
                logits_filter=logits_filter,
                width_pages=int(getattr(self.options, "step_graph_width_pages", 8)),
                # Rows are an exact key and walk down one at a time as a group
                # finishes, and the step count is a key too.
                max_captures=4 * int(getattr(self.options, "step_graph_max_captures", 64)),
            )
        return self._loop_graphs

    def _prefill_cache(self):
        """The prefill prompt-forward graph cache, or ``None`` when prefills run eager.

        Where steps capture, for a decoder that declares
        ``supports_prefill_graphs``.
        """
        if self._prefill_graphs is not _UNSET:
            return self._prefill_graphs
        if self.kv_manager() is None:
            # Dense storage -- or the memory probe's prefill, which forces it for
            # the call.  Deciding now would resolve the step graphs against a
            # dense state and keep that answer for the life of the engine.
            return None
        self._prefill_graphs = None
        decoder = self._decoder()
        if bool(getattr(decoder, "supports_prefill_graphs", False)) and self._step_graphs():
            from ..decoder_graph import DecoderPrefillGraphCache

            self._prefill_graphs = DecoderPrefillGraphCache(
                decoder,
                self.kv_manager(),
                width_pages=int(getattr(self.options, "step_graph_width_pages", 8)),
                max_captures=int(getattr(self.options, "step_graph_max_captures", 64)),
            )
        return self._prefill_graphs

    def _loops(self, group: Any) -> bool:
        """Whether ``group`` decodes through the loop graph.

        Greedy rows only: a beam group reorders its rows every step, and a
        sampled draw is not a function of the logits the graph could replay.
        And only where it pays (:meth:`_loop_pays`).
        """
        if isinstance(group, ArBeamGroup) or not group.requests:
            return False
        if int(getattr(self.options, "loop_steps", 0)) < 1:
            return False
        if any(o is not None and o.sampling for o in group.opts):
            return False
        return self._loop_cache() is not None and self._loop_pays(len(group.requests))

    def _loop_width(self, group: Any) -> int:
        """Steps the next launch for ``group`` runs: ``loop_steps``, 1 if it does not loop.

        Never more than the row with the least generation room has left, so no
        row writes past its position budget (the cap is a step count, known
        before the launch).  A fixed width, not a tuned one: the width is part of
        a loop graph's key, and widths that followed timings kept capturing new
        shapes on live traffic (Whisper tiny's backlog halved).
        """
        if not self._loops(group):
            return 1
        room = int(min(m - len(t) for m, t in zip(group.max_new, group.tokens)))
        return max(1, min(int(getattr(self.options, "loop_steps", 1)), room))

    def _loop_pays(self, rows: int) -> bool:
        """Whether looping pays at this batch width, from timed single steps.

        A loop saves the host's per-step work and costs the steps a finishing
        row computes past its end, so it pays where a step is host-bound and
        not where it is GPU-bound.  On a single step the two add up: at a host
        time half the GPU's, the GPU idles a third of the time, which a loop
        recovers, against ~0.3 of a step per row wasted at its end.  Measured on
        an RTX 5090, the median single step's host time over its GPU time is
        0.43-0.62 for Whisper tiny at 16-64 rows, where a loop took the backlog
        +30 % and single-request latency -29 %, and 0.03-0.14 for Whisper
        large-v3, where it gained nothing and a fixed eight steps per launch
        cost 30 % of single-request latency.

        So every width starts on single steps (whose graphs a GPU-bound model
        needs anyway), the first :data:`_TIMED_STEPS` of each width bucket are
        timed (a step that captured a graph is not a sample), and the bucket
        loops from then on if the median step's host ms is at least
        :data:`_HOST_TO_STEP` times its GPU ms.  Deciding on loops instead
        captured loop graphs a GPU-bound model never used: Qwen2-Audio's warm-up
        took 10 s instead of 4.5.  An undecided bucket follows the nearest
        decided one.
        """
        bucket = _rows_bucket(rows)
        verdict = self._loop_verdict.get(bucket)
        if verdict is not None:
            return verdict
        ratios = self._step_ratios.get(bucket)
        if ratios is not None and len(ratios) >= _TIMED_STEPS:
            # The median: one step that compiled a kernel or aligned a long
            # transcript is a host outlier, not the step's cost.
            ratio = sorted(ratios)[len(ratios) // 2]
            verdict = ratio >= _HOST_TO_STEP
            self._loop_verdict[bucket] = verdict
            logger.info(
                "%s: %d-row decode groups %s (host time %.2fx GPU time per step)",
                self.decode_type,
                bucket,
                (
                    f"loop up to {int(getattr(self.options, 'loop_steps', 0))} steps per launch"
                    if verdict
                    else "step one token per launch"
                ),
                ratio,
            )
            return verdict
        if self._loop_verdict:
            nearest = min(
                self._loop_verdict,
                key=lambda b: (abs(b.bit_length() - bucket.bit_length()), b),
            )
            return self._loop_verdict[nearest]
        return False

    def _step(self, tokens: torch.Tensor, state: Any) -> Tuple[torch.Tensor, Any]:
        """One batched decoder step, replayed from a graph where one exists.

        The graph path advances the KV here because the captured step runs no
        Python; the eager path advances it inside the forward.  Both leave the
        state in the same place, which is what lets the two be compared
        token-for-token.
        """
        graphs = self._step_graphs()
        if graphs is not None:
            kv = state.get("kv") if isinstance(state, dict) else None
            if kv is not None:
                # The whole state: an AED's paged cross-attention is part of
                # what the captured step reads.
                logits = graphs.step(tokens, state)
                if logits is not None:
                    kv.commit(1)
                    return logits, state
        return self._decoder().step(tokens, state)

    @staticmethod
    def _release(state: Any) -> None:
        """Return a dropped group's paged pages to the pool.

        A no-op for dense storage, which owns nothing outside its own tensors.
        Called where a group ends *without* a ``select`` — the last row of a
        group finishing, or an abort taking it — because ``select`` is what frees
        rows the rest of the time and there is no other owner to notice.  Every
        paged part of the state is freed, not only ``kv``: an AED's
        cross-attention window holds more pages than its self-attention does.
        """
        if not isinstance(state, dict):
            return
        for part in state.values():
            free = getattr(part, "free", None)
            if callable(free):
                free()

    # ------------------------------------------------------------------
    # Admission budgeting
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

        An AED row also holds its cross-attention K/V over the whole encoder
        window (``decoder_cache_spec.cross_attention_len`` positions, same
        geometry), dense or paged alike, so that is added: a budget without it
        would admit rows by a fraction of their real footprint.

        Paged storage reserves whole pages, so there a row costs its ceiling
        rounded up to pages, per slot.  That is the unit the pool is sized in
        and admits by.  Counted in tokens instead, the executor's byte budget
        came out one row wider than the pool for budgets in a gap of
        ``rows x rounding`` tokens below each page boundary.  Qwen2-Audio's rows
        are 1,242 tokens but 1,248 paged, so the gap is 6 tokens per row: 7 % of
        budgets at 15 rows, 12 % at 26.  In it, the last row admission promised
        was refused at prefill and rejected.

        ``None`` when the model does not declare ``decoder_cache_spec``; the
        budget then stays off rather than guessing a footprint.
        """
        spec = getattr(self._model, "decoder_cache_spec", None)
        if spec is None:
            return None
        per_token = self._kv_bytes_per_token(spec)
        if self._pages_kv():
            return per_token * self._kv_block_tokens() * self._pages_per_row(spec)
        cross_len = int(getattr(spec, "cross_attention_len", 0) or 0)
        return per_token * (int(self._position_budget()) + cross_len)

    def _kv_bytes_per_token(self, spec: Any) -> int:
        """K and V bytes for one position across every decoder layer.

        With ``quantization.kv_cache="fp8_e4m3"`` that is one byte per element
        plus a 4-byte scale per (token, kv-head) -- 1.88x fewer bytes than fp16
        at head_dim 64, 1.94x at 128 -- and it is what the VRAM-derived budget
        and the per-row admission count, so the setting admits more rows rather
        than merely using less memory per row.
        """
        from oasr.functionals.kv_quant import kv_cache_bytes_per_token

        return kv_cache_bytes_per_token(
            num_layers=int(spec.num_layers),
            n_kv_head=int(spec.n_kv_head),
            head_dim=int(spec.head_dim),
            dtype=self._config.dtype,
            kv_cache=self.kv_cache_format,
        )

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
        """Per-row generation cap: the request's, else the engine's, clamped.

        ``capacity`` is the row's remaining position budget in the decoder.  The
        engine's ``max_new_tokens`` is a ceiling as well as the default: the pool
        and the admission budget give every row that many positions, so one
        request asking for more would reserve past its share and get its whole
        prefill batch refused.  A clamp that binds is reported the way every cap
        is, as ``finish_reason="length"``.
        """
        ceiling = int(self.options.max_new_tokens)
        opts = getattr(request, "decoding", None)
        requested = (
            opts.max_new_tokens if opts is not None and opts.max_new_tokens is not None else ceiling
        )
        return max(1, min(int(requested), ceiling, capacity))

    @staticmethod
    def _row_capacity(prompt_len: int, max_new: int) -> int:
        """Positions a row's decoder KV must hold: its prompt plus its cap.

        A row writes its prompt, then one position per generated token except
        the last, which is never fed back: ``prompt_len + max_new - 1`` at most,
        so this keeps one spare.  It is also within :meth:`_position_budget`,
        which is what the pool and the admission budget give each row.  The
        extra ``+ 1`` this once added put a row at the ceiling one position past
        that budget, which is a whole page when the budget is a multiple of the
        page size.
        """
        return int(prompt_len) + int(max_new)

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

    def probe_prefill(self, enc_out: torch.Tensor, enc_lengths: torch.Tensor) -> int:
        """Prefill ``enc_out``'s rows once, for measurement, and return the KV they held.

        The engine's memory probe calls this to measure the prefill transient:
        the peak it observes, minus the KV bytes returned here, which the budget
        charges separately.  Nothing outlives the call.  No group is registered,
        and the decoder KV is dense and dropped on return.  The paged pool is
        never touched: it is sized from the budget this measurement feeds, so it
        must not exist yet.
        """
        requests = [Request(streaming=False) for _ in range(int(enc_out.size(0)))]
        saved = self._kv_manager
        self._kv_manager = None
        try:
            plan = self._prefill(requests, enc_out, enc_lengths)
            return _held_bytes(plan.state)
        finally:
            self._kv_manager = saved

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
        fresh = ArGroup(
            requests=list(requests),
            state=plan.state,
            last_logits=plan.logits,
            max_new=list(plan.max_new),
            opts=opts,
            align_enc=align_enc,
            no_speech=self.prefill_no_speech(requests, plan.no_speech_logits),
        )
        if not self._absorb(fresh):
            self._groups.append(fresh)

    def _absorb(self, fresh: ArGroup) -> bool:
        """Join ``fresh`` into a group that is already generating, if it can.

        Returns whether it did.  The condition is the decoder's to answer — its
        state is opaque here — so this only walks the candidates and asks; a
        surface with no ``merge`` keeps one group per encoded micro-batch, which
        is what every AR family did before per-row KV offsets existed.

        Beam groups are skipped: their rows are a per-request slot grid advanced
        by a group-wide ``steps`` counter, so joining two of them would need a
        second, per-request counter and a rule for what a shared ``max_new``
        means across it.  Nothing declares them mergeable, so nothing tries.
        """
        if not bool(getattr(self.options, "merge_groups", True)):
            return False
        decoder = self._decoder()
        merge = getattr(decoder, "merge", None)
        can_merge = getattr(decoder, "can_merge", None)
        if not callable(merge) or not callable(can_merge):
            return False
        for group in self._groups:
            if isinstance(group, ArBeamGroup) or not can_merge(group.state, fresh.state):
                continue
            # The rows a loop in flight ends must leave before the join: its
            # tokens are listed by the rows it was launched with.
            self._drain(group, self._ready)
            if not group.requests:
                continue
            logits = torch.cat([group.last_logits, fresh.last_logits], dim=0)
            try:
                merged = merge(group.state, fresh.state)
            except Exception as exc:
                # A dense merge releases each source layer as it copies it, so
                # one that raised part-way (an OOM on the copy) has already
                # destroyed part of *both* caches.  The incumbent cannot go on
                # generating, and the fresh rows are refused by the caller.
                self._groups = [g for g in self._groups if g is not group]
                self._failed.extend(self._fail_group(group, exc))
                self._release(fresh.state)
                raise
            group.extend(fresh, merged, logits)
            return True
        return False

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
        return bool(self._groups) or bool(self._failed) or bool(self._ready)

    def advance(self, budget: "StepBudget") -> List[RequestOutput]:
        """Advance the groups round-robin within ``budget``, one group at a time.

        A group whose step raises is failed alone (:meth:`_fail_group`): its
        rows get error finals and its KV goes back.  Without that, the group
        was off the queue with its pages still held, and the exception left
        ``step()`` and failed every request in flight through the dispatcher.
        """
        outputs: List[RequestOutput] = self._failed + self._ready
        self._failed, self._ready = [], []
        advanced: List[ArGroup] = []
        while self._groups:
            # Round-robin: advance the front group one launch -- one batched
            # step, or several when it loops on the device -- and re-queue it
            # at the back if it still has active rows.
            steps = budget.take_up_to(self._loop_width(self._groups[0]))
            if not steps:
                break
            group = self._groups.pop(0)
            try:
                if isinstance(group, ArBeamGroup):
                    self._advance_beam_group(group, outputs)
                else:
                    self._advance_group(group, outputs, steps)
            except Exception as exc:  # noqa: BLE001 — one group must not take the tick
                outputs.extend(self._fail_group(group, exc, outputs))
                continue
            if group.requests:
                self._groups.append(group)
                if not any(g is group for g in advanced):
                    advanced.append(group)
        if self.emit_partials:
            for group in advanced:
                try:
                    if isinstance(group, ArBeamGroup):
                        outputs.extend(self._beam_partials([group]))
                    else:
                        outputs.extend(self._partials([group]))
                except Exception as exc:  # noqa: BLE001 — as above
                    self._groups = [g for g in self._groups if g is not group]
                    outputs.extend(self._fail_group(group, exc, outputs))
        return outputs

    def _fail_group(
        self,
        group: ArGroup,
        exc: BaseException,
        emitted: Sequence[RequestOutput] = (),
    ) -> List[RequestOutput]:
        """Error finals for ``group``'s rows, with its decoder KV handed back.

        ``group`` is already off the queue.  A row that finished before the
        failure keeps its final, which is in ``emitted``; every other row gets
        ``finish_reason="error"``.  ``error_stage`` is ``decode_oom`` for an
        allocation that failed (device memory, or the decoder-KV pool) and
        ``decode_step`` otherwise.
        """
        done = {out.request_id for out in emitted if out.finished}
        failed = [req for req in group.requests if req.request_id not in done]
        oom = isinstance(exc, (torch.cuda.OutOfMemoryError, DecoderKvExhausted))
        stage = "decode_oom" if oom else "decode_step"
        logger.warning(
            "%s: decode group of %d row(s) failed (%s: %s); failing requests %s",
            self.decode_type,
            len(group.requests),
            type(exc).__name__,
            exc,
            [req.request_id for req in failed],
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        try:
            self._release(group.state)
        except Exception:  # pragma: no cover - the backstop is PagedDecoderKv.__del__
            logger.exception("%s: releasing a failed group's decoder KV raised", self.decode_type)
        group.state = {}
        group.requests = []
        if isinstance(group, ArGroup):
            group.pending_read = None
        return [
            RequestOutput(
                request_id=req.request_id,
                text="",
                tokens=[[]],
                finished=True,
                finish_reason="error",
                error_stage=stage,
            )
            for req in failed
        ]

    def _advance_group(self, group: ArGroup, outputs: List[RequestOutput], steps: int = 1) -> None:
        """Advance ``group`` by ``steps`` tokens per row.

        Several steps run as one loop graph when :meth:`_loop_width` allowed
        them; a loop the cache cannot serve (a shape it refused, a capture that
        failed) falls back to that many single steps, which the budget already
        paid for.
        """
        if self._loops(group) and self._advance_group_loop(group, outputs, steps):
            return
        # A single step reads its token at once, so it starts from tokens read.
        self._drain(group, outputs)
        for _ in range(steps):
            if not group.requests:
                return
            self._advance_group_once(group, outputs)

    def _advance_group_once(self, group: ArGroup, outputs: List[RequestOutput]) -> None:
        """Select one token per active row, retire finished rows, take one step.

        Finals go into ``outputs`` as rows finish, before the ``select`` and the
        step, so a step that then fails cannot take them with it.
        """
        t0 = time.perf_counter()
        logits = self._process_logits(group.last_logits.float(), group)
        next_tokens = select_next_tokens(logits, group.opts)  # (B_active,)

        t_wait = time.perf_counter()
        toks = next_tokens.cpu().tolist()
        waited = time.perf_counter() - t_wait
        # The read above waited out the step this group took last, so its GPU
        # time can be read now.
        timed, group.timed_step = group.timed_step, None
        if timed is not None:
            self._note_step(*timed)
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

        keep_idx = self._retire_rows(group, finished_rows, reasons, outputs)
        if keep_idx is not None:
            next_tokens = next_tokens.index_select(0, keep_idx)
        if not group.requests:
            return
        rows = len(group.requests)
        started: Optional["torch.cuda.Event"] = None
        graphs = self._step_graphs()
        captured = graphs.num_captured if graphs is not None else 0
        if self._times_steps(group, rows):
            started = torch.cuda.Event(enable_timing=True)
            started.record()
        with torch.no_grad():
            group.last_logits, group.state = self._step(next_tokens, group.state)
        # A step that captured its graph spent its host time capturing -- a
        # 7B's capture is hundreds of milliseconds -- and is not a sample: two
        # of them made Qwen2-Audio's steps look host-bound.
        if started is not None and (graphs is None or graphs.num_captured == captured):
            ended = torch.cuda.Event(enable_timing=True)
            ended.record()
            group.timed_step = (rows, started, ended, time.perf_counter() - t0 - waited)

    def _times_steps(self, group: ArGroup, rows: int) -> bool:
        """Whether this step is timed toward a loop verdict for its width."""
        if not group.last_logits.is_cuda or self._loop_cache() is None:
            return False
        return _rows_bucket(rows) not in self._loop_verdict

    def _note_step(
        self,
        rows: int,
        started: "torch.cuda.Event",
        ended: "torch.cuda.Event",
        host_s: float,
    ) -> None:
        """Fold one timed single step into its width bucket's means."""
        bucket = _rows_bucket(rows)
        if bucket in self._loop_verdict:
            return
        gpu_ms = max(started.elapsed_time(ended), 1e-3)
        self._step_ratios.setdefault(bucket, []).append(host_s * 1e3 / gpu_ms)

    def _advance_group_loop(self, group: ArGroup, outputs: List[RequestOutput], steps: int) -> bool:
        """``steps`` greedy steps in one loop-graph replay; ``False`` if not served.

        The loop has already fed every row's tokens back by the time they are
        read, so the state and the pending logits are those after the last step
        for every row.  A row ends at its first EOS or at its cap; whatever it
        chose after that is dropped with it.

        With ``loop_readahead`` the tokens are read one launch late: this loop is
        launched first, then the previous one's tokens are read and its finished
        rows retired while the GPU runs this one.
        """
        cache = self._loop_cache()
        if cache is None:
            return False
        ahead = bool(getattr(self.options, "loop_readahead", False))
        previous = group.pending_read
        if previous is not None:
            # No row may run past its cap counting the tokens still in flight;
            # a loop that would have to waits for them to be read.
            room = min(m - len(t) for m, t in zip(group.max_new, group.tokens))
            if room - previous.steps < steps:
                self._drain(group, outputs)
                previous = None
                if not group.requests:
                    return True
                room = min(m - len(t) for m, t in zip(group.max_new, group.tokens))
                steps = min(steps, room)
        device = group.last_logits.device
        # A row is choosing its first token only if no launch has chosen one
        # for it: a row in the loop still in flight has, though its tokens are
        # not read yet.  Treating it as fresh re-applies the first-position
        # filter -- Whisper's begin-suppress list holds EOT -- one launch late.
        launched = {id(req) for req in previous.requests} if previous is not None else set()
        fresh = to_device(
            [not t and id(req) not in launched for req, t in zip(group.requests, group.tokens)],
            dtype=torch.bool,
            device=device,
        )
        with torch.no_grad():
            served = cache.loop(group.state, group.last_logits, fresh, steps)
        if served is None:
            return False
        tokens, logits = served
        group.state["kv"].commit(steps)
        group.last_logits = logits.clone()  # a replay buffer; the next replay reuses it
        landed: Optional["torch.cuda.Event"] = None
        if tokens.is_cuda:
            # Into pinned memory behind the loop, ahead of any later replay.
            host = tokens.to("cpu", non_blocking=True)
            landed = torch.cuda.Event()
            landed.record(torch.cuda.current_stream(tokens.device))
        else:  # pragma: no cover - a loop cache only exists on CUDA
            host = tokens.clone()
        group.pending_read = _LoopRead(list(group.requests), host, landed, steps)
        if previous is not None:
            self._collect_read(group, previous, outputs)
        if not ahead:
            self._drain(group, outputs)
        return True

    def _drain(self, group: ArGroup, outputs: List[RequestOutput]) -> None:
        """Read ``group``'s in-flight loop tokens now, finishing whatever they end."""
        read, group.pending_read = group.pending_read, None
        if read is not None:
            self._collect_read(group, read, outputs)

    def _collect_read(
        self, group: ArGroup, read: "_LoopRead", outputs: List[RequestOutput]
    ) -> None:
        """Append one loop's tokens to the rows still in ``group``; retire the
        rows they finish.

        ``read`` lists its rows as they were at launch.  Rows retired since (by
        an earlier read, an abort) are skipped; the rest are found by identity,
        since retiring removes rows and reorders none.
        """
        if read.landed is not None:
            read.landed.synchronize()  # the one wait for the device per loop
        chosen_rows = read.tokens.tolist()
        index = {id(req): row for row, req in enumerate(group.requests)}
        finished_rows: List[int] = []
        reasons: Dict[int, str] = {}
        for req, chosen in zip(read.requests, chosen_rows):
            row = index.get(id(req))
            if row is None:
                continue
            generated = group.tokens[row]
            for tok in chosen:
                if self._is_eos(tok):
                    finished_rows.append(row)
                    reasons[row] = "stop"
                    break
                generated.append(tok)
                if len(generated) >= group.max_new[row]:
                    finished_rows.append(row)
                    reasons[row] = "length"
                    break
        pending = group.pending_read
        keep_idx = self._retire_rows(group, sorted(finished_rows), reasons, outputs)
        if keep_idx is not None:
            group.last_logits = group.last_logits.index_select(0, keep_idx)
        group.pending_read = pending if group.requests else None

    def _retire_rows(
        self,
        group: ArGroup,
        finished_rows: List[int],
        reasons: Dict[int, str],
        outputs: List[RequestOutput],
    ) -> Optional[torch.Tensor]:
        """Finalize ``finished_rows`` and drop them from the group.

        Returns the device index of the rows kept, for the caller to apply to
        what it holds per row, or ``None`` when no row finished or none is left.
        """
        for row in finished_rows:
            outputs.append(self._finalize_row(group, row, reasons[row]))
        if not finished_rows:
            return None
        keep = [r for r in range(len(group.requests)) if r not in finished_rows]
        group.keep_rows(keep)
        if not keep:
            # Every row finished together, so no ``select`` runs to hand the
            # group's storage back; the group itself is about to be dropped.
            self._release(group.state)
            return None
        device = group.last_logits.device
        keep_idx = to_device(keep, dtype=torch.long, device=device)
        kv = group.state.get("kv") if isinstance(group.state, dict) else None
        if isinstance(kv, PagedDecoderKv) and device.type == "cuda":
            # Paged rows take their index on the host: a device one is read back,
            # and that read waits for a loop still in flight.
            group.state = self._decoder().select(
                group.state, torch.tensor(keep, dtype=torch.long).pin_memory()
            )
        else:
            group.state = self._decoder().select(group.state, keep_idx)
        return keep_idx

    # ------------------------------------------------------------------
    # Beam advance
    # ------------------------------------------------------------------

    def _advance_beam_group(self, group: ArBeamGroup, outputs: List[RequestOutput]) -> None:
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

        finals, keep = self._retire_finished_requests(group)
        outputs.extend(finals)
        if group.requests:
            # Restrict the fed tokens to the surviving slots, in the same flat
            # order ``_retire_finished_requests`` left the state in.
            if keep is not None:
                chosen = chosen.index_select(0, keep)
            with torch.no_grad():
                group.last_logits, group.state = self._step(chosen, group.state)

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

    def prefill_no_speech(
        self, requests: List[Request], logits: Optional[torch.Tensor]
    ) -> List[Optional[float]]:
        """``P(no speech)`` per row from :attr:`Prefill.no_speech_logits`.

        The default is "this family has no such token", which is the honest
        answer for every AR family but Whisper.  Overridden rather than
        conditioned on ``hasattr`` so a family that gains the token later
        declares it in one place.
        """
        return [None] * len(requests)

    def _finalize_row(self, group: ArGroup, row: int, reason: str) -> RequestOutput:
        tokens = group.tokens[row]
        out = RequestOutput(
            request_id=group.requests[row].request_id,
            text=self._row_text(group, row),
            tokens=[tokens],
            finished=True,
            finish_reason=reason,
        )
        if row < len(group.no_speech):
            out.no_speech_prob = group.no_speech[row]
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
        # A request whose group already failed holds no KV, only its error final.
        self._failed = [out for out in self._failed if out.request_id != request.request_id]
        for group in self._groups:
            if request not in group.requests:
                continue
            if not isinstance(group, ArBeamGroup):
                self._drain(group, self._ready)
                if request not in group.requests:
                    break  # the loop in flight had already finished it
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
            else:
                self._release(group.state)
            break
        self._groups = [g for g in self._groups if g.requests]
        # The aborted request's own final, if the drain produced one, is moot.
        self._ready = [out for out in self._ready if out.request_id != request.request_id]

    def release_graphs(self) -> None:
        """Release the decode graphs, after any loop still running has finished.

        A loop launched with ``loop_readahead`` may still be executing, and its
        tokens' copy with it; the pool its graph replays from is what this frees.
        """
        if torch.cuda.is_available() and any(
            getattr(g, "pending_read", None) is not None for g in self._groups
        ):
            torch.cuda.synchronize()
        super().release_graphs()

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
