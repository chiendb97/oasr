# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Request representation and lifecycle state for the ASR engine."""

from __future__ import annotations

import dataclasses
import enum
import time
import uuid
from dataclasses import dataclass
from typing import ClassVar, Deque, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from oasr.cache import StreamContext


class RequestState(enum.Enum):
    """Lifecycle state of an ASR request."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


# Default priority level (lower value = higher priority).
# Streaming requests default to this; offline can be bumped lower-priority.
DEFAULT_PRIORITY = 0

#: Sampling-temperature bounds for :class:`DecodingOptions`.  ``0`` means greedy;
#: any other value must land in ``[MIN_TEMPERATURE, MAX_TEMPERATURE]`` so
#: ``logits / temperature`` can neither overflow to ``inf`` nor flatten the
#: distribution into a no-op.  The serving layer clamps to the same range.
MIN_TEMPERATURE = 0.01
MAX_TEMPERATURE = 100.0


@dataclass
class DecodingOptions:
    """Per-request decoding options.

    Engine-level knobs (kernel beam width, tick budgets, decoder type) stay on
    :class:`~oasr.engine.config.EngineConfig`; this carries only what may vary
    request to request.  Every field has a "no effect" default, so an absent /
    default-constructed options object reproduces today's behaviour exactly.

    Attributes
    ----------
    n_best : int
        How many hypotheses to detokenize into
        :attr:`RequestOutput.nbest_texts` on the **final** output (the serving
        layer maps ``max_alternatives`` here).  ``1`` (default) fills only
        :attr:`RequestOutput.text`.  Only beam families (CTC / WFST /
        rescoring) produce more than one hypothesis; greedy families ignore
        values above what they emit.  Interim streaming partials always carry
        the best hypothesis only.
    max_new_tokens : int, optional
        Per-request generation cap for the incremental AR strategies
        (AED / LLM), overriding ``EngineConfig.max_new_tokens``; still clamped
        by the model's position-embedding capacity.  Ignored by
        frame-synchronous families.
    temperature : float
        ``0.0`` (default) — greedy.  ``> 0`` enables sampling for the AR
        strategies: logits are divided by the temperature before the
        ``top_k`` / ``top_p`` filters and a multinomial draw.  Sampling uses
        the process-global torch generator (seed with ``torch.manual_seed``
        for reproducibility).
    top_k : int
        Keep only the ``k`` highest-probability tokens before sampling.
        ``0`` (default) disables the filter.  Only meaningful with
        ``temperature > 0``.
    top_p : float
        Nucleus sampling — keep the smallest set of tokens whose cumulative
        probability reaches ``top_p``.  ``1.0`` (default) disables the
        filter.  Only meaningful with ``temperature > 0``.
    prompt : str, optional
        Per-request user prompt for the speech-LLM strategy, overriding
        ``EngineConfig.llm_prompt`` / the checkpoint default.  Ignored by
        every other family (Whisper's SOT sequence is checkpoint-fixed apart
        from the ``task`` / ``language`` slots below).
    task : str, optional
        ``"transcribe"`` or ``"translate"`` for the families whose prompt
        carries a task token (Whisper AED).  ``None`` (default) keeps the
        checkpoint's own task.  This is what backs OpenAI's
        ``/v1/audio/translations``: before it existed the task was frozen in
        the checkpoint's ``forced_decoder_ids`` at conversion time.
    language : str, optional
        ISO-639 primary subtag (``"en"``, ``"fr"``) for the families whose
        prompt carries a language token.  ``None`` (default) keeps the
        checkpoint's own.  The serving layer reduces a BCP-47 tag
        (``"en-US"``) before it reaches here.

    Both ``task`` and ``language`` are **validated against the running decode
    family** at admission (:meth:`DecodeStrategy.validate_options`), so a
    family that cannot honour them rejects the request instead of silently
    transcribing under the checkpoint's default.
    """

    n_best: int = 1
    max_new_tokens: Optional[int] = None
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    prompt: Optional[str] = None
    task: Optional[str] = None
    language: Optional[str] = None

    #: Task values any family may declare support for.  Mirrored by
    #: ``oasr_wire::TASKS`` on the other side of the PyO3 boundary.
    TASKS: ClassVar[Tuple[str, ...]] = ("transcribe", "translate")

    def __post_init__(self) -> None:
        if self.n_best < 1:
            raise ValueError(f"n_best must be >= 1, got {self.n_best!r}")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1 or None, got {self.max_new_tokens!r}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature!r}")
        # A temperature between 0 and MIN_TEMPERATURE divides the logits by a
        # near-zero number: the result overflows to ±inf and ``torch.multinomial``
        # then raises *inside* the decoder step, for the whole batched group.
        # Values that small are numerically indistinguishable from greedy anyway,
        # so ask for greedy explicitly instead.
        if 0.0 < self.temperature < MIN_TEMPERATURE:
            raise ValueError(
                f"temperature must be 0 (greedy) or >= {MIN_TEMPERATURE}, got "
                f"{self.temperature!r}"
            )
        if self.temperature > MAX_TEMPERATURE:
            raise ValueError(f"temperature must be <= {MAX_TEMPERATURE}, got {self.temperature!r}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k!r}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p!r}")
        if self.task is not None and self.task not in self.TASKS:
            raise ValueError(f"task must be one of {list(self.TASKS)}, got {self.task!r}")
        if self.language is not None:
            # Whether *this checkpoint* knows the language is the strategy's
            # question; the shape is this one's.  A tag that still carries a
            # region ("en-US") would miss the token table and decode in the
            # checkpoint's default language, confidently.
            if not self.language.isalpha() or not self.language.islower():
                raise ValueError(
                    "language must be a lowercase ISO-639 primary subtag such "
                    f'as "en", got {self.language!r}'
                )

    @property
    def sampling(self) -> bool:
        """Whether this request draws tokens instead of taking the argmax."""
        return self.temperature > 0.0

    @classmethod
    def coerce(cls, value: Union["DecodingOptions", Mapping, None]) -> Optional["DecodingOptions"]:
        """Normalise an options value from a Python or PyO3 caller.

        Accepts ``None`` (no options), an existing :class:`DecodingOptions`,
        or a plain mapping (the Rust dispatcher passes a dict) whose ``None``
        values are treated as "use the default".
        """
        if value is None or isinstance(value, cls):
            return value
        # Drive the key set off the dataclass, never a literal list: a hardcoded
        # tuple silently drops any field added to one side and not the other,
        # with no error at either end.  ``option_keys`` is the same set the Rust
        # front-end asserts against at startup.
        kwargs = {k: value[k] for k in cls.option_keys() if k in value and value[k] is not None}
        return cls(**kwargs)

    @classmethod
    def option_keys(cls) -> Tuple[str, ...]:
        """The per-request option names, in declaration order.

        The single source of truth for the option table that crosses PyO3.
        ``oasr-wire``'s ``DecodingParams`` must produce exactly these keys;
        :func:`assert_matches_wire_keys` checks that at engine construction so a
        rename fails fast instead of silently dropping the option.
        """
        return tuple(f.name for f in dataclasses.fields(cls))

    @classmethod
    def assert_matches_wire_keys(cls, keys: Iterable[str]) -> None:
        """Fail loudly if the Rust option table has drifted from this dataclass.

        Called once from the PyO3 boundary at startup.  Without it, adding a
        field on one side only means requests carrying that option are accepted
        and ignored — the exact silent drift S9 catalogued, and unobservable
        from either end.
        """
        theirs, ours = set(keys), set(cls.option_keys())
        if theirs != ours:
            raise ValueError(
                "per-request decoding option tables disagree across the PyO3 "
                f"boundary: only in Rust {sorted(theirs - ours)}, only in Python "
                f"{sorted(ours - theirs)}. Update oasr_wire::DecodingParams and "
                "oasr.engine.DecodingOptions together."
            )


@dataclass
class RequestOutput:
    """Output produced by the engine for a single request.

    Attributes
    ----------
    request_id : str
        Identifier matching the originating :class:`Request`.
    text : str
        Detokenized transcript (best hypothesis).
    tokens : List[List[int]]
        N-best token ID sequences (outer list = N-best, inner = tokens).
    scores : List[float], optional
        Log-probability scores per hypothesis.
    finished : bool
        ``True`` when decoding is complete; ``False`` for partial streaming
        results.
    timestamps : List[Tuple[float, float]], optional
        Per-token ``(start_s, end_s)`` times for the **best** hypothesis,
        aligned with ``tokens[0]``.  Emitted by decode families that produce
        alignments (Paraformer's CIF fire positions); ``None`` otherwise.
    nbest_texts : List[str], optional
        Detokenized transcripts for the top hypotheses, aligned with
        ``tokens`` rows (``nbest_texts[0] == text``).  Filled on final
        outputs when the request asked for ``DecodingOptions.n_best > 1``
        and the decode family produced multiple hypotheses; ``None``
        otherwise.
    finish_reason : str, optional
        Why generation stopped, for the incremental AR families:
        ``"stop"`` (EOS emitted) or ``"length"`` (``max_new_tokens`` hit).
        ``"error"`` when the executor could not run the request to completion.
        ``None`` for frame-synchronous families (they always consume the
        full audio) and for partial outputs.
    error_stage : str, optional
        Which stage failed, when ``finish_reason == "error"`` — e.g.
        ``"offline_forward"``, ``"streaming_forward"``, ``"prefill_oom"``.
        Set alongside ``finish_reason`` rather than folded into it so the
        serving layer's error envelope stays a stable two-value vocabulary
        while ``oasr_requests_failed_total{stage}`` still says *where*.
    """

    request_id: str
    text: str
    tokens: List[List[int]]
    scores: Optional[List[float]] = None
    finished: bool = False
    timestamps: Optional[List[Tuple[float, float]]] = None
    nbest_texts: Optional[List[str]] = None
    finish_reason: Optional[str] = None
    error_stage: Optional[str] = None


class Request:
    """A single ASR inference request.

    Parameters
    ----------
    audio : Tensor or ndarray
        A raw **waveform** — ``torch.Tensor`` of shape ``(num_samples,)`` or
        ``(1, num_samples)``, or a NumPy array — at the model sample rate.
        The engine is waveform-only; file decoding happens at the entry point
        (the serving front-end via ``oasr-asr``, or the bench/test harness),
        never inside the engine.
    request_id : str, optional
        Unique identifier.  Auto-generated (UUID4 hex) if not provided.
    streaming : bool
        If ``True``, process via the chunk-by-chunk streaming path with
        paged attention cache.  If ``False``, use the single-pass offline path.
    sample_rate : int
        Sample rate of the audio in Hz.  Must equal the model's own rate — the
        engine does not resample and derives every frame count from
        ``FeatureConfig.sample_rate``; admission rejects a mismatch.  Prefer
        letting :meth:`ASREngine.add_request` default it.
    decoding : DecodingOptions, optional
        Per-request decoding options (n-best, generation cap, sampling,
        prompt).  ``None`` keeps every engine default.
    """

    def __init__(
        self,
        audio: Optional[Union[torch.Tensor, "np.ndarray"]] = None,
        request_id: Optional[str] = None,
        streaming: bool = False,
        sample_rate: int = 16000,
        priority: int = DEFAULT_PRIORITY,
        decoding: Optional[DecodingOptions] = None,
    ) -> None:
        self.request_id: str = request_id or uuid.uuid4().hex
        self.decoding: Optional[DecodingOptions] = decoding
        # The engine's single audio input slot — always a **waveform** (1-D
        # float32 samples) or ``None``; the engine never takes file paths
        # (decode at the entry point).  Its role depends on the mode:
        #   * offline — the input waveform.  ``prepare_offline`` canonicalises
        #     it in place (→ 1-D float32 CPU), ``collate`` consumes it and
        #     then clears it to ``None`` once the GPU feature tensor owns the
        #     batch.
        #   * streaming — ``None`` for the chunk-by-chunk API
        #     (``add_streaming_request`` + ``feed_chunk``), or a pre-loaded
        #     waveform that ``StreamingExecutor.admit`` splits into chunks
        #     (``transcribe(..., streaming=True)``).
        # ndarray / ``(1, T)`` / non-float32 inputs are accepted and normalised
        # on first use — there is no separate "raw vs. normalised" field.
        self.audio: Optional[Union[torch.Tensor, "np.ndarray"]] = audio
        self.streaming: bool = streaming
        self.sample_rate: int = sample_rate
        self.priority: int = priority
        self.arrival_time: float = time.monotonic()

        # Lifecycle state
        self.state: RequestState = RequestState.WAITING

        # Populated by InputProcessor
        self.features: Optional[torch.Tensor] = None  # (1, T, F)
        self.feature_lengths: Optional[torch.Tensor] = None  # (1,)
        # Number of feature frames.  For offline this starts as a cheap
        # sample-count-derived estimate so the scheduler can bucket before
        # features are extracted, and is overwritten with the exact value
        # after the batched extraction runs.
        self.num_frames: int = 0

        # --- Streaming audio-chunk state (all populated by InputProcessor) ---
        # Queue of audio-sample chunks awaiting feature extraction.  Each
        # element is a CPU float32 1-D tensor of raw samples.  Streaming
        # ingests strictly left-to-right from this queue; the scheduler never
        # reaches ahead of the currently-enqueued chunks (no future audio).
        self.audio_chunks: Optional[Deque[torch.Tensor]] = None
        # Residual samples from the last fbank call — the suffix of the
        # combined (tail + new) waveform that didn't fit in a whole frame.
        # They get prepended to the next audio chunk so frame boundaries
        # stay aligned across streaming invocations.
        self.audio_tail: Optional[torch.Tensor] = None
        # Device-side feature ring.  Grown by extraction; encoder chunks are
        # sliced from ``features[feature_cursor : feature_cursor + window]``.
        self.feature_buffer: Optional[torch.Tensor] = None
        # Number of valid feature frames currently in ``feature_buffer``.
        self.feature_frames: int = 0
        # Feature-frame index of the next encoder chunk's start.
        self.feature_cursor: int = 0
        # Flips to True when the final audio chunk has been enqueued (no
        # more audio will arrive).  Triggers fbank flush + last-window forward.
        self.audio_final: bool = False

        # Running total of audio samples enqueued via ``feed_chunk`` /
        # ``prepare_streaming``.  Used by :meth:`append_streaming_chunk` to
        # update ``num_frames`` in O(1) instead of summing the whole deque.
        self.samples_enqueued: int = 0

        # Assigned by Scheduler
        self.stream_id: Optional[int] = None
        # Assigned by ModelRunner when admitting the stream — a fixed slot
        # id in ``[0, max_batch_size)`` indexing into the engine's
        # persistent batched block_table, cache_seqlens, CNN cache and
        # feature buffer. Released back to the slot pool on free_stream.
        self.slot_id: Optional[int] = None

        # Populated by ModelRunner (streaming only)
        self.stream_context: Optional[StreamContext] = None
        self.offset: int = 0  # encoder output frame offset
        # Set by the streaming backend when this stream's encoder cache can grow
        # no further (paged pool / block-table capacity reached with unlimited
        # history).  The executor finalizes such streams with the transcript
        # decoded so far and ``finish_reason="length"`` rather than letting the
        # allocator raise mid-forward.
        self.cache_exhausted: bool = False

        # Final output
        self.output: Optional[RequestOutput] = None

    @property
    def is_finished(self) -> bool:
        return self.state == RequestState.FINISHED

    @property
    def has_pending_audio(self) -> bool:
        """True if audio samples still need to be turned into features."""
        return bool(self.audio_chunks) or (
            self.audio_final and self.audio_tail is not None and self.audio_tail.numel() > 0
        )

    def has_ready_encoder_chunk(self, window: int) -> bool:
        """True if ``feature_buffer`` holds enough frames for a full window.

        At final-flush time (no more audio coming) we emit whatever remains,
        even if it's shorter than ``window``.
        """
        available = self.feature_frames - self.feature_cursor
        if available >= window:
            return True
        if self.audio_final and not self.audio_chunks and available > 0:
            return True
        return False

    @property
    def waited_for(self) -> float:
        """Seconds spent in the waiting queue (monotonic clock)."""
        return time.monotonic() - self.arrival_time

    def __repr__(self) -> str:
        if self.audio is None:
            audio_repr = "None"
        elif isinstance(self.audio, str):
            audio_repr = self.audio
        else:
            audio_repr = type(self.audio).__name__
        return (
            f"Request(id={self.request_id[:8]}, state={self.state.value}, "
            f"streaming={self.streaming}, audio={audio_repr})"
        )
