# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Speech-detector registry — the eighth extension axis.

Deliberately shaped like :mod:`oasr.features.registry`: a frozen spec object
carrying the callable plus the properties callers need, a ``register_*``
function, a lazy import of the built-ins, and a lookup that raises naming the
registered kinds.  A new detector is a subclass plus one registration call; no
engine edit, no ``EngineConfig`` field.

Two declarations on :class:`VadSpec` carry real consequences, so set them
deliberately:

* ``consumes`` says what the engine must feed the detector, and it is what makes
  *"an ASR-derived detector cannot pre-segment"* a fact of the type rather than a
  comment.  ``register_vad`` refuses a spec claiming ``"presegment"`` with an
  ASR-derived ``consumes``, at registration time, because discovering it on the
  first request means discovering it in production.
* ``framing`` is a **function of the config**, not a constant, for the same
  reason ``ExtractorSpec.framing`` is: span and hop are config-derived.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from .config import VadConfig
    from .detector import SpeechDetector

logger = logging.getLogger(__name__)

__all__ = [
    "VadFraming",
    "VadSpec",
    "register_vad",
    "get_vad_spec",
    "build_detector",
    "list_vad",
    "CONSUMES",
    "ROLES",
    "ASR_CONSUMES",
]

#: What the engine feeds a detector.  ``waveform`` detectors are the only ones
#: that can run *before* the ASR; every other value names a tensor the ASR
#: itself produced, which is what makes them free and also what makes them
#: unable to pre-segment.
CONSUMES: Tuple[str, ...] = (
    "waveform",
    "asr_log_probs",  # (B, T, V) CTC log-softmax — ctc_blank
    "asr_frames",  # (B, T) per-frame activity indicator — transducer_blank
    "asr_alphas",  # (B, T) CIF weights, already in [0, 1] — cif_alpha
    "asr_prefill_logits",  # (B, V) logits at the first generated position — aed_no_speech
)

#: The subset of ``CONSUMES`` that requires the ASR to have run already.
ASR_CONSUMES: Tuple[str, ...] = tuple(c for c in CONSUMES if c.startswith("asr_"))

#: Pipeline roles a detector can declare.
#:
#: ``presegment`` — can run ahead of the encoder, so it can drive offline
#: segmentation and streaming gating.  ``stream`` — can consume a growing
#: buffer incrementally.  ``posthoc`` — can only label audio the ASR already
#: transcribed.
ROLES: Tuple[str, ...] = ("presegment", "stream", "posthoc")


@dataclass(frozen=True)
class VadFraming:
    """How a detector's output frames map onto audio, in samples.

    The VAD twin of :class:`~oasr.features.registry.StreamingFraming`, and
    deliberately the same four integers: a sliding window over a growing sample
    buffer is a sliding window whatever produces it.

    Attributes
    ----------
    span : int
        Samples one output frame reads, from its own start.
    hop : int
        Samples between consecutive frame starts.
    history : int
        Leading buffer samples that are context only — they influence frame
        values but do not start a frame.
    prefill : int
        Zero samples the buffer starts with, standing in for the implicit left
        padding of a one-shot pass.
    """

    span: int
    hop: int
    history: int = 0
    prefill: int = 0

    def __post_init__(self) -> None:
        if self.span <= 0 or self.hop <= 0:
            raise ValueError(f"span and hop must be positive, got {self.span}/{self.hop}")
        if self.history < 0 or self.prefill < 0:
            raise ValueError("history and prefill must be >= 0")

    def frames_for(self, num_samples: int) -> int:
        """Frames a buffer of ``num_samples`` can emit (``>= 0``)."""
        if num_samples < self.history + self.span:
            return 0
        return (num_samples - self.history - self.span) // self.hop + 1

    @property
    def min_samples(self) -> int:
        """Buffered samples needed before the first frame can be emitted."""
        return self.history + self.span

    def seconds_per_frame(self, sample_rate: int) -> float:
        """Seconds one output frame advances.

        The only time base a VAD span may be reported in.  Everything the engine
        publishes — word timings, token timestamps, segment boundaries — is
        seconds of *audio*, never wall clock, which is also why Google computes
        its speech-event offsets from bytes received rather than server time.
        """
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        return self.hop / float(sample_rate)


@dataclass(frozen=True)
class VadSpec:
    """One registered speech detector plus the properties callers need."""

    kind: str
    factory: Callable[..., "SpeechDetector"]
    consumes: str
    #: ``None`` for an ASR-derived kind: its grid is the encoder's, and the
    #: engine resolves it from the running model rather than from this config.
    framing: Optional[Callable[["VadConfig"], VadFraming]] = None
    modes: Tuple[str, ...] = ("stream",)
    #: Dotted model attribute paths an ASR-derived kind needs, checked with the
    #: same machinery ``CAPABILITIES`` uses so a failure names the missing member.
    requires: Tuple[str, ...] = ()
    #: True when the detector carries per-stream recurrent state across chunks.
    stateful: bool = False
    #: Shortest silence this detector's signal can tell apart from its own
    #: sparsity, in milliseconds.  ``0`` for a genuine frame-level detector.
    #:
    #: The ASR-derived signals are **peaky**, and this is the number that says
    #: so.  A CTC head emits a non-blank at a handful of frames per second and
    #: blank everywhere else, so ``1 - P(blank)`` sits near zero *inside* a word:
    #: measured on read speech at a 40 ms frame rate, only 15 % of frames clear
    #: 0.5 and runs below threshold reach 840 ms without any pause in the audio.
    #: Handing that trace a 100 ms minimum-silence would shred one utterance
    #: into dozens of segments.  WeNet reaches the same conclusion from the other
    #: end and ships a 1 s trailing-blank rule; this field is what lets the
    #: engine raise a preset to meet it instead of leaving the operator to
    #: discover it from the output.
    min_silence_floor_ms: int = 0
    #: One line on what it is, quoted by ``list_vad`` and by the error a bad
    #: ``backend`` raises.
    doc: str = ""

    @property
    def is_asr_derived(self) -> bool:
        return self.consumes in ASR_CONSUMES

    def can(self, role: str) -> bool:
        """Whether this detector declares ``role``."""
        return role in self.modes

    def framing_for(self, config: "VadConfig") -> VadFraming:
        """This detector's frame grid for ``config``."""
        if self.framing is None:
            raise NotImplementedError(
                f"the {self.kind!r} detector declares no waveform framing; its frame "
                "grid is the encoder's, so ask the running model for it"
            )
        return self.framing(config)


_REGISTRY: Dict[str, VadSpec] = {}

#: Entry-point group for out-of-tree detectors.  ``oasr.features`` and
#: ``oasr.engine.streaming_backend`` do not have one; this axis should, because a
#: third-party VAD model is the obvious plugin.
_ENTRY_POINT_GROUP = "oasr.vad"
_ENTRY_POINTS_LOADED = False


def register_vad(spec: VadSpec) -> VadSpec:
    """Register ``spec`` under its ``kind`` (idempotent; last write wins).

    Validates the declarations against each other here rather than at first use.
    An ASR-derived detector that claims ``"presegment"`` is not a detector with a
    limitation to document — it is a contradiction, since the thing it consumes
    is produced by the stage it claims to precede.
    """
    if spec.consumes not in CONSUMES:
        raise ValueError(
            f"vad {spec.kind!r}: consumes must be one of {list(CONSUMES)}, got {spec.consumes!r}"
        )
    bad_roles = sorted(set(spec.modes) - set(ROLES))
    if bad_roles:
        raise ValueError(f"vad {spec.kind!r}: unknown modes {bad_roles}; valid: {list(ROLES)}")
    if not spec.modes:
        raise ValueError(f"vad {spec.kind!r}: declares no modes, so nothing could ever use it")
    if spec.is_asr_derived and "presegment" in spec.modes:
        raise ValueError(
            f"vad {spec.kind!r} consumes {spec.consumes!r} but claims the 'presegment' "
            "role: an ASR-derived detector reads what the encoder produced, so it "
            "cannot run before the encoder. Declare ('stream',) and/or ('posthoc',)."
        )
    if not spec.is_asr_derived and spec.framing is None:
        raise ValueError(
            f"vad {spec.kind!r} consumes waveform but declares no framing; without it "
            "nothing can convert its output frames to seconds"
        )
    if spec.kind in _REGISTRY:
        logger.debug("Overriding speech detector for %r", spec.kind)
    _REGISTRY[spec.kind] = spec
    return spec


def _ensure_builtins() -> None:
    """Import the built-in detectors so their registration runs.

    Lazy, to avoid an import cycle (each detector module imports this one) and to
    keep torch off the ``import oasr`` path for callers that only want the config
    objects.
    """
    global _ENTRY_POINTS_LOADED
    if not _REGISTRY:
        from . import detectors  # noqa: F401
    if not _ENTRY_POINTS_LOADED:
        _ENTRY_POINTS_LOADED = True
        _load_entry_point_vads()


def _load_entry_point_vads() -> None:
    """Import any out-of-tree detectors advertised on the ``oasr.vad`` group.

    A broken plugin warns and is skipped rather than taking down the process:
    a third-party detector that fails to import should cost its own kind, not
    every other one.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.10 is unsupported anyway
        return
    try:
        found = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - older importlib.metadata signature
        found = entry_points().get(_ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
    for ep in found:
        try:
            ep.load()
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not be fatal
            logger.warning("could not load VAD plugin %r from %r: %s", ep.name, ep.value, exc)


def get_vad_spec(kind: str) -> VadSpec:
    """Resolve a detector's **spec** without constructing it.

    The two-phase lookup matters for the same reason
    ``get_streaming_backend_class`` exists: the engine has to read cheap
    declarations — ``consumes``, ``modes``, ``requires`` — to decide whether a
    configuration is serviceable at all, and that decision has to happen before
    anything allocates.
    """
    _ensure_builtins()
    spec = _REGISTRY.get(kind)
    if spec is None:
        raise NotImplementedError(
            f"No speech detector registered for vad backend={kind!r}. "
            f"Registered: {sorted(_REGISTRY)}. Add one with "
            "oasr.vad.register_vad(VadSpec(...))."
        )
    return spec


def build_detector(
    config: "VadConfig",
    *,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    **kwargs: object,
) -> "SpeechDetector":
    """Construct the detector named by ``config.backend``.

    Extra keyword arguments are passed through to the factory — that is how an
    ASR-derived detector receives the ids it needs (``blank_id``,
    ``no_speech_token_id``) without the registry knowing what they are.
    """
    if config.backend is None:
        raise ValueError(
            "vad backend is unset; the engine resolves 'auto' from the decode "
            "family before calling build_detector"
        )
    spec = get_vad_spec(config.backend)
    return spec.factory(config, device=device, dtype=dtype, **kwargs)


def list_vad() -> List[str]:
    """Names of all registered detectors."""
    _ensure_builtins()
    return sorted(_REGISTRY)


def describe_vad() -> List[Tuple[str, str, str]]:
    """``(kind, consumes, doc)`` for every registered detector, for ``--help``."""
    _ensure_builtins()
    return [(s.kind, s.consumes, s.doc) for s in (_REGISTRY[k] for k in sorted(_REGISTRY))]
