# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Voice-activity configuration: what to detect with, and how to turn it into turns.

Three groups of knobs, and they come from three different traditions:

* **selection** — ``backend`` / ``model_dir`` / ``mode``, the registry selector and
  the pipeline role.
* **segmentation** — the hysteresis knobs, in the Silero/OpenAI vocabulary
  (``threshold`` / ``neg_threshold`` / ``min_speech_ms`` / ``min_silence_ms`` /
  ``speech_pad_ms`` / ``max_speech_s``).  pyannote, NeMo and Riva spell the same
  six concepts ``onset`` / ``offset`` / ``min_duration_on`` / ``min_duration_off``
  / ``pad_onset`` / ``pad_offset``; the equivalence is in ``docs/vad.md``.  This
  vocabulary is the one OASR's OpenAI-compatible surface already speaks.
* **endpointing** — a rule disjunction in Kaldi's shape
  (:class:`EndpointRule`), with Riva's windowed activity test standing in for
  Kaldi's silence-phone traceback.

**Two presets, not one default.**  The same detector is configured an order of
magnitude differently depending on what it feeds: Silero ships
``min_silence_duration_ms=100, speech_pad_ms=30`` for turn-taking, and
faster-whisper re-tunes the *same model* to ``2000 / 400`` for long-form
pre-segmentation, because aggressive segmentation clips word onsets and costs
WER.  A single default set would be wrong for one of the two uses, so the
unset knobs are filled from a preset chosen by service mode — see
:meth:`VadConfig.resolve`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "EndpointRule",
    "VadConfig",
    "DEFAULT_ENDPOINT_RULES",
    "MODES",
    "PRESETS",
]


@dataclass(frozen=True)
class EndpointRule:
    """One clause of the endpointing disjunction.

    A rule fires when **all** of its conditions hold; the endpointer fires when
    **any** rule does.  This is Kaldi's ``OnlineEndpointRule`` minus
    ``max_relative_cost``, which is deliberately absent rather than
    approximated: it is the gap between the best cost over all active tokens and
    the best cost over tokens that can reach a WFST final state, and no
    end-to-end decoder has one.  Every downstream port (WeNet, sherpa,
    sherpa-onnx, Vosk) drops it for the same reason.  Substituting a
    plausible-looking proxy would endpoint on a number that does not mean what
    the field name says.

    Attributes
    ----------
    must_contain_nonsilence : bool
        Require that something has actually been decoded in this turn.  The
        endpointer derives it the way Kaldi does — ``utterance_length >
        trailing_silence`` — rather than asking the decode strategy, so it holds
        for a family that reports no tokens until it finalizes.
    min_trailing_silence_s : float
        Silence at the end of the turn, in seconds.
    min_utterance_length_s : float
        Total turn length, in seconds.  Non-zero only for the hard cap.
    """

    must_contain_nonsilence: bool
    min_trailing_silence_s: float
    min_utterance_length_s: float = 0.0

    def __post_init__(self) -> None:
        if self.min_trailing_silence_s < 0:
            raise ValueError(
                f"min_trailing_silence_s must be >= 0, got {self.min_trailing_silence_s!r}"
            )
        if self.min_utterance_length_s < 0:
            raise ValueError(
                f"min_utterance_length_s must be >= 0, got {self.min_utterance_length_s!r}"
            )


#: Kaldi's five rules, collapsed to three the way WeNet's ``CtcEndpointConfig``
#: does — rules 2 and 3 there differ only by ``max_relative_cost``, which does
#: not survive the port, so keeping both would give two identical clauses.
#:
#: ``1.0 s`` rather than the ~500 ms hosted APIs converge on: the providers that
#: get away with 500 ms pair the timer with a *turn-confidence* signal (Deepgram
#: Flux ``eot_threshold``, AssemblyAI ``end_of_turn_confidence_threshold``,
#: OpenAI ``semantic_vad``), so their silence threshold is a floor on a confident
#: decision rather than the decision itself.  A pure-silence rule has to be
#: longer or it cuts people off mid-sentence; the open-source consensus for one
#: is 1.0-2.4 s (WeNet 1.0, sherpa-onnx 1.2, Kaldi 2.0 without its cost gate).
DEFAULT_ENDPOINT_RULES: Tuple[EndpointRule, ...] = (
    # Nothing was ever decoded — a stream of pure silence or noise.
    EndpointRule(must_contain_nonsilence=False, min_trailing_silence_s=5.0),
    # The ordinary end of a turn.
    EndpointRule(must_contain_nonsilence=True, min_trailing_silence_s=1.0),
    # Hard cap, so a turn that never pauses still yields.
    EndpointRule(
        must_contain_nonsilence=False, min_trailing_silence_s=0.0, min_utterance_length_s=20.0
    ),
)

#: Pipeline roles.  ``observe`` and ``endpoint`` leave the encoder's input
#: untouched; ``segment`` is the only one that changes what the model sees.
MODES: Tuple[str, ...] = ("off", "observe", "endpoint", "segment")

#: Unset segmentation knobs are filled from one of these.
PRESETS: Dict[str, Dict[str, Any]] = {
    # Silero's own defaults, as used by sherpa-onnx for streaming endpointing.
    "turn": {
        "threshold": 0.5,
        "min_speech_ms": 250,
        "min_silence_ms": 100,
        "speech_pad_ms": 30,
        "max_speech_s": 20.0,
    },
    # faster-whisper's re-tuning of the same knobs for pre-ASR segmentation:
    # long silences before a cut, generous padding around it, and no minimum
    # speech duration (a short segment is still worth transcribing).
    "segment": {
        "threshold": 0.5,
        "min_speech_ms": 0,
        "min_silence_ms": 2000,
        "speech_pad_ms": 400,
        "max_speech_s": 30.0,
    },
}


#: Numeric fields, and how to read one out of a ``--vad-option k=v`` string.
#: Everything absent from this table is a string field and needs no conversion.
_STR_COERCIBLE = {
    "sample_rate": int,
    "frame_ms": float,
    "hop_ms": float,
    "threshold": float,
    "neg_threshold": float,
    "min_speech_ms": int,
    "min_silence_ms": int,
    "speech_pad_ms": int,
    "max_speech_s": float,
    "activity_window_ms": int,
    "activity_threshold": float,
    "speech_start_timeout_s": float,
    "speech_end_timeout_s": float,
}


@dataclass
class VadConfig:
    """Engine-level voice-activity configuration.

    Rides on :class:`~oasr.engine.config.EngineConfig` as a single ``vad`` field,
    the way ``feature_config`` does, rather than a dozen flat fields —
    ``EngineConfig`` had already accumulated nine per-family fields once, and the
    ``options_cls`` axis exists to stop that recurring.

    Every segmentation knob defaults to ``None`` meaning "take the preset", so a
    config that names only ``mode`` is fully specified.

    Attributes
    ----------
    backend : str, optional
        Registry key of the detector (``"energy"``, ``"ctc_blank"``, ...).
        ``None`` means **auto**: the engine picks the ASR-derived detector the
        running decode family declares, and fails at construction if the family
        declares none and the mode needs one.
    model_dir : str, optional
        Checkpoint directory for a detector that has weights.
    mode : str
        One of :data:`MODES`.  ``"off"`` (default) must leave every transcript
        byte-identical — that is the negative control the whole axis rests on.
    device : str, optional
        Where the detector runs; ``None`` follows the engine.
    sample_rate : int
        Rate of the audio the waveform detectors see.  Stamped by the engine
        from ``FeatureConfig.sample_rate``; the engine does not resample, so a
        detector running at any other rate would report spans in a different
        second than the transcript does.
    frame_ms, hop_ms : float
        Analysis geometry for detectors whose framing is configurable (the
        energy baseline).  A detector with a *trained* window — Silero's 512
        samples, MarbleNet's 20 ms — declares its own framing and these are not
        consulted for it.
    preset : str, optional
        ``"turn"`` or ``"segment"``.  ``None`` lets :meth:`resolve` pick from the
        service mode.
    threshold : float, optional
        Probability at or above which a frame enters speech.
    neg_threshold : float, optional
        Probability below which a frame leaves speech.  ``None`` derives
        ``threshold - 0.15`` (floored at 0.01), which is Silero's own rule; the
        gap is what stops a trace hovering at the threshold from chattering.
    min_speech_ms : int, optional
        Speech runs shorter than this are dropped.  In streaming this also
        delays the ``speech_started`` event by that much — the event is stamped
        at the true onset, so the timestamp is unaffected, only its arrival.
    min_silence_ms : int, optional
        Silence shorter than this does not end a run.
    speech_pad_ms : int, optional
        Padding added to each side of an emitted segment.  Inert in ``observe``
        and ``endpoint`` modes, where the encoder sees every sample anyway; it
        becomes load-bearing wherever audio is actually dropped, which is why
        faster-whisper raises it from 30 ms to 400 ms for its pre-ASR path.
    max_speech_s : float, optional
        A run this long is cut even without a silence.
    endpoint_rules : tuple of EndpointRule, optional
        ``None`` uses :data:`DEFAULT_ENDPOINT_RULES`.
    activity_window_ms : int
        Width of the trailing window the endpointer's activity test looks at.
    activity_threshold : float
        Fraction of that window which must be non-silent for the turn's
        trailing-silence counter to reset.  This is Riva's
        ``start_history`` / ``start_threshold`` idea, and it is why the counter
        is robust where WeNet's and sherpa-onnx's plain run-length counters are
        not: one spurious non-blank frame in a pause resets a run-length counter
        to zero, and never resets this one.
    speech_start_timeout_s : float, optional
        Close the stream if speech never begins.  **Cancelled for the rest of the
        stream** once the first speech-start fires — Google's specified
        behaviour, copied because an existing STT client already expects it.
    speech_end_timeout_s : float, optional
        Close the stream this long after speech ends; reset by a new
        speech-start.
    """

    backend: Optional[str] = None
    model_dir: Optional[str] = None
    mode: str = "off"
    device: Optional[str] = None

    # Audio geometry for the waveform detectors.  The engine stamps
    # ``sample_rate`` from the checkpoint's own frontend at construction, because
    # OASR is single-rate by design and a detector running at a different rate
    # than the model would report boundaries in a different second.
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0

    preset: Optional[str] = None
    threshold: Optional[float] = None
    neg_threshold: Optional[float] = None
    min_speech_ms: Optional[int] = None
    min_silence_ms: Optional[int] = None
    speech_pad_ms: Optional[int] = None
    max_speech_s: Optional[float] = None

    endpoint_rules: Optional[Tuple[EndpointRule, ...]] = None
    activity_window_ms: int = 300
    activity_threshold: float = 0.2

    speech_start_timeout_s: Optional[float] = None
    speech_end_timeout_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"vad mode must be one of {list(MODES)}, got {self.mode!r}")
        if self.preset is not None and self.preset not in PRESETS:
            raise ValueError(f"vad preset must be one of {sorted(PRESETS)}, got {self.preset!r}")
        for name in ("threshold", "neg_threshold", "activity_threshold"):
            value = getattr(self, name)
            if value is not None and not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}")
        for name in ("min_speech_ms", "min_silence_ms", "speech_pad_ms", "activity_window_ms"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
        if self.sample_rate <= 0:
            raise ValueError(f"vad sample_rate must be positive, got {self.sample_rate!r}")
        if self.frame_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("vad frame_ms and hop_ms must be positive")
        if self.hop_ms > self.frame_ms:
            raise ValueError(
                f"vad hop_ms ({self.hop_ms}) must be <= frame_ms ({self.frame_ms}); a hop "
                "longer than the window would skip audio between frames"
            )
        if self.max_speech_s is not None and float(self.max_speech_s) <= 0:
            raise ValueError(f"max_speech_s must be > 0 or None, got {self.max_speech_s!r}")
        for name in ("speech_start_timeout_s", "speech_end_timeout_s"):
            value = getattr(self, name)
            if value is None:
                continue
            # Google bounds both at 0.5-60 s.  Below half a second the timer
            # fires inside the segmenter's own hysteresis and the stream closes
            # on a hesitation; above a minute it cannot close a stalled stream
            # before the idle timeout does.
            if not 0.5 <= float(value) <= 60.0:
                raise ValueError(f"{name} must be in [0.5, 60] seconds, got {value!r}")
        # ``backend`` is validated against the registry rather than a literal
        # list, so registering an out-of-tree detector makes its name legal with
        # no edit here — the same rule FeatureConfig applies to feature_type.
        if self.backend is not None:
            from .registry import list_vad

            kinds = list_vad()
            if self.backend not in kinds:
                raise ValueError(f"vad backend must be one of {kinds}, got {self.backend!r}")

    @property
    def speech_pad_seconds(self) -> float:
        """``speech_pad_ms`` in seconds, with the unresolved case as zero.

        Derived here rather than at each of the three use sites -- the
        segmenter, the streaming gate and the frontend-window budget -- so the
        ``or 0`` guard for an unresolved config is stated once.
        """
        return float(self.speech_pad_ms or 0) / 1000.0

    @property
    def enabled(self) -> bool:
        """Whether any VAD work should run at all."""
        return self.mode != "off"

    @property
    def emits_events(self) -> bool:
        """Whether this mode produces speech-activity events."""
        return self.mode in ("observe", "endpoint", "segment")

    @property
    def endpoints(self) -> bool:
        """Whether this mode ends the **request** on detected silence.

        ``"segment"`` is deliberately not in this set even though it also ends
        turns.  The two do different things with the same detection: ``endpoint``
        stops recognising and hands the client its result (Google's
        ``single_utterance``), while ``segment`` closes the turn, resets the
        encoder and keeps going on the same connection.  Folding them together
        would make ``segment`` terminate every stream at its first pause.
        """
        return self.mode == "endpoint"

    @property
    def gates_encoder(self) -> bool:
        """Whether this mode decides which audio the encoder sees.

        True only for ``"segment"``, and it is the line that separates a mode
        that *labels* audio from one that *drops* it: a detector serving this
        mode has to run ahead of the encoder, and its mistakes cost transcript
        rather than metadata.
        """
        return self.mode == "segment"

    def resolve(self, service_mode: str) -> "VadConfig":
        """Return a copy with every optional segmentation knob filled in.

        The preset is chosen from the service mode when it was not named:
        an offline engine is doing pre-ASR segmentation and wants
        faster-whisper's long silences and generous padding, while a streaming
        engine is doing turn-taking and wants Silero's short ones.  Getting this
        from the mode rather than from a single global default is the whole
        point of having two presets.

        ``mode="segment"`` takes the segmentation preset in **either** service
        mode, because it is the mode that drops audio and the padding is what
        keeps a word onset out of the part that gets dropped.
        """
        # The preset follows what the VAD is *for*, not only where it runs:
        # streaming ``segment`` drops audio the encoder never sees, so it wants
        # faster-whisper's long silences and generous padding for exactly the
        # reason faster-whisper does — clipping a word onset costs a word.  An
        # observing or endpointing stream drops nothing and wants Silero's
        # turn-taking numbers.
        wants_segmentation = service_mode == "offline" or self.mode == "segment"
        preset_name = self.preset or ("segment" if wants_segmentation else "turn")
        preset = PRESETS[preset_name]
        threshold = self.threshold if self.threshold is not None else preset["threshold"]
        neg = self.neg_threshold
        if neg is None:
            # Silero's rule.  The floor matters: a threshold below 0.15 would
            # otherwise produce a non-positive exit threshold, and a run that
            # can never exit is a segment that never closes.
            neg = max(float(threshold) - 0.15, 0.01)
        return replace(
            self,
            preset=preset_name,
            threshold=float(threshold),
            neg_threshold=float(neg),
            min_speech_ms=(
                preset["min_speech_ms"] if self.min_speech_ms is None else int(self.min_speech_ms)
            ),
            min_silence_ms=(
                preset["min_silence_ms"]
                if self.min_silence_ms is None
                else int(self.min_silence_ms)
            ),
            speech_pad_ms=(
                preset["speech_pad_ms"] if self.speech_pad_ms is None else int(self.speech_pad_ms)
            ),
            max_speech_s=(
                preset["max_speech_s"] if self.max_speech_s is None else float(self.max_speech_s)
            ),
            endpoint_rules=(
                DEFAULT_ENDPOINT_RULES
                if self.endpoint_rules is None
                else tuple(self.endpoint_rules)
            ),
        )

    @classmethod
    def coerce(cls, value: Any) -> Optional["VadConfig"]:
        """Normalise a config value from a Python or JSON caller.

        Accepts ``None``, an existing :class:`VadConfig`, or a mapping (the Rust
        front-end passes one).  ``None`` values in the mapping mean "unset", so a
        serialized default round-trips to the same object.
        """
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"vad config must be a VadConfig or a dict, got {type(value).__name__}")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown vad config keys {unknown}; valid keys: {sorted(known)}")
        kwargs = {k: v for k, v in value.items() if v is not None}
        rules = kwargs.pop("endpoint_rules", None)
        if rules is not None:
            kwargs["endpoint_rules"] = tuple(
                r if isinstance(r, EndpointRule) else EndpointRule(**r) for r in rules
            )
        # ``--vad-option k=v`` can only carry strings, the same constraint
        # ``--decode-option`` has; type them here from the declared field rather
        # than in the serving crate, which would otherwise need a copy of this
        # table and could drift from it.
        for name, convert in _STR_COERCIBLE.items():
            raw = kwargs.get(name)
            if isinstance(raw, str):
                try:
                    kwargs[name] = convert(raw)
                except ValueError:
                    raise ValueError(
                        f"vad option {name}={raw!r} is not a {convert.__name__}"
                    ) from None
        return cls(**kwargs)
