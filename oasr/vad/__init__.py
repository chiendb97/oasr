# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Voice activity detection — the speech-activity extension axis.

Three components, and only the first varies per model:

.. code-block:: text

    SpeechDetector  ->  SpeechSegmenter  ->  Endpointer
    (registry axis)     (shared policy)      (shared policy, streaming)
    p(speech) per       hysteresis over      Kaldi rule disjunction over
    frame, on a         p[t] -> segments     trailing silence + turn state
    declared grid       and events

Writing the last two once is what makes a neural VAD and a CTC blank posterior
produce identical segment semantics, identical knobs and identical events — the
same split :mod:`oasr.features` already uses, where ``ExtractorSpec`` declares
the grid and ``StreamingFraming`` owns the arithmetic for every frontend.

Adding a detector is a subclass plus one :func:`register_vad` call; there is no
engine edit and no ``EngineConfig`` field.  Out-of-tree detectors can arrive
through the ``oasr.vad`` entry-point group instead.
"""

from __future__ import annotations

from .config import (
    DEFAULT_ENDPOINT_RULES,
    MODES,
    PRESETS,
    EndpointRule,
    VadConfig,
)
from .endpointer import (
    END_TIMEOUT,
    START_TIMEOUT,
    EndpointDecision,
    Endpointer,
    rule_names,
)
from .registry import (
    ASR_CONSUMES,
    CONSUMES,
    ROLES,
    VadFraming,
    VadSpec,
    build_detector,
    describe_vad,
    get_vad_spec,
    list_vad,
    register_vad,
)
from .segmenter import (
    SPEECH_STARTED,
    SPEECH_STOPPED,
    SpeechSegment,
    SpeechSegmenter,
    VadEvent,
)

__all__ = [
    # configuration
    "VadConfig",
    "EndpointRule",
    "DEFAULT_ENDPOINT_RULES",
    "MODES",
    "PRESETS",
    # the axis
    "VadSpec",
    "VadFraming",
    "register_vad",
    "get_vad_spec",
    "build_detector",
    "list_vad",
    "describe_vad",
    "CONSUMES",
    "ASR_CONSUMES",
    "ROLES",
    # policy
    "SpeechSegmenter",
    "SpeechSegment",
    "VadEvent",
    "SPEECH_STARTED",
    "SPEECH_STOPPED",
    "Endpointer",
    "EndpointDecision",
    "START_TIMEOUT",
    "END_TIMEOUT",
    "rule_names",
]


def __getattr__(name: str):
    """Expose :class:`SpeechDetector` without importing torch at package import.

    ``oasr.vad.VadConfig`` is reachable from a CPU-only or torch-free context
    (the Rust front-end validates a config object before any engine exists);
    the detector base class is not, because it types its tensors.
    """
    if name in ("SpeechDetector", "VadState", "as_rows"):
        from . import detector as _detector

        return getattr(_detector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
