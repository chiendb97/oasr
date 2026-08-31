# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Speech activity read out of the ASR model's own per-frame output.

This is what runs when no separate VAD model is configured, and on this codebase
it is very nearly free: every signal below is already computed by a model that is
running anyway.  The CTC one is *already implemented* — ``GpuStreamingDecoder``
compares ``log_prob[..., blank_id]`` against a threshold every streaming tick,
DMAs the result into pinned host memory, and uses it to skip decoder work under
the name ``is_speech_mask``.

It is also not the consolation prize for streaming.  Deepgram documents the
failure of its own acoustic endpointer plainly — background noise keeps the VAD
hot, so silence never registers and the endpoint never fires — and their fix was
to add a *decoder-derived* signal (``utterance_end_ms``, watching gaps between
finalized word timings) that "works effectively despite background noise".  A
blank posterior is that signal: the acoustic model has already decided the fan
in the room is not a token, so a detector reading its output inherits that
judgement for free.

The limitation is structural and is declared rather than papered over: these
detectors consume what the encoder produced, so they cannot run *before* it.
:func:`~oasr.vad.registry.register_vad` refuses a spec that claims otherwise.
Every kind here declares ``("stream", "posthoc")`` and never ``"presegment"``,
and every one of them declares a ``min_silence_floor_ms``, because the signals
are peaky: fed a preset tuned for a frame-level detector they would shred one
utterance into dozens of segments.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import torch

from oasr.layers import AvgPool1d, MaxPool1d, Softmax

from ..config import VadConfig
from ..detector import SpeechDetector
from ..registry import VadSpec, register_vad

__all__ = [
    "CtcBlankDetector",
    "FrameActivityDetector",
    "CifAlphaDetector",
    "AedNoSpeechDetector",
    "build_ctc_blank",
    "build_transducer_blank",
    "build_cif_alpha",
    "build_aed_no_speech",
]


class _AsrDetector(SpeechDetector):
    """Shared plumbing: an ASR-derived detector is told its frame rate."""

    def __init__(
        self,
        config: VadConfig,
        *,
        seconds_per_frame: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(seconds_per_frame=seconds_per_frame, device=device, dtype=dtype)
        self._cfg = config
        self._width = 1
        self._pool: Optional[MaxPool1d] = None

    def _set_dilation(self, dilate_s: float) -> None:
        """Widen each frame by ``dilate_s`` on both sides, as a frame count."""
        half = max(0, int(round(float(dilate_s) / self.seconds_per_frame)))
        self._width = 2 * half + 1
        # A stride-1 max pool with half-width padding *is* a morphological
        # dilation.  Through the waist, so it takes the OASR kernel wherever
        # the dtype is served and is counted as out-of-scope where it is not --
        # a bare ``F.max_pool1d`` would be neither.
        self._pool = MaxPool1d(self._width, stride=1, padding=self._width // 2)

    def _dilate(self, probs: torch.Tensor) -> torch.Tensor:
        """Widen each frame's value over its neighbours, on the ``(B, T)`` trace."""
        if self._pool is None or self._width <= 1 or probs.size(1) < self._width:
            return probs
        # BTC is the waist's layout, so a per-frame trace is a one-channel
        # sequence -- ``unsqueeze(-1)``, not the NCL ``unsqueeze(1)`` torch
        # wants.  That is the point of the layout: no transpose either side.
        #
        # Annotated rather than returned directly: ``nn.Module.__call__`` is
        # ``Any`` to mypy, so the layer's own return type does not survive it.
        dilated: torch.Tensor = self._pool(probs.unsqueeze(-1))
        return dilated.squeeze(-1)

    def _finish(
        self, probs: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The exit every per-frame kind shares: derive the counts, mask the tail."""
        frame_lengths = lengths.to(device=probs.device, dtype=torch.int64)
        return self._mask_padding(probs, frame_lengths), frame_lengths


class CtcBlankDetector(_AsrDetector):
    """``1 - P(blank)`` per encoder frame.

    The threshold used for *decode skipping* (``GpuDecoderConfig.blank_threshold``,
    0.98) is deliberately not reused here.  It exists to skip decoder work and is
    permissive on purpose; wiring it to endpointing would couple decode-skip
    aggressiveness to turn-detection sensitivity — two knobs with different
    owners and different failure modes.  This detector emits the raw posterior
    and lets ``VadConfig.threshold`` decide.
    """

    kind: ClassVar[str] = "ctc_blank"

    def __init__(
        self,
        config: VadConfig,
        *,
        blank_id: int,
        seconds_per_frame: float,
        dilate_s: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(config, seconds_per_frame=seconds_per_frame, device=device, dtype=dtype)
        self._blank_id = int(blank_id)
        # A short dilation, for two reasons that are not the same one.  It closes
        # the one- and two-frame blanks *between* the tokens of a word, and it
        # makes the endpointer's windowed activity test see a continuous run
        # rather than a spike train.  It is deliberately far too short to bridge
        # a real pause — that is what the declared ``min_silence_floor_ms`` is
        # for, because bridging 840 ms of in-word blank by dilation alone would
        # smear every boundary by 420.
        self._set_dilation(dilate_s)

    def detect_from_asr(
        self, tensor: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim != 3:
            raise ValueError(f"ctc_blank expects (B, T, V) log-probs, got {tuple(tensor.shape)}")
        if not 0 <= self._blank_id < tensor.size(-1):
            raise ValueError(
                f"blank_id {self._blank_id} is outside the vocabulary ({tensor.size(-1)})"
            )
        # fp32 before exp: the head emits in the engine dtype, and exp() of a
        # bf16 log-prob near zero loses most of the resolution that distinguishes
        # a confident blank from a marginal one.
        blank = tensor[..., self._blank_id].to(torch.float32)
        probs = (1.0 - blank.exp()).clamp_(0.0, 1.0)
        return self._finish(self._dilate(probs), lengths)


class FrameActivityDetector(_AsrDetector):
    """A per-frame activity indicator the decode family already recorded.

    The transducer's greedy loop knows, at every encoder frame, whether it
    emitted a label or advanced on blank, and it records the emission frames
    anyway for word timings.  The strategy scatters those into a dense
    ``(B, T)`` indicator; this widens each spike into a run.

    **Why the dilation is not optional.**  An emission indicator is *sparse* —
    one frame per token, so a few frames per second — while the audio between
    two tokens of the same word is obviously still speech.  Fed raw to the
    segmenter it would split inside every word, because the inter-token gap is
    routinely longer than ``min_silence_ms``.  Widening each emission by
    ``dilate_s`` on each side turns the spike train into the run it stands for.

    The cost is real and worth stating: segment boundaries carry roughly
    ``dilate_s`` of slack.  A transducer's boundaries are approximate anyway —
    standard RNN-T training lets the model delay non-blank emission by hundreds
    of milliseconds to exploit future context, which is what FastEmit and
    alignment-restricted training exist to bound — so a detector reading
    emissions inherits that delay whatever it does with it.

    Greedy only.  Under beam search the device-side hypothesis buffer carries
    labels rather than frames, which is the same reason the transducer declares
    no ``word_timing_modes`` under beam.
    """

    kind: ClassVar[str] = "transducer_blank"

    def __init__(
        self,
        config: VadConfig,
        *,
        seconds_per_frame: float,
        dilate_s: float = 0.2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(config, seconds_per_frame=seconds_per_frame, device=device, dtype=dtype)
        self._set_dilation(dilate_s)

    def detect_from_asr(
        self, tensor: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim != 2:
            raise ValueError(
                f"transducer_blank expects a (B, T) indicator, got {tuple(tensor.shape)}"
            )
        probs = tensor.to(torch.float32).clamp_(0.0, 1.0)
        return self._finish(self._dilate(probs), lengths)


class CifAlphaDetector(_AsrDetector):
    """Paraformer's CIF weights, smoothed into an activity rate.

    ``alphas`` is a sigmoid per encoder frame whose sum over an utterance is the
    token count, so it is *not* a speech posterior: a long vowel carries little
    weight per frame while a consonant cluster carries a lot.  What it is, is a
    token **rate**, and a token rate is high in speech and zero in silence.  So
    the frames are boxcar-averaged over roughly a syllable and scaled.

    ``gain`` is a heuristic, not a calibration, and it is the weakest of the four
    ASR-derived signals.  It is exposed rather than hidden because a checkpoint
    whose CIF is differently scaled will need it changed, and a hidden constant
    would make that look like a model bug.
    """

    kind: ClassVar[str] = "cif_alpha"

    def __init__(
        self,
        config: VadConfig,
        *,
        seconds_per_frame: float,
        smooth_frames: int = 5,
        gain: float = 4.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(config, seconds_per_frame=seconds_per_frame, device=device, dtype=dtype)
        width = max(1, int(smooth_frames))
        # Odd, so ``avg_pool1d`` with symmetric padding returns exactly T frames;
        # an even width returns T+1 and every span downstream shifts by half a
        # frame.  Set directly rather than through ``_set_dilation``: this is a
        # smoothing window in frames, not a widening in seconds.
        self._width = width if width % 2 == 1 else width + 1
        self._gain = float(gain)
        self._smooth = AvgPool1d(
            self._width, stride=1, padding=self._width // 2, count_include_pad=False
        )

    def detect_from_asr(
        self, tensor: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim != 2:
            raise ValueError(f"cif_alpha expects (B, T) weights, got {tuple(tensor.shape)}")
        alphas = tensor.to(torch.float32).clamp_min(0.0)
        if self._width > 1 and alphas.size(1) >= self._width:
            alphas = self._smooth(alphas.unsqueeze(-1)).squeeze(-1)
        probs = (alphas * self._gain).clamp_(0.0, 1.0)
        return self._finish(probs, lengths)


class AedNoSpeechDetector(_AsrDetector):
    """Whisper's ``<|nospeech|>`` probability at the first generated position.

    One "frame" spanning the whole decoding window, because that is genuinely all
    this signal covers: Whisper reads it once per 30 s window, so it says nothing
    about silence in the remaining 29 seconds.  Reporting it as a fine-grained
    trace would be inventing resolution the model never produced.

    Worth knowing before trusting it: OpenAI's own gate is ``no_speech_prob >
    0.6`` **and** ``avg_logprob <= -1.0``, and hallucinations on silence are
    typically *high-confidence*, which flips the second condition and emits the
    hallucination anyway.  This detector exposes the number because callers ask
    for it; it is not a segmenter and the ``presegment`` role is unavailable to
    it for a second, independent reason.
    """

    kind: ClassVar[str] = "aed_no_speech"

    def __init__(
        self,
        config: VadConfig,
        *,
        no_speech_token_id: int,
        seconds_per_frame: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(config, seconds_per_frame=seconds_per_frame, device=device, dtype=dtype)
        self._token_id = int(no_speech_token_id)
        self._softmax = Softmax()

    def detect_from_asr(
        self, tensor: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del lengths  # the window is the frame; a per-row length would be 1 either way
        if tensor.ndim != 2:
            raise ValueError(
                f"aed_no_speech expects (B, V) prefill logits, got {tuple(tensor.shape)}"
            )
        if not 0 <= self._token_id < tensor.size(-1):
            raise ValueError(
                f"no_speech_token_id {self._token_id} is outside the vocabulary "
                f"({tensor.size(-1)}); this checkpoint's converter did not record one"
            )
        no_speech = self._softmax(tensor.to(torch.float32))[:, self._token_id]
        probs = (1.0 - no_speech).clamp_(0.0, 1.0).unsqueeze(1)
        frame_lengths = torch.ones(probs.size(0), dtype=torch.int64, device=probs.device)
        return probs, frame_lengths


# ---------------------------------------------------------------------------
# Factories and registration
# ---------------------------------------------------------------------------


def _require(name: str, kind: str, kwargs: Dict[str, Any]) -> Any:
    """Pull a mandatory factory argument, or say which caller failed to supply it."""
    if name not in kwargs or kwargs[name] is None:
        raise ValueError(
            f"the {kind!r} detector needs {name!r}, which the engine supplies from the "
            "running model; it is missing, so this detector was built by hand without it"
        )
    return kwargs[name]


def build_ctc_blank(
    config: VadConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> CtcBlankDetector:
    """Factory for the registry.  Unknown kwargs are other kinds' extras."""
    return CtcBlankDetector(
        config,
        blank_id=int(_require("blank_id", "ctc_blank", kwargs)),
        seconds_per_frame=float(_require("seconds_per_frame", "ctc_blank", kwargs)),
        dilate_s=float(kwargs.get("dilate_s", 0.1)),
        device=device,
        dtype=dtype,
    )


def build_transducer_blank(
    config: VadConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> FrameActivityDetector:
    """Factory for the registry.  Unknown kwargs are other kinds' extras."""
    return FrameActivityDetector(
        config,
        seconds_per_frame=float(_require("seconds_per_frame", "transducer_blank", kwargs)),
        dilate_s=float(kwargs.get("dilate_s", 0.2)),
        device=device,
        dtype=dtype,
    )


def build_cif_alpha(
    config: VadConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> CifAlphaDetector:
    """Factory for the registry.  Unknown kwargs are other kinds' extras."""
    return CifAlphaDetector(
        config,
        seconds_per_frame=float(_require("seconds_per_frame", "cif_alpha", kwargs)),
        smooth_frames=int(kwargs.get("smooth_frames", 5)),
        gain=float(kwargs.get("gain", 4.0)),
        device=device,
        dtype=dtype,
    )


def build_aed_no_speech(
    config: VadConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> AedNoSpeechDetector:
    """Factory for the registry.  Unknown kwargs are other kinds' extras."""
    return AedNoSpeechDetector(
        config,
        no_speech_token_id=int(_require("no_speech_token_id", "aed_no_speech", kwargs)),
        seconds_per_frame=float(_require("seconds_per_frame", "aed_no_speech", kwargs)),
        device=device,
        dtype=dtype,
    )


register_vad(
    VadSpec(
        kind="ctc_blank",
        factory=build_ctc_blank,
        consumes="asr_log_probs",
        modes=("stream", "posthoc"),
        min_silence_floor_ms=1000,
        doc="1 - P(blank) per encoder frame, from the CTC head's own log-probs",
    )
)

register_vad(
    VadSpec(
        kind="transducer_blank",
        factory=build_transducer_blank,
        consumes="asr_frames",
        modes=("stream", "posthoc"),
        min_silence_floor_ms=1000,
        doc="emission-frame activity from the transducer greedy loop (greedy only)",
    )
)

register_vad(
    VadSpec(
        kind="cif_alpha",
        factory=build_cif_alpha,
        consumes="asr_alphas",
        modes=("posthoc",),
        min_silence_floor_ms=500,
        doc="Paraformer CIF token rate, boxcar-smoothed (heuristic gain)",
    )
)

register_vad(
    VadSpec(
        kind="aed_no_speech",
        factory=build_aed_no_speech,
        consumes="asr_prefill_logits",
        modes=("posthoc",),
        doc="Whisper <|nospeech|> probability; one frame per decoding window",
    )
)
