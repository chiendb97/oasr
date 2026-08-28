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
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional, Tuple

import torch
import torch.nn.functional as F

from ..config import VadConfig
from ..detector import SpeechDetector

__all__ = [
    "CtcBlankDetector",
    "FrameActivityDetector",
    "CifAlphaDetector",
    "AedNoSpeechDetector",
]


def _int64(lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    return lengths.to(device=device, dtype=torch.int64)


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
        dilate_s: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)  # type: ignore[arg-type]
        self._blank_id = int(blank_id)
        half = max(0, int(round(float(dilate_s) / self.seconds_per_frame)))
        self._width = 2 * half + 1

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
        if self._width > 1 and probs.size(1) >= self._width:
            # A short dilation, for two reasons that are not the same one.  It
            # closes the one- and two-frame blanks *between* the tokens of a
            # word, and it makes the endpointer's windowed activity test see a
            # continuous run rather than a spike train.  It is deliberately far
            # too short to bridge a real pause — that is what the declared
            # ``min_silence_floor_ms`` is for, because bridging 840 ms of
            # in-word blank by dilation alone would smear every boundary by 420.
            probs = F.max_pool1d(
                probs.unsqueeze(1),
                kernel_size=self._width,
                stride=1,
                padding=self._width // 2,
            ).squeeze(1)
        frame_lengths = _int64(lengths, probs.device)
        return self._mask_padding(probs, frame_lengths), frame_lengths


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

    def __init__(self, config: VadConfig, *, dilate_s: float = 0.2, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)  # type: ignore[arg-type]
        half = max(0, int(round(float(dilate_s) / self.seconds_per_frame)))
        self._width = 2 * half + 1

    def detect_from_asr(
        self, tensor: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim != 2:
            raise ValueError(
                f"transducer_blank expects a (B, T) indicator, got {tuple(tensor.shape)}"
            )
        probs = tensor.to(torch.float32).clamp_(0.0, 1.0)
        if self._width > 1 and probs.size(1) >= self._width:
            # Morphological dilation, which is what max-pooling with stride 1 is.
            probs = F.max_pool1d(
                probs.unsqueeze(1),
                kernel_size=self._width,
                stride=1,
                padding=self._width // 2,
            ).squeeze(1)
        frame_lengths = _int64(lengths, probs.device)
        return self._mask_padding(probs, frame_lengths), frame_lengths


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
        smooth_frames: int = 5,
        gain: float = 4.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)  # type: ignore[arg-type]
        width = max(1, int(smooth_frames))
        # Odd, so ``avg_pool1d`` with symmetric padding returns exactly T frames;
        # an even width returns T+1 and every span downstream shifts by half a
        # frame.
        self._width = width if width % 2 == 1 else width + 1
        self._gain = float(gain)

    def detect_from_asr(
        self, tensor: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim != 2:
            raise ValueError(f"cif_alpha expects (B, T) weights, got {tuple(tensor.shape)}")
        alphas = tensor.to(torch.float32).clamp_min(0.0)
        if self._width > 1 and alphas.size(1) >= self._width:
            alphas = F.avg_pool1d(
                alphas.unsqueeze(1),
                kernel_size=self._width,
                stride=1,
                padding=self._width // 2,
                count_include_pad=False,
            ).squeeze(1)
        probs = (alphas * self._gain).clamp_(0.0, 1.0)
        frame_lengths = _int64(lengths, probs.device)
        return self._mask_padding(probs, frame_lengths), frame_lengths


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

    def __init__(self, config: VadConfig, *, no_speech_token_id: int, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)  # type: ignore[arg-type]
        self._token_id = int(no_speech_token_id)

    def detect_from_asr(
        self, tensor: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim != 2:
            raise ValueError(
                f"aed_no_speech expects (B, V) prefill logits, got {tuple(tensor.shape)}"
            )
        if not 0 <= self._token_id < tensor.size(-1):
            raise ValueError(
                f"no_speech_token_id {self._token_id} is outside the vocabulary "
                f"({tensor.size(-1)}); this checkpoint's converter did not record one"
            )
        no_speech = tensor.to(torch.float32).softmax(dim=-1)[:, self._token_id]
        probs = (1.0 - no_speech).clamp_(0.0, 1.0).unsqueeze(1)
        frame_lengths = torch.ones(probs.size(0), dtype=torch.int64, device=probs.device)
        del lengths  # the window is the frame; a per-row length would be 1 either way
        return probs, frame_lengths
