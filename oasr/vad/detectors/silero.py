# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Silero VAD v5 — the neural detector, rebuilt on ``oasr.layers``.

Upstream ships one TorchScript archive (MIT, ~309 K parameters for the 16 kHz
model).  OASR does **not** run that archive: rule 2 says a model is composed from
the layer waist, and the reason applies here more than anywhere.  A detector that
gates the encoder in ``vad.mode="segment"`` runs on every chunk of every stream,
so it wants the same kernel paths, the same dtype policy and the same CUDA-graph
eligibility as everything else — and vendoring a scripted module (or an ONNX
runtime) would put a second inference stack in the process to get none of that.
What it costs is a weight conversion and a parity oracle, both here.

Why Silero and not one of the other candidates: OASR's whole segmentation
vocabulary is already Silero's — ``threshold`` / ``neg_threshold`` /
``min_speech_ms`` / ``min_silence_ms`` / ``speech_pad_ms`` — and both shipped
presets are its numbers (``turn``) and faster-whisper's re-tuning of the *same*
model (``segment``).  The detector the knobs were designed around is the one that
should ship with them.

The architecture, read out of the archive rather than from the paper:

```
512 samples (+64 carried context)
  │  reflect-pad 64 right                       -> 640
  ├─ Conv1d(1 -> 258, k=256, s=128)             -> (4, 258)   "conv-STFT"
  │  magnitude = sqrt(re^2 + im^2)              -> (4, 129)
  ├─ Conv1d(129 -> 128, k=3, s=1, p=1) + ReLU   -> (4, 128)
  ├─ Conv1d(128 ->  64, k=3, s=2, p=1) + ReLU   -> (2,  64)
  ├─ Conv1d( 64 ->  64, k=3, s=2, p=1) + ReLU   -> (1,  64)
  ├─ Conv1d( 64 -> 128, k=3, s=1, p=1) + ReLU   -> (1, 128)
  ├─ LSTM(128 -> 128), one step per 512-sample frame
  └─ ReLU -> Linear(128 -> 1) -> sigmoid        -> p(speech)
```

Two of those lines are deliberate substitutions, and both are exact:

* the "STFT" is a convolution against a fixed basis, so it *is* a ``Conv1d`` —
  the archive stores the basis as a buffer only because upstream builds it in
  ``__init__``;
* the head is a 1x1 convolution over a length-one sequence, which is a linear
  map.  ``Linear`` is the honest layer and it batches every frame of every stream
  into one GEMM instead of one convolution per frame.

Everything before the recurrence is frame-independent, so a call over ``F``
frames runs the front half **once** at ``B*F`` rows and only steps the LSTM ``F``
times.  The 64 samples of left context that upstream keeps inside the module are
carried in :class:`SileroVadState` instead, which is what lets the frame grid
stay a plain ``hop == span == 512`` and the streaming stage's carry rule work
unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from oasr.layers import LSTM, Conv1d, Linear, Relu, Sigmoid

from ..config import VadConfig
from ..detector import SpeechDetector, VadState
from ..registry import VadFraming, VadSpec, register_vad

logger = logging.getLogger(__name__)

__all__ = [
    "SileroVadNet",
    "SileroVadState",
    "SileroDetector",
    "silero_framing",
    "convert_silero_state_dict",
    "load_silero_weights",
    "build_silero",
]

#: Per sample rate: ``(window, context, filter_length, hop_length)`` in samples.
#: The archive carries a 16 kHz and an 8 kHz model; both are supported because
#: both weights are in the file, and refusing one would be a gap invented here
#: rather than one upstream has.
_GEOMETRY: Dict[int, Tuple[int, int, int, int]] = {
    16000: (512, 64, 256, 128),
    8000: (256, 32, 128, 64),
}

#: Hidden width of the encoder output and of the recurrent state.
_HIDDEN = 128

#: Filenames tried, in order, when ``model_dir`` names a directory.
_CANDIDATES = ("silero_vad.jit", "silero_vad.pt", "silero_vad.pth", "model.pt")

#: Frames embedded per slice: bounds the front half's intermediates, about
#: 12 KB a frame (the reflect-padded window, the spectrum and four convs).
_EMBED_FRAMES = 8192

#: Which frames of an ``unfold`` view are real: all (``None``), a row index
#: into a 2-D view, or a ``(rows, frames)`` pair into a 3-D one.
_FrameIndex = Union[None, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _real_frames(lens: torch.Tensor, total: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(row, frame)`` of every real frame, row by row, on ``lens``'s device.

    Expanded on the device from the per-row counts, so a call moves ``B``
    integers instead of building, pinning and copying ``N`` of them.
    """
    device = lens.device
    rows = torch.repeat_interleave(
        torch.arange(lens.numel(), device=device), lens, output_size=total
    )
    firsts = torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens, output_size=total)
    return rows, torch.arange(total, device=device) - firsts


def _stage(host: torch.Tensor, device: torch.device) -> torch.Tensor:
    """A host index on ``device``, copied from pinned memory without a stream drain.

    A pageable copy to the GPU waits for everything queued on the stream, and
    the streaming detector is issued behind the encoder forward.
    """
    if device.type != "cuda":
        return host.to(device)
    return host.pin_memory().to(device, non_blocking=True)


def silero_framing(config: VadConfig) -> VadFraming:
    """The analysis grid, which is fixed by the weights and not configurable.

    ``history`` stays **0** even though each frame reads 64 samples from before
    its own start: that context is carried in the detector's state, exactly as
    upstream carries it in the module.  Declaring it as framing history instead
    would shift every reported boundary by 4 ms and, worse, break the streaming
    stage's rule for how many samples a call consumed.
    """
    rate = int(config.sample_rate)
    if rate not in _GEOMETRY:
        raise ValueError(
            f"silero VAD serves {sorted(_GEOMETRY)} Hz, not {rate}; its window is a "
            "trained constant, so it cannot be resampled onto another grid"
        )
    window = _GEOMETRY[rate][0]
    return VadFraming(span=window, hop=window)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


class SileroVadNet(nn.Module):
    """Silero VAD v5, composed from :mod:`oasr.layers`.

    Parameters
    ----------
    sample_rate : int
        ``16000`` or ``8000``; picks the geometry and the weight set.

    Notes
    -----
    Runs in **float32** and says so rather than following the engine's dtype.
    The magnitude spectrum is a sum of squares of numbers spanning the waveform's
    whole dynamic range, and the model is 1.2 MB — there is nothing to save by
    halving it and a real precision floor to lose.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if sample_rate not in _GEOMETRY:
            raise ValueError(f"sample_rate must be one of {sorted(_GEOMETRY)}, got {sample_rate}")
        window, context, filter_length, hop_length = _GEOMETRY[sample_rate]
        self.sample_rate = int(sample_rate)
        self.window = window
        self.context = context
        self.filter_length = filter_length
        self.hop_length = hop_length
        #: Real bins of the conv-STFT; the basis holds real and imaginary
        #: halves stacked, so the convolution has twice this many outputs.
        self.cutoff = filter_length // 2 + 1
        #: Encoder output width and recurrent state width -- a trained
        #: constant like the geometry above, read through the instance so
        #: every shape in this class comes from one place.
        self.hidden_size = _HIDDEN

        self.stft = Conv1d(
            1,
            2 * self.cutoff,
            filter_length,
            stride=hop_length,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.encoder = nn.ModuleList(
            [
                Conv1d(self.cutoff, 128, 3, stride=1, padding=1, device=device, dtype=dtype),
                Conv1d(128, 64, 3, stride=2, padding=1, device=device, dtype=dtype),
                Conv1d(64, 64, 3, stride=2, padding=1, device=device, dtype=dtype),
                # The last width is the one `embed` reshapes to and the LSTM
                # consumes, so it is the constant above; the 128/64 before it
                # are independent intermediates that merely happen to match.
                Conv1d(64, self.hidden_size, 3, stride=1, padding=1, device=device, dtype=dtype),
            ]
        )
        self.activation = Relu()
        self.lstm = LSTM(
            self.hidden_size, self.hidden_size, batch_first=True, device=device, dtype=dtype
        )
        self.head = Linear(self.hidden_size, 1, device=device, dtype=dtype)
        self.output_activation = Sigmoid()

    # -- pieces --------------------------------------------------------------

    def spectrogram(self, windows: torch.Tensor) -> torch.Tensor:
        """``(N, window + context)`` sample windows → ``(N, T, cutoff)`` magnitudes."""
        x = windows.unsqueeze(1)
        # Right-only reflect pad, upstream's convention.  A symmetric pad would
        # shift the frame grid by half a hop against the weights.
        x = F.pad(x, (0, self.context), mode="reflect")
        spectrum = self.stft(x.transpose(1, 2))  # (N, T, 2 * cutoff)
        real, imag = spectrum[..., : self.cutoff], spectrum[..., self.cutoff :]
        return torch.sqrt(real * real + imag * imag)

    def embed(self, windows: torch.Tensor) -> torch.Tensor:
        """``(N, window + context)`` → ``(N, hidden)``, the pre-recurrent half."""
        hidden = self.spectrogram(windows)
        for conv in self.encoder:
            hidden = self.activation(conv(hidden))
        # The encoder's strides collapse the frame axis to one step per window.
        return hidden.reshape(hidden.size(0), self.hidden_size)

    def embed_frames(self, windows: torch.Tensor, index: _FrameIndex) -> torch.Tensor:
        """``embed`` over the frames of ``windows`` that ``index`` names, in order.

        ``windows`` is an ``unfold`` view, ``(G, width)`` or ``(B, F, width)``,
        which costs nothing until it is read.  ``index`` names the real frames:
        ``None`` (all of them), a ``(N,)`` row index for the 2-D form, or a
        ``(rows, frames)`` pair for the 3-D one.  Gathering only those keeps a
        ragged batch's padding out of the front half -- reshaping the view
        would copy every padded window first -- and gathering in slices bounds
        the intermediates (about 12 KB a frame), so an hour of audio is not one
        1.4 GB spectrogram.
        """
        if index is None:
            flat = windows.reshape(-1, windows.size(-1))
            total = int(flat.size(0))
        else:
            total = int(index[0].numel()) if isinstance(index, tuple) else int(index.numel())

        def gather(start: int, stop: int) -> torch.Tensor:
            if index is None:
                return flat[start:stop]
            if isinstance(index, tuple):
                return windows[index[0][start:stop], index[1][start:stop]]
            return windows.index_select(0, index[start:stop])

        if total <= _EMBED_FRAMES:
            return self.embed(gather(0, total).contiguous())
        out = windows.new_empty(total, self.hidden_size)
        for start in range(0, total, _EMBED_FRAMES):
            stop = min(total, start + _EMBED_FRAMES)
            out[start:stop] = self.embed(gather(start, stop).contiguous())
        return out

    def _recur(
        self,
        sequence: torch.Tensor,
        frame_lengths: Sequence[int],
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step the LSTM per row, for exactly that row's number of frames.

        One recurrence for the whole batch, ragged or not: ``lengths`` packs
        the rows (``oasr.layers.LSTM``), so each row advances its own number of
        frames and ends on its own state.  Grouping rows by frame count and
        running one call per group did the same, but a ragged batch is one
        group per row -- an offline admission batch always is, and so is a
        streaming tick whose streams were fed different amounts -- and then the
        one sequential part of this model ran once per row.
        """
        out, (new_h, new_c) = self.lstm(sequence, (hidden, cell), lengths=frame_lengths)
        return out, new_h, new_c

    def frame_probabilities(
        self,
        windows: torch.Tensor,
        index: _FrameIndex,
        counts: Sequence[int],
        hidden: torch.Tensor,
        cell: torch.Tensor,
        slots: Optional[torch.Tensor] = None,
        final_state: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Speech probability of every real frame, rows concatenated in order.

        The frames ``index`` names in ``windows`` (see :meth:`embed_frames`)
        are row 0's ``counts[0]`` frames, then row 1's, and so on; ``hidden``
        and ``cell`` are ``(1, len(counts), hidden)``.  Returns ``(probs (N,),
        hidden, cell)`` with ``N = sum(counts)``.  Only the recurrence needs a
        ``(row, frame)`` layout, so only it is padded: ``slots`` is where each
        frame sits in it (``row * max(counts) + frame``, on the device), built
        here when the caller has not already.

        ``final_state=False`` is for a caller that discards the state (a whole
        waveform from a zero state): a ragged batch then runs as one plain
        recurrence over the padding.  The recurrence is causal, so every real
        frame's output is exactly its own; only the returned state is not --
        a short row's ran on past its end.  The packed call that keeps it right
        is host-bound in cuDNN (1.1-2.9 ms at 64 rows of 20-200 frames, against
        0.36 ms for the plain one), so it is paid only where the state matters.
        """
        counts = [int(n) for n in counts]
        batch, longest = len(counts), max(counts, default=0)
        embedded = self.embed_frames(windows, index)
        if all(n == longest for n in counts):
            sequence = embedded.view(batch, longest, self.hidden_size)
            outputs, hidden, cell = self._recur(sequence, counts, hidden, cell)
            flat = outputs.reshape(-1, self.hidden_size)
        else:
            if slots is None:
                lens = _stage(torch.tensor(counts, dtype=torch.int64), embedded.device)
                rows, frames = _real_frames(lens, int(embedded.size(0)))
                slots = rows * longest + frames
            sequence = embedded.new_zeros(batch * longest, self.hidden_size)
            sequence.index_copy_(0, slots, embedded)
            outputs, hidden, cell = self._recur(
                sequence.view(batch, longest, self.hidden_size),
                counts if final_state else [longest] * batch,
                hidden,
                cell,
            )
            flat = outputs.reshape(-1, self.hidden_size).index_select(0, slots)
        probs = self.output_activation(self.head(self.activation(flat)))
        return probs.reshape(-1), hidden, cell

    def forward(
        self,
        windows: torch.Tensor,
        frame_lengths: Sequence[int],
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(B, F, window + context)`` windows → ``(B, F)`` speech probabilities.

        ``frame_lengths`` is a **host** sequence: it decides which frames are
        real, which is control flow, and reading it off the device inside the
        model would put a synchronisation in the step loop.  Frames past a
        row's length come back as zero.
        """
        batch, n_frames, _ = windows.shape
        counts = [int(n) for n in frame_lengths]
        if n_frames == 0 or not any(counts):
            return windows.new_zeros(batch, n_frames), hidden, cell
        if all(n == n_frames for n in counts):
            probs, hidden, cell = self.frame_probabilities(windows, None, counts, hidden, cell)
            return probs.view(batch, n_frames), hidden, cell
        # The gather pair, then the recurrence's slots (rows as wide as the
        # longest one) and the output's (as wide as ``windows``).
        lens = _stage(torch.tensor(counts, dtype=torch.int64), windows.device)
        rows, frames = _real_frames(lens, sum(counts))
        flat, hidden, cell = self.frame_probabilities(
            windows, (rows, frames), counts, hidden, cell, slots=rows * max(counts) + frames
        )
        probs = flat.new_zeros(batch * n_frames)
        probs.index_copy_(0, rows * n_frames + frames, flat)
        return probs.view(batch, n_frames), hidden, cell


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def convert_silero_state_dict(state: Dict[str, torch.Tensor], sample_rate: int) -> Dict[str, Any]:
    """Upstream archive parameters → :class:`SileroVadNet` names.

    The archive holds both models under ``_model.`` (16 kHz) and ``_model_8k.``,
    so the rate selects the prefix.  Convolution weights are left in torch's
    ``(out, in, kernel)`` layout on purpose: :class:`oasr.layers.Conv1d` has a
    load hook that transposes them into its native KSC layout, and doing it here
    as well would transpose them twice.
    """
    prefix = "_model." if int(sample_rate) == 16000 else "_model_8k."
    picked = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not picked:
        raise ValueError(
            f"no {prefix!r} parameters in this checkpoint (found "
            f"{sorted({k.split('.')[0] for k in state})}); it does not look like a "
            "Silero VAD archive"
        )
    out: Dict[str, Any] = {
        "stft.weight": picked["stft.forward_basis_buffer"],
        "lstm.weight_ih_l0": picked["decoder.rnn.weight_ih"],
        "lstm.weight_hh_l0": picked["decoder.rnn.weight_hh"],
        "lstm.bias_ih_l0": picked["decoder.rnn.bias_ih"],
        "lstm.bias_hh_l0": picked["decoder.rnn.bias_hh"],
        # A 1x1 convolution over a length-one sequence, as a linear map.
        "head.weight": picked["decoder.decoder.2.weight"].squeeze(-1),
        "head.bias": picked["decoder.decoder.2.bias"],
    }
    for i in range(4):
        out[f"encoder.{i}.weight"] = picked[f"encoder.{i}.reparam_conv.weight"]
        out[f"encoder.{i}.bias"] = picked[f"encoder.{i}.reparam_conv.bias"]
    return out


def _looks_upstream(state: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith(("_model.", "_model_8k.")) for k in state)


def load_silero_weights(model_dir: str, sample_rate: int) -> Dict[str, Any]:
    """Read Silero weights from ``model_dir``, converting the upstream form.

    Accepts the upstream TorchScript archive directly, so the common case is
    "download the file, point at it" with no conversion step — the archive is
    1.2 MB and ``torch.jit.load`` is in the dependency set already.  A path may
    name the file or the directory holding it; a plain ``torch.save`` of either
    the upstream parameters or already-converted ones works too.
    """
    path = os.path.expanduser(str(model_dir))
    if os.path.isdir(path):
        for name in _CANDIDATES:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"no Silero VAD weights in {model_dir!r}: looked for "
                f"{list(_CANDIDATES)}. Fetch the upstream archive with "
                "`curl -L -o silero_vad.jit https://raw.githubusercontent.com/"
                "snakers4/silero-vad/master/src/silero_vad/data/silero_vad.jit`."
            )
    if not os.path.exists(path):
        raise FileNotFoundError(f"vad.model_dir={model_dir!r} does not exist")

    if path.endswith(".jit"):
        state = dict(torch.jit.load(path, map_location="cpu").state_dict())
    else:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        state = dict(loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded)
    if _looks_upstream(state):
        return convert_silero_state_dict(state, sample_rate)
    return state


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


class SileroVadState(VadState):
    """One stream's recurrent state plus its carried left context.

    The context is state, not framing: upstream keeps the previous chunk's last
    64 samples inside the module and prepends them to the next call, so a frame
    still *advances* a whole window.  Holding it here keeps
    ``VadFraming(span=hop)`` honest and keeps the streaming stage's "the call
    consumed ``frames * hop`` samples" rule true.
    """

    __slots__ = ("hidden", "cell", "context")

    def __init__(self, hidden: torch.Tensor, cell: torch.Tensor, context: torch.Tensor) -> None:
        #: ``(1, B, 128)`` each — the layer axis the recurrent waist expects.
        self.hidden = hidden
        self.cell = cell
        #: ``(B, context_samples)`` of audio preceding the next frame.
        self.context = context


class SileroDetector(SpeechDetector):
    """Silero VAD v5 as an OASR speech detector.

    Parameters
    ----------
    config : VadConfig
        ``sample_rate`` selects the geometry; ``model_dir`` says where the
        weights are.
    model_dir : str
        Directory or file holding the weights.  Required — a neural detector
        with random weights would report speech activity that looks like a
        distribution and means nothing, which is worse than refusing.
    """

    kind: ClassVar[str] = "silero"

    def __init__(
        self,
        config: VadConfig,
        *,
        model_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        framing = silero_framing(config)
        # Deliberately **not** the engine's dtype (see SileroVadNet): the
        # magnitude spectrum sums squares across the waveform's whole dynamic
        # range, and there is nothing to save by halving a 1.2 MB model.
        del dtype
        super().__init__(
            seconds_per_frame=framing.seconds_per_frame(config.sample_rate),
            device=torch.device(device) if device is not None else torch.device("cpu"),
            dtype=torch.float32,
        )
        if not model_dir:
            raise ValueError(
                "vad.backend='silero' needs its weights: pass --vad-model-dir (or "
                "VadConfig.model_dir) pointing at the directory holding "
                "silero_vad.jit. Unlike the ASR-derived detectors this one is a "
                "separate model and there is nothing sensible to fall back to."
            )
        self._framing = framing
        self._context = _GEOMETRY[int(config.sample_rate)][1]
        #: :meth:`detect_many`'s reused host buffer and zero pads.
        self._staging: Optional[torch.Tensor] = None
        self._zero_pads: Dict[int, torch.Tensor] = {}
        net = SileroVadNet(int(config.sample_rate))
        net.load_state_dict(load_silero_weights(model_dir, int(config.sample_rate)))
        self._net = net.to(device=self._device, dtype=self._dtype).eval()
        for parameter in self._net.parameters():
            parameter.requires_grad_(False)
        logger.info(
            "silero VAD loaded from %s (%d Hz, %d ms frames)",
            model_dir,
            config.sample_rate,
            round(1000 * self.seconds_per_frame),
        )

    @property
    def framing(self) -> VadFraming:
        return self._framing

    @property
    def net(self) -> SileroVadNet:
        return self._net

    def _zeros(self, samples: int) -> torch.Tensor:
        pad = self._zero_pads.get(samples)
        if pad is None:
            pad = self._zero_pads[samples] = torch.zeros(samples, dtype=self._dtype)
        return pad

    def _host_buffer(self, samples: int) -> torch.Tensor:
        """:meth:`detect_many`'s flat layout on the host, reused across calls.

        Only the first call that needs a size pays for it: a fresh 100 MB
        buffer is mostly page faults.  Reuse is safe because a host call is
        finished when it returns.
        """
        buf = self._staging
        if buf is None or buf.numel() < samples:
            size = max(samples, 2 * (0 if buf is None else buf.numel()))
            buf = self._staging = torch.empty(size, dtype=self._dtype)
        return buf[:samples]

    # -- state ---------------------------------------------------------------

    def new_state(self, batch: int) -> SileroVadState:
        width = self._net.hidden_size
        zeros = torch.zeros(1, batch, width, dtype=self._dtype, device=self._device)
        return SileroVadState(
            zeros,
            zeros.clone(),
            torch.zeros(batch, self._context, dtype=self._dtype, device=self._device),
        )

    def stack_states(self, states: Sequence[Optional[VadState]]) -> SileroVadState:
        rows = [s if isinstance(s, SileroVadState) else self.new_state(1) for s in states]
        return SileroVadState(
            torch.cat([r.hidden for r in rows], dim=1),
            torch.cat([r.cell for r in rows], dim=1),
            torch.cat([r.context for r in rows], dim=0),
        )

    def unstack_states(self, state: Optional[VadState], count: int) -> List[Optional[VadState]]:
        if not isinstance(state, SileroVadState):
            return [self.new_state(1) for _ in range(count)]
        return [
            SileroVadState(
                state.hidden[:, i : i + 1],
                state.cell[:, i : i + 1],
                state.context[i : i + 1],
            )
            for i in range(count)
        ]

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def detect(
        self, waveform: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-shot over a whole waveform, from a zero state.

        Routed through the incremental form rather than duplicating it, so the
        two cannot disagree — which they would otherwise be free to do exactly
        where it is hardest to notice, at a chunk boundary.
        """
        probs, frame_lengths, _ = self.detect_streaming(waveform, lengths, None)
        return probs, frame_lengths

    @torch.no_grad()
    def detect_many(self, waves: Sequence[torch.Tensor]) -> List[List[float]]:
        """Whole waveforms of any lengths, from a zero state, in one call.

        Laid out flat rather than padded: row ``i`` owns
        ``(frames_i + 1) * window`` samples of one buffer, its (zero) left
        context and then its whole frames, so every row's frame ``k`` is the
        same ``unfold`` window, and only real frames are gathered from it.  A
        padded batch of an admission's utterances is mostly padding (the
        longest of 64 random LibriSpeech utterances is ~4x their mean), and
        each padded sample would be copied to the device, cut into windows and
        run through the front half.  The recurrence runs once over the padded
        batch: from a zero state with the state discarded, padding past a row's
        end changes none of its frames (:meth:`SileroVadNet.frame_probabilities`).
        One read-back for the whole call.
        """
        hop, context = self._framing.hop, self._context
        counts = [max(0, (int(w.numel()) - hop) // hop + 1) for w in waves]
        rows: List[List[float]] = [[] for _ in waves]
        live = [i for i, n in enumerate(counts) if n > 0]
        if not live:
            return rows
        lens = [counts[i] for i in live]
        firsts = [0] * len(live)  # each row's first window, in windows of the buffer
        for j in range(1, len(live)):
            firsts[j] = firsts[j - 1] + lens[j - 1] + 1
        # Row by row: its (zero) left context, its whole frames, then zeros to
        # the next window boundary.  On the GPU each row is copied straight
        # into its place, in one call (an engine's request audio is usually
        # page-locked already, so the copies are DMAs with no host pass); on
        # the host, one ``cat`` into a reused buffer.
        samples = (firsts[-1] + lens[-1] + 1) * hop
        sources = [waves[i].reshape(-1)[: counts[i] * hop] for i in live]
        small = torch.tensor(lens + firsts, dtype=torch.int64)
        device = self._device or torch.device("cpu")
        if device.type == "cuda":
            flat = torch.zeros(samples, dtype=self._dtype, device=device)
            places = [
                flat[first * hop + context : first * hop + context + src.numel()]
                for first, src in zip(firsts, sources)
            ]
            torch._foreach_copy_(places, sources, non_blocking=True)
            small = small.pin_memory().to(device, non_blocking=True)
        else:
            lead, tail = self._zeros(context), self._zeros(hop - context)
            parts: List[torch.Tensor] = []
            for src in sources:
                parts += [lead, src, tail]
            flat = torch.cat(parts, out=self._host_buffer(samples))
        windows = flat.unfold(0, context + hop, hop)
        lens_d, firsts_d = small[: len(live)], small[len(live) :]
        total = sum(lens)
        rows_d, frames_d = _real_frames(lens_d, total)
        state = self.new_state(len(live))
        probs, _h, _c = self._net.frame_probabilities(
            windows,
            firsts_d[rows_d] + frames_d,
            lens,
            state.hidden,
            state.cell,
            slots=rows_d * max(lens) + frames_d,
            final_state=False,
        )
        values = probs.to("cpu").tolist()
        offset = 0
        for i, n in zip(live, lens):
            rows[i] = values[offset : offset + n]
            offset += n
        return rows

    @torch.no_grad()
    def detect_streaming(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[VadState],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[VadState]]:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"waveform must be (B, T) or (T,), got {tuple(waveform.shape)}")
        wave = waveform.to(device=self._device, dtype=self._dtype)
        batch, samples = wave.shape
        window = self._framing.hop

        if not isinstance(state, SileroVadState) or state.context.size(0) != batch:
            state = self.new_state(batch)
        assert isinstance(state, SileroVadState)

        lengths_dev = lengths.to(device=self._device, dtype=torch.int64)
        frame_lengths = torch.clamp((lengths_dev - window) // window + 1, min=0)
        n_frames = max(0, (samples - window) // window + 1)
        if n_frames == 0:
            return wave.new_zeros(batch, 0), frame_lengths, state

        # Context first, then the audio: frame k reads [k*window - context,
        # (k+1)*window) of the stream, which after prepending is a plain slide.
        padded = torch.cat([state.context, wave], dim=1)
        windows = padded.unfold(1, self._context + window, window)[:, :n_frames]

        host_lengths = [int(n) for n in frame_lengths.tolist()]
        probs, hidden, cell = self._net(windows, host_lengths, state.hidden, state.cell)

        context = self._next_context(wave, frame_lengths, state.context)
        masked = self._mask_padding(probs, frame_lengths)
        return masked, frame_lengths, SileroVadState(hidden, cell, context)

    def _next_context(
        self, wave: torch.Tensor, frame_lengths: torch.Tensor, previous: torch.Tensor
    ) -> torch.Tensor:
        """The last ``context`` samples each row actually consumed.

        Gathered per row rather than sliced, because rows in one call consume
        different numbers of frames and a single slice would hand the short ones
        audio they have not reached — a left context from the future, which the
        model would happily turn into a probability.
        """
        window, context = self._framing.hop, self._context
        starts = frame_lengths * window - context
        index = starts.clamp(min=0).unsqueeze(1) + torch.arange(
            context, device=wave.device
        ).unsqueeze(0)
        gathered = torch.gather(wave, 1, index.clamp(max=max(0, wave.size(1) - 1)))
        advanced = (frame_lengths > 0).unsqueeze(1)
        return torch.where(advanced, gathered, previous)


def build_silero(
    config: VadConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> SileroDetector:
    """Factory for the registry.  Unknown kwargs are the engine's ASR-only extras."""
    model_dir = kwargs.get("model_dir") or config.model_dir
    return SileroDetector(
        config,
        model_dir=str(model_dir) if model_dir else None,
        device=device,
        dtype=dtype,
    )


register_vad(
    VadSpec(
        kind="silero",
        factory=build_silero,
        consumes="waveform",
        framing=silero_framing,
        # A waveform detector, so it can run ahead of the encoder: ``presegment``
        # drives the offline fan-out and the streaming per-window gate, and
        # ``stream`` says it can do that incrementally.
        modes=("presegment", "stream", "posthoc"),
        # The LSTM's hidden state crosses chunk boundaries, and so do the 64
        # samples of carried left context (see SileroVadState).
        stateful=True,
        # A neural detector with random weights would report an activity trace
        # that looks like a distribution and means nothing, so the engine refuses
        # at construction and names the flag rather than letting it happen.
        needs_weights=True,
        doc="Silero VAD v5 (MIT, 309K params); needs --vad-model-dir",
    )
)
