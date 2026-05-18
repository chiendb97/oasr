"""Feature-extraction (FBANK / MFCC) benchmark routines.

Subroutines
-----------
* ``fbank_preprocess`` — DC-removal + pre-emphasis + windowing + zero-pad.
* ``mel_log``           — mel filterbank + log-floor + log over a power spectrum.
* ``dct_lifter``        — DCT-II + cepstral lifter.
* ``fbank_pipeline``    — full FBANK end-to-end (waveform → log-mel features).
* ``mfcc_pipeline``     — full MFCC end-to-end (waveform → MFCC).

Backends
--------
* ``cuda``       — OASR's CUDA kernel chain (``oasr.feature.*`` + ``oasr.rfft_power``).
* ``torch``      — pure-PyTorch GPU equivalent (vectorized over the batch).
* ``torchaudio`` — per-utterance loop over ``torchaudio.compliance.kaldi``
                    (only for the pipeline subroutines).

Default shapes target standard 16 kHz Kaldi-style FBANK / MFCC
(``frame_length=25 ms``, ``frame_shift=10 ms``, ``n_fft=512``,
``num_mel=80``, ``num_ceps=13``).
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from typing import Any, Callable, Dict, Tuple

import torch

import oasr
from benchmarks.routines.bench_utils import (
    BenchResult,
    OutputWriter,
    bench_fn,
    check_close,
    compute_bandwidth_tb_s,
    dtype_size,
    parse_dtype,
    run_main,
)

SUBROUTINES = [
    "fbank_preprocess",
    "mel_log",
    "dct_lifter",
    "fbank_pipeline",
    "mfcc_pipeline",
]

# ---------------------------------------------------------------------------
# Default configs (16 kHz Kaldi-style FBANK / MFCC)
# ---------------------------------------------------------------------------

# Per-kernel configs: shape the intermediate tensor directly.
# Pipeline configs: shape the input waveform via (batch, audio_seconds).
DEFAULT_CONFIGS: Dict[str, list[dict[str, Any]]] = {
    "fbank_preprocess": [
        {"batch": 32, "num_frames": 250, "frame_length": 400, "n_fft": 512},
        {"batch": 64, "num_frames": 500, "frame_length": 400, "n_fft": 512},
        {"batch": 64, "num_frames": 500, "frame_length": 800, "n_fft": 1024},
    ],
    "mel_log": [
        {"batch": 32, "num_frames": 250, "n_freq": 257, "num_mel": 80},
        {"batch": 64, "num_frames": 500, "n_freq": 257, "num_mel": 80},
        {"batch": 64, "num_frames": 500, "n_freq": 513, "num_mel": 80},
    ],
    "dct_lifter": [
        {"batch": 32, "num_frames": 250, "num_mel": 23, "num_ceps": 13},
        {"batch": 64, "num_frames": 500, "num_mel": 23, "num_ceps": 13},
        {"batch": 64, "num_frames": 500, "num_mel": 80, "num_ceps": 40},
    ],
    "fbank_pipeline": [
        {"batch": 8,  "audio_seconds": 5.0},
        {"batch": 32, "audio_seconds": 5.0},
        {"batch": 64, "audio_seconds": 5.0},
        {"batch": 32, "audio_seconds": 10.0},
    ],
    "mfcc_pipeline": [
        {"batch": 8,  "audio_seconds": 5.0},
        {"batch": 32, "audio_seconds": 5.0},
        {"batch": 64, "audio_seconds": 5.0},
        {"batch": 32, "audio_seconds": 10.0},
    ],
}

PROFILE_CONFIGS: Dict[str, tuple] = {
    "fbank_preprocess": (64, 500, 400, 512),
    "mel_log": (64, 500, 257, 80),
    "dct_lifter": (64, 500, 23, 13),
    "fbank_pipeline": (32, 5.0),
    "mfcc_pipeline": (32, 5.0),
}


def get_default_configs() -> Dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Shared helpers (Kaldi-style mel bank / Povey window — match torchaudio)
# ---------------------------------------------------------------------------


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


@lru_cache(maxsize=8)
def _povey_window(frame_length: int, device_str: str) -> torch.Tensor:
    """Povey window: ``(0.5 - 0.5*cos(2π i/(N-1)))**0.85``."""
    device = torch.device(device_str)
    i = torch.arange(frame_length, device=device, dtype=torch.float32)
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * i / (frame_length - 1))
    return w.pow(0.85)


@lru_cache(maxsize=8)
def _mel_bank(
    num_mel: int,
    n_fft: int,
    sample_rate: int,
    low_freq: float,
    high_freq: float,
    device_str: str,
) -> torch.Tensor:
    """Kaldi-style triangular mel filterbank, shape ``(num_mel, n_fft//2+1)``."""
    device = torch.device(device_str)
    nyquist = 0.5 * sample_rate
    if high_freq <= 0.0:
        high_freq = nyquist + high_freq
    num_bins_fft = n_fft // 2 + 1

    def to_mel(f: torch.Tensor) -> torch.Tensor:
        return 1127.0 * torch.log(1.0 + f / 700.0)

    mel_low = to_mel(torch.tensor(low_freq))
    mel_high = to_mel(torch.tensor(high_freq))
    mel_edges = torch.linspace(mel_low.item(), mel_high.item(), num_mel + 2)
    bin_hz = torch.arange(num_bins_fft, dtype=torch.float32) * (sample_rate / n_fft)
    bin_mel = to_mel(bin_hz)

    left = mel_edges[:-2].unsqueeze(1)
    center = mel_edges[1:-1].unsqueeze(1)
    right = mel_edges[2:].unsqueeze(1)
    bin_mel = bin_mel.unsqueeze(0)
    up = (bin_mel - left) / (center - left)
    down = (right - bin_mel) / (right - center)
    return torch.clamp(torch.minimum(up, down), min=0.0).to(device=device, dtype=torch.float32)


@lru_cache(maxsize=8)
def _dct_matrix(num_ceps: int, num_mel: int, device_str: str) -> torch.Tensor:
    """Kaldi-style orthonormal DCT-II matrix, shape ``(num_ceps, num_mel)``."""
    device = torch.device(device_str)
    n = torch.arange(num_mel, dtype=torch.float32) + 0.5  # (num_mel,)
    k = torch.arange(num_ceps, dtype=torch.float32).unsqueeze(1)  # (num_ceps, 1)
    mat = torch.cos(math.pi / num_mel * k * n) * math.sqrt(2.0 / num_mel)
    mat[0] *= 1.0 / math.sqrt(2.0)
    return mat.to(device=device, dtype=torch.float32)


@lru_cache(maxsize=8)
def _cepstral_lifter(num_ceps: int, lifter: float, device_str: str) -> torch.Tensor:
    """Kaldi cepstral lifter weights, shape ``(num_ceps,)``."""
    device = torch.device(device_str)
    if lifter <= 0:
        return torch.ones(num_ceps, device=device, dtype=torch.float32)
    n = torch.arange(num_ceps, dtype=torch.float32, device=device)
    return 1.0 + 0.5 * lifter * torch.sin(math.pi * n / lifter)


# ---------------------------------------------------------------------------
# Setup: per-kernel benchmarks
# ---------------------------------------------------------------------------


def setup_fbank_preprocess(batch, num_frames, frame_length, n_fft, dtype=torch.float32):
    device = "cuda"
    frames = torch.randn(batch, num_frames, frame_length, device=device, dtype=dtype)
    window = _povey_window(frame_length, device)
    coef = 0.97

    def oasr_fn():
        return oasr.feature.fbank_preprocess(
            frames, window, n_fft=n_fft, preemph_coef=coef
        )

    def torch_fn():
        # Equivalent vectorized batched op chain matching the OASR kernel.
        f = frames - frames.mean(dim=-1, keepdim=True)
        preem = torch.empty_like(f)
        preem[..., 1:] = f[..., 1:] - coef * f[..., :-1]
        preem[..., 0] = f[..., 0] - coef * f[..., 0]
        windowed = preem * window
        if n_fft > frame_length:
            return torch.nn.functional.pad(windowed, (0, n_fft - frame_length))
        return windowed.contiguous()

    return oasr_fn, torch_fn


def setup_mel_log(batch, num_frames, n_freq, num_mel, dtype=torch.float32):
    device = "cuda"
    power = torch.rand(batch, num_frames, n_freq, device=device, dtype=dtype) + 1e-6
    n_fft = (n_freq - 1) * 2
    mel_mat = _mel_bank(num_mel, n_fft, 16000, 20.0, 0.0, device)
    eps = torch.finfo(dtype).tiny

    def oasr_fn():
        return oasr.feature.mel_log(power, mel_mat, log_floor=eps)

    def torch_fn():
        return torch.matmul(power, mel_mat.t()).clamp_min(eps).log()

    return oasr_fn, torch_fn


def setup_dct_lifter(batch, num_frames, num_mel, num_ceps, dtype=torch.float32):
    device = "cuda"
    log_mel = torch.randn(batch, num_frames, num_mel, device=device, dtype=dtype)
    dct = _dct_matrix(num_ceps, num_mel, device)
    lifter = _cepstral_lifter(num_ceps, 22.0, device)

    def oasr_fn():
        return oasr.feature.dct_lifter(log_mel, dct, lifter=lifter)

    def torch_fn():
        return torch.matmul(log_mel, dct.t()) * lifter

    return oasr_fn, torch_fn


# ---------------------------------------------------------------------------
# Setup: end-to-end pipelines
# ---------------------------------------------------------------------------


def _oasr_fbank_pipeline(
    waveforms: torch.Tensor,
    *,
    frame_length: int,
    frame_shift: int,
    n_fft: int,
    window: torch.Tensor,
    mel_mat: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """OASR-kernel chain: unfold → fbank_preprocess → rfft_power → mel_log."""
    frames = waveforms.unfold(-1, frame_length, frame_shift).contiguous()
    preprocessed = oasr.feature.fbank_preprocess(frames, window, n_fft=n_fft)
    power = oasr.rfft_power(preprocessed)
    return oasr.feature.mel_log(power, mel_mat, log_floor=eps)


def _torch_fbank_pipeline(
    waveforms: torch.Tensor,
    *,
    frame_length: int,
    frame_shift: int,
    n_fft: int,
    window: torch.Tensor,
    mel_mat: torch.Tensor,
    preemph: float,
    eps: float,
) -> torch.Tensor:
    """Pure torch batched pipeline (matches batched.batched_fbank)."""
    frames = waveforms.unfold(-1, frame_length, frame_shift)
    f = frames - frames.mean(dim=-1, keepdim=True)
    preem = torch.empty_like(f)
    preem[..., 1:] = f[..., 1:] - preemph * f[..., :-1]
    preem[..., 0] = f[..., 0] - preemph * f[..., 0]
    windowed = preem * window
    if n_fft > frame_length:
        windowed = torch.nn.functional.pad(windowed, (0, n_fft - frame_length))
    spec = torch.fft.rfft(windowed, n=n_fft)
    power = spec.real.pow(2) + spec.imag.pow(2)
    mel = torch.matmul(power, mel_mat.t())
    return torch.log(mel.clamp_min(eps))


def _torchaudio_fbank_loop(waveforms: torch.Tensor, sample_rate: int, num_mel: int) -> torch.Tensor:
    """Per-utterance torchaudio.compliance.kaldi.fbank loop (the legacy default)."""
    import torchaudio

    out = []
    for i in range(waveforms.size(0)):
        out.append(
            torchaudio.compliance.kaldi.fbank(
                waveforms[i:i + 1],
                sample_frequency=float(sample_rate),
                num_mel_bins=num_mel,
                frame_length=25.0,
                frame_shift=10.0,
                dither=0.0,
                energy_floor=0.0,
                preemphasis_coefficient=0.97,
                window_type="povey",
                low_freq=20.0,
                high_freq=0.0,
                snip_edges=True,
            )
        )
    return torch.stack(out, dim=0)


def setup_fbank_pipeline(batch, audio_seconds, dtype=torch.float32):
    device = "cuda"
    sample_rate = 16000
    frame_length = 400
    frame_shift = 160
    n_fft = _next_power_of_two(frame_length)
    num_mel = 80
    preemph = 0.97
    eps = torch.finfo(dtype).tiny

    n_samples = int(audio_seconds * sample_rate)
    waveforms = torch.randn(batch, n_samples, device=device, dtype=dtype)
    window = _povey_window(frame_length, device)
    mel_mat = _mel_bank(num_mel, n_fft, sample_rate, 20.0, 0.0, device)

    def oasr_fn():
        return _oasr_fbank_pipeline(
            waveforms,
            frame_length=frame_length,
            frame_shift=frame_shift,
            n_fft=n_fft,
            window=window,
            mel_mat=mel_mat,
            eps=eps,
        )

    def torch_fn():
        return _torch_fbank_pipeline(
            waveforms,
            frame_length=frame_length,
            frame_shift=frame_shift,
            n_fft=n_fft,
            window=window,
            mel_mat=mel_mat,
            preemph=preemph,
            eps=eps,
        )

    def torchaudio_fn():
        return _torchaudio_fbank_loop(waveforms, sample_rate, num_mel)

    return oasr_fn, torch_fn, torchaudio_fn


def _oasr_mfcc_pipeline(
    waveforms: torch.Tensor,
    *,
    frame_length: int,
    frame_shift: int,
    n_fft: int,
    window: torch.Tensor,
    mel_mat: torch.Tensor,
    dct: torch.Tensor,
    lifter: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    log_mel = _oasr_fbank_pipeline(
        waveforms,
        frame_length=frame_length,
        frame_shift=frame_shift,
        n_fft=n_fft,
        window=window,
        mel_mat=mel_mat,
        eps=eps,
    )
    return oasr.feature.dct_lifter(log_mel, dct, lifter=lifter)


def _torch_mfcc_pipeline(
    waveforms: torch.Tensor,
    *,
    frame_length: int,
    frame_shift: int,
    n_fft: int,
    window: torch.Tensor,
    mel_mat: torch.Tensor,
    dct: torch.Tensor,
    lifter: torch.Tensor,
    preemph: float,
    eps: float,
) -> torch.Tensor:
    log_mel = _torch_fbank_pipeline(
        waveforms,
        frame_length=frame_length,
        frame_shift=frame_shift,
        n_fft=n_fft,
        window=window,
        mel_mat=mel_mat,
        preemph=preemph,
        eps=eps,
    )
    return torch.matmul(log_mel, dct.t()) * lifter


def _torchaudio_mfcc_loop(
    waveforms: torch.Tensor, sample_rate: int, num_mel: int, num_ceps: int
) -> torch.Tensor:
    import torchaudio

    out = []
    for i in range(waveforms.size(0)):
        out.append(
            torchaudio.compliance.kaldi.mfcc(
                waveforms[i:i + 1],
                sample_frequency=float(sample_rate),
                num_mel_bins=num_mel,
                num_ceps=num_ceps,
                cepstral_lifter=22.0,
                frame_length=25.0,
                frame_shift=10.0,
                dither=0.0,
                energy_floor=0.0,
                preemphasis_coefficient=0.97,
                window_type="povey",
                low_freq=20.0,
                high_freq=0.0,
                snip_edges=True,
            )
        )
    return torch.stack(out, dim=0)


def setup_mfcc_pipeline(batch, audio_seconds, dtype=torch.float32):
    device = "cuda"
    sample_rate = 16000
    frame_length = 400
    frame_shift = 160
    n_fft = _next_power_of_two(frame_length)
    num_mel = 23
    num_ceps = 13
    preemph = 0.97
    eps = torch.finfo(dtype).tiny

    n_samples = int(audio_seconds * sample_rate)
    waveforms = torch.randn(batch, n_samples, device=device, dtype=dtype)
    window = _povey_window(frame_length, device)
    mel_mat = _mel_bank(num_mel, n_fft, sample_rate, 20.0, 0.0, device)
    dct = _dct_matrix(num_ceps, num_mel, device)
    lifter = _cepstral_lifter(num_ceps, 22.0, device)

    def oasr_fn():
        return _oasr_mfcc_pipeline(
            waveforms,
            frame_length=frame_length,
            frame_shift=frame_shift,
            n_fft=n_fft,
            window=window,
            mel_mat=mel_mat,
            dct=dct,
            lifter=lifter,
            eps=eps,
        )

    def torch_fn():
        return _torch_mfcc_pipeline(
            waveforms,
            frame_length=frame_length,
            frame_shift=frame_shift,
            n_fft=n_fft,
            window=window,
            mel_mat=mel_mat,
            dct=dct,
            lifter=lifter,
            preemph=preemph,
            eps=eps,
        )

    def torchaudio_fn():
        return _torchaudio_mfcc_loop(waveforms, sample_rate, num_mel, num_ceps)

    return oasr_fn, torch_fn, torchaudio_fn


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------

KERNEL_SETUP = {
    "fbank_preprocess": setup_fbank_preprocess,
    "mel_log": setup_mel_log,
    "dct_lifter": setup_dct_lifter,
}

PIPELINE_SETUP = {
    "fbank_pipeline": setup_fbank_pipeline,
    "mfcc_pipeline": setup_mfcc_pipeline,
}


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--num-frames", type=int, default=None, help="Frames per utterance")
    parser.add_argument("--frame-length", type=int, default=None,
                        help="Frame length in samples (preprocess)")
    parser.add_argument("--n-fft", type=int, default=None, help="FFT length")
    parser.add_argument("--n-freq", type=int, default=None,
                        help="Frequency bin count = n_fft/2+1 (mel_log)")
    parser.add_argument("--num-mel", type=int, default=None, help="Number of mel bins")
    parser.add_argument("--num-ceps", type=int, default=None, help="Number of cepstral coefficients")
    parser.add_argument("--audio-seconds", type=float, default=None,
                        help="Audio duration per utterance (pipeline subroutines)")


# ---------------------------------------------------------------------------
# Bytes accounting (rough — used only for the bandwidth metric)
# ---------------------------------------------------------------------------


def _bytes_accessed(subroutine: str, cfg: dict, dtype: torch.dtype) -> int:
    elem = dtype_size(dtype)
    if subroutine == "fbank_preprocess":
        b, n, l, nf = cfg["batch"], cfg["num_frames"], cfg["frame_length"], cfg["n_fft"]
        return (b * n * l + l + b * n * nf) * elem
    if subroutine == "mel_log":
        b, n, f, m = cfg["batch"], cfg["num_frames"], cfg["n_freq"], cfg["num_mel"]
        return (b * n * f + m * f + b * n * m) * elem
    if subroutine == "dct_lifter":
        b, n, m, c = cfg["batch"], cfg["num_frames"], cfg["num_mel"], cfg["num_ceps"]
        return (b * n * m + c * m + b * n * c) * elem
    if subroutine in ("fbank_pipeline", "mfcc_pipeline"):
        b, secs = cfg["batch"], cfg["audio_seconds"]
        n_samples = int(secs * 16000)
        out_dim = 13 if subroutine == "mfcc_pipeline" else 80
        n_frames = max(0, (n_samples - 400) // 160 + 1)
        return (b * n_samples + b * n_frames * out_dim) * elem
    return 0


def _shape_str(subroutine: str, cfg: dict) -> str:
    if subroutine == "fbank_preprocess":
        return (
            f"[B={cfg['batch']}, N={cfg['num_frames']}, "
            f"L={cfg['frame_length']}, n_fft={cfg['n_fft']}]"
        )
    if subroutine == "mel_log":
        return (
            f"[B={cfg['batch']}, N={cfg['num_frames']}, "
            f"n_freq={cfg['n_freq']}, num_mel={cfg['num_mel']}]"
        )
    if subroutine == "dct_lifter":
        return (
            f"[B={cfg['batch']}, N={cfg['num_frames']}, "
            f"num_mel={cfg['num_mel']}, num_ceps={cfg['num_ceps']}]"
        )
    if subroutine in ("fbank_pipeline", "mfcc_pipeline"):
        return f"[B={cfg['batch']}, {cfg['audio_seconds']}s]"
    return str(cfg)


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


def _setup_for_config(subroutine: str, cfg: dict, dtype: torch.dtype):
    """Return ``(oasr_fn, torch_fn[, torchaudio_fn])`` for *cfg*."""
    if subroutine in KERNEL_SETUP:
        return KERNEL_SETUP[subroutine](**cfg, dtype=dtype)
    if subroutine in PIPELINE_SETUP:
        return PIPELINE_SETUP[subroutine](**cfg, dtype=dtype)
    raise ValueError(f"Unknown subroutine '{subroutine}'")


def get_fn_map(subroutine: str, *fns: Callable) -> Dict[str, Callable]:
    """Map backend name → callable. Pipelines expose the torchaudio backend too."""
    if subroutine in PIPELINE_SETUP:
        oasr_fn, torch_fn, torchaudio_fn = fns
        return {"cuda": oasr_fn, "torch": torch_fn, "torchaudio": torchaudio_fn}
    oasr_fn, torch_fn = fns[:2]
    return {"cuda": oasr_fn, "torch": torch_fn}


# ---------------------------------------------------------------------------
# Refcheck — different reference for kernel vs pipeline subroutines
# ---------------------------------------------------------------------------


def _refcheck(subroutine: str, fn_map: Dict[str, Callable], shape_str: str,
              allow_mismatch: bool, verbose: bool) -> bool:
    """Compare backends. Returns True if all checks pass (or were skipped)."""
    if "cuda" not in fn_map:
        return True

    oasr_out = fn_map["cuda"]()
    if subroutine in KERNEL_SETUP and "torch" in fn_map:
        ref = fn_map["torch"]()
        passed, max_diff = check_close(oasr_out, ref, atol=1e-2, rtol=1e-2)
        if not passed:
            print(f"  [ERROR] {shape_str}: oasr vs torch max_diff={max_diff:.4g}")
            return allow_mismatch
        if verbose:
            print(f"  [OK] {shape_str}: oasr vs torch max_diff={max_diff:.4g}")
        return True

    if subroutine in PIPELINE_SETUP:
        ok = True
        if "torch" in fn_map:
            ref = fn_map["torch"]()
            passed, max_diff = check_close(oasr_out, ref, atol=5e-2, rtol=5e-2)
            tag = "OK" if passed else "ERROR"
            print(f"  [{tag}] {shape_str}: oasr vs torch max_diff={max_diff:.4g}")
            ok = ok and (passed or allow_mismatch)
        if "torchaudio" in fn_map:
            ref = fn_map["torchaudio"]()
            # Pipelines and torchaudio use Kaldi-equivalent ops but different
            # FFT order; tolerate moderate differences.
            passed, max_diff = check_close(oasr_out, ref, atol=5e-2, rtol=5e-2)
            tag = "OK" if passed else "WARN"
            print(f"  [{tag}] {shape_str}: oasr vs torchaudio max_diff={max_diff:.4g}")
            # Don't gate on the torchaudio comparison — it's primarily a
            # performance reference, not a strict bit-for-bit ground truth.
        return ok

    return True


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _resolve_configs(args, subroutine: str) -> list[dict[str, Any]]:
    if subroutine == "fbank_preprocess":
        if all(getattr(args, k, None) is not None
               for k in ("batch", "num_frames", "frame_length", "n_fft")):
            return [{
                "batch": args.batch,
                "num_frames": args.num_frames,
                "frame_length": args.frame_length,
                "n_fft": args.n_fft,
            }]
    elif subroutine == "mel_log":
        if all(getattr(args, k, None) is not None
               for k in ("batch", "num_frames", "n_freq", "num_mel")):
            return [{
                "batch": args.batch,
                "num_frames": args.num_frames,
                "n_freq": args.n_freq,
                "num_mel": args.num_mel,
            }]
    elif subroutine == "dct_lifter":
        if all(getattr(args, k, None) is not None
               for k in ("batch", "num_frames", "num_mel", "num_ceps")):
            return [{
                "batch": args.batch,
                "num_frames": args.num_frames,
                "num_mel": args.num_mel,
                "num_ceps": args.num_ceps,
            }]
    elif subroutine in ("fbank_pipeline", "mfcc_pipeline"):
        if args.batch is not None and args.audio_seconds is not None:
            return [{"batch": args.batch, "audio_seconds": args.audio_seconds}]
    return DEFAULT_CONFIGS.get(subroutine, [])


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", None) or SUBROUTINES[0]
    if subroutine not in SUBROUTINES:
        print(f"[ERROR] Unknown subroutine '{subroutine}'. Available: {SUBROUTINES}")
        return

    dtype_str = getattr(args, "dtype", "float32")
    dtype = parse_dtype(dtype_str)
    if dtype != torch.float32:
        print(f"  [WARNING] feature kernels only support float32, "
              f"got {dtype_str}. Forcing float32.")
        dtype_str = "float32"
        dtype = torch.float32

    do_check = getattr(args, "refcheck", False)
    allow_mismatch = getattr(args, "allow_output_mismatch", False)
    dry_run_iters = getattr(args, "dry_run_iters", 5)
    num_iters = getattr(args, "num_iters", 30)
    use_cuda_events = getattr(args, "use_cuda_events", False)
    verbose = getattr(args, "verbosity", 0) >= 1

    configs = _resolve_configs(args, subroutine)

    for cfg in configs:
        fns = _setup_for_config(subroutine, cfg, dtype)
        fn_map = get_fn_map(subroutine, *fns)
        backends = getattr(args, "backends", None) or list(fn_map.keys())

        bytes_accessed = _bytes_accessed(subroutine, cfg, dtype)
        shape_str = _shape_str(subroutine, cfg)

        if do_check:
            ok = _refcheck(subroutine, fn_map, shape_str, allow_mismatch, verbose)
            if not ok and not allow_mismatch:
                continue

        for backend in backends:
            if backend not in fn_map:
                print(f"  [WARNING] Unknown backend '{backend}', skipping")
                continue
            median_ms, std_ms = bench_fn(
                fn_map[backend],
                dry_run_iters=dry_run_iters,
                num_iters=num_iters,
                use_cuda_events=use_cuda_events,
            )
            bw = compute_bandwidth_tb_s(bytes_accessed, median_ms)
            output.write_result(BenchResult(
                routine="feature",
                subroutine=subroutine,
                backend=backend,
                shape=shape_str,
                dtype=dtype_str,
                median_ms=median_ms,
                std_ms=std_ms,
                bandwidth_tb_s=bw,
            ))


# ---------------------------------------------------------------------------
# Standalone entry — used by bench_extract_feature.py wrapper
# ---------------------------------------------------------------------------


def run_standalone(variants: Tuple[str, ...] = ()) -> None:
    if not variants:
        variants = tuple(SUBROUTINES)

    pcfg = {k: PROFILE_CONFIGS[k] for k in variants if k in PROFILE_CONFIGS}
    setup_funcs = {sub: _make_profile_setup(sub) for sub in variants if sub in PROFILE_CONFIGS}

    def benchmark():
        output = OutputWriter()
        for sub in variants:
            output.write_header(f"{sub} kernel benchmark")
            for cfg in DEFAULT_CONFIGS.get(sub, []):
                fns = _setup_for_config(sub, cfg, torch.float32)
                fn_map = get_fn_map(sub, *fns)
                bytes_accessed = _bytes_accessed(sub, cfg, torch.float32)
                shape_str = _shape_str(sub, cfg)
                for backend, fn in fn_map.items():
                    median_ms, std_ms = bench_fn(fn)
                    bw = compute_bandwidth_tb_s(bytes_accessed, median_ms)
                    output.write_result(BenchResult(
                        routine="feature", subroutine=sub, backend=backend,
                        shape=shape_str, dtype="float32",
                        median_ms=median_ms, std_ms=std_ms,
                        bandwidth_tb_s=bw,
                    ))
        output.finalize()

    run_main("Feature Extraction", pcfg, setup_funcs, benchmark)


def _make_profile_setup(subroutine: str):
    cfg_tuple = PROFILE_CONFIGS[subroutine]

    def _setup():
        if subroutine in PIPELINE_SETUP:
            fns = PIPELINE_SETUP[subroutine](*cfg_tuple)
            return fns[0], fns[1]  # (oasr, torch) — torchaudio omitted from legacy 2-tuple
        return KERNEL_SETUP[subroutine](*cfg_tuple)

    return _setup
