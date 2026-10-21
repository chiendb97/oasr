# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Calibration: observers, and audio → model through the checkpoint's frontend."""

from .forward_loop import (
    CalibrationSet,
    CheckpointFrontend,
    asr_forward_loop,
    load_waveform,
    no_tf32,
    select_manifest,
)
from .observers import (
    AbsMaxPerChannel,
    AbsMaxPerTensor,
    HistogramPerTensor,
    Observer,
    clip_mse,
    clip_percentile,
)

__all__ = [
    "AbsMaxPerChannel",
    "AbsMaxPerTensor",
    "CalibrationSet",
    "CheckpointFrontend",
    "HistogramPerTensor",
    "Observer",
    "asr_forward_loop",
    "clip_mse",
    "clip_percentile",
    "load_waveform",
    "no_tf32",
    "select_manifest",
]
