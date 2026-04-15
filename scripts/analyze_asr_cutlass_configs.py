#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
enumerate_asr_configs.py — Enumerate GEMM and CONV2D problem sizes for ASR
model families and query nvMatmulHeuristics for all supported CUTLASS configs.

Supported model families:  conformer, zipformer, branchformer, paraformer, transducer
Supported model sizes:     base, large
Supported CUTLASS targets: CUTLASS2, CUTLASS3 (where supported by nvMatmulHeuristics)

Usage
-----
    # Dry run (no GPU queries) — just enumerate problem sizes:
    python benchmarks/enumerate_asr_configs.py \\
        --families conformer branchformer --sizes base \\
        --batches 1 8 64 --durations 4 16 \\
        --output /tmp/asr_configs.jsonl --dry-run -v

    # Full run with nvMatmulHeuristics queries:
    python benchmarks/enumerate_asr_configs.py \\
        --output results/asr_cutlass_configs.jsonl -vv

    # Single family / GPU subset:
    python benchmarks/enumerate_asr_configs.py \\
        --families conformer --sizes large \\
        --gpus A100_SXM_80GB H100_SXM \\
        --output out.jsonl -v

Approximation strategy
----------------------
Each ASR model is reduced to a deterministic set of GEMM problems derived from
its published hyperparameters.  The key assumptions are:

  * Frame rate: 100 frames/s (10 ms frame shift).
  * Subsampling: 4× for all families except Zipformer, which has per-stack strides
    on top of a global 4× factor.
  * Attention: full MHSA; head_dim = d_model // num_heads.  Q/K/V/O projections
    are separate GEMMs; QK and AV are batched GEMMs (batchSize = batch * num_heads).
  * Feed-forward: two separate GEMMs (expand + contract).  Macaron-style encoders
    have two FF sub-blocks per layer (same problem sizes, distinct op_name labels).
  * Conformer conv module: pointwise-expand (d_model → 2*d_model) and
    pointwise-contract (d_model → d_model) GEMMs.  Depthwise conv1d is not a GEMM.
  * CONV2D: only the input subsampling stack generates 2D convolutions.
    nvMatmulHeuristics covers GEMM only; CONV2D records use status="nmh_unsupported".

References
----------
  * nvMatmulHeuristics Python API:
    https://docs.nvidia.com/cuda/nvidia-matmul-heuristics/api_python.html
  * WeNet Conformer: https://github.com/wenet-e2e/wenet
  * ESPnet Branchformer: https://arxiv.org/abs/2207.02971
  * Zipformer (icefall): https://arxiv.org/abs/2310.11230
  * Paraformer (FunASR): https://arxiv.org/abs/2206.08317
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

log = logging.getLogger("enumerate_asr_configs")

# ─────────────────────────────────────────────────────────────────────────────
# nvMatmulHeuristics import
# ─────────────────────────────────────────────────────────────────────────────

try:
    from nvMatmulHeuristics import (
        NvMatmulHeuristicsInterface,
        NvMatmulHeuristicsNvidiaGpu,
        NvMatmulHeuristicsTarget,
        boolsToNvMatmulHeuristicsLayout,
    )
    NMH_AVAILABLE = True
except ImportError:
    NMH_AVAILABLE = False
    log.warning(
        "nvMatmulHeuristics not importable (install nvidia-matmul-heuristics). "
        "Use --dry-run to skip GPU queries."
    )

# ─────────────────────────────────────────────────────────────────────────────
# GPU target definitions
# ─────────────────────────────────────────────────────────────────────────────

# Map GPU label (used on CLI) → (NvMatmulHeuristicsNvidiaGpu enum value, SM string)
# Extend this table when new GPUs are added to the enum.
GPU_TABLE: Dict[str, Tuple[Any, str]] = {}
if NMH_AVAILABLE:
    GPU_TABLE = {
        # Ampere (SM80 / SM86)
        "A100_SXM_80GB":  (NvMatmulHeuristicsNvidiaGpu.A100_SXM_80GB, "sm_80"),
        "A100_PCIE_80GB": (NvMatmulHeuristicsNvidiaGpu.A100_PCIE_80GB, "sm_80"),
        "A30_PCIE":       (NvMatmulHeuristicsNvidiaGpu.A30_PCIE,       "sm_80"),
        "A10_PCIE":       (NvMatmulHeuristicsNvidiaGpu.A10_PCIE,       "sm_86"),
        "A40_PCIE":       (NvMatmulHeuristicsNvidiaGpu.A40_PCIE,       "sm_86"),
        "RTX_3090":       (NvMatmulHeuristicsNvidiaGpu.RTX_3090,       "sm_86"),
        "RTX_A6000":      (NvMatmulHeuristicsNvidiaGpu.RTX_A6000,      "sm_86"),
        # Ada (SM89)
        "L20":            (NvMatmulHeuristicsNvidiaGpu.L20,            "sm_89"),
        "L40":            (NvMatmulHeuristicsNvidiaGpu.L40,            "sm_89"),
        "L40S":           (NvMatmulHeuristicsNvidiaGpu.L40S,           "sm_89"),
        "L4":             (NvMatmulHeuristicsNvidiaGpu.L4,             "sm_89"),
        "RTX_4090":       (NvMatmulHeuristicsNvidiaGpu.RTX_4090,       "sm_89"),
        "RTX_6000_ADA":   (NvMatmulHeuristicsNvidiaGpu.RTX_6000_ADA,  "sm_89"),
        # Hopper (SM90)
        "H100_SXM":       (NvMatmulHeuristicsNvidiaGpu.H100_SXM,      "sm_90"),
        "H100_PCIE":      (NvMatmulHeuristicsNvidiaGpu.H100_PCIE,     "sm_90"),
        "H100_NVL":       (NvMatmulHeuristicsNvidiaGpu.H100_NVL,      "sm_90"),
        "H200_SXM":       (NvMatmulHeuristicsNvidiaGpu.H200_SXM,      "sm_90"),
        "H20_SXM":        (NvMatmulHeuristicsNvidiaGpu.H20_SXM,       "sm_90"),
        # Blackwell (SM100 / SM120)
        "B200":           (NvMatmulHeuristicsNvidiaGpu.B200,           "sm_100"),
        "GB200_NVL":      (NvMatmulHeuristicsNvidiaGpu.GB200_NVL,     "sm_100"),
        "GB300_NVL":      (NvMatmulHeuristicsNvidiaGpu.GB300_NVL,     "sm_100"),
        "RTX_5080":       (NvMatmulHeuristicsNvidiaGpu.RTX_5080,      "sm_120"),
        "RTX_5090":       (NvMatmulHeuristicsNvidiaGpu.RTX_5090,      "sm_120"),
        "RTX_PRO_6000":   (NvMatmulHeuristicsNvidiaGpu.RTX_PRO_6000,  "sm_120"),
    }

# Default GPU set: one representative per SM generation (data-center focused)
DEFAULT_GPUS = [
    "A100_SXM_80GB",   # SM80
    "A10_PCIE",        # SM86
    "L40S",            # SM89
    "H100_SXM",        # SM90
    "H200_SXM",        # SM90
    "B200",            # SM100
]

# CUTLASS targets to query: (label, NvMatmulHeuristicsTarget enum)
CUTLASS_TARGETS: List[Tuple[str, Any]] = []
if NMH_AVAILABLE:
    CUTLASS_TARGETS = [
        ("cutlass2", NvMatmulHeuristicsTarget.CUTLASS),
        ("cutlass3", NvMatmulHeuristicsTarget.CUTLASS3),
    ]

# Precision string → dtype label mapping.
# Format: A_dtype / B_dtype / C_dtype (H=FP16, B=BF16, S=FP32, I=INT8).
# We query both FP32-accumulate and same-dtype-accumulate variants.
DTYPE_PRECISIONS: Dict[str, List[str]] = {
    "float16":  ["HSS", "HHS"],   # FP16 inputs: FP32-acc and FP16-acc
    "bfloat16": ["BSS"],          # BF16 inputs: FP32-acc
}

# Maximum number of configs to request per query.  The API returns ≤ this many.
MAX_CONFIGS_PER_QUERY = 512


# ═════════════════════════════════════════════════════════════════════════════
# Problem size dataclasses
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GemmProblem:
    """
    A single GEMM: C[M×N] = A[M×K] × B[K×N].

    batch=1 → standard GEMM.  batch>1 → batched GEMM (e.g. attention QK/AV).
    transA / transB reflect the memory layout of the operands.
    """
    M: int
    N: int
    K: int
    batch: int = 1
    transA: bool = False    # False = row-major A (most linear layers)
    transB: bool = True     # True = weight matrix stored transposed (PyTorch convention)
    op_name: str = ""       # human-readable label for traceability


@dataclass(frozen=True)
class Conv2dProblem:
    """A single CONV2D in NCHW layout (N=batch, C=in-ch, H=time, W=freq)."""
    N: int      # batch
    C: int      # input channels
    H: int      # input height (time frames)
    W: int      # input width  (frequency bins)
    K: int      # output channels
    R: int      # kernel height
    S: int      # kernel width
    pad_h: int = 1
    pad_w: int = 1
    stride_h: int = 1
    stride_w: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    op_name: str = ""


@dataclass
class ConfigRecord:
    """One output row written to the JSONL file."""
    # ── model identity
    model_family: str
    model_size: str
    batch_size: int
    audio_duration_sec: int
    # ── op identity
    op_type: str              # "gemm" | "conv2d"
    function_name: str        # e.g. "ff1_expand", "attn_qk"
    problem_size: Dict[str, Any]
    dtype: str
    precision: str            # nvMatmulHeuristics precision string (e.g. "HSS")
    # ── GPU / CUTLASS
    gpu_target: str           # human-readable GPU label (e.g. "A100_SXM_80GB")
    gpu_arch: str             # SM string (e.g. "sm_80")
    cutlass_family: str       # "cutlass2" | "cutlass3" | "dry_run" | "nmh_unsupported"
    cutlass_config: Dict[str, Any]
    estimated_runtime_s: float = 0.0
    # ── extra metadata
    extra_meta: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"   # success | unsupported | no_config | error | dry_run
    error_msg: str = ""


# ═════════════════════════════════════════════════════════════════════════════
# Model architecture definitions
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ConvSubsamplingSpec:
    """A stack of Conv2D layers applied to the raw (B, 1, T, mel_bins) spectrogram."""

    # Each tuple: (in_ch, out_ch, kernel_h, kernel_w, stride_h, stride_w, pad)
    layers: List[Tuple[int, int, int, int, int, int, int]]

    @property
    def time_reduction(self) -> int:
        """Total time downsampling factor (product of all stride_h values)."""
        factor = 1
        for _, _, _, _, sh, _sw, _ in self.layers:
            factor *= sh
        return factor


# Standard 2×stride-2 Conv2D subsampling used by Conformer / Paraformer / Branchformer.
# Input:  (B, 1, T, 80)
# After layer 1: (B, C, T//2, 40)
# After layer 2: (B, C, T//4, 20)
def _conv2d_subsampling(ch: int) -> ConvSubsamplingSpec:
    return ConvSubsamplingSpec(layers=[
        (1,  ch, 3, 3, 2, 2, 1),
        (ch, ch, 3, 3, 2, 2, 1),
    ])


@dataclass
class EncoderConfig:
    """
    Unified per-stack encoder configuration.

    Fields
    ------
    d_model           Hidden dimension.
    num_layers        Number of encoder layers in this stack.
    num_heads         Number of attention heads.
    ff_units          Feed-forward inner dimension.
    cnn_kernel        Kernel size of the depthwise conv1d (not a GEMM).
    macaron           If True, two half-FF blocks per layer instead of one.
    has_conv_module   If True, include pointwise-expand and pointwise-contract GEMMs.
    skip_ff           If True, skip FF GEMM generation (Branchformer uses cgMLP branch).
    subsampling       Conv2D subsampling spec applied before this encoder stack.
    global_subsampling  Non-Conv2D time reduction factor applied before this encoder
                      (e.g. LFR=6 for Paraformer).  Ignored when subsampling is set.
    qk_head_dim       Per-head dimension for Q and K projections.  When None, defaults
                      to d_model // num_heads (standard attention).  Set for Zipformer
                      whose Q/K dim is decoupled from d_model (icefall default: 32).
    v_head_dim        Per-head dimension for the V projection and the output projection
                      input.  When None, defaults to d_model // num_heads.  Set for
                      Zipformer (icefall default: 12).
    frame_rate        Input frame rate in frames per second.
    mel_bins          Number of input mel-filterbank bins.
    """
    d_model: int
    num_layers: int
    num_heads: int
    ff_units: int
    cnn_kernel: int
    macaron: bool = True
    has_conv_module: bool = True
    skip_ff: bool = False
    subsampling: Optional[ConvSubsamplingSpec] = None
    global_subsampling: int = 1
    qk_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    frame_rate: int = 100
    mel_bins: int = 80


@dataclass
class ModelSpec:
    """Container for a complete ASR model's sub-network configurations."""
    name: str
    size: str
    encoder: EncoderConfig

    # Optional attention decoder (Paraformer NAR decoder)
    decoder_d_model: Optional[int] = None
    decoder_num_layers: Optional[int] = None
    decoder_num_heads: Optional[int] = None
    decoder_ff_units: Optional[int] = None

    # Optional RNNT predictor + joiner
    predictor_dim: Optional[int] = None
    predictor_num_layers: Optional[int] = None
    joiner_dim: Optional[int] = None

    # Shared output vocabulary
    vocab_size: Optional[int] = None

    # Branchformer cgMLP branch
    cgmlp_units: Optional[int] = None

    # Zipformer: list of per-stack encoder configs + time strides
    zipformer_stacks: Optional[List[EncoderConfig]] = None
    zipformer_strides: Optional[List[int]] = None


# ── Conformer ─────────────────────────────────────────────────────────────────
# Base: WeNet AISHELL train_conformer.yaml (wenet-e2e/wenet).
#   output_size=256, num_blocks=12, attention_heads=4, linear_units=2048,
#   cnn_module_kernel=15, input_layer=conv2d, macaron_style implicit (Conformer default).
# Large: Conformer paper Table 1 "Conformer-L" (arXiv:2005.08100, INTERSPEECH 2020).
#   encoder_dim=512, num_layers=17, attention_heads=8, conv_kernel_size=32.
#   WeNet does not publish a public "large" AISHELL config; paper values are used.

CONFORMER_BASE = ModelSpec(
    name="conformer", size="base",
    encoder=EncoderConfig(
        d_model=256, num_layers=12, num_heads=4, ff_units=2048,
        cnn_kernel=15, macaron=True, has_conv_module=True,
        subsampling=_conv2d_subsampling(256),
    ),
    vocab_size=5000,
)

CONFORMER_LARGE = ModelSpec(
    name="conformer", size="large",
    encoder=EncoderConfig(
        d_model=512, num_layers=17, num_heads=8, ff_units=2048,
        cnn_kernel=32, macaron=True, has_conv_module=True,
        subsampling=_conv2d_subsampling(512),
    ),
    vocab_size=5000,
)

# ── Branchformer ──────────────────────────────────────────────────────────────
# WeNet AISHELL train_u2++_branchformer.yaml (wenet-e2e/wenet).
#   output_size=256, num_blocks=24, attention_heads=4, cgmlp_linear_units=2048,
#   cgmlp_conv_kernel=63, merge_method=concat (Linear(2D→D) merge).
# Each block has two parallel branches: MHSA and cgMLP; no separate FF module
# (skip_ff=True).  The cgMLP depthwise conv kernel=63 (not a GEMM, not enumerated).
# Large: no authoritative public config found; scaled to d_model=512 keeping all
#   other counts identical to base (num_blocks=24, cgmlp_conv_kernel=63).

BRANCHFORMER_BASE = ModelSpec(
    name="branchformer", size="base",
    encoder=EncoderConfig(
        d_model=256, num_layers=24, num_heads=4, ff_units=2048,
        cnn_kernel=63, macaron=False, has_conv_module=False, skip_ff=True,
        subsampling=_conv2d_subsampling(256),
    ),
    cgmlp_units=2048,
    vocab_size=5000,
)

BRANCHFORMER_LARGE = ModelSpec(
    name="branchformer", size="large",
    encoder=EncoderConfig(
        d_model=512, num_layers=24, num_heads=8, ff_units=3072,
        cnn_kernel=63, macaron=False, has_conv_module=False, skip_ff=True,
        subsampling=_conv2d_subsampling(512),
    ),
    cgmlp_units=3072,
    vocab_size=5000,
)

# ── Paraformer ────────────────────────────────────────────────────────────────
# FunASR / ModelScope SANMEncoder + ParaformerSANMDecoder.
#
# Large: iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
#   (fetched config.yaml from ModelScope).
#   encoder: SANMEncoder output_size=512, num_blocks=50, attention_heads=4,
#            linear_units=2048, kernel_size=11, input_layer=pe (no Conv2D).
#   decoder: ParaformerSANMDecoder, num_blocks=16, attention_heads=4,
#            linear_units=2048.
#   Frontend: LFR (Low Frame Rate) stacking m=7, n=6 → T_prime ≈ T // 6.
#             Represented as global_subsampling=6 (no Conv2D emitted).
#   Reference: arXiv:2206.08317; ModelScope model card.
#
# Base: no public config.yaml found; values are approximated by scaling the
#   large config down (256 dim, 12 blocks, 6-layer decoder).
#   All non-size parameters (SANM kernel=11, LFR=6, no Conv2D) match the large.

PARAFORMER_BASE = ModelSpec(
    name="paraformer", size="base",
    encoder=EncoderConfig(
        d_model=256, num_layers=12, num_heads=4, ff_units=2048,
        cnn_kernel=11, macaron=False, has_conv_module=False,
        subsampling=None, global_subsampling=6,
    ),
    decoder_d_model=256, decoder_num_layers=6,
    decoder_num_heads=4, decoder_ff_units=2048,
    vocab_size=8404,
)

PARAFORMER_LARGE = ModelSpec(
    name="paraformer", size="large",
    encoder=EncoderConfig(
        d_model=512, num_layers=50, num_heads=4, ff_units=2048,
        cnn_kernel=11, macaron=False, has_conv_module=False,
        subsampling=None, global_subsampling=6,
    ),
    decoder_d_model=512, decoder_num_layers=16,
    decoder_num_heads=4, decoder_ff_units=2048,
    vocab_size=8404,
)

# ── Transducer (RNNT) ─────────────────────────────────────────────────────────
# Conformer encoder + LSTM predictor + linear joiner.
# Encoder: identical to the corresponding Conformer base/large (verified against
#   WeNet AISHELL train_conformer.yaml for base; Conformer paper Table 1 for large).
# Predictor / joiner: no public WeNet transducer config found (train_transducer.yaml
#   does not exist in the main branch).  icefall pruned_transducer_stateless7 uses
#   a stateless (embedding-only) predictor with --decoder-dim=512, --joiner-dim=512;
#   these values are adopted here as representative defaults for LSTM-based RNNT.
# Reference: arXiv:1211.3711; icefall egs/librispeech/ASR/pruned_transducer_stateless7.

TRANSDUCER_BASE = ModelSpec(
    name="transducer", size="base",
    encoder=EncoderConfig(
        d_model=256, num_layers=12, num_heads=4, ff_units=2048,
        cnn_kernel=15, macaron=True, has_conv_module=True,
        subsampling=_conv2d_subsampling(256),
    ),
    predictor_dim=512, predictor_num_layers=2,
    joiner_dim=512, vocab_size=5000,
)

TRANSDUCER_LARGE = ModelSpec(
    name="transducer", size="large",
    encoder=EncoderConfig(
        d_model=512, num_layers=17, num_heads=8, ff_units=2048,
        cnn_kernel=32, macaron=True, has_conv_module=True,
        subsampling=_conv2d_subsampling(512),
    ),
    predictor_dim=1024, predictor_num_layers=2,
    joiner_dim=1024, vocab_size=5000,
)

# ── Zipformer ─────────────────────────────────────────────────────────────────
# k2/icefall Zipformer-M (base) and Zipformer-L (large).
# Multi-stack encoder with global 4× Conv1D subsampling (no Conv2D mel subsampling).
# Per-stack sequence length = ceil(T_global / stride), T_global = ceil(T / 4).
# Reference: arXiv:2310.11230; icefall egs/librispeech/ASR/zipformer/train.py.
#
# Column order: (d_model, num_heads, ff_units, num_layers, time_stride, cnn_kernel)
#
# Base = Zipformer-M (icefall train.py argparse defaults, verified):
#   --encoder-dim       "192,256,384,512,384,256"
#   --feedforward-dim   "512,768,1024,1536,1024,768"
#   --num-encoder-layers "2,2,3,4,3,2"
#   --num-heads         "4,4,4,8,4,4"
#   --downsampling-factor "1,2,4,8,4,2"
#   --cnn-module-kernel "31,31,15,15,15,31"
#
# Large = Zipformer-L (icefall run.sh large variant, derived from known recipes):
#   --encoder-dim       "192,256,512,768,512,256"
#   --feedforward-dim   "512,768,1536,2048,1536,768"
#   --num-encoder-layers "2,2,4,5,4,2"
#   --num-heads         "4,4,4,8,4,4"     (same as medium)
#   --downsampling-factor "1,2,4,8,4,2"   (same as medium)
#   --cnn-module-kernel "31,31,15,15,15,31" (same as medium)
#
# Zipformer uses decoupled Q/K and V head dimensions (icefall defaults):
#   --query-head-dim 32   (qk_head_dim)
#   --value-head-dim 12   (v_head_dim)
# These differ from standard d_model // num_heads and materially change the
# attention Q/K/V projection and BMM problem sizes.

_ZM_STACKS = [
    # (d_model, num_heads, ff_units, num_layers, time_stride, cnn_kernel)
    (192, 4, 512,  2, 1, 31),
    (256, 4, 768,  2, 2, 31),
    (384, 4, 1024, 3, 4, 15),
    (512, 8, 1536, 4, 8, 15),
    (384, 4, 1024, 3, 4, 15),
    (256, 4, 768,  2, 2, 31),
]

_ZL_STACKS = [
    (192, 4, 512,  2, 1, 31),
    (256, 4, 768,  2, 2, 31),
    (512, 4, 1536, 4, 4, 15),
    (768, 8, 2048, 5, 8, 15),
    (512, 4, 1536, 4, 4, 15),
    (256, 4, 768,  2, 2, 31),
]

# Zipformer Q/K and V head dims (icefall --query-head-dim / --value-head-dim defaults)
_ZF_QK_HEAD_DIM = 32
_ZF_V_HEAD_DIM  = 12


def _zipformer_stacks(table) -> Tuple[List[EncoderConfig], List[int]]:
    confs = [
        EncoderConfig(
            d_model=dm, num_heads=nh, ff_units=ff, num_layers=nl,
            cnn_kernel=ck, macaron=False, has_conv_module=True,
            subsampling=None,
            qk_head_dim=_ZF_QK_HEAD_DIM,
            v_head_dim=_ZF_V_HEAD_DIM,
        )
        for dm, nh, ff, nl, _, ck in table
    ]
    strides = [s for _, _, _, _, s, _ in table]
    return confs, strides


_ZM_CONFS, _ZM_STRIDES = _zipformer_stacks(_ZM_STACKS)
_ZL_CONFS, _ZL_STRIDES = _zipformer_stacks(_ZL_STACKS)

# Use the first stack as the representative "encoder" for field access;
# actual GEMM generation iterates zipformer_stacks / zipformer_strides.
ZIPFORMER_BASE = ModelSpec(
    name="zipformer", size="base",
    encoder=_ZM_CONFS[0],
    zipformer_stacks=_ZM_CONFS,
    zipformer_strides=_ZM_STRIDES,
    vocab_size=500,
)

ZIPFORMER_LARGE = ModelSpec(
    name="zipformer", size="large",
    encoder=_ZL_CONFS[0],
    zipformer_stacks=_ZL_CONFS,
    zipformer_strides=_ZL_STRIDES,
    vocab_size=500,
)

# Registry: (family, size) → ModelSpec
MODEL_REGISTRY: Dict[Tuple[str, str], ModelSpec] = {
    ("conformer",    "base"):  CONFORMER_BASE,
    ("conformer",    "large"): CONFORMER_LARGE,
    ("branchformer", "base"):  BRANCHFORMER_BASE,
    ("branchformer", "large"): BRANCHFORMER_LARGE,
    ("paraformer",   "base"):  PARAFORMER_BASE,
    ("paraformer",   "large"): PARAFORMER_LARGE,
    ("transducer",   "base"):  TRANSDUCER_BASE,
    ("transducer",   "large"): TRANSDUCER_LARGE,
    ("zipformer",    "base"):  ZIPFORMER_BASE,
    ("zipformer",    "large"): ZIPFORMER_LARGE,
}


# ═════════════════════════════════════════════════════════════════════════════
# Problem size derivation
# ═════════════════════════════════════════════════════════════════════════════

def _ceil_div(a: int, b: int) -> int:
    return math.ceil(a / b)


def _encoder_gemm_problems(enc: EncoderConfig, batch: int, T_prime: int) -> List[GemmProblem]:
    """
    Derive GEMM problem sizes for one encoder layer.

    The caller is responsible for repeating these across num_layers.
    BT = batch * T_prime is the effective M for all linear projections.
    """
    D = enc.d_model
    H = enc.num_heads
    F = enc.ff_units
    BH = batch * H       # batched attention: batch_count = batch * num_heads
    BT = batch * T_prime

    # Per-head Q/K and V dimensions.
    # Standard attention: qk_dim = v_dim = d_model // num_heads.
    # Zipformer: qk_head_dim=32, v_head_dim=12 (decoupled from d_model).
    qk_dim = enc.qk_head_dim if enc.qk_head_dim is not None else D // H
    v_dim  = enc.v_head_dim  if enc.v_head_dim  is not None else D // H

    problems: List[GemmProblem] = []

    # ── Feed-forward sub-blocks ─────────────────────────────────────────────
    if not enc.skip_ff:
        # macaron = 2 half-FF blocks, standard = 1 full FF block.
        # Both generate the same problem sizes; we label them ff1 / ff2.
        for i in range(2 if enc.macaron else 1):
            tag = f"ff{i + 1}"
            problems.append(GemmProblem(M=BT, N=F, K=D, op_name=f"{tag}_expand"))
            problems.append(GemmProblem(M=BT, N=D, K=F, op_name=f"{tag}_contract"))

    # ── Multi-head self-attention ────────────────────────────────────────────
    # Q, K projections output H * qk_dim features.
    # V projection outputs H * v_dim features.
    # Output projection maps H * v_dim → d_model.
    # (For standard attention all four are d_model × d_model.)
    for proj in ("q_proj", "k_proj"):
        problems.append(GemmProblem(M=BT, N=H * qk_dim, K=D, op_name=f"attn_{proj}"))
    problems.append(GemmProblem(M=BT, N=H * v_dim, K=D,       op_name="attn_v_proj"))
    problems.append(GemmProblem(M=BT, N=D,         K=H * v_dim, op_name="attn_out_proj"))

    # QK score: (B*H, T', qk_dim) @ (B*H, qk_dim, T')
    problems.append(GemmProblem(
        M=T_prime, N=T_prime, K=qk_dim, batch=BH, op_name="attn_qk",
    ))
    # Context: (B*H, T', T') @ (B*H, T', v_dim)
    problems.append(GemmProblem(
        M=T_prime, N=v_dim, K=T_prime, batch=BH, op_name="attn_av",
    ))

    # ── Conformer conv module ────────────────────────────────────────────────
    # Pointwise expand:   (BT, D) → (BT, 2D)   [for GLU gating]
    # Pointwise contract: (BT, D) → (BT, D)
    if enc.has_conv_module:
        problems.append(GemmProblem(M=BT, N=2 * D, K=D, op_name="conv_pw_expand"))
        problems.append(GemmProblem(M=BT, N=D,     K=D, op_name="conv_pw_contract"))

    return problems


def _branchformer_cgmlp_problems(spec: ModelSpec, batch: int, T_prime: int) -> List[GemmProblem]:
    """
    Extra GEMM problems for Branchformer's cgMLP branch per layer:
      cgMLP expand:   (BT, d_model) → (BT, 2 * cgmlp_units)
      cgMLP contract: (BT, cgmlp_units) → (BT, d_model)
      Branch merge:   (BT, 2 * d_model) → (BT, d_model)
    """
    if spec.cgmlp_units is None:
        return []
    D  = spec.encoder.d_model
    CU = spec.cgmlp_units
    BT = batch * T_prime
    return [
        GemmProblem(M=BT, N=2 * CU, K=D,  op_name="cgmlp_expand"),
        GemmProblem(M=BT, N=D,      K=CU, op_name="cgmlp_contract"),
        GemmProblem(M=BT, N=D,      K=2 * D, op_name="branch_merge"),
    ]


def _decoder_gemm_problems(spec: ModelSpec, batch: int, T_enc: int) -> List[GemmProblem]:
    """
    GEMM problems for an attention decoder (Paraformer-style NAR decoder).

    Decoder output length is approximated as T_enc // 4 (typical Paraformer CIF ratio).
    Includes: self-attention, cross-attention (Q from decoder, K/V from encoder), FF.
    """
    if spec.decoder_d_model is None:
        return []

    D  = spec.decoder_d_model
    H  = spec.decoder_num_heads
    F  = spec.decoder_ff_units
    NL = spec.decoder_num_layers
    hd = D // H
    T_dec = max(1, T_enc // 4)
    BH_dec = batch * H
    BT_dec = batch * T_dec
    BT_enc = batch * T_enc

    problems: List[GemmProblem] = []
    for _ in range(NL):
        # Self-attention
        for proj in ("q", "k", "v"):
            problems.append(GemmProblem(M=BT_dec, N=D, K=D, op_name=f"dec_self_{proj}"))
        problems.append(GemmProblem(M=BT_dec, N=D, K=D, op_name="dec_self_out"))
        problems.append(GemmProblem(
            M=T_dec, N=T_dec, K=hd, batch=BH_dec, op_name="dec_self_qk",
        ))
        problems.append(GemmProblem(
            M=T_dec, N=hd, K=T_dec, batch=BH_dec, op_name="dec_self_av",
        ))

        # Cross-attention  (Q: decoder, K/V: encoder)
        problems.append(GemmProblem(M=BT_dec, N=D, K=D, op_name="dec_cross_q"))
        problems.append(GemmProblem(M=BT_enc, N=D, K=D, op_name="dec_cross_k"))
        problems.append(GemmProblem(M=BT_enc, N=D, K=D, op_name="dec_cross_v"))
        problems.append(GemmProblem(M=BT_dec, N=D, K=D, op_name="dec_cross_out"))
        problems.append(GemmProblem(
            M=T_dec, N=T_enc, K=hd, batch=BH_dec, op_name="dec_cross_qk",
        ))
        problems.append(GemmProblem(
            M=T_dec, N=hd, K=T_enc, batch=BH_dec, op_name="dec_cross_av",
        ))

        # FF
        problems.append(GemmProblem(M=BT_dec, N=F, K=D, op_name="dec_ff_expand"))
        problems.append(GemmProblem(M=BT_dec, N=D, K=F, op_name="dec_ff_contract"))

    return problems


def _transducer_gemm_problems(spec: ModelSpec, batch: int, T_enc: int) -> List[GemmProblem]:
    """
    GEMM problems for the RNNT predictor (LSTM) and joiner (linear).

    LSTM: each cell gate requires two matrix multiplications:
      input  gate: (batch, input_dim)   @ (input_dim,   4 * hidden_dim)
      hidden gate: (batch, hidden_dim)  @ (hidden_dim,  4 * hidden_dim)

    Joiner: two encoder/predictor projections + vocabulary linear.
    """
    if spec.predictor_dim is None:
        return []

    pd = spec.predictor_dim
    jd = spec.joiner_dim or pd
    V  = spec.vocab_size or 5000
    NL = spec.predictor_num_layers or 2
    enc_d = spec.encoder.d_model
    problems: List[GemmProblem] = []

    # LSTM predictor layers
    input_dim = pd  # first layer uses predictor embedding dim (same as pd)
    for layer in range(NL):
        in_d = input_dim if layer == 0 else pd
        problems.append(GemmProblem(
            M=batch, N=4 * pd, K=in_d, op_name=f"lstm_layer{layer + 1}_input",
        ))
        problems.append(GemmProblem(
            M=batch, N=4 * pd, K=pd, op_name=f"lstm_layer{layer + 1}_hidden",
        ))

    # Joiner: project encoder output (batch*T_enc, enc_d) → (batch*T_enc, jd)
    problems.append(GemmProblem(
        M=batch * T_enc, N=jd, K=enc_d, op_name="joiner_enc_proj",
    ))
    # Joiner: project predictor output (batch, pd) → (batch, jd)
    problems.append(GemmProblem(
        M=batch, N=jd, K=pd, op_name="joiner_pred_proj",
    ))
    # Joiner output: (batch * T_enc, jd) → (batch * T_enc, V)
    problems.append(GemmProblem(
        M=batch * T_enc, N=V, K=jd, op_name="joiner_output",
    ))

    return problems


def _conv2d_problems(spec_sub: ConvSubsamplingSpec, batch: int, T: int) -> List[Conv2dProblem]:
    """Derive Conv2D problems for the input subsampling stack."""
    problems: List[Conv2dProblem] = []
    H, W = T, 80  # time frames × mel bins
    for i, (in_ch, out_ch, kH, kW, sH, sW, pad) in enumerate(spec_sub.layers):
        problems.append(Conv2dProblem(
            N=batch, C=in_ch, H=H, W=W,
            K=out_ch, R=kH, S=kW,
            pad_h=pad, pad_w=pad,
            stride_h=sH, stride_w=sW,
            op_name=f"conv2d_subsample_L{i + 1}",
        ))
        H = _ceil_div(H, sH)
        W = _ceil_div(W, sW)
    return problems


def derive_problems(
    spec: ModelSpec, batch: int, duration_sec: int,
) -> Tuple[List[GemmProblem], List[Conv2dProblem]]:
    """
    Derive all GEMM and CONV2D problem sizes for a given (spec, batch, duration).

    Returns
    -------
    gemm_problems  : list of GemmProblem (not deduplicated yet)
    conv2d_problems: list of Conv2dProblem (not deduplicated yet)
    """
    frame_rate = spec.encoder.frame_rate
    T = duration_sec * frame_rate  # raw input frames

    # ── CONV2D subsampling ───────────────────────────────────────────────────
    conv2d_problems: List[Conv2dProblem] = []
    if spec.encoder.subsampling is not None:
        conv2d_problems = _conv2d_problems(spec.encoder.subsampling, batch, T)
        T_prime = _ceil_div(T, spec.encoder.subsampling.time_reduction)
    elif spec.zipformer_stacks is not None:
        # Zipformer: global 4× Conv1D subsampling before per-stack strides (no Conv2D).
        T_prime = _ceil_div(T, 4)
    else:
        # Models without Conv2D subsampling (e.g. Paraformer with LFR frontend).
        # global_subsampling encodes the effective frame-rate reduction
        # (e.g. LFR stacking factor ≈ 6 for Paraformer; 1 = no reduction).
        T_prime = _ceil_div(T, spec.encoder.global_subsampling)

    # ── GEMM: encoder ────────────────────────────────────────────────────────
    gemm_problems: List[GemmProblem] = []

    if spec.zipformer_stacks is not None:
        # Zipformer: iterate per-stack with per-stack time stride
        T4 = _ceil_div(T, 4)  # after global 4× subsampling
        for enc_stack, stride in zip(spec.zipformer_stacks, spec.zipformer_strides):
            T_stack = _ceil_div(T4, stride)
            layer_problems = _encoder_gemm_problems(enc_stack, batch, T_stack)
            for _ in range(enc_stack.num_layers):
                gemm_problems.extend(layer_problems)
    else:
        layer_problems = _encoder_gemm_problems(spec.encoder, batch, T_prime)
        for _ in range(spec.encoder.num_layers):
            gemm_problems.extend(layer_problems)

    # ── GEMM: Branchformer cgMLP ─────────────────────────────────────────────
    if spec.cgmlp_units is not None:
        cgmlp = _branchformer_cgmlp_problems(spec, batch, T_prime)
        for _ in range(spec.encoder.num_layers):
            gemm_problems.extend(cgmlp)

    # ── GEMM: NAR decoder ───────────────────────────────────────────────────
    gemm_problems.extend(_decoder_gemm_problems(spec, batch, T_prime))

    # ── GEMM: RNNT predictor + joiner ────────────────────────────────────────
    gemm_problems.extend(_transducer_gemm_problems(spec, batch, T_prime))

    # ── GEMM: CTC/output projection ──────────────────────────────────────────
    if spec.vocab_size is not None:
        gemm_problems.append(GemmProblem(
            M=batch * T_prime,
            N=spec.vocab_size,
            K=spec.encoder.d_model if spec.zipformer_stacks is None
            else spec.zipformer_stacks[-1].d_model,
            op_name="ctc_output",
        ))

    return gemm_problems, conv2d_problems


def dedup_gemm(problems: List[GemmProblem]) -> List[GemmProblem]:
    """Remove GEMM duplicates (same M/N/K/batch/layout), keeping first occurrence."""
    seen: set = set()
    result: List[GemmProblem] = []
    for p in problems:
        key = (p.M, p.N, p.K, p.batch, p.transA, p.transB)
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def dedup_conv2d(problems: List[Conv2dProblem]) -> List[Conv2dProblem]:
    """Remove Conv2D duplicates."""
    seen: set = set()
    result: List[Conv2dProblem] = []
    for p in problems:
        key = (p.N, p.C, p.H, p.W, p.K, p.R, p.S,
               p.pad_h, p.pad_w, p.stride_h, p.stride_w,
               p.dilation_h, p.dilation_w)
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# nvMatmulHeuristics query layer
# ═════════════════════════════════════════════════════════════════════════════

def _gemm_config_to_dict(cfg) -> Dict[str, Any]:
    """Convert a GemmConfig object to a plain dict (all fields)."""
    return {
        "layout":      cfg.layout,
        "precision":   cfg.precision,
        "stages":      cfg.stages,
        "split_k":     cfg.split_k,
        "cta_tile_m":  cfg.cta_tile_m,
        "cta_tile_n":  cfg.cta_tile_n,
        "cta_tile_k":  cfg.cta_tile_k,
        "warp_tile_m": cfg.warp_tile_m,
        "warp_tile_n": cfg.warp_tile_n,
        "warp_tile_k": cfg.warp_tile_k,
        "instr_tile_m":cfg.instr_tile_m,
        "instr_tile_n":cfg.instr_tile_n,
        "instr_tile_k":cfg.instr_tile_k,
        "cluster_m":   cfg.cluster_m,
        "cluster_n":   cfg.cluster_n,
        "swizzle_factor": cfg.swizzle_factor,
        "cta_order":   cfg.cta_order,
    }


class NmhQueryEngine:
    """
    Manages NvMatmulHeuristicsInterface instances (one per CUTLASS target) and
    provides a high-level query interface.  Instances are created lazily and
    reused across calls.
    """

    def __init__(self, gpu_label: str, gpu_enum: Any, flags: int = 0):
        self._gpu_label = gpu_label
        self._gpu_enum  = gpu_enum
        self._flags     = flags
        # target_label → NvMatmulHeuristicsInterface
        self._interfaces: Dict[str, NvMatmulHeuristicsInterface] = {}

    def _get_interface(self, cutlass_label: str, cutlass_target: Any,
                       precision: str) -> NvMatmulHeuristicsInterface:
        """Return (or create) an interface for the given CUTLASS target + precision."""
        key = f"{cutlass_label}_{precision}"
        if key not in self._interfaces:
            hw = NvMatmulHeuristicsInterface(
                backend=cutlass_target,
                precision=precision,
                flags=self._flags,
            )
            # Bind hardware descriptor to the requested GPU
            desc = hw.createHardwareDescriptor()
            hw.setHardwarePredefinedGpu(desc, self._gpu_enum)
            hw._hardware_descriptor = desc  # keep reference to prevent GC
            self._interfaces[key] = hw
        return self._interfaces[key]

    def query_gemm(
        self,
        problem: GemmProblem,
        cutlass_label: str,
        cutlass_target: Any,
        precision: str,
        count: int = MAX_CONFIGS_PER_QUERY,
    ) -> List[Dict[str, Any]]:
        """
        Query nvMatmulHeuristics for GEMM configs.

        Returns list of result dicts, each with keys:
          cutlass_config, estimated_runtime_s
        """
        iface = self._get_interface(cutlass_label, cutlass_target, precision)
        layout = boolsToNvMatmulHeuristicsLayout(problem.transA, problem.transB)
        nmh_problem = iface.makeNvMatmulHeuristicsProblem(
            problem.M, problem.N, problem.K, layout, batch_size=problem.batch,
        )
        raw_results = iface.get(nmh_problem, count, iface._hardware_descriptor)
        return [
            {
                "cutlass_config":    _gemm_config_to_dict(r["kernel"]),
                "estimated_runtime_s": r.get("runtime", 0.0),
            }
            for r in raw_results
        ]


# ═════════════════════════════════════════════════════════════════════════════
# Core enumeration loop
# ═════════════════════════════════════════════════════════════════════════════

def enumerate_configs(
    model_families: List[str],
    model_sizes: List[str],
    batch_sizes: List[int],
    audio_durations: List[int],
    dtypes: List[str],
    gpu_labels: List[str],
    dry_run: bool = False,
) -> Iterator[ConfigRecord]:
    """
    Generator yielding one ConfigRecord per (model, batch, duration, dtype,
    precision, GPU, CUTLASS target, config).

    In --dry-run mode GPU queries are skipped; one placeholder is emitted
    per problem size so callers can inspect the enumerated shapes without GPU access.
    """
    combos = list(itertools.product(model_families, model_sizes))
    n_total = len(combos) * len(batch_sizes) * len(audio_durations)
    done = 0

    # Build one NmhQueryEngine per GPU (reused across all problems on that GPU)
    engines: Dict[str, NmhQueryEngine] = {}
    if NMH_AVAILABLE and not dry_run:
        for label in gpu_labels:
            if label not in GPU_TABLE:
                log.warning("Unknown GPU label %r — skipping", label)
                continue
            gpu_enum, _sm = GPU_TABLE[label]
            engines[label] = NmhQueryEngine(label, gpu_enum)

    for family, size in combos:
        spec = MODEL_REGISTRY.get((family, size))
        if spec is None:
            log.warning("No model spec for %s/%s — skipping", family, size)
            continue

        for batch, duration in itertools.product(batch_sizes, audio_durations):
            done += 1
            log.info(
                "[%d/%d] %s/%s  batch=%d  duration=%ds",
                done, n_total, family, size, batch, duration,
            )

            try:
                gemm_raw, conv2d_raw = derive_problems(spec, batch, duration)
            except Exception as exc:
                log.error(
                    "Problem derivation failed for %s/%s b=%d d=%d: %s",
                    family, size, batch, duration, exc,
                )
                continue

            gemm_list  = dedup_gemm(gemm_raw)
            conv2d_list = dedup_conv2d(conv2d_raw)
            log.debug(
                "  → %d GEMM + %d Conv2D problems (after dedup)",
                len(gemm_list), len(conv2d_list),
            )

            # ── GEMM ─────────────────────────────────────────────────────────
            for dtype in dtypes:
                precisions = DTYPE_PRECISIONS.get(dtype, ["HSS"])

                for gpu_label in gpu_labels:
                    if gpu_label not in GPU_TABLE:
                        continue
                    _, gpu_arch = GPU_TABLE[gpu_label]

                    for p in gemm_list:
                        prob_dict = {
                            "M": p.M, "N": p.N, "K": p.K,
                            "batch": p.batch,
                            "transA": p.transA, "transB": p.transB,
                        }
                        base = dict(
                            model_family=family, model_size=size,
                            batch_size=batch, audio_duration_sec=duration,
                            op_type="gemm", function_name=p.op_name,
                            problem_size=prob_dict, dtype=dtype,
                            gpu_target=gpu_label, gpu_arch=gpu_arch,
                        )

                        if dry_run:
                            yield ConfigRecord(
                                **base, precision="dry_run",
                                cutlass_family="dry_run", cutlass_config={},
                                status="dry_run",
                            )
                            continue

                        engine = engines.get(gpu_label)
                        if engine is None:
                            continue

                        for prec in precisions:
                            for cl, ct in CUTLASS_TARGETS:
                                yield from _query_gemm_and_emit(
                                    engine=engine,
                                    problem=p,
                                    base_kwargs=base,
                                    precision=prec,
                                    cutlass_label=cl,
                                    cutlass_target=ct,
                                )

            # ── CONV2D ───────────────────────────────────────────────────────
            # nvMatmulHeuristics covers GEMM only.  We still emit one record per
            # Conv2D problem so the shape is visible in the output.
            for dtype in dtypes:
                for gpu_label in gpu_labels:
                    if gpu_label not in GPU_TABLE:
                        continue
                    _, gpu_arch = GPU_TABLE[gpu_label]

                    for p in conv2d_list:
                        prob_dict = {
                            "N": p.N, "C": p.C, "H": p.H, "W": p.W,
                            "K": p.K, "R": p.R, "S": p.S,
                            "stride_h": p.stride_h, "stride_w": p.stride_w,
                            "pad_h": p.pad_h, "pad_w": p.pad_w,
                            "dilation_h": p.dilation_h, "dilation_w": p.dilation_w,
                        }
                        yield ConfigRecord(
                            model_family=family, model_size=size,
                            batch_size=batch, audio_duration_sec=duration,
                            op_type="conv2d", function_name=p.op_name,
                            problem_size=prob_dict, dtype=dtype,
                            precision="n/a",
                            gpu_target=gpu_label, gpu_arch=gpu_arch,
                            cutlass_family="nmh_unsupported",
                            cutlass_config={},
                            status="nmh_unsupported",
                            error_msg=(
                                "nvMatmulHeuristics does not support Conv2D; "
                                "use cuDNN or CUTLASS conv heuristics."
                            ),
                        )


def _query_gemm_and_emit(
    engine: "NmhQueryEngine",
    problem: GemmProblem,
    base_kwargs: Dict[str, Any],
    precision: str,
    cutlass_label: str,
    cutlass_target: Any,
) -> Iterator[ConfigRecord]:
    """Run one GEMM heuristic query and yield ConfigRecord objects."""
    try:
        results = engine.query_gemm(problem, cutlass_label, cutlass_target, precision)
    except Exception as exc:
        err = str(exc)
        lower = err.lower()
        if any(kw in lower for kw in ("unsupported", "not supported", "invalid", "no kernel")):
            status = "unsupported"
        else:
            status = "error"
        log.debug(
            "  NMH query failed  %s/%s  prec=%s  gpu=%s: %s",
            cutlass_label, problem.op_name, precision, base_kwargs.get("gpu_target"), exc,
        )
        yield ConfigRecord(
            **base_kwargs,
            precision=precision,
            cutlass_family=cutlass_label,
            cutlass_config={},
            status=status,
            error_msg=err,
        )
        return

    if not results:
        yield ConfigRecord(
            **base_kwargs,
            precision=precision,
            cutlass_family=cutlass_label,
            cutlass_config={},
            status="no_config",
        )
        return

    for r in results:
        yield ConfigRecord(
            **base_kwargs,
            precision=precision,
            cutlass_family=cutlass_label,
            cutlass_config=r["cutlass_config"],
            estimated_runtime_s=r.get("estimated_runtime_s", 0.0),
            status="success",
        )


# ═════════════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════════════

class JsonlWriter:
    """
    JSONL writer with cross-family merging.

    Records that share the same (model_size, batch_size, audio_duration_sec,
    op_type, problem_size, dtype, precision, gpu_target, cutlass_family,
    cutlass_config, status) but come from different model families are
    consolidated into a single record:

      * model_family  → "encoder"  (the shared encoder label)
      * extra_meta["source_families"] → sorted list of all contributing families

    Records from the same family that are exact duplicates (same merge key +
    same model_family) are silently dropped.

    All records are buffered in memory and flushed to disk on __exit__.  This
    ensures full cross-family visibility before writing.  Pass merge=False
    (--no-dedup) to disable merging and write every record as-is.
    """

    def __init__(self, path: Path, merge: bool = True):
        self._path  = path
        self._merge = merge
        # merge_key (str) → record dict; updated in-place when families merge
        self._records: Dict[str, dict] = {}
        self._fh = None
        self._merged_count = 0

    def __enter__(self) -> "JsonlWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *_) -> None:
        if self._fh:
            for d in self._records.values():
                self._fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            self._fh.close()
        log.info(
            "Wrote %d records to %s  (%d cross-family merges performed)",
            len(self._records), self._path, self._merged_count,
        )

    def write(self, record: ConfigRecord) -> None:
        d = asdict(record)

        if not self._merge:
            # Raw mode: write immediately, no buffering or merging.
            if self._fh:
                self._fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            return

        # Build the merge key from all fields that identify a unique
        # (problem × GPU × config) tuple, intentionally excluding model_family
        # so that records from different families that map to the same problem
        # are treated as candidates for merging.
        merge_key = json.dumps(
            {
                k: d[k]
                for k in (
                    "model_size", "batch_size", "audio_duration_sec",
                    "op_type", "problem_size", "dtype", "precision",
                    "gpu_target", "cutlass_family", "cutlass_config", "status",
                )
            },
            sort_keys=True,
        )

        if merge_key not in self._records:
            # First occurrence: store verbatim; seed source_families list.
            d["extra_meta"] = dict(d.get("extra_meta") or {})
            d["extra_meta"]["source_families"] = [d["model_family"]]
            self._records[merge_key] = d
            return

        existing = self._records[merge_key]
        incoming_family = d["model_family"]

        # Same family, same key → exact duplicate; discard silently.
        if incoming_family in existing["extra_meta"]["source_families"]:
            return

        # Different family for the same problem → merge.
        existing["model_family"] = "encoder"
        existing["extra_meta"]["source_families"].append(incoming_family)
        existing["extra_meta"]["source_families"].sort()
        self._merged_count += 1


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

_ALL_FAMILIES = sorted({f for f, _ in MODEL_REGISTRY})
_ALL_SIZES    = ["base", "large"]
_ALL_BATCHES  = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
_ALL_DURS     = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
_ALL_DTYPES   = ["float16", "bfloat16"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="enumerate_asr_configs",
        description="Enumerate ASR GEMM/CONV2D configs via nvMatmulHeuristics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--families", nargs="+", default=_ALL_FAMILIES,
        choices=_ALL_FAMILIES, metavar="FAMILY",
        help=f"Model families (default: all).  Choices: {_ALL_FAMILIES}",
    )
    p.add_argument(
        "--sizes", nargs="+", default=_ALL_SIZES,
        choices=_ALL_SIZES, metavar="SIZE",
        help="Model sizes: base | large  (default: both)",
    )
    p.add_argument(
        "--batches", nargs="+", type=int, default=_ALL_BATCHES, metavar="B",
        help=f"Batch sizes (default: {_ALL_BATCHES})",
    )
    p.add_argument(
        "--durations", nargs="+", type=int, default=_ALL_DURS, metavar="SEC",
        help=f"Audio durations in seconds (default: {_ALL_DURS})",
    )
    p.add_argument(
        "--dtypes", nargs="+", default=_ALL_DTYPES,
        choices=["float16", "bfloat16", "float32"], metavar="DTYPE",
        help=f"Data types (default: {_ALL_DTYPES})",
    )
    p.add_argument(
        "--gpus", nargs="+",
        default=DEFAULT_GPUS,
        choices=sorted(GPU_TABLE) if GPU_TABLE else [],
        metavar="GPU",
        help=(
            f"GPU targets (default: {DEFAULT_GPUS}).  "
            f"Available: {sorted(GPU_TABLE) if GPU_TABLE else '(install nvMatmulHeuristics)'}"
        ),
    )
    p.add_argument(
        "--output", "-o", default="asr_cutlass_configs.jsonl", metavar="FILE",
        help="Output JSONL file (default: asr_cutlass_configs.jsonl)",
    )
    p.add_argument(
        "--no-dedup", action="store_true",
        help=(
            "Disable cross-family merging and write every record as-is, "
            "including duplicates across model families."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Skip GPU heuristic queries.  "
            "Emits one placeholder record per problem size for shape inspection."
        ),
    )
    p.add_argument(
        "--list-problems", action="store_true",
        help=(
            "Print derived problem sizes to stdout as JSONL and exit "
            "(implies --dry-run, no output file written)."
        ),
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase logging verbosity (-v INFO, -vv DEBUG)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)]
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    if args.list_problems:
        args.dry_run = True

    if not NMH_AVAILABLE and not args.dry_run:
        log.error(
            "nvMatmulHeuristics is not installed (found under nvidia-matmul-heuristics "
            "but not importable).  Run:  python -c 'import nvMatmulHeuristics' to diagnose, "
            "or pass --dry-run to enumerate problem sizes without GPU queries."
        )
        return 1

    log.info(
        "families=%s  sizes=%s  batches=%s  durations=%s  dtypes=%s  gpus=%s",
        args.families, args.sizes, args.batches, args.durations,
        args.dtypes, args.gpus,
    )

    if args.list_problems:
        # Print problem shapes only; skip CUTLASS queries and file I/O.
        seen_problems: set = set()
        for family, size in itertools.product(args.families, args.sizes):
            spec = MODEL_REGISTRY.get((family, size))
            if spec is None:
                continue
            for batch, duration in itertools.product(args.batches, args.durations):
                try:
                    gemm_raw, conv2d_raw = derive_problems(spec, batch, duration)
                except Exception as exc:
                    log.error("derive_problems failed: %s", exc)
                    continue
                for p in dedup_gemm(gemm_raw):
                    key = ("gemm", p.M, p.N, p.K, p.batch)
                    if key not in seen_problems:
                        seen_problems.add(key)
                        print(json.dumps({
                            "model_family": family, "model_size": size,
                            "batch_size": batch, "audio_duration_sec": duration,
                            "op_type": "gemm", "function_name": p.op_name,
                            "M": p.M, "N": p.N, "K": p.K, "batch_count": p.batch,
                        }))
                for p in dedup_conv2d(conv2d_raw):
                    key = ("conv2d", p.N, p.C, p.H, p.W, p.K)
                    if key not in seen_problems:
                        seen_problems.add(key)
                        print(json.dumps({
                            "model_family": family, "model_size": size,
                            "batch_size": batch, "audio_duration_sec": duration,
                            "op_type": "conv2d", "function_name": p.op_name,
                            "N": p.N, "C": p.C, "H": p.H, "W": p.W,
                            "K": p.K, "R": p.R, "S": p.S,
                            "stride": (p.stride_h, p.stride_w),
                        }))
        return 0

    output_path = Path(args.output)
    log.info("Output → %s", output_path.resolve())

    with JsonlWriter(output_path, merge=not args.no_dedup) as writer:
        for record in enumerate_configs(
            model_families=args.families,
            model_sizes=args.sizes,
            batch_sizes=args.batches,
            audio_durations=args.durations,
            dtypes=args.dtypes,
            gpu_labels=args.gpus,
            dry_run=args.dry_run,
        ):
            writer.write(record)

    return 0


if __name__ == "__main__":
    sys.exit(main())
