# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Load icefall Zipformer checkpoints into OASR.

icefall checkpoints store only the model state-dict (the architecture config
comes from CLI args), so the config defaults to the LibriSpeech "M" recipe (see
:class:`ZipformerModelConfig`); the CTC vocab is read from the checkpoint when
possible.  The architecture-specific key remapping lives in
:meth:`ZipformerModel.load_weights`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch

from .config import ZipformerEncoderConfig, ZipformerModelConfig
from .model import ZipformerModel

logger = logging.getLogger(__name__)

_NAMED_CANDIDATES = ("pretrained.pt", "model.pt", "checkpoint.pt", "cpu_jit.pt")
# icefall recipes universally use pos_head_dim=4; this disambiguates the
# (num_heads, *_head_dim) factorization, which is otherwise degenerate in the
# weight shapes.
_POS_HEAD_DIM = 4


def infer_encoder_config(sd: Mapping[str, torch.Tensor]) -> ZipformerEncoderConfig:
    """Infer a :class:`ZipformerEncoderConfig` from icefall checkpoint shapes.

    Recovers per-stack ``downsampling_factor`` / ``encoder_dim`` /
    ``num_encoder_layers`` / ``num_heads`` / ``query_head_dim`` /
    ``value_head_dim`` / ``feedforward_dim`` / ``cnn_module_kernel`` plus
    ``pos_dim`` from the parameter shapes, assuming ``pos_head_dim == 4``.
    ``feature_dim`` is not uniquely recoverable and defaults to 80.
    """
    stacks = sorted(
        {int(m.group(1)) for k in sd if (m := re.match(r"encoder\.encoders\.(\d+)\.", k))}
    )
    ds, edim, nlayers, nheads, qhd, vhd, ffd, cnnk = [], [], [], [], [], [], [], []
    pos_dim = None
    for i in stacks:
        pre = f"encoder.encoders.{i}."
        if (pre + "downsample.bias") in sd:
            ds.append(sd[pre + "downsample.bias"].shape[0])
            lpre = pre + "encoder.layers."
        else:
            ds.append(1)
            lpre = pre + "layers."
        nlayers.append(
            len({int(m.group(1)) for k in sd if (m := re.match(re.escape(lpre) + r"(\d+)\.", k))})
        )
        l0 = lpre + "0."
        edim.append(sd[l0 + "norm.bias"].shape[0])
        ffd.append(sd[l0 + "feed_forward2.in_proj.weight"].shape[0])
        cnnk.append(sd[l0 + "conv_module1.depthwise_conv.weight"].shape[-1])
        lin_pos = sd[l0 + "self_attn_weights.linear_pos.weight"].shape  # (H*pos_head_dim, pos_dim)
        in_proj = sd[l0 + "self_attn_weights.in_proj.weight"].shape[0]  # (2*qhd+pos_head_dim)*H
        v_in = sd[l0 + "self_attn1.in_proj.weight"].shape[0]  # H*vhd
        H = lin_pos[0] // _POS_HEAD_DIM
        nheads.append(H)
        qhd.append((in_proj // H - _POS_HEAD_DIM) // 2)
        vhd.append(v_in // H)
        pos_dim = lin_pos[1]
    # Single-value tuples for fields icefall keeps constant across stacks.
    qhd_t = (qhd[0],) if len(set(qhd)) == 1 else tuple(qhd)
    vhd_t = (vhd[0],) if len(set(vhd)) == 1 else tuple(vhd)
    return ZipformerEncoderConfig(
        feature_dim=80,
        downsampling_factor=tuple(ds),
        encoder_dim=tuple(edim),
        num_encoder_layers=tuple(nlayers),
        num_heads=tuple(nheads),
        query_head_dim=qhd_t,
        pos_head_dim=(_POS_HEAD_DIM,),
        value_head_dim=vhd_t,
        feedforward_dim=tuple(ffd),
        cnn_module_kernel=tuple(cnnk),
        pos_dim=int(pos_dim),
        causal=False,
    )


def _extract_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    """icefall checkpoints are sometimes ``{'model': sd, ...}`` and sometimes raw ``sd``."""
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj


class IcefallConverter:
    """Checkpoint converter for icefall Zipformer experiment directories.

    Implements the :class:`~oasr.models.registry.CheckpointConverter` protocol.
    """

    def _find_ckpt(self, ckpt_dir: Path, checkpoint_name: Optional[str] = None) -> Optional[Path]:
        ckpt_dir = Path(ckpt_dir)
        # icefall keeps checkpoints under exp/; search both the dir and exp/.
        search_dirs = [ckpt_dir, ckpt_dir / "exp"]
        if checkpoint_name:
            for d in search_dirs:
                if (d / checkpoint_name).exists():
                    return d / checkpoint_name
        for d in search_dirs:
            for c in _NAMED_CANDIDATES:
                if (d / c).exists():
                    return d / c
        for d in search_dirs:
            epochs = sorted(d.glob("epoch-*.pt"))
            if epochs:
                return epochs[-1]
            pts = sorted(d.glob("*.pt"))
            if pts:
                return pts[0]
        return None

    def detect(self, ckpt_dir: Path) -> bool:
        """An icefall dir has a ``.pt`` checkpoint but no WeNet ``train.yaml``."""
        ckpt_dir = Path(ckpt_dir)
        if (ckpt_dir / "train.yaml").exists():
            return False  # that's a WeNet/Conformer experiment dir
        return self._find_ckpt(ckpt_dir) is not None

    def build_config(self, ckpt_dir: Path) -> ZipformerModelConfig:
        """Build the model config, inferring the encoder architecture + vocab from
        the checkpoint shapes (falls back to the LibriSpeech "M" defaults)."""
        config = ZipformerModelConfig()
        ckpt = self._find_ckpt(Path(ckpt_dir))
        if ckpt is not None:
            try:
                sd = _extract_state_dict(torch.load(str(ckpt), map_location="cpu"))
                config.encoder = infer_encoder_config(sd)
                w = sd.get("ctc_output.1.weight")
                if w is not None:
                    vocab = int(w.shape[0])
                    # GEMM kernels require N % 8 == 0; pad like the Conformer loader.
                    if vocab % 8 != 0:
                        vocab = (vocab // 8 + 1) * 8
                    config.vocab_size = vocab
            except Exception:  # pragma: no cover - best-effort inference
                logger.warning(
                    "Could not infer Zipformer config from %s; using 'M' defaults.",
                    ckpt, exc_info=True,
                )
        logger.info(
            "Zipformer config: vocab_size=%s encoder_dim=%s num_encoder_layers=%s",
            config.vocab_size, config.encoder.encoder_dim, config.encoder.num_encoder_layers,
        )
        return config

    def build_aux(self, ckpt_dir: Path) -> dict:
        return {}

    def load_state_dict(
        self, ckpt_dir: Path, checkpoint_name: str, map_location: Any
    ) -> Mapping[str, torch.Tensor]:
        ckpt = self._find_ckpt(Path(ckpt_dir), checkpoint_name)
        if ckpt is None:
            raise FileNotFoundError(f"No icefall checkpoint (*.pt) found under {ckpt_dir}")
        return _extract_state_dict(torch.load(str(ckpt), map_location=map_location))


def load_icefall_checkpoint(
    ckpt_dir: str,
    checkpoint_name: str = "pretrained.pt",
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[ZipformerModel, ZipformerModelConfig]:
    """Convenience loader for an icefall Zipformer checkpoint directory.

    Thin wrapper around :func:`oasr.models.registry.build_model_from_checkpoint`.
    """
    from ..registry import build_model_from_checkpoint

    return build_model_from_checkpoint(
        ckpt_dir, checkpoint_name=checkpoint_name, device=device, dtype=dtype
    )
