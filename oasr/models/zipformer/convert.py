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
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch

from .config import ZipformerModelConfig
from .model import ZipformerModel

logger = logging.getLogger(__name__)

_NAMED_CANDIDATES = ("pretrained.pt", "model.pt", "checkpoint.pt", "cpu_jit.pt")


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
        if checkpoint_name and (ckpt_dir / checkpoint_name).exists():
            return ckpt_dir / checkpoint_name
        for c in _NAMED_CANDIDATES:
            if (ckpt_dir / c).exists():
                return ckpt_dir / c
        epochs = sorted(ckpt_dir.glob("epoch-*.pt"))
        if epochs:
            return epochs[-1]
        pts = sorted(ckpt_dir.glob("*.pt"))
        return pts[0] if pts else None

    def detect(self, ckpt_dir: Path) -> bool:
        """An icefall dir has a ``.pt`` checkpoint but no WeNet ``train.yaml``."""
        ckpt_dir = Path(ckpt_dir)
        if (ckpt_dir / "train.yaml").exists():
            return False  # that's a WeNet/Conformer experiment dir
        return self._find_ckpt(ckpt_dir) is not None

    def build_config(self, ckpt_dir: Path) -> ZipformerModelConfig:
        config = ZipformerModelConfig()
        ckpt = self._find_ckpt(Path(ckpt_dir))
        if ckpt is not None:
            try:
                sd = _extract_state_dict(torch.load(str(ckpt), map_location="cpu"))
                w = sd.get("ctc_output.1.weight")
                if w is not None:
                    vocab = int(w.shape[0])
                    # GEMM kernels require N % 8 == 0; pad like the Conformer loader.
                    if vocab % 8 != 0:
                        vocab = (vocab // 8 + 1) * 8
                    config.vocab_size = vocab
            except Exception:  # pragma: no cover - best-effort vocab probe
                logger.debug("Could not probe vocab_size from %s", ckpt, exc_info=True)
        logger.info("Zipformer config: vocab_size=%s (defaults to LibriSpeech 'M')",
                    config.vocab_size)
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
