# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Complete checkpoint-conversion bundle.

A :class:`ConvertedCheckpoint` is everything a checkpoint conversion produces:
model config + weights + aux buffers (as before), **plus** the tokenizer /
feature / decoding metadata that used to be re-derived (or lost) engine-side.
Converters that implement ``convert()`` emit it directly; legacy 4-method
converters (``detect`` / ``build_config`` / ``build_aux`` / ``load_state_dict``)
keep working through :func:`convert_checkpoint`, which fills the metadata with
the historical engine-side sniffing behavior.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from oasr.features import FeatureSpec
from oasr.tokenizers import TokenizerSpec

logger = logging.getLogger(__name__)


@dataclass
class DecodingDefaults:
    """Decoding metadata that travels with the checkpoint."""

    default_decode_type: str = "ctc"
    blank_id: int = 0
    unk_id: Optional[int] = None
    sos_id: Optional[int] = None
    eos_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_decode_type": self.default_decode_type,
            "blank_id": self.blank_id,
            "unk_id": self.unk_id,
            "sos_id": self.sos_id,
            "eos_id": self.eos_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecodingDefaults":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ConvertedCheckpoint:
    """Everything one checkpoint conversion yields.

    ``source_format`` records which converter path produced the bundle
    (``"wenet"`` / ``"icefall"`` / ``"native"`` / ``"legacy"``).  Native-format
    state dicts are already in OASR's own key space and load via
    ``model.load_state_dict`` (strict); every other format goes through the
    architecture's ``model.load_weights`` name mapping.
    """

    architecture: str
    model_config: Any  # BaseModelConfig (kept loose to avoid an import cycle)
    aux: Dict[str, Any] = field(default_factory=dict)
    state_dict: Mapping[str, torch.Tensor] = field(default_factory=dict)
    tokenizer: Optional[TokenizerSpec] = None
    features: Optional[FeatureSpec] = None
    decoding: DecodingDefaults = field(default_factory=DecodingDefaults)
    source_format: str = "legacy"


def sniff_legacy_tokenizer_spec(ckpt_dir: Path) -> Optional[TokenizerSpec]:
    """Historical engine-side tokenizer discovery, as a spec.

    Mirrors ``EngineConfig.__post_init__``: the first of ``units.txt`` /
    ``words.txt`` becomes a ``symbol_table`` spec.  Used only for legacy
    converters that do not emit a :class:`TokenizerSpec` themselves.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    for fname in ("units.txt", "words.txt"):
        candidate = ckpt_dir / fname
        if candidate.exists():
            return TokenizerSpec(kind="symbol_table", files={"table": str(candidate)})
    return None


def sniff_legacy_sentencepiece(ckpt_dir: Path) -> Optional[str]:
    """First ``*.model`` file in the dir (the historical sniffing rule)."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    for fname in sorted(os.listdir(ckpt_dir)):
        if fname.endswith(".model"):
            return str(ckpt_dir / fname)
    return None


def convert_checkpoint(
    architecture: str,
    converter: Any,
    ckpt_dir: Path,
    checkpoint_name: str = "final.pt",
    map_location: Any = "cpu",
) -> ConvertedCheckpoint:
    """Run *converter* on *ckpt_dir*, returning a complete bundle.

    Uses the converter's own ``convert()`` when implemented; otherwise adapts
    the legacy 4-method protocol, filling tokenizer / feature / decoding
    metadata with the historical sniffing defaults so behavior is unchanged.
    """
    ckpt_dir = Path(ckpt_dir)
    convert = getattr(converter, "convert", None)
    if callable(convert):
        bundle = convert(ckpt_dir, checkpoint_name=checkpoint_name, map_location=map_location)
        if bundle.architecture != architecture:
            logger.warning(
                "Converter for %r emitted architecture %r", architecture, bundle.architecture
            )
        return bundle

    return ConvertedCheckpoint(
        architecture=architecture,
        model_config=converter.build_config(ckpt_dir),
        aux=converter.build_aux(ckpt_dir),
        state_dict=converter.load_state_dict(ckpt_dir, checkpoint_name, map_location),
        tokenizer=sniff_legacy_tokenizer_spec(ckpt_dir),
        features=None,  # legacy converters: engine-side FeatureConfig default
        decoding=DecodingDefaults(),
        source_format="legacy",
    )
