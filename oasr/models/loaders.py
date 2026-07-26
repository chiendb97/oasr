# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``from_pretrained`` — unified Wenet / Icefall / HuggingFace checkpoint loading.

Resolves a local checkpoint directory **or** a HuggingFace Hub repo id to a live,
weight-loaded model + config, reusing the format-converter registry
(:func:`~oasr.models.registry.build_model_from_checkpoint`).  Local dirs load
directly; Hub ids are downloaded via ``huggingface_hub.snapshot_download`` first.
The architecture (Wenet conformer / Icefall zipformer / …) is auto-detected by
the registered converters — ``from_pretrained`` adds only the *source*
resolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from .base import BaseAsrModel, BaseModelConfig, LoadReport
from .registry import build_model_from_checkpoint, instantiate_from_bundle, load_checkpoint_bundle

if TYPE_CHECKING:
    from oasr.checkpoints import DecodingDefaults
    from oasr.features import FeatureSpec
    from oasr.tokenizers import TokenizerSpec

logger = logging.getLogger(__name__)


@dataclass
class PretrainedModel:
    """A loaded model plus the checkpoint-derived metadata that travels with it.

    Returned by :func:`load_pretrained`.  The engine consumes the specs
    (tokenizer / features / decoding) instead of sniffing ``ckpt_dir`` paths;
    ``load_report`` is ``None`` for native checkpoints (strict load, nothing
    dropped).
    """

    model: BaseAsrModel
    config: BaseModelConfig
    architecture: str
    tokenizer_spec: Optional["TokenizerSpec"]
    feature_spec: Optional["FeatureSpec"]
    decoding: "DecodingDefaults"
    load_report: Optional[LoadReport]


def _resolve_to_local_dir(
    model_id_or_path: Union[str, Path],
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
) -> str:
    """Resolve a checkpoint source to a local directory.

    A path that exists on disk is used as-is; otherwise the argument is treated
    as a HuggingFace Hub repo id and downloaded with ``snapshot_download``.
    """
    p = Path(model_id_or_path)
    if p.exists() and p.is_dir():
        return str(p)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            f"{model_id_or_path!r} is not a local directory and huggingface_hub "
            "is not installed.  Pass a local checkpoint dir, or "
            "`pip install huggingface_hub` to load from the Hub."
        ) from exc

    logger.info("Downloading %r from the HuggingFace Hub ...", str(model_id_or_path))
    return snapshot_download(
        repo_id=str(model_id_or_path),
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
    )


def from_pretrained(
    model_id_or_path: Union[str, Path],
    *,
    checkpoint_name: str = "final.pt",
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    architecture: Optional[str] = None,
) -> Tuple[BaseAsrModel, BaseModelConfig]:
    """Load an ASR model + config from a local dir or a HuggingFace Hub id.

    Native OASR checkpoints (``oasr_config.json``, written by ``oasr-convert``)
    load directly with no format conversion; other dirs go through the detected
    (or explicitly overridden) format converter.

    Args:
        model_id_or_path: Local checkpoint directory, or a HuggingFace Hub repo
            id (e.g. ``"Zengwei/icefall-asr-librispeech-zipformer-..."``).
        checkpoint_name: Weights filename inside the resolved dir (Wenet
            ``final.pt`` / Icefall ``pretrained.pt`` — override per format).
        device: Device to map tensors onto.
        dtype: Optional dtype to cast the model into after loading.
        revision: Hub revision (branch / tag / commit) when downloading.
        cache_dir: Hub cache directory override.
        allow_patterns: Restrict which Hub files are downloaded.
        architecture: Explicit registry key, skipping format detection.

    Returns:
        ``(model, config)`` — the live model in eval mode and its config.
    """
    local_dir = _resolve_to_local_dir(
        model_id_or_path,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
    )
    return build_model_from_checkpoint(
        local_dir, checkpoint_name, device=device, dtype=dtype, architecture=architecture
    )


def load_pretrained(
    model_id_or_path: Union[str, Path],
    *,
    checkpoint_name: str = "final.pt",
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    architecture: Optional[str] = None,
) -> PretrainedModel:
    """:func:`from_pretrained`, but returning the full :class:`PretrainedModel`.

    Same resolution pipeline; additionally surfaces the converter-emitted
    tokenizer / feature / decoding specs and the weight-load report, so callers
    (the engine, ``oasr-convert``) never re-derive checkpoint metadata.
    """
    local_dir = _resolve_to_local_dir(
        model_id_or_path,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
    )
    # The bundle's state dict always lands host-side: mapping it onto the GPU
    # would keep a full second weight copy resident (the bundle stays alive
    # for its specs) while the model moves over — an 8.4B-parameter speech-LLM
    # checkpoint then cannot fit at all.  ``instantiate_from_bundle`` moves the
    # weight-loaded model to ``device`` as its final step.
    arch, bundle = load_checkpoint_bundle(
        local_dir, checkpoint_name, map_location="cpu", architecture=architecture
    )
    model, config, report = instantiate_from_bundle(arch, bundle, device=device, dtype=dtype)
    logger.info("Loaded %r model from %s (eval mode)", arch, local_dir)
    return PretrainedModel(
        model=model,
        config=config,
        architecture=arch,
        tokenizer_spec=bundle.tokenizer,
        feature_spec=bundle.features,
        decoding=bundle.decoding,
        load_report=report,
    )
