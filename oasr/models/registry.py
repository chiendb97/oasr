# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Model registry + checkpoint factory.

Maps an architecture name (e.g. ``"conformer"``) to the model class, its config
class, and a format converter that knows how to read a checkpoint directory.
:func:`build_model_from_checkpoint` is the single generic entry point the engine
uses to turn a checkpoint dir into a live, weight-loaded model — analogous to
vLLM / SGLang's model registry + loader split.

Adding a new architecture is a self-contained, three-line registration in the
architecture's package ``__init__`` (see ``oasr/models/conformer/__init__.py``);
no engine or registry edits are required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import torch

from .base import BaseAsrModel, BaseModelConfig

try:  # Python 3.8 compatibility for Protocol
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

logger = logging.getLogger(__name__)


@runtime_checkable
class CheckpointConverter(Protocol):
    """Format-specific checkpoint reader.

    A converter is responsible for everything format-specific: detecting that a
    directory is in its format, translating the on-disk config into a
    :class:`BaseModelConfig`, building auxiliary buffers (e.g. CMVN) passed to
    ``Model.from_config``, and loading the raw external state-dict.  The
    architecture-specific name-mapping happens later in ``Model.load_weights``.
    """

    def detect(self, ckpt_dir: Path) -> bool:
        """Return True if *ckpt_dir* looks like this converter's format."""
        ...

    def build_config(self, ckpt_dir: Path) -> BaseModelConfig:
        ...

    def build_aux(self, ckpt_dir: Path) -> Dict[str, Any]:
        ...

    def load_state_dict(
        self, ckpt_dir: Path, checkpoint_name: str, map_location: Any
    ) -> Mapping[str, torch.Tensor]:
        ...


@dataclass(frozen=True)
class ModelEntry:
    """One registered architecture."""

    model_cls: Type[BaseAsrModel]
    config_cls: Type[BaseModelConfig]
    converter: CheckpointConverter


_REGISTRY: Dict[str, ModelEntry] = {}


def register_model(
    name: str,
    *,
    model_cls: Type[BaseAsrModel],
    config_cls: Type[BaseModelConfig],
    converter: CheckpointConverter,
) -> None:
    """Register an architecture under *name* (idempotent; last write wins)."""
    if name in _REGISTRY:
        logger.debug("Overriding model registration for %r", name)
    _REGISTRY[name] = ModelEntry(model_cls, config_cls, converter)


def _ensure_builtins() -> None:
    """Import built-in model packages so their ``register_model`` calls run."""
    # Importing each package triggers its __init__ registration. Kept lazy to
    # avoid an import cycle (each arch imports this module to register).
    if "conformer" not in _REGISTRY:
        import oasr.models.conformer  # noqa: F401
    if "zipformer" not in _REGISTRY:
        import oasr.models.zipformer  # noqa: F401


def get_model_entry(name: str) -> ModelEntry:
    _ensure_builtins()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown model architecture {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def list_models() -> List[str]:
    """Names of all registered architectures."""
    _ensure_builtins()
    return sorted(_REGISTRY)


def resolve_architecture(ckpt_dir: Path) -> str:
    """Detect the architecture of a checkpoint directory.

    Probes each registered converter's :meth:`CheckpointConverter.detect`;
    falls back to ``"conformer"`` (the historical default) when none claim it.
    """
    _ensure_builtins()
    for name, entry in _REGISTRY.items():
        try:
            if entry.converter.detect(ckpt_dir):
                return name
        except Exception:  # pragma: no cover - detection must never hard-fail
            logger.debug("detect() raised for %r", name, exc_info=True)
    return "conformer"


def build_model_from_checkpoint(
    ckpt_dir: Union[str, Path],
    checkpoint_name: str = "final.pt",
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[BaseAsrModel, BaseModelConfig]:
    """Generic factory: checkpoint dir → ``(live model, config)`` in eval mode.

    Detects the architecture, builds the config + aux buffers via the format
    converter, instantiates the model, loads weights via the model's own
    :meth:`~oasr.models.base.BaseAsrModel.load_weights`, then moves it to the
    target device / dtype and sets eval mode.

    Args:
        ckpt_dir: Path to the checkpoint/experiment directory.
        checkpoint_name: Weights filename inside *ckpt_dir* (default ``final.pt``).
        device: Device to map tensors onto.
        dtype: Optional dtype to cast the model into after loading.
    """
    _ensure_builtins()
    path = Path(ckpt_dir)
    arch = resolve_architecture(path)
    entry = get_model_entry(arch)
    converter = entry.converter

    config = converter.build_config(path)
    aux = converter.build_aux(path)
    model = entry.model_cls.from_config(config, **aux)

    state_dict = converter.load_state_dict(path, checkpoint_name, map_location=device)
    model.load_weights(state_dict)

    model = model.to(device=device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.eval()
    logger.info("Loaded %r model from %s (eval mode)", arch, path)
    return model, config
