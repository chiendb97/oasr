# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Model registry + checkpoint factory.

Maps an architecture name (e.g. ``"conformer"``) to the model class, its config
class, and a format converter that knows how to read a checkpoint directory.
:func:`build_model_from_checkpoint` is the single generic entry point the engine
uses to turn a checkpoint dir into a live, weight-loaded model — analogous to
vLLM / SGLang's model registry + loader split.

Resolution precedence (see ``docs/design/multi_paradigm.md`` §7.1):

1. native OASR format (``oasr_config.json``) — loaded directly, no conversion;
2. explicit ``architecture=`` override — that converter, no sniffing;
3. converter ``detect()`` sniffing — exactly one match wins; multiple matches
   raise (ambiguity), zero matches fall back to ``"conformer"`` with a
   :class:`DeprecationWarning` (this fallback becomes an error in a future
   release — pass ``architecture=`` for loosely-structured dirs).

Adding a new architecture is a self-contained, three-line registration in the
architecture's package ``__init__`` (see ``oasr/models/conformer/__init__.py``);
no engine or registry edits are required.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import torch

from .base import BaseAsrModel, BaseModelConfig, LoadReport

if TYPE_CHECKING:
    from oasr.checkpoints import ConvertedCheckpoint

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

    Optional extensions (not part of the runtime-checkable protocol):

    * ``convert(ckpt_dir, checkpoint_name=..., map_location=...)`` →
      :class:`~oasr.checkpoints.ConvertedCheckpoint` — emit the complete
      bundle (tokenizer / feature / decoding specs travel with the
      checkpoint).  Converters without it go through the legacy adapter in
      :func:`oasr.checkpoints.convert_checkpoint`.
    * ``expected_unused_prefixes: Tuple[str, ...]`` — checkpoint key prefixes
      that are *expected* to be dropped (e.g. icefall's ``simple_*_proj``
      training-only projections); dropping these is logged at DEBUG only.
    * ``capability_drop_hints: Mapping[str, str]`` — key prefix → description
      of the capability lost when those weights are dropped (e.g. the U2++
      ``decoder.*`` rescoring branch); dropping these logs one WARNING naming
      the capability.
    """

    def detect(self, ckpt_dir: Path) -> bool:
        """Return True if *ckpt_dir* looks like this converter's format."""
        ...

    def build_config(self, ckpt_dir: Path) -> BaseModelConfig: ...

    def build_aux(self, ckpt_dir: Path) -> Dict[str, Any]: ...

    def load_state_dict(
        self, ckpt_dir: Path, checkpoint_name: str, map_location: Any
    ) -> Mapping[str, torch.Tensor]: ...


@dataclass(frozen=True)
class ModelEntry:
    """One registered architecture."""

    model_cls: Type[BaseAsrModel]
    config_cls: Type[BaseModelConfig]
    converter: CheckpointConverter


_REGISTRY: Dict[str, ModelEntry] = {}

# Historical default when no converter claims a checkpoint dir.  Deprecated:
# scheduled to become a hard error listing registered candidates.
_FALLBACK_ARCHITECTURE = "conformer"


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


def resolve_architecture(ckpt_dir: Path, architecture: Optional[str] = None) -> str:
    """Resolve the architecture of a checkpoint directory.

    Native checkpoints answer from ``oasr_config.json``; an explicit
    *architecture* skips sniffing; otherwise every registered converter's
    :meth:`CheckpointConverter.detect` is probed — exactly one claim wins,
    multiple claims raise ``ValueError`` (pass ``architecture=``), and zero
    claims fall back to ``"conformer"`` with a :class:`DeprecationWarning`.
    """
    _ensure_builtins()
    path = Path(ckpt_dir)

    from oasr.checkpoints.native import is_native_checkpoint, read_native_config

    if is_native_checkpoint(path):
        native_arch = read_native_config(path)["architecture"]
        if architecture is not None and architecture != native_arch:
            raise ValueError(
                f"architecture={architecture!r} conflicts with native checkpoint "
                f"{path} (architecture {native_arch!r})"
            )
        return native_arch

    if architecture is not None:
        get_model_entry(architecture)  # validate eagerly
        return architecture

    matches = []
    for name, entry in _REGISTRY.items():
        try:
            if entry.converter.detect(path):
                matches.append(name)
        except Exception:  # pragma: no cover - detection must never hard-fail
            logger.debug("detect() raised for %r", name, exc_info=True)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous checkpoint format at {path}: detected by {sorted(matches)}. "
            "Pass architecture=<name> to disambiguate."
        )
    warnings.warn(
        f"No registered converter detected the format of {path}; falling back to "
        f"architecture '{_FALLBACK_ARCHITECTURE}'. This fallback is deprecated and "
        "will become an error — pass architecture=<name> explicitly "
        f"(registered: {sorted(_REGISTRY)}).",
        DeprecationWarning,
        stacklevel=2,
    )
    return _FALLBACK_ARCHITECTURE


def load_checkpoint_bundle(
    ckpt_dir: Union[str, Path],
    checkpoint_name: str = "final.pt",
    map_location: Any = "cpu",
    architecture: Optional[str] = None,
) -> Tuple[str, "ConvertedCheckpoint"]:
    """Checkpoint dir → ``(architecture, ConvertedCheckpoint)``.

    Native checkpoints load directly (no conversion); everything else runs the
    detected (or overridden) format converter through
    :func:`oasr.checkpoints.convert_checkpoint`.
    """
    _ensure_builtins()
    path = Path(ckpt_dir)

    from oasr.checkpoints import convert_checkpoint
    from oasr.checkpoints.native import is_native_checkpoint, load_native

    if is_native_checkpoint(path):
        bundle = load_native(path, map_location=map_location)
        if architecture is not None and architecture != bundle.architecture:
            raise ValueError(
                f"architecture={architecture!r} conflicts with native checkpoint "
                f"{path} (architecture {bundle.architecture!r})"
            )
        return bundle.architecture, bundle

    arch = resolve_architecture(path, architecture)
    entry = get_model_entry(arch)
    bundle = convert_checkpoint(arch, entry.converter, path, checkpoint_name, map_location)
    return arch, bundle


def _log_load_report(report: Optional[LoadReport], converter: Any, arch: str) -> None:
    """Surface dropped-weight accounting: no checkpoint tensor vanishes silently."""
    if report is None:
        return
    expected = tuple(getattr(converter, "expected_unused_prefixes", ()))
    hints: Mapping[str, str] = getattr(converter, "capability_drop_hints", {})

    unexpected = [k for k in report.dropped if not any(k.startswith(p) for p in expected)]
    if len(unexpected) < len(report.dropped):
        logger.debug(
            "%s: %d expected-unused checkpoint tensors skipped (%s)",
            arch,
            len(report.dropped) - len(unexpected),
            expected,
        )
    leftover = list(unexpected)
    for prefix, capability in hints.items():
        hit = [k for k in leftover if k.startswith(prefix)]
        if hit:
            logger.warning(
                "%s checkpoint carries %d %r tensors that were NOT loaded — %s.",
                arch,
                len(hit),
                prefix + "*",
                capability,
            )
            leftover = [k for k in leftover if not k.startswith(prefix)]
    if leftover:
        prefixes = sorted({k.split(".")[0] for k in leftover})
        logger.warning(
            "%s checkpoint has %d unrecognized tensors that were dropped "
            "(prefixes: %s); first few: %s",
            arch,
            len(leftover),
            prefixes,
            leftover[:5],
        )


def instantiate_from_bundle(
    arch: str,
    bundle: "ConvertedCheckpoint",
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[BaseAsrModel, BaseModelConfig, Optional[LoadReport]]:
    """Bundle → live, weight-loaded model in eval mode (+ the load report)."""
    entry = get_model_entry(arch)
    model = entry.model_cls.from_config(bundle.model_config, **bundle.aux)

    if bundle.source_format == "native":
        from oasr.checkpoints.native import load_native_weights

        load_native_weights(model, dict(bundle.state_dict))
        report: Optional[LoadReport] = None
    else:
        report = model.load_weights(bundle.state_dict)
        _log_load_report(report, entry.converter, arch)

    model = model.to(device=device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.eval()
    return model, bundle.model_config, report


def build_model_from_checkpoint(
    ckpt_dir: Union[str, Path],
    checkpoint_name: str = "final.pt",
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    architecture: Optional[str] = None,
) -> Tuple[BaseAsrModel, BaseModelConfig]:
    """Generic factory: checkpoint dir → ``(live model, config)`` in eval mode.

    Resolves the architecture (native format first, then ``architecture=``
    override, then converter detection), builds the bundle via the format
    converter (or reads the native format directly), instantiates the model,
    and loads weights via the model's own
    :meth:`~oasr.models.base.BaseAsrModel.load_weights`.

    Args:
        ckpt_dir: Path to the checkpoint/experiment directory.
        checkpoint_name: Weights filename inside *ckpt_dir* (default ``final.pt``).
        device: Device to map tensors onto.
        dtype: Optional dtype to cast the model into after loading.
        architecture: Explicit registry key, skipping format detection.
    """
    arch, bundle = load_checkpoint_bundle(
        ckpt_dir, checkpoint_name, map_location=device, architecture=architecture
    )
    model, config, _report = instantiate_from_bundle(arch, bundle, device=device, dtype=dtype)
    logger.info("Loaded %r model from %s (eval mode)", arch, ckpt_dir)
    return model, config
