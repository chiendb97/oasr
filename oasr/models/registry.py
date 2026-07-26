# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Model registry + checkpoint factory.

Maps an architecture name (e.g. ``"conformer"``) to the model class, its config
class, and a format converter that knows how to read a checkpoint directory.
:func:`build_model_from_checkpoint` is the single generic entry point the engine
uses to turn a checkpoint dir into a live, weight-loaded model — analogous to
vLLM / SGLang's model registry + loader split.

Resolution precedence (see ``.artifacts/multi_paradigm.md`` §7.1):

1. native OASR format (``oasr_config.json``) — loaded directly, no conversion;
2. explicit ``architecture=`` override — that converter, no sniffing;
3. converter ``detect()`` sniffing — every registered converter is probed and
   the claims are ranked by ``detect_specificity`` (a converter that read the
   architecture out of a config file outranks one matching filenames only), so
   the most specific claim wins; a tie at the top raises, and **zero** claims
   raise too (pass ``architecture=`` for loosely-structured dirs).

Adding a new architecture is a self-contained, three-line registration in the
architecture's package ``__init__`` (see ``oasr/models/conformer/__init__.py``);
no engine or registry edits are required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

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
    * ``expected_unused_prefixes: Tuple[str, ...]`` — checkpoint keys that are
      *expected* to be dropped; logged at DEBUG only.  Each entry matches as a
      key **prefix** (icefall's ``simple_am_proj``) or as a dotted **component**
      anywhere in the key (WeNet's ``concat_linear``, which lives at
      ``encoder.encoders.N.concat_linear.*``).
    * ``capability_drop_hints: Mapping[str, str]`` — key prefix → description
      of the capability lost when those weights are dropped (e.g. the U2++
      ``decoder.*`` rescoring branch); dropping these logs one WARNING naming
      the capability.
    """

    #: How *specific* this converter's :meth:`detect` is.  When several converters
    #: claim the same directory the registry keeps only the highest-specificity
    #: group, so a weak filename-based matcher cannot shadow a converter that read
    #: the architecture out of a config file.  Use one of the ``DETECT_*``
    #: constants; the default is the weakest level.
    detect_specificity: ClassVar[int]

    def detect(self, ckpt_dir: Path) -> bool:
        """Return True if *ckpt_dir* looks like this converter's format.

        Declare only **positive** markers.  Do not add negative guards for other
        formats ("return False if train.yaml exists") — that puts one format's
        knowledge inside another's converter, so adding a 7th format means editing
        an unrelated one.  Set :attr:`detect_specificity` instead and let the
        registry rank.
        """
        ...

    def build_config(self, ckpt_dir: Path) -> BaseModelConfig: ...

    def build_aux(self, ckpt_dir: Path) -> Dict[str, Any]: ...

    def load_state_dict(
        self, ckpt_dir: Path, checkpoint_name: str, map_location: Any
    ) -> Mapping[str, torch.Tensor]: ...


#: ``detect()`` specificity levels, ranked.  A directory can satisfy several
#: converters at once — a FunASR Paraformer dir carries a ``model.pt`` that looks
#: like icefall's, a WeNet dir carries a ``final.pt`` — and the right answer is
#: always the converter that identified the format most precisely.  Ranking
#: replaced the alternative, which was negative guards ("``return False`` if
#: ``config.yaml`` exists") living inside an *unrelated* converter.
#:
#: ``DETECT_KEYED_VALUE`` — a named config file whose declared field names this
#: architecture (``config.json: model_type == "whisper"``).  Unambiguous.
DETECT_KEYED_VALUE = 30
#: ``DETECT_NAMED_CONFIG`` — the presence of a framework-specific config file
#: (WeNet's ``train.yaml``).  Identifies the framework, not the architecture.
DETECT_NAMED_CONFIG = 20
#: ``DETECT_ASSET_LAYOUT`` — filename / asset conventions only (an ``exp/`` layout,
#: ``epoch-*.pt``, a ``tokens.txt`` beside the weights).  The weakest signal, and
#: the default for a converter that declares nothing.
DETECT_ASSET_LAYOUT = 10


@dataclass(frozen=True)
class ModelEntry:
    """One registered architecture."""

    model_cls: Type[BaseAsrModel]
    config_cls: Type[BaseModelConfig]
    converter: CheckpointConverter


_REGISTRY: Dict[str, ModelEntry] = {}

# The historical default when no converter claimed a checkpoint dir.  No longer
# used for resolution — kept only so the error message can name what used to
# happen, since a caller hitting it was very likely relying on the guess.
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


#: Built-in model packages, imported on first registry access so their
#: ``register_model`` calls run.  A list, not an if-chain: adding an
#: architecture is one entry, and it cannot drift out of sync with
#: ``models/__init__`` the way six hand-written guards did.
_BUILTIN_PACKAGES: Tuple[str, ...] = (
    "conformer",
    "zipformer",
    "transducer",
    "whisper",
    "paraformer",
    "speech_llm",
)

#: setuptools entry-point group for out-of-tree architectures.  A third-party
#: package declares ``[project.entry-points."oasr.models"] my_arch = "pkg.mod"``
#: and its module is imported on first registry access, so ``register_model``
#: runs without anyone editing this file.  That closes the last "adding a model
#: means editing engine core" gap in the extensibility scorecard.
_ENTRY_POINT_GROUP = "oasr.models"

_builtins_loaded = False


def _ensure_builtins() -> None:
    """Import built-in + entry-point model packages so registration runs."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    # Kept lazy to avoid an import cycle (each arch imports this module to
    # register).  The flag makes this a no-op after the first call rather than
    # a per-access membership scan.
    import importlib

    for pkg in _BUILTIN_PACKAGES:
        importlib.import_module(f"oasr.models.{pkg}")
    _load_entry_point_models()
    _builtins_loaded = True


def _load_entry_point_models() -> None:
    """Import any third-party architectures advertised via entry points.

    A broken plugin must not take down the built-in models with it, so each
    import failure is warned about and skipped — a user who installed an
    incompatible plugin should still be able to run Conformer.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8
        return
    try:
        found = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - Python < 3.10 selection API
        found = entry_points().get(_ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
    for ep in found:
        try:
            ep.load()
        except Exception as exc:  # pragma: no cover - depends on installed pkgs
            logger.warning(
                "could not load model plugin %r from entry point group %r: %s",
                getattr(ep, "name", ep),
                _ENTRY_POINT_GROUP,
                exc,
            )


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
    :meth:`CheckpointConverter.detect` is probed and the claims are **ranked** by
    ``detect_specificity`` — the most specific claim wins, a tie at the top raises
    ``ValueError``, and no claim at all raises too (pass ``architecture=``).

    Ranking is what lets each ``detect()`` state only positive markers: several
    formats share filenames (a FunASR dir carries a ``model.pt``, a WeNet dir a
    ``final.pt``), which previously forced *one* converter to carry ``return False``
    guards naming the *others*' markers.
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

    matches: List[Tuple[int, str]] = []
    for name, entry in _REGISTRY.items():
        try:
            if entry.converter.detect(path):
                specificity = int(
                    getattr(entry.converter, "detect_specificity", DETECT_ASSET_LAYOUT)
                )
                matches.append((specificity, name))
        except Exception:  # pragma: no cover - detection must never hard-fail
            logger.debug("detect() raised for %r", name, exc_info=True)

    if matches:
        # Several converters legitimately claim one directory (a FunASR dir holds a
        # ``model.pt`` that also satisfies icefall's asset rule).  Keep the most
        # specific claim; only a genuine tie is ambiguous.
        best = max(s for s, _ in matches)
        winners = sorted(name for s, name in matches if s == best)
        if len(winners) == 1:
            if len(matches) > 1:
                logger.debug(
                    "%s detected as %r (specificity %d); also matched by %s",
                    path,
                    winners[0],
                    best,
                    sorted(n for s, n in matches if s != best),
                )
            return winners[0]
        raise ValueError(
            f"Ambiguous checkpoint format at {path}: detected by {winners} at the "
            f"same specificity ({best}). Pass architecture=<name> to disambiguate."
        )
    raise ValueError(
        f"No registered converter recognized the checkpoint format at {path}. "
        f"Pass architecture=<name> explicitly (registered: {sorted(_REGISTRY)}), or "
        "convert the directory first with `oasr-convert <src> <dst>`.\n"
        "This used to fall back to "
        f"architecture={_FALLBACK_ARCHITECTURE!r} with a DeprecationWarning, which "
        "guessed WeNet/Conformer for anything unrecognized and then failed deep "
        "inside weight loading with a shape error — the guess is now refused at the "
        "point where the information is actually missing."
    )


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

    def _is_expected(key: str) -> bool:
        # A declaration matches either as a key **prefix** (icefall's
        # ``simple_am_proj``) or as a dotted **component** anywhere in the key
        # (WeNet's ``concat_linear``, which appears as
        # ``encoder.encoders.N.concat_linear.weight``).  Prefix-only matching
        # cannot express the second, and without it a normal WeNet checkpoint
        # reports two dozen "unrecognized tensors" — and worse, they fall through
        # to the ``decoder.`` capability hint, which then claims attention
        # rescoring is unavailable on a checkpoint whose decoder loaded fine.
        return any(key.startswith(p) or f".{p}" in key for p in expected)

    unexpected = [k for k in report.dropped if not _is_expected(k)]
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

    # Cast on the host first, then move: a big LLM checkpoint held fp32 may
    # not fit the GPU at all (8.4B fp32 = 33.6 GB), and the bf16 transfer is
    # half the PCIe traffic.  CPU vs GPU casts round identically.
    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device=device)
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
    # State dict host-side, model moved last — see the matching note in
    # ``loaders.load_pretrained`` (a GPU-mapped bundle would double-book VRAM).
    arch, bundle = load_checkpoint_bundle(
        ckpt_dir, checkpoint_name, map_location="cpu", architecture=architecture
    )
    model, config, _report = instantiate_from_bundle(arch, bundle, device=device, dtype=dtype)
    logger.info("Loaded %r model from %s (eval mode)", arch, ckpt_dir)
    return model, config
