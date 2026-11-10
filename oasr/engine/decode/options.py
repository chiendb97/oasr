# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Strategy-owned decode option declarations and resolution.

Values resolve from dataclass defaults, compatible legacy fields, then generic
``decode_options`` overrides. Unknown keys fail rather than being ignored.
"""

from __future__ import annotations

import dataclasses
import functools
import types
import typing
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


def option(default: Any, *, legacy: Optional[str] = None, doc: str = "") -> Any:
    """Declare an option field.

    ``legacy`` names the flat :class:`EngineConfig` attribute this option used
    to live on.  It must carry the *same default*, so reading it
    unconditionally is equivalent to reading the option default when the caller
    set nothing.  ``doc`` is surfaced by :func:`describe_options` for
    ``--decode-option`` help and the docs.
    """
    return dataclasses.field(default=default, metadata={"legacy": legacy, "doc": doc})


def option_factory(factory, *, legacy: Optional[str] = None, doc: str = "") -> Any:
    """:func:`option` for a mutable / lazily-built default (e.g. a sub-config).

    The factory runs only when the owning strategy is constructed, which is what
    keeps a Whisper engine from building a CTC beam config it never reads.
    """
    return dataclasses.field(default_factory=factory, metadata={"legacy": legacy, "doc": doc})


def describe_options(options_cls: Optional[type]) -> List[Dict[str, Any]]:
    """``[{name, default, doc, legacy}]`` for one family's options."""
    if options_cls is None:
        return []
    out = []
    for f in dataclasses.fields(options_cls):
        default = None if f.default is dataclasses.MISSING else f.default
        out.append(
            {
                "name": f.name,
                "default": default,
                "doc": f.metadata.get("doc", ""),
                "legacy": f.metadata.get("legacy"),
            }
        )
    return out


def build_options(options_cls: Optional[type], config: Any) -> Any:
    """Resolve one family's options from defaults + legacy fields + overrides."""
    overrides: Mapping[str, Any] = getattr(config, "decode_options", None) or {}

    if options_cls is None:
        if overrides:
            raise ValueError(
                f"decode_options={dict(overrides)!r} was given, but this decode "
                "family declares no options (options_cls is None)."
            )
        return None

    names = {f.name for f in dataclasses.fields(options_cls)}
    unknown = sorted(set(overrides) - names)
    if unknown:
        raise ValueError(
            f"unknown decode_options {unknown} for {options_cls.__name__}; "
            f"valid keys: {sorted(names)}"
        )

    option_types = _option_types(options_cls)
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(options_cls):
        annotation = option_types[f.name]
        legacy_name = f.metadata.get("legacy")
        if legacy_name is not None and hasattr(config, legacy_name):
            value = getattr(config, legacy_name)
            # ``None`` from a legacy field means "unset" for the option kinds
            # that have a non-None default (a lazily-built sub-config); for
            # options whose own default is None it is the value.
            if value is not None:
                kwargs[f.name] = _as_sub_config(
                    value, annotation, functools.partial(_field_default, f), legacy_name
                )
        if f.name in overrides:
            raw = overrides[f.name]
            # ``--decode-option k=v`` can only carry strings.  Type them here,
            # against the option's declared type — the serving crate must not
            # need a copy of every family's option table to do it.
            if isinstance(raw, str):
                try:
                    raw = coerce_option_value(raw, annotation)
                except ValueError as exc:
                    raise ValueError(f"decode option {f.name}={raw!r}: {exc}") from exc
            # A mapping overlays what the legacy field resolved to, else the
            # option's own default -- so ``{"wfst_search_beam": 30}`` keeps the
            # WFST family's ``search_type="wfst"`` rather than the bare
            # dataclass's ``"prefix_beam"``.
            kwargs[f.name] = _as_sub_config(
                raw,
                annotation,
                functools.partial(_resolved_or_default, kwargs, f),
                f"decode option {f.name}",
            )
    return options_cls(**kwargs)


def _field_default(f: "dataclasses.Field[Any]") -> Any:
    """A fresh copy of one option field's default (its factory runs per call)."""
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return f.default


def _resolved_or_default(kwargs: Mapping[str, Any], f: "dataclasses.Field[Any]") -> Any:
    """What the legacy field already resolved this option to, else its default."""
    return kwargs[f.name] if f.name in kwargs else _field_default(f)


def _sub_config_class(annotation: Any) -> Optional[type]:
    """The dataclass an option is declared as (``T`` or ``Optional[T]``), else ``None``."""
    target, _ = _unwrap_optional(annotation)
    if isinstance(target, type) and dataclasses.is_dataclass(target):
        return target
    return None


def _as_sub_config(value: Any, annotation: Any, base: Callable[[], Any], where: str) -> Any:
    """Resolve a value for an option declared as a dataclass (a decoder sub-config).

    An instance is kept.  A mapping -- what a JSON or YAML caller has, such as
    ``asrbench -o decode_options={decoder_config: {...}}`` -- names fields to
    change on ``base()``; its string values are typed against the
    sub-config's own annotations, the same way ``--decode-option`` strings are.
    Anything else is refused here.  Handed over as-is, a mapping used to reach
    the decoder and fail at its first attribute access (``'dict' object has no
    attribute 'wfst_max_streams'``), far from the option that caused it.
    Options that are not dataclasses pass through untouched.
    """
    cls = _sub_config_class(annotation)
    if cls is None or isinstance(value, cls):
        return value
    if value is None and _unwrap_optional(annotation)[1]:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{where} must be a {cls.__name__} or a mapping of its fields, "
            f"got {type(value).__name__}"
        )
    start = base()
    if not isinstance(start, cls):
        start = cls()
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(value) - names)
    if unknown:
        raise ValueError(
            f"unknown {cls.__name__} fields {unknown} in {where}; valid fields: {sorted(names)}"
        )
    types_ = typing.get_type_hints(cls)
    changes: Dict[str, Any] = {}
    for key, raw in value.items():
        if isinstance(raw, str):
            try:
                raw = coerce_option_value(raw, types_[key])
            except ValueError as exc:
                raise ValueError(f"{where}: {key}={raw!r}: {exc}") from exc
        changes[key] = _as_sub_config(
            raw, types_[key], functools.partial(getattr, start, key), f"{where}.{key}"
        )
    # ``replace`` builds a new object (a caller's legacy instance is not
    # mutated) and runs the sub-config's own ``__post_init__`` range checks.
    return dataclasses.replace(start, **changes)


def _option_types(options_cls: type) -> Dict[str, Any]:
    """Each option's declared type, its annotation resolved.

    The annotation, not the default, says what a string means: ``nbest`` is
    ``Optional[int]`` with a ``None`` default, and typing from the default left
    ``nbest=4`` the string ``"4"``, which the options' own validation then
    compared with an int and raised ``TypeError``.
    """
    return typing.get_type_hints(options_cls)


#: Spellings of "unset" for an ``Optional`` option that is not a string.
_NONE_SPELLINGS = ("none", "null")


def _unwrap_optional(annotation: Any) -> Tuple[Any, bool]:
    """``(T, True)`` for ``Optional[T]`` / ``T | None``; ``(annotation, False)`` otherwise."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        members = typing.get_args(annotation)
        rest = [a for a in members if a is not type(None)]
        if len(rest) == 1 and len(members) == 2:
            return rest[0], True
    return annotation, False


def coerce_option_value(raw: str, annotation: Any) -> Any:
    """Parse a ``--decode-option k=v`` string as the option's declared type.

    The wire only carries strings, so the annotation is what says whether
    ``"4"`` means the int 4 or the string "4".  ``Optional[int]`` and
    ``Optional[float]`` also take ``none`` / ``null`` for ``None``;
    ``Optional[str]`` keeps the string.  A type a string cannot spell (a
    decoder sub-config) is refused rather than handed over as a string for the
    first attribute access to trip on.  Range checks stay with the options
    dataclass's own ``__post_init__``.
    """
    target, optional = _unwrap_optional(annotation)
    if target is Any or target is str:
        return raw
    if optional and raw.strip().lower() in _NONE_SPELLINGS:
        return None
    if target is bool:
        low = raw.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if target is int:
        return int(raw)
    if target is float:
        return float(raw)
    name = getattr(target, "__name__", repr(target))
    raise ValueError(
        f"a {name} cannot be given as a string; set it in Python "
        "(EngineConfig(decode_options={...}))"
    )


def parse_decode_options(pairs, options_cls: Optional[type]) -> Dict[str, Any]:
    """Turn ``["k=v", ...]`` into a typed ``decode_options`` dict.

    Used by the serving CLI so a new family's knobs are reachable the moment it
    registers, with no new flag.  Typing is driven by the family's declared
    types, so this stays correct as families come and go.
    """
    types = _option_types(options_cls) if options_cls is not None else {}
    names = {d["name"] for d in describe_options(options_cls)}
    out: Dict[str, Any] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"--decode-option expects k=v, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if key not in names:
            raise ValueError(f"unknown decode option {key!r}; valid keys: {sorted(names)}")
        out[key] = coerce_option_value(raw, types[key])
    return out


__all__ = [
    "option",
    "option_factory",
    "describe_options",
    "build_options",
    "coerce_option_value",
    "parse_decode_options",
]
