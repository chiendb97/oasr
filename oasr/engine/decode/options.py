# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Strategy-owned decode option declarations and resolution.

Values resolve from dataclass defaults, compatible legacy fields, then generic
``decode_options`` overrides. Unknown keys fail rather than being ignored.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Mapping, Optional


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

    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(options_cls):
        legacy_name = f.metadata.get("legacy")
        if legacy_name is not None and hasattr(config, legacy_name):
            value = getattr(config, legacy_name)
            # ``None`` from a legacy field means "unset" for the option kinds
            # that have a non-None default (a lazily-built sub-config); for
            # options whose own default is None it is the value.
            if value is not None:
                kwargs[f.name] = value
        if f.name in overrides:
            raw = overrides[f.name]
            # ``--decode-option k=v`` can only carry strings.  Type them here,
            # against the option's declared default — the serving crate must not
            # need a copy of every family's option table to do it.
            default = None if f.default is dataclasses.MISSING else f.default
            if isinstance(raw, str) and not isinstance(default, str) and default is not None:
                try:
                    raw = coerce_option_value(raw, default)
                except ValueError as exc:
                    raise ValueError(f"decode option {f.name}={raw!r}: {exc}") from exc
            kwargs[f.name] = raw
    return options_cls(**kwargs)


def coerce_option_value(raw: str, default: Any) -> Any:
    """Parse a ``--decode-option k=v`` string against the option's default type.

    The wire only carries strings, so the default is what says whether ``"4"``
    means the int 4 or the string "4".  Unknown/None defaults stay strings —
    the option dataclass's own ``__post_init__`` is the validator, not this.
    """
    if isinstance(default, bool):
        low = raw.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def parse_decode_options(pairs, options_cls: Optional[type]) -> Dict[str, Any]:
    """Turn ``["k=v", ...]`` into a typed ``decode_options`` dict.

    Used by the serving CLI so a new family's knobs are reachable the moment it
    registers, with no new flag.  Typing is driven by the family's declared
    defaults, so this stays correct as families come and go.
    """
    defaults = {d["name"]: d["default"] for d in describe_options(options_cls)}
    out: Dict[str, Any] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"--decode-option expects k=v, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if key not in defaults:
            raise ValueError(f"unknown decode option {key!r}; valid keys: {sorted(defaults)}")
        out[key] = coerce_option_value(raw, defaults[key])
    return out


__all__ = [
    "option",
    "option_factory",
    "describe_options",
    "build_options",
    "coerce_option_value",
    "parse_decode_options",
]
