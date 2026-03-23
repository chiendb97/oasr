# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Core data types for the autotuning module."""

import dataclasses
from typing import Any, Dict, Tuple


@dataclasses.dataclass(frozen=True)
class OpKey:
    """Identifies an operation type.

    Attributes:
        family: Kernel family, e.g. ``"gemm"``, ``"conv"``, ``"norm"``.
        op: Operation name within the family, e.g. ``"gemm"``, ``"conv2d"``.
        flags: Extra frozen flags that affect kernel choice,
               e.g. ``("activation_type=2",)``.
    """

    family: str
    op: str
    flags: Tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ProfileKey:
    """Cache lookup key combining operation, shape, dtype, and device arch.

    Attributes:
        op_key: The operation identifier.
        shape_sig: Canonicalized shape signature (operation-specific).
                   For GEMM: ``(M, N, K)``.
                   For Conv2D: ``(N, H, W, IC, K, R, S, stride_h, ...)``.
        dtype: Data type string, e.g. ``"float16"``, ``"bfloat16"``.
        device_sm: GPU SM version, e.g. ``80``, ``86``, ``90``.
    """

    op_key: OpKey
    shape_sig: Tuple[int, ...]
    dtype: str
    device_sm: int

    def to_str(self) -> str:
        """Stable, deterministic string for use as a JSON/dict key."""
        flags_part = ",".join(self.op_key.flags) if self.op_key.flags else ""
        shape_part = ",".join(str(s) for s in self.shape_sig)
        parts = [
            self.op_key.family,
            self.op_key.op,
            flags_part,
            f"({shape_part})",
            self.dtype,
            f"sm{self.device_sm}",
        ]
        return "|".join(parts)

    @classmethod
    def from_str(cls, s: str) -> "ProfileKey":
        """Parse a ``ProfileKey`` from its ``to_str()`` representation."""
        parts = s.split("|")
        family, op, flags_str, shape_str, dtype, sm_str = parts
        flags = tuple(flags_str.split(",")) if flags_str else ()
        shape_inner = shape_str.strip("()")
        shape_sig = tuple(int(x) for x in shape_inner.split(",")) if shape_inner else ()
        device_sm = int(sm_str.replace("sm", ""))
        return cls(
            op_key=OpKey(family=family, op=op, flags=flags),
            shape_sig=shape_sig,
            dtype=dtype,
            device_sm=device_sm,
        )


@dataclasses.dataclass(frozen=True)
class Tactic:
    """A specific backend + configuration combination.

    Attributes:
        backend: Backend name, e.g. ``"cutlass"``, ``"cudnn"``.
        config: Frozen key-value pairs for backend-specific knobs,
                e.g. ``(("tile_m", 128), ("tile_n", 128))``.
    """

    backend: str
    config: Tuple[Tuple[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "backend": self.backend,
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tactic":
        """Deserialize from a dict."""
        config = tuple(sorted(d.get("config", {}).items()))
        return cls(backend=d["backend"], config=config)


@dataclasses.dataclass
class TuneResult:
    """Profiling result for one tactic.

    Attributes:
        tactic: The tactic that was profiled.
        median_ms: Median execution time in milliseconds.
        min_ms: Minimum execution time in milliseconds.
        max_ms: Maximum execution time in milliseconds.
        status: One of ``"ok"``, ``"error"``, ``"unsupported"``.
        error_msg: Error message if ``status != "ok"``.
    """

    tactic: Tactic
    median_ms: float
    min_ms: float
    max_ms: float
    status: str
    error_msg: str = ""
