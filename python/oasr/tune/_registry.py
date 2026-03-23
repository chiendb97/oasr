# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Pluggable backend registry for autotuning."""

import dataclasses
import threading
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from ._types import OpKey, Tactic


@dataclasses.dataclass
class BackendEntry:
    """A registered backend for an operation.

    Attributes:
        tactic: The tactic this entry represents.
        is_available: Callable returning ``True`` if the backend can run
                      on the current system (e.g. cuDNN is installed).
        get_runner: Callable returning the kernel function. The returned
                    callable must accept the same arguments as the TVM-FFI
                    module call (e.g. ``runner(out, input, filter, ...)``).
        is_fallback: If ``True``, this entry is used when no tuned config
                     exists and tuning mode is off.
    """

    tactic: Tactic
    is_available: Callable[[], bool]
    get_runner: Callable[[], Callable]
    is_fallback: bool = False


class BackendRegistry:
    """Registry mapping operations to candidate backend entries."""

    def __init__(self) -> None:
        self._entries: Dict[OpKey, List[BackendEntry]] = defaultdict(list)
        self._lock = threading.Lock()

    def register(self, op_key: OpKey, entry: BackendEntry) -> None:
        """Register a backend entry for an operation."""
        with self._lock:
            self._entries[op_key].append(entry)

    def get_candidates(self, op_key: OpKey) -> List[BackendEntry]:
        """Return all *available* backend entries for an operation."""
        entries = self._entries.get(op_key, [])
        return [e for e in entries if e.is_available()]

    def get_fallback(self, op_key: OpKey) -> Optional[BackendEntry]:
        """Return the fallback entry for an operation, or ``None``."""
        for entry in self._entries.get(op_key, []):
            if entry.is_fallback and entry.is_available():
                return entry
        # If no explicit fallback, use the first available entry.
        candidates = self.get_candidates(op_key)
        return candidates[0] if candidates else None

    def get_entry_for_tactic(
        self, op_key: OpKey, tactic: Tactic
    ) -> Optional[BackendEntry]:
        """Find the entry matching a specific tactic, or ``None``."""
        for entry in self._entries.get(op_key, []):
            if entry.tactic == tactic and entry.is_available():
                return entry
        return None


# Module-level global registry, populated by backend modules on import.
_global_registry = BackendRegistry()
