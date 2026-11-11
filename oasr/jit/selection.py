# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""A measured selection table: one tuning-DB family a dispatcher consults per shape.

The shared half of the families whose region names a *path* -- a lane, its tile,
or "today's rule" -- rather than a compiled variant of one kernel
(:mod:`oasr.jit.recurrent_select`, :mod:`oasr.jit.mlp_select`): the decoded tier
views of the user and system files, the walk from a signature key and a dynamic
size to a servable choice, the warning for an entry this process cannot serve,
the "no file at all" fast path, and the per-epoch reset.  The family module owns
everything that is about its kernels: the codec, the signature, servability.

Resolution is the database's: user tier, then system; a region rounds the
dynamic size up; a size above every bounded region was never measured there and
falls through; an entry naming a choice this process cannot serve is said once
and skipped.  The answer is a pure function of the shape and the tuning snapshot,
never of CUDA-graph capture state (AGENTS.md rule 11).
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("oasr.jit.selection")

__all__ = ["MeasuredTable", "AUTO_ID"]

#: The sentinel every such family stores for "measured; today's rule is best here".
AUTO_ID = "auto"

#: ``(dynamic hi | None, config id, decoded choice | None)``.
_Rule = Tuple[Optional[int], str, Any]


@functools.lru_cache(maxsize=1)
def _target_sm() -> int:
    from oasr.jit.core import _get_target_sm

    return int(_get_target_sm())


class MeasuredTable:
    """The tiers of one family, decoded per epoch, and the walk to a choice.

    ``decode(params)`` turns a stored config into a choice (raising on one it
    does not know); ``unusable(choice, shape)`` says why a choice cannot serve
    *shape* here, or ``None``.
    """

    def __init__(
        self,
        family: str,
        op: str,
        decode: Callable[[Dict[str, Any]], Any],
        unusable: Callable[[Any, Any], Optional[str]],
    ) -> None:
        self.family = family
        self.op = op
        self._decode_fn = decode
        self._unusable = unusable
        self._views: Dict[Tuple[int, int], List[Tuple[str, Dict[str, List[_Rule]]]]] = {}
        self._warned: set = set()
        #: The epoch at which no file of this family was loaded for this arch.
        self.inactive_epoch = -1
        from oasr.tune import database

        database.register_reload_hook(self._on_reload)
        self.epoch: Callable[[], int] = database.epoch

    # ------------------------------------------------------------------ tiers

    def _decode(self, params: Optional[Dict[str, Any]]) -> Any:
        if params is None:
            return None
        try:
            return self._decode_fn(params)
        except (KeyError, TypeError, ValueError):
            return None

    def _table(self, tf) -> Dict[str, List[_Rule]]:
        from oasr.tune.database import parse_sig_key

        table: Dict[str, List[_Rule]] = {}
        decoded: Dict[str, Any] = {}
        for key, entry in tf.entries.items():
            family, fields = parse_sig_key(key)
            if family != self.family or fields.get("op") != self.op or not entry.regions:
                continue
            rules: List[_Rule] = []
            for hi, cid in entry.regions:
                if cid not in decoded:
                    decoded[cid] = self._decode(tf.configs.get(cid))
                rules.append((hi, cid, decoded[cid]))
            table[key] = rules
        return table

    def views(self, sm: Optional[int] = None) -> List[Tuple[str, Dict[str, List[_Rule]]]]:
        """The decoded user and system tables for arch family *sm*, this epoch."""
        from oasr.tune import database

        sm_ = _target_sm() if sm is None else int(sm)
        epoch = database.epoch()
        key = (epoch, sm_)
        views = self._views.get(key)
        if views is not None:
            return views
        views = [(tier, self._table(tf)) for tier, tf in database.tiers(self.family, sm_).ordered()]
        for stale in [k for k in self._views if k[0] != epoch]:
            del self._views[stale]
        self._views[key] = views
        return views

    def active(self, epoch: int) -> bool:
        """Whether any file of this family is loaded for this arch this epoch."""
        if self.inactive_epoch == epoch:
            return False
        try:
            views = self.views()
        except Exception:  # noqa: BLE001 -- no device, no target: nothing to consult
            views = []
        if not views:
            self.inactive_epoch = epoch
            return False
        return True

    # ------------------------------------------------------------- resolution

    def resolve(
        self, key: str, size: int, shape: Any, sm: Optional[int] = None
    ) -> Tuple[Optional[str], Any]:
        """``(tier, choice)`` for signature *key* at dynamic *size*.

        ``choice`` is ``None`` for "today's rule": a region naming
        :data:`AUTO_ID`, or no covering entry (``tier`` is then ``None``).
        """
        for tier, table in self.views(sm):
            rules = table.get(key)
            if not rules:
                continue
            rule = None
            for r in rules:
                if r[0] is None or size <= r[0]:
                    rule = r
                    break
            if rule is None:  # above every bounded region: never measured there
                continue
            _, cid, choice = rule
            if choice == AUTO_ID:
                return tier, None
            reason = "cannot be decoded" if choice is None else self._unusable(choice, shape)
            if reason is None:
                return tier, choice
            self._note_unusable(tier, key, cid, reason)
        return None, None

    def _note_unusable(self, tier: str, key: str, cid: str, reason: str) -> None:
        wkey = (tier, key, cid)
        if wkey in self._warned:
            return
        self._warned.add(wkey)
        logger.warning(
            "%s %s-tier entry %s names %s, which %s; falling through",
            self.family,
            tier,
            key,
            cid,
            reason,
        )

    def _on_reload(self) -> None:
        self._views.clear()
        self.inactive_epoch = -1
