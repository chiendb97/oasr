# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Runners: the unit a tuning region selects.

A *runner* is one backend that can serve a kernel family's operations -- the
OASR CUTLASS lanes, the fused CTC-head launcher, cuBLAS, and, registered by
later work, external libraries.  A *tactic* is one configuration inside a
runner.  Tuning is a joint search over ``(runner, tactic)`` pairs under the one
benchmark protocol (:mod:`oasr.tune.bench`), and a region of a tuning file
names the winner.

The file format needs nothing new for it.  Every config a tuning file stores
already carries a ``kind`` -- ``"cutlass"``, ``"default"``, ``"torch"``,
``"fused"``, ``"heuristic"`` -- and each kind is owned by exactly one runner of
its family.  A config whose kind no runner in this process owns, or whose
runner cannot run here, is skipped where it would have been served (the lookup
falls through to the next tier) and said once.  It never makes the rest of the
file unreadable, which it did when an unknown kind raised inside the decoder.

What a runner declares
----------------------
:class:`RunnerTraits` holds what the tuner and the runtime may rely on without
running anything:

* ``capturable`` -- its calls can be recorded in a CUDA graph.  A shape a
  captured key reaches may only be served by a capturable runner (AGENTS
  rule 11: the graph and eager must run the same kernel).
* ``batch_invariant`` -- a runner-wide guarantee that an output row depends
  only on its own input row, never on M.  Finer, per-tactic classes (the
  CUTLASS data-parallel GEMMs, :func:`oasr.jit.gemm.is_batch_invariant`) stay
  with their family.
* ``numerics`` -- ``"bitwise"`` when every tactic computes the same bits as the
  family's reference, else ``"tolerance"``.  A family with a bitwise contract
  (``scaled_mm``) admits only bitwise runners.
* ``external`` -- the kernels are not OASR's.  A measured win by an external
  runner is counted as a policy hit (``oasr.layers._backend.policy_hits``):
  a library being faster *today* is a decision and a standing argument for
  kernel work, never a silent bypass.
* ``compiled`` -- its tactics are OASR JIT variants, which cost a build and a
  slot in the production compile set.  The compile budget and the coverage
  basis count only these.

:meth:`Runner.versions` names the library versions a runner's timings depend
on.  They join the tuning file's soft validators, and an entry is reported
stale only when a runner *it* references has changed.

Registering one
---------------
Subclass :class:`Runner`, set its class attributes, and pass an instance to
:func:`register_runner` -- no tuner, database or dispatcher edit (AGENTS
rule 1).  A runner the family's dispatcher does not special-case is served
through :meth:`Runner.bind`, whose callable takes the operation's own calling
convention, output first (rule 4).  For the GEMM family that is the convention
of :mod:`oasr.functionals.gemm_torch`.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger("oasr.tune")

__all__ = [
    "RunnerTraits",
    "RunnerChoice",
    "Runner",
    "register_runner",
    "unregister_runner",
    "get_runner",
    "runner_for_kind",
    "runner_for_backend",
    "runners",
    "kind_of",
    "runner_of",
    "runner_name_of",
    "unservable",
    "runner_versions",
    "register_backend_entries",
]

_NUMERICS = ("bitwise", "tolerance")


@dataclass(frozen=True)
class RunnerTraits:
    """What a runner guarantees for every tactic it serves (see the module docstring)."""

    capturable: bool = True
    batch_invariant: bool = False
    numerics: str = "tolerance"
    external: bool = False
    compiled: bool = False

    def __post_init__(self) -> None:
        if self.numerics not in _NUMERICS:
            raise ValueError(f"numerics must be one of {_NUMERICS}, got {self.numerics!r}")


@dataclass(frozen=True)
class RunnerChoice:
    """A selection served by a runner the family's resolver does not special-case.

    A family resolver returns this where it would return one of its own config
    objects, and the dispatcher binds it through :meth:`Runner.bind`.  The
    params are kept as canonical JSON, so the choice is hashable and its
    parameters round-trip exactly into a tuning file.
    """

    kind: str
    config_id: str
    params_json: str

    @classmethod
    def of(cls, kind: str, config_id: str, params: Mapping[str, Any]) -> "RunnerChoice":
        body = dict(params)
        body["kind"] = kind
        return cls(str(kind), str(config_id), json.dumps(body, sort_keys=True))

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters as the tuning file stores them, ``kind`` included."""
        out: Dict[str, Any] = json.loads(self.params_json)
        return out


def _params_digest(params: Mapping[str, Any]) -> str:
    body = {k: v for k, v in params.items() if k != "kind"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:10]


class Runner:
    """One backend of one kernel family.  Subclass, set the attributes, register.

    Attributes:
        name: Stable identifier, unique within the family (``"cublas"``).
        family: The tuning-DB kernel family it serves (``"gemm"``).
        kinds: Config ``kind`` values it owns in tuning files.  The first is
            the kind its own tactics are written under.
        ops: The family's operations it can serve; empty means all of them.
        backends: ``Tactic.backend`` names it owns in the ``oasr.autotune()``
            registry, so a measured tactic maps back to this runner.
        traits: What it guarantees (:class:`RunnerTraits`).
        issue_class: The launch path its calls share host issue cost with.  An
            arm eliminated before its issue time was measured borrows the
            median of its class (``oasr.tune.partition``).  Empty: its own name.
    """

    name: str = ""
    family: str = ""
    kinds: Tuple[str, ...] = ()
    ops: Tuple[str, ...] = ()
    backends: Tuple[str, ...] = ()
    traits: RunnerTraits = RunnerTraits()
    issue_class: str = ""

    @property
    def issue_group(self) -> str:
        return self.issue_class or self.name

    # ------------------------------------------------------------------ status

    def unavailable(self) -> Optional[str]:
        """Why this runner cannot run in this process, or ``None`` when it can.

        A missing library or an unsupported device belongs here.  Cached by the
        caller for the epoch, so it may import or query the device.
        """
        return None

    def versions(self) -> Dict[str, Any]:
        """Library versions its timings depend on (soft validators); ``{}`` if none."""
        return {}

    def supports(self, op: str, M: int, N: int, K: int, dtype: Any) -> Optional[str]:
        """Why this runner refuses one problem, or ``None`` when it can serve it.

        Consulted where a tuned selection would be served, so a refusal falls
        through to the next tier instead of failing at launch.
        """
        return None

    def serves_op(self, op: str) -> bool:
        return not self.ops or op in self.ops

    # ------------------------------------------------------------ the codec

    def config_id(self, params: Mapping[str, Any]) -> str:
        """The stable id a tactic is stored under.  Override for readable ids."""
        return f"{self.name}_{_params_digest(params)}"

    def decode(self, params: Mapping[str, Any], config_id: Optional[str] = None) -> RunnerChoice:
        """A tuning-file config as a :class:`RunnerChoice`."""
        kind = str(params.get("kind", self.kinds[0]))
        cid = config_id if config_id is not None else self.config_id(params)
        return RunnerChoice.of(kind, cid, params)

    def choice_from_tactic(self, tactic: Any, sm: int) -> RunnerChoice:
        """The choice an ``oasr.autotune()`` tactic of this runner stands for."""
        params = {"kind": self.kinds[0], **dict(tactic.config)}
        return self.decode(params)

    # ------------------------------------------------------------- dispatch

    def bind(self, op: str, choice: Any) -> Callable[..., None]:
        """The production callable for *choice* on *op*, in the op's calling convention.

        Called once per distinct shape, when the dispatcher first plans it.
        The capture paths warm up eagerly first, so that is normally before any
        capture, but a first sighting inside one is not ruled out: bind may
        plan and import on the host, and must not launch or synchronise device
        work (allocate a persistent workspace at registration, or lazily
        outside capture).  The returned callable must not allocate per call.
        """
        raise NotImplementedError(
            f"runner {self.family}.{self.name} is dispatched by its family's own fast path"
        )

    def backend_entries(self) -> Iterable[Tuple[Any, Any]]:
        """``(OpKey, BackendEntry)`` pairs to add to the ``oasr.autotune()`` registry.

        The built-in runners' entries are registered by ``oasr.tune.backends``;
        a new runner contributes its candidates here instead, so the build and
        the runtime autotuner measure it without an edit to either.
        """
        return ()

    def __repr__(self) -> str:
        return f"<Runner {self.family}.{self.name} kinds={self.kinds}>"


# =============================================================================
# Registry
# =============================================================================

_lock = threading.RLock()
_RUNNERS: Dict[Tuple[str, str], Runner] = {}
_BY_KIND: Dict[Tuple[str, str], Runner] = {}
_BY_BACKEND: Dict[Tuple[str, str], Runner] = {}
_BUILTIN_LOADED = False
_ENTRIES_REGISTERED: set = set()


def _ensure_builtin() -> None:
    global _BUILTIN_LOADED
    if _BUILTIN_LOADED:
        return
    with _lock:
        if _BUILTIN_LOADED:
            return
        _BUILTIN_LOADED = True
        from oasr.tune.runners import builtin  # noqa: F401 -- registers on import


def register_runner(runner: Runner, *, replace: bool = False) -> Runner:
    """Register *runner*; a kind or backend another runner of the family owns is refused.

    ``replace=True`` swaps out a runner of the same ``(family, name)`` (tests,
    reloading a plugin).  Returns *runner* so a module can register at import.
    """
    if not runner.name or not runner.family:
        raise ValueError(f"{runner!r}: a runner needs a name and a family")
    if not runner.kinds:
        raise ValueError(f"{runner!r}: a runner must own at least one config kind")
    _ensure_builtin()
    with _lock:
        key = (runner.family, runner.name)
        old = _RUNNERS.get(key)
        if old is not None and not replace:
            raise ValueError(f"runner {runner.family}.{runner.name} is already registered")
        if old is not None:
            _drop(old)
        for kind in runner.kinds:
            owner = _BY_KIND.get((runner.family, kind))
            if owner is not None:
                if old is not None:
                    _restore(old)
                raise ValueError(
                    f"config kind {kind!r} of family {runner.family!r} is owned by "
                    f"{owner.name!r}; {runner.name!r} cannot claim it"
                )
        for backend in runner.backends:
            owner = _BY_BACKEND.get((runner.family, backend))
            if owner is not None:
                if old is not None:
                    _restore(old)
                raise ValueError(
                    f"autotuner backend {backend!r} of family {runner.family!r} is owned by "
                    f"{owner.name!r}; {runner.name!r} cannot claim it"
                )
        _restore(runner)
    _clear_caches()
    return runner


def _drop(runner: Runner) -> None:
    _RUNNERS.pop((runner.family, runner.name), None)
    for kind in runner.kinds:
        _BY_KIND.pop((runner.family, kind), None)
    for backend in runner.backends:
        _BY_BACKEND.pop((runner.family, backend), None)


def _restore(runner: Runner) -> None:
    _RUNNERS[(runner.family, runner.name)] = runner
    for kind in runner.kinds:
        _BY_KIND[(runner.family, kind)] = runner
    for backend in runner.backends:
        _BY_BACKEND[(runner.family, backend)] = runner


def unregister_runner(family: str, name: str) -> None:
    """Remove a registered runner (tests; a built-in comes back only by re-registering it)."""
    _ensure_builtin()
    with _lock:
        runner = _RUNNERS.get((family, name))
        if runner is not None:
            _drop(runner)
            _ENTRIES_REGISTERED.discard((family, name))
    _clear_caches()


def get_runner(family: str, name: str) -> Optional[Runner]:
    _ensure_builtin()
    return _RUNNERS.get((family, name))


def runner_for_kind(family: str, kind: str) -> Optional[Runner]:
    """The runner that owns config *kind* in *family*, or ``None``."""
    _ensure_builtin()
    return _BY_KIND.get((family, kind))


def runner_for_backend(family: str, backend: str) -> Optional[Runner]:
    """The runner that owns ``oasr.autotune()`` backend name *backend*, or ``None``."""
    _ensure_builtin()
    return _BY_BACKEND.get((family, backend))


def runners(family: Optional[str] = None) -> List[Runner]:
    """Registered runners, of *family* or of every family, in registration order."""
    _ensure_builtin()
    with _lock:
        return [r for (fam, _), r in _RUNNERS.items() if family is None or fam == family]


# =============================================================================
# Choices
# =============================================================================


def kind_of(choice: Any) -> str:
    """The config kind a family choice is stored under.

    A family's sentinels are strings naming their kind (``"torch"``,
    ``"fused"``, ``"heuristic"``); a :class:`RunnerChoice` carries its own; any
    other object is one of the family's compiled kernel configs, and the
    family's default config is one of those too.
    """
    if isinstance(choice, RunnerChoice):
        return choice.kind
    if isinstance(choice, str):
        return choice
    return "cutlass"


def runner_of(family: str, choice: Any) -> Optional[Runner]:
    """The registered runner that serves *choice*, or ``None`` when none owns its kind."""
    return runner_for_kind(family, kind_of(choice))


def runner_name_of(family: str, choice: Any) -> Optional[str]:
    runner = runner_of(family, choice)
    return None if runner is None else runner.name


#: ``(family, name) -> unavailable() result``, per tuning-DB epoch.
_UNAVAILABLE: Dict[Tuple[str, str], Optional[str]] = {}


def _unavailable(runner: Runner) -> Optional[str]:
    key = (runner.family, runner.name)
    if key not in _UNAVAILABLE:
        try:
            _UNAVAILABLE[key] = runner.unavailable()
        except Exception as exc:  # noqa: BLE001 -- a probe that raises means "cannot run"
            _UNAVAILABLE[key] = f"availability probe raised {type(exc).__name__}: {exc}"
    return _UNAVAILABLE[key]


def unservable(
    family: str, op: str, choice: Any, M: int, N: int, K: int, dtype: Any
) -> Optional[str]:
    """Why *choice* cannot serve this problem in this process, or ``None`` when it can.

    The runner-level half of a resolver's check -- an owner for the kind, the
    runner available here, the op served, the problem supported.  Whether a
    compiled kernel config is in this process's compile set stays with the
    family, which owns that set.
    """
    runner = runner_of(family, choice)
    if runner is None:
        return f"no runner of family {family!r} owns config kind {kind_of(choice)!r}"
    why = _unavailable(runner)
    if why is not None:
        return f"runner {runner.name!r} is unavailable: {why}"
    if not runner.serves_op(op):
        return f"runner {runner.name!r} does not serve {family}.{op}"
    try:
        refusal = runner.supports(op, M, N, K, dtype)
    except Exception as exc:  # noqa: BLE001
        refusal = f"supports() raised {type(exc).__name__}: {exc}"
    if refusal is not None:
        return f"runner {runner.name!r} refuses {op} M={M} N={N} K={K}: {refusal}"
    return None


# =============================================================================
# Soft validators and the autotuner registry
# =============================================================================


@functools.lru_cache(maxsize=None)
def runner_versions(family: str) -> Dict[str, Dict[str, Any]]:
    """``{runner name: versions()}`` for every available runner of *family* that has any.

    Cached per tuning-DB epoch: :func:`oasr.tune.database.reload` clears it.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for runner in runners(family):
        if _unavailable(runner) is not None:
            continue
        try:
            v = runner.versions()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[tune.runners] %s.versions() raised: %s", runner.name, exc)
            continue
        if v:
            out[runner.name] = dict(v)
    return out


def _clear_caches() -> None:
    _UNAVAILABLE.clear()
    runner_versions.cache_clear()


def register_backend_entries(registry: Any) -> int:
    """Add every registered runner's :meth:`Runner.backend_entries` to *registry* once.

    Returns how many entries were added.  Idempotent per runner, so the
    autotuner may call it on every registration pass.
    """
    added = 0
    for runner in runners():
        key = (runner.family, runner.name)
        if key in _ENTRIES_REGISTERED or _unavailable(runner) is not None:
            continue
        for op_key, entry in runner.backend_entries():
            registry.register(op_key, entry)
            added += 1
        _ENTRIES_REGISTERED.add(key)
    return added


def _register_reload_hook() -> None:
    from oasr.tune import database

    database.register_reload_hook(_clear_caches)


_register_reload_hook()
