# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for GEMM operations."""

import functools
import logging
from typing import Optional

import torch

# Module-attribute indirection, not ``from ... import``: the per-call
# ``from oasr.jit.gemm import select_default_config`` cost ~0.3 us of importlib
# machinery, but binding the name locally would also break monkeypatching of
# the heuristic at its source, which the dispatch tests rely on.
import oasr.jit.gemm as _jit_gemm
from oasr.api_logging import oasr_api
from oasr.tune import is_tuning_enabled

logger = logging.getLogger("oasr.functionals.gemm")


@functools.cache
def _get_gemm_module():
    from oasr.jit.gemm import gen_gemm_module

    return gen_gemm_module().build_and_load()


@functools.cache
def _get_bmm_module():
    from oasr.jit.gemm import gen_bmm_module

    return gen_bmm_module().build_and_load()


@functools.cache
def _get_group_gemm_module():
    from oasr.jit.gemm import gen_group_gemm_module

    return gen_group_gemm_module().build_and_load()


@functools.cache
def _get_gemm_log_softmax_module():
    from oasr.jit.gemm import gen_gemm_log_softmax_module

    return gen_gemm_log_softmax_module().build_and_load()


def _default_gemm_fn():
    from oasr.jit.gemm import GEMM_DEFAULT, gemm_func_name

    return getattr(_get_gemm_module(), gemm_func_name(GEMM_DEFAULT))


def _default_gemm_activation_fn():
    from oasr.jit.gemm import GEMM_DEFAULT, gemm_activation_func_name

    return getattr(_get_gemm_module(), gemm_activation_func_name(GEMM_DEFAULT))


def _default_bmm_fn():
    from oasr.jit.gemm import GEMM_DEFAULT, bmm_func_name

    return getattr(_get_bmm_module(), bmm_func_name(GEMM_DEFAULT))


def _default_group_gemm_fn():
    from oasr.jit.gemm import GEMM_DEFAULT, group_gemm_func_name

    return getattr(_get_group_gemm_module(), group_gemm_func_name(GEMM_DEFAULT))


@functools.cache
def _target_sm() -> int:
    from oasr.jit.core import _get_target_sm

    return _get_target_sm()


def _gemm_io(A: torch.Tensor, N: int, out: Optional[torch.Tensor]):
    """``(A, out)`` → ``(A_contig, out, M)``.  No reshaping: the launcher flattens.

    The ``contiguous()`` is not just a precondition the launchers impose — it is
    faster than the alternative, which is worth stating because it looks like a
    hidden cost.  Zipformer is the architecture that hits it: 11% of its GEMM
    calls arrive with ``A`` a ``(T, B, C)`` view of a contiguous ``(B, T, C)``,
    38 MiB of copies per 8 utterances, an upper bound of 3.2% of wall time.

    Dims 0 and 1 of that view cannot be merged, so ``A.reshape(M, K)`` would copy
    too; the only genuinely copy-free route is a strided *batched* GEMM, and that
    decomposes into ``T`` separate ``(B, C) @ (C, N)`` matmuls — four rows each.
    Measured at Zipformer's own shapes, graph-captured:

        T=478 B=4 C=192 N=192   copy + one GEMM  8.80 us   strided batched 11.63
        T=478 B=4 C=192 N=384   copy + one GEMM  9.13 us   strided batched 22.88
        T=295 B=4 C=192 N=768   copy + one GEMM  9.05 us   strided batched 27.35
        T=148 B=4 C=384 N=384   copy + one GEMM  5.83 us   strided batched 14.01

    So the copy *buys* 1.3-3.0x by turning a pathological batch of thin matmuls
    into one wide GEMM.  Removing it would be a regression.
    """
    if not A.is_contiguous():
        A = A.contiguous()
    if out is None:
        shape = A.shape
        out = A.new_empty(shape[0], N) if len(shape) == 2 else A.new_empty(*shape[:-1], N)
    return A, out, A.numel() // A.shape[-1]


@functools.cache
def _gemm_fn(compile_name: str, activation: bool):
    """Resolve a compiled GEMM variant by ``compile_name`` (cached → a stable
    callable per shape, which keeps CUDA-graph capture/replay deterministic)."""
    suffix = "_activation" if activation else ""
    return getattr(_get_gemm_module(), f"gemm_{compile_name}{suffix}")


# ---------------------------------------------------------------------------
# Resolved-dispatch cache
# ---------------------------------------------------------------------------
#
# The choice is a pure function of ``(op, M, N, K, dtype)`` — that is what makes
# it CUDA-graph safe — so resolving it once per distinct shape is sufficient, and
# the per-call work drops to one dict lookup plus one identity compare.  Measured
# at the LSTM input projection (M=64, N=2560, K=640), per call:
#
#     select_default_config          0.56 us   dict get + str(dtype) + rule scan
#     choice.compile_name            0.72 us   a *property*, rebuilt every access
#     _gemm_fn(name, act) cache hit  0.08 us
#     getattr(choice, "split_k", 1)  ~0.1 us
#
# ``compile_name`` was the surprise: it looks like an attribute and is a list
# comprehension plus a ``"_".join``, so the "cached" variant lookup paid to
# rebuild its own cache key on every call.
#
# Keyed on the resolver's identity as well, because the dispatch tests
# monkeypatch ``oasr.jit.gemm.select_default_config`` and a shape-only cache
# would serve them the previous test's decision.  Holding the reference (rather
# than an ``id()``) also rules out address reuse after a patched lambda is
# collected.
_PLAN_CUTLASS = 0
_PLAN_TORCH = 1
_PLAN_FUSED = 2
_PLAN_DEFAULT = 3

#: ``(op, activation, M, N, K, dtype)`` -> ``(resolver, kind, fn, split_k)``.
_PLANS: dict = {}

#: A model with a wide spread of M values (long-form, varlen) would otherwise
#: grow this without bound.  Rebuilding an entry costs ~1.4 us, so dropping the
#: whole table on overflow is cheaper than tracking recency.
_PLAN_CACHE_MAX = 4096

_PLAN_WARNED: set = set()


def _plan(op: str, activation: bool, M: int, N: int, K: int, dtype) -> tuple:
    """Resolve — and memoise — the dispatch decision for one shape.

    Returns ``(resolver, kind, fn, split_k)``.  ``kind`` is one of the
    ``_PLAN_*`` constants; ``fn`` is the compiled variant for ``_PLAN_CUTLASS``
    and ``None`` otherwise.
    """
    resolver = _jit_gemm.select_default_config
    key = (op, activation, M, N, K, dtype)
    plan: Optional[tuple] = _PLANS.get(key)
    if plan is not None and plan[0] is resolver:
        return plan
    try:
        from oasr.jit.gemm import GEMM_DEFAULT

        choice = resolver(op, M, N, K, dtype, _target_sm())
        if choice == "torch":
            plan = (resolver, _PLAN_TORCH, None, 1)
        elif choice == "fused":
            plan = (resolver, _PLAN_FUSED, None, 1)
        elif choice is GEMM_DEFAULT and op == "gemm_log_softmax":
            # The CTC head's rule-less fallback is the fused launcher, not a
            # composed CUTLASS variant (see _dispatch_gemm_log_softmax).
            plan = (resolver, _PLAN_DEFAULT, None, 1)
        else:
            # split_k is a CUTLASS 2.x runtime argument; the SM90+ configs have
            # no such knob (their collective mainloop pipelines K itself), so it
            # is read defensively.  Reading it unconditionally used to raise
            # AttributeError on SM90 and the bare ``except`` turned that into a
            # silent fall-through -- which made the whole shape-aware rule table
            # inert on Hopper.
            name = choice.compile_name
            fn = _bmm_fn(name) if op == "bmm" else _gemm_fn(name, activation)
            plan = (resolver, _PLAN_CUTLASS, fn, getattr(choice, "split_k", 1))
    except Exception:
        # A shape pinned to the fixed default is a degradation, not a crash, and
        # it used to be completely silent.  Say it once per (op, N, K).
        plan = (resolver, _PLAN_DEFAULT, None, 1)
        wkey = (op, N, K)
        if wkey not in _PLAN_WARNED:
            _PLAN_WARNED.add(wkey)
            logger.warning(
                "GEMM dispatch for %s N=%d K=%d could not be resolved; "
                "using the fixed default variant",
                op,
                N,
                K,
                exc_info=True,
            )
    if len(_PLANS) >= _PLAN_CACHE_MAX:
        _PLANS.clear()
    _PLANS[key] = plan
    return plan


def _dispatch_gemm(out, A, B, C, N, K, M) -> None:
    """Shape-aware non-tuning dispatch for ``gemm``.

    Picks a per-shape config via the heuristic rules (CUTLASS variant or torch/
    cuBLAS); any failure falls through to the fixed ``GEMM_DEFAULT`` so a bad rule
    can never break inference.

    ``A``/``out`` arrive N-D — the CUTLASS launchers flatten internally.  Only
    the cuBLAS escape needs 2-D views, and it is the rare branch (5 of 27 rules),
    so it pays for its own reshapes.

    The decision itself is memoised per shape by :func:`_plan`; only the launch
    remains per-invocation.
    """
    _, kind, fn, split_k = _plan("gemm", False, M, N, K, A.dtype)
    try:
        if kind == _PLAN_CUTLASS:
            fn(out, A, B, C, split_k)
            return
        if kind == _PLAN_TORCH:
            from oasr.functionals.gemm_torch import torch_gemm

            torch_gemm(out.reshape(M, N), A.reshape(M, K), B, C, 1)
            return
    except Exception:
        pass
    _default_gemm_fn()(out, A, B, C, 1)


def _dispatch_gemm_activation(out, A, B, C, activation_type, N, K, M) -> None:
    """Shape-aware non-tuning dispatch for ``gemm_activation`` (see ``_dispatch_gemm``)."""
    _, kind, fn, split_k = _plan("gemm_activation", True, M, N, K, A.dtype)
    try:
        if kind == _PLAN_CUTLASS:
            fn(out, A, B, C, activation_type, split_k)
            return
        if kind == _PLAN_TORCH:
            from oasr.functionals.gemm_torch import torch_gemm_activation

            torch_gemm_activation(out.reshape(M, N), A.reshape(M, K), B, C, activation_type, 1)
            return
    except Exception:
        pass
    _default_gemm_activation_fn()(out, A, B, C, activation_type, 1)


@functools.cache
def _bmm_fn(compile_name: str):
    """Resolve a compiled BMM variant by ``compile_name`` (cached, graph-safe)."""
    return getattr(_get_bmm_module(), f"bmm_{compile_name}")


def _dispatch_bmm(out, A, B, N, K) -> None:
    """Shape-aware non-tuning dispatch for ``bmm``.

    The rule table is keyed on ``("bmm", N, K)`` with M thresholds (the batch
    dimension is not part of the key).  BMM has no Stream-K / parallel split-K
    variants and no runtime split-K argument, so a rule choice only selects a
    tile.  Any failure falls through to the fixed default.
    """
    _, kind, fn, _sk = _plan("bmm", False, A.shape[1], N, K, A.dtype)
    try:
        if kind == _PLAN_CUTLASS:
            fn(out, A, B)
            return
        if kind == _PLAN_TORCH:
            from oasr.functionals.gemm_torch import torch_bmm

            torch_bmm(out, A, B)
            return
    except Exception:
        pass
    _default_bmm_fn()(out, A, B)


def _log_softmax_inplace(out2d) -> None:
    """Row-wise in-place log_softmax via the OASR online kernel."""
    from oasr.functionals.softmax import _get_softmax_module

    _get_softmax_module().log_softmax(out2d, out2d)


def _dispatch_gemm_log_softmax(out, A, B, C, N, K, M) -> None:
    """Shape-aware non-tuning dispatch for ``gemm_log_softmax`` (the CTC head).

    Choices (per the ``("gemm_log_softmax", N, K)`` rules):
      * ``"torch"``   — cuBLAS addmm + torch log_softmax (in place);
      * ``"fused"``   — the legacy single-call CUTLASS launcher (fixed 16x128
        tile; best at tiny streaming M);
      * a ``CutlassGemmConfig`` — that GEMM variant composed with the OASR
        online log_softmax kernel on the same buffer (identical numerics to
        the fused path, but the GEMM tile is shape-selected).

    Falls back to the legacy fused launcher (the historical behaviour) when no
    rule matches.

    An unaligned ``N``/``K`` used to be quietly rerouted to cuBLAS here, which
    made this the one GEMM entry point with a different contract from the rest
    of the family — ``gemm`` failed on the same input.  The launchers now all
    raise the same actionable error (``CHECK_GEMM_ALIGNMENT``), and the fix is
    to pad the projection at the model layer, which every in-tree caller does.
    """
    _, kind, fn, split_k = _plan("gemm_log_softmax", False, M, N, K, A.dtype)
    try:
        if kind == _PLAN_TORCH:
            from oasr.functionals.gemm_torch import torch_gemm_log_softmax

            torch_gemm_log_softmax(out.reshape(M, N), A.reshape(M, K), B, C, 1)
            return
        if kind == _PLAN_CUTLASS:
            fn(out, A, B, C, split_k)
            # The online log_softmax kernel normalises the trailing dim, so it
            # takes the N-D buffer as-is.
            _log_softmax_inplace(out)
            return
    except Exception:
        pass
    _get_gemm_log_softmax_module().gemm_log_softmax(out, A, B, C, 1)


@oasr_api
def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """General matrix multiplication: D = A @ B + C.

    Args:
        A: Input tensor [M, K] or [batch, M', K].
        B: Input tensor [N, K].
        C: Optional bias tensor [M, N].
        out: Optional pre-allocated output tensor.

    Returns:
        Output tensor [M, N].
    """
    K = A.shape[-1]
    N = B.shape[0]
    A, out, M = _gemm_io(A, N, out)

    if is_tuning_enabled():
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        get_tuner().dispatch(
            op_key=OpKey("gemm", "gemm"),
            shape_sig=(M, N, K),
            dtype=A.dtype,
            device=A.device,
            runner_args=(out.reshape(-1, N), A.reshape(-1, K), B, C),
        )
        return out

    _dispatch_gemm(out, A, B, C, N, K, M)
    return out


@oasr_api
def bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched matrix multiplication: D[b] = A[b] @ B[b].

    Args:
        A: Input [batch_size, M, K].
        B: Input [batch_size, N, K].
        out: Optional pre-allocated output tensor.

    Returns:
        Output [batch_size, M, N].
    """
    batch_size, M, K = A.shape
    N = B.shape[1]
    if out is None:
        out = torch.empty(batch_size, M, N, device=A.device, dtype=A.dtype)

    from oasr.tune import is_tuning_enabled

    if is_tuning_enabled():
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        get_tuner().dispatch(
            op_key=OpKey("gemm", "bmm"),
            shape_sig=(batch_size, M, N, K),
            dtype=A.dtype,
            device=A.device,
            runner_args=(out, A, B),
        )
        return out

    _dispatch_bmm(out, A, B, N, K)
    return out


@oasr_api
def group_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    offset: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Grouped GEMM with variable problem sizes.

    Args:
        A: Input tensor [L, K].
        B: Input tensor [B, N, K].
        offset: Offset tensor [B] (cumulative row counts).
        out: Optional pre-allocated output tensor.

    Returns:
        Output tensor [L, N].
    """
    L, K = A.shape
    B_count = B.shape[0]
    N = B.shape[1]
    if out is None:
        out = torch.empty(L, N, device=A.device, dtype=A.dtype)

    from oasr.tune import is_tuning_enabled

    if is_tuning_enabled():
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        get_tuner().dispatch(
            op_key=OpKey("gemm", "group_gemm"),
            shape_sig=(L, B_count, N, K),
            dtype=A.dtype,
            device=A.device,
            runner_args=(out, A, B, offset),
        )
        return out

    _default_group_gemm_fn()(out, A, B, offset)
    return out


@oasr_api
def gemm_activation(
    A: torch.Tensor,
    B: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    activation_type: int = 2,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GEMM with fused activation: D = activation(A @ B + C).

    Args:
        A: Input tensor [M, K] or [batch, M', K].
        B: Input tensor [N, K].
        C: Optional bias tensor.
        activation_type: Activation type (0=RELU, 1=tanh-GELU, 2=SWISH,
            4=exact-erf GELU).
        out: Optional pre-allocated output tensor.

    Returns:
        Output tensor [M, N].
    """
    K = A.shape[-1]
    N = B.shape[0]
    A, out, M = _gemm_io(A, N, out)

    if is_tuning_enabled():
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        get_tuner().dispatch(
            op_key=OpKey("gemm", "gemm_activation"),
            shape_sig=(M, N, K),
            dtype=A.dtype,
            device=A.device,
            runner_args=(out.reshape(-1, N), A.reshape(-1, K), B, C, activation_type),
        )
        return out

    _dispatch_gemm_activation(out, A, B, C, activation_type, N, K, M)
    return out


@oasr_api
def gemm_log_softmax(
    A: torch.Tensor,
    B: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused GEMM + log_softmax: D = log_softmax(A @ B.T + C, dim=-1).

    Drop-in replacement for ``F.log_softmax(F.linear(A, B, C), dim=-1)`` such
    as the CTC head ``F.log_softmax(self.ctc_lo(hidden_states), dim=2)``.

    Args:
        A: Activations [*, K]. Leading dims are flattened to M = prod(A.shape[:-1]).
        B: Weight matrix [N, K] (e.g. ``nn.Linear.weight``).
        C: Optional bias [N], broadcast over rows.
        out: Optional pre-allocated output [*, N] with A's dtype.

    Returns:
        Output tensor with the same leading shape as A and last dim N.
    """
    K = A.shape[-1]
    N = B.shape[0]
    A, out, M = _gemm_io(A, N, out)

    if is_tuning_enabled():
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        get_tuner().dispatch(
            op_key=OpKey("gemm", "gemm_log_softmax"),
            shape_sig=(M, N, K),
            dtype=A.dtype,
            device=A.device,
            runner_args=(out.reshape(-1, N), A.reshape(-1, K), B, C),
        )
        return out

    _dispatch_gemm_log_softmax(out, A, B, C, N, K, M)
    return out
