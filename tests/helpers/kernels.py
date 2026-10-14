# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Assertions and gates every kernel test needs.

Each of these existed in three to twenty-three copies.  That is not only
duplication: a copied assertion drifts, and the weakest copy is the one that
decides what the suite actually checks.  Half the hand-written
destination-passing tests compared ``data_ptr`` and nothing else, so they
would have passed on a launcher that wrote the right address and the wrong
numbers.  :func:`assert_dest_passing` checks both, once.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Tuple

import pytest
import torch
import torch.nn.functional as F

from oasr.functionals.activation import _ACTIVATION_NAME_TO_ID

__all__ = [
    "ACTIVATIONS",
    "ACTIVATION_IDS",
    "assert_dest_passing",
    "assert_graph_replay",
    "capture_graph",
    "device_sm",
    "occupy_free_memory",
    "requires_cute",
    "requires_sm",
    "written_sentinels",
]


def device_sm() -> int:
    """Compute capability as the JIT spells it, e.g. 90 for Hopper, 120 for SM120."""
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def requires_sm(*sms: int, what: str = "kernel"):
    """Skip unless the running device is one of ``sms``."""
    return pytest.mark.skipif(
        device_sm() not in sms,
        reason=f"the {what} is built for SM{'/'.join(str(s) for s in sms)} only",
    )


def requires_cute(jit_module: Any, what: str):
    """Skip unless a CuTeDSL kernel exists for this arch.

    ``jit_module`` is the ``oasr.jit`` module that owns the kernel; it declares
    the arches it was written for in ``_SUPPORTED_SM``.  Reading the declaration
    is the point -- a hand-maintained arch list in a test is a second source of
    truth that goes stale the day a kernel gains an arch.
    """
    return pytest.mark.skipif(
        device_sm() not in getattr(jit_module, "_SUPPORTED_SM", ()),
        reason=f"no CuTeDSL {what} kernel for this arch",
    )


#: name -> torch reference, for every activation the fused epilogues accept.
#: Keyed off the production table so a new activation cannot be added to the
#: kernels and silently left untested.  Note ``gelu`` maps to the *tanh*
#: approximation: that is what the CUDA epilogue computes, and fusing it under
#: the exact-erf name is the silent accuracy change AGENTS.md warns about.
_TORCH_REF: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "gelu": lambda t: F.gelu(t, approximate="tanh"),
    "gelu_tanh": lambda t: F.gelu(t, approximate="tanh"),
    "gelu_erf": F.gelu,
    "swish": F.silu,
    "silu": F.silu,
}
ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    name: _TORCH_REF[name] for name in _ACTIVATION_NAME_TO_ID
}
#: name -> the integer id the launchers take.
ACTIVATION_IDS = dict(_ACTIVATION_NAME_TO_ID)


def assert_dest_passing(
    fn: Callable[..., Any],
    *args: Any,
    out: torch.Tensor,
    expected: torch.Tensor | None = None,
    outs: Iterable[torch.Tensor] = (),
    **kwargs: Any,
) -> Any:
    """Rule 4: the output tensor is the first parameter, and it is *the* output.

    Asserts the returned tensor aliases ``out`` -- and, when ``expected`` is
    given, that ``out`` holds the right numbers.  A ``data_ptr`` check alone
    passes on a launcher that writes the correct address and the wrong values.

    ``outs`` names any further destination tensors a multi-output launcher
    takes (``add_rms_norm_residual``'s ``residual_out``, say); each is checked
    for aliasing against the matching element of the returned tuple.
    """
    ret = fn(*args, out=out, **kwargs)
    primary = ret[0] if isinstance(ret, tuple) else ret
    assert primary.data_ptr() == out.data_ptr(), f"{fn.__name__} did not write into out="
    for i, extra in enumerate(outs, start=1):
        assert ret[i].data_ptr() == extra.data_ptr(), f"{fn.__name__} ignored destination #{i}"
    if expected is not None:
        torch.testing.assert_close(out, expected)
    return ret


def capture_graph(launch: Callable[[], Any], *, warmup: int = 1) -> Tuple[Any, Any]:
    """Warm ``launch`` up on a side stream, capture it, and return ``(graph, result)``.

    ``result`` is what the captured call returned: its storage is the graph's
    output, rewritten by every ``graph.replay()``.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            launch()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = launch()
    return graph, result


#: The byte :func:`occupy_free_memory` fills with: every bit set, which reads as
#: NaN in any floating dtype and as -1 in any signed integer one.
_SENTINEL = 0xFF


def occupy_free_memory() -> List[torch.Tensor]:
    """Hand every free block of the small allocator pool to an all-ones sentinel.

    A captured graph keeps the addresses it was captured with, so storage freed
    after a capture is a defect that stays invisible until something else lands
    there -- and whether anything does is up to the allocator.  This makes
    something land there: afterwards, a replay that reads freed memory reads NaN
    (or -1), and one that writes into it changes a sentinel, which
    :func:`written_sentinels` counts.  The small pool serves every request up to
    1 MiB on the current stream, which covers a lengths vector or a positional
    table; 512 bytes is its block granule, so the sentinels tile it exactly.
    """
    stats = torch.cuda.memory_stats()
    free = stats["reserved_bytes.small_pool.current"] - stats["allocated_bytes.small_pool.current"]
    return [
        torch.full((512,), _SENTINEL, dtype=torch.uint8, device="cuda") for _ in range(free // 512)
    ]


def written_sentinels(sentinels: List[torch.Tensor]) -> int:
    """How many of :func:`occupy_free_memory`'s sentinels something wrote into."""
    if not sentinels:
        return 0
    return int(torch.stack(sentinels).ne(_SENTINEL).any(dim=1).sum())


def assert_graph_replay(
    launch: Callable[[], Any],
    *,
    out: torch.Tensor,
    mutate: Callable[[], None],
    expected: Callable[[], torch.Tensor],
    warmup: int = 3,
    **tolerance: float,
) -> None:
    """Capture ``launch`` into a CUDA graph and prove a replay recomputes.

    Two failures this catches and an eager test cannot: an allocation inside
    the launcher, and a host-side trip count that is read from data rather than
    from shapes.  ``mutate`` writes new values *through the captured buffers*,
    so a replay that reproduced the captured result instead of recomputing
    fails against ``expected()``.
    """
    graph, _ = capture_graph(launch, warmup=warmup)
    mutate()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected(), **tolerance)
