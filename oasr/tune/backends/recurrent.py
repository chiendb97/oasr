# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUTLASS recurrent-tactic registrations.

The recurrent launcher contains all variants in one JIT module.  Tactics only
select an M tile and reduction strategy, so profiling never recompiles code.
"""

import functools
from typing import Callable, List, Tuple

from oasr.tune.autotuner import BackendEntry, OpKey, Tactic, _global_registry


@functools.cache
def _get_recurrent_module():
    from oasr.jit.recurrent import gen_recurrent_module

    return gen_recurrent_module().build_and_load()


def _make_runner(function_name: str, tactic: int, split_k_slices: int):
    def get_runner():
        function = getattr(_get_recurrent_module(), function_name)

        def call(*args):
            function(*args, tactic, split_k_slices)

        return call

    return get_runner


@functools.cache
def _has_tma_tactics() -> bool:
    """Is this device one whose target compiles the CUTLASS 3.x recurrent arms?

    The launcher only builds them for SM90 and SM100 — the two targets whose 3.x
    OpClassTensorOp builders accept FP16/BF16 — and refuses the tactic ids
    everywhere else. Gate here so other GPUs never profile a tactic that raises.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor in (90, 100)


def _always_available() -> bool:
    return True


#: ``(name, tactic id, split_k_slices, availability predicate)``.
_TacticSpec = Tuple[str, int, int, Callable[[], bool]]

_COMMON_TACTICS: Tuple[_TacticSpec, ...] = (
    ("fused_16x64", 0, 1, _always_available),
    ("fused_32x64", 1, 1, _always_available),
    ("fused_64x64", 2, 1, _always_available),
    ("stream_k", 3, 1, _always_available),
    ("parallel_split_k_2", 4, 2, _always_available),
    ("parallel_split_k_4", 4, 4, _always_available),
    ("parallel_split_k_8", 4, 8, _always_available),
    # CUTLASS 3.x TMA warp-specialized. The collective mainloop owns its own K
    # pipeline, so these carry no split-K variants.
    ("tma_64", 6, 1, _has_tma_tactics),
    ("tma_128", 7, 1, _has_tma_tactics),
)


def _register(op: str, function_name: str, include_serial_split_k: bool) -> None:
    tactics: List[_TacticSpec] = list(_COMMON_TACTICS)
    if include_serial_split_k:
        tactics.extend(
            (
                ("serial_split_k_2", 5, 2, _always_available),
                ("serial_split_k_4", 5, 4, _always_available),
                ("serial_split_k_8", 5, 8, _always_available),
            )
        )
    for name, tactic, split_k_slices, is_available in tactics:
        _global_registry.register(
            OpKey("recurrent", op),
            BackendEntry(
                tactic=Tactic(
                    "cutlass_recurrent",
                    config=(("name", name), ("split_k_slices", split_k_slices)),
                ),
                is_available=is_available,
                get_runner=_make_runner(function_name, tactic, split_k_slices),
                is_fallback=name == "fused_32x64",
            ),
        )


_register("lstm", "lstm_gemm_layer", include_serial_split_k=True)
_register("rnn_tanh", "rnn_gemm_layer", include_serial_split_k=False)
_register("rnn_relu", "rnn_gemm_layer", include_serial_split_k=False)
