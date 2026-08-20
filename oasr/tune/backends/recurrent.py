# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUTLASS recurrent-tactic registrations.

The recurrent launcher contains all variants in one JIT module.  Tactics only
select an M tile and reduction strategy, so profiling never recompiles code.
"""

import functools

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


_COMMON_TACTICS = (
    ("fused_16x64", 0, 1),
    ("fused_32x64", 1, 1),
    ("fused_64x64", 2, 1),
    ("stream_k", 3, 1),
    ("parallel_split_k_2", 4, 2),
    ("parallel_split_k_4", 4, 4),
    ("parallel_split_k_8", 4, 8),
)


def _register(op: str, function_name: str, include_serial_split_k: bool) -> None:
    tactics = list(_COMMON_TACTICS)
    if include_serial_split_k:
        tactics.extend(
            (
                ("serial_split_k_2", 5, 2),
                ("serial_split_k_4", 5, 4),
                ("serial_split_k_8", 5, 8),
            )
        )
    for name, tactic, split_k_slices in tactics:
        _global_registry.register(
            OpKey("recurrent", op),
            BackendEntry(
                tactic=Tactic(
                    "cutlass_recurrent",
                    config=(("name", name), ("split_k_slices", split_k_slices)),
                ),
                is_available=lambda: True,
                get_runner=_make_runner(function_name, tactic, split_k_slices),
                is_fallback=name == "fused_32x64",
            ),
        )


_register("lstm", "lstm_gemm_layer", include_serial_split_k=True)
_register("rnn_tanh", "rnn_gemm_layer", include_serial_split_k=False)
_register("rnn_relu", "rnn_gemm_layer", include_serial_split_k=False)
