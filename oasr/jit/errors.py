# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Exceptions a JIT-compiled launcher raises, by their TVM-FFI error kind.

A launcher throws ``TVM_FFI_THROW(Kind)``; TVM-FFI maps the kind to the Python
class registered under that name, and to ``RuntimeError`` when none is.  The
classes here subclass ``RuntimeError``, so a caller that catches that still
catches them.
"""

from __future__ import annotations

__all__ = ["VariantUnsupportedError", "register"]


class VariantUnsupportedError(RuntimeError):
    """A compiled kernel variant cannot run this problem.

    Raised by ``CHECK_GEMM_STATUS`` (``csrc/tvm_ffi_utils.h``) when CUTLASS's
    ``can_implement`` refuses the problem or the variant has no path for the
    arguments -- before anything is launched, so the device is untouched.  It is
    the one launcher failure the GEMM / BMM / Conv1D dispatch answers with the
    default variant; any other failure is a plain ``RuntimeError`` and reaches the
    caller, because relaunching after it would hide a CUDA error until a later
    call trips over it.
    """


#: TVM-FFI error kind -> the class it raises as.
_KINDS = {"OasrVariantUnsupported": VariantUnsupportedError}

_registered = False


def register() -> None:
    """Map OASR's error kinds to their classes.  Idempotent.

    Called before the first library loads (``JitSpec.build_and_load``), so no
    launcher can raise one of these kinds as a bare ``RuntimeError``.
    """
    global _registered
    if _registered:
        return
    import tvm_ffi

    for kind, cls in _KINDS.items():
        tvm_ffi.error.register_error(kind, cls)
    _registered = True
