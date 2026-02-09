# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""C extension backend and dtype helpers for OASR layers."""


def _backend():
    """Return the oasr._C extension module (lazy import)."""
    try:
        import oasr._C as _C
        return _C
    except ImportError:
        from importlib.util import find_spec

        if find_spec("oasr") and getattr(__import__("oasr"), "_C", None) is not None:
            import oasr._C as _C
            return _C
        raise ImportError(
            "oasr._C extension not found. Build the project with pip install -e . or set PYTHONPATH."
        )


def _torch_dtype_to_oasr(dtype):
    """Map torch dtype to oasr DataType."""
    import torch

    _C = _backend()
    m = {
        torch.float32: _C.DataType.FP32,
        torch.float16: _C.DataType.FP16,
        torch.bfloat16: getattr(_C.DataType, "BF16", _C.DataType.FP16),
    }
    return m.get(dtype, _C.DataType.FP16)
