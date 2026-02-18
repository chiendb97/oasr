# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""C extension dtype mapping for OASR layers."""


import torch

import oasr._C as _C


_DTYPE_TO_OASR = {
    torch.float32: _C.DataType.FP32,
    torch.float16: _C.DataType.FP16,
    torch.bfloat16: _C.DataType.BF16
}

def _torch_dtype_to_oasr(dtype):
    """Map torch dtype to oasr DataType."""

    return _DTYPE_TO_OASR.get(dtype, _C.DataType.FP16)
