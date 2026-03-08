# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""C extension dtype mapping for OASR layers."""


import torch

import oasr


TORCH_DTYPE_TO_OASR_DTYPE = {
    torch.float32: oasr.DataType.FP32,
    torch.float16: oasr.DataType.FP16,
    torch.bfloat16: oasr.DataType.BF16
}

STR_ACTIVATION_TO_OASR_ACTIVATION = {
    "swish": oasr.ActivationType.SWISH,
    "silu": oasr.ActivationType.SWISH,
    "relu": oasr.ActivationType.RELU,
    "gelu": oasr.ActivationType.GELU
}


def torch_dtype_to_oasr_dtype(dtype):
    return TORCH_DTYPE_TO_OASR_DTYPE.get(dtype, oasr.DataType.FP16)

def str_activation_to_oasr_activation(activation):
    return STR_ACTIVATION_TO_OASR_ACTIVATION.get(activation.lower(), oasr.ActivationType.SWISH)