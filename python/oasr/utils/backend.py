# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""C extension dtype mapping for OASR layers."""


import torch
import torch.nn as nn
import oasr


DTYPE_TO_OASR_DTYPE = {
    torch.float32: oasr.DataType.FP32,
    torch.float16: oasr.DataType.FP16,
    torch.bfloat16: oasr.DataType.BF16
}

ACTIVATION_TYPE_TO_OASR_ACTIVATION_TYPE = {
    "swish": oasr.ActivationType.SWISH,
    "silu": oasr.ActivationType.SWISH,
    "relu": oasr.ActivationType.RELU,
    "gelu": oasr.ActivationType.GELU
}

NORM_TYPE_TO_OASR_NORM_TYPE = {
    "layer_norm": oasr.NormType.LAYER_NORM,
    "rms_norm": oasr.NormType.RMS_NORM,
    "batch_norm": oasr.NormType.BATCH_NORM,
    "group_norm": oasr.NormType.GROUP_NORM,
}

NORM_TYPE_TO_OASR_NORM = {
    "layer_norm": oasr.layers.norm.LayerNorm,
    "rms_norm": oasr.layers.norm.RMSNorm,
    "batch_norm": oasr.layers.norm.BatchNorm1d,
    "group_norm": oasr.layers.norm.GroupNorm,
}

ACTIVATION_TYPE_TO_TORCH_ACTIVATION = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
}


def get_dtype(dtype):
    return DTYPE_TO_OASR_DTYPE.get(dtype, oasr.DataType.FP16)


def get_activation_type(activation_type):
    return ACTIVATION_TYPE_TO_OASR_ACTIVATION_TYPE.get(activation_type.lower(), oasr.ActivationType.SWISH)

def get_activation(activation_type):
    return ACTIVATION_TYPE_TO_TORCH_ACTIVATION.get(activation_type.lower(), nn.SiLU())


def get_norm_type(norm_type):
    return NORM_TYPE_TO_OASR_NORM_TYPE.get(norm_type.lower(), oasr.NormType.BATCH_NORM)

def get_norm(norm_type):
    return NORM_TYPE_TO_OASR_NORM.get(norm_type.lower(), oasr.layers.norm.BatchNorm1d)