# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint bundles + the native OASR on-disk format.

* :class:`ConvertedCheckpoint` — everything a conversion yields (config,
  weights, aux, tokenizer / feature / decoding specs).
* :func:`convert_checkpoint` — adapter running any converter (new ``convert()``
  or legacy 4-method protocol) into a bundle.
* :mod:`~oasr.checkpoints.native` — round-trippable ``oasr_config.json`` +
  ``model.safetensors`` format; ``oasr-convert`` CLI in
  :mod:`~oasr.checkpoints.convert`.
"""

from .bundle import (
    ConvertedCheckpoint,
    DecodingDefaults,
    convert_checkpoint,
    sniff_legacy_sentencepiece,
    sniff_legacy_tokenizer_spec,
)
from .native import (
    FORMAT_VERSION,
    NATIVE_CONFIG_NAME,
    NATIVE_WEIGHTS_NAME,
    is_native_checkpoint,
    load_native,
    load_native_weights,
    read_native_config,
    register_aux_builder,
    save_native,
)

__all__ = [
    "ConvertedCheckpoint",
    "DecodingDefaults",
    "convert_checkpoint",
    "sniff_legacy_sentencepiece",
    "sniff_legacy_tokenizer_spec",
    "FORMAT_VERSION",
    "NATIVE_CONFIG_NAME",
    "NATIVE_WEIGHTS_NAME",
    "is_native_checkpoint",
    "load_native",
    "load_native_weights",
    "read_native_config",
    "register_aux_builder",
    "save_native",
]
