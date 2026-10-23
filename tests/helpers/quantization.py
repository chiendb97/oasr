# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Builders the quantization tests share across families.

The quantization package, its layer seam (``tests/models``) and its engine
integration (``tests/engine``) all need a small Conformer whose affine norms are
not the identity — a fold into ``γ = 1, β = 0`` would pass an invariance check
that a real checkpoint fails.  Imports stay inside the functions so collection
does not pull the model registry in.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["randomize_norms", "tiny_conformer", "tiny_conformer_config"]


def tiny_conformer_config(*, causal: bool = False, output_size: int = 32, **encoder: Any):
    """The config :func:`tiny_conformer` builds (also what ``save_native`` records)."""
    from oasr.models.conformer.config import ConformerEncoderConfig, ConformerModelConfig

    kwargs: dict[str, Any] = {
        "input_size": 80,
        "output_size": output_size,
        "attention_heads": 2,
        "linear_units": 64,
        "num_blocks": 2,
        "cnn_module_kernel": 7,
        "cnn_module_norm": "layer_norm",
        "causal": causal,
    }
    kwargs.update(encoder)
    return ConformerModelConfig(vocab_size=32, encoder=ConformerEncoderConfig(**kwargs))


def tiny_conformer(**overrides: Any):
    """A 2-block macaron + conv Conformer (pre-norm, LayerNorm conv norm), eval mode.

    Built in the default dtype on the host, with its norms randomized
    (:func:`randomize_norms`), so smoothing folds have something to move.
    ``overrides`` go to :func:`tiny_conformer_config`.
    """
    from oasr.models.conformer.model import ConformerModel

    torch.manual_seed(0)
    model = ConformerModel.from_config(tiny_conformer_config(**overrides)).eval()
    randomize_norms(model)
    return model


def randomize_norms(model: torch.nn.Module, seed: int = 1) -> None:
    """Give every LayerNorm / RMSNorm a non-trivial affine (γ in [0.5, 2], β in [-0.5, 0.5])."""
    from oasr.layers.norm import LayerNorm, RMSNorm

    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (LayerNorm, RMSNorm)):
                module.weight.copy_(torch.rand(module.weight.shape, generator=g) * 1.5 + 0.5)
                if module.bias is not None:
                    module.bias.copy_(torch.rand(module.bias.shape, generator=g) - 0.5)
