# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Round-to-nearest: no transform; the weights quantize as they are.

The baseline every smarter algorithm is measured against, and the online path
(``EngineConfig.quantization``) — dynamic per-token activation scales need no
calibration, so RTN W8A8 needs no data at all.
"""

from __future__ import annotations

from typing import Optional

from torch import nn

from ..layout import QuantLayout
from ..spec import QuantizationSpec
from .base import ForwardLoop, QuantAlgorithm, register_quant_algorithm


@register_quant_algorithm("rtn")
class RoundToNearest(QuantAlgorithm):
    """Identity transform."""

    def apply(
        self,
        model: nn.Module,
        layout: QuantLayout,
        spec: QuantizationSpec,
        forward_loop: Optional[ForwardLoop],
    ) -> QuantizationSpec:
        return spec
