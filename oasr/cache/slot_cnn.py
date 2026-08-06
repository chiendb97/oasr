# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``SlotCnnCache`` — the convolutional left-context descriptor, by its old name.

Kept as an alias of the generic :class:`~oasr.cache.state.SlotTensor` (whose
``slot_axis`` defaults to ``1``, this cache's layout) so the Conformer chunk
forward's parameter name, the CUDA-graph capture site and the public
``oasr.cache`` export all keep working unchanged.  New code should declare a
:class:`~oasr.cache.state.StreamStateSpec` and read the descriptor back out of
:meth:`~oasr.cache.state.SlotStateCache.views`; see ``oasr/cache/state.py`` for
why the single-tensor form was not general enough.
"""

from __future__ import annotations

from oasr.cache.state import SlotTensor

#: Slot-indexed CNN cache descriptor: ``(num_layers, max_batch_size,
#: cnn_cache_frames, hidden_dim)`` buffer + the ``(B,)`` active slot ids.
#: :meth:`~oasr.cache.state.SlotTensor.gather` materialises the per-batch left
#: context at the top of the encoder; :meth:`~oasr.cache.state.SlotTensor.scatter`
#: writes the post-chunk tail back in place at the bottom.
SlotCnnCache = SlotTensor

__all__ = ["SlotCnnCache"]
