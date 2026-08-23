# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CuTeDSL fused recurrent-step kernels.

What is here: an Ampere-style composition -- cp.async multistage ring, swizzled
shared memory, ``ldmatrix.x4``, warp-level ``mma.sync`` m16n8k16, FP32
accumulate -- which is what SM80 through SM120 all actually run for FP16/BF16.
GeForce Blackwell has no FP16 tcgen05 path, so its own CuTeDSL GEMM uses the same
warp-level atom.

What is deliberately not here, and why:

* **TMA loads.** Available from SM90, and SM120's reference GEMM uses them for A,
  B *and* C.  This kernel's epilogue consumes two extra tensors beyond A/B/C
  (``previous_c``) and writes two (``h`` and ``c``), and routing those through TMA
  descriptors is a large change for a part of the kernel that touches each element
  exactly once.  TMA belongs on the K loop if anywhere; it is a measured follow-up,
  not an assumed win.
* **wgmma / tcgen05.** SM90's warpgroup MMA and SM100's tcgen05 need their own
  mainloop and epilogue, and neither can be validated on the GeForce Blackwell
  part this was developed on.  Declared as a gap rather than shipped untested.
"""

from oasr.kernels.cute.recurrent.step import RecurrentStepCute

__all__ = ["RecurrentStepCute"]
