# Performance benchmarks for OASR kernels
#
# Each kernel has its own benchmark file:
#   bench_gemm.py           - GEMM
#   bench_bmm.py            - Batched GEMM (BMM)
#   bench_group_gemm.py     - Grouped GEMM
#   bench_depthwise_conv1d.py - Depthwise Conv1D (standard + causal)
#   bench_pointwise_conv1d.py - Pointwise Conv1D (with/without activation)
#   bench_conv_block.py     - Conformer conv block (end-to-end)
#   bench_conv2d.py         - Conv2D NHWC (with/without fused activation)
#   bench_layer_norm.py     - LayerNorm (standard, fused add, fused activation)
#   bench_rms_norm.py       - RMSNorm
#   bench_group_norm.py     - GroupNorm
#   bench_batch_norm.py     - BatchNorm (standard, fused swish, fused activation)
#   bench_glu.py            - GLU activation
#   bench_swish.py          - Swish (SiLU) activation
