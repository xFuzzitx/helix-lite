"""HELIX-Lite — KVQuant nuq2 (2-bit non-uniform KV cache quantization).

Adaptation of SqueezeAILab/KVQuant (Hooper et al., 2024) for vLLM 0.20.1
and Qwen2.5-7B-Instruct-1M on sm_86. Three layers:

1. ``nuq``: pure-PyTorch reference implementation of non-uniform
   quantization with dense + sparse outliers. Used for calibration and
   correctness checks.

2. ``calibration``: port of KVQuant's Fisher-information-weighted
   k-means calibration loop, modernised for current ``transformers``
   and Qwen2 architecture (per-channel pre-RoPE Keys, per-token Values).

3. ``triton_kernels`` (TODO): Triton kernels that pack/unpack 2-bit nuq
   values with their dense outliers, hooked into vLLM's PagedAttention
   via a custom ``AttentionImpl``.

The reference paper is ``arXiv:2401.18079``; the upstream repo
``github.com/SqueezeAILab/KVQuant`` targets transformers + Llama and
flat KV cache, neither of which we use directly. We keep its
calibration math and re-implement everything else.
"""
from .nuq import quantize_nuq, dequantize_nuq, find_outliers
from .scales import KVScales, PerChannelScale, PerTokenScale

__all__ = [
    "quantize_nuq",
    "dequantize_nuq",
    "find_outliers",
    "KVScales",
    "PerChannelScale",
    "PerTokenScale",
]
