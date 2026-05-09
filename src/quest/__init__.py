"""HELIX-Lite — Quest top-K page selection (PR3).

Decode-time KV-cache sparsity: at each decode step, score every KV
page (block of ~16 tokens) by an upper bound on its contribution to
attention, then keep only the top-K pages. This collapses the work
from O(seq_len) to O(K * page_size) per attention call, with the
seq_len-dependent part now in the upper-bound scoring (which is much
cheaper than full attention because it operates on a (page_count,
heads, head_dim) tensor instead of (seq_len, heads, head_dim)).

The math is from Quest (Tang et al., ICML'24). Implementation order
mirrors what worked for nuq4 in PR1b:
1. Pure-PyTorch reference (this module's :mod:`reference`).
2. Triton kernels matching the reference bit-for-bit.
3. vLLM integration via custom AttentionImpl + page-stat updates on
   cache writes.

Streaming-LLM attention sinks (PR4) come for free here: we always
include the first ``sink_pages`` pages in the selected set,
regardless of their upper-bound score. Two PRs in one slice.
"""
from .reference import (
    compute_page_stats,
    page_upper_bound,
    topk_pages_with_sinks,
    selected_attention,
)

__all__ = [
    "compute_page_stats",
    "page_upper_bound",
    "topk_pages_with_sinks",
    "selected_attention",
]
