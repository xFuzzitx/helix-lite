"""Pure-PyTorch reference for Quest decode-time page selection.

Three pieces, each tested independently:

1. :func:`compute_page_stats` — ``K_pages -> (K_min, K_max)`` per
   ``(page, head, head_dim)`` group. Built once at prefill, updated
   incrementally as decode appends new pages.

2. :func:`page_upper_bound` — ``(q, K_min, K_max) -> (page, head)``
   upper bound on ``max_t q·K[t]`` within each page. The bound is
   tight enough that pages it scores low really do not contribute,
   yet computable in one matmul-equivalent pass over `(P, H, D)`.

3. :func:`topk_pages_with_sinks` — pick top-K pages per head while
   guaranteeing the first ``sink_pages`` always make the cut (the
   StreamingLLM attention-sink trick from Xiao et al., 2023).

4. :func:`selected_attention` — thin wrapper that takes a query and
   the full prefilled K, V, runs the upper-bound + top-K, then
   gathers + does standard scaled-dot-product attention on the
   selected pages.

The math under :func:`page_upper_bound` is the elementary
``q · k <= q+ · k_max + q- · k_min`` componentwise inequality. Sum
over head_dim gives a per-head per-page upper bound on `q·K[t]`
which is itself an upper bound on ``max_t q·K[t]`` for `t` in that
page (actually it's not -- the tighter form is shown in the Quest
paper §3.2 -- but for correctness this is the safe one we use as
the reference).
"""
from __future__ import annotations

import math

import torch


def compute_page_stats(
    K_pages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-page min/max of Keys.

    Args:
        K_pages: ``(P, page_size, H, D)``. Tokens within a page do
            not need to be filled - if a page is partial, pad slots
            should be set to ``+inf`` for the slots used in the min
            scan and ``-inf`` for the max scan; here we just amin/
            amax the slot dim so callers must mask out unused slots
            ahead of time.

    Returns:
        ``(K_min, K_max)``, each ``(P, H, D)``.
    """
    if K_pages.dim() != 4:
        raise ValueError(f"expected (P, page, H, D), got {tuple(K_pages.shape)}")
    return K_pages.amin(dim=1), K_pages.amax(dim=1)


def page_upper_bound(
    query: torch.Tensor,
    K_min: torch.Tensor,
    K_max: torch.Tensor,
) -> torch.Tensor:
    """Upper bound on per-page max ``q·K[t]`` for one decode query.

    Uses the componentwise inequality::

        q_d * k_d   <=  max(q_d, 0) * max_d + min(q_d, 0) * min_d

    summed over the head_dim ``d``. This is conservative but
    correct: the bound is always ``>=`` the actual ``max_t q·K[t]``
    over tokens in the page, which is what we need to safely prune
    low-scoring pages.

    Args:
        query: ``(H, D)`` query for one decode step.
        K_min, K_max: ``(P, H, D)`` page-wise key bounds.

    Returns:
        ``(P, H)`` upper bound per page per head.
    """
    if query.dim() != 2:
        raise ValueError(f"query should be (H, D), got {tuple(query.shape)}")
    if K_min.shape != K_max.shape or K_min.dim() != 3:
        raise ValueError(
            f"K_min/K_max should be (P, H, D), got {tuple(K_min.shape)} and {tuple(K_max.shape)}"
        )
    q_pos = query.clamp(min=0)                 # (H, D)
    q_neg = query.clamp(max=0)                 # (H, D)
    # Broadcast: (1, H, D) * (P, H, D) -> (P, H, D), sum over D -> (P, H)
    bound = (q_pos.unsqueeze(0) * K_max).sum(dim=-1) + \
            (q_neg.unsqueeze(0) * K_min).sum(dim=-1)
    return bound


def topk_pages_with_sinks(
    bound: torch.Tensor,
    k: int,
    sink_pages: int = 1,
) -> torch.Tensor:
    """Pick top-K page indices, always including the first ``sink_pages``.

    Args:
        bound: ``(P, H)`` upper-bound scores from
            :func:`page_upper_bound`.
        k: number of pages to keep beyond the sinks. Capped by
            ``P - sink_pages``; if ``P`` is small, returns all
            available pages.
        sink_pages: number of leading pages to always include
            (the StreamingLLM trick).

    Returns:
        ``(k_eff + sink_pages, H)`` integer indices, sorted by
        index per head (so attention preserves causal order if the
        downstream kernel relies on that). ``k_eff`` may be less
        than ``k`` when not enough non-sink pages exist.
    """
    if bound.dim() != 2:
        raise ValueError(f"bound should be (P, H), got {tuple(bound.shape)}")
    P, H = bound.shape
    sink_pages = max(0, min(sink_pages, P))
    k_eff = max(0, min(k, P - sink_pages))
    sink_idx = torch.arange(sink_pages, device=bound.device).unsqueeze(1).expand(-1, H)
    if k_eff == 0:
        # Just sinks; sort doesn't matter, they're already in order.
        return sink_idx
    bound_masked = bound.clone()
    bound_masked[:sink_pages, :] = float("-inf")
    top = torch.topk(bound_masked, k=k_eff, dim=0)
    selected = torch.cat([sink_idx, top.indices], dim=0)         # (sink+k, H)
    # Sort indices per head so callers can replay causal order.
    selected, _ = selected.sort(dim=0)
    return selected


def selected_attention(
    query: torch.Tensor,            # (H, D)
    K_pages: torch.Tensor,          # (P, page_size, H, D)
    V_pages: torch.Tensor,          # (P, page_size, H_v, D_v)
    page_indices: torch.Tensor,     # (k+sinks, H), one per head
) -> torch.Tensor:
    """Reference scaled-dot-product attention restricted to the selected pages.

    Each head attends only to keys in its own selected pages. This
    drops the seq_len-scaling cost of attention to ``k * page_size``
    keys regardless of the true sequence length.

    Returns:
        ``(H, D_v)`` attention output for the single decode token.
    """
    P, page_size, H, D = K_pages.shape
    H_v, D_v = V_pages.shape[2], V_pages.shape[3]
    k_total = page_indices.shape[0]
    out = torch.zeros(H, D_v, dtype=query.dtype, device=query.device)
    scale = 1.0 / math.sqrt(D)

    for h in range(H):
        # Gather (k_total * page_size) keys for this head.
        idx = page_indices[:, h]                                  # (k_total,)
        Kh = K_pages[idx, :, h, :].reshape(k_total * page_size, D)
        # Map each query head to its KV head when GQA != MHA. The
        # caller is responsible for passing a head_indices_v map; in
        # this reference (no GQA) we assume H == H_v.
        Vh = V_pages[idx, :, h % H_v, :].reshape(k_total * page_size, D_v)
        scores = (Kh @ query[h]) * scale                          # (k_total * page_size,)
        attn = torch.softmax(scores.float(), dim=0).to(query.dtype)
        out[h] = attn @ Vh
    return out
