"""Behavioural tests for the Quest reference path.

Three things the reference must guarantee, each pinned by a test:

* ``compute_page_stats`` returns the per-page min/max correctly.
* ``page_upper_bound`` is *actually* an upper bound on the true
  per-page max ``q · K[t]``.
* ``topk_pages_with_sinks`` always includes the sink pages and
  returns indices sorted per head.
* ``selected_attention`` reduces to standard SDPA when all pages
  are selected.

Run:
    PYTHONPATH=src python src/quest/tests/test_reference.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest.reference import (
    compute_page_stats,
    page_upper_bound,
    selected_attention,
    topk_pages_with_sinks,
)


def test_page_stats_correct_min_max() -> None:
    torch.manual_seed(0)
    K = torch.randn(4, 16, 2, 8)              # (P, page, H, D)
    K_min, K_max = compute_page_stats(K)
    # spot-check page 1, head 0, dim 3
    expected_min = K[1, :, 0, 3].min()
    expected_max = K[1, :, 0, 3].max()
    assert torch.allclose(K_min[1, 0, 3], expected_min)
    assert torch.allclose(K_max[1, 0, 3], expected_max)


def test_upper_bound_is_actual_upper_bound() -> None:
    """For every page and every head, the computed bound must >= the
    true max_t q·K[t] across tokens t in that page."""
    torch.manual_seed(1)
    P, page, H, D = 8, 16, 4, 32
    K = torch.randn(P, page, H, D)
    q = torch.randn(H, D)
    K_min, K_max = compute_page_stats(K)
    bound = page_upper_bound(q, K_min, K_max)         # (P, H)
    # True max per page per head
    # K @ q.T over the head_dim, reduce max over the page slot.
    # einsum: (P, page, H, D) , (H, D) -> (P, page, H)
    scores = torch.einsum("pthd,hd->pth", K, q)
    true_max = scores.amax(dim=1)                     # (P, H)
    diff = bound - true_max
    # bound should be >= true_max (allowing 1e-5 fp slack)
    assert (diff >= -1e-4).all(), (
        f"bound is below true max somewhere: min slack {diff.min().item():.6f}"
    )


def test_topk_includes_sink_pages_and_is_sorted() -> None:
    torch.manual_seed(2)
    P, H = 16, 4
    bound = torch.randn(P, H)
    # Force the sinks (page 0) to have low score so they wouldn't
    # naturally make the cut.
    bound[0, :] = -100.0
    selected = topk_pages_with_sinks(bound, k=4, sink_pages=2)
    # sink_pages + k = 2 + 4 = 6 entries per head
    assert selected.shape == (6, 4)
    # Sinks (indices 0 and 1) must be present for every head
    for h in range(H):
        idx_set = set(selected[:, h].tolist())
        assert 0 in idx_set, f"head {h} missing sink page 0"
        assert 1 in idx_set, f"head {h} missing sink page 1"
        # And the column should be sorted
        assert torch.all(selected[:-1, h] <= selected[1:, h]), (
            f"head {h} indices not sorted: {selected[:, h].tolist()}"
        )


def test_topk_handles_small_page_count() -> None:
    bound = torch.randn(3, 2)
    # k larger than P - sinks; expect to get clipped, not crash
    selected = topk_pages_with_sinks(bound, k=10, sink_pages=1)
    # sinks + min(k, P-sinks) = 1 + 2 = 3
    assert selected.shape == (3, 2)


def test_selected_attention_reduces_to_full_when_all_pages_picked() -> None:
    torch.manual_seed(3)
    P, page, H, D = 4, 8, 2, 16
    K = torch.randn(P, page, H, D)
    V = torch.randn(P, page, H, D)
    q = torch.randn(H, D)
    # All pages, in order, for both heads
    all_pages = torch.arange(P).unsqueeze(1).expand(-1, H)
    out_quest = selected_attention(q, K, V, all_pages)

    # Reference dense SDPA over the full (P*page) tokens
    K_flat = K.reshape(P * page, H, D)
    V_flat = V.reshape(P * page, H, D)
    scale = 1.0 / math.sqrt(D)
    out_dense = torch.zeros(H, D)
    for h in range(H):
        scores = (K_flat[:, h] @ q[h]) * scale            # (P*page,)
        attn = torch.softmax(scores.float(), dim=0).to(q.dtype)
        out_dense[h] = attn @ V_flat[:, h]
    assert torch.allclose(out_quest, out_dense, atol=1e-3, rtol=1e-3), (
        f"selected (all pages) != dense; max diff "
        f"{(out_quest - out_dense).abs().max().item():.4e}"
    )


def test_selected_attention_drops_low_scoring_pages_by_design() -> None:
    """Sanity check: if we select only one page that's far from a
    second page where the high-scoring token actually lives, the
    Quest path should miss it. This is *expected* behaviour and is
    what distinguishes Quest from full attention - the test pins it
    so we notice if a refactor changes the contract."""
    torch.manual_seed(4)
    P, page, H, D = 4, 4, 1, 8
    K = torch.zeros(P, page, H, D)
    V = torch.zeros(P, page, H, D)
    K[2, 0, 0] = torch.tensor([1.0] + [0.0] * 7)        # high-scoring token
    V[2, 0, 0] = torch.tensor([5.0] + [0.0] * 7)        # distinctive value
    q = torch.tensor([[1.0] + [0.0] * 7])
    # Force-select only page 0 (low-scoring); page 2's high token is excluded
    page_indices = torch.tensor([[0]])
    out = selected_attention(q, K, V, page_indices)
    # The high V (5.0) shouldn't appear in the output because page 2 was excluded
    assert out[0, 0].item() < 1.0, (
        f"Quest output should not see page 2 when only page 0 is selected; got {out[0, 0]:.3f}"
    )


if __name__ == "__main__":
    tests = [
        ("page_stats_correct_min_max", test_page_stats_correct_min_max),
        ("upper_bound_is_actual_upper_bound", test_upper_bound_is_actual_upper_bound),
        ("topk_includes_sink_pages_and_is_sorted",
         test_topk_includes_sink_pages_and_is_sorted),
        ("topk_handles_small_page_count", test_topk_handles_small_page_count),
        ("selected_attention_reduces_to_full_when_all_pages_picked",
         test_selected_attention_reduces_to_full_when_all_pages_picked),
        ("selected_attention_drops_low_scoring_pages_by_design",
         test_selected_attention_drops_low_scoring_pages_by_design),
    ]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok  {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {name}: {e}")
        except Exception as e:
            fails += 1
            print(f"  ERR  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
