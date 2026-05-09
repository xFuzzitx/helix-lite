"""Tests for the EpisodeStore flat pool.

Run:
    PYTHONPATH=src python src/emllm/tests/test_episode_store.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emllm.episode_store import EpisodeStore


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"  # tests use GPU 0; production picks GPU 1
    return "cpu"


def test_add_and_repr() -> None:
    store = EpisodeStore(emb_dim=16, device=_device(), capacity=10)
    assert store.num_episodes == 0
    e = torch.randn(16)
    ep = store.add(e, (0, 100), surprise=2.5)
    assert ep.index == 0
    assert ep.token_range == (0, 100)
    assert ep.surprise == 2.5
    assert store.num_episodes == 1
    assert "1/10" in repr(store)


def test_topk_returns_self_for_query_equal_to_stored() -> None:
    store = EpisodeStore(emb_dim=8, device=_device(), capacity=4)
    e0 = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
    e1 = torch.tensor([0, 1.0, 0, 0, 0, 0, 0, 0])
    e2 = torch.tensor([0, 0, 1.0, 0, 0, 0, 0, 0])
    store.add(e0, (0, 10))
    store.add(e1, (10, 20))
    store.add(e2, (20, 30))
    top = store.topk(e1, k=1)
    assert len(top) == 1
    ep, score = top[0]
    assert ep.token_range == (10, 20), f"got {ep.token_range}"
    assert score > 0.99, f"cosine should be ~1, got {score}"


def test_topk_orders_by_similarity() -> None:
    store = EpisodeStore(emb_dim=4, device=_device(), capacity=8)
    # Three episodes spaced along the (1,1) direction with varying magnitude
    store.add(torch.tensor([1.0, 1.0, 0.0, 0.0]), (0, 1))
    store.add(torch.tensor([1.0, 0.5, 0.0, 0.0]), (1, 2))
    store.add(torch.tensor([0.0, 0.0, 1.0, 0.0]), (2, 3))   # orthogonal to query
    q = torch.tensor([1.0, 1.0, 0.0, 0.0])
    top = store.topk(q, k=3)
    # First should be the exact direction match
    assert top[0][0].token_range == (0, 1)
    # The orthogonal one should come last
    assert top[-1][0].token_range == (2, 3)
    assert top[-1][1] < 0.05, f"orthogonal cosine should be ~0, got {top[-1][1]}"


def test_capacity_exhaustion_raises() -> None:
    store = EpisodeStore(emb_dim=4, device=_device(), capacity=2)
    store.add(torch.zeros(4), (0, 1))
    store.add(torch.zeros(4), (1, 2))
    try:
        store.add(torch.zeros(4), (2, 3))
        raise AssertionError("expected RuntimeError on capacity exhaustion")
    except RuntimeError:
        pass


def test_empty_store_topk_returns_empty() -> None:
    store = EpisodeStore(emb_dim=4, device=_device(), capacity=2)
    assert store.topk(torch.zeros(4), k=4) == []


def test_dot_metric_distinct_from_cosine() -> None:
    store = EpisodeStore(emb_dim=4, device=_device(), capacity=4)
    # Two co-linear vectors of different magnitudes
    store.add(torch.tensor([1.0, 0, 0, 0]), (0, 1))
    store.add(torch.tensor([5.0, 0, 0, 0]), (1, 2))
    q = torch.tensor([1.0, 0, 0, 0])
    cos = store.topk(q, k=2, metric="cosine")
    dot = store.topk(q, k=2, metric="dot")
    # Cosine should rank both ~1 (cosine doesn't care about magnitude)
    assert abs(cos[0][1] - cos[1][1]) < 0.05, f"cosine should be ~equal, got {[s for _, s in cos]}"
    # Dot should rank the larger-magnitude one strictly higher
    assert dot[0][0].token_range == (1, 2), "dot should put the bigger vector first"


if __name__ == "__main__":
    tests = [
        ("add_and_repr", test_add_and_repr),
        ("topk_returns_self_for_query_equal_to_stored",
         test_topk_returns_self_for_query_equal_to_stored),
        ("topk_orders_by_similarity", test_topk_orders_by_similarity),
        ("capacity_exhaustion_raises", test_capacity_exhaustion_raises),
        ("empty_store_topk_returns_empty", test_empty_store_topk_returns_empty),
        ("dot_metric_distinct_from_cosine", test_dot_metric_distinct_from_cosine),
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
