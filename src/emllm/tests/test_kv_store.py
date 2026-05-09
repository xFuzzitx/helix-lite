"""Tests for KVEpisodeStore + assemble_kv.

Run:
    PYTHONPATH=src python src/emllm/tests/test_kv_store.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emllm.episode_store import Episode
from emllm.kv_store import KVEpisodeStore
from emllm.hot_swap import HotSwapConfig, assemble_kv


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def test_add_kv_and_get_kv() -> None:
    store = KVEpisodeStore(emb_dim=8, device=_device(), capacity=4)
    e = store.add(torch.randn(8), (0, 16))
    K = torch.randn(2, 16, 4, 8)        # (num_layers, T, H_kv, D)
    V = torch.randn(2, 16, 4, 8)
    store.add_kv(e, K, V)
    assert store.has_kv(e.index)
    chunk = store.get_kv(e.index)
    assert chunk.K.shape == (2, 16, 4, 8)
    assert chunk.V.shape == (2, 16, 4, 8)
    assert chunk.length == 16
    assert chunk.num_layers == 2


def test_gather_kv_concats_along_time() -> None:
    store = KVEpisodeStore(emb_dim=8, device=_device(), capacity=4)
    e0 = store.add(torch.randn(8), (0, 8))
    e1 = store.add(torch.randn(8), (8, 24))
    e2 = store.add(torch.randn(8), (24, 28))
    store.add_kv(e0, torch.randn(2, 8, 4, 8), torch.randn(2, 8, 4, 8))
    store.add_kv(e1, torch.randn(2, 16, 4, 8), torch.randn(2, 16, 4, 8))
    store.add_kv(e2, torch.randn(2, 4, 4, 8), torch.randn(2, 4, 4, 8))

    K, V, lens = store.gather_kv_for_episodes([e0.index, e2.index], _device())
    # 8 + 4 = 12 along time; layers/heads/dim preserved
    assert K.shape == (2, 12, 4, 8)
    assert V.shape == (2, 12, 4, 8)
    assert lens == [8, 4]


def test_assemble_kv_orders_sinks_cold_hot() -> None:
    """assemble_kv must concatenate in [sinks ; cold ; hot] order."""
    store = KVEpisodeStore(emb_dim=4, device=_device(), capacity=8)
    # Two cold episodes
    e0 = store.add(torch.tensor([1.0, 0, 0, 0]), (0, 5))
    e1 = store.add(torch.tensor([0, 1.0, 0, 0]), (5, 8))
    store.add_kv(e0, torch.randn(2, 5, 1, 4), torch.randn(2, 5, 1, 4))
    store.add_kv(e1, torch.randn(2, 3, 1, 4), torch.randn(2, 3, 1, 4))

    # Sinks: 4 tokens (fp16 to match store dtype)
    sink_K = torch.randn(2, 4, 1, 4, dtype=torch.float16)
    sink_V = torch.randn(2, 4, 1, 4, dtype=torch.float16)
    # Hot: 16 tokens
    hot_K = torch.randn(2, 16, 1, 4, dtype=torch.float16)
    hot_V = torch.randn(2, 16, 1, 4, dtype=torch.float16)

    cfg = HotSwapConfig(hot_window=16, top_m=2, sink_tokens=4)
    result = assemble_kv(
        hot_K=hot_K, hot_V=hot_V,
        sink_K=sink_K, sink_V=sink_V,
        store=store,
        query_embedding=torch.tensor([1.0, 0, 0, 0]),
        cfg=cfg,
        target_device=_device(),
    )
    # Total 4 + (5 + 3) + 16 = 28
    assert result.K_combined.shape == (2, 28, 1, 4)
    assert result.sink_len == 4
    assert result.cold_len == 8
    assert result.hot_len == 16
    assert result.total_len == 28
    # The cold KV should be the gather_kv output for the queried episodes,
    # in retrieval order.
    expected_cold_K, _, _ = store.gather_kv_for_episodes([e0.index, e1.index], _device())
    assert torch.allclose(result.K_combined[:, 4:12], expected_cold_K)
    # Sinks first
    assert torch.allclose(result.K_combined[:, :4], sink_K.to(_device()))
    # Hot last
    assert torch.allclose(result.K_combined[:, 12:], hot_K.to(_device()))


def test_assemble_kv_with_no_cold_kv_falls_through() -> None:
    """If no cold episode has a KVChunk attached, only sinks+hot survive."""
    store = KVEpisodeStore(emb_dim=4, device=_device(), capacity=4)
    store.add(torch.tensor([1.0, 0, 0, 0]), (0, 5))   # no add_kv
    cfg = HotSwapConfig(top_m=4, sink_tokens=2, hot_window=8)
    sink_K = torch.randn(2, 2, 1, 4); sink_V = torch.randn(2, 2, 1, 4)
    hot_K = torch.randn(2, 8, 1, 4); hot_V = torch.randn(2, 8, 1, 4)
    res = assemble_kv(hot_K, hot_V, sink_K, sink_V, store,
                      torch.tensor([1.0, 0, 0, 0]), cfg, _device())
    assert res.cold_len == 0
    assert res.K_combined.shape == (2, 10, 1, 4)


def test_kv_store_repr_shows_with_kv_count() -> None:
    store = KVEpisodeStore(emb_dim=4, device=_device(), capacity=4)
    e0 = store.add(torch.zeros(4), (0, 1))
    store.add(torch.zeros(4), (1, 2))      # no KV
    store.add_kv(e0, torch.zeros(2, 1, 1, 4), torch.zeros(2, 1, 1, 4))
    s = repr(store)
    assert "with_kv=1" in s, s
    assert "2/4" in s, s


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); sys.exit(0)
    tests = [
        ("add_kv_and_get_kv", test_add_kv_and_get_kv),
        ("gather_kv_concats_along_time", test_gather_kv_concats_along_time),
        ("assemble_kv_orders_sinks_cold_hot", test_assemble_kv_orders_sinks_cold_hot),
        ("assemble_kv_with_no_cold_kv_falls_through",
         test_assemble_kv_with_no_cold_kv_falls_through),
        ("kv_store_repr_shows_with_kv_count", test_kv_store_repr_shows_with_kv_count),
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
