"""Sanity test: the Quest selector picks blocks containing the highest q·k
matches, given calibrated stats from a CompactKVPool.

We construct a tiny pool, manually plant a "high-attention" block by
making its K dominate the dot-product with a known query, then check
that `select_top_blocks_for_layer` puts that block in the top-K.
"""
from __future__ import annotations

import sys
import torch

if not torch.cuda.is_available():
    print("CUDA required", file=sys.stderr); sys.exit(0)

from kvquant.compact_pool import CompactKVPool
from kvquant.scales import KVScales, PerChannelScale, PerTokenScale
from kvquant.quest_selector import select_top_blocks_for_layer


def _scales(L=1, H=2, D=8, T_cal=32, NL_K=16, NL_V=4):
    g = torch.Generator(device="cuda").manual_seed(0)
    keys = [PerChannelScale(
        poles=torch.linspace(-1.5, 1.5, NL_K, dtype=torch.float16, device="cuda")
              .reshape(1, 1, -1).expand(H, D, NL_K).contiguous(),
        upper_threshold=torch.full((H, D), 2.0, dtype=torch.float16, device="cuda"),
        lower_threshold=torch.full((H, D), -2.0, dtype=torch.float16, device="cuda"),
        num_bits=4,
    ) for _ in range(L)]
    vals = [PerTokenScale(
        poles=torch.linspace(-1.5, 1.5, NL_V, dtype=torch.float16, device="cuda")
              .reshape(1, 1, -1).expand(H, T_cal, NL_V).contiguous(),
        upper_threshold=torch.full((H, T_cal), 2.0, dtype=torch.float16, device="cuda"),
        lower_threshold=torch.full((H, T_cal), -2.0, dtype=torch.float16, device="cuda"),
        num_bits=2,
    ) for _ in range(L)]
    return KVScales(per_layer_keys=keys, per_layer_values=vals,
                    first_few_fp16=4, num_bits=2)


def test_quest_selector_picks_high_attention_block():
    H, D = 2, 8
    BS = 4
    NB = 6
    T_cal = 32
    scales = _scales(H=H, D=D, T_cal=T_cal)
    pool = CompactKVPool(num_layers=1, num_blocks=NB, block_size=BS,
                          num_kv_heads=H, head_size=D, scales=scales)

    # Plant tokens in blocks 0..4 (slots 0..19). Block 3 has K aligned
    # with our query direction (all 1.0); the rest have K = 0.
    T = 5 * BS
    key = torch.zeros(T, H, D, dtype=torch.float16, device="cuda")
    high_block = 3
    key[high_block * BS:(high_block + 1) * BS] = 1.0  # all heads, all dims = 1
    value = torch.zeros_like(key)
    slot_mapping = torch.arange(T, dtype=torch.int64, device="cuda")
    abs_pos = torch.arange(T, dtype=torch.int64, device="cuda")
    pool.write_kv(0, key, value, slot_mapping, abs_pos)

    # Query points in the +1 direction → high block should win
    query = torch.ones(H, D, dtype=torch.float16, device="cuda")
    used_block_ids = torch.arange(5, dtype=torch.int64, device="cuda")
    selected, first_pos = select_top_blocks_for_layer(
        pool, layer_idx=0, query=query, used_block_ids=used_block_ids,
        block_size=BS, top_k=1, sink_blocks=0,
    )
    assert int(selected.shape[0]) == 1, f"expected 1 block, got {int(selected.shape[0])}"
    assert int(selected[0].item()) == high_block, \
        f"expected block {high_block}, got {int(selected[0].item())}"
    # first_pos = block_id * BS
    assert int(first_pos[0].item()) == high_block * BS


def test_quest_selector_includes_sinks():
    H, D = 2, 8
    BS = 4
    NB = 6
    scales = _scales(H=H, D=D)
    pool = CompactKVPool(num_layers=1, num_blocks=NB, block_size=BS,
                          num_kv_heads=H, head_size=D, scales=scales)
    # All zero K — top-K is ambiguous. Sinks should still be included.
    T = 5 * BS
    k = torch.zeros(T, H, D, dtype=torch.float16, device="cuda")
    pool.write_kv(0, k, k, torch.arange(T, dtype=torch.int64, device="cuda"),
                  torch.arange(T, dtype=torch.int64, device="cuda"))
    query = torch.ones(H, D, dtype=torch.float16, device="cuda")
    used = torch.arange(5, dtype=torch.int64, device="cuda")
    selected, _ = select_top_blocks_for_layer(
        pool, layer_idx=0, query=query, used_block_ids=used,
        block_size=BS, top_k=1, sink_blocks=2,
    )
    # Sinks are blocks 0 and 1 of used_block_ids (= block ids 0 and 1)
    assert 0 in selected.tolist()
    assert 1 in selected.tolist()
