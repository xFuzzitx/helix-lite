"""Quest top-K block selection driven by the compact KV pool's stats.

The compact pool stores per-block ``(k_min, k_max)`` in fp32 — populated
atomically during write_kv. PR3 Quest math (``src/quest/reference.py``)
takes those stats and a query, computes the conservative upper bound
``q_pos·k_max + q_neg·k_min`` per page-per-head, and returns the
top-K page ids.

This module bridges the two: given a compact pool, a query, the
list of *used* blocks for the active sequence, and the desired
keep-K + sinks, it returns a flat list of block ids (union across
heads) and the matching staging-buffer block-first-positions for
the unpack-to-staging kernel.

For Phase 1B v1 we union the per-head top-K so the staging buffer
holds the same set of blocks for every head. This is permissive vs
strict per-head selection (slightly more memory, slightly more FA
work), but it keeps the call into vLLM's existing
``flash_attn_varlen_func`` straightforward.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make src/quest reachable
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from quest.reference import page_upper_bound, topk_pages_with_sinks  # noqa: E402

from .compact_pool import CompactKVPool


def select_top_blocks_for_layer(
    pool: CompactKVPool,
    layer_idx: int,
    query: torch.Tensor,           # (H, D) query for one decode step (head-first)
    used_block_ids: torch.Tensor,  # (NU,) int64 — blocks holding this seq's tokens
    block_size: int,
    top_k: int,
    sink_blocks: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the union-of-heads top-K blocks the decode query attends to.

    Args:
        pool: CompactKVPool. Its k_min/k_max for layer_idx are read.
        layer_idx: which transformer layer.
        query: ``(H, D)`` fp16 decode query (already divided by head_dim if
            the caller wants — we apply no scaling here).
        used_block_ids: int64 block ids that currently hold tokens for
            this sequence, in ascending order. Length NU.
        block_size: tokens per block.
        top_k: number of non-sink blocks to keep.
        sink_blocks: leading blocks always kept (StreamingLLM-style attention sinks).

    Returns:
        ``(selected_block_ids, block_first_positions)``, both int64
        on the same device. The unpack kernel uses them directly.
    """
    # Gather only the stats for blocks this sequence has touched
    k_min_active = pool.k_min[layer_idx, used_block_ids]  # (NU, H, D) fp32
    k_max_active = pool.k_max[layer_idx, used_block_ids]
    # Convert query to fp32 to match
    q_f = query.float()
    # (NU, H)
    bound = page_upper_bound(q_f, k_min_active, k_max_active)

    # Use sink_blocks // 1 leading slots of used_block_ids as sinks
    NU = used_block_ids.shape[0]
    sinks = max(0, min(sink_blocks, NU))
    keep = max(0, min(top_k, NU - sinks))

    # topk_pages_with_sinks returns per-head ids into [0, NU)
    selected_per_head = topk_pages_with_sinks(bound, k=keep, sink_pages=sinks)
    # (sinks + keep, H) — union across H
    unique_idx_in_used = torch.unique(selected_per_head.reshape(-1))
    # Sort by original block id so we preserve causal order
    selected_block_ids = used_block_ids[unique_idx_in_used.long()]
    selected_block_ids, _ = selected_block_ids.sort()

    # block_first_positions: for each selected block id, the absolute
    # starting token position is block_id * block_size (assuming the
    # sequence started at slot 0 and used blocks in id-order).
    # For a single sequence in v1 that's correct.
    block_first_positions = selected_block_ids * block_size
    return selected_block_ids, block_first_positions
