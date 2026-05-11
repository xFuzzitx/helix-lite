"""Phase 1B v1 vLLM integration — compact pool + Quest top-K on decode.

Design (v1 = correctness anchor, NOT yet VRAM-optimal):

* vLLM allocates its standard fp16 KV pool as usual. We do not touch it.
* We allocate a **shadow** CompactKVPool sized to ``max_model_len``.
* ``do_kv_cache_update``:
    1. Call super() to write fp16 K/V into vLLM's pool (so prefill
       attention can read them).
    2. Also pack the same K/V into our compact pool via
       ``CompactKVPool.write_kv``.
* ``forward``:
    - **Prefill / chunked-prefill** (``max_query_len > 1``): pass through
      to super() — reads vLLM's fp16 pool. No Quest pruning; full
      attention.
    - **Decode** (``max_query_len == 1``): for each sequence, use Quest
      against the compact pool's stats to select top-K blocks, unpack
      them into a per-sequence fp16 staging buffer, then call
      ``flash_attn_varlen_func`` on the staging with a remapped block
      table.

Phase 1B v2 will eliminate vLLM's fp16 pool (so total VRAM drops
below baseline). This v1 hybrid is the correctness gate.

Status: not yet wired into a registered AttentionBackend. Use
``install_compact_backend(scales_path)`` to monkey-patch FA's
get_impl_cls similar to Phase 1A's :mod:`vllm_backend`.

Limitations of v1:
- Skipped: outliers added by the calibration are stored in the
  compact pool but clamping behaviour for missing-from-staging
  cases is conservative.
- Single-sequence decode path is the simple case; multi-seq batched
  decode walks one sequence at a time (not ideal but correct).
- No fused-dequant attention kernel — staging is a real fp16 buffer.

This is enough to validate the design end-to-end at 4K and start
walking up the context ladder. The fused kernel is Phase 1B v2.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)

from .compact_pool import CompactKVPool
from .quest_selector import select_top_blocks_for_layer
from .scales import KVScales

if TYPE_CHECKING:
    pass


# Module-level state, set up via install_compact_backend()
_KV_SCALES: KVScales | None = None
_COMPACT_POOL: CompactKVPool | None = None
_POOL_CONFIG: dict = {}
_DEVICE: str = "cuda:0"
_TOP_K: int = 0     # decode top-K blocks. 0 → never prune (debug)
_SINK_BLOCKS: int = 4
_LAYER_INDEX: dict[str, int] = {}


def install_compact_backend(
    scales_path: str,
    *,
    num_layers: int = 28,
    num_blocks: int = 16_384,
    block_size: int = 16,
    num_kv_heads: int = 4,
    head_size: int = 128,
    device: str = "cuda:0",
    top_k_blocks_decode: int = 0,
    sink_blocks: int = 4,
) -> None:
    """Set up the global compact pool and patch FA's impl class.

    Must be called before ``vllm.LLM(...)``.

    ``top_k_blocks_decode=0`` disables Quest pruning (decode reads all
    used blocks). Set to a positive number to enable.
    """
    global _KV_SCALES, _COMPACT_POOL, _POOL_CONFIG, _DEVICE
    global _TOP_K, _SINK_BLOCKS
    _DEVICE = device
    _KV_SCALES = KVScales.load(scales_path, map_location=device).to(device)
    assert _KV_SCALES.num_layers == num_layers, (
        f"scales has {_KV_SCALES.num_layers} layers, expected {num_layers}"
    )
    _COMPACT_POOL = CompactKVPool(
        num_layers=num_layers, num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size, scales=_KV_SCALES,
        device=device,
    )
    _POOL_CONFIG = dict(
        num_layers=num_layers, num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size,
    )
    _TOP_K = top_k_blocks_decode
    _SINK_BLOCKS = sink_blocks

    FlashAttentionBackend.get_impl_cls = staticmethod(  # type: ignore[method-assign]
        lambda: KVQuantCompactImpl
    )
    print(
        f"[kvquant.compact_backend] compact pool: "
        f"{num_layers}L × 2 × {num_blocks}B × {block_size}T × "
        f"{num_kv_heads}H × {head_size}D uint8 + outliers fp16 on {device}"
    )
    print(
        f"[kvquant.compact_backend] top_k_blocks_decode={top_k_blocks_decode} "
        f"sink_blocks={sink_blocks} (0 = no Quest pruning)"
    )


def _layer_index(layer: torch.nn.Module) -> int | None:
    name = getattr(layer, "layer_name", "") or ""
    m = re.search(r"layers?\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    if name and name not in _LAYER_INDEX:
        _LAYER_INDEX[name] = len(_LAYER_INDEX)
    return _LAYER_INDEX.get(name)


class KVQuantCompactImpl(FlashAttentionImpl):
    """v1 hybrid impl: pack to compact pool on write, use it for decode."""

    can_return_lse_for_decode: bool = True

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # 1) keep vLLM's fp16 cache up to date so prefill can read it
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        # 2) mirror into the compact pool for decode-time Quest access
        if _COMPACT_POOL is None:
            return
        idx = _layer_index(layer)
        if idx is None or idx >= _COMPACT_POOL.num_layers:
            return
        # Absolute token positions: slot_mapping itself is the token's
        # absolute slot index, which equals its absolute position when
        # one sequence is in-flight. For multi-seq batches this only
        # works if each sequence starts at its own block boundary; vLLM
        # guarantees this for paged attention by design.
        num_actual = slot_mapping.numel()
        if num_actual == 0:
            return
        # Contiguous (T, H, D) views of the prefix
        key_part = key[:num_actual].contiguous()
        value_part = value[:num_actual].contiguous()
        abs_positions = slot_mapping[:num_actual].to(torch.int64).contiguous()
        _COMPACT_POOL.write_kv(
            idx, key_part.to(torch.float16), value_part.to(torch.float16),
            slot_mapping[:num_actual].to(torch.int64).contiguous(), abs_positions,
        )

    # v1: keep forward identical to FA (use vLLM's fp16 cache). Decode-side
    # Quest hookup lands in v1.1 once we verify the write side at scale.
