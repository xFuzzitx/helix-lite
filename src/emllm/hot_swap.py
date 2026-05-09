"""Hot/cold KV attention swap for the EM-LLM retrieval path.

The idea (EM-LLM, Liu et al., ICLR'25):

* Keep a small **hot window** of the most recent KV on GPU 0 (the
  model's normal KV cache).
* Push older KV chunks, segmented by Bayesian surprise, into the
  **cold store** on GPU 1, indexed by an embedding.
* At each decode step, query the cold store with the current
  hidden state, retrieve the top-M episodes, transfer their KV
  back to GPU 0 transiently, and run attention over the
  ``[hot ; retrieved-cold]`` concatenation.

This module provides the orchestrator. The model's per-layer K, V
projections are computed normally; this code re-builds the KV that
attention sees by mixing the hot cache with cold-retrieved chunks.

For the smoke test we run this with HF transformers (full attention,
no paging), since vLLM-side integration is the same plumbing job
as the nuq4 / Quest backends and is being tracked separately in
``src/kvquant/vllm_integration.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .kv_store import KVEpisodeStore


@dataclass
class HotSwapConfig:
    """Tuning knobs for the hot/cold attention swap."""

    hot_window: int = 4096          # tokens kept fully on GPU 0
    top_m: int = 8                  # cold episodes retrieved per query
    sink_tokens: int = 4            # always-include first N tokens (StreamingLLM)
    metric: str = "cosine"


@dataclass
class SwapResult:
    """Output of one swap operation."""

    K_combined: torch.Tensor       # (num_layers, T_combined, H_kv, D)
    V_combined: torch.Tensor
    sink_len: int
    cold_len: int
    hot_len: int

    @property
    def total_len(self) -> int:
        return self.sink_len + self.cold_len + self.hot_len


def assemble_kv(
    hot_K: torch.Tensor,                    # (num_layers, T_hot, H_kv, D)
    hot_V: torch.Tensor,
    sink_K: torch.Tensor | None,            # (num_layers, sink_tokens, H_kv, D)
    sink_V: torch.Tensor | None,
    store: KVEpisodeStore,
    query_embedding: torch.Tensor,          # (emb_dim,)
    cfg: HotSwapConfig,
    target_device: torch.device | str,
) -> SwapResult:
    """Build the K, V seen by attention from hot + retrieved-cold chunks.

    Args:
        hot_K, hot_V: the most recent KV on GPU 0. Last
            ``cfg.hot_window`` tokens of the conversation.
        sink_K, sink_V: the first ``cfg.sink_tokens`` tokens (kept
            verbatim from the very start of the prompt). May be
            ``None`` when sinks are disabled.
        store: a :class:`KVEpisodeStore` populated with cold episodes.
        query_embedding: vector to score cold episodes against.
        cfg: knobs.
        target_device: where the output tensors should live (usually
            cuda:0).

    Returns:
        A :class:`SwapResult` holding the concatenated K, V plus
        the sub-lengths so callers know how many tokens are sinks
        / cold / hot in causal order.
    """
    # 1. Score cold episodes by the query and pick top-M.
    top = store.topk(query_embedding, k=cfg.top_m, metric=cfg.metric)
    if not top:
        cold_K = cold_V = None
        cold_len = 0
    else:
        cold_idx = [ep.index for ep, _score in top if store.has_kv(ep.index)]
        if cold_idx:
            cold_K, cold_V, _ = store.gather_kv_for_episodes(cold_idx, target_device)
            cold_len = cold_K.shape[1]
        else:
            cold_K = cold_V = None
            cold_len = 0

    # 2. Move hot to target if needed
    hot_K = hot_K.to(target_device)
    hot_V = hot_V.to(target_device)
    hot_len = hot_K.shape[1]

    # 3. Optional sinks at the very start
    sink_len = 0
    parts_K = []
    parts_V = []
    if sink_K is not None:
        sink_K = sink_K.to(target_device)
        sink_V = sink_V.to(target_device)
        sink_len = sink_K.shape[1]
        parts_K.append(sink_K)
        parts_V.append(sink_V)
    if cold_K is not None:
        parts_K.append(cold_K)
        parts_V.append(cold_V)
    parts_K.append(hot_K)
    parts_V.append(hot_V)

    K_combined = torch.cat(parts_K, dim=1)
    V_combined = torch.cat(parts_V, dim=1)
    return SwapResult(
        K_combined=K_combined, V_combined=V_combined,
        sink_len=sink_len, cold_len=cold_len, hot_len=hot_len,
    )
