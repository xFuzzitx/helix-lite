"""Flat episode pool intended to live on GPU 1.

Storage strategy v1:
* ``embeddings``: ``(max_episodes, emb_dim)`` fp16 tensor on GPU 1.
  Mean-pooled last-layer hidden states per episode.
* ``token_ranges``: list of ``(start, end)`` byte-or-token spans into
  the original prompt. End-exclusive.
* Lookup: ``query @ embeddings.T`` then top-K. Plain matmul is fine
  for tens of thousands of episodes; FAISS is overkill until we
  cross 100K.

What lives where:
* The pool itself (embeddings + ranges) is on GPU 1.
* The query is computed on GPU 0 (where the model lives) and copied
  over for the matmul, then top-K indices come back. PCIe gen4 is
  ~32 GB/s so a single fp16 vector hop is sub-millisecond.

Future versions will also store the per-episode KV chunks on GPU 1
so :class:`hot_swap` can transfer them back to GPU 0 on demand.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Episode:
    """One unit of episodic memory."""

    index: int
    token_range: tuple[int, int]
    surprise: float = 0.0


class EpisodeStore:
    """Bounded flat pool of episode embeddings on a chosen device.

    Args:
        device: CUDA device that holds the pool. Defaults to the
            second GPU when present, else GPU 0.
        emb_dim: model hidden size (Qwen2.5-7B = 3584).
        capacity: max number of episodes the pool can hold; allocated
            up-front to avoid reallocation jitter.
        dtype: storage dtype for embeddings. fp16 saves bandwidth on
            the cross-device top-K hops; fp32 only if you need the
            extra precision for cosine-style retrieval.
    """

    def __init__(
        self,
        emb_dim: int,
        device: str | torch.device | None = None,
        capacity: int = 100_000,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if device is None:
            device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
        self.device = torch.device(device)
        self.dtype = dtype
        self.capacity = capacity
        self.emb_dim = emb_dim

        self.embeddings = torch.zeros(capacity, emb_dim, dtype=dtype, device=self.device)
        # Norms are cached for cosine-style scoring; updated on add().
        self.norms = torch.zeros(capacity, dtype=dtype, device=self.device)
        self.episodes: list[Episode] = []

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def add(self, embedding: torch.Tensor, token_range: tuple[int, int],
            surprise: float = 0.0) -> Episode:
        """Append one episode.

        Args:
            embedding: ``(emb_dim,)`` tensor (any device, any dtype);
                will be cast to ``self.dtype`` and moved to ``self.device``.
            token_range: ``(start, end)`` token positions, end-exclusive.
            surprise: optional log of the boundary's KL value, for
                later debugging or weighted retrieval.
        """
        if self.num_episodes >= self.capacity:
            raise RuntimeError(
                f"EpisodeStore is full ({self.capacity} episodes); "
                "this version does not support eviction yet."
            )
        if embedding.numel() != self.emb_dim:
            raise ValueError(
                f"embedding has {embedding.numel()} elements, "
                f"expected emb_dim={self.emb_dim}"
            )
        idx = self.num_episodes
        emb = embedding.to(device=self.device, dtype=self.dtype).reshape(self.emb_dim)
        self.embeddings[idx].copy_(emb)
        self.norms[idx] = emb.norm()
        ep = Episode(index=idx, token_range=token_range, surprise=surprise)
        self.episodes.append(ep)
        return ep

    def topk(self, query: torch.Tensor, k: int = 8,
             metric: str = "cosine") -> list[tuple[Episode, float]]:
        """Return the ``k`` highest-scoring episodes for ``query``.

        Args:
            query: ``(emb_dim,)`` tensor on any device.
            k: number of episodes to return; capped by ``num_episodes``.
            metric: ``"cosine"`` (default), or ``"dot"`` for raw dot
                product.

        Returns:
            ``[(Episode, score), ...]`` sorted descending by score.
        """
        if self.num_episodes == 0:
            return []
        k = min(k, self.num_episodes)
        q = query.to(device=self.device, dtype=self.dtype).reshape(self.emb_dim)
        active = self.embeddings[: self.num_episodes]                    # (N, D)
        scores = active @ q                                              # (N,)
        if metric == "cosine":
            q_norm = q.norm().clamp_min(1e-6)
            scores = scores / (self.norms[: self.num_episodes].clamp_min(1e-6) * q_norm)
        elif metric != "dot":
            raise ValueError(f"unknown metric: {metric}")
        top = torch.topk(scores, k=k, dim=0)
        out: list[tuple[Episode, float]] = []
        for rank in range(k):
            idx = int(top.indices[rank].item())
            out.append((self.episodes[idx], float(top.values[rank].item())))
        return out

    def memory_bytes(self) -> int:
        return int(self.embeddings.numel() * self.embeddings.element_size())

    def __repr__(self) -> str:
        mb = self.memory_bytes() / 1e6
        return (
            f"EpisodeStore(num={self.num_episodes}/{self.capacity}, "
            f"dim={self.emb_dim}, dev={self.device}, mem={mb:.1f} MB)"
        )
