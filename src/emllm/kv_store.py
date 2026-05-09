"""Per-episode KV chunk storage on GPU 1.

PR5a's :class:`EpisodeStore` only kept embeddings (~3.5 MB for 1000
episodes); for the actual hot/cold attention swap we also need the
KV tensors per episode, so the model can attend back to a past
episode without recomputing it.

Storage shape per episode:
    K: (num_layers, T_episode, num_kv_heads, head_dim) fp16
    V: same shape

For Qwen2.5-7B (28 layers, 4 KV heads, 128 head_dim) and an average
episode of 256 tokens, that's 28*256*4*128*2 bytes = ~7.3 MB for K
plus ~7.3 MB for V, ~15 MB per episode. 1500 episodes = ~22 GB,
which fits a 24 GB GPU 1 with margin for the embeddings.

This module is the storage layer; the actual attention swap lives
in :mod:`hot_swap`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .episode_store import Episode, EpisodeStore


@dataclass
class KVChunk:
    """Per-layer K, V tensors for one episode."""

    K: torch.Tensor  # (num_layers, T_episode, num_kv_heads, head_dim)
    V: torch.Tensor  # same shape

    @property
    def num_layers(self) -> int:
        return self.K.shape[0]

    @property
    def length(self) -> int:
        return self.K.shape[1]

    def memory_bytes(self) -> int:
        return int(
            self.K.numel() * self.K.element_size()
            + self.V.numel() * self.V.element_size()
        )

    def to(self, device: torch.device | str) -> "KVChunk":
        return KVChunk(K=self.K.to(device), V=self.V.to(device))


class KVEpisodeStore(EpisodeStore):
    """EpisodeStore that also stores KV chunks per episode.

    Identical interface plus :meth:`add_kv` / :meth:`get_kv` /
    :meth:`gather_kv_for_episodes`. KV tensors live on the same
    device as the embeddings (defaults to ``cuda:1``); callers move
    them back to ``cuda:0`` for attention computations.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # episode_idx -> KVChunk; sparse so we can support stores
        # where embeddings are added without KV (e.g. during retrieval
        # tests on a budget GPU).
        self._kv: dict[int, KVChunk] = {}

    def add_kv(self, episode: Episode, K: torch.Tensor, V: torch.Tensor) -> None:
        """Attach K, V to an existing episode (added via :meth:`add`).

        Args:
            episode: the Episode object returned by :meth:`add`.
            K, V: ``(num_layers, T_episode, num_kv_heads, head_dim)``
                tensors. Will be moved to ``self.device`` and cast to
                ``self.dtype`` (default fp16).
        """
        if K.shape != V.shape:
            raise ValueError(f"K and V must share shape, got {tuple(K.shape)} vs {tuple(V.shape)}")
        if K.dim() != 4:
            raise ValueError(f"K must be (num_layers, T, H, D), got {tuple(K.shape)}")
        chunk = KVChunk(
            K=K.to(device=self.device, dtype=self.dtype),
            V=V.to(device=self.device, dtype=self.dtype),
        )
        self._kv[episode.index] = chunk

    def get_kv(self, episode_idx: int) -> KVChunk:
        if episode_idx not in self._kv:
            raise KeyError(f"no KV stored for episode {episode_idx}")
        return self._kv[episode_idx]

    def has_kv(self, episode_idx: int) -> bool:
        return episode_idx in self._kv

    def gather_kv_for_episodes(
        self,
        episode_indices: list[int],
        target_device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Concatenate KV chunks for the given episodes along the time axis.

        Returns:
            ``(K_cat, V_cat, lengths)`` where ``K_cat`` and ``V_cat``
            have shape ``(num_layers, sum_T, num_kv_heads, head_dim)``
            on ``target_device``, and ``lengths`` is the per-episode
            token count in concat order.
        """
        if not episode_indices:
            raise ValueError("episode_indices is empty")
        chunks = []
        lengths = []
        for idx in episode_indices:
            c = self.get_kv(idx)
            chunks.append(c)
            lengths.append(c.length)
        K_cat = torch.cat([c.K for c in chunks], dim=1).to(target_device)
        V_cat = torch.cat([c.V for c in chunks], dim=1).to(target_device)
        return K_cat, V_cat, lengths

    def kv_memory_bytes(self) -> int:
        return sum(c.memory_bytes() for c in self._kv.values())

    def __repr__(self) -> str:
        emb_mb = self.memory_bytes() / 1e6
        kv_mb = self.kv_memory_bytes() / 1e6
        return (
            f"KVEpisodeStore(num={self.num_episodes}/{self.capacity}, "
            f"with_kv={len(self._kv)}, dim={self.emb_dim}, dev={self.device}, "
            f"emb={emb_mb:.1f} MB, kv={kv_mb:.0f} MB)"
        )
