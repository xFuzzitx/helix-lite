"""HELIX-Lite — EM-LLM episodic memory on GPU 1 (PR5).

Components:

- :mod:`segmenter`  - Bayesian-surprise segmentation. As tokens stream
  through the model, watch the predicted-distribution KL between
  consecutive positions; spikes above a threshold mark episode
  boundaries. Cheap (just KL on logits) and content-aware.

- :mod:`episode_store` - flat pool on GPU 1 holding ``(embedding,
  token_range)`` per episode. Lookup is plain matmul-then-topK for
  now; FAISS-GPU is a follow-up once we have enough episodes that
  exact search starts to hurt.

- (Future) ``hot_swap.py`` - given a query at decode time, fetch the
  top-M episodes' KV chunks across PCIe back to GPU 0 for transient
  attention.
"""
from .segmenter import BayesianSurpriseSegmenter
from .episode_store import EpisodeStore, Episode
from .kv_store import KVEpisodeStore, KVChunk
from .hot_swap import HotSwapConfig, SwapResult, assemble_kv

__all__ = [
    "BayesianSurpriseSegmenter",
    "EpisodeStore",
    "Episode",
    "KVEpisodeStore",
    "KVChunk",
    "HotSwapConfig",
    "SwapResult",
    "assemble_kv",
]
