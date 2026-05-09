"""Calibration loop — collect per-channel/per-token KV statistics on Qwen2.

The output of this module is a populated :class:`scales.KVScales`,
saved as a single ``.pt`` file that the inference path will load.

High-level algorithm (KVQuant §3, adapted for Qwen2 GQA):

1. Load HuggingFace ``Qwen2ForCausalLM`` (fp16) plus a tokeniser.
2. Register forward hooks on every decoder layer's self-attention to
   capture pre-RoPE Keys and post-projection Values.
3. Run the model on a calibration dataset (default: 32 prompts at 4K
   tokens each, sampled from a long-context corpus).
4. For each layer, accumulate Keys along the sequence dimension and
   Values along the channel dimension, building per-group histograms.
5. Per scale group (per (head, head_dim) for K, per (head, position)
   for V):
   * Pick outlier band as the [outlier_pct, 1-outlier_pct] quantile.
   * Run weighted k-means with ``2**num_bits`` clusters on the
     non-outlier values; cluster centres become ``poles``.
6. Pack into :class:`KVScales` and save.

This module is *intentionally* not a vLLM-coupled path: calibration
runs once and offline against a stock HF checkpoint, and its product
(scales file) is loaded by the Triton/vLLM inference path.

NOTE (2026-05-09): scaffold + signatures only; implementation pending.
The math is fully specified in :mod:`nuq` and :mod:`scales`, so the
calibration loop is mainly plumbing (hooks, batching, k-means).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .scales import KVScales


@dataclass
class CalibrationConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M"
    num_prompts: int = 32
    seqlen: int = 4096
    num_bits: int = 2
    outlier_pct: float = 0.005  # 0.5 %% tail on each side
    first_few_fp16: int = 4
    cap_outliers: int = -1  # -1 = no cap; KVQuant paper uses 21
    dataset: str = "wikitext"  # or "c4", "redpajama"
    device: str = "cuda:0"
    seed: int = 0
    output_path: str = "scales/qwen2_5_7b_1m_nuq2.pt"


def run_calibration(cfg: CalibrationConfig) -> KVScales:
    """Run the calibration loop and return a populated :class:`KVScales`.

    NOTE: implementation pending. The expected steps are documented in
    the module docstring; once filled in, this should:

    * Load model + tokeniser
    * Build ``cfg.num_prompts`` prompts of length ``cfg.seqlen``
    * Capture KVs via forward hooks
    * Compute thresholds + run weighted k-means per scale group
    * Construct + return ``KVScales``

    Until then, callers should not invoke this function.
    """
    raise NotImplementedError(
        "calibration.run_calibration() is the next step in PR1b; "
        "the pure-PyTorch reference (nuq.py) and dataclasses (scales.py) "
        "are in place and tested. See module docstring for the planned "
        "algorithm."
    )


def kmeans_weighted(
    values: torch.Tensor,
    weights: torch.Tensor | None,
    num_clusters: int,
    num_iters: int = 25,
) -> torch.Tensor:
    """Weighted Lloyd's algorithm for 1-D inputs.

    Used to fit ``num_clusters`` (= ``2**num_bits``) reconstruction
    levels per scale group. ``weights`` should be Fisher-information
    estimates if available, else uniform.

    NOTE: implementation pending. Sklearn's ``KMeans`` is fine for the
    reference path; for production we may want a torch-native
    implementation that runs on GPU.
    """
    raise NotImplementedError(
        "calibration.kmeans_weighted() pending. For the first pass, "
        "sklearn.cluster.KMeans on the cpu-staged tensor is acceptable."
    )
