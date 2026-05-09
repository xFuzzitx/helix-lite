"""Pure-PyTorch reference implementation of nuq2 quantisation.

This module is the spec for what the Triton kernels must reproduce
bit-for-bit. The math comes from KVQuant (Hooper et al., 2024) but
re-expressed for vectorised tensor ops over Qwen2-shaped KVs.

Core ideas, condensed:

* Each value is mapped to its nearest *pole* in a calibrated lookup
  table (non-uniform quantisation), with ``2**num_bits`` poles.
* Values outside an outlier band ``[lower_th, upper_th]`` are kept in
  full precision (dense outliers); the rest are quantised.
* For Keys, scales are shared per-channel along ``head_dim``; for
  Values, scales are shared per-token. The shape conventions mirror
  ``KVScales`` in :mod:`scales`.
"""
from __future__ import annotations

import torch


def round_to_nearest_pole(values: torch.Tensor, poles: torch.Tensor) -> torch.Tensor:
    """Map each value to the nearest pole.

    Args:
        values: arbitrary float tensor, shape ``(*S,)``.
        poles: ``(*S_broadcast, num_levels)``. The last dim enumerates
            the candidate reconstruction levels for each scale group.

    Returns:
        Reconstructed tensor with the same shape as ``values``,
        containing one of the ``poles`` values per element.
    """
    # values: (*S,)         -> unsqueeze last for level-broadcast
    # poles:  (*S, num_levels)
    diff = (values.unsqueeze(-1) - poles).abs()
    idx = diff.argmin(dim=-1)
    return torch.gather(poles, -1, idx.unsqueeze(-1)).squeeze(-1)


def find_outliers(
    values: torch.Tensor,
    upper_threshold: torch.Tensor,
    lower_threshold: torch.Tensor,
    cap_outliers: int = -1,
    first_few_fp16: int = -1,
    seq_dim: int = -2,
) -> torch.Tensor:
    """Boolean mask marking outlier positions.

    A value is an outlier when it falls outside ``[lower_threshold,
    upper_threshold]``. The thresholds broadcast against ``values``
    along whatever dimensions ``values`` has beyond the threshold rank.

    Args:
        values: input tensor.
        upper_threshold / lower_threshold: per-group thresholds, shape
            broadcastable with ``values`` once the seq/head_dim slot is
            unsqueezed away.
        cap_outliers: if positive, keep only the top-``cap_outliers``
            largest-deviation outliers per row; the rest are clamped
            and quantised. ``-1`` disables capping.
        first_few_fp16: if positive, force the first N positions along
            ``seq_dim`` to be kept in fp16 (attention-sink trick).
        seq_dim: which dim of ``values`` is the sequence dim; only
            used by ``first_few_fp16``.

    Returns:
        Boolean tensor with the same shape as ``values``: ``True`` for
        outlier positions.
    """
    mask = (values < lower_threshold) | (values > upper_threshold)

    if cap_outliers > 0:
        # Distance from band centre, normalised by half-band width
        zero_point = (upper_threshold + lower_threshold) / 2
        half_band = (upper_threshold - lower_threshold) / 2
        # Avoid division by zero; positions with zero band fall back to abs
        half_band = half_band.clamp(min=1e-8)
        normalised = (values - zero_point) / half_band
        # Keep only top-k by absolute normalised deviation along the
        # last dim (matches KVQuant's per-token outlier cap)
        magnitude = normalised.abs() * mask.to(normalised.dtype)
        topk = torch.topk(magnitude, k=cap_outliers, dim=-1)
        capped = torch.zeros_like(mask)
        capped.scatter_(-1, topk.indices, True)
        mask = capped

    if first_few_fp16 > 0:
        idx = [slice(None)] * values.dim()
        idx[seq_dim] = slice(0, first_few_fp16)
        mask[tuple(idx)] = True

    return mask


def quantize_nuq(
    values: torch.Tensor,
    poles: torch.Tensor,
    upper_threshold: torch.Tensor,
    lower_threshold: torch.Tensor,
    cap_outliers: int = -1,
    first_few_fp16: int = -1,
    seq_dim: int = -2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference (slow) nuq2 quantisation.

    Args:
        values, poles, upper_threshold, lower_threshold: see
            :func:`round_to_nearest_pole` and :func:`find_outliers`.
        cap_outliers / first_few_fp16 / seq_dim: forwarded to
            :func:`find_outliers`.

    Returns:
        ``(reconstructed, outlier_mask)``. ``reconstructed`` has the
        same shape and dtype as ``values``; outlier positions hold the
        original fp16 value, the rest hold the nearest pole. The mask
        records which positions are outliers, so the caller can store
        them densely (and the quantised positions sparsely as 2-bit
        codes, downstream).
    """
    outlier_mask = find_outliers(
        values,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        cap_outliers=cap_outliers,
        first_few_fp16=first_few_fp16,
        seq_dim=seq_dim,
    )
    quantised = round_to_nearest_pole(values, poles)
    out = torch.where(outlier_mask, values, quantised)
    return out, outlier_mask


def dequantize_nuq(
    codes: torch.Tensor,
    outliers: torch.Tensor,
    outlier_mask: torch.Tensor,
    poles: torch.Tensor,
) -> torch.Tensor:
    """Round-trip companion for the reference quantiser.

    Args:
        codes: integer tensor of pole indices, same shape as the
            original values.
        outliers: dense fp16 tensor at outlier positions (other
            positions are zero).
        outlier_mask: boolean tensor marking outlier positions.
        poles: ``(*S, num_levels)`` reconstruction table.

    Returns:
        Tensor with the same shape as ``codes`` in fp16/bf16, holding
        the dequantised KV.
    """
    reconstructed = torch.gather(poles, -1, codes.unsqueeze(-1)).squeeze(-1)
    return torch.where(outlier_mask, outliers, reconstructed.to(outliers.dtype))
