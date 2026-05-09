"""Calibrated quantization parameters used at inference time.

These dataclasses are populated by ``calibration.py`` and then consumed
by both the pure-PyTorch reference path and the eventual Triton kernels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PerChannelScale:
    """Calibrated nuq parameters for Keys (per-channel along ``head_dim``).

    KVQuant quantises Keys *pre-RoPE*, sharing scaling factors across the
    sequence dimension and within each head_dim channel. For a layer
    with ``num_kv_heads`` heads of dim ``head_dim``:

    * ``poles`` are the ``2**num_bits`` reconstruction levels per channel,
      shape ``(num_kv_heads, head_dim, num_levels)``.
    * ``upper_threshold`` / ``lower_threshold`` mark the outlier band
      (anything outside is stored densely in fp16), shape
      ``(num_kv_heads, head_dim)``.
    """

    poles: torch.Tensor
    upper_threshold: torch.Tensor
    lower_threshold: torch.Tensor
    num_bits: int = 2

    def to(self, device: torch.device | str) -> "PerChannelScale":
        return PerChannelScale(
            poles=self.poles.to(device),
            upper_threshold=self.upper_threshold.to(device),
            lower_threshold=self.lower_threshold.to(device),
            num_bits=self.num_bits,
        )


@dataclass
class PerTokenScale:
    """Calibrated nuq parameters for Values (per-token).

    Values exhibit token-wise outlier patterns; KVQuant therefore shares
    a scale across head_dim within each token slot. For a calibration
    of length ``T`` tokens:

    * ``poles``: ``(num_kv_heads, T, num_levels)``
    * ``upper_threshold`` / ``lower_threshold``: ``(num_kv_heads, T)``

    At inference, scales for unseen positions are extrapolated from the
    last calibrated token (the "tail" hypothesis - Values stabilise
    after a few thousand tokens).
    """

    poles: torch.Tensor
    upper_threshold: torch.Tensor
    lower_threshold: torch.Tensor
    num_bits: int = 2

    def to(self, device: torch.device | str) -> "PerTokenScale":
        return PerTokenScale(
            poles=self.poles.to(device),
            upper_threshold=self.upper_threshold.to(device),
            lower_threshold=self.lower_threshold.to(device),
            num_bits=self.num_bits,
        )


@dataclass
class KVScales:
    """Full set of calibrated scales for one model.

    ``per_layer_keys[i]`` and ``per_layer_values[i]`` hold the scales
    for transformer layer ``i``. ``first_few_fp16`` mirrors the attention
    sink trick: the first N tokens are kept in fp16 even after
    quantisation kicks in.
    """

    per_layer_keys: list[PerChannelScale] = field(default_factory=list)
    per_layer_values: list[PerTokenScale] = field(default_factory=list)
    first_few_fp16: int = 4
    num_bits: int = 2

    def __post_init__(self) -> None:
        if self.per_layer_keys and self.per_layer_values:
            assert len(self.per_layer_keys) == len(self.per_layer_values), (
                "key and value scales must cover the same number of layers"
            )

    @property
    def num_layers(self) -> int:
        return len(self.per_layer_keys)

    def to(self, device: torch.device | str) -> "KVScales":
        return KVScales(
            per_layer_keys=[s.to(device) for s in self.per_layer_keys],
            per_layer_values=[s.to(device) for s in self.per_layer_values],
            first_few_fp16=self.first_few_fp16,
            num_bits=self.num_bits,
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "per_layer_keys": [
                    {
                        "poles": s.poles,
                        "upper_threshold": s.upper_threshold,
                        "lower_threshold": s.lower_threshold,
                        "num_bits": s.num_bits,
                    }
                    for s in self.per_layer_keys
                ],
                "per_layer_values": [
                    {
                        "poles": s.poles,
                        "upper_threshold": s.upper_threshold,
                        "lower_threshold": s.lower_threshold,
                        "num_bits": s.num_bits,
                    }
                    for s in self.per_layer_values
                ],
                "first_few_fp16": self.first_few_fp16,
                "num_bits": self.num_bits,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> "KVScales":
        d = torch.load(path, map_location=map_location, weights_only=True)
        return cls(
            per_layer_keys=[PerChannelScale(**s) for s in d["per_layer_keys"]],
            per_layer_values=[PerTokenScale(**s) for s in d["per_layer_values"]],
            first_few_fp16=d["first_few_fp16"],
            num_bits=d["num_bits"],
        )
