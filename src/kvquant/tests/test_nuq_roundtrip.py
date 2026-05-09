"""Round-trip and correctness checks for the reference nuq2 implementation.

Run with:
    python -m pytest src/kvquant/tests/test_nuq_roundtrip.py -v
or just:
    python src/kvquant/tests/test_nuq_roundtrip.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvquant.nuq import (
    dequantize_nuq,
    find_outliers,
    quantize_nuq,
    round_to_nearest_pole,
)


def test_round_to_nearest_pole_simple() -> None:
    values = torch.tensor([0.1, 0.4, -0.6, 1.2])
    # Single shared pole table for all positions
    poles = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).expand(4, 5)
    out = round_to_nearest_pole(values, poles)
    expected = torch.tensor([0.0, 0.5, -0.5, 1.0])
    assert torch.allclose(out, expected), f"got {out}"


def test_outliers_threshold_band() -> None:
    values = torch.tensor([[-2.0, -0.4, 0.1, 0.7, 3.5]])  # (1, 5)
    upper = torch.tensor([[1.0]])
    lower = torch.tensor([[-1.0]])
    mask = find_outliers(values, upper, lower)
    assert mask.tolist() == [[True, False, False, False, True]]


def test_outliers_first_few_fp16() -> None:
    values = torch.zeros(5, 4)
    upper = torch.full((5, 4), 1.0)
    lower = torch.full((5, 4), -1.0)
    mask = find_outliers(values, upper, lower, first_few_fp16=2, seq_dim=0)
    assert mask[:2].all()
    assert not mask[2:].any()


def test_quantize_dequantize_roundtrip_no_outliers() -> None:
    torch.manual_seed(0)
    values = torch.linspace(-1, 1, 256)
    # 4 evenly spaced poles -> nuq2
    poles = torch.tensor([-1.0, -0.33, 0.33, 1.0]).expand(256, 4)
    upper = torch.full((256,), 100.0)  # nothing is an outlier
    lower = torch.full((256,), -100.0)
    out, mask = quantize_nuq(values, poles, upper, lower)
    assert mask.sum() == 0, "no positions should be marked as outliers"
    # Each value rounded to nearest pole; max error == max pole spacing / 2
    err = (out - values).abs().max().item()
    assert err < 0.34, f"reconstruction error {err:.3f} too high"


def test_quantize_dequantize_with_outliers_preserves_them() -> None:
    torch.manual_seed(0)
    values = torch.tensor([-5.0, -0.3, 0.0, 0.4, 7.0])
    poles = torch.tensor([-0.5, 0.0, 0.5]).expand(5, 3)
    upper = torch.full((5,), 1.0)
    lower = torch.full((5,), -1.0)
    out, mask = quantize_nuq(values, poles, upper, lower)
    # Outlier positions keep the original value bit-for-bit
    assert out[0].item() == -5.0
    assert out[-1].item() == 7.0
    assert mask.tolist() == [True, False, False, False, True]


def test_full_dequant_pipeline_recovers_outliers_and_codes() -> None:
    """Closed-loop test: quantize -> store codes + outliers -> dequant."""
    torch.manual_seed(0)
    values = torch.randn(8, 16)
    # Compute per-row min/max as a stand-in for calibrated thresholds.
    upper = values.quantile(0.95, dim=-1, keepdim=True).expand_as(values)
    lower = values.quantile(0.05, dim=-1, keepdim=True).expand_as(values)
    poles = torch.linspace(-1, 1, 4).expand(8, 16, 4)
    # Reference path
    reconstructed_ref, mask = quantize_nuq(values, poles, upper, lower)
    # Simulate code-storage path: codes = pole index, outliers = dense fp16
    diff = (values.unsqueeze(-1) - poles).abs()
    codes = diff.argmin(dim=-1)
    outliers = torch.where(mask, values, torch.zeros_like(values))
    recovered = dequantize_nuq(codes, outliers, mask, poles)
    assert torch.allclose(recovered, reconstructed_ref, atol=1e-6), (
        f"max diff {(recovered - reconstructed_ref).abs().max().item():.3e}"
    )


if __name__ == "__main__":
    tests = [
        ("round_to_nearest_pole_simple", test_round_to_nearest_pole_simple),
        ("outliers_threshold_band", test_outliers_threshold_band),
        ("outliers_first_few_fp16", test_outliers_first_few_fp16),
        ("quantize_dequantize_roundtrip_no_outliers",
         test_quantize_dequantize_roundtrip_no_outliers),
        ("quantize_dequantize_with_outliers_preserves_them",
         test_quantize_dequantize_with_outliers_preserves_them),
        ("full_dequant_pipeline_recovers_outliers_and_codes",
         test_full_dequant_pipeline_recovers_outliers_and_codes),
    ]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok  {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {name}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
