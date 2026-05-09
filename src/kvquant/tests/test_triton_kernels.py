"""Triton-vs-reference parity check for nuq2 pack/unpack.

The reference path in :mod:`kvquant.nuq` is the spec; the Triton
kernels must reproduce it exactly (modulo the fp32->fp16 round-trip
the kernel performs internally).

Run with:
    python src/kvquant/tests/test_triton_kernels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvquant.nuq import quantize_nuq
from kvquant.triton_kernels import (
    pack_keys_per_channel,
    pack_values_per_token,
    unpack_keys_per_channel,
    unpack_values_per_token,
)


def _make_calibrated_inputs(T: int, D: int, seed: int = 0):
    """Build a (T, D) input plus calibrated per-channel poles/thresholds."""
    torch.manual_seed(seed)
    values = torch.randn(T, D, dtype=torch.float16, device="cuda")
    # nuq2 -> 4 reconstruction levels per channel. Place them at
    # quantiles to simulate a real calibration.
    fp32_vals = values.float()
    qs = torch.tensor([0.125, 0.375, 0.625, 0.875], device="cuda")
    poles = torch.empty(D, 4, dtype=torch.float16, device="cuda")
    upper = torch.empty(D, dtype=torch.float16, device="cuda")
    lower = torch.empty(D, dtype=torch.float16, device="cuda")
    for d in range(D):
        col = fp32_vals[:, d]
        poles[d] = col.quantile(qs).to(torch.float16)
        upper[d] = col.quantile(0.99).to(torch.float16)
        lower[d] = col.quantile(0.01).to(torch.float16)
    return values, poles, upper, lower


def test_pack_unpack_roundtrip_matches_reference() -> None:
    T, D = 64, 128
    values, poles, upper, lower = _make_calibrated_inputs(T, D)
    # --- Triton path ---
    codes, outlier_v, outlier_m = pack_keys_per_channel(values, poles, upper, lower)
    triton_recon = unpack_keys_per_channel(codes, outlier_v, outlier_m, poles)
    # --- Reference (pure-PyTorch) path ---
    poles_b = poles.unsqueeze(0).expand(T, D, 4)         # (T, D, 4)
    upper_b = upper.unsqueeze(0).expand(T, D)            # (T, D)
    lower_b = lower.unsqueeze(0).expand(T, D)
    ref_recon, ref_mask = quantize_nuq(values, poles_b, upper_b, lower_b)

    # The Triton kernel computes everything in fp32 internally then
    # casts back to fp16; the reference path uses fp16 throughout.
    # Allow a small tolerance for the arg-min ties this can shift.
    diff = (triton_recon.float() - ref_recon.float()).abs()
    max_err = diff.max().item()
    assert max_err < 5e-3, f"max recon diff {max_err:.3e} exceeds tolerance"

    triton_mask = (outlier_m != 0)
    # Mask agreement should be perfect because the comparison is on
    # the exact same fp16 thresholds.
    mismatch = (triton_mask != ref_mask).sum().item()
    total = ref_mask.numel()
    assert mismatch == 0, f"{mismatch}/{total} outlier-mask positions differ"


def test_codes_in_range() -> None:
    T, D = 32, 64
    values, poles, upper, lower = _make_calibrated_inputs(T, D)
    codes, _, _ = pack_keys_per_channel(values, poles, upper, lower)
    # nuq2 -> 4 levels -> codes in [0, 3]
    assert codes.max().item() <= 3
    assert codes.min().item() >= 0


def _make_calibrated_inputs_per_token(T: int, D: int, seed: int = 0):
    """Build a (T, D) input plus per-token poles/thresholds (Values style)."""
    torch.manual_seed(seed)
    values = torch.randn(T, D, dtype=torch.float16, device="cuda")
    fp32_vals = values.float()
    qs = torch.tensor([0.125, 0.375, 0.625, 0.875], device="cuda")
    poles = torch.empty(T, 4, dtype=torch.float16, device="cuda")
    upper = torch.empty(T, dtype=torch.float16, device="cuda")
    lower = torch.empty(T, dtype=torch.float16, device="cuda")
    for t in range(T):
        row = fp32_vals[t]
        poles[t] = row.quantile(qs).to(torch.float16)
        upper[t] = row.quantile(0.99).to(torch.float16)
        lower[t] = row.quantile(0.01).to(torch.float16)
    return values, poles, upper, lower


def test_per_token_pack_unpack_roundtrip_matches_reference() -> None:
    T, D = 64, 128
    values, poles, upper, lower = _make_calibrated_inputs_per_token(T, D)
    codes, outlier_v, outlier_m = pack_values_per_token(values, poles, upper, lower)
    triton_recon = unpack_values_per_token(codes, outlier_v, outlier_m, poles)

    # Reference: poles broadcast across head_dim, thresholds same.
    poles_b = poles.unsqueeze(1).expand(T, D, 4)         # (T, D, 4)
    upper_b = upper.unsqueeze(1).expand(T, D)
    lower_b = lower.unsqueeze(1).expand(T, D)
    ref_recon, ref_mask = quantize_nuq(values, poles_b, upper_b, lower_b)

    diff = (triton_recon.float() - ref_recon.float()).abs()
    max_err = diff.max().item()
    assert max_err < 5e-3, f"per-token max diff {max_err:.3e}"
    triton_mask = (outlier_m != 0)
    mismatch = (triton_mask != ref_mask).sum().item()
    assert mismatch == 0, f"{mismatch} per-token outlier-mask mismatches"


def test_outlier_path_preserves_value_bit_for_bit() -> None:
    T, D = 16, 32
    values, poles, upper, lower = _make_calibrated_inputs(T, D)
    # Force one position to be a guaranteed outlier
    values[0, 0] = torch.tensor(99.0, dtype=torch.float16, device="cuda")
    codes, outlier_v, outlier_m = pack_keys_per_channel(values, poles, upper, lower)
    assert outlier_m[0, 0].item() == 1, "manual outlier was not flagged"
    assert outlier_v[0, 0].item() == 99.0, "outlier value was clobbered"
    recon = unpack_keys_per_channel(codes, outlier_v, outlier_m, poles)
    assert recon[0, 0].item() == 99.0, "outlier was not preserved on dequant"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        sys.exit(0)
    tests = [
        ("pack_unpack_roundtrip_matches_reference",
         test_pack_unpack_roundtrip_matches_reference),
        ("codes_in_range", test_codes_in_range),
        ("outlier_path_preserves_value_bit_for_bit",
         test_outlier_path_preserves_value_bit_for_bit),
        ("per_token_pack_unpack_roundtrip_matches_reference",
         test_per_token_pack_unpack_roundtrip_matches_reference),
    ]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok  {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {name}: {e}")
        except Exception as e:
            fails += 1
            print(f"  ERR  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
