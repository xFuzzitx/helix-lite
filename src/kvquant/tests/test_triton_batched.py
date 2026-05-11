"""Parity tests: batched-head kernels vs per-head reference kernels.

The batched kernels in :mod:`kvquant.triton_batched` collapse the
Python loop over heads into a single launch grid. They must produce
bit-identical results to the per-head versions in
:mod:`kvquant.triton_kernels` — otherwise vLLM integration will
silently corrupt long-context outputs.
"""
from __future__ import annotations

import torch

if not torch.cuda.is_available():
    import sys
    print("CUDA required, skipping", file=sys.stderr)
    sys.exit(0)

from kvquant.triton_kernels import (
    pack_keys_per_channel, unpack_keys_per_channel,
    pack_values_per_token, unpack_values_per_token,
)
from kvquant.triton_batched import (
    pack_keys_per_channel_batched, unpack_keys_per_channel_batched,
    pack_values_per_token_batched, unpack_values_per_token_batched,
)


def _make_keys(T=128, H=4, D=128, L=16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn(T, H, D, generator=g, device="cuda", dtype=torch.float16)
    poles = torch.randn(H, D, L, generator=g, device="cuda", dtype=torch.float16).sort(dim=-1).values
    sigma = values.abs().mean()
    upper = (sigma * 1.5).expand(H, D).to(torch.float16).contiguous()
    lower = -upper
    return values, poles, upper, lower


def _make_values(T=128, H=4, D=128, L=16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn(T, H, D, generator=g, device="cuda", dtype=torch.float16)
    poles = torch.randn(H, T, L, generator=g, device="cuda", dtype=torch.float16).sort(dim=-1).values
    sigma = values.abs().mean()
    upper = (sigma * 1.5).expand(H, T).to(torch.float16).contiguous()
    lower = -upper
    return values, poles, upper, lower


def _per_head_pack_keys(values, poles, upper, lower):
    T, H, D = values.shape
    codes = torch.empty((T, H, D), dtype=torch.uint8, device=values.device)
    out_v = torch.empty((T, H, D), dtype=torch.float16, device=values.device)
    out_m = torch.empty((T, H, D), dtype=torch.uint8, device=values.device)
    for h in range(H):
        c, v, m = pack_keys_per_channel(
            values[:, h, :].contiguous(),
            poles[h].contiguous(),
            upper[h].contiguous(),
            lower[h].contiguous(),
        )
        codes[:, h, :] = c
        out_v[:, h, :] = v
        out_m[:, h, :] = m
    return codes, out_v, out_m


def _per_head_pack_values(values, poles, upper, lower):
    T, H, D = values.shape
    codes = torch.empty((T, H, D), dtype=torch.uint8, device=values.device)
    out_v = torch.empty((T, H, D), dtype=torch.float16, device=values.device)
    out_m = torch.empty((T, H, D), dtype=torch.uint8, device=values.device)
    for h in range(H):
        c, v, m = pack_values_per_token(
            values[:, h, :].contiguous(),
            poles[h].contiguous(),
            upper[h].contiguous(),
            lower[h].contiguous(),
        )
        codes[:, h, :] = c
        out_v[:, h, :] = v
        out_m[:, h, :] = m
    return codes, out_v, out_m


def test_pack_keys_per_channel_batched_matches_per_head():
    values, poles, upper, lower = _make_keys()
    c_ref, v_ref, m_ref = _per_head_pack_keys(values, poles, upper, lower)
    c_bat, v_bat, m_bat = pack_keys_per_channel_batched(values, poles, upper, lower)
    torch.testing.assert_close(c_bat, c_ref, atol=0, rtol=0)
    torch.testing.assert_close(v_bat, v_ref, atol=0, rtol=0)
    torch.testing.assert_close(m_bat, m_ref, atol=0, rtol=0)


def test_unpack_keys_per_channel_batched_matches_per_head():
    values, poles, upper, lower = _make_keys()
    c_bat, v_bat, m_bat = pack_keys_per_channel_batched(values, poles, upper, lower)
    out_bat = unpack_keys_per_channel_batched(c_bat, v_bat, m_bat, poles)

    T, H, D = values.shape
    out_ref = torch.empty_like(values)
    for h in range(H):
        out_ref[:, h, :] = unpack_keys_per_channel(
            c_bat[:, h, :].contiguous(),
            v_bat[:, h, :].contiguous(),
            m_bat[:, h, :].contiguous(),
            poles[h].contiguous(),
        )
    torch.testing.assert_close(out_bat, out_ref, atol=0, rtol=0)


def test_pack_values_per_token_batched_matches_per_head():
    values, poles, upper, lower = _make_values()
    c_ref, v_ref, m_ref = _per_head_pack_values(values, poles, upper, lower)
    c_bat, v_bat, m_bat = pack_values_per_token_batched(values, poles, upper, lower)
    torch.testing.assert_close(c_bat, c_ref, atol=0, rtol=0)
    torch.testing.assert_close(v_bat, v_ref, atol=0, rtol=0)
    torch.testing.assert_close(m_bat, m_ref, atol=0, rtol=0)


def test_unpack_values_per_token_batched_matches_per_head():
    values, poles, upper, lower = _make_values()
    c_bat, v_bat, m_bat = pack_values_per_token_batched(values, poles, upper, lower)
    out_bat = unpack_values_per_token_batched(c_bat, v_bat, m_bat, poles)

    T, H, D = values.shape
    out_ref = torch.empty_like(values)
    for h in range(H):
        out_ref[:, h, :] = unpack_values_per_token(
            c_bat[:, h, :].contiguous(),
            v_bat[:, h, :].contiguous(),
            m_bat[:, h, :].contiguous(),
            poles[h].contiguous(),
        )
    torch.testing.assert_close(out_bat, out_ref, atol=0, rtol=0)


def test_batched_roundtrip_keys_within_quant_error():
    """End-to-end: pack then unpack via batched kernels; non-outlier
    positions should round-trip to their nearest pole."""
    values, poles, upper, lower = _make_keys(T=256, H=4, D=128, L=16)
    codes, out_v, out_m = pack_keys_per_channel_batched(values, poles, upper, lower)
    recon = unpack_keys_per_channel_batched(codes, out_v, out_m, poles)
    # Outliers should be exact
    out_mask = out_m.bool()
    torch.testing.assert_close(recon[out_mask], values[out_mask], atol=0, rtol=0)
    # Non-outliers: each is its nearest pole (per channel per head)
    # Sanity: max error bounded by 0.5 × max pole spacing
    max_spacing = (poles[..., 1:] - poles[..., :-1]).max()
    err = (recon - values).abs().to(torch.float32)
    assert err[~out_mask].max() <= max_spacing.to(torch.float32) * 0.51, (
        f"max non-outlier err {err[~out_mask].max()} > 0.5 × spacing {max_spacing}"
    )
