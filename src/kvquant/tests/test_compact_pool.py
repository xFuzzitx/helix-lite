"""Parity tests for the Phase 1B compact KV pool.

Approach: build a tiny pool (small L, NB, BS, H, D), generate fake
K/V, write through ``CompactKVPool.write_kv``, then unpack the codes
back to fp16 using the existing reference path
(:func:`kvquant.nuq.dequantize_nuq`) and verify they match the
reference quant→dequant on the same input.

Also checks block_stats min/max equal the reconstructed K min/max
along the position axis within each block.
"""
from __future__ import annotations

import sys
import torch

if not torch.cuda.is_available():
    print("CUDA required", file=sys.stderr)
    sys.exit(0)

from kvquant.compact_pool import CompactKVPool
from kvquant.scales import KVScales, PerChannelScale, PerTokenScale


def _make_tiny_scales(L=2, H=2, D=16, T_cal=32, NL_K=16, NL_V=4, seed=0):
    """Build a small KVScales that matches the input fake K/V."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = []
    vals = []
    for _ in range(L):
        # K scales: per-channel (H, D, NL)
        k_poles = torch.randn(H, D, NL_K, generator=g, device="cuda",
                              dtype=torch.float16).sort(dim=-1).values
        k_upper = torch.full((H, D), 1.5, dtype=torch.float16, device="cuda")
        k_lower = -k_upper
        keys.append(PerChannelScale(poles=k_poles, upper_threshold=k_upper,
                                     lower_threshold=k_lower, num_bits=4))
        # V scales: per-token (H, T_cal, NL)
        v_poles = torch.randn(H, T_cal, NL_V, generator=g, device="cuda",
                              dtype=torch.float16).sort(dim=-1).values
        v_upper = torch.full((H, T_cal), 1.5, dtype=torch.float16, device="cuda")
        v_lower = -v_upper
        vals.append(PerTokenScale(poles=v_poles, upper_threshold=v_upper,
                                   lower_threshold=v_lower, num_bits=2))
    return KVScales(per_layer_keys=keys, per_layer_values=vals,
                    first_few_fp16=4, num_bits=2)


def _reference_quant(values: torch.Tensor, poles, upper, lower):
    """Pure-pytorch reference of round_to_nearest_pole + outlier passthrough.

    Args (``values`` is (D,)):
        poles: (D, NL) for per-channel (K) scales, OR (NL,) for per-token (V)
        upper, lower: same broadcast pattern as poles minus the NL axis.
    """
    values_2d = values[:, None]  # (D, 1)
    if poles.dim() == 2:
        # per-channel: (D, NL)
        diff = (values_2d - poles).abs()
        best = diff.argmin(dim=-1)
        recon = poles.gather(-1, best[:, None]).squeeze(-1)
        out_mask = (values > upper) | (values < lower)
    else:
        # per-token shared across D: poles (NL,), upper/lower scalars
        diff = (values_2d - poles[None, :]).abs()  # (D, NL)
        best = diff.argmin(dim=-1)
        recon = poles[best]
        out_mask = (values > upper) | (values < lower)
    return torch.where(out_mask, values, recon), out_mask


def test_compact_pool_pack_round_trip():
    """Pack random K/V; unpack via codes + outliers; compare to reference."""
    torch.manual_seed(0)
    L, NB, BS, H, D = 2, 4, 8, 2, 16
    T_cal = 32
    scales = _make_tiny_scales(L=L, H=H, D=D, T_cal=T_cal)
    pool = CompactKVPool(num_layers=L, num_blocks=NB, block_size=BS,
                          num_kv_heads=H, head_size=D, scales=scales)

    # 10 tokens, mapped to consecutive slots in block 0 and 1
    T = 10
    key = torch.randn(T, H, D, dtype=torch.float16, device="cuda")
    value = torch.randn(T, H, D, dtype=torch.float16, device="cuda")
    slot_mapping = torch.arange(T, dtype=torch.int64, device="cuda")  # slot 0..9
    abs_positions = torch.arange(T, dtype=torch.int64, device="cuda")

    for L_idx in range(L):
        pool.write_kv(L_idx, key, value, slot_mapping, abs_positions)

    # For each token / layer / head, manually compare against reference
    for L_idx in range(L):
        k_scale = scales.per_layer_keys[L_idx]
        v_scale = scales.per_layer_values[L_idx]
        for t in range(T):
            block_id = t // BS
            in_block = t % BS
            for h in range(H):
                # K: per-channel scales
                k_ref, k_outmask_ref = _reference_quant(
                    key[t, h].to(torch.float32),
                    k_scale.poles[h].to(torch.float32),
                    k_scale.upper_threshold[h].to(torch.float32),
                    k_scale.lower_threshold[h].to(torch.float32),
                )
                k_codes_pool = pool.k_codes[L_idx, block_id, in_block, h]
                k_outv_pool = pool.k_outv[L_idx, block_id, in_block, h]
                k_outm_pool = pool.k_outm[L_idx, block_id, in_block, h].bool()
                # Reconstruct from pool
                k_recon = torch.gather(k_scale.poles[h], -1,
                                       k_codes_pool.long()[:, None]).squeeze(-1)
                k_recon = torch.where(k_outm_pool, k_outv_pool, k_recon)
                torch.testing.assert_close(k_recon.float(), k_ref.float(),
                                           atol=1e-2, rtol=1e-2)
                assert torch.equal(k_outm_pool, k_outmask_ref), \
                    f"K outlier mask mismatch at L={L_idx} t={t} h={h}"

                # V: per-token scales (use abs_positions[t] clamped to T_cal-1)
                pos = min(int(abs_positions[t].item()), T_cal - 1)
                v_ref, v_outmask_ref = _reference_quant(
                    value[t, h].to(torch.float32),
                    v_scale.poles[h, pos].to(torch.float32),
                    v_scale.upper_threshold[h, pos].to(torch.float32),
                    v_scale.lower_threshold[h, pos].to(torch.float32),
                )
                v_codes_pool = pool.v_codes[L_idx, block_id, in_block, h]
                v_outv_pool = pool.v_outv[L_idx, block_id, in_block, h]
                v_outm_pool = pool.v_outm[L_idx, block_id, in_block, h].bool()
                # V poles are (NL,) for this (h, pos) — fancy-index by codes
                v_recon = v_scale.poles[h, pos][v_codes_pool.long()]
                v_recon = torch.where(v_outm_pool, v_outv_pool, v_recon)
                torch.testing.assert_close(v_recon.float(), v_ref.float(),
                                           atol=1e-2, rtol=1e-2)


def test_compact_pool_pack_unpack_round_trip():
    """Pack tokens, then unpack the blocks via the staging kernel.
    Each (block, position, head, dim) should match the in-flight
    reference quant→dequant on the same input."""
    torch.manual_seed(2)
    L, NB, BS, H, D = 1, 3, 8, 2, 16
    T_cal = 64
    scales = _make_tiny_scales(L=L, H=H, D=D, T_cal=T_cal)
    pool = CompactKVPool(num_layers=L, num_blocks=NB, block_size=BS,
                          num_kv_heads=H, head_size=D, scales=scales)

    # Fill blocks 0 and 1 (slots 0..15) — leave block 2 empty
    T = 16
    key = torch.randn(T, H, D, dtype=torch.float16, device="cuda")
    value = torch.randn_like(key)
    slot_mapping = torch.arange(T, dtype=torch.int64, device="cuda")
    abs_positions = torch.arange(T, dtype=torch.int64, device="cuda")
    pool.write_kv(0, key, value, slot_mapping, abs_positions)

    # Unpack blocks 0 and 1
    block_id_list = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    block_first_position = torch.tensor([0, BS], dtype=torch.int64, device="cuda")
    NK = block_id_list.shape[0]
    staging_k = torch.empty((NK, BS, H, D), dtype=torch.float16, device="cuda")
    staging_v = torch.empty_like(staging_k)
    pool.unpack_to_staging(0, block_id_list, block_first_position,
                            staging_k, staging_v)

    # Compare against in-flight reference for every position
    k_scale = scales.per_layer_keys[0]
    v_scale = scales.per_layer_values[0]
    for k_idx in range(NK):
        block_id = int(block_id_list[k_idx].item())
        start = int(block_first_position[k_idx].item())
        for in_block in range(BS):
            abs_t = block_id * BS + in_block  # equals original slot
            pos = min(start + in_block, T_cal - 1)
            for h in range(H):
                k_ref, _ = _reference_quant(
                    key[abs_t, h].to(torch.float32),
                    k_scale.poles[h].to(torch.float32),
                    k_scale.upper_threshold[h].to(torch.float32),
                    k_scale.lower_threshold[h].to(torch.float32),
                )
                v_ref, _ = _reference_quant(
                    value[abs_t, h].to(torch.float32),
                    v_scale.poles[h, pos].to(torch.float32),
                    v_scale.upper_threshold[h, pos].to(torch.float32),
                    v_scale.lower_threshold[h, pos].to(torch.float32),
                )
                torch.testing.assert_close(
                    staging_k[k_idx, in_block, h].float(),
                    k_ref.float(), atol=1e-2, rtol=1e-2,
                    msg=f"K mismatch at kept={k_idx} pos={in_block} h={h}",
                )
                torch.testing.assert_close(
                    staging_v[k_idx, in_block, h].float(),
                    v_ref.float(), atol=1e-2, rtol=1e-2,
                    msg=f"V mismatch at kept={k_idx} pos={in_block} h={h}",
                )


def test_compact_pool_block_stats():
    """k_min/k_max per block must match the actual min/max of reconstructed K."""
    torch.manual_seed(1)
    L, NB, BS, H, D = 1, 2, 4, 2, 8
    T_cal = 16
    scales = _make_tiny_scales(L=L, H=H, D=D, T_cal=T_cal)
    pool = CompactKVPool(num_layers=L, num_blocks=NB, block_size=BS,
                          num_kv_heads=H, head_size=D, scales=scales)

    # Fill block 0 (slots 0..3) and block 1 (slots 4..7)
    T = 8
    key = torch.randn(T, H, D, dtype=torch.float16, device="cuda")
    value = torch.randn_like(key)
    slot_mapping = torch.arange(T, dtype=torch.int64, device="cuda")
    abs_positions = torch.arange(T, dtype=torch.int64, device="cuda")

    pool.write_kv(0, key, value, slot_mapping, abs_positions)

    # For each block, compute reference reconstructed-K min/max along positions
    k_scale = scales.per_layer_keys[0]
    for block_id in range(NB):
        slot_range = slice(block_id * BS, (block_id + 1) * BS)
        block_keys = key[slot_range]  # (BS, H, D)
        # Quantize each token via reference
        recons = []
        for bs in range(block_keys.shape[0]):
            for h in range(H):
                r, _ = _reference_quant(
                    block_keys[bs, h].to(torch.float32),
                    k_scale.poles[h].to(torch.float32),
                    k_scale.upper_threshold[h].to(torch.float32),
                    k_scale.lower_threshold[h].to(torch.float32),
                )
                recons.append(r)
        recons = torch.stack(recons).reshape(BS, H, D)
        ref_min = recons.amin(dim=0)
        ref_max = recons.amax(dim=0)
        pool_min = pool.k_min[0, block_id]  # fp32
        pool_max = pool.k_max[0, block_id]
        torch.testing.assert_close(pool_min, ref_min,
                                   atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(pool_max, ref_max,
                                   atol=1e-2, rtol=1e-2)
