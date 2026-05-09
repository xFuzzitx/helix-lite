"""Triton kernels for nuq2 KV cache packing/unpacking.

Layout convention (single layer, single head):

* Input KV: ``(T, D)`` fp16 (one token of one head, D = head_dim).
* Quantised storage:
    - ``codes``     : ``(T, D)`` uint8 (one 2-bit code per element,
                      stored byte-padded for now; bit-packing into
                      uint32 is a follow-up optim once correctness is
                      proven).
    - ``outlier_v`` : ``(T, D)`` fp16, zero where not outlier; holds
                      the dense fp16 value at outlier positions.
    - ``outlier_m`` : ``(T, D)`` bool, True at outlier positions.

This first-pass storage is *only* 2x compression rather than 8x, but
it's the simplest correctness anchor. Once attention works end to
end we replace ``codes`` with bit-packed uint32 and ``outlier_v`` with
a (CSR-ish) sparse representation, which gets us to the target 8x.

The kernels are batched over (T, D) and broadcast scales according
to the scale-group conventions in :mod:`scales`.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# -- Pack: fp16 -> codes (2-bit, byte-stored for now) + dense outliers ----

@triton.jit
def _pack_per_channel_kernel(
    values_ptr,        # (T, D) fp16, K activations for one head
    poles_ptr,         # (D, NUM_LEVELS) fp16, per-channel poles
    upper_ptr,         # (D,) fp16
    lower_ptr,         # (D,) fp16
    codes_ptr,         # (T, D) uint8 (only NUM_BITS=2 used)
    outlier_v_ptr,     # (T, D) fp16, dense outlier holder
    outlier_m_ptr,     # (T, D) uint8 (0/1)
    T: tl.constexpr,
    D: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < D
    mask_2d = mask_t[:, None] & mask_d[None, :]

    # Load (BLOCK_T, BLOCK_D) values
    val_addr = values_ptr + offs_t[:, None] * D + offs_d[None, :]
    vals = tl.load(val_addr, mask=mask_2d, other=0.0).to(tl.float32)

    # Per-channel thresholds (BLOCK_D,)
    upper = tl.load(upper_ptr + offs_d, mask=mask_d, other=1e9).to(tl.float32)
    lower = tl.load(lower_ptr + offs_d, mask=mask_d, other=-1e9).to(tl.float32)
    # Outlier mask
    is_outlier = (vals > upper[None, :]) | (vals < lower[None, :])
    is_outlier = is_outlier & mask_2d

    # Find nearest pole index for each value. NUM_LEVELS is small
    # (4 for nuq2, 16 for nuq4) so we unroll over the level axis.
    best_idx = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.int32)
    best_diff = tl.full((BLOCK_T, BLOCK_D), float("inf"), dtype=tl.float32)
    for lvl in tl.static_range(NUM_LEVELS):
        pole = tl.load(poles_ptr + offs_d * NUM_LEVELS + lvl, mask=mask_d, other=0.0).to(tl.float32)
        diff = tl.abs(vals - pole[None, :])
        is_better = diff < best_diff
        best_diff = tl.where(is_better, diff, best_diff)
        best_idx = tl.where(is_better, tl.full((BLOCK_T, BLOCK_D), lvl, dtype=tl.int32), best_idx)

    # Write outputs
    code_addr = codes_ptr + offs_t[:, None] * D + offs_d[None, :]
    tl.store(code_addr, best_idx.to(tl.uint8), mask=mask_2d)

    out_v_addr = outlier_v_ptr + offs_t[:, None] * D + offs_d[None, :]
    out_m_addr = outlier_m_ptr + offs_t[:, None] * D + offs_d[None, :]
    tl.store(
        out_v_addr,
        tl.where(is_outlier, vals, tl.zeros_like(vals)).to(tl.float16),
        mask=mask_2d,
    )
    tl.store(
        out_m_addr,
        tl.where(is_outlier, tl.full((BLOCK_T, BLOCK_D), 1, dtype=tl.uint8),
                 tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.uint8)),
        mask=mask_2d,
    )


def pack_keys_per_channel(
    values: torch.Tensor,        # (T, D) fp16
    poles: torch.Tensor,         # (D, NUM_LEVELS) fp16
    upper: torch.Tensor,         # (D,) fp16
    lower: torch.Tensor,         # (D,) fp16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise per-channel Keys with the Triton kernel.

    Returns ``(codes uint8 (T,D), outlier_v fp16 (T,D), outlier_m
    uint8 (T,D))``.

    This is the per-head call; callers tile over (B, H) themselves
    to keep the kernel signature simple.
    """
    assert values.is_cuda and values.dtype == torch.float16
    T, D = values.shape
    NUM_LEVELS = poles.shape[-1]
    codes = torch.empty(T, D, dtype=torch.uint8, device=values.device)
    out_v = torch.empty(T, D, dtype=torch.float16, device=values.device)
    out_m = torch.empty(T, D, dtype=torch.uint8, device=values.device)
    BLOCK_T = 32
    BLOCK_D = 64
    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(D, BLOCK_D))
    _pack_per_channel_kernel[grid](
        values, poles.contiguous(), upper, lower,
        codes, out_v, out_m,
        T=T, D=D, NUM_LEVELS=NUM_LEVELS,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )
    return codes, out_v, out_m


# -- Unpack: codes + dense outliers -> fp16 --------------------------------

@triton.jit
def _unpack_per_channel_kernel(
    codes_ptr,         # (T, D) uint8
    outlier_v_ptr,     # (T, D) fp16
    outlier_m_ptr,     # (T, D) uint8
    poles_ptr,         # (D, NUM_LEVELS) fp16
    out_ptr,           # (T, D) fp16
    T: tl.constexpr,
    D: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < D
    mask_2d = mask_t[:, None] & mask_d[None, :]

    code_addr = codes_ptr + offs_t[:, None] * D + offs_d[None, :]
    codes = tl.load(code_addr, mask=mask_2d, other=0).to(tl.int32)

    # Reconstruct from poles via gather. NUM_LEVELS is small, unroll.
    recon = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    for lvl in tl.static_range(NUM_LEVELS):
        pole = tl.load(poles_ptr + offs_d * NUM_LEVELS + lvl, mask=mask_d, other=0.0).to(tl.float32)
        match = (codes == lvl)
        recon = tl.where(match, pole[None, :], recon)

    # Apply outliers
    out_v = tl.load(outlier_v_ptr + offs_t[:, None] * D + offs_d[None, :],
                    mask=mask_2d, other=0.0).to(tl.float32)
    out_m = tl.load(outlier_m_ptr + offs_t[:, None] * D + offs_d[None, :],
                    mask=mask_2d, other=0).to(tl.int32)
    final = tl.where(out_m != 0, out_v, recon)

    tl.store(
        out_ptr + offs_t[:, None] * D + offs_d[None, :],
        final.to(tl.float16),
        mask=mask_2d,
    )


def unpack_keys_per_channel(
    codes: torch.Tensor,        # (T, D) uint8
    outlier_v: torch.Tensor,    # (T, D) fp16
    outlier_m: torch.Tensor,    # (T, D) uint8
    poles: torch.Tensor,        # (D, NUM_LEVELS) fp16
) -> torch.Tensor:
    """Invert :func:`pack_keys_per_channel`."""
    assert codes.is_cuda and codes.dtype == torch.uint8
    T, D = codes.shape
    NUM_LEVELS = poles.shape[-1]
    out = torch.empty(T, D, dtype=torch.float16, device=codes.device)
    BLOCK_T = 32
    BLOCK_D = 64
    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(D, BLOCK_D))
    _unpack_per_channel_kernel[grid](
        codes, outlier_v, outlier_m, poles.contiguous(), out,
        T=T, D=D, NUM_LEVELS=NUM_LEVELS,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )
    return out
