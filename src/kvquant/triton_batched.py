"""Batched multi-head pack/unpack kernels for nuq quantisation.

The per-head kernels in :mod:`triton_kernels` operate on shape
``(T, D)`` and force callers to loop over heads in Python. For Qwen2.5-7B
(num_kv_heads=4, num_layers=28) that means 4 launches × 28 layers × 2
(pack+unpack) = 224 launches per forward, which dominates wall-clock
at long context. These kernels fold the head axis into the launch
grid, collapsing the launch count to 28 × 2 = 56 and unblocking
long-context measurement.

Layout convention:
    values : (T, H, D)  fp16
    poles  : (H, D, L)  fp16   for Keys (per-channel-per-head)
             (H, T, L)  fp16   for Values (per-token-per-head)
    upper, lower : same as poles minus the L axis

Grid is ``(cdiv(T, BLOCK_T), H, cdiv(D, BLOCK_D))``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# --- per-channel (Keys): poles shape (H, D, L) -----------------------------

@triton.jit
def _pack_per_channel_batched(
    values_ptr,        # (T, H, D)
    poles_ptr,         # (H, D, L)
    upper_ptr,         # (H, D)
    lower_ptr,         # (H, D)
    codes_ptr,         # (T, H, D) uint8
    outlier_v_ptr,     # (T, H, D) fp16
    outlier_m_ptr,     # (T, H, D) uint8
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < D
    mask_2d = mask_t[:, None] & mask_d[None, :]

    # values stride: T*H*D, layer-flat. addr = t*H*D + pid_h*D + d
    base_v = pid_h * D
    val_addr = values_ptr + offs_t[:, None] * (H * D) + base_v + offs_d[None, :]
    vals = tl.load(val_addr, mask=mask_2d, other=0.0).to(tl.float32)

    # poles, upper, lower: per (H, D, *)
    poles_base = pid_h * D * NUM_LEVELS + offs_d * NUM_LEVELS
    upper = tl.load(upper_ptr + pid_h * D + offs_d, mask=mask_d, other=1e9).to(tl.float32)
    lower = tl.load(lower_ptr + pid_h * D + offs_d, mask=mask_d, other=-1e9).to(tl.float32)
    is_outlier = (vals > upper[None, :]) | (vals < lower[None, :])
    is_outlier = is_outlier & mask_2d

    best_idx = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.int32)
    best_diff = tl.full((BLOCK_T, BLOCK_D), float("inf"), dtype=tl.float32)
    for lvl in tl.static_range(NUM_LEVELS):
        pole = tl.load(poles_ptr + poles_base + lvl, mask=mask_d, other=0.0).to(tl.float32)
        diff = tl.abs(vals - pole[None, :])
        is_better = diff < best_diff
        best_diff = tl.where(is_better, diff, best_diff)
        best_idx = tl.where(is_better, tl.full((BLOCK_T, BLOCK_D), lvl, dtype=tl.int32), best_idx)

    out_addr = offs_t[:, None] * (H * D) + base_v + offs_d[None, :]
    tl.store(codes_ptr + out_addr, best_idx.to(tl.uint8), mask=mask_2d)
    tl.store(outlier_v_ptr + out_addr,
             tl.where(is_outlier, vals, tl.zeros_like(vals)).to(tl.float16),
             mask=mask_2d)
    tl.store(outlier_m_ptr + out_addr,
             tl.where(is_outlier, tl.full((BLOCK_T, BLOCK_D), 1, dtype=tl.uint8),
                      tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.uint8)),
             mask=mask_2d)


@triton.jit
def _unpack_per_channel_batched(
    codes_ptr,
    outlier_v_ptr,
    outlier_m_ptr,
    poles_ptr,
    out_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < D
    mask_2d = mask_t[:, None] & mask_d[None, :]

    base_v = pid_h * D
    addr = offs_t[:, None] * (H * D) + base_v + offs_d[None, :]
    codes = tl.load(codes_ptr + addr, mask=mask_2d, other=0).to(tl.int32)

    poles_base = pid_h * D * NUM_LEVELS + offs_d * NUM_LEVELS
    recon = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    for lvl in tl.static_range(NUM_LEVELS):
        pole = tl.load(poles_ptr + poles_base + lvl, mask=mask_d, other=0.0).to(tl.float32)
        match = (codes == lvl)
        recon = tl.where(match, pole[None, :], recon)

    out_v = tl.load(outlier_v_ptr + addr, mask=mask_2d, other=0.0).to(tl.float32)
    out_m = tl.load(outlier_m_ptr + addr, mask=mask_2d, other=0).to(tl.int32)
    final = tl.where(out_m != 0, out_v, recon)
    tl.store(out_ptr + addr, final.to(tl.float16), mask=mask_2d)


def pack_keys_per_channel_batched(
    values: torch.Tensor,        # (T, H, D) fp16
    poles: torch.Tensor,         # (H, D, L) fp16
    upper: torch.Tensor,         # (H, D)
    lower: torch.Tensor,         # (H, D)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched-head version of :func:`pack_keys_per_channel`."""
    assert values.is_cuda and values.dtype == torch.float16
    assert values.is_contiguous(), "values must be contiguous (T, H, D)"
    T, H, D = values.shape
    NUM_LEVELS = poles.shape[-1]
    codes = torch.empty((T, H, D), dtype=torch.uint8, device=values.device)
    out_v = torch.empty((T, H, D), dtype=torch.float16, device=values.device)
    out_m = torch.empty((T, H, D), dtype=torch.uint8, device=values.device)
    BLOCK_T = 32
    BLOCK_D = 64
    grid = (triton.cdiv(T, BLOCK_T), H, triton.cdiv(D, BLOCK_D))
    _pack_per_channel_batched[grid](
        values, poles.contiguous(), upper.contiguous(), lower.contiguous(),
        codes, out_v, out_m,
        T=T, H=H, D=D, NUM_LEVELS=NUM_LEVELS,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )
    return codes, out_v, out_m


def unpack_keys_per_channel_batched(
    codes: torch.Tensor,
    outlier_v: torch.Tensor,
    outlier_m: torch.Tensor,
    poles: torch.Tensor,
) -> torch.Tensor:
    """Batched-head version of :func:`unpack_keys_per_channel`."""
    assert codes.is_cuda and codes.dtype == torch.uint8
    assert codes.is_contiguous()
    T, H, D = codes.shape
    NUM_LEVELS = poles.shape[-1]
    out = torch.empty((T, H, D), dtype=torch.float16, device=codes.device)
    BLOCK_T = 32
    BLOCK_D = 64
    grid = (triton.cdiv(T, BLOCK_T), H, triton.cdiv(D, BLOCK_D))
    _unpack_per_channel_batched[grid](
        codes, outlier_v, outlier_m, poles.contiguous(), out,
        T=T, H=H, D=D, NUM_LEVELS=NUM_LEVELS,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )
    return out


# --- per-token (Values): poles shape (H, T, L), tail-extrapolated -----------
# We gather scales by absolute token position. For Phase 1A simplicity the
# caller materialises the gathered (H, T, L) and (H, T) tensors and passes
# them in. This duplicates memory but keeps the kernel simple.

@triton.jit
def _pack_per_token_batched(
    values_ptr,        # (T, H, D)
    poles_ptr,         # (H, T, L)
    upper_ptr,         # (H, T)
    lower_ptr,         # (H, T)
    codes_ptr,
    outlier_v_ptr,
    outlier_m_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < D
    mask_2d = mask_t[:, None] & mask_d[None, :]

    base_v = pid_h * D
    val_addr = values_ptr + offs_t[:, None] * (H * D) + base_v + offs_d[None, :]
    vals = tl.load(val_addr, mask=mask_2d, other=0.0).to(tl.float32)

    # per-(H, T) scales
    upper = tl.load(upper_ptr + pid_h * T + offs_t, mask=mask_t, other=1e9).to(tl.float32)
    lower = tl.load(lower_ptr + pid_h * T + offs_t, mask=mask_t, other=-1e9).to(tl.float32)
    is_outlier = (vals > upper[:, None]) | (vals < lower[:, None])
    is_outlier = is_outlier & mask_2d

    best_idx = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.int32)
    best_diff = tl.full((BLOCK_T, BLOCK_D), float("inf"), dtype=tl.float32)
    for lvl in tl.static_range(NUM_LEVELS):
        pole = tl.load(poles_ptr + pid_h * T * NUM_LEVELS + offs_t * NUM_LEVELS + lvl,
                       mask=mask_t, other=0.0).to(tl.float32)
        diff = tl.abs(vals - pole[:, None])
        is_better = diff < best_diff
        best_diff = tl.where(is_better, diff, best_diff)
        best_idx = tl.where(is_better, tl.full((BLOCK_T, BLOCK_D), lvl, dtype=tl.int32), best_idx)

    out_addr = offs_t[:, None] * (H * D) + base_v + offs_d[None, :]
    tl.store(codes_ptr + out_addr, best_idx.to(tl.uint8), mask=mask_2d)
    tl.store(outlier_v_ptr + out_addr,
             tl.where(is_outlier, vals, tl.zeros_like(vals)).to(tl.float16),
             mask=mask_2d)
    tl.store(outlier_m_ptr + out_addr,
             tl.where(is_outlier, tl.full((BLOCK_T, BLOCK_D), 1, dtype=tl.uint8),
                      tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.uint8)),
             mask=mask_2d)


@triton.jit
def _unpack_per_token_batched(
    codes_ptr,
    outlier_v_ptr,
    outlier_m_ptr,
    poles_ptr,         # (H, T, L)
    out_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < D
    mask_2d = mask_t[:, None] & mask_d[None, :]

    base_v = pid_h * D
    addr = offs_t[:, None] * (H * D) + base_v + offs_d[None, :]
    codes = tl.load(codes_ptr + addr, mask=mask_2d, other=0).to(tl.int32)

    recon = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    for lvl in tl.static_range(NUM_LEVELS):
        pole = tl.load(poles_ptr + pid_h * T * NUM_LEVELS + offs_t * NUM_LEVELS + lvl,
                       mask=mask_t, other=0.0).to(tl.float32)
        match = (codes == lvl)
        recon = tl.where(match, pole[:, None], recon)

    out_v = tl.load(outlier_v_ptr + addr, mask=mask_2d, other=0.0).to(tl.float32)
    out_m = tl.load(outlier_m_ptr + addr, mask=mask_2d, other=0).to(tl.int32)
    final = tl.where(out_m != 0, out_v, recon)
    tl.store(out_ptr + addr, final.to(tl.float16), mask=mask_2d)


def pack_values_per_token_batched(
    values: torch.Tensor,        # (T, H, D) fp16
    poles: torch.Tensor,         # (H, T, L) fp16   (already tail-extrapolated)
    upper: torch.Tensor,         # (H, T)
    lower: torch.Tensor,         # (H, T)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched-head version of :func:`pack_values_per_token`.

    The caller is responsible for tail-extrapolation of the (H, T, L)
    poles tensor — typically by gathering ``scales[h, clamp(t, T_cal-1)]``.
    """
    assert values.is_cuda and values.dtype == torch.float16
    assert values.is_contiguous()
    T, H, D = values.shape
    NUM_LEVELS = poles.shape[-1]
    assert poles.shape == (H, T, NUM_LEVELS), f"poles shape {poles.shape} != ({H},{T},{NUM_LEVELS})"
    codes = torch.empty((T, H, D), dtype=torch.uint8, device=values.device)
    out_v = torch.empty((T, H, D), dtype=torch.float16, device=values.device)
    out_m = torch.empty((T, H, D), dtype=torch.uint8, device=values.device)
    BLOCK_T = 32
    BLOCK_D = 64
    grid = (triton.cdiv(T, BLOCK_T), H, triton.cdiv(D, BLOCK_D))
    _pack_per_token_batched[grid](
        values, poles.contiguous(), upper.contiguous(), lower.contiguous(),
        codes, out_v, out_m,
        T=T, H=H, D=D, NUM_LEVELS=NUM_LEVELS,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )
    return codes, out_v, out_m


def unpack_values_per_token_batched(
    codes: torch.Tensor,
    outlier_v: torch.Tensor,
    outlier_m: torch.Tensor,
    poles: torch.Tensor,
) -> torch.Tensor:
    """Batched-head version of :func:`unpack_values_per_token`."""
    assert codes.is_cuda and codes.dtype == torch.uint8
    assert codes.is_contiguous()
    T, H, D = codes.shape
    NUM_LEVELS = poles.shape[-1]
    out = torch.empty((T, H, D), dtype=torch.float16, device=codes.device)
    BLOCK_T = 32
    BLOCK_D = 64
    grid = (triton.cdiv(T, BLOCK_T), H, triton.cdiv(D, BLOCK_D))
    _unpack_per_token_batched[grid](
        codes, outlier_v, outlier_m, poles.contiguous(), out,
        T=T, H=H, D=D, NUM_LEVELS=NUM_LEVELS,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )
    return out
