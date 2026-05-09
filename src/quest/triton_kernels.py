"""Triton kernels for Quest decode-time page selection.

The hot path of Quest at decode is the per-page upper-bound score:
for ``P`` pages, ``H`` heads, ``D`` head_dim, compute one fp16 number
per ``(page, head)``. The reference implementation in
:mod:`reference` does this in two broadcast-multiply-then-sum
operations; this Triton kernel fuses them into a single pass over
``(P, H, D)`` with per-block reduction across ``D``.

Storage convention matches :mod:`reference`:
* ``query`` ``(H, D)`` fp16
* ``K_min``, ``K_max`` ``(P, H, D)`` fp16
* output ``bound`` ``(P, H)`` fp16

For Qwen2.5-7B (H=4 KV heads, D=128) and 16-token pages, an 8K-token
context has 512 pages; this kernel takes about 0.1 ms on a 3090.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _page_upper_bound_kernel(
    query_ptr,         # (H, D) fp16
    kmin_ptr,          # (P, H, D) fp16
    kmax_ptr,          # (P, H, D) fp16
    out_ptr,           # (P, H) fp16
    P: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Sum over D in BLOCK_D tiles, accumulator in fp32 for numerical safety.
    acc = tl.zeros((), dtype=tl.float32)
    for d_off in range(0, D, BLOCK_D):
        offs_d = d_off + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        q_addr = query_ptr + pid_h * D + offs_d
        kmin_addr = kmin_ptr + pid_p * H * D + pid_h * D + offs_d
        kmax_addr = kmax_ptr + pid_p * H * D + pid_h * D + offs_d

        q = tl.load(q_addr, mask=mask_d, other=0.0).to(tl.float32)
        kmin = tl.load(kmin_addr, mask=mask_d, other=0.0).to(tl.float32)
        kmax = tl.load(kmax_addr, mask=mask_d, other=0.0).to(tl.float32)

        q_pos = tl.maximum(q, 0.0)
        q_neg = tl.minimum(q, 0.0)
        acc += tl.sum(q_pos * kmax + q_neg * kmin, axis=0)

    tl.store(out_ptr + pid_p * H + pid_h, acc.to(tl.float16))


def page_upper_bound_triton(
    query: torch.Tensor,
    K_min: torch.Tensor,
    K_max: torch.Tensor,
) -> torch.Tensor:
    """Triton-accelerated counterpart of :func:`reference.page_upper_bound`.

    Args mirror the reference; output is ``(P, H)`` fp16 on the same
    device as the inputs.
    """
    if query.dim() != 2 or K_min.dim() != 3 or K_max.shape != K_min.shape:
        raise ValueError(
            f"shape mismatch: query={tuple(query.shape)}, "
            f"K_min={tuple(K_min.shape)}, K_max={tuple(K_max.shape)}"
        )
    H, D = query.shape
    P = K_min.shape[0]
    assert K_min.shape == (P, H, D), \
        f"K_min should be (P, H, D)=({P},{H},{D}), got {tuple(K_min.shape)}"
    assert query.is_cuda and K_min.is_cuda and K_max.is_cuda

    bound = torch.empty(P, H, dtype=torch.float16, device=query.device)
    BLOCK_D = 64 if D > 32 else 32
    grid = (P, H)
    _page_upper_bound_kernel[grid](
        query.contiguous().to(torch.float16),
        K_min.contiguous().to(torch.float16),
        K_max.contiguous().to(torch.float16),
        bound,
        P=P, H=H, D=D, BLOCK_D=BLOCK_D,
    )
    return bound
