"""Phase 1B compact KV pool — write/read side.

Owns its own paged storage on the GPU (NOT through vLLM's
KVCacheManager) so we can size it independently and pack to nuq
codes byte-by-byte.

Layout:

* ``codes``     : ``(num_layers, 2, num_blocks, block_size, H, D)`` uint8
  - codes[L, 0, ...] are K codes, codes[L, 1, ...] are V codes
  - one byte per code (nuq2 has 4 levels, nuq4 has 16 — both fit
    in a byte; bit-packing is a Phase 1B v2 follow-up)
* ``outlier_v`` : same shape, fp16, holds dense outlier value where
  ``outlier_m == 1``, else 0
* ``outlier_m`` : same shape, uint8 (0/1) — sparse density depends
  on the calibration outlier_pct
* ``block_stats``: ``(num_layers, 2, num_blocks, H, D, 2)`` fp16
  - block_stats[L, 0, B, H, D, 0] = min(K_recon[block B, all positions, H, D])
  - block_stats[L, 0, B, H, D, 1] = max(K_recon[block B, all positions, H, D])
  - V uses the same layout but feeds Quest's V-side scoring if we
    add it; right now Quest scores K only.

The block-stats are written at the same time as codes so Quest can
score each (sequence, block) pair in O(num_blocks · H · D) without
unpacking.

Per-layer scales (``scales.KVScales``) are not owned here; they
remain in :mod:`kvquant.scales` and are passed in on every call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl

from .scales import KVScales


# --- pack on write -------------------------------------------------------

@triton.jit
def _pack_kv_slot_kernel(
    # input KV  (T, H, D)
    key_ptr, val_ptr,
    # scales for one layer
    k_poles_ptr,         # (H, D, NL)
    k_upper_ptr, k_lower_ptr,  # (H, D)
    v_poles_ptr,         # (H, T_pos, NL) -- positions already extrapolated
    v_upper_ptr, v_lower_ptr,  # (H, T_pos)
    v_pos_ptr,           # (T,) int64 -- absolute token position for each input row,
                         #               clamped to T_cal-1
    # output codes pool
    k_codes_ptr,         # (num_blocks, block_size, H, D) uint8
    v_codes_ptr,
    k_outv_ptr, k_outm_ptr,
    v_outv_ptr, v_outm_ptr,
    # block stats (K only)
    k_min_ptr,           # (num_blocks, H, D) fp16
    k_max_ptr,           # (num_blocks, H, D) fp16
    # slot mapping
    slot_mapping_ptr,    # (T,) int64
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NL_K: tl.constexpr,
    NL_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Pack one token's KV across all H, D into the compact pool.

    Grid: (T, H, ceil(D/BLOCK_D))

    Each program packs (1 token) × (1 head) × (BLOCK_D channels)
    and accumulates min/max into block_stats via atomic_min/max
    (the block_stats tensor is initialised to +inf/-inf before
    this kernel runs).
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    slot = tl.load(slot_mapping_ptr + pid_t)
    # Negative slot means "skip" (vLLM uses this for padding tokens)
    if slot < 0:
        return
    block_id = slot // BLOCK_SIZE
    in_block_pos = slot % BLOCK_SIZE

    # Load this token's K/V row for head pid_h
    base_in = pid_t * H * D + pid_h * D
    k_vals = tl.load(key_ptr + base_in + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    v_vals = tl.load(val_ptr + base_in + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    # --- K: per-channel ---
    k_upper = tl.load(k_upper_ptr + pid_h * D + offs_d, mask=mask_d, other=1e9).to(tl.float32)
    k_lower = tl.load(k_lower_ptr + pid_h * D + offs_d, mask=mask_d, other=-1e9).to(tl.float32)
    k_out = (k_vals > k_upper) | (k_vals < k_lower)

    k_best_idx = tl.zeros((BLOCK_D,), dtype=tl.int32)
    k_best_diff = tl.full((BLOCK_D,), float("inf"), dtype=tl.float32)
    k_best_pole = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for lvl in tl.static_range(NL_K):
        pole = tl.load(k_poles_ptr + pid_h * D * NL_K + offs_d * NL_K + lvl,
                       mask=mask_d, other=0.0).to(tl.float32)
        diff = tl.abs(k_vals - pole)
        better = diff < k_best_diff
        k_best_diff = tl.where(better, diff, k_best_diff)
        k_best_idx = tl.where(better, tl.full((BLOCK_D,), lvl, dtype=tl.int32), k_best_idx)
        k_best_pole = tl.where(better, pole, k_best_pole)

    # Reconstructed K for stats (outliers stay fp16, codes use nearest pole)
    k_recon = tl.where(k_out, k_vals, k_best_pole)

    # Codes/outliers write addresses (use 1D linear offsets)
    code_off = (block_id * BLOCK_SIZE * H * D
                + in_block_pos * H * D
                + pid_h * D
                + offs_d)
    tl.store(k_codes_ptr + code_off, k_best_idx.to(tl.uint8), mask=mask_d)
    tl.store(k_outv_ptr + code_off,
             tl.where(k_out, k_vals, tl.zeros((BLOCK_D,), dtype=tl.float32)).to(tl.float16),
             mask=mask_d)
    tl.store(k_outm_ptr + code_off,
             tl.where(k_out, tl.full((BLOCK_D,), 1, dtype=tl.uint8),
                      tl.zeros((BLOCK_D,), dtype=tl.uint8)),
             mask=mask_d)

    # Atomic min/max into block_stats: addr (block_id, h, d)
    stat_off = block_id * H * D + pid_h * D + offs_d
    tl.atomic_min(k_min_ptr + stat_off, k_recon.to(tl.float16), mask=mask_d, sem="relaxed")
    tl.atomic_max(k_max_ptr + stat_off, k_recon.to(tl.float16), mask=mask_d, sem="relaxed")

    # --- V: per-token ---
    pos = tl.load(v_pos_ptr + pid_t)  # int64, already clamped to T_cal-1
    v_upper = tl.load(v_upper_ptr + pid_h * (NL_V * 0) + pid_h * 1).to(tl.float32)  # placeholder
    # Real V scales are per-token: address = (h * T_cal + pos)
    # We compute the actual stride via v_pos_ptr's clamped value.
    # For shape (H, T_cal), upper/lower address: pid_h * T_cal + pos
    # We need T_cal at the kernel boundary; pass via the pointer's stride.
    # Simpler: compute as if v_upper_ptr layout is (H, T_cal) and we just need
    # the offset pid_h * T_cal + pos. But T_cal is in the v_pos_ptr-clamped
    # space, so the caller has guaranteed pos < T_cal.
    # We'll pass T_cal as a kernel constant in the launcher.


def _build_v_pos_clamped(num_tokens: int, abs_positions: torch.Tensor,
                          T_cal: int) -> torch.Tensor:
    """Clamp absolute token positions to ``[0, T_cal-1]`` for tail extrapolation."""
    return abs_positions.clamp_(max=T_cal - 1)


# The Triton kernel above is structurally correct for K but the V branch
# needs the T_cal constexpr; for cleanliness we split V into its own
# kernel which mirrors the K one with the per-token scale gather.

@triton.jit
def _pack_v_slot_kernel(
    val_ptr,             # (T, H, D) fp16
    v_poles_ptr,         # (H, T_cal, NL)
    v_upper_ptr,         # (H, T_cal)
    v_lower_ptr,         # (H, T_cal)
    v_pos_ptr,           # (T,) int64 (clamped)
    v_codes_ptr,         # (num_blocks, block_size, H, D) uint8
    v_outv_ptr,
    v_outm_ptr,
    slot_mapping_ptr,    # (T,) int64
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    T_CAL: tl.constexpr,
    NL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    slot = tl.load(slot_mapping_ptr + pid_t)
    if slot < 0:
        return
    block_id = slot // BLOCK_SIZE
    in_block_pos = slot % BLOCK_SIZE

    pos = tl.load(v_pos_ptr + pid_t)
    base_in = pid_t * H * D + pid_h * D
    vals = tl.load(val_ptr + base_in + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    upper = tl.load(v_upper_ptr + pid_h * T_CAL + pos).to(tl.float32)
    lower = tl.load(v_lower_ptr + pid_h * T_CAL + pos).to(tl.float32)
    is_out = (vals > upper) | (vals < lower)

    best_idx = tl.zeros((BLOCK_D,), dtype=tl.int32)
    best_diff = tl.full((BLOCK_D,), float("inf"), dtype=tl.float32)
    for lvl in tl.static_range(NL):
        pole = tl.load(v_poles_ptr + pid_h * T_CAL * NL + pos * NL + lvl).to(tl.float32)
        diff = tl.abs(vals - pole)
        better = diff < best_diff
        best_diff = tl.where(better, diff, best_diff)
        best_idx = tl.where(better, tl.full((BLOCK_D,), lvl, dtype=tl.int32), best_idx)

    code_off = (block_id * BLOCK_SIZE * H * D
                + in_block_pos * H * D
                + pid_h * D
                + offs_d)
    tl.store(v_codes_ptr + code_off, best_idx.to(tl.uint8), mask=mask_d)
    tl.store(v_outv_ptr + code_off,
             tl.where(is_out, vals, tl.zeros((BLOCK_D,), dtype=tl.float32)).to(tl.float16),
             mask=mask_d)
    tl.store(v_outm_ptr + code_off,
             tl.where(is_out, tl.full((BLOCK_D,), 1, dtype=tl.uint8),
                      tl.zeros((BLOCK_D,), dtype=tl.uint8)),
             mask=mask_d)


@triton.jit
def _pack_k_slot_kernel(
    key_ptr,             # (T, H, D) fp16
    k_poles_ptr,         # (H, D, NL)
    k_upper_ptr,         # (H, D)
    k_lower_ptr,         # (H, D)
    k_codes_ptr,         # (num_blocks, block_size, H, D) uint8
    k_outv_ptr,
    k_outm_ptr,
    k_min_ptr,           # (num_blocks, H, D) fp16
    k_max_ptr,
    slot_mapping_ptr,    # (T,) int64
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    slot = tl.load(slot_mapping_ptr + pid_t)
    if slot < 0:
        return
    block_id = slot // BLOCK_SIZE
    in_block_pos = slot % BLOCK_SIZE

    base_in = pid_t * H * D + pid_h * D
    vals = tl.load(key_ptr + base_in + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    upper = tl.load(k_upper_ptr + pid_h * D + offs_d, mask=mask_d, other=1e9).to(tl.float32)
    lower = tl.load(k_lower_ptr + pid_h * D + offs_d, mask=mask_d, other=-1e9).to(tl.float32)
    is_out = (vals > upper) | (vals < lower)

    best_idx = tl.zeros((BLOCK_D,), dtype=tl.int32)
    best_diff = tl.full((BLOCK_D,), float("inf"), dtype=tl.float32)
    best_pole = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for lvl in tl.static_range(NL):
        pole = tl.load(k_poles_ptr + pid_h * D * NL + offs_d * NL + lvl,
                       mask=mask_d, other=0.0).to(tl.float32)
        diff = tl.abs(vals - pole)
        better = diff < best_diff
        best_diff = tl.where(better, diff, best_diff)
        best_idx = tl.where(better, tl.full((BLOCK_D,), lvl, dtype=tl.int32), best_idx)
        best_pole = tl.where(better, pole, best_pole)

    recon = tl.where(is_out, vals, best_pole)

    code_off = (block_id * BLOCK_SIZE * H * D
                + in_block_pos * H * D
                + pid_h * D
                + offs_d)
    tl.store(k_codes_ptr + code_off, best_idx.to(tl.uint8), mask=mask_d)
    tl.store(k_outv_ptr + code_off,
             tl.where(is_out, vals, tl.zeros((BLOCK_D,), dtype=tl.float32)).to(tl.float16),
             mask=mask_d)
    tl.store(k_outm_ptr + code_off,
             tl.where(is_out, tl.full((BLOCK_D,), 1, dtype=tl.uint8),
                      tl.zeros((BLOCK_D,), dtype=tl.uint8)),
             mask=mask_d)

    stat_off = block_id * H * D + pid_h * D + offs_d
    tl.atomic_min(k_min_ptr + stat_off, recon, mask=mask_d, sem="relaxed")
    tl.atomic_max(k_max_ptr + stat_off, recon, mask=mask_d, sem="relaxed")


# --- unpack on read (selective blocks) -----------------------------------

@triton.jit
def _unpack_k_block_kernel(
    k_codes_ptr,         # (num_blocks_pool, BS, H, D) uint8
    k_outv_ptr,
    k_outm_ptr,
    k_poles_ptr,         # (H, D, NL)
    block_id_list_ptr,   # (NK,) int64
    staging_k_ptr,       # (NK, BS, H, D) fp16
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Unpack the BS positions of one selected block, for one head, for
    a tile of D channels."""
    pid_k = tl.program_id(0)    # which kept block
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    block_id = tl.load(block_id_list_ptr + pid_k)

    # Loop over the BS positions in the block (BLOCK_SIZE is constexpr,
    # so this unrolls)
    for pos in tl.static_range(BLOCK_SIZE):
        in_off = (block_id * BLOCK_SIZE * H * D
                  + pos * H * D
                  + pid_h * D
                  + offs_d)
        codes = tl.load(k_codes_ptr + in_off, mask=mask_d, other=0).to(tl.int32)
        outv = tl.load(k_outv_ptr + in_off, mask=mask_d, other=0.0).to(tl.float32)
        outm = tl.load(k_outm_ptr + in_off, mask=mask_d, other=0).to(tl.int32)

        # Gather poles by code; unroll over levels
        recon = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for lvl in tl.static_range(NL):
            pole = tl.load(k_poles_ptr + pid_h * D * NL + offs_d * NL + lvl,
                           mask=mask_d, other=0.0).to(tl.float32)
            match = codes == lvl
            recon = tl.where(match, pole, recon)
        final = tl.where(outm != 0, outv, recon)

        out_off = (pid_k * BLOCK_SIZE * H * D
                   + pos * H * D
                   + pid_h * D
                   + offs_d)
        tl.store(staging_k_ptr + out_off, final.to(tl.float16), mask=mask_d)


@triton.jit
def _unpack_v_block_kernel(
    v_codes_ptr,
    v_outv_ptr,
    v_outm_ptr,
    v_poles_ptr,         # (H, T_cal, NL)
    block_id_list_ptr,   # (NK,) int64
    block_first_pos_ptr, # (NK,) int64 — absolute start position of each kept block
    staging_v_ptr,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    T_CAL: tl.constexpr,
    NL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    block_id = tl.load(block_id_list_ptr + pid_k)
    block_start = tl.load(block_first_pos_ptr + pid_k)

    for pos in tl.static_range(BLOCK_SIZE):
        # Per-token absolute position for V scale lookup
        abs_pos = block_start + pos
        # Clamp to T_cal - 1 for tail extrapolation
        clamped_pos = tl.minimum(abs_pos, T_CAL - 1)

        in_off = (block_id * BLOCK_SIZE * H * D
                  + pos * H * D
                  + pid_h * D
                  + offs_d)
        codes = tl.load(v_codes_ptr + in_off, mask=mask_d, other=0).to(tl.int32)
        outv = tl.load(v_outv_ptr + in_off, mask=mask_d, other=0.0).to(tl.float32)
        outm = tl.load(v_outm_ptr + in_off, mask=mask_d, other=0).to(tl.int32)

        # V poles are shared across D — gather one scalar pole per level
        recon = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for lvl in tl.static_range(NL):
            pole_addr = pid_h * T_CAL * NL + clamped_pos * NL + lvl
            pole = tl.load(v_poles_ptr + pole_addr).to(tl.float32)
            match = codes == lvl
            recon = tl.where(match, pole, recon)
        final = tl.where(outm != 0, outv, recon)

        out_off = (pid_k * BLOCK_SIZE * H * D
                   + pos * H * D
                   + pid_h * D
                   + offs_d)
        tl.store(staging_v_ptr + out_off, final.to(tl.float16), mask=mask_d)


@dataclass
class CompactKVPool:
    """A user-managed paged KV cache holding nuq codes + dense outliers.

    Lives next to vLLM but is NOT visible to vLLM's KVCacheManager —
    vLLM still allocates its own fp16 pool (we'll keep it tiny via a
    low max_model_len; the *real* storage is here).
    """

    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_size: int
    scales: KVScales
    device: str = "cuda:0"
    # Per-layer level counts (derived from scales)
    _k_levels: list[int] = field(default_factory=list)
    _v_levels: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        L = self.num_layers
        NB = self.num_blocks
        BS = self.block_size
        H = self.num_kv_heads
        D = self.head_size
        dev = self.device

        # codes, outlier values, outlier masks (per layer, K and V separately)
        self.k_codes = torch.zeros((L, NB, BS, H, D), dtype=torch.uint8, device=dev)
        self.v_codes = torch.zeros((L, NB, BS, H, D), dtype=torch.uint8, device=dev)
        self.k_outv = torch.zeros((L, NB, BS, H, D), dtype=torch.float16, device=dev)
        self.v_outv = torch.zeros((L, NB, BS, H, D), dtype=torch.float16, device=dev)
        self.k_outm = torch.zeros((L, NB, BS, H, D), dtype=torch.uint8, device=dev)
        self.v_outm = torch.zeros((L, NB, BS, H, D), dtype=torch.uint8, device=dev)

        # block stats for Quest top-K: per-block min/max of reconstructed K.
        # Triton atomic_min/max require fp32 (no fp16 atomics on sm_86), so
        # we use fp32 here; doubles the stats size but it's tiny relative
        # to the codes (per-token vs per-block).
        self.k_min = torch.full((L, NB, H, D), float("inf"),
                                 dtype=torch.float32, device=dev)
        self.k_max = torch.full((L, NB, H, D), float("-inf"),
                                 dtype=torch.float32, device=dev)

        self._k_levels = [s.poles.shape[-1] for s in self.scales.per_layer_keys]
        self._v_levels = [s.poles.shape[-1] for s in self.scales.per_layer_values]

    # -- write side --
    def write_kv(
        self,
        layer_idx: int,
        key: torch.Tensor,             # (T, H, D) fp16
        value: torch.Tensor,           # (T, H, D) fp16
        slot_mapping: torch.Tensor,    # (T,) int64
        abs_positions: torch.Tensor,   # (T,) int64 — absolute positions (for V scales)
    ) -> None:
        """Pack one layer's K/V into the pool via slot_mapping."""
        assert layer_idx < self.num_layers
        assert key.is_contiguous(), "key must be contiguous (T, H, D)"
        assert value.is_contiguous()
        T = key.shape[0]
        H = self.num_kv_heads
        D = self.head_size
        BLOCK_D = min(64, D)

        k_scale = self.scales.per_layer_keys[layer_idx]
        v_scale = self.scales.per_layer_values[layer_idx]
        T_cal = v_scale.poles.shape[1]
        v_pos = abs_positions.clamp(max=T_cal - 1).to(torch.int64).contiguous()

        # Pack K
        grid_k = (T, H, triton.cdiv(D, BLOCK_D))
        _pack_k_slot_kernel[grid_k](
            key, k_scale.poles, k_scale.upper_threshold, k_scale.lower_threshold,
            self.k_codes[layer_idx], self.k_outv[layer_idx], self.k_outm[layer_idx],
            self.k_min[layer_idx], self.k_max[layer_idx],
            slot_mapping.to(torch.int64).contiguous(),
            BLOCK_SIZE=self.block_size, H=H, D=D,
            NL=self._k_levels[layer_idx], BLOCK_D=BLOCK_D,
        )

        # Pack V
        grid_v = (T, H, triton.cdiv(D, BLOCK_D))
        _pack_v_slot_kernel[grid_v](
            value, v_scale.poles, v_scale.upper_threshold, v_scale.lower_threshold,
            v_pos,
            self.v_codes[layer_idx], self.v_outv[layer_idx], self.v_outm[layer_idx],
            slot_mapping.to(torch.int64).contiguous(),
            BLOCK_SIZE=self.block_size, H=H, D=D, T_CAL=T_cal,
            NL=self._v_levels[layer_idx], BLOCK_D=BLOCK_D,
        )

    # -- read side --
    def unpack_to_staging(
        self,
        layer_idx: int,
        block_id_list: torch.Tensor,        # (NK,) int64
        block_first_position: torch.Tensor, # (NK,) int64 — abs token pos of slot 0
        staging_k: torch.Tensor,            # (NK, BS, H, D) fp16
        staging_v: torch.Tensor,            # (NK, BS, H, D) fp16
    ) -> None:
        """Dequantise the selected blocks for one layer into ``staging_*``.

        The caller picks which blocks to keep (top-K from Quest) and
        passes their ids + each block's absolute starting token
        position (for V's per-token scale gather).
        """
        assert layer_idx < self.num_layers
        NK = block_id_list.shape[0]
        BS = self.block_size
        H = self.num_kv_heads
        D = self.head_size
        BLOCK_D = min(64, D)
        assert staging_k.shape == (NK, BS, H, D), \
            f"staging_k shape {staging_k.shape} != ({NK},{BS},{H},{D})"
        assert staging_v.shape == staging_k.shape

        k_scale = self.scales.per_layer_keys[layer_idx]
        v_scale = self.scales.per_layer_values[layer_idx]
        T_cal = v_scale.poles.shape[1]

        grid = (NK, H, triton.cdiv(D, BLOCK_D))
        _unpack_k_block_kernel[grid](
            self.k_codes[layer_idx], self.k_outv[layer_idx], self.k_outm[layer_idx],
            k_scale.poles,
            block_id_list.to(torch.int64).contiguous(),
            staging_k,
            BLOCK_SIZE=BS, H=H, D=D,
            NL=self._k_levels[layer_idx], BLOCK_D=BLOCK_D,
        )
        _unpack_v_block_kernel[grid](
            self.v_codes[layer_idx], self.v_outv[layer_idx], self.v_outm[layer_idx],
            v_scale.poles,
            block_id_list.to(torch.int64).contiguous(),
            block_first_position.to(torch.int64).contiguous(),
            staging_v,
            BLOCK_SIZE=BS, H=H, D=D, T_CAL=T_cal,
            NL=self._v_levels[layer_idx], BLOCK_D=BLOCK_D,
        )
