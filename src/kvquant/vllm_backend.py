"""KVQuant vLLM v1 integration — Phase 1A: math-wrapper.

Subclasses ``vllm.v1.attention.backends.flash_attn.FlashAttentionImpl``
and injects nuq quant→dequant on K/V *before* they're written into
the (still fp16) paged KV cache. The pool keeps its native fp16
layout — Phase 1A buys NO VRAM win, it only proves the calibrated
scales + Triton pack/unpack survive vLLM's prefill+decode pipeline
end-to-end on a real Qwen2.5-7B model.

Once Phase 1A passes a NIAH at 32 K (the needle survives the
round-trip on every attention call), Phase 1B can subclass the
backend to actually compact the KV pool to int4/int2 storage and
unlock the VRAM savings that get us to 1M tokens.

Usage::

    # Before LLM(...)
    from kvquant.vllm_backend import install_kvquant_backend
    install_kvquant_backend("scales/qwen2_5_7b_1m_nuq4_v3.pt", device="cuda:0")

    from vllm import LLM
    llm = LLM(model="graelo/Qwen2.5-7B-Instruct-1M-AWQ", ...)
    # Every Attention layer's forward will now apply nuq quant→dequant
    # on its K/V before storing them in the KV cache.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)

from .scales import KVScales, PerChannelScale, PerTokenScale
from .triton_kernels import (
    pack_keys_per_channel,
    pack_values_per_token,
    unpack_keys_per_channel,
    unpack_values_per_token,
)

if TYPE_CHECKING:
    pass


# Module-level state. Set by install_kvquant_backend() before vLLM's
# LLM() constructs the AttentionImpl instances, so each impl can find
# its layer scales at forward time without threading the scales through
# the vLLM config plumbing (which has no public hook for arbitrary kwargs).
_KV_SCALES: KVScales | None = None
_DEVICE: str = "cuda:0"
_AUTO_LAYER_INDEX: dict[str, int] = {}
_WARNED_LAYER_NAMES: set[str] = set()


def install_kvquant_backend(scales_path: str, device: str = "cuda:0") -> None:
    """Load calibrated scales globally and monkey-patch the FA backend
    so every attention layer routes through :class:`KVQuantAttentionImpl`.

    Must be called BEFORE :class:`vllm.LLM` (or any code path that
    triggers attention-backend selection).
    """
    global _KV_SCALES, _DEVICE
    _DEVICE = device
    _KV_SCALES = KVScales.load(scales_path, map_location=device).to(device)
    print(
        f"[kvquant.vllm_backend] loaded {scales_path}: "
        f"{_KV_SCALES.num_layers} layers @ nuq{_KV_SCALES.num_bits}, "
        f"first_few_fp16={_KV_SCALES.first_few_fp16}, device={device}"
    )

    # Monkey-patch the FA backend's get_impl_cls to return our subclass.
    # The selector resolves FlashAttentionBackend → get_impl_cls() →
    # FlashAttentionImpl. We replace that lookup with our subclass.
    FlashAttentionBackend.get_impl_cls = staticmethod(  # type: ignore[method-assign]
        lambda: KVQuantAttentionImpl
    )
    print(
        "[kvquant.vllm_backend] patched FlashAttentionBackend.get_impl_cls "
        "→ KVQuantAttentionImpl"
    )


def _resolve_layer_index(layer: torch.nn.Module) -> int | None:
    """Map a vLLM Attention module to a transformer layer index.

    vLLM v1 stores ``layer_name = prefix`` on each :class:`Attention`,
    typically ``"model.layers.<N>.self_attn.attn"``. We extract <N>.
    For models that don't follow the convention we fall back to a
    per-name auto-counter (only stable if backend was just installed
    and all layers are constructed in source order — fine for inference).
    """
    name = getattr(layer, "layer_name", None) or ""
    if not name:
        return None
    m = re.search(r"layers?\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    # Auto-assign on first sight
    if name not in _AUTO_LAYER_INDEX:
        _AUTO_LAYER_INDEX[name] = len(_AUTO_LAYER_INDEX)
        if name not in _WARNED_LAYER_NAMES:
            print(f"[kvquant.vllm_backend] auto-indexing layer {name!r}")
            _WARNED_LAYER_NAMES.add(name)
    return _AUTO_LAYER_INDEX[name]


def _quant_dequant_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    k_scale: PerChannelScale,
    v_scale: PerTokenScale,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply nuq pack→unpack to ``(key, value)``.

    Shapes:
        key, value: ``(T, H, D)``  fp16 or bf16
        k_scale.poles: ``(H, D, L)``  fp16
        v_scale.poles: ``(H, T_cal, L)``  fp16 — extrapolated by tail

    Returns reconstructed ``(key_q, value_q)`` in the original dtype.
    The KV cache will store these reconstructed values; from this point
    onward the math is identical to a "real" compact-pool implementation
    that stores codes and unpacks on read.
    """
    orig_dtype = key.dtype
    if key.numel() == 0:
        return key, value

    T, H, D = key.shape
    assert v_scale.poles.dim() == 3, f"expected (H,T,L), got {v_scale.poles.shape}"
    H_kv = k_scale.poles.shape[0]
    assert H == H_kv, f"key heads {H} != scale heads {H_kv}"

    # Cast to fp16 for the kernels (they assert fp16).
    k_f16 = key.to(torch.float16)
    v_f16 = value.to(torch.float16)

    # Per-token V scales: tail-extrapolate by clamping the time index
    # to the last calibrated slot. For Phase 1A this is a single broadcast
    # of the last calibrated scale across all positions — the Values'
    # outlier statistics stabilise after a few thousand tokens.
    T_cal = v_scale.poles.shape[1]
    t_idx = torch.arange(T, device=value.device).clamp_(max=T_cal - 1)

    key_q = torch.empty_like(k_f16)
    value_q = torch.empty_like(v_f16)

    for h in range(H):
        # K: per-channel along head_dim
        k_codes, k_outv, k_outm = pack_keys_per_channel(
            k_f16[:, h, :].contiguous(),
            k_scale.poles[h].contiguous(),
            k_scale.upper_threshold[h].contiguous(),
            k_scale.lower_threshold[h].contiguous(),
        )
        key_q[:, h, :] = unpack_keys_per_channel(
            k_codes, k_outv, k_outm, k_scale.poles[h].contiguous(),
        )

        # V: per-token (tail-extrapolated)
        v_codes, v_outv, v_outm = pack_values_per_token(
            v_f16[:, h, :].contiguous(),
            v_scale.poles[h, t_idx].contiguous(),
            v_scale.upper_threshold[h, t_idx].contiguous(),
            v_scale.lower_threshold[h, t_idx].contiguous(),
        )
        value_q[:, h, :] = unpack_values_per_token(
            v_codes, v_outv, v_outm, v_scale.poles[h, t_idx].contiguous(),
        )

    return key_q.to(orig_dtype), value_q.to(orig_dtype)


class KVQuantAttentionImpl(FlashAttentionImpl):
    """FA impl that round-trips K/V through nuq quant→dequant before storage.

    Phase 1A only. Phase 1B will replace the pool layout itself.
    """

    can_return_lse_for_decode: bool = True

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _KV_SCALES is None or attn_metadata is None or key.numel() == 0:
            # Profiling run / not installed / empty step — pass through.
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        idx = _resolve_layer_index(layer)
        if idx is None or idx >= _KV_SCALES.num_layers:
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        # Only quantise the tokens that will actually be cached this step.
        # FA caches the first num_actual_tokens; anything past is padding.
        num_actual = attn_metadata.num_actual_tokens
        if num_actual == 0:
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        k_scale = _KV_SCALES.per_layer_keys[idx]
        v_scale = _KV_SCALES.per_layer_values[idx]

        with torch.no_grad():
            k_part = key[:num_actual]
            v_part = value[:num_actual]
            k_q, v_q = _quant_dequant_kv(k_part, v_part, k_scale, v_scale)
            # Write back in-place; tokens past num_actual are padding that
            # FA ignores via slot_mapping, but cloning the unaltered tail
            # is cheap so we just overwrite the prefix.
            key = torch.cat([k_q, key[num_actual:]], dim=0) if key.shape[0] > num_actual else k_q
            value = torch.cat([v_q, value[num_actual:]], dim=0) if value.shape[0] > num_actual else v_q

        return super().forward(
            layer, query, key, value, kv_cache, attn_metadata,
            output, output_scale, output_block_scale,
        )
