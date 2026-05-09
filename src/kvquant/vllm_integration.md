# vLLM v1 integration plan for nuq4 KV cache

## Status (2026-05-09)

* `nuq.py` — pure-PyTorch reference (6/6 tests)
* `triton_kernels.py` — Triton pack/unpack for Keys (per-channel) and Values (per-token), 4/4 parity tests vs reference
* `calibration.py` — runs against HF `Qwen2.5-7B-Instruct-1M`, produces `KVScales` saved as `.pt`. v3 nuq4 calibration validated: NIAH needle preserved at 1.5K tokens after quantize→dequantize on every layer's KV
* `validate_scales.py` — closed-loop NIAH: hooks every layer, replaces KV with quant→dequant in the forward path, checks the secret password survives
* `vllm_backend.py` — **NOT YET WRITTEN** (this doc is the plan)

## What's not yet done

The code above proves the math, the calibration, and the kernels all line up: a Qwen2 model can run with quant→dequant injected after every `k_proj` and `v_proj` and still find the needle (`'BANANA-7392'`). What remains is plumbing it into vLLM v1 in a way that **actually saves VRAM**, not just simulates the math.

## The integration in vLLM v1

vLLM 0.20.1 v1 stores KV in paged blocks (typically 16 tokens per block) accessed through a `BlockTable`. The relevant entry points:

### Read path
`vllm.v1.attention.backends.flash_attn.FlashAttentionImpl.forward()` (~line 682) calls into `reshape_and_cache_flash` (cache write) and `flash_attn_with_kvcache` (cache read + attention). Both kernels assume contiguous fp16/bf16 blocks of shape `(num_blocks, block_size, num_kv_heads, head_dim)`.

### Allocator
`vllm.v1.core.kv_cache_manager.KVCacheManager` allocates these blocks at startup based on `gpu_memory_utilization` and `max_model_len`. The shape and dtype come from `FlashAttentionBackend.get_kv_cache_shape()`.

### Where nuq4 has to plug in
1. **Custom backend class** subclassing `FlashAttentionBackend` whose `get_kv_cache_shape()` returns *two* tensors: codes (uint8) and outliers (fp16, sparse-ish). For correctness-first this can stay dense.
2. **Custom `AttentionImpl`** subclassing `FlashAttentionImpl`:
   - `__init__`: load `KVScales` from disk, move to GPU.
   - `forward`:
     - Before `reshape_and_cache_flash`: pack the new KV with the calibrated scales into the codes/outlier buffers.
     - Before `flash_attn_with_kvcache`: unpack the relevant blocks back into a temporary fp16 buffer and pass that to FA.
3. **Allocator override**: `get_kv_cache_shape` must report the smaller per-token byte count so vLLM allocates fewer blocks for the same memory budget.

### Cost estimate
* **~2 days** to wire the dense-outlier MVP (no real VRAM win — both fp16 KV and codes coexist).
* **~1 week** to add the custom allocator path so the codes are the only thing kept resident.
* **~1 week** to fuse the dequant into the FA prefix path (so we don't pay a memcpy per attention call).

This is "boring plumbing" that needs careful per-layer-kv-head bookkeeping, not new research, but it's invasive enough that we punt it from PR1b.

## Why we ship now without it

The HELIX vision (PR0..PR5) needs forward progress on:
- PR2 MInference (sparse prefill, Qwen ships the pattern config already)
- PR3 Quest (sparse decode top-K)
- PR5 EM-LLM (the GPU-1 retrieval store)

PR1a (AWQ INT4 weights) already gives us 128K context on a single 3090 with the needle preserved. The PR1b kernels and scales are committed and unit-tested; when we eventually need >256K on one GPU we resume the integration with the math already settled.
