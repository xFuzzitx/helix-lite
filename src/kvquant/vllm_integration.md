# vLLM v1 integration for KVQuant nuq KV cache

## Status (2026-05-11)

* `nuq.py` — pure-PyTorch reference (6/6 tests)
* `triton_kernels.py` — per-head Triton pack/unpack for Keys (per-channel) and Values (per-token), 4/4 parity tests vs reference
* `triton_batched.py` — multi-head batched pack/unpack kernels, 5/5 bit-exact parity tests vs the per-head versions
* `calibration.py` — runs against HF `Qwen2.5-7B-Instruct-1M`, produces `KVScales` saved as `.pt`. Validated nuq4 v3 (outlier_pct=0.02, 16 levels) preserves the needle through every layer; nuq2 v1/v2 failed at layer 27, v3 (outlier_pct=0.05) in progress at time of writing.
* `validate_scales.py` — closed-loop NIAH: hooks every layer, replaces KV with quant→dequant in the forward path, checks the secret password survives
* **`vllm_backend.py` — Phase 1A** ✓: subclasses `FlashAttentionImpl`; injects nuq pack/unpack on K/V before they reach `reshape_and_cache_flash`. The KV pool stays fp16 (no VRAM win), but every cache write goes through the calibrated quant→dequant round-trip. NIAH ✓ on `graelo/Qwen2.5-7B-Instruct-1M-AWQ` + nuq4_v3 at 4K, 32K, 128K, 200K, 256K. 256K = natural fp16-pool ceiling on a 24 GB 3090 with the AWQ weights.
* **Phase 1B — not yet written** (this doc is the plan for the remaining work)

## Phase 1A — what we ship today

The math wrapper (`KVQuantAttentionImpl` in `vllm_backend.py`) is installed via:

```python
from kvquant.vllm_backend import install_kvquant_backend
install_kvquant_backend("scales/qwen2_5_7b_1m_nuq4_v3.pt", device="cuda:0")
# then construct vllm.LLM(...) as usual
```

`install_kvquant_backend` monkey-patches `FlashAttentionBackend.get_impl_cls` to return our subclass. On every forward, the subclass:

1. Looks up the calibrated scales for the current layer (by `layer.layer_name` → regex → index).
2. Runs the **batched** Triton kernels (`triton_batched.py`) to pack K/V to codes + outliers, then unpacks them back to fp16.
3. Hands the reconstructed K/V to `FlashAttentionImpl.forward`, which writes the (already quant-rounded) values to the standard fp16 KV pool.

The cache stays fp16 so the FA kernels need zero changes. Every cache *write* is the quant→dequant'd version, so subsequent reads attend over the same numbers a real compact-pool implementation would produce — Phase 1A is the math-correctness anchor before we touch the allocator.

## What we ship in the CLI today

`helix-cli --doc <file> --nuq-scales <scales.pt> "<question>"` wires
Phase 1A into the queryable model. Recommended scales after the
2026-05-11 autoloop: `scales/mixed_nuq2v3_nuq4v3_cut16.pt`
(16 shallow layers nuq2 + 12 deep layers nuq4, 6.29× weighted
nominal compression, NIAH ✓ at 4K/32K/128K/256K).

## Phase 1B — the path to true 1M

vLLM 0.20.1 v1 stores KV in paged blocks (typically 16 tokens per block) accessed through a `BlockTable`. The relevant entry points (mapped 2026-05-11 against vLLM 0.20.1):

### Read/write path
`vllm.v1.attention.backends.flash_attn.FlashAttentionImpl.forward()` (line 682) calls into `reshape_and_cache_flash` (cache write) and `flash_attn_varlen_func` (cache read + attention). Both kernels expect contiguous fp16/bf16 blocks of shape `(num_blocks, block_size, num_kv_heads, head_size)`.

### Allocator
`vllm.v1.core.kv_cache_manager.KVCacheManager` allocates these blocks at startup based on `gpu_memory_utilization` and `max_model_len`. The shape comes from `FlashAttentionBackend.get_kv_cache_shape()` (line 138). For FA: `(2, num_blocks, block_size, num_kv_heads, head_size)`.

### Where compact storage plugs in
1. **Custom backend** subclassing `AttentionBackend` (not `FlashAttentionBackend` — we need to register at the enum level via `vllm.v1.attention.backends.registry.register_backend(AttentionBackendEnum.CUSTOM, ...)`).
2. **`get_kv_cache_shape`** returning the compact byte count. For nuq4 the natural compaction is `(2, num_blocks, block_size, num_kv_heads, head_size // 2)` of uint8 — two 4-bit codes per byte. For nuq2 it's `head_size // 4`. Outliers ride along in a sibling tensor — either appended to the same buffer (logical offset) or allocated separately and tracked alongside the block table.
3. **Custom `AttentionImpl`**:
   * `__init__`: load `KVScales` from disk, move to GPU, capture per-layer-id.
   * `forward` write side: replace `reshape_and_cache_flash` with a Triton kernel that packs each (block_size, num_kv_heads, head_size) slot into the compact layout and stores outliers to the sibling buffer.
   * `forward` read side: replace `flash_attn_varlen_func` with a *fused-dequant attention kernel* that walks the block table, unpacks codes on-the-fly into a per-warp register buffer, applies the outlier patches, and runs FA's online-softmax dot product against the query. Materialising the full dequantised cache into a temp fp16 buffer would defeat the VRAM saving on long-context decode.

### Cost estimate
* **~3 days** to wire `get_kv_cache_shape` + Triton pack-on-write + naive dequant-then-FA on read (no VRAM win on decode — full cache rebuilt per step).
* **~1 week** for the fused-dequant attention kernel that walks blocks and never materialises the cache (this is where the 4×/8× nominal compression turns into actual VRAM savings).
* **~3 days** of long-context calibration: nuq4 v3 was calibrated at seqlen=4096; extrapolating to 1M via the tail hypothesis is unvalidated. A 32K-128K calibration pass with the same dataset is the safer recipe before declaring "1M attention".
* **~1 week** of NIAH/RULER/BABILong at increasing contexts to catch layer-specific quality drops as the V-scales extrapolate further out.

### Why Phase 1B is gated on Phase 1A passing

Phase 1A proves the **math** integrates (correct scales × correct kernels × correct layer routing). Phase 1B is purely a **memory-layout** engineering problem after that. Running Phase 1B without Phase 1A is how you ship a 256K → 1M jump and discover three weeks in that the V-scale gather has been wrong for layers 14-27 the whole time.

### Why a "halfway Phase 1B" doesn't work

Any compact-pool implementation that **materialises a fp16 staging buffer** for `flash_attn_varlen_func` to read pays peak VRAM = compact pool + staging. At 1M decode the staging needs to span the whole sequence (FA reads all blocks), so the peak is the same as the original fp16 pool — *no net win*. The win only appears when the dequant happens **inside** the attention kernel as it walks blocks, which is the fused-kernel work in step 2 of the cost estimate above.

For prefill-only / chunked-prefill workflows, a "compact pool + per-chunk staging" intermediate would work because each chunk's referenced block set is small. But for the user-facing "load 1M doc, ask one question" path the bottleneck is the decode step. So Phase 1B.1 (correctness anchor, no VRAM win) and Phase 1B.2 (fused kernel, real win) are stages of the same project — there's no shippable intermediate that delivers measurable VRAM savings at 1M without the fused kernel.

## Why nuq2 calibration is its own task

KVQuant's published numbers (Hooper et al. 2024) show nuq2 holds quality on Llama-7B with Fisher-information-weighted k-means. Our calibration uses uniform-weight k-means (a simplification noted in `calibration.py`). The empirical result on Qwen2.5-7B-Instruct-1M:

| version | num_bits | outlier_pct | layer-27 V mean err | needle |
|---|---|---|---|---|
| v1 nuq2 | 2 | 0.005 | 1.40 | ✗ |
| v2 nuq2 | 2 | 0.020 | 0.97 | ✗ |
| v3 nuq4 | 4 | 0.020 | 0.18 | ✓ |
| v3 nuq2 | 2 | 0.050 | (in progress) | (in progress) |

If nuq2 v3 still misses the needle, the next levers are:
- **Higher outlier_pct** (0.08 → 0.10) — trades compression for fidelity at deep-layer Values.
- **Mixed-bit** — deep layers (20-27) at nuq4, shallow at nuq2; the KVQuant paper suggests this as a production recipe.
- **Fisher weighting** — most invasive, requires gradient capture during calibration. Defer until both knobs above are exhausted.
