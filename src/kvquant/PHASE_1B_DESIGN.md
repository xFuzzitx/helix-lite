# PR1c Phase 1B — design (2026-05-11)

## The previous wall

A naïve compact-pool-with-fp16-staging approach hits the same peak
VRAM as fp16 baseline at 1M decode: `flash_attn_varlen_func` reads
every block in the KV pool through `block_table`, so the staging
buffer must be sized for the whole sequence. Net peak VRAM during a
forward = compact_pool + staging = fp16_baseline. No win.

A fused-dequant attention kernel removes the staging by walking
blocks inline (read codes + dequant in registers, never materialise
full fp16 cache). That's 1–2 weeks of Triton/CUDA on sm_86 — real,
but multi-day, hard-to-rollback work.

## The new idea — stack Phase 1B on top of PR3 (Quest)

PR3 is already validated: `src/quest/triton_kernels.py` produces a
per-page upper-bound score `q_pos · k_max + q_neg · k_min` and
`src/quest/reference.py:topk_pages_with_sinks` returns the top-K
block ids that the decode query actually needs to attend to. Smoke
on real Qwen layer-14 shows **cos 0.97 vs dense at 25 % KV loaded**.

So instead of materialising fp16 for *all* blocks per decode, we:

1. Compute Quest scores against the compact-stored K (statistics
   over each block: per-channel min/max of codes' reconstructed K).
2. Select the top-K blocks the query actually attends to.
3. Materialise fp16 staging **only for those blocks** (≈ 25 % of
   total).
4. Call vLLM's existing `flash_attn_varlen_func` on the staging,
   with a remapped `block_table` pointing into the smaller buffer.

Numbers at 1M context, Qwen2.5-7B, nuq4 byte-packed:

| component                  | size            |
|---                          |---              |
| compact pool (uint8 codes)  | 14 GB → **7 GB** |
| staging for top-25 %         | 14 GB → **3.5 GB** |
| **peak during forward**     | **10.5 GB**     |
| (vs. fp16 baseline)         | 14 GB           |

At nuq2 mixed-cut16 (6.29× weighted) + Quest 25 %:
peak ≈ (14 / 6.29) + (14 × 0.25) ≈ 2.2 + 3.5 = **5.7 GB at 1M decode**.

With AWQ weights (5.4 GB) + activations (≈ 3 GB), total ≈ 14 GB on
GPU 0 — comfortable headroom under 24 GB. **1M genuinely fits.**

## What still needs writing

### 1. `kvquant.compact_pool.CompactKVPool`

A user-managed shadow pool replacing vLLM's fp16 allocation:

* shape `(num_layers, 2, num_blocks, block_size, num_kv_heads, head_size)` uint8
* outliers in `(num_layers, 2, num_blocks, block_size, num_kv_heads, head_size)` fp16 + uint8 mask
* per-layer K/V scales (already calibrated, mixed-bit cut=16)
* Triton pack kernel using `slot_mapping` (input layout `(T, H, D)` fp16 → output codes via vLLM's slot indexing)

The pool is allocated **alongside** vLLM's fp16 pool (which we'll
keep at a much smaller `max_model_len` so it doesn't dominate VRAM).

### 2. `kvquant.compact_pool.unpack_to_staging`

Given a `block_id_list` (top-K from Quest) + the compact pool +
per-layer scales, produce a packed `(K, num_blocks_kept, block_size, H, D)` fp16
staging buffer.

* one Triton kernel call per layer
* outputs live in a pre-allocated staging pinned-memory region
* `remapped_block_table[seq][slot] = staging_index_for_original_block`

### 3. Quest-driven block selection per forward

Adapt `src/quest/reference.py::topk_pages_with_sinks` to consume the
compact pool's per-block statistics. Each block's per-channel K
min/max is computed at write time and stored next to the codes
(extra ~`(num_blocks × 2 × num_kv_heads × head_size)` fp16 — small
relative to the compact pool itself).

### 4. Custom `KVQuantQuestAttentionBackend(AttentionBackend)`

* `get_kv_cache_shape` returns a tiny shape — we want vLLM to
  allocate a small dummy pool; the *real* storage is the
  CompactKVPool which we own.
* Custom `AttentionImpl.forward` orchestrates Quest top-K + unpack +
  FA on staging.
* Custom `do_kv_cache_update` writes codes to the compact pool.

### 5. Outlier handling

For Phase 1B v1 we **drop outliers** (clamp values outside the
calibrated band). The needle-survival history shows nuq4 + clamp is
acceptable at deep layers; the mixed-cut=16 calibration we already
have absorbs the worst layers at nuq4. If quality regresses, add a
sparse outlier side-buffer in v2.

## Order of operations

1. Pack-on-write Triton kernel for the compact pool (single layer
   first, smoke at random data).
2. Unpack-to-staging Triton kernel (single layer, parity vs
   reference).
3. Compact pool data structure + write side wired into a dummy
   AttentionImpl.
4. Integrate Quest top-K selection.
5. Replace vLLM's KV cache with the compact pool at runtime.
6. NIAH smoke at 4K → 32K → 128K → 256K → 512K → 1M.

Each step is a separate commit. The session ends when the next NIAH
context breaks; we commit the working state and iterate.
