# HELIX-Lite — PR Plan

Each PR is **independent and mergeable on its own**. Together they compose the full stack. Acceptance criteria use the baseline benchmark as reference.

---

## PR0 — Baseline ✅ DONE (2026-05-09)

Run `benchmarks/run_baseline.py` on Qwen2.5-7B-Instruct-1M. Save numbers as the reference all later PRs are compared against.

**Result TP=1** (`benchmarks/results/baseline_20260509-090458.json`):

| ctx     | mem (GB) | KV (GB) | throughput   | needle |
|---------|----------|---------|--------------|--------|
|  4,000  |   21.67  |  0.08   |  2,791 t/s   |   ✓    |
| 32,000  |   22.30  |  0.63   |  3,395 t/s   |   ✓    |
| 128,000 |   FAIL   |  —      |  —           |   —    |

**Result TP=2** (`benchmarks/results/baseline_20260509-090942.json`) — both 3090s, gmu=0.92:

| ctx      | mem/GPU0 (GB) | KV Δ (GB) | throughput   | elapsed | needle |
|----------|---------------|-----------|--------------|---------|--------|
|   4,000  |     24.04     |   ~0      |  2,193 t/s   |  1.6 s  |   ✓    |
|  32,000  |     24.65     |   0.61    |  3,969 t/s   |  7.2 s  |   ✓    |
| 128,000  |     24.65     |   ~0      |  2,726 t/s   | 41.8 s  |   ✓    |

(KV Δ near-zero at 128K means scratch was already pre-allocated; absolute KV is ~6 GB split across both GPUs.)

**Settings** (vLLM 0.20.1, torch 2.11.0+cu130, sm_86):
- `enforce_eager=True` (CUDA graphs segfault on sm_86 + FLASHINFER)
- `gpu_memory_utilization=0.85`
- `attention_backend` = auto (FLASH_ATTN selected)
- `max_model_len=32000`
- env: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**Workarounds applied** :
- `dual_chunk_attention_config` removed from `config.json` (DCA unsupported in vLLM 0.20.1 v1 engine; backed up as `config.json.original`). Cost: model degrades >256K ctx; addressed in PR5.
- `sparse_attention_config.json` renamed to `.disabled` (re-enable in PR2 with proper MInference path).

**128K verdict** :
- Single 3090 (TP=1): VRAM-bound. KV scratch needs 6.84 GiB → leaves <500 MB for forward activations → OOM. Motivation for PR1.
- Both 3090s (TP=2): works, 41.8 s prefill, 2,726 t/s effective. Throughput near-identical to TP=1 at 32K (no NCCL bottleneck visible at this scale, all-reduce is bandwidth-cheap relative to compute).

**Acceptance**:
- [x] Model loads
- [x] Generation works at 4K, 32K, 128K (TP=2)
- [x] Needle-in-haystack passes at 32K and 128K (BANANA-7392 retrieved)
- [x] Memory and throughput recorded for both TP=1 and TP=2
- [ ] 1M and beyond — requires PR1 (KVQuant) and PR3 (Quest sparse decode)

---

## PR1a — AWQ INT4 weights ✅ DONE (2026-05-09)

Quick-win path before vendoring KVQuant. The HELIX vision needs **GPU 1 free** for EM-LLM (PR5), so we needed a way to fit 128K on a single 3090.

**FP8 attempt (rejected)**: `kv_cache_dtype=fp8` and `cortecs/Qwen2.5-7B-Instruct-1M-FP8-Dynamic` both produced garbage outputs (Chinese tokens, broken needle). Root cause: Ampere (sm_86) has no native FP8; vLLM's Marlin FP8 kernels work for memory but degrade correctness on this generation. Documented and skipped.

**AWQ INT4 result** (`graelo/Qwen2.5-7B-Instruct-1M-AWQ`, TP=1, gmu=0.85):

| ctx     | mem (GB) | throughput | elapsed | needle |
|---------|----------|------------|---------|--------|
|   4,000 |  21.73   |  3,272 t/s |  1.1 s  |   ✓    |
|  32,000 |  22.34   |  3,450 t/s |  8.3 s  |   ✓    |
| 128,000 |  22.34   |  1,822 t/s | 62.5 s  |   ✓    |

Weights compressed 4× (14 → 3.5 GB), KV stays fp16, INT4 kernels are native sm_86. Output quality preserved across all context lengths.

**Trade vs TP=2 fp16 baseline**: -33% throughput at 128K (1,822 vs 2,726 t/s). Worth it because GPU 1 is now free for retrieval, episodic store, JEPA decoder.

---

## PR1b — KVQuant nuq2 (2-bit KV cache, the original plan)

**Goal**: 8× compression of KV cache via 2-bit non-uniform quantization with dense+sparse outliers.

**References**:
- Paper: [KVQuant — arXiv:2401.18079](https://arxiv.org/abs/2401.18079)
- Repo: [SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant)

**What to do**:
1. Vendored fork of KVQuant kernels into `src/kvquant/`.
2. Hook into vLLM's KV cache manager (the `cache_kv` and `gather_kv` paths).
3. Calibration step: sample ~16 prompts at varying lengths, compute per-channel scales.
4. Inference: K stored per-channel, V per-token, both at 2-bit nuq with dense outliers.

**Acceptance**:
- [ ] KV memory at 1M context drops from ~33 GB → ~4.2 GB (8×)
- [ ] **1M context now fits on a single 3090** (weights 14 + KV 4 + activations 3 = 21 GB)
- [ ] RULER@32K regresses ≤2pt vs PR0 baseline
- [ ] RULER@128K regresses ≤5pt
- [ ] Throughput at 32K within 90% of baseline

**Risks**:
- vLLM's KV cache layout assumes contiguous fp16; integration may need a custom CacheEngine subclass
- nuq2 sparse-outlier path needs Triton kernels — Triton supports sm_86 but FA3 doesn't, so we stick with FA2

---

## PR2 — MInference Vertical-Slash (sparse prefill)

**Goal**: ~10× prefill speedup at 1M via dynamic sparse attention patterns (A-shape, Vertical-Slash, Block-Sparse) classified per-head offline.

**References**:
- Paper: [MInference — arXiv:2407.02490](https://arxiv.org/abs/2407.02490) (NeurIPS'24)
- Repo: [microsoft/MInference](https://github.com/microsoft/MInference)
- Already integrated in vLLM partially; we leverage existing PRs.

**What to do**:
1. `pip install minference` (HF distributes wheels)
2. Patch the model's forward pass to invoke `MInference.search_pattern` once during warmup, then dispatch per-head sparse Triton kernels at prefill.
3. **Decode unchanged** (MInference is prefill-only).

**Acceptance**:
- [ ] TTFT at 128K drops from baseline by ≥5×
- [ ] TTFT at 1M < 60s (with PR1 active so KV fits)
- [ ] RULER@128K regresses ≤3pt vs PR1
- [ ] No degradation on multi-needle RULER tasks at 32K

**Risks**:
- Pattern classification can mis-classify on out-of-distribution prompts; mitigation: per-head thresholds tuned on RULER

---

## PR3 — Quest (decode-time top-K page selection)

**Goal**: Decode-time KV sparsity. At each decode step, only the top-K most relevant *pages* of KV are loaded into the attention computation. Page = block of ~16 tokens.

**References**:
- Paper: [Quest — arXiv:2406.10774](https://arxiv.org/abs/2406.10774) (ICML'24)
- Repo: [mit-han-lab/Quest](https://github.com/mit-han-lab/Quest)

**What to do**:
1. For each KV page, store min/max keys per head — this is the "criticality bound".
2. At decode, the query computes per-page upper-bound scores against (min, max), then top-K.
3. Only the top-K pages contribute to the actual attention.
4. Budget: K such that loaded pages span ~64K-128K tokens worth of KV (regardless of full context).

**Acceptance**:
- [ ] Decode throughput at 1M ≥ 7× the throughput at 1M without Quest (target from paper)
- [ ] RULER passkey@1M ≥ 90%
- [ ] RULER multi-needle@1M ≥ 70% (Quest may degrade on tasks needing all tokens)
- [ ] Active KV working set during decode ≤ 16 GB (so room for prefix cache)

**Risks**:
- Quest works best with KVQuant *off* (because page-level decisions are made on min/max keys, which are corrupted by aggressive quantization). **May need to keep keys in fp8 instead of 2-bit** — re-check with ablation.

---

## PR4 — StreamingLLM attention sinks

**Goal**: Numerical stability in long context by always retaining the first 4 tokens as "attention sinks", regardless of any other KV eviction strategy.

**References**:
- Paper: [StreamingLLM — arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- Repo: [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm)

**What to do**:
- Trivial: ensure that PR3's top-K page selection always includes page 0.
- ~10 LOC change.

**Acceptance**:
- [ ] No perplexity spike at any context length
- [ ] Generation stays coherent past 1M tokens (subjective sanity check)

---

## PR5 — EM-LLM episodic store on GPU 1 (5M retrieval)

**Goal**: Extend effective context to **5M tokens** via episodic memory + kNN retrieval. **GPU 1** hosts the entire episode store.

**References**:
- Paper: [EM-LLM — arXiv:2407.09450](https://arxiv.org/abs/2407.09450) (ICLR'25)
- Project: [em-llm.github.io](https://em-llm.github.io/)

**What to do**:
1. Stream incoming tokens through the model; segment via Bayesian surprise (KL between consecutive token distributions).
2. Each episode → a fixed-size embedding (mean-pool of last layer hidden states).
3. Episode embeddings + their KV chunks live on GPU 1.
4. At decode, when context overflows GPU 0's hot window: query GPU 1 via kNN retrieval, load top-M episodes' KV back to GPU 0 transiently.
5. Benchmark at 5M tokens (synthetic concatenation + needle).

**Acceptance**:
- [ ] 5M context "works" (model generates coherently, doesn't OOM)
- [ ] Needle-in-haystack@5M ≥ 50% (retrieval-style)
- [ ] BABILong@1M ≥ 60% (multi-hop where retrieval helps)
- [ ] Latency overhead ≤ 20% vs PR4 at the same effective context

**Risks**:
- PCIe gen4 (~32 GB/s) is the bottleneck for GPU 0 ↔ GPU 1 transfers
- Needs careful pipelining: prefetch likely-needed episodes while decoding

---

## Final — Evaluation + open-source release

**Goal**: full eval suite, documented repo.

**Suite obligatoire**:
- RULER avg at 32K, 128K, 1M, 5M
- MRCR-V2 4-needle and 8-needle at 128K, 1M
- BABILong at 1M, 5M
- NIAH at 1M, 5M
- NoLiMa (no literal match) at 128K, 1M

**Comparisons**:
- vs Qwen2.5-7B-1M dense baseline (PR0)
- vs Qwen2.5-7B-1M with naive RAG fallback at 5M
- vs Llama-3.1-8B-128K (different base)

**Repo deliverable**:
- README with benchmarks
- Reproducible install scripts (already done in `setup/`)
- Each PR isolatable so users can pick and choose
- Apache 2.0 or MIT license
- Submit to vLLM upstream as KV-cache optimization PRs where applicable

---

## Timeline (réaliste, solo, 2× 3090)

| Step | Durée | Compute | Cumulative |
|---|---|---|---|
| Setup + PR0 baseline | 3 jours | inférence pure | week 1 |
| PR1 KVQuant | 3 semaines | calibration ~12h GPU | week 4 |
| PR2 MInference | 1 semaine | warmup ~6h GPU | week 5 |
| PR3 Quest | 3 semaines | benchmark ~24h GPU | week 8 |
| PR4 sinks | 2 jours | <1h GPU | week 8 |
| PR5 EM-LLM | 4 semaines | ~50h GPU pour gen episode store | week 12 |
| Evaluation + release | 2 semaines | ~30h GPU bench complet | week 14 |

**Total : ~14 semaines (3.5 mois) en solo.** Compute total : ~120h GPU. Aucune phase d'entraînement — uniquement inférence et calibration.
