# Cutting-Edge Techniques for Extending Small LLMs to 5M+ Token Context Windows

*Research compiled: May 2026. Target: take an existing 1B–10B model from 128K → 5M tokens, with a path to 50M.*

---

## Executive Summary

**Three bottlenecks govern 5M-context.** Attention compute is O(N²) — at 5M tokens with a 7B model, prefill on dense softmax attention is impossible on consumer or even single-server hardware. KV cache memory at 5M with standard GQA is ~164 GB for a 7B (Llama-class), roughly 2× a single H100. And quality — measured by RULER, NIAH, MRCR, BABILong — collapses long before claimed length: Llama 4 advertises 10M but effective is ~5–6.5M; Gemini 2.5 Pro retains only 16.4% on MRCR-V2 8-needle at 1M; Sonnet 4.5 retains 21%. The current public state of the art is **MiniMax-Text-01 at 0.91 RULER at 1M** (456B MoE, 7:1 lightning:softmax hybrid).

**The convergent production recipe.** Every model that demonstrably handles ≥1M with usable recall combines: (1) **a sub-quadratic core** — linear/lightning attention or hierarchical sparse selection, (2) **a small fraction of full softmax layers** to anchor recall (typically 1 in 4–8 layers), (3) **MLA or aggressive KV compression** (DeepSeek V4 cuts KV to ~2% of GQA), (4) **a progressive RoPE-base + length curriculum** (Qwen ramps base 10K → 10M across five stages), and (5) **an indexer-then-attend mechanism** that selects relevant tokens before full attention runs (DeepSeek's Lightning Indexer, MInference's Vertical-Slash, NSA's compress/select/slide).

**For a small model, distill — don't pretrain.** MOHAWK and Mamba-in-the-Llama show transformer→hybrid conversion works at <1% of from-scratch data: Phi-Mamba in 3B tokens, Llama3-8B-Instruct in tens of billions. Combined with the architectural recipe above, a 7B → 5M conversion is a small-team task (10–25K H100-hours), not a frontier-lab task. Achieving 50M, however, still requires ARMT-style segment recurrence or DeepSeek V4-style compression cascades — public models with verified recall above 1M are still rare in 2026.

---

## 1. Why 5M is Hard — The Numbers

For a Llama-class **7B** model (32 layers, 32 Q heads, 8 KV heads, head_dim 128, fp16):

| Quantity | At 128K | At 1M | At 5M |
|---|---|---|---|
| KV cache (GQA, fp16) | 4.2 GB | 33 GB | **164 GB** |
| KV cache (MHA, fp16) | 17 GB | 131 GB | 655 GB |
| Dense attention FLOPs (one prefill) | ~3.5 PFLOP | ~2.1 EFLOP | ~52 EFLOP |
| Wall-clock dense prefill (H100 BF16) | ~1 s | ~10 min | ~4 hours |

Even setting attention compute aside, **KV memory at 5M alone exceeds two H100s with no room for weights or activations.** This is why every credible 1M+ system either compresses the KV cache aggressively, replaces softmax attention with sub-quadratic mixers, or both.

Quality-wise, the Sparse Frontier paper (arXiv:2504.17768) finds that **retrieval tolerates 95% attention sparsity**, but **multi-hop reasoning and aggregation collapse below 50–67% retention.** The recall budget at 5M needs to be at least 32K–128K tokens to keep RULER above 80%, regardless of how those tokens are selected.

---

## 2. Technique Families

### 2.1 Position Embedding Extension (post-hoc, fine-tune-light)

| Technique | Mechanism | Demonstrated context | Cost |
|---|---|---|---|
| **YaRN** (arXiv:2309.00071) | NTK-by-parts ramp + softmax temperature scaling | Llama-2 → 128K | ~64 GPU-hr / 400M tokens |
| **LongRoPE** (arXiv:2402.13753) | Evolutionary search over per-dim RoPE factors; progressive PI→search→retune | Llama-2 → **2M** | A few thousand GPU-hr |
| **LongRoPE2** (arXiv:2502.20082) | **Needle-driven PPL** in evolutionary search + mixed-context-window training. Diagnoses that high-RoPE-dim are *under-trained* | Llama-3 8B → **true effective 128K with only 10B tokens (80× cheaper than Meta's recipe)** | ~10B tokens |
| **Nvidia UltraLong** (arXiv:2504.06214) | Llama-3.1-8B + 1B continued-pretrain tokens + YaRN. **100% NIAH at 4M** | 8B → 4M | 256 H100s × 13 hr ≈ 3,300 H100-hr |
| **PoSE** (arXiv:2309.10400) | Skip-wisE position augmentation: train at 2K, simulate any offset up to L_target | Llama-2 → 128K from 2K window | Reduces VRAM/time ~10× |
| **LongLoRA / S²-Attn** (arXiv:2309.12307) | Shifted sparse attention at training only + LoRA | Llama-2 13B → 64K on 8×A100 in 52 hr | LoRA-cheap |
| **iRoPE (Llama 4 Scout, 2025)** | Every 4th layer drops RoPE (NoPE for global) + chunked attention with RoPE on others + inference-time temperature scaling | 256K trained, 10M claimed | Pretrain-time |
| **Critical attention scaling** (arXiv:2510.05554) | Theory: β_n ∝ log n rescaling justifies YaRN/Qwen | (theoretical) | – |

**Key takeaway:** for 7B → 5M, **LongRoPE2-style search + Nvidia UltraLong's continued-pretrain recipe + PoSE for cheap simulation** is the position-embedding stack. iRoPE's chunked + NoPE pattern is the only published recipe that's actually demonstrated at 10M.

### 2.2 Sparse / Efficient Attention

| Technique | Mechanism | Notes | Train? |
|---|---|---|---|
| **DeepSeek NSA** (arXiv:2502.11089, ACL 2025 Best Paper) | Three branches: compressed (block-aggregated MLP), selected (top-16 blocks), sliding-window (512); merged via sigmoid gate | 9× forward / 6× backward / 11.6× decode at 64K vs FlashAttention-2 | Native pretraining required |
| **DeepSeek V3.2 DSA** (arXiv:2512.02556) | Lightning Indexer over MLA-compressed KV → **top-k=2048** selection per query. **O(L·k) linear**, fine-grained per-token | API price cut ~50%, quality parity with V3.1 | Continued-pretraining; not from-scratch |
| **MoBA** (arXiv:2502.13189, Moonshot/Kimi) | MoE-style routing applied to attention: blocks + parameter-less top-k gate | 6.5× at 1M, FlashAttention-2 compatible. Production at Kimi 1M | Continued-pretrain ~100–300 GPU-hr from a dense base |
| **MInference** (arXiv:2407.02490, Microsoft) | Per-head offline pattern classification (A-shape / Vertical-Slash / Block-Sparse) → dynamic Triton kernels | **~10× prefill at 1M, drop-in to vLLM/SGLang**, no retraining | Inference-only |
| **Quest** (arXiv:2406.10774) | Page-level KV with min/max key bounds + query-aware criticality | 7× attention speedup, 2.23× decode, negligible loss | Inference-only |
| **StreamingLLM** (arXiv:2309.17453) | Sliding window + retain first ~4 "sink" tokens | Streams to 4M without losing fluency, but **forgets evicted tokens** (fails NIAH at long range) | Inference-only |
| **Sliding window + global** (Longformer, BigBird, Gemma 3) | Local window + a few global tokens. Gemma 3 alternates 5 local : 1 global with separate RoPE bases | KV memory drops from 60% → <15% | Pretrain-time |
| **LongNet / Dilated** (arXiv:2307.02486) | Exponentially expanding dilation, log-depth dependency | Claims 1B-token feasibility but few independent reproductions | – |
| **MagicPIG / RetrievalAttention** | LSH or ANN over KV in CPU RAM | Decoding-only, retrieval-style sparse | Inference-only |

**The Sparse Frontier finding** (arXiv:2504.17768): retrieval tasks tolerate 95% sparsity; aggregation and reasoning fail at 50–67%; **the sparsity budget should grow sublinearly with N** — this is the right design principle.

**Stack to ship without retraining:** MInference (10× prefill drop-in) + Quest at decode + StreamingLLM sinks. **With ~100–300 GPU-hr budget:** convert to MoBA or fine-tune for NSA-style pattern.

### 2.3 SSM / Hybrid Architectures (the harder rewrite)

The 2024–2026 industry default is **6–8 sub-quadratic layers per 1 full-attention layer**:

| Architecture | SSM/Linear : Full ratio | Demonstrated context | Notes |
|---|---|---|---|
| **MiniMax-Text-01** (arXiv:2501.08313) | 7 lightning : 1 softmax (per 8-layer block) | Trained 1M, deployed 4M, **0.91 RULER at 1M** | 456B / 45.9B-active MoE |
| **Samba** (arXiv:2406.07522, MS) | Mamba → MLP → SWA → MLP | Trained at 4K, **extrapolates to 1M zero-shot** with 4K finetuning | 3.8B params |
| **Qwen3-Next 80B-A3B** (HF, 2025) | 75% Gated DeltaNet : 25% full | 1M, ~10× throughput vs Qwen3-32B at 32K+ | KV at 1M ≈ 25 GB |
| **Jamba 1.5** (AI21, arXiv:2403.19887) | 1 attn : 7 Mamba per 8-layer block + MoE | 256K | 398B / 94B-active |
| **Nemotron-H** (NVIDIA, arXiv:2504.03624) | 24 Mamba-2 + 4 attn + 24 MLP | – | Up to 3× faster |
| **Mamba-3** (ICLR 2026, arXiv:2603.15569) | Pure SSM with complex state + MIMO | "Inference-first" successor to Mamba-2 | Stronger length-gen |
| **RWKV-7 "Goose"** | Dynamic State Evolution + Generalized Delta Rule | Breaks TC0 expressivity ceiling | Pure recurrent decode |

**The key finding for small models:** *distillation works.* You don't have to pretrain from scratch.

| Distillation method | Data | Result |
|---|---|---|
| **MOHAWK** (arXiv:2408.10189) | **3B tokens (<1% of from-scratch)** | Phi-Mamba (1.5B) |
| **Mamba in the Llama** (arXiv:2408.15237) | Tens of billions | Llama3-8B-Instruct → hybrid; ¼ attention layers retained; 5× throughput; "almost perfect NIAH at 20× distillation length" |
| **Llamba** (Cartesia, arXiv:2502.14458) | <0.1% data | 8B at 12× decode throughput |
| **RADLADS** (arXiv:2505.03005) / **Attention-Bridge** (arXiv:2510.19266) | – | Generalized any-transformer → linear conversion |

### 2.4 KV Cache Optimization

| Method | Mechanism | Reduction | Train? |
|---|---|---|---|
| **MLA** (DeepSeek-V2/V3, arXiv:2412.19437) | Low-rank latent KV with **weight absorption** so latent is consumed directly | **128K context: 488 GB → 7.6 GB (~64× vs MHA, 12× vs GQA)** | Pretrain-time |
| **YOCO** (arXiv:2405.05254) | Self-decoder produces KV once; cross-decoder reuses globally | 80× for 65B, 9.38× at 1M for 3B; prefill 71.8× faster at 1M | Pretrain-time |
| **CLA** (arXiv:2405.12981) | Adjacent layers share KV | Additional 2–3× on top of MQA | Pretrain-time |
| **KVQuant nuq2** (arXiv:2401.18079) | 2-bit non-uniform + dense+sparse outliers | **8×; LLaMA-7B at 1M on a single A100, 10M on 8 GPUs** | Inference-only |
| **KIVI** (arXiv:2402.02750) | Asymmetric: K per-channel, V per-token, 2-bit | 2.6× peak; 4× larger batch | Inference-only |
| **Quest** | Query-aware page-level top-K | 7× attention; "negligible accuracy loss" | Inference-only |
| **PyramidKV** (arXiv:2406.02069) | Pyramidal allocation: more cache lower, less upper | ~8× retention; 2.2× throughput | Inference-only |
| **SnapKV** (arXiv:2404.14469) | Observation window predicts important tokens during prefill | 4–8× | Inference-only |
| **H2O** (arXiv:2306.14048) | Heavy hitters + recent window | ~5×; 29× throughput | Inference-only |
| **MiniCache** (arXiv:2405.14366) | Cross-layer merging via magnitude-direction decomposition | 5.02× w/ 4-bit, ~5× throughput | Inference-only |
| **RocketKV** (arXiv:2502.14051) | Two-stage coarse eviction + fine top-k | **400× compression, 3.7× speedup, training-free** | Inference-only |
| **PagedAttention** (arXiv:2309.06180) | Block-paged KV; eliminates fragmentation | Substrate (vLLM); not optional | – |
| **LayerKV / InfiniGen / HEADINFER** | Layer or head-wise CPU offload + speculative prefetch | **Single 24 GB GPU serves 4M Llama-3-8B** (HEADINFER) | Inference-only |
| **SGLang RadixAttention** (arXiv:2312.07104) | Token-level radix-tree prefix cache | 5× faster on multi-turn / shared prefixes | – |

**Math: getting 5M onto a single H100.** Starting at 164 GB GQA fp16 baseline:

| Step | Technique | Factor | Running total |
|---|---|---|---|
| 0 | GQA fp16 | 1× | 164 GB |
| 1 | KVQuant nuq2 | 8× | **20.5 GB** |
| 2 | Quest top-K (load 25%) | 4× active | 5.1 GB working set, 20.5 GB resident |
| 3 | LayerKV CPU offload of cold pages | shifts to CPU | ~5–8 GB GPU active |

20.5 GB resident KV + ~14 GB 7B fp16 weights = ~35 GB total, fits H100 80GB with headroom for activations. **No retraining required.** If you can pretrain with MLA, KV drops to ~14 GB before any quantization.

### 2.5 Memory and Recurrent Mechanisms

| Method | Footprint | Demonstrated | Train? |
|---|---|---|---|
| **ARMT** (arXiv:2407.04841) | Constant per segment | **80% on BABILong at 50M tokens** — only verified result at this scale, GPT-2 base | Curriculum fine-tune |
| **Activation Beacon** (arXiv:2401.03462) | Linear in #beacons (8× compression) | Llama-2-7B 4K → 400K with comparable performance, 2× speedup | Plug-in module on frozen LLM |
| **RMT / AutoCompressor** | Constant per segment | – | Fine-tune |
| **EM-LLM** (ICLR 2025, arXiv:2407.09450) | Linear (non-parametric) | **Validated at 10M tokens**, beats InfLLM | **Inference-only** |
| **YOCO** | Linear with single shared cache | 1M near-perfect needle retrieval | From scratch |
| **Titans** (arXiv:2501.00663) | Constant (MLP memory) | 2M+ NIAH | From scratch; **code never released, weak reproducibility** |
| **Infini-attention** (arXiv:2404.07143) | Constant per layer | 1M passkey claim | **HF replication failed** — gating did not converge. *Skip.* |
| **MemGPT** (arXiv:2310.08560) | Unbounded (disk) | – | Inference-only (tool calls) |
| **LongMem / Memorizing Transformers** | Linear in stored tokens | PG-19 perplexity | Foundational, pre-1M era |
| **Larimar / CAMELoT / EdgeInfinite / LCM** | Various | – | – |

**Verdict:** for a 5M+ system on a small existing transformer, **don't extend full attention** — the KV and IO costs are prohibitive even with Flash/Ring. Use **local full-attention (32K–128K) + Activation Beacon for compressed past + EM-LLM for retrieval at >1M + RAG fallback** for static corpora. ARMT is the only architecture with a verified 50M result, but requires curriculum fine-tuning.

---

## 3. What Production Models Actually Do

| Model | Context | RULER at full | Architecture | KV trick | Length training |
|---|---|---|---|---|---|
| **MiniMax-Text-01** | 4M | **0.910 at 1M** | 7 lightning : 1 softmax MoE | – | Trained at 1M, extrap 4M |
| **Gemini 2.5 Pro** | 1M | LOFT-hard 1M = 69.8%; MRCR-V2 8-needle 1M = 16.4% | Sparse MoE multimodal | undisclosed | "hill-climbing" against LOFT/MRCR |
| **Gemini 1.5 Pro** | 1M (2M Pro) | (paper claims good) | sparse MoE | – | "modeling and data advances" |
| **Llama 4 Scout** | 10M claimed | effective ~5–6.5M | iRoPE + chunked attn + temperature scaling | GQA | Trained 256K + mid-train |
| **DeepSeek V3.2** | 128K | (parity with V3.1) | MLA + DSA Lightning Indexer top-2048 | MLA | 2-stage continued |
| **DeepSeek V4** (Apr 2026) | 1M | MRCR 0.59 at 1M | CSA (4×) + HCA (128×) alternating | FP8/FP4 to 2% of GQA | Native |
| **Qwen2.5-1M (7B/14B)** | 1M | 92.7 RULER avg at 256K; >80% passkey 1M with DCA | Dense + DCA + MInference | GQA | 5-stage 4K → 256K, RoPE base 10K → 10M |
| **Magic LTM-2-mini** | 100M | – (HashHop only) | Non-attention sequence mixer (~1000× cheaper FLOPs) | unknown | unknown |
| **Sonnet 4.5/4.6 / Opus 4.7** | 1M | MRCR-V2 1M = 21% | undisclosed | undisclosed; flat pricing implies sub-quadratic | undisclosed |
| **Yi-Lightning** | – | – | 3 SWA : 1 full + cross-layer KV reuse | 82.8% mem cut | – |
| **Jamba 1.5** | 256K | top-tier | 1 attn : 7 Mamba + MoE | – | – |
| **Qwen3-Next 80B-A3B** | 1M | – | 75% GDN : 25% full + MoE | KV at 1M ≈ 25 GB | – |

**Five convergent design choices** across every model that cracks 1M with usable recall:

1. **Sub-quadratic main mixer** (linear attention, SSM, sparse selection) with **a small fraction of full softmax** to anchor recall.
2. **MoE for capacity decoupled from attention cost** — 100B–1.6T total, 13–50B active.
3. **Progressive context curriculum + aggressive RoPE base scaling** — train 4K, ramp to 256K, RoPE base 10K → 10M.
4. **KV compression to ≤10% of GQA at inference** — MLA, FP8/FP4, cross-layer reuse.
5. **Indexer-then-attend** — cheap relevance scoring before full attention runs.

The pricing tells: **DeepSeek V3.2 cut API price ~50% on the same hardware** when it switched to DSA, and **Anthropic charges flat across 1M (no quadratic premium)** — strong evidence both have shipped sub-quadratic prefill in production.

---

## 4. Three Roadmaps for a 7B → 5M-Context Model

### Roadmap A — Cheap path (~500 H100-hours, inference-only)

Start: a 128K-extended 7B (Qwen2.5-7B-1M, Llama-3.1-8B-Instruct, or Mistral-Nemo).

**Stack:**
1. **vLLM PagedAttention + automatic prefix cache** (substrate)
2. **KVQuant nuq2** for 8× KV compression
3. **MInference** drop-in for 10× prefill at 1M
4. **Quest** page-level top-K at decode (budget 64K–128K)
5. **LayerKV** CPU offload for cold pages
6. **StreamingLLM** sinks (always keep first 4 tokens)
7. **EM-LLM** episodic retrieval beyond 1M
8. **RAG fallback** for static corpora

**Outcome:** 7B handling ~1–2M with usable recall, reaches 5M via EM-LLM retrieval (not true attention). RULER at 1M ≈ 70–80%; RULER at 5M unknown (gap from extrapolation).

**Limit:** the underlying RoPE and attention patterns were never trained at 5M, so true cross-context reasoning at 5M is broken. This is the path if you need *capability* but not *quality*.

### Roadmap B — Medium path (~5K H100-hours, fine-tuning + sparse attention)

Same base. Add:

1. **LongRoPE2 evolutionary search** of per-dim RoPE factors using needle-driven PPL targeting 5M
2. **Continued pretraining** ~10–20B tokens of long-form (books, code repos, mined web ≥1M tokens), Nvidia UltraLong recipe scaled
3. **PoSE** to expose all relative offsets up to 5M while training in 32K windows
4. **LongLoRA S²-Attn** for cheap full-pass training
5. **MoBA continued-pretraining** (~100–300 GPU-hr) to make attention block-sparse at training, FlashAttention-2-compatible at inference
6. Deploy with the Roadmap A inference stack (KVQuant + Quest + LayerKV + sinks)

**Outcome:** true effective context 1–2M with NIAH retention >80% at 1M; degraded but usable at 5M (50–70% NIAH expected). Roughly the Qwen2.5-1M class of result.

**Compute:** ~256–512 H100s × 24–48 hours via Ring/Ulysses sequence parallelism + FlashAttention-3.

### Roadmap C — Aggressive path (~25K H100-hours, hybrid conversion via distillation)

This is the path that *actually* targets 5M with quality at 1M:

1. **Distillation to hybrid**: apply **Mamba-in-the-Llama** or **MOHAWK** to the 7B, replacing ~75% of attention layers with **Mamba-2 or Gated DeltaNet**, retaining ~25% as **sliding-window attention** (Samba-style 4K window). Use Llama3-8B-Instruct or Qwen3-7B as teacher. ~10–30B token distillation budget.
2. **iRoPE** on the retained attention layers: every 4th drops RoPE entirely; chunked attention with RoPE on others; inference-time temperature scaling on NoPE.
3. **Length curriculum**: 4K → 64K → 256K → 1M progressive, with PoSE simulation.
4. **MLA** at the retained attention layers (if you have the retraining budget), or fall back to GQA.
5. **NSA or DSA-style sparse selection** on the retained attention.
6. **Activation Beacon** as a learned per-layer compressed-memory side-module to handle 1M → 5M extrapolation.
7. Deploy with Roadmap A inference stack.

**Outcome:** 7B at 5M with NIAH/MRCR retention competitive with Qwen2.5-1M at 1M, plus a credible extrapolation curve to 5M from the SSM/linear core (cf. Samba's 4K → 1M zero-shot).

**Compute:** 256 H100s × ~4 days for distillation, plus length-curriculum training. ~10–25K H100-hours total. Within reach of a small team.

---

## 5. Path to 50M

Above 5M, only three approaches have any public evidence:

1. **ARMT-style segment recurrence with associative memory** (arXiv:2407.04841): 80% on BABILong at 50M tokens on a GPT-2 base. Requires curriculum fine-tuning. **The only verified 50M-token result.**
2. **Compression cascades** (DeepSeek V4): CSA (4× sequence compression) + HCA (128× compression) alternating per layer, with FP4-quantized indexers. V4 demonstrates 1M with MRCR 0.59; the architecture conceptually scales further.
3. **Sub-quadratic mixers + retrieval offload** (Magic LTM-2-mini): claimed 100M via a non-attention sequence mixer (~1000× cheaper FLOPs than dense attention at 100M). No public RULER. Likely state-space / linear-attention family.

**The honest 2026 frontier:** > 80% recall at 1M is now table stakes. Getting the same recall at 10M is shipped (Llama 4) but quality is degraded. Getting it at 50M is unsolved publicly outside ARMT-style fine-tuning on synthetic benchmarks.

For a 5M-context system targeting future 50M, the cleanest add-on is **ARMT segment-recurrent memory tokens combined with EM-LLM retrieval**. Architecturally: keep the Roadmap C hybrid backbone, add a Beacon-like compressed-memory module per layer, and a non-parametric retrieval store on the side.

---

## 6. Validation — Do Not Trust Average PPL

Average perplexity drops with context but does not measure recall or reasoning. The benchmarks that matter:

- **RULER** (NVIDIA, arXiv:2404.06654) — variable-length needle/multi-needle/multi-hop, the field's standard.
- **Needle-in-a-Haystack** (NIAH) — single needle retrieval, easiest test, often passed by methods that fail RULER.
- **MRCR-V2** (multi-round coreference, Gemini) — 8-needle is the hardest commodity test; **at 1M, all public models score below 60%**.
- **BABILong** — segment-friendly multi-hop reasoning, where ARMT shines at 50M.
- **InfiniteBench** (arXiv:2402.13718) — long-form code, math, retrieval.
- **LOFT-hard** — Gemini 2.5 Pro's lead benchmark (87% at 128K, 69.8% at 1M).
- **NoLiMa** — needle without literal match, exposes naive lexical-retrieval bypasses.

A reasonable evaluation gate before claiming 5M: **>80% RULER avg at 1M, >70% RULER avg at 4M, >60% MRCR-V2 8-needle at 1M.** No public open-source 7B has hit this combination as of May 2026.

---

## 7. References

### Position embedding
- YaRN — https://arxiv.org/abs/2309.00071 — https://github.com/jquesnelle/yarn
- LongRoPE — https://arxiv.org/abs/2402.13753 — https://github.com/microsoft/LongRoPE
- LongRoPE2 — https://arxiv.org/abs/2502.20082
- PoSE — https://arxiv.org/abs/2309.10400 — https://github.com/dwzhu-pku/PoSE
- LongLoRA — https://arxiv.org/abs/2309.12307 — https://github.com/dvlab-research/LongLoRA
- Nvidia UltraLong — https://arxiv.org/abs/2504.06214
- Llama 4 iRoPE — https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Critical attention scaling — https://arxiv.org/abs/2510.05554

### Sparse / efficient attention
- DeepSeek NSA — https://arxiv.org/abs/2502.11089
- DeepSeek V3.2 DSA — https://arxiv.org/abs/2512.02556
- MoBA — https://arxiv.org/abs/2502.13189 — https://github.com/MoonshotAI/MoBA
- MInference — https://arxiv.org/abs/2407.02490 — https://github.com/microsoft/MInference
- Quest — https://arxiv.org/abs/2406.10774 — https://github.com/mit-han-lab/Quest
- StreamingLLM — https://arxiv.org/abs/2309.17453 — https://github.com/mit-han-lab/streaming-llm
- The Sparse Frontier — https://arxiv.org/abs/2504.17768
- Gemma 3 (alternating local/global) — https://arxiv.org/abs/2503.19786
- LongNet — https://arxiv.org/abs/2307.02486

### SSM / hybrid
- Mamba-2 / SSD — https://arxiv.org/abs/2405.21060
- Samba — https://arxiv.org/abs/2406.07522 — https://github.com/microsoft/Samba
- Jamba — https://arxiv.org/abs/2403.19887
- MiniMax-Text-01 — https://arxiv.org/abs/2501.08313 — model card https://huggingface.co/MiniMaxAI/MiniMax-Text-01
- Qwen3-Next — https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Gated DeltaNet — https://arxiv.org/abs/2412.06464
- Titans — https://arxiv.org/abs/2501.00663
- MOHAWK — https://arxiv.org/abs/2408.10189
- Mamba in the Llama — https://arxiv.org/abs/2408.15237
- Llamba — https://arxiv.org/abs/2502.14458
- RADLADS — https://arxiv.org/abs/2505.03005
- Mamba-3 — https://arxiv.org/abs/2603.15569
- Nemotron-H — https://arxiv.org/abs/2504.03624
- Zamba2 — https://arxiv.org/abs/2405.16712
- xLSTM — https://arxiv.org/abs/2405.04517

### KV cache
- KVQuant — https://arxiv.org/abs/2401.18079 — https://github.com/SqueezeAILab/KVQuant
- KIVI — https://arxiv.org/abs/2402.02750
- Quest, MInference, StreamingLLM (above)
- PyramidKV — https://arxiv.org/abs/2406.02069
- SnapKV — https://arxiv.org/abs/2404.14469
- H2O — https://arxiv.org/abs/2306.14048
- MiniCache — https://arxiv.org/abs/2405.14366
- MLA (DeepSeek-V3) — https://arxiv.org/abs/2412.19437
- YOCO — https://arxiv.org/abs/2405.05254
- CLA — https://arxiv.org/abs/2405.12981
- RocketKV — https://arxiv.org/abs/2502.14051
- HEADINFER — https://arxiv.org/abs/2502.12574
- vLLM PagedAttention — https://arxiv.org/abs/2309.06180
- SGLang RadixAttention — https://arxiv.org/abs/2312.07104
- NVIDIA kvpress — https://github.com/NVIDIA/kvpress

### Memory mechanisms
- Activation Beacon — https://arxiv.org/abs/2401.03462
- ARMT — https://arxiv.org/abs/2407.04841 — https://github.com/RodkinIvan/associative-recurrent-memory-transformer
- EM-LLM — https://arxiv.org/abs/2407.09450 — https://em-llm.github.io/
- Infini-attention — https://arxiv.org/abs/2404.07143 (skip — failed reproductions)
- HF Infini-attention reproduction failure — https://huggingface.co/blog/infini-attention
- MemGPT — https://arxiv.org/abs/2310.08560
- BABILong — https://github.com/booydar/babilong

### Production models
- Gemini 2.5 — https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf
- Gemini 1.5 — https://arxiv.org/abs/2403.05530
- Magic LTM-2-mini — https://magic.dev/blog/100m-token-context-windows
- Qwen2.5-1M — https://arxiv.org/abs/2501.15383
- DeepSeek V4 — https://huggingface.co/blog/deepseekv4
- Yi-Lightning — https://arxiv.org/abs/2412.01253

### Benchmarks
- RULER — https://arxiv.org/abs/2404.06654
- InfiniteBench — https://arxiv.org/abs/2402.13718
