# HELIX-Lite — Long-context inference stack on 2× RTX 3090

Goal: serve **Qwen2.5-7B-Instruct-1M** at effective 1M context with SOTA quality, plus retrieval-based extension to ~5M, on 2× RTX 3090 (48 GB total VRAM).

Repo: https://github.com/xFuzzitx/helix-lite

## Status

- [x] **Setup** — NVIDIA 595.71.05 + CUDA 13.0, torch 2.11.0+cu130, vLLM 0.20.1, FlashInfer 0.6.8
- [x] **Baseline** — TP=1 @ 32K (3,395 t/s), TP=2 @ 128K (2,726 t/s, NIAH ✓) ([details](docs/PR_PLAN.md#pr0--baseline--done-2026-05-09))
- [x] **PR1a** — AWQ INT4 weights → 128K on a single 3090, **8/8 multi-needle recall** at 32K and 128K (frees GPU 1 for PR5)
- [x] **PR1b** — nuq4 KV: math + scales + Triton kernels validated end-to-end (vLLM integration deferred, see `src/kvquant/vllm_integration.md`)
- [x] **PR1c Phase 1A** — KVQuant math-wrapper in vLLM `FlashAttentionImpl`: per-layer nuq quant→dequant on K/V, `kv_cache` stays fp16 (no VRAM win yet). NIAH ✓ at 4K/32K/128K/200K/256K on Qwen2.5-7B-Instruct-1M-AWQ. 256K = natural ceiling of fp16 pool on a 24 GB 3090; Phase 1B (compact bit-packed pool + fused-dequant attention kernel) is the path to true 1M attention.
- [x] **PR1c Phase 2 calibration** — pure nuq2 v3 (outlier=0.05) failed needle at layer 27 (consistent with v1/v2); **mixed nuq2/nuq4 with cut=16** (16 shallow layers nuq2 + 12 deep nuq4) ✓ end-to-end in vLLM smoke at 4K/32K/128K/256K. **6.29× weighted nominal compression** vs fp16, calibration ready for Phase 1B's compact pool.
- [ ] **PR1c Phase 1B** — compact bit-packed KV pool + fused-dequant attention kernel to turn the 6.29× nominal into real VRAM savings → true 1M attention on 1× 3090.
- [ ] **PR2** — MInference Vertical-Slash — deferred (vLLM 0.20.1 lacks the merged kernels; details in `docs/PR_PLAN.md#pr2`)
- [x] **PR3 + PR4** — Quest top-K page selection w/ attention-sink prefix: math + Triton kernel + 8/8 tests; smoke on real Qwen2 layer-14 shows cos 0.97 vs dense at 25%% KV loaded (4× decode speedup, quality preserved)
- [x] **PR5a** — EM-LLM scaffold: surprise segmenter + episode pool on GPU 1 (11/11 unit tests, end-to-end smoke on real Qwen2 with 50%% top-1 self-recall)
- [x] **PR5b** — KV-chunk transfer & hot/cold attention swap: validated cos 0.93-0.98 vs dense at hot=2K + top-M=8 (39%% of context loaded, 5M scales as O(hot + M·episode_len))
- [x] **Release v0** — `helix-cli` ships the queryable model: vanilla AWQ at ≤128K and EM-RAG (segment + retrieve + answer) for longer docs
- [x] **Eval v0** — multi-needle NIAH for both paths (numbers below; retrieval is the EM-RAG bottleneck at >32K)
- [x] **Eval v0.1** — mid-layer indexer embeddings: 38%→75% @128K, 38%→62% @200K (+100% / +67%), no change to segmenter/pool/index
- [x] **Eval v0.2** — top-M proportional to ctx (top-M=256 at ≥512K): 0%→62% @1M, 38%→62% @512K. Sweet spot found; top-M=384 already regresses (distractors).
- [x] **Eval v0.3** — LLM-as-reranker on wider cosine top-K (1024 → top-M=64): **8/8 (100%) @256K**, plateaus at 5/8 @512K-1M (2 needles structurally outside cosine top-1024).
- [x] **Eval v0.4** — dedicated similarity encoder (BGE-M3, 568M, GPU 1): **8/8 (100%) @256K / 512K / 1M**, top-M=64, no rerank needed. RAG-killer goal hit on 2× RTX 3090.

## Eval results (v0)

Multi-needle NIAH, 8 distinct needles spread evenly across the doc,
graded by exact substring match on the answer.

**Vanilla AWQ path** (full doc in context):

| ctx | episodes | recall |
|-----|----------|--------|
| 32 K  | — | **8/8** (100%) |
| 128 K | — | **8/8** (100%) |

**EM-RAG path** (segment + retrieve + answer):

| ctx | episodes | v0 last/M=64 | v0.1 mid/M=64 | v0.2 mid/M=256 | v0.3 rerank topk=1024 | **v0.4 BGE-M3** |
|-----|---------:|---:|---:|---:|---:|---:|
| 32 K   | 202  | 6/8 (75%) | **7/8 (88%)** | — | — | — |
| 128 K  | 811  | 3/8 (38%) | **6/8 (75%)** | — | — | — |
| 200 K  | 1272 | 3/8 (38%) | **5/8 (62%)** | — | — | — |
| 256 K  | 1635 | —         | 6/8 (75%)     | 6/8 (75%) | 8/8 (100%) | **8/8 (100%)** |
| 512 K  | 3259 | —         | 3/8 (38%)     | 5/8 (62%) | 5/8 (62%) | **8/8 (100%)** |
| 1 M    | 6378 | —         | 0/8 (0%) 💀   | 5/8 (62%) | 5/8 (62%) | **8/8 (100%)** ⭐ |

v0 used the indexer's *last* layer hidden states (layer 28/28) as
episode embeddings. The last layer is specialised for next-token
prediction; its outputs are dominated by syntactic prediction
signal, not the semantic content needed for retrieval. Switching
to a *mid* layer (~layer 14/28) with the same max-abs pooling
roughly doubles recall at 128K (38% → 75%) and pushes 200K from
38% → 62%, without any change to the segmenter, pool function,
or retrieval index.

v0.2 (top-M=256) — at very long contexts the episode count grows
to thousands (6378 episodes at 1M tokens), so top-M=64 gives only
~1% selectivity and recall collapses to 0/8 at 1M. Scaling top-M
proportionally to keep ~4% selectivity restores recall: **5/8 at
1M ctx**, up from 0/8. Top-M=384 was tried as well but does worse
(4/8 at 1M) because the extra retrieved episodes act as distractors
that dilute the answer span — quality of top-M=256 matters more
than quantity beyond that.

Sweet spot: `indexer_layer="mid"` + `top_m=256` on 2× RTX 3090
sustains ≥62% recall from 32K up to 1M tokens.

v0.3 (rerank topk=1024 → top-M=64) — wider cosine top-K=1024 then
LLM-as-reranker on the indexer model. **Reaches 8/8 recall at 256K**
(up from 6/8). At 512K and 1M the AWQ recall plateaus at 5/8 because
two specific needles ("TY-3407" at depth 0.36, "PL-4350" at 0.77)
fall out of the top-1024 cosine retrieval entirely — i.e. the
mid-layer max-abs embedding is structurally blind to those values.
A first rerank attempt with topk=256 was useless (same recall as
cosine pure) because the needles weren't in the top-256 either —
the rerank can only sharpen a candidate set that already contains
the answer.

Cost: rerank adds ~500s of forward passes per ctx at topk=1024.
The wider-cosine + rerank path is exposed as `--rerank --rerank-topk N`
in the bench and `EMRAGConfig.rerank=True`, opt-in.

v0.4 (BGE-M3 dedicated embedding model) — replaces the indexer
hidden-state pooling with a real similarity encoder. Qwen2.5-7B
stays on GPU 0 for the segmenter (it needs *logits* to measure
surprise at episode boundaries), while BGE-M3 (568M, 1024-dim,
~1.1 GB BF16) lives on GPU 1 and produces every episode and query
embedding. The two GPUs are used in parallel and the dedicated
encoder is roughly an order of magnitude more selective than raw
hidden-state max-abs pooling.

Result: **8/8 (100%) recall at 256K, 512K, AND 1M**, with top-M=64
and no rerank. Both previously "structurally invisible" needles
(TY-3407 at depth 0.36, PL-4350 at 0.77) are recovered.

Cost: BGE-M3 load adds ~30s once per session. Per-context: a few
seconds to re-embed all episodes (1635 → ~5s, 6378 → ~15s).
Retrieval Phase A drops from 502s (rerank topk=1024) to 0.0s
because the cosine top-K is small (top-M=64) and BGE-M3 query
encoding is sub-second.

Use it via `--embedding-model BAAI/bge-m3` or
`EMRAGConfig.embedding_model="BAAI/bge-m3"`. Other sentence
encoders work too (Qwen3-Embedding, E5-Mistral) — CLS-token
pooling with L2-normalised vectors is the contract.

HyDE query expansion was tried and failed (-66% at 128K — generic
hypothetical passages drift away from the rare-token needle style).

Reproduce:

```bash
# v0.4: BGE-M3 dedicated encoder (recommended) — 8/8 at 256K, 512K, 1M
PYTHONPATH=src python benchmarks/quality/run_em_rag_multi_needle.py \
  --ctx 256000 512000 1000000 \
  --num-needles 8 --top-m 64 --max-doc-tokens 1100000 \
  --embedding-model BAAI/bge-m3

# v0.2: hidden-state pooling, mid layer + top-M=256 (no extra encoder)
PYTHONPATH=src python benchmarks/quality/run_em_rag_multi_needle.py \
  --ctx 32000 128000 256000 512000 1000000 \
  --num-needles 8 --top-m 256 --max-doc-tokens 1100000

# v0.1: same but top-M=64 (highlights the 0% collapse at 1M)
PYTHONPATH=src python benchmarks/quality/run_em_rag_multi_needle.py \
  --ctx 32000 128000 200000 --num-needles 8 --top-m 64

# v0 baseline (last layer, for comparison)
PYTHONPATH=src python benchmarks/quality/run_em_rag_multi_needle.py \
  --ctx 32000 128000 200000 --num-needles 8 --top-m 64 --indexer-layer last
```

Raw JSON in [`benchmarks/results/`](benchmarks/results/).

## Query the model

Once the venv is set up (see *Setup* below), the CLI is one command:

```bash
# vanilla path: full doc in context, up to 128K tokens, 1 GPU
./helix-cli --doc path/to/document.txt "Your question here"

# EM-LLM RAG path: works on docs > 128K — segments + retrieves the
# most relevant episodes before answering. Bump --top-m for longer
# docs (ratio top-m/episodes drives recall; see Eval below).
./helix-cli --doc large_book.txt --em-rag --top-m 64 "Who killed Roger Ackroyd?"

# multi-turn shell over a single document
./helix-cli --doc paper.pdf.txt --repl

# PR1c math-wrapper: route K/V through calibrated nuq quant→dequant
# in vLLM's attention. Validates calibration up to 256K; fp16 pool
# unchanged so no VRAM savings until Phase 1B (fused-dequant kernel).
./helix-cli --doc paper.pdf.txt \
  --nuq-scales scales/mixed_nuq2v3_nuq4v3_cut16.pt \
  "Your question"
```

First run downloads `graelo/Qwen2.5-7B-Instruct-1M-AWQ` (≈ 5 GB) to
the HF cache configured in `.env`. Subsequent runs are warm. Vanilla
mode loads in ~3 s and answers a 2 K-token-doc query in under a second.

## Setup

```bash
# 1. Install NVIDIA driver + CUDA toolkit (sudo, reboot required)
sudo bash setup/01_install_nvidia.sh
sudo reboot

# 2. After reboot — Python env + vLLM
bash setup/02_install_python.sh

# 3. Verify everything works
bash setup/03_verify.sh
# Expected: 2× RTX 3090 visible, torch.cuda OK, vllm imports
```

## Hardware

| Component | Spec |
|---|---|
| GPUs | 2× RTX 3090 24 GB (Ampere sm_86, no NVLink) |
| Total VRAM | 48 GB |
| Memory bandwidth | 936 GB/s per GPU |
| Inter-GPU | PCIe gen4 (~32 GB/s) |
| Host | Debian 13 (trixie), Linux 6.12 |

## Architecture target

```
GPU 0 (24 GB)                          GPU 1 (24 GB)
┌────────────────────────┐             ┌────────────────────────┐
│ Qwen2.5-7B-1M (fp16)   │             │ KV cold pages offload  │
│   weights:    ~14 GB   │             │ EM-LLM episodic store  │
│   KV (KVQuant): ~4 GB  │  ◄──PCIe──► │ JEPA decoder (later)   │
│   activations: ~3 GB   │             │                        │
│ Total: ~21 GB          │             │ Total: ~22 GB          │
└────────────────────────┘             └────────────────────────┘
```

At **1M tokens**: everything resides on GPU 0. GPU 1 idle/standby.
At **5M tokens**: hot 1M on GPU 0, cold 4M offloaded to GPU 1 + retrieval index.

## Layout

```
helix-lite/
├── setup/                    # install scripts (run once)
│   ├── 01_install_nvidia.sh  # drivers + CUDA (sudo)
│   ├── 02_install_python.sh  # venv + vLLM + deps
│   └── 03_verify.sh          # sanity checks
├── src/                      # modules per PR
│   ├── kvquant/              # PR1
│   ├── minference/           # PR2
│   ├── quest/                # PR3
│   ├── sinks/                # PR4
│   └── emllm/                # PR5
├── benchmarks/               # eval scripts
│   ├── run_baseline.py       # first script — loads model, measures @ 32K
│   ├── ruler/                # NVIDIA RULER suite
│   ├── niah/                 # needle-in-haystack
│   └── babilong/             # BABILong @ 1M / 5M
├── docs/
│   └── PR_PLAN.md            # detailed plan per PR
└── requirements.txt
```

## Plan détaillé

Voir [`docs/PR_PLAN.md`](docs/PR_PLAN.md) pour les critères d'acceptation et le détail technique de chaque PR.

## Références

- Recherche initiale : [`docs/research/5M_context_research.md`](docs/research/5M_context_research.md)
- Architecture HELIX (vision complète) : [`docs/research/HELIX_architecture.md`](docs/research/HELIX_architecture.md)
- Auto-critique : [`docs/research/HELIX_critique.md`](docs/research/HELIX_critique.md)
- Réalité 2× 3090 : [`docs/research/setup_2x3090_realistic.md`](docs/research/setup_2x3090_realistic.md)
