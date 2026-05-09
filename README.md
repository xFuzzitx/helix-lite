# HELIX-Lite — Long-context inference stack on 2× RTX 3090

Goal: serve **Qwen2.5-7B-Instruct-1M** at effective 1M context with SOTA quality, plus retrieval-based extension to ~5M, on 2× RTX 3090 (48 GB total VRAM).

## Status

- [x] **Setup** — NVIDIA 595.71.05 + CUDA 13.0, torch 2.11.0+cu130, vLLM 0.20.1, FlashInfer 0.6.8
- [x] **Baseline** — TP=1 @ 32K (3,395 t/s), TP=2 @ 128K (2,726 t/s, NIAH ✓) ([details](docs/PR_PLAN.md#pr0--baseline--done-2026-05-09))
- [x] **PR1a** — AWQ INT4 weights → 128K on a single 3090 with NIAH ✓ (frees GPU 1 for PR5)
- [x] **PR1b** — nuq4 KV: math + scales + Triton kernels validated end-to-end (vLLM integration deferred, see `src/kvquant/vllm_integration.md`)
- [ ] **PR1c** — KVQuant nuq2 (2-bit KV, 8× compression) — unlocks 1M+
- [ ] **PR2** — MInference Vertical-Slash — deferred (vLLM 0.20.1 lacks the merged kernels; details in `docs/PR_PLAN.md#pr2`)
- [ ] **PR3** — Quest top-K page selection (decode)
- [ ] **PR4** — StreamingLLM attention sinks
- [x] **PR5a** — EM-LLM scaffold: surprise segmenter + episode pool on GPU 1 (11/11 unit tests, end-to-end smoke on real Qwen2 with 50%% top-1 self-recall)
- [ ] **PR5b** — KV-chunk transfer & hot/cold attention swap (retrieval to 5M)
- [ ] **Release** — full RULER/MRCR/BABILong + open-source repo

## Quickstart

```bash
# 1. Install NVIDIA driver + CUDA toolkit (sudo, reboot required)
sudo bash setup/01_install_nvidia.sh
sudo reboot

# 2. After reboot — Python env + vLLM
bash setup/02_install_python.sh

# 3. Verify everything works
bash setup/03_verify.sh
# Expected: 2× RTX 3090 visible, torch.cuda OK, vllm imports

# 4. Run the baseline (downloads ~14 GB model on first run)
source .venv/bin/activate
python benchmarks/run_baseline.py
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
