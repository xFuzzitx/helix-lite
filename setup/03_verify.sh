#!/usr/bin/env bash
# Sanity check: drivers loaded, GPUs visible, Python env imports cleanly
# Usage: bash 03_verify.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_DIR/.venv"

echo "===================================================="
echo " HELIX-Lite — verification"
echo "===================================================="

echo ""
echo "=== 1) NVIDIA driver + GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv

echo ""
echo "=== 2) nvcc (CUDA toolkit) ==="
nvcc --version | tail -2

echo ""
echo "=== 3) Python env ==="
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV — run setup/02_install_python.sh first"
    exit 1
fi
source "$VENV/bin/activate"

python <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (built): {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} | {p.total_memory/1e9:.1f} GB | sm_{p.major}{p.minor}")
    free, total = torch.cuda.mem_get_info(i)
    print(f"           free: {free/1e9:.1f} GB / {total/1e9:.1f} GB")

import vllm; print(f"vllm: {vllm.__version__}")
import transformers; print(f"transformers: {transformers.__version__}")
import flash_attn; print(f"flash_attn: {flash_attn.__version__}")
PY

echo ""
echo "=== 4) Disk space (need ~30 GB free for model + cache) ==="
df -h "$PROJECT_DIR" | head -2

echo ""
echo "=== 5) Memory math at 1M context ==="
python <<'PY'
# Qwen2.5-7B-Instruct-1M: 28 layers, 28 Q heads, 4 KV heads (GQA), head_dim 128
LAYERS = 28
KV_HEADS = 4
HEAD_DIM = 128
BYTES_PER_FP16 = 2

for ctx in [32_000, 128_000, 1_000_000, 5_000_000]:
    kv_bytes = ctx * LAYERS * KV_HEADS * HEAD_DIM * 2 * BYTES_PER_FP16
    kv_2bit  = kv_bytes / 8  # KVQuant nuq2
    print(f"  context {ctx:>10,} | KV fp16: {kv_bytes/1e9:6.2f} GB | KV 2-bit: {kv_2bit/1e9:6.2f} GB")
PY

echo ""
echo "===================================================="
echo " ✓ All checks passed"
echo "===================================================="
