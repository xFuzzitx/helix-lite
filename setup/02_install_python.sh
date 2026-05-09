#!/usr/bin/env bash
# Setup Python venv + vLLM + dependencies
#
# Usage: bash 02_install_python.sh
# (No sudo needed — installs user-local in .venv/)
#
# Prerequisites: 01_install_nvidia.sh + reboot

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_DIR/.venv"

echo "===================================================="
echo " HELIX-Lite — Python env + vLLM install"
echo "===================================================="
echo ""

echo "[0/6] Sanity check: nvidia-smi..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Run setup/01_install_nvidia.sh first and reboot."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "ERROR: nvidia-smi failed. Driver may not be loaded — did you reboot?"
    exit 1
}

echo ""
echo "[1/6] Checking Python 3.11 or 3.12..."
PYTHON=""
for v in 3.12 3.11; do
    if command -v python$v &> /dev/null; then
        PYTHON=$(command -v python$v)
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "  Python 3.11/3.12 not found, installing python3.12 from Debian repos..."
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip
    PYTHON=$(command -v python3.12)
fi
echo "  ✓ using $PYTHON"

echo ""
echo "[2/6] Creating venv at $VENV..."
if [[ -d "$VENV" ]]; then
    echo "  → venv already exists, skipping creation"
else
    "$PYTHON" -m venv "$VENV"
fi

echo ""
echo "[3/6] Activating + upgrading pip..."
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools

echo ""
echo "[4/6] Installing PyTorch (CUDA 12.4)..."
# vLLM 0.6.4+ requires torch>=2.5.0 with CUDA 12.4
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "[5/6] Installing vLLM + dependencies (~10 min, downloads ~3 GB)..."
pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "[6/6] Verifying imports..."
python -c "
import torch
print(f'  ✓ torch {torch.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
print(f'  ✓ GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'    GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB, sm_{p.major}{p.minor})')

import vllm
print(f'  ✓ vllm {vllm.__version__}')

import transformers
print(f'  ✓ transformers {transformers.__version__}')

import flash_attn
print(f'  ✓ flash_attn {flash_attn.__version__}')
"

echo ""
echo "===================================================="
echo " ✓ Install complete"
echo "===================================================="
echo ""
echo " Activate the env in your shell:"
echo "   source $VENV/bin/activate"
echo ""
echo " Run sanity verification:"
echo "   bash setup/03_verify.sh"
echo ""
echo " Run baseline (downloads ~14 GB model first time):"
echo "   python benchmarks/run_baseline.py"
echo "===================================================="
