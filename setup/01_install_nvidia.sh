#!/usr/bin/env bash
# Install NVIDIA driver + CUDA toolkit on Debian 13 (trixie)
# Hardware: 2× RTX 3090 (Ampere sm_86)
#
# Usage: sudo bash 01_install_nvidia.sh
# A reboot is required after this script completes.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: this script must be run with sudo"
    echo "Usage: sudo bash $0"
    exit 1
fi

echo "===================================================="
echo " HELIX-Lite — NVIDIA driver install (Debian 13)"
echo "===================================================="
echo ""

echo "[1/7] Detecting Debian sources file format..."
SOURCES_NEW=/etc/apt/sources.list.d/debian.sources
SOURCES_OLD=/etc/apt/sources.list

if [[ -f "$SOURCES_NEW" ]]; then
    echo "  → using deb822 format at $SOURCES_NEW"
    if ! grep -q "non-free-firmware" "$SOURCES_NEW"; then
        sed -i '/^Components:/ s/main/main contrib non-free non-free-firmware/' "$SOURCES_NEW"
        echo "  ✓ added contrib non-free non-free-firmware"
    else
        echo "  ✓ already enabled"
    fi
elif [[ -f "$SOURCES_OLD" ]]; then
    echo "  → using legacy format at $SOURCES_OLD"
    sed -i '/^deb / s/main$/main contrib non-free non-free-firmware/' "$SOURCES_OLD"
    echo "  ✓ added contrib non-free non-free-firmware"
else
    echo "ERROR: no Debian sources file found"
    exit 1
fi

echo ""
echo "[2/7] apt-get update..."
apt-get update

echo ""
echo "[3/7] Installing kernel headers + build tools..."
apt-get install -y linux-headers-"$(uname -r)" build-essential dkms pkg-config

echo ""
echo "[4/7] Installing NVIDIA driver + CUDA toolkit + firmware..."
# nvidia-driver pulls in libcuda1, kmod, and the kernel module via DKMS
# nvidia-cuda-toolkit pulls in nvcc and the runtime libs
apt-get install -y \
    nvidia-driver \
    nvidia-smi \
    nvidia-cuda-toolkit \
    firmware-misc-nonfree \
    libnvidia-ml1

echo ""
echo "[5/7] Blacklisting nouveau (Debian usually handles this, double-check)..."
cat > /etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
update-initramfs -u

echo ""
echo "[6/7] Verifying package install..."
dpkg -l | grep -E "^ii\s+(nvidia-driver|nvidia-cuda-toolkit)" || true

echo ""
echo "[7/7] Done. Driver version installed:"
apt-cache policy nvidia-driver | head -2

echo ""
echo "===================================================="
echo " ⚠️  REBOOT REQUIRED"
echo "===================================================="
echo ""
echo " Run:   sudo reboot"
echo ""
echo " After reboot, verify with:"
echo "   nvidia-smi"
echo ""
echo " You should see 2× RTX 3090, 24576 MiB each."
echo ""
echo " Then run:"
echo "   bash setup/02_install_python.sh"
echo "===================================================="
