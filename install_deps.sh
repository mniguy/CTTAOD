#!/bin/bash
set -e

# ============================================================
# CTTAOD Environment Setup Script
#
# Known issues handled:
#   - setuptools >= 71: pkg_resources.packaging missing -> downgrade
#   - CLIP: pkg_resources build error -> manual git install
#   - fasttext: GCC > 8 compile error + not used in codebase -> skip
#   - mmcv-full: not used in core codebase -> skip
#   - build order: pybind11/numpy must precede other C++ packages
#   - pip install -e . fails with setuptools 59 -> use setup.py develop
#   - torch iJIT_NotifyEvent: conda-built torch in venv -> use pip torch
#   - CUDA mismatch: torch cu113 vs system CUDA 12/13 -> detect and match
# ============================================================

echo "======================================================"
echo "  CTTAOD Environment Setup"
echo "======================================================"

# --- GCC check ---
echo ""
echo "=== [0/5] Checking GCC version (requires <= 8) ==="
GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
if [ "$GCC_VERSION" -gt 8 ]; then
    echo "WARNING: GCC $GCC_VERSION detected. C++ extensions may fail."
    echo ""
    echo "  Fix options:"
    echo "  [apt]   sudo add-apt-repository ppa:ubuntu-toolchain-r/test"
    echo "          sudo apt update && sudo apt install gcc-8 g++-8"
    echo "          sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 --slave /usr/bin/g++ g++ /usr/bin/g++-8"
    echo "  [conda] conda install -c conda-forge gcc=8 gxx=8"
    echo ""
    read -p "Continue anyway? [y/N] " answer
    [[ "$answer" =~ ^[Yy]$ ]] || exit 1
else
    echo "GCC $GCC_VERSION OK"
fi

# --- PyTorch install (CUDA-aware) ---
echo ""
echo "=== [1/5] Installing PyTorch ==="
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch already installed, skipping."
else
    CUDA_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1)
    if [ -z "$CUDA_MAJOR" ]; then
        echo "nvcc not found. Installing CPU-only PyTorch."
        pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
    elif [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "CUDA $CUDA_MAJOR detected. Installing PyTorch with cu124."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    else
        echo "CUDA $CUDA_MAJOR detected. Installing PyTorch 1.11.0+cu113."
        pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
            --extra-index-url https://download.pytorch.org/whl/cu113
    fi
fi

# --- setuptools downgrade ---
echo ""
echo "=== [2/5] Setting setuptools to 59.5.0 (torch cpp_extension compatibility) ==="
pip install "setuptools==59.5.0" -q

# --- CLIP ---
echo ""
echo "=== [3/5] Installing CLIP ==="
if python -c "import clip" 2>/dev/null; then
    echo "CLIP already installed, skipping."
else
    rm -rf /tmp/clip
    git clone https://github.com/openai/CLIP.git /tmp/clip -q
    cd /tmp/clip
    git checkout d50d76daa670286dd6cacf3bcd80b5e4823fc8e1 -q
    python setup.py install -q
    cd -
fi

# --- Build dependencies first (order matters) ---
echo ""
echo "=== [4/5] Installing requirements ==="
pip install numpy==1.23.3 Cython==0.29.32 pybind11==2.12.0 -q

# Skip: clip (manual), mmcv-full (unused), fasttext (unused + broken on GCC>8),
#        numpy/Cython/pybind11 (already installed above),
#        torch/torchvision/torchaudio (installed above)
grep -v -E \
    "^clip |^mmcv-full|^fasttext==|^numpy==|^Cython==|^pybind11==|^torch==|^torchvision==|^torchaudio==" \
    requirements.txt > /tmp/req_filtered.txt

pip install --no-build-isolation -r /tmp/req_filtered.txt

# --- detectron2 ---
echo ""
echo "=== [5/5] Installing detectron2 (dev mode) ==="
# Note: 'pip install -e .' fails with setuptools 59 (no build_editable hook)
#       Use setup.py develop instead.
python setup.py develop

echo ""
echo "======================================================"
echo "  Done! Verify with:"
echo "  python -c \"import torch, detectron2; print('OK')\""
echo "======================================================"
