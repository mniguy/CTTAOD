#!/bin/bash
set -e

echo "=== Step 1: Install PyTorch (CUDA-compatible) ==="
CUDA_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1)
if python -c "import torch" 2>/dev/null; then
    echo "PyTorch already installed, skipping."
elif [ -z "$CUDA_MAJOR" ]; then
    echo "No CUDA found. Installing CPU-only PyTorch."
    pip install torch torchvision torchaudio
elif [ "$CUDA_MAJOR" -ge 12 ]; then
    echo "CUDA $CUDA_MAJOR detected -> installing PyTorch with cu124."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "CUDA $CUDA_MAJOR detected -> installing PyTorch 1.11.0+cu113."
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
        --extra-index-url https://download.pytorch.org/whl/cu113
fi

echo "=== Step 2: Install requirements ==="
pip install -r requirements.txt

echo "=== Step 3: Install detectron2 ==="
pip install -e .

echo ""
echo "=== Done! Verify: ==="
python -c "import torch, detectron2; print('torch', torch.__version__); print('detectron2 OK')"