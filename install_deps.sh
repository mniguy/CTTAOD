#!/bin/bash
set -e

# ── Step 1: PyTorch ──────────────────────────────────────────────────────────
echo "=== Step 1: Install PyTorch (CUDA-compatible) ==="
CUDA_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1)
if python3 -c "import torch" 2>/dev/null; then
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

# ── Step 2: Patch torch CUDA version check ───────────────────────────────────
# nvcc major version may differ from torch.version.cuda (e.g. nvcc 13.0 vs cu124).
# PyTorch raises RuntimeError on major mismatch, blocking C++ extension builds.
# Downgrade to warning so compilation still proceeds (CUDA is runtime-compatible).
echo "=== Step 2: Patch torch CUDA major-version check ==="
CPP_EXT=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'utils/cpp_extension.py'))")
if grep -q "raise RuntimeError(CUDA_MISMATCH_MESSAGE" "$CPP_EXT"; then
    sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE/warnings.warn(CUDA_MISMATCH_MESSAGE/' "$CPP_EXT"
    echo "Patched: $CPP_EXT"
else
    echo "Already patched or not needed, skipping."
fi

# ── Step 3: Pin setuptools for pkg_resources compatibility ───────────────────
# setuptools ≥80 removed pkg_resources from its distribution.
# CLIP's setup.py uses pkg_resources.parse_requirements(), so we pin to <80.
echo "=== Step 3: Pin setuptools for CLIP compatibility ==="
pip install 'setuptools<80' wheel

# ── Step 3b: Install CLIP separately ─────────────────────────────────────────
# CLIP uses pkg_resources in setup.py and must be installed before the main
# requirements.txt (where it is commented out) to avoid build-isolation issues.
echo "=== Step 3b: Install CLIP ==="
pip install --no-build-isolation \
    git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1

# ── Step 4: requirements.txt ─────────────────────────────────────────────────
# --no-build-isolation: lets pip reuse already-installed torch/numpy during
# C-extension builds (pycocotools, etc.).
echo "=== Step 4: Install requirements ==="
pip install --no-build-isolation -r requirements.txt

# ── Step 5: detectron2 (editable) ────────────────────────────────────────────
echo "=== Step 5: Install detectron2 ==="
pip install --no-build-isolation -e .

echo ""
echo "=== Done! ==="
python3 -c "import torch, detectron2; print('torch', torch.__version__); print('detectron2 OK')"
