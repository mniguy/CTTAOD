#!/bin/bash
set -e

# ── Step 1: PyTorch ──────────────────────────────────────────────────────────
echo "=== Step 1: Install PyTorch (CUDA-compatible) ==="
CUDA_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1)
# Check if installed torch is compatible with the current CUDA environment.
# A torch built for cu102 must be replaced when NVCC reports CUDA 12+.
TORCH_CUDA_OK=false
if python3 -c "import torch" 2>/dev/null; then
    TORCH_CU=$(python3 -c "import torch; print(torch.version.cuda or '')" 2>/dev/null)
    TORCH_CU_MAJOR=$(echo "$TORCH_CU" | cut -d. -f1)
    if [ -n "$CUDA_MAJOR" ] && [ -n "$TORCH_CU_MAJOR" ] && [ "$TORCH_CU_MAJOR" = "$CUDA_MAJOR" ]; then
        TORCH_CUDA_OK=true
    elif [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -ge 12 ] && echo "$TORCH_CU" | grep -qE '^1[2-9]'; then
        TORCH_CUDA_OK=true
    elif [ -z "$CUDA_MAJOR" ]; then
        TORCH_CUDA_OK=true  # CPU build is fine without CUDA
    fi
fi
if $TORCH_CUDA_OK; then
    echo "PyTorch already installed and CUDA-compatible ($(python3 -c 'import torch; print(torch.__version__)') / cu${TORCH_CU_MAJOR}), skipping."
elif [ -z "$CUDA_MAJOR" ]; then
    echo "No CUDA found. Installing CPU-only PyTorch."
    pip install torch torchvision torchaudio
elif [ "$CUDA_MAJOR" -ge 12 ]; then
    echo "CUDA $CUDA_MAJOR detected -> installing PyTorch 2.6.0+cu124."
    pip install "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "torchaudio==2.6.0+cu124" \
        --index-url https://download.pytorch.org/whl/cu124
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

# ── Step 3c: HDF5 system library + h5py binary wheel ─────────────────────────
# h5py 3.9.0 has no Python 3.12 wheel; building from source requires libhdf5-dev.
# Pre-install a binary wheel here so requirements.txt sees h5py as satisfied.
echo "=== Step 3c: Pre-install h5py ==="
if command -v apt-get &>/dev/null; then
    apt-get install -y --no-install-recommends libhdf5-dev 2>/dev/null \
        && echo "  libhdf5-dev installed." \
        || echo "  Warning: apt-get libhdf5-dev failed — trying binary wheel only."
fi
pip install --prefer-binary 'h5py>=3.9.0'

# ── Step 3d: Pre-install C-extension build deps ───────────────────────────────
# pip does not guarantee install order within requirements.txt, so Cython/numpy
# may not be present when pycocotools or other Cython extensions are compiled.
# Install them explicitly here so the build environment is ready.
echo "=== Step 3d: Pre-install build dependencies (Cython, numpy, pybind11) ==="
pip install "numpy>=1.23" "Cython>=3.0" "pybind11>=2.12.0"

# ── Step 3e: Pre-install problematic C-extension packages ─────────────────────
# pycocotools>=2.0.7 no longer ships pre-generated _mask.c; it must be produced
# by Cython at build time.  Installing it separately (after Cython is present)
# avoids the "No such file or directory: pycocotools/_mask.c" error.
#
# fasttext and mmcv-full require a working C++ toolchain + CUDA headers that
# may not be present.  Use pre-built wheels where possible:
#   • fasttext-wheel — pure-Python wrapper with bundled native binary
#   • mmcv-full      — pull from OpenMMLab's wheel index matching torch+CUDA ver
echo "=== Step 3e: Pre-install pycocotools, fasttext, mmcv-full ==="

# pycocotools: build from source now that Cython is available
pip install --no-build-isolation --no-cache-dir "pycocotools>=2.0.7"

# fasttext: fasttext-wheel provides the same 'import fasttext' API as prebuilt binary.
# Track success so we can skip fasttext==0.9.2 in requirements.txt (which fails source build).
FASTTEXT_INSTALLED=false
if pip install fasttext-wheel 2>/dev/null; then
    FASTTEXT_INSTALLED=true
    echo "  fasttext-wheel installed."
elif pip install --no-build-isolation fasttext 2>/dev/null; then
    FASTTEXT_INSTALLED=true
    echo "  fasttext installed from source."
else
    echo "  Warning: fasttext installation failed — will be skipped in requirements.txt."
fi

# mmcv-full: try prebuilt wheels across torch/CUDA version combos, then fall
# back to a no-ops CPU-only build (MMCV_WITH_OPS=0) so the install doesn't abort.
TORCH_VER=$(python3 -c "import torch; v=torch.__version__.split('+')[0]; parts=v.split('.'); print(parts[0]+'.'+parts[1])")
CUDA_VER=$(python3 -c "import torch; cv=torch.version.cuda or ''; parts=cv.split('.'); print('cu'+''.join(parts[:2]) if cv else 'cpu')")
echo "  Detected: torch=${TORCH_VER}, cuda_tag=${CUDA_VER}"

MMCV_INSTALLED=false
for TF in "$TORCH_VER" "2.2" "2.1" "2.0" "1.13"; do
    for CV in "$CUDA_VER" "cu121" "cu118" "cu117"; do
        MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/${CV}/torch${TF}/index.html"
        if pip install "mmcv-full==1.6.0" -f "$MMCV_INDEX" 2>/dev/null; then
            MMCV_INSTALLED=true
            echo "  mmcv-full installed via ${CV}/torch${TF} index."
            break 2
        fi
    done
done

if ! $MMCV_INSTALLED; then
    # mmcv-full 1.x source build fails against torch 2.x headers (c10::bit_cast removed).
    # Fall back to mmcv==1.6.0 (same Python API, no CUDA ops) which installs from PyPI
    # as a pure-Python wheel. The codebase only needs mmcv.utils.ConfigDict.
    echo "  No prebuilt mmcv-full wheel found. Falling back to mmcv==1.6.0 (no CUDA ops)..."
    if pip install "mmcv==1.6.0" --no-build-isolation 2>/dev/null; then
        MMCV_INSTALLED=true
        echo "  mmcv==1.6.0 installed (no CUDA ops — sufficient for ConfigDict usage)."
    else
        echo "  Warning: mmcv installation failed — mmcv-full will be skipped in requirements.txt."
    fi
fi

# ── Step 4: requirements.txt ─────────────────────────────────────────────────
# --no-build-isolation: lets pip reuse already-installed torch/numpy during
# C-extension builds (pycocotools, etc.).
# Filter out packages already handled above to prevent failed source rebuilds.
echo "=== Step 4: Install requirements ==="
TEMP_REQS=$(mktemp --suffix=.txt)
cp requirements.txt "$TEMP_REQS"
if $FASTTEXT_INSTALLED; then sed -i '/^fasttext==/d' "$TEMP_REQS"; fi
# mmcv-full 1.x cannot compile against torch 2.x headers (c10::bit_cast removed).
# Always remove from requirements.txt; use whatever was pre-installed above (if any).
sed -i '/^mmcv-full==/d' "$TEMP_REQS"
# Step 1 already installed a CUDA-matched torch. Prevent requirements.txt from
# downgrading it to the pinned 1.11.0 (cu102), which would break step 5's build.
sed -i '/^torch==/d;/^torchvision==/d;/^torchaudio==/d' "$TEMP_REQS"
pip install --no-build-isolation --no-cache-dir -r "$TEMP_REQS"
rm -f "$TEMP_REQS"

# ── Step 4.5: Symlink nvidia pip-package headers into CUDA include dir ────────
# Vast.ai instances ship CUDA 13.x runtime-only (no full toolkit headers).
# PyTorch pip wheels bundle headers like cusparse.h inside nvidia-* packages,
# but NVCC only searches /usr/local/cuda/include — so detectron2's C++ build
# fails with "fatal error: cusparse.h: No such file or directory".
# Fix: symlink every header from the installed nvidia packages into CUDA include.
echo "=== Step 4.5: Symlink nvidia package headers into CUDA include ==="
CUDA_INC=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), '../../nvidia'))" 2>/dev/null)
if [ -z "$CUDA_INC" ]; then
    echo "Could not locate nvidia packages, skipping."
else
    for inc_dir in "$CUDA_INC"/*/include; do
        [ -d "$inc_dir" ] || continue
        for header in "$inc_dir"/*.h "$inc_dir"/*.hpp; do
            [ -f "$header" ] || continue
            dest="/usr/local/cuda/include/$(basename "$header")"
            [ -e "$dest" ] || ln -s "$header" "$dest"
        done
    done
    echo "Done symlinking headers from $CUDA_INC"
fi

# ── Step 5: detectron2 ───────────────────────────────────────────────────────
# _C.cpython-310-x86_64-linux-gnu.so is already compiled in the repo root.
# Rebuilding it fails due to pybind11 version conflict between pip-installed
# pybind11 and torch's bundled headers.  Instead, register the repo path via a
# .pth file so Python finds detectron2 (and the existing _C.so) without any build.
echo "=== Step 5: Register detectron2 via .pth (pre-compiled _C.so reused) ==="
SITE_PKGS=$(python3 -c "import site; print(site.getsitepackages()[0])")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$REPO_ROOT" > "$SITE_PKGS/detectron2-dev.pth"
echo "  Registered: $REPO_ROOT -> $SITE_PKGS/detectron2-dev.pth"

echo ""
echo "=== Done! ==="
python3 -c "import torch, detectron2; print('torch', torch.__version__); print('detectron2 OK')"
