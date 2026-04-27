#!/bin/bash
# Run the CTTAOD container on a Vast.ai instance (or any CUDA host).
# Usage: bash docker/run_vastai.sh [image] [data_dir] [models_dir]
#
# Defaults:
#   image      = cttaod:latest  (or set CTTAOD_IMAGE env var)
#   data_dir   = $PWD/datasets
#   models_dir = $PWD/models
#
# The container mounts:
#   /workspace/datasets  ← your dataset directory
#   /workspace/models    ← checkpoints & source stats
#   /workspace/outputs   ← experiment outputs (created if absent)

set -e

IMAGE=${1:-${CTTAOD_IMAGE:-cttaod:latest}}
DATA_DIR=${2:-$PWD/datasets}
MODELS_DIR=${3:-$PWD/models}
OUTPUTS_DIR=$PWD/outputs

mkdir -p "$OUTPUTS_DIR"

echo "=== Launching $IMAGE ==="
echo "  datasets → $DATA_DIR"
echo "  models   → $MODELS_DIR"
echo "  outputs  → $OUTPUTS_DIR"

docker run --gpus all -it --ipc=host \
  --shm-size=8g \
  -v "$DATA_DIR":/workspace/datasets \
  -v "$MODELS_DIR":/workspace/models \
  -v "$OUTPUTS_DIR":/workspace/outputs \
  "$IMAGE" \
  bash
