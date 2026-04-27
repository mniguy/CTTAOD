#!/bin/bash
# Build the Vast.ai-compatible Docker image.
# Usage: bash docker/build_vastai.sh [tag]
#
# The image is self-contained: PyTorch 2.5.1 + CUDA 12.4 + compiled detectron2.
# Push to Docker Hub or a registry so Vast.ai instances can pull it directly.
#
# Example push:
#   docker tag cttaod:latest <your-dockerhub>/cttaod:latest
#   docker push <your-dockerhub>/cttaod:latest

set -e

TAG=${1:-cttaod:latest}

# TORCH_CUDA_ARCH_LIST covers the most common Vast.ai GPU generations.
# Ampere (A100/A10/A6000): 8.0 / 8.6
# Ada (RTX 4090/4080):     8.9
# Hopper (H100):           9.0
# Add "7.5" for Turing (RTX 2080/T4) or "6.1" for Pascal (1080 Ti).
ARCH_LIST="8.0;8.6;8.9;9.0"

echo "=== Building image: $TAG  (TORCH_CUDA_ARCH_LIST=$ARCH_LIST) ==="

docker build \
  --build-arg TORCH_CUDA_ARCH_LIST="$ARCH_LIST" \
  -t "$TAG" \
  -f docker/Dockerfile.vastai \
  .

echo ""
echo "=== Build complete: $TAG ==="
echo "To push: docker push $TAG"
