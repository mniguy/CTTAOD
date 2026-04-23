#!/usr/bin/env bash
# =============================================================================
# Exp 0: Evaluation Protocol Setup (Cityscapes → Cityscapes-C)
#
# Runs three sub-experiments in order:
#   (1) Collect source feature statistics from Cityscapes train
#   (2) Source-only baseline (no adaptation) evaluated on all 15 corruptions
#   (3) Baseline CTTAOD: build full (T+1)×T evaluation matrix
#
# Usage:
#   bash scripts/exp0_cityscapes.sh [CHECKPOINT_PATH]
#
# CHECKPOINT_PATH defaults to ./models/checkpoints/faster_rcnn_R50_cityscapes.pth
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_R50_cityscapes.pth}"
CFG="../configs/TTA/Cityscapes_R50.yaml"
STATS_PATH="../models/stats/Cityscapes_R50_stats.pt"

mkdir -p ../models/stats ../models/checkpoints ../results/exp0

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Collect source feature statistics (Cityscapes train, clean)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Step 1: Collecting source feature statistics ==="
python train_net.py \
    --config-file "$CFG" \
    --eval-only \
    MODEL.WEIGHTS "$CKPT" \
    TEST.ONLINE_ADAPTATION False \
    TEST.CONTINUAL_DOMAIN False \
    TEST.COLLECT_FEATURES True \
    TEST.EVAL_MATRIX False \
    DATASETS.TEST '("cityscapes_det_train",)' \
    OUTPUT_DIR ../outputs/Cityscapes_TTA/exp0_collect_stats

# The stats file is saved to models/{checkpoint_name}_feature_stats_new.pt
# Move it to the expected path
CKPT_BASE=$(basename "$CKPT" .pth)
mv "../models/${CKPT_BASE}_feature_stats_new.pt" "$STATS_PATH" 2>/dev/null || true
echo "Source stats saved to $STATS_PATH"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Source-only baseline (no adaptation)
#   Evaluates frozen source model on all 15 Cityscapes-C corruptions.
#   This produces row-0 of the evaluation matrix.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Source-only baseline (no adaptation) ==="
python train_net.py \
    --config-file "$CFG" \
    --eval-only \
    MODEL.WEIGHTS "$CKPT" \
    TEST.ONLINE_ADAPTATION False \
    TEST.CONTINUAL_DOMAIN True \
    TEST.EVAL_MATRIX True \
    TEST.ADAPTATION.ASRI_ALPHA 0.0 \
    TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
    OUTPUT_DIR ../outputs/Cityscapes_TTA/exp0_source_only

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Baseline CTTAOD — full evaluation matrix
#   Adapter-only adaptation, continual, KL alignment, ASRI_ALPHA=0.0
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 3: Baseline CTTAOD — full evaluation matrix ==="
python train_net.py \
    --config-file "$CFG" \
    --eval-only \
    MODEL.WEIGHTS "$CKPT" \
    TEST.ONLINE_ADAPTATION True \
    TEST.CONTINUAL_DOMAIN True \
    TEST.EVAL_MATRIX True \
    TEST.ADAPTATION.CONTINUAL True \
    TEST.ADAPTATION.WHERE "adapter" \
    TEST.ADAPTATION.GLOBAL_ALIGN "KL" \
    TEST.ADAPTATION.FOREGROUND_ALIGN "KL" \
    TEST.ADAPTATION.ASRI_ALPHA 0.0 \
    TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
    OUTPUT_DIR ../outputs/Cityscapes_TTA/exp0_baseline

# Copy results to results/exp0/
cp ../outputs/Cityscapes_TTA/exp0_baseline/eval_matrix/eval_matrix.npy \
   ../results/exp0/eval_matrix_baseline.npy 2>/dev/null || true
cp ../outputs/Cityscapes_TTA/exp0_baseline/eval_matrix/eval_matrix_per_class.npy \
   ../results/exp0/eval_matrix_baseline_per_class.npy 2>/dev/null || true
cp ../outputs/Cityscapes_TTA/exp0_baseline/eval_matrix/metrics.json \
   ../results/exp0/metrics_baseline.json 2>/dev/null || true
cp ../outputs/Cityscapes_TTA/exp0_source_only/eval_matrix/eval_matrix.npy \
   ../results/exp0/eval_matrix_source_only.npy 2>/dev/null || true

echo ""
echo "=== Exp 0 complete. Results in results/exp0/ ==="
cat ../results/exp0/metrics_baseline.json 2>/dev/null || true