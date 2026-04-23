#!/usr/bin/env bash
# =============================================================================
# Exp 0: Evaluation Protocol Setup (COCO → COCO-C)
#
# Runs three sub-experiments in order:
#   (1) Collect source feature statistics from COCO val
#   (2) Source-only baseline (no adaptation) evaluated on all 15 corruptions
#   (3) Baseline CTTAOD: build full (T+1)×T evaluation matrix
#
# Usage:
#   bash scripts/exp0_coco.sh [CHECKPOINT_PATH]
#
# CHECKPOINT_PATH defaults to ./models/checkpoints/faster_rcnn_r50_coco.pth
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"

mkdir -p ../models/stats ../models/checkpoints ../results/exp0

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Collect source feature statistics (COCO val, clean)
# Skip this step if COCO_R50_stats.pt already exists (e.g. downloaded from Drive)
# ─────────────────────────────────────────────────────────────────────────────
if [ -f "$STATS_PATH" ]; then
    echo "=== Step 1: Source stats already exist at $STATS_PATH — skipping collection ==="
else
    echo "=== Step 1: Collecting source feature statistics ==="
    python train_net.py \
        --config-file "$CFG" \
        --eval-only \
        MODEL.WEIGHTS "$CKPT" \
        TEST.ONLINE_ADAPTATION False \
        TEST.CONTINUAL_DOMAIN False \
        TEST.COLLECT_FEATURES True \
        TEST.EVAL_MATRIX False \
        OUTPUT_DIR ../outputs/COCO_TTA/exp0_collect_stats

    CKPT_BASE=$(basename "$CKPT" .pth)
    mv "../models/${CKPT_BASE}_feature_stats_new.pt" "$STATS_PATH" 2>/dev/null || true
    echo "Source stats saved to $STATS_PATH"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Source-only baseline (no adaptation)
#   Evaluates frozen source model on all 15 COCO-C corruptions.
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
    OUTPUT_DIR ../outputs/COCO_TTA/exp0_source_only

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
    OUTPUT_DIR ../outputs/COCO_TTA/exp0_baseline

cp ../outputs/COCO_TTA/exp0_baseline/eval_matrix/eval_matrix.npy \
   ../results/exp0/eval_matrix_baseline.npy 2>/dev/null || true
cp ../outputs/COCO_TTA/exp0_baseline/eval_matrix/eval_matrix_per_class.npy \
   ../results/exp0/eval_matrix_baseline_per_class.npy 2>/dev/null || true
cp ../outputs/COCO_TTA/exp0_baseline/eval_matrix/metrics.json \
   ../results/exp0/metrics_baseline.json 2>/dev/null || true
cp ../outputs/COCO_TTA/exp0_source_only/eval_matrix/eval_matrix.npy \
   ../results/exp0/eval_matrix_source_only.npy 2>/dev/null || true

echo ""
echo "=== Exp 0 complete. Results in results/exp0/ ==="
cat ../results/exp0/metrics_baseline.json 2>/dev/null || true