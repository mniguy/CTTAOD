#!/usr/bin/env bash
# =============================================================================
# Exp 1: Forgetting Cause Diagnosis
#
# Runs three variants to isolate the cause of catastrophic forgetting:
#
#   Variant A — Adapter Reset + EMA Prototype (isolates prototype drift)
#     - Reset adapter weights at each domain boundary
#     - Keep EMA prototype running (no injection)
#     → If BWT still bad: prototype drift is the main cause → ASRI is correct
#
#   Variant B — Adapter Continual + Oracle Prototype (isolates adapter drift)
#     - Let adapters accumulate across domains (normal CTTAOD)
#     - Replace target prototype with source prototype (ASRI_ALPHA=1.0)
#     → If BWT still bad: adapter weight drift is the main cause
#
#   Baseline — from Exp 0 (already computed, just reference)
#
# Usage:
#   bash scripts/exp1_diagnosis.sh [CHECKPOINT_PATH]
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_R50_cityscapes.pth}"
CFG="../configs/TTA/Cityscapes_R50.yaml"
STATS_PATH="../models/stats/Cityscapes_R50_stats.pt"

mkdir -p ../results/exp1

# ─────────────────────────────────────────────────────────────────────────────
# Variant A: Adapter Reset + EMA Prototype (no injection, adapter resets)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Exp 1 Variant A: Adapter Reset + EMA Prototype ==="
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
    TEST.ADAPTATION.ORACLE_PROTOTYPE False \
    TEST.ADAPTATION.ADAPTER_RESET True \
    TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
    OUTPUT_DIR ../outputs/Cityscapes_TTA/exp1_varA

cp ../outputs/Cityscapes_TTA/exp1_varA/eval_matrix/eval_matrix.npy \
   ../results/exp1/eval_matrix_varA.npy 2>/dev/null || true
cp ../outputs/Cityscapes_TTA/exp1_varA/eval_matrix/metrics.json \
   ../results/exp1/metrics_varA.json 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Variant B: Adapter Continual + Oracle Prototype (ASRI_ALPHA=1.0)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Exp 1 Variant B: Adapter Continual + Oracle Prototype (ASRI α=1.0) ==="
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
    TEST.ADAPTATION.ASRI_ALPHA 1.0 \
    TEST.ADAPTATION.ORACLE_PROTOTYPE True \
    TEST.ADAPTATION.ADAPTER_RESET False \
    TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
    OUTPUT_DIR ../outputs/Cityscapes_TTA/exp1_varB

cp ../outputs/Cityscapes_TTA/exp1_varB/eval_matrix/eval_matrix.npy \
   ../results/exp1/eval_matrix_varB.npy 2>/dev/null || true
cp ../outputs/Cityscapes_TTA/exp1_varB/eval_matrix/metrics.json \
   ../results/exp1/metrics_varB.json 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Print diagnosis summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Exp 1 Diagnosis Summary ==="
echo ""
echo "--- Baseline CTTAOD ---"
cat ../results/exp0/metrics_baseline.json 2>/dev/null || echo "(run exp0 first)"
echo ""
echo "--- Variant A (Adapter Reset) ---"
cat ../results/exp1/metrics_varA.json 2>/dev/null || true
echo ""
echo "--- Variant B (Oracle Prototype) ---"
cat ../results/exp1/metrics_varB.json 2>/dev/null || true
echo ""
echo "Interpretation:"
echo "  If VarA BWT << 0 but VarB BWT ≈ 0  → Prototype drift is the main cause → ASRI is correct"
echo "  If VarA BWT ≈ 0 but VarB BWT << 0  → Adapter interference is the main cause"
echo "  If both bad                         → Both contribute"