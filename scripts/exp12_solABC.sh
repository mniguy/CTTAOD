#!/usr/bin/env bash
# =============================================================================
# Exp 12: Sol A/B/C ported from ContinualTTA_ObjectDetection
#
# Methods:
#   E0  baseline                — no proto_method override (CTTAOD default path)
#   E1  Sol-A reset             — cosine-sim drop triggers prototype reset to source anchor
#         sweep SWITCH_COSIM_THR ∈ {0.20, 0.30, 0.40}
#   E2  Sol-B dual_memory       — KL t_dist mean blended with frozen source anchor
#         sweep SOURCE_ANCHOR_ALPHA ∈ {0.1, 0.3, 0.5}
#   E3  Sol-C adaptive_gamma    — cosine-scaled gamma, mean-based EMA, clamped
#         (no extra sweep — relies on existing EMA_GAMMA)
#
# Reference base: COCO R50, fg+global KL, legacy gamma EMA path (EMA_BETA=0,
# SWEMA_K=0, ASRI_ALPHA=0) so proto_method branch is the only active update rule.
#
# Outputs:
#   results/exp12/metrics_<tag>.json
#   results/exp12/eval_matrix_<tag>.npy
#
# Usage:
#   bash scripts/exp12_solABC.sh [CHECKPOINT_PATH]
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/datasets}"
mkdir -p ../results/exp12

BASE_ARGS=(
    --config-file "$CFG"
    --eval-only
    MODEL.WEIGHTS "$CKPT"
    TEST.ONLINE_ADAPTATION True
    TEST.CONTINUAL_DOMAIN True
    TEST.EVAL_MATRIX True
    TEST.ADAPTATION.CONTINUAL True
    TEST.ADAPTATION.WHERE "adapter"
    TEST.ADAPTATION.GLOBAL_ALIGN "KL"
    TEST.ADAPTATION.FOREGROUND_ALIGN "KL"
    TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH"
    TEST.ADAPTATION.EMA_BETA 0.0
    TEST.ADAPTATION.SWEMA_K 0
    TEST.ADAPTATION.ASRI_ALPHA 0.0
    TEST.ADAPTATION.ADAPTER_RESET False
    TEST.ADAPTATION.ORACLE_PROTOTYPE False
)

collect() {
    local TAG="$1"
    local OUT="$2"
    cp "${OUT}/eval_matrix/metrics.json"    "../results/exp12/metrics_${TAG}.json"    2>/dev/null || true
    cp "${OUT}/eval_matrix/eval_matrix.npy" "../results/exp12/eval_matrix_${TAG}.npy" 2>/dev/null || true
}

# ─────────────────────────────────────────────────────────────────────────────
# E0 : baseline
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== E0 : baseline (proto_method=baseline) ==="
OUT="../outputs/COCO_TTA/exp12_e0_baseline"
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.PROTO_METHOD "baseline" \
    OUTPUT_DIR "$OUT"
collect "e0_baseline" "$OUT"

# ─────────────────────────────────────────────────────────────────────────────
# E1 : Sol-A reset — sweep SWITCH_COSIM_THR
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== E1 : Sol-A reset ==="
for THR in 0.20 0.30 0.40; do
    THR_TAG=$(echo "$THR" | sed 's/\./_/g')
    OUT="../outputs/COCO_TTA/exp12_e1_solA_thr${THR_TAG}"
    echo "  SWITCH_COSIM_THR = $THR"
    python train_net.py "${BASE_ARGS[@]}" \
        TEST.ADAPTATION.PROTO_METHOD "reset" \
        TEST.ADAPTATION.SWITCH_COSIM_THR "$THR" \
        OUTPUT_DIR "$OUT"
    collect "e1_solA_thr${THR_TAG}" "$OUT"
done

# ─────────────────────────────────────────────────────────────────────────────
# E2 : Sol-B dual_memory — sweep SOURCE_ANCHOR_ALPHA
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== E2 : Sol-B dual_memory ==="
for A in 0.1 0.3 0.5; do
    A_TAG=$(echo "$A" | sed 's/\./_/g')
    OUT="../outputs/COCO_TTA/exp12_e2_solB_a${A_TAG}"
    echo "  SOURCE_ANCHOR_ALPHA = $A"
    python train_net.py "${BASE_ARGS[@]}" \
        TEST.ADAPTATION.PROTO_METHOD "dual_memory" \
        TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$A" \
        OUTPUT_DIR "$OUT"
    collect "e2_solB_a${A_TAG}" "$OUT"
done

# ─────────────────────────────────────────────────────────────────────────────
# E3 : Sol-C adaptive_gamma
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== E3 : Sol-C adaptive_gamma ==="
OUT="../outputs/COCO_TTA/exp12_e3_solC"
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.PROTO_METHOD "adaptive_gamma" \
    OUTPUT_DIR "$OUT"
collect "e3_solC" "$OUT"

echo ""
echo "======================================================"
echo "  Exp 12 done. Results: ../results/exp12/"
echo "======================================================"
