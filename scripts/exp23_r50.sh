#!/usr/bin/env bash
# =============================================================================
# Exp 23: Alpha & Lambda Validation on ResNet-50
#
# Runs specified (alpha, lambda) pairs on ResNet-50 to verify results from
# exp22 (Swin-T) transfer to this backbone.
#
# Usage:
#   bash scripts/exp23_r50.sh [CHECKPOINT_PATH]
#
# Optional env:
#   EXP23_COMBINATIONS="0.4:0.1 0.5:0.1 0.3:0.1"   space-separated alpha:lambda pairs
#   EXP23_LOG_PERIOD=10
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
PYTHON_BIN="${PYTHON:-python3}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"
FISHER_PATH="../models/stats/COCO_R50_fisher.pt"

EXP23_COMBINATIONS="${EXP23_COMBINATIONS:-0.5:0.001 0.5:0.01 0.4:0.001 0.4:0.1}"
EXP23_LOG_PERIOD="${EXP23_LOG_PERIOD:-10}"

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/datasets}"
mkdir -p ../results/exp23

tag_float() {
    echo "$1" | tr '.' '_'
}

collect() {
    local tag="$1"
    local out="$2"
    cp "${out}/eval_matrix/metrics.json" "../results/exp23/metrics_${tag}.json" 2>/dev/null || true
    cp "${out}/eval_matrix/run_meta.json" "../results/exp23/meta_${tag}.json"   2>/dev/null || true
    cp "${out}/drift/drift_log.jsonl"     "../results/exp23/drift_${tag}.jsonl" 2>/dev/null || true
}

ensure_fisher() {
    if [ ! -f "$FISHER_PATH" ]; then
        echo "=== Pre-step: computing R50 Fisher for adapter EWC ==="
        "$PYTHON_BIN" compute_fisher.py \
            --config-file "$CFG" \
            MODEL.WEIGHTS "$CKPT" \
            TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
            TEST.ADAPTATION.WHERE "adapter"
    fi
}

COMMON_ARGS=(
    --config-file "$CFG"
    --eval-only
    MODEL.WEIGHTS "$CKPT"
    TEST.CONTINUAL_DOMAIN True
    TEST.ADAPTATION.CONTINUAL True
    TEST.ADAPTATION.WHERE "adapter"
    TEST.ADAPTATION.GLOBAL_ALIGN "KL"
    TEST.ADAPTATION.FOREGROUND_ALIGN "KL"
    TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH"
    TEST.ADAPTATION.EWC_FISHER_PATH "$FISHER_PATH"
    TEST.ADAPTATION.EMA_BETA 0.0
    TEST.ADAPTATION.SWEMA_K 0
    TEST.ADAPTATION.ASRI_ALPHA 0.0
    TEST.ADAPTATION.ADAPTER_RESET False
    TEST.ADAPTATION.ORACLE_PROTOTYPE False
    TEST.ADAPTATION.PROTO_METHOD "dual_memory"
    TEST.ADAPTATION.SKIP_REDUNDANT None
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA_ADAPTIVE False
    TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE False
    TEST.ADAPTATION.DRIFT_LOG True
    TEST.ADAPTATION.DRIFT_LOG_PERIOD "$EXP23_LOG_PERIOD"
    TEST.ONLINE_ADAPTATION True
)

ensure_fisher

echo ""
echo "=== exp23 R50: combinations=${EXP23_COMBINATIONS} ==="
for pair in $EXP23_COMBINATIONS; do
    alpha="${pair%%:*}"
    lambda="${pair##*:}"
    tag="alpha$(tag_float "$alpha")_lam$(tag_float "$lambda")"
    out="../outputs/COCO_TTA/exp23_${tag}"
    echo "--- alpha=${alpha}, lambda=${lambda} ---"
    "$PYTHON_BIN" train_net.py "${COMMON_ARGS[@]}" \
        TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$alpha" \
        TEST.ADAPTATION.EWC_LAMBDA "$lambda" \
        OUTPUT_DIR "$out"
    collect "$tag" "$out"
done

"$PYTHON_BIN" summarize_exp23_r50.py --results-dir ../results/exp23
