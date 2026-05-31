#!/usr/bin/env bash
# =============================================================================
# Exp 20: Swin-T backbone robustness checks for Sol-B + EWC
#
# Purpose:
#   Build the fair Swin-T comparison table, collect drift diagnostics, and run
#   the fixed/adaptive alpha and lambda follow-ups needed for the paper claim.
#
# Usage:
#   bash scripts/exp20_swint_backbone_plan.sh [CHECKPOINT_PATH]
#
# Optional env:
#   EXP20_RUNS="whw solb_ewc alpha adaptive_alpha"   default core comparison
#   EXP20_LOG_PERIOD=10                              drift log interval
#   EXP20_ALPHA_VALUES="0.2 0.3 0.4 0.5 0.6"
#   EXP20_LAMBDA_VALUES="1.0 3.0 10.0 30.0"
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_swinT_coco.pth}"
PYTHON_BIN="${PYTHON:-python3}"
CFG="../configs/TTA/COCO_swinT.yaml"
BASE_CFG="../configs/Base/COCO_faster_rcnn_swinT_FPN_1x.yaml"
STATS_PATH="../models/stats/COCO_swinT_stats.pt"
FISHER_PATH="../models/stats/COCO_swinT_fisher.pt"

EXP20_RUNS="${EXP20_RUNS:-adaptive_alpha adaptive_alpha_lambda}"
EXP20_LOG_PERIOD="${EXP20_LOG_PERIOD:-10}"
EXP20_ALPHA_VALUES="${EXP20_ALPHA_VALUES:-0.2 0.3 0.4 0.5 0.6}"
EXP20_LAMBDA_VALUES="${EXP20_LAMBDA_VALUES:-1.0 3.0 10.0 30.0}"
EXP20_BASE_ALPHA="${EXP20_BASE_ALPHA:-0.4}"
EXP20_BASE_LAMBDA="${EXP20_BASE_LAMBDA:-10.0}"

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/datasets}"
mkdir -p ../results/exp20

has_run() {
    local key="$1"
    [[ " $EXP20_RUNS " == *" $key "* ]]
}

tag_float() {
    echo "$1" | tr '.' '_'
}

collect() {
    local tag="$1"
    local out="$2"
    cp "${out}/eval_matrix/metrics.json" "../results/exp20/metrics_${tag}.json" 2>/dev/null || true
    cp "${out}/eval_matrix/run_meta.json" "../results/exp20/meta_${tag}.json" 2>/dev/null || true
    cp "${out}/drift/drift_log.jsonl" "../results/exp20/drift_${tag}.jsonl" 2>/dev/null || true
}

ensure_fisher() {
    if [ ! -f "$FISHER_PATH" ]; then
        echo "=== Pre-step: computing Swin-T Fisher for adapter EWC ==="
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
    TEST.ADAPTATION.PROTO_METHOD "baseline"
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA_ADAPTIVE False
    TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE False
    TEST.ADAPTATION.DRIFT_LOG True
    TEST.ADAPTATION.DRIFT_LOG_PERIOD "$EXP20_LOG_PERIOD"
)

run_variant() {
    local key="$1"
    local tag="$2"
    local label="$3"
    shift 3
    if ! has_run "$key"; then
        echo "=== Skip ${key}/${tag}: not in EXP20_RUNS ==="
        return
    fi
    local out="../outputs/COCO_TTA/exp20_${tag}"
    echo ""
    echo "=== ${key}: ${label} ==="
    "$PYTHON_BIN" train_net.py "${COMMON_ARGS[@]}" "$@" OUTPUT_DIR "$out"
    collect "$tag" "$out"
}

if has_run "direct"; then
    out="../outputs/COCO_TTA/exp20_direct"
    echo ""
    echo "=== direct: source model without online adaptation ==="
    "$PYTHON_BIN" train_net.py \
        --config-file "$BASE_CFG" \
        --eval-only \
        MODEL.WEIGHTS "$CKPT" \
        TEST.CONTINUAL_DOMAIN True \
        TEST.ONLINE_ADAPTATION False \
        OUTPUT_DIR "$out"
    collect direct "$out"
fi

if [[ " $EXP20_RUNS " == *" ewc "* || " $EXP20_RUNS " == *" solb_ewc "* || " $EXP20_RUNS " == *" alpha "* || " $EXP20_RUNS " == *" lambda "* || " $EXP20_RUNS " == *" adaptive_alpha"* ]]; then
    ensure_fisher
fi

run_variant whw whw \
    "WHW-style Ours: baseline prototype + full backward" \
    TEST.ONLINE_ADAPTATION True \
    TEST.ADAPTATION.SKIP_REDUNDANT None \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

run_variant whw_skip whw_skip \
    "WHW-style Ours-Skip: skip redundant updates" \
    TEST.ONLINE_ADAPTATION True \
    TEST.ADAPTATION.SKIP_REDUNDANT "stat-period-ema" \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

run_variant ewc ewc \
    "EWC only: baseline prototype + adapter EWC" \
    TEST.ONLINE_ADAPTATION True \
    TEST.ADAPTATION.SKIP_REDUNDANT None \
    TEST.ADAPTATION.PROTO_METHOD "baseline" \
    TEST.ADAPTATION.EWC_LAMBDA "$EXP20_BASE_LAMBDA"

run_variant solb solb \
    "Sol-B only: source anchor alpha=${EXP20_BASE_ALPHA}" \
    TEST.ONLINE_ADAPTATION True \
    TEST.ADAPTATION.SKIP_REDUNDANT None \
    TEST.ADAPTATION.PROTO_METHOD "dual_memory" \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$EXP20_BASE_ALPHA" \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

run_variant solb_ewc solb_ewc \
    "Sol-B + EWC: alpha=${EXP20_BASE_ALPHA}, lambda=${EXP20_BASE_LAMBDA}" \
    TEST.ONLINE_ADAPTATION True \
    TEST.ADAPTATION.SKIP_REDUNDANT None \
    TEST.ADAPTATION.PROTO_METHOD "dual_memory" \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$EXP20_BASE_ALPHA" \
    TEST.ADAPTATION.EWC_LAMBDA "$EXP20_BASE_LAMBDA"

if has_run "alpha"; then
    for alpha in $EXP20_ALPHA_VALUES; do
        atag="$(tag_float "$alpha")"
        run_variant alpha "alpha_${atag}" \
            "fixed alpha sweep: alpha=${alpha}, lambda=${EXP20_BASE_LAMBDA}" \
            TEST.ONLINE_ADAPTATION True \
            TEST.ADAPTATION.SKIP_REDUNDANT None \
            TEST.ADAPTATION.PROTO_METHOD "dual_memory" \
            TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$alpha" \
            TEST.ADAPTATION.EWC_LAMBDA "$EXP20_BASE_LAMBDA"
    done
fi

if has_run "lambda"; then
    for lambda in $EXP20_LAMBDA_VALUES; do
        ltag="$(tag_float "$lambda")"
        run_variant lambda "lambda_${ltag}" \
            "lambda sweep: alpha=${EXP20_BASE_ALPHA}, lambda=${lambda}" \
            TEST.ONLINE_ADAPTATION True \
            TEST.ADAPTATION.SKIP_REDUNDANT None \
            TEST.ADAPTATION.PROTO_METHOD "dual_memory" \
            TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$EXP20_BASE_ALPHA" \
            TEST.ADAPTATION.EWC_LAMBDA "$lambda"
    done
fi

run_variant adaptive_alpha adaptive_alpha \
    "adaptive alpha: confidence/count-gated source anchor" \
    TEST.ONLINE_ADAPTATION True \
    TEST.ADAPTATION.SKIP_REDUNDANT None \
    TEST.ADAPTATION.PROTO_METHOD "dual_memory" \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$EXP20_BASE_ALPHA" \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA_ADAPTIVE True \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA_MIN 0.2 \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA_MAX 0.6 \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA_COUNT_REF 32.0 \
    TEST.ADAPTATION.EWC_LAMBDA "$EXP20_BASE_LAMBDA"

run_variant adaptive_alpha_lambda adaptive_alpha_lambda \
    "adaptive alpha + pressure-adaptive EWC lambda" \
    TEST.ONLINE_ADAPTATION True \
    TEST.ADAPTATION.SKIP_REDUNDANT None \
    TEST.ADAPTATION.PROTO_METHOD "dual_memory" \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$EXP20_BASE_ALPHA" \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA_ADAPTIVE True \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA_MIN 0.2 \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA_MAX 0.6 \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA_COUNT_REF 32.0 \
    TEST.ADAPTATION.EWC_LAMBDA "$EXP20_BASE_LAMBDA" \
    TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE True \
    TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE_MODE "pressure" \
    TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE_BETA 1.0 \
    TEST.ADAPTATION.EWC_LAMBDA_MIN_SCALE 0.1

"$PYTHON_BIN" summarize_exp20_swint.py --results-dir ../results/exp20
