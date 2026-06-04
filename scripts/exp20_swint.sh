#!/usr/bin/env bash
# =============================================================================
# Exp 20: Swin-T Sol-B + EWC alpha/lambda pair runs
#
# Purpose:
#   Run selected Swin-T COCO -> COCO-C continual adaptation experiments with only
#   two knobs exposed per run:
#     - SOURCE_ANCHOR_ALPHA for Sol-B dual memory
#     - EWC_LAMBDA for adapter EWC
#
# Usage:
#   bash scripts/exp20_swint.sh [ALPHA:LAMBDA ...]
#
# Examples:
#   bash scripts/exp20_swint.sh 0.4:10.0
#   bash scripts/exp20_swint.sh 0.4:10.0 0.5:3.0 0.6:1.0
#   EXP20_PAIRS="0.4:10.0 0.5:3.0" bash scripts/exp20_swint.sh
#
# Optional env:
#   EXP20_PAIRS="0.4:10.0"   pair set used when CLI pairs are omitted
#   EXP20_DRIFT_LOG=True     enable drift diagnostics
#   EXP20_LOG_PERIOD=10      drift log interval when enabled
#   CKPT=...                 checkpoint path override
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

PYTHON_BIN="${PYTHON:-python3}"
CFG="../configs/TTA/COCO_swinT.yaml"
STATS_PATH="../models/stats/COCO_swinT_stats.pt"
FISHER_PATH="../models/stats/COCO_swinT_fisher.pt"

CKPT="${CKPT:-../models/checkpoints/faster_rcnn_swinT_coco.pth}"
EXP20_DRIFT_LOG="${EXP20_DRIFT_LOG:-False}"
EXP20_LOG_PERIOD="${EXP20_LOG_PERIOD:-10}"

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/datasets}"
mkdir -p ../results/exp20

if [ "$#" -gt 0 ]; then
    # Backward-compatible convenience: `script 0.4 10.0` means one pair.
    if [ "$#" -eq 2 ] && [[ "$1" != *:* ]] && [[ "$2" != *:* ]]; then
        PAIRS=("$1:$2")
    else
        PAIRS=("$@")
    fi
elif [ -n "${EXP20_PAIRS:-}" ]; then
    PAIRS=()
    for pair in ${EXP20_PAIRS//,/ }; do
        PAIRS+=("$pair")
    done
else
    PAIRS=("0.4:1.0")
fi

validate_float() {
    local name="$1"
    local value="$2"
    "$PYTHON_BIN" -c 'import sys; float(sys.argv[1])' "$value" 2>/dev/null || {
        echo "Invalid ${name}: ${value}"
        exit 1
    }
}

lambda_needs_fisher() {
    "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) > 0.0 else 1)' "$1"
}

tag_float() {
    echo "$1" | tr '.' '_'
}

parse_pair() {
    local pair="$1"
    if [[ "$pair" != *:* ]]; then
        echo "Invalid pair '${pair}'. Use ALPHA:LAMBDA, e.g. 0.4:10.0"
        exit 1
    fi
    EXP20_ALPHA="${pair%%:*}"
    EXP20_LAMBDA="${pair##*:}"
    validate_float "alpha" "$EXP20_ALPHA"
    validate_float "lambda" "$EXP20_LAMBDA"
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

for pair in "${PAIRS[@]}"; do
    parse_pair "$pair"
    if lambda_needs_fisher "$EXP20_LAMBDA"; then
        ensure_fisher
        break
    fi
done

run_pair() {
    local pair="$1"
    parse_pair "$pair"

    local alpha_tag
    local lambda_tag
    local tag
    local out
    local metrics_out
    local meta_out
    local drift_out
    local summary_out
    alpha_tag="$(tag_float "$EXP20_ALPHA")"
    lambda_tag="$(tag_float "$EXP20_LAMBDA")"
    tag="alpha${alpha_tag}_lam${lambda_tag}"
    out="../outputs/COCO_TTA/exp20_${tag}"
    metrics_out="../results/exp20/metrics_${tag}.json"
    meta_out="../results/exp20/meta_${tag}.json"
    drift_out="../results/exp20/drift_${tag}.jsonl"
    summary_out="../results/exp20/summary_${tag}.json"

    echo ""
    echo "=== Exp20 Swin-T Sol-B + EWC ==="
    echo "alpha=${EXP20_ALPHA}, lambda=${EXP20_LAMBDA}"
    echo "output=${out}"

    rm -f "$metrics_out" "$meta_out" "$drift_out" "$summary_out"

    "$PYTHON_BIN" train_net.py \
        --config-file "$CFG" \
        --eval-only \
        MODEL.WEIGHTS "$CKPT" \
        TEST.ONLINE_ADAPTATION True \
        TEST.CONTINUAL_DOMAIN True \
        TEST.EVAL_MATRIX False \
        TEST.ADAPTATION.CONTINUAL True \
        TEST.ADAPTATION.WHERE "adapter" \
        TEST.ADAPTATION.GLOBAL_ALIGN "KL" \
        TEST.ADAPTATION.FOREGROUND_ALIGN "KL" \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        TEST.ADAPTATION.EWC_FISHER_PATH "$FISHER_PATH" \
        TEST.ADAPTATION.PROTO_METHOD "dual_memory" \
        TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$EXP20_ALPHA" \
        TEST.ADAPTATION.EWC_LAMBDA "$EXP20_LAMBDA" \
        TEST.ADAPTATION.EMA_BETA 0.0 \
        TEST.ADAPTATION.SWEMA_K 0 \
        TEST.ADAPTATION.ASRI_ALPHA 0.0 \
        TEST.ADAPTATION.ADAPTER_RESET False \
        TEST.ADAPTATION.ORACLE_PROTOTYPE False \
        TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA_ADAPTIVE False \
        TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE False \
        TEST.ADAPTATION.DRIFT_LOG "$EXP20_DRIFT_LOG" \
        TEST.ADAPTATION.DRIFT_LOG_PERIOD "$EXP20_LOG_PERIOD" \
        OUTPUT_DIR "$out"

    cp "${out}/eval_matrix/metrics.json" "$metrics_out" 2>/dev/null || true
    cp "${out}/eval_matrix/run_meta.json" "$meta_out" 2>/dev/null || true
    cp "${out}/drift/drift_log.jsonl" "$drift_out" 2>/dev/null || true

    if [ ! -f "$metrics_out" ]; then
        echo "Missing metrics: ${out}/eval_matrix/metrics.json"
        exit 1
    fi

    "$PYTHON_BIN" - "$metrics_out" "$summary_out" "$tag" "$EXP20_ALPHA" "$EXP20_LAMBDA" <<'PYEOF'
import json
import math
import sys

metrics_path, summary_path, tag, alpha, lam = sys.argv[1:6]

corruptions = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)

def get_ap(metrics, domain):
    key = "coco_2017_val" if domain == "original" else f"coco_2017_val-{domain}"
    val = metrics.get(key)
    if isinstance(val, dict) and isinstance(val.get("bbox"), dict):
        val = val["bbox"]
    if isinstance(val, dict) and finite(val.get("AP")):
        return float(val["AP"])
    return None

with open(metrics_path) as fp:
    metrics = json.load(fp)

per_domain = {d: get_ap(metrics, d) for d in corruptions}
clean = get_ap(metrics, "original")
vals15 = [v for v in per_domain.values() if finite(v)]
vals16 = vals15 + ([clean] if finite(clean) else [])
summary = {
    "tag": tag,
    "alpha": float(alpha),
    "lambda": float(lam),
    "AP15": sum(vals15) / len(vals15) if vals15 else None,
    "AP16": sum(vals16) / len(vals16) if vals16 else None,
    "clean": clean,
    "per_domain_AP": per_domain,
}

with open(summary_path, "w") as fp:
    json.dump(summary, fp, indent=2)

def fmt(v):
    return f"{v:.4f}" if finite(v) else "n/a"

print("")
print("=== Exp20 Result ===")
print(f"tag={tag}")
print(f"AP15={fmt(summary['AP15'])}  AP16={fmt(summary['AP16'])}  clean={fmt(clean)}")
print(f"saved={summary_path}")
PYEOF
}

for pair in "${PAIRS[@]}"; do
    run_pair "$pair"
done

echo ""
echo "=== Exp20 complete. Results in results/exp20/ ==="
