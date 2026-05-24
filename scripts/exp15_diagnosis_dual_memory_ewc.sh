#!/usr/bin/env bash
# =============================================================================
# Exp 15: Two-Coupled-Drift Diagnosis with Sol-B Dual Memory + EWC
#
# Purpose:
#   Repeat the exp14 diagnosis, but replace the prototype-side solution:
#     exp14: Sol-A prototype reset + EWC
#     exp15: Sol-B dual memory + EWC
#
# Runs:
#   D0  baseline                  normal prototype + continual adapter
#   D1  oracle prototype          source prototype in fg KL objective; adapter drifts
#   D2  adapter reset             adapter reset at each domain boundary; prototype drifts
#   D3  EWC only                  normal prototype + EWC(lambda=10) on adapter
#   D4  dual memory only          dual_memory(alpha) + no EWC
#   D5  current Sol-B candidate   dual_memory(alpha) + EWC(lambda=10)
#
# Outputs per run:
#   results/exp15/metrics_<tag>.json
#   results/exp15/eval_matrix_<tag>.npy
#   results/exp15/eval_matrix_per_class_<tag>.npy
#   results/exp15/drift_<tag>.jsonl
#
# Output summary:
#   results/exp15/exp15_summary.json
#
# Usage:
#   bash scripts/exp15_diagnosis_dual_memory_ewc.sh [CHECKPOINT_PATH]
#
# Optional env:
#   EXP15_EVAL_MATRIX=True|False        default True. True is needed for BWT/FWT.
#   EXP15_LOG_PERIOD=N                  default 1. Log every N adaptation iterations.
#   EXP15_RUNS="d0 d1 d5"               default all runs.
#   EXP15_SOURCE_ANCHOR_ALPHA=0.40      default 0.40, chosen from exp13 Sol-B sweep.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"
FISHER_PATH="../models/stats/COCO_R50_fisher.pt"
EXP15_EVAL_MATRIX="${EXP15_EVAL_MATRIX:-True}"
EXP15_LOG_PERIOD="${EXP15_LOG_PERIOD:-1}"
EXP15_RUNS="${EXP15_RUNS:-d0 d1 d2 d3 d4 d5}"
EXP15_SOURCE_ANCHOR_ALPHA="${EXP15_SOURCE_ANCHOR_ALPHA:-0.40}"
ALPHA_TAG="${EXP15_SOURCE_ANCHOR_ALPHA//./_}"

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/datasets}"
mkdir -p ../results/exp15

if [ ! -f "$FISHER_PATH" ]; then
    echo "=== Pre-step: computing Fisher information for adapter drift/EWC ==="
    python compute_fisher.py \
        --config-file "$CFG" \
        MODEL.WEIGHTS "$CKPT" \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        TEST.ADAPTATION.WHERE "adapter"
fi

BASE_ARGS=(
    --config-file "$CFG"
    --eval-only
    MODEL.WEIGHTS "$CKPT"
    TEST.ONLINE_ADAPTATION True
    TEST.CONTINUAL_DOMAIN True
    TEST.EVAL_MATRIX "$EXP15_EVAL_MATRIX"
    TEST.ADAPTATION.CONTINUAL True
    TEST.ADAPTATION.WHERE "adapter"
    TEST.ADAPTATION.GLOBAL_ALIGN "KL"
    TEST.ADAPTATION.FOREGROUND_ALIGN "KL"
    TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH"
    TEST.ADAPTATION.EWC_FISHER_PATH "$FISHER_PATH"
    TEST.ADAPTATION.EMA_BETA 0.0
    TEST.ADAPTATION.SWEMA_K 0
    TEST.ADAPTATION.ASRI_ALPHA 0.0
    TEST.ADAPTATION.DRIFT_LOG True
    TEST.ADAPTATION.DRIFT_LOG_PERIOD "$EXP15_LOG_PERIOD"
    TEST.ADAPTATION.ADAPTER_RESET False
    TEST.ADAPTATION.ORACLE_PROTOTYPE False
    TEST.ADAPTATION.EWC_LAMBDA 0.0
    TEST.ADAPTATION.PROTO_METHOD "baseline"
)

has_run() {
    local key="$1"
    [[ " $EXP15_RUNS " == *" $key "* ]]
}

collect() {
    local tag="$1"
    local out="$2"
    cp "${out}/eval_matrix/metrics.json" "../results/exp15/metrics_${tag}.json" 2>/dev/null || true
    cp "${out}/eval_matrix/eval_matrix.npy" "../results/exp15/eval_matrix_${tag}.npy" 2>/dev/null || true
    cp "${out}/eval_matrix/eval_matrix_per_class.npy" "../results/exp15/eval_matrix_per_class_${tag}.npy" 2>/dev/null || true
    cp "${out}/drift/drift_log.jsonl" "../results/exp15/drift_${tag}.jsonl" 2>/dev/null || true
}

run_variant() {
    local key="$1"
    local tag="$2"
    local label="$3"
    shift 3
    if ! has_run "$key"; then
        echo "=== Skip ${key}/${tag}: not in EXP15_RUNS ==="
        return
    fi
    local out="../outputs/COCO_TTA/exp15_${tag}"
    echo ""
    echo "=== ${key^^}: ${label} ==="
    python train_net.py "${BASE_ARGS[@]}" "$@" OUTPUT_DIR "$out"
    collect "$tag" "$out"
}

run_variant d0 d0_baseline \
    "baseline: normal prototype + continual adapter" \
    TEST.ADAPTATION.PROTO_METHOD "baseline" \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

run_variant d1 d1_oracle_proto \
    "oracle prototype: source prototype in fg KL target" \
    TEST.ADAPTATION.ORACLE_PROTOTYPE True \
    TEST.ADAPTATION.PROTO_METHOD "baseline" \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

run_variant d2 d2_adapter_reset \
    "adapter reset: removes cross-domain adapter accumulation, keeps prototype memory" \
    TEST.ADAPTATION.ADAPTER_RESET True \
    TEST.ADAPTATION.PROTO_METHOD "baseline" \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

run_variant d3 d3_ewc10 \
    "EWC only: constrains adapter drift, leaves prototype update unchanged" \
    TEST.ADAPTATION.PROTO_METHOD "baseline" \
    TEST.ADAPTATION.EWC_LAMBDA 10.0

run_variant d4 "d4_dual_memory_a${ALPHA_TAG}" \
    "dual memory only: Sol-B alpha=${EXP15_SOURCE_ANCHOR_ALPHA}, no EWC" \
    TEST.ADAPTATION.PROTO_METHOD "dual_memory" \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$EXP15_SOURCE_ANCHOR_ALPHA" \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

run_variant d5 "d5_ewc10_dual_memory_a${ALPHA_TAG}" \
    "Sol-B candidate: EWC(lambda=10) + dual memory alpha=${EXP15_SOURCE_ANCHOR_ALPHA}" \
    TEST.ADAPTATION.PROTO_METHOD "dual_memory" \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$EXP15_SOURCE_ANCHOR_ALPHA" \
    TEST.ADAPTATION.EWC_LAMBDA 10.0

python3 - <<'PYEOF'
import glob
import json
import math
import os
from statistics import mean

RESULT_DIR = "../results/exp15"

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
]

LABELS = {
    "d0_baseline": "Baseline",
    "d1_oracle_proto": "Oracle proto",
    "d2_adapter_reset": "Adapter reset",
    "d3_ewc10": "EWC only",
}

def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)

def avg(vals):
    vals = [v for v in vals if finite(v)]
    return mean(vals) if vals else None

def last(vals):
    vals = [v for v in vals if finite(v)]
    return vals[-1] if vals else None

def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    xbar = mean(x for x, _ in pairs)
    ybar = mean(y for _, y in pairs)
    num = sum((x - xbar) * (y - ybar) for x, y in pairs)
    denx = sum((x - xbar) ** 2 for x, _ in pairs)
    deny = sum((y - ybar) ** 2 for _, y in pairs)
    if denx <= 0 or deny <= 0:
        return None
    return num / math.sqrt(denx * deny)

def read_json(path):
    try:
        with open(path) as fp:
            return json.load(fp)
    except Exception:
        return {}

def read_drift(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def get_ap(metrics, corruption):
    return metrics.get(f"coco_2017_val-{corruption}", {}).get("AP")

def avg_ap15(metrics):
    vals = [get_ap(metrics, c) for c in CORRUPTIONS]
    vals = [v for v in vals if finite(v)]
    return sum(vals) / len(CORRUPTIONS) if len(vals) == len(CORRUPTIONS) else None

def mean_ap16(metrics):
    vals = [get_ap(metrics, c) for c in CORRUPTIONS]
    clean = metrics.get("coco_2017_val", {}).get("AP")
    vals = [v for v in vals if finite(v)]
    if len(vals) != len(CORRUPTIONS) or not finite(clean):
        return None
    return (sum(vals) + clean) / (len(CORRUPTIONS) + 1)

def label_for(tag):
    if tag in LABELS:
        return LABELS[tag]
    if tag.startswith("d4_dual_memory"):
        return "Dual memory"
    if tag.startswith("d5_ewc10_dual_memory"):
        return "EWC + dual memory"
    return tag

summary = []
for metrics_path in sorted(glob.glob(os.path.join(RESULT_DIR, "metrics_*.json"))):
    tag = os.path.basename(metrics_path).replace("metrics_", "").replace(".json", "")
    metrics = read_json(metrics_path)
    rows = read_drift(os.path.join(RESULT_DIR, f"drift_{tag}.jsonl"))

    proto_drift = [r.get("proto_drift_source_mean") for r in rows]
    proto_cos = [r.get("proto_cos_source_mean") for r in rows]
    adapter_l2 = [r.get("adapter_l2") for r in rows]
    adapter_fisher = [r.get("adapter_fisher") for r in rows]
    fg_score = [r.get("fg_score_mean") for r in rows]
    fg_boxes = [r.get("fg_num_boxes") for r in rows]
    fg_loss = [r.get("losses", {}).get("fg_align") for r in rows]
    gl_loss = [r.get("losses", {}).get("global_align") for r in rows]
    reset_count = sum(int(r.get("proto_reset_count") or 0) for r in rows)

    item = {
        "tag": tag,
        "label": label_for(tag),
        "BWT": metrics.get("BWT"),
        "FWT": metrics.get("FWT"),
        "avg_AP15": avg_ap15(metrics),
        "mean_AP16": mean_ap16(metrics),
        "clean_AP": metrics.get("coco_2017_val", {}).get("AP"),
        "n_logged_steps": len(rows),
        "proto_drift_mean": avg(proto_drift),
        "proto_drift_final": last(proto_drift),
        "proto_cos_source_mean": avg(proto_cos),
        "adapter_l2_mean": avg(adapter_l2),
        "adapter_l2_final": last(adapter_l2),
        "adapter_fisher_mean": avg(adapter_fisher),
        "adapter_fisher_final": last(adapter_fisher),
        "fg_score_mean": avg(fg_score),
        "fg_boxes_mean": avg(fg_boxes),
        "fg_align_loss_mean": avg(fg_loss),
        "global_align_loss_mean": avg(gl_loss),
        "proto_reset_count_total": reset_count,
        "corr_proto_drift_adapter_l2": corr(proto_drift, adapter_l2),
        "corr_proto_drift_adapter_fisher": corr(proto_drift, adapter_fisher),
        "corr_proto_drift_fg_loss": corr(proto_drift, fg_loss),
        "corr_adapter_fisher_fg_loss": corr(adapter_fisher, fg_loss),
    }
    item.update({c: get_ap(metrics, c) for c in CORRUPTIONS})
    summary.append(item)

with open(os.path.join(RESULT_DIR, "exp15_summary.json"), "w") as fp:
    json.dump(summary, fp, indent=2)

print("\n=== Exp 15 Summary ===")
print(
    f"{'run':<34} {'avg15':>8} {'mean16':>8} {'BWT':>8} {'FWT':>8}"
    f" {'protoD':>8} {'adptL2':>9} {'adptF':>9} {'score':>8} {'resets':>7}"
)
print("-" * 122)
for r in summary:
    def fmt(v, width=8, prec=3):
        return f"{v:>{width}.{prec}f}" if finite(v) else f"{'nan':>{width}}"
    print(
        f"{r['tag']:<34}"
        f" {fmt(r.get('avg_AP15'))}"
        f" {fmt(r.get('mean_AP16'))}"
        f" {fmt(r.get('BWT'))}"
        f" {fmt(r.get('FWT'))}"
        f" {fmt(r.get('proto_drift_mean'))}"
        f" {fmt(r.get('adapter_l2_final'), 9)}"
        f" {fmt(r.get('adapter_fisher_final'), 9)}"
        f" {fmt(r.get('fg_score_mean'))}"
        f" {int(r.get('proto_reset_count_total') or 0):>7}"
    )
print(f"\nSaved to {os.path.join(RESULT_DIR, 'exp15_summary.json')}")
PYEOF

echo ""
echo "=== Exp 15 complete. Results in results/exp15/ ==="
