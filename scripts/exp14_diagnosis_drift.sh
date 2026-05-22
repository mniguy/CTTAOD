#!/usr/bin/env bash
# =============================================================================
# Exp 14: Two-Coupled-Drift Diagnosis for Continual TTAOD
#
# Purpose:
#   Collect causal and trajectory evidence for the paper's central claim:
#   catastrophic forgetting in continual TTAOD is driven by two coupled drifts:
#     (1) prototype/statistical-memory drift
#     (2) adapter/parametric-memory drift
#
# Runs:
#   D0  baseline                  normal prototype + continual adapter
#   D1  oracle prototype          source prototype in fg KL objective; adapter drifts
#   D2  adapter reset             adapter reset at each domain boundary; prototype drifts
#   D3  EWC only                  normal prototype + EWC(lambda=10) on adapter
#   D4  prototype reset only      reset(thr=0.40) + no EWC
#   D5  current method            reset(thr=0.40) + EWC(lambda=10)
#
# Outputs per run:
#   results/exp14/metrics_<tag>.json
#   results/exp14/eval_matrix_<tag>.npy
#   results/exp14/eval_matrix_per_class_<tag>.npy
#   results/exp14/drift_<tag>.jsonl
#
# Output summary:
#   results/exp14/exp14_summary.json
#
# Usage:
#   bash scripts/exp14_two_drift_diagnosis.sh [CHECKPOINT_PATH]
#
# Optional env:
#   EXP14_EVAL_MATRIX=True|False   default True. True is needed for BWT/FWT.
#   EXP14_LOG_PERIOD=N             default 1. Log every N adaptation iterations.
#   EXP14_RUNS="d0 d1 d5"          default all runs.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"
FISHER_PATH="../models/stats/COCO_R50_fisher.pt"
EXP14_EVAL_MATRIX="${EXP14_EVAL_MATRIX:-True}"
EXP14_LOG_PERIOD="${EXP14_LOG_PERIOD:-1}"
EXP14_RUNS="${EXP14_RUNS:-d0 d1 d2 d3 d4 d5}"

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/datasets}"
mkdir -p ../results/exp14

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
    TEST.EVAL_MATRIX "$EXP14_EVAL_MATRIX"
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
    TEST.ADAPTATION.DRIFT_LOG_PERIOD "$EXP14_LOG_PERIOD"
    TEST.ADAPTATION.ADAPTER_RESET False
    TEST.ADAPTATION.ORACLE_PROTOTYPE False
    TEST.ADAPTATION.EWC_LAMBDA 0.0
    TEST.ADAPTATION.PROTO_METHOD "baseline"
)

has_run() {
    local key="$1"
    [[ " $EXP14_RUNS " == *" $key "* ]]
}

collect() {
    local tag="$1"
    local out="$2"
    cp "${out}/eval_matrix/metrics.json" "../results/exp14/metrics_${tag}.json" 2>/dev/null || true
    cp "${out}/eval_matrix/eval_matrix.npy" "../results/exp14/eval_matrix_${tag}.npy" 2>/dev/null || true
    cp "${out}/eval_matrix/eval_matrix_per_class.npy" "../results/exp14/eval_matrix_per_class_${tag}.npy" 2>/dev/null || true
    cp "${out}/drift/drift_log.jsonl" "../results/exp14/drift_${tag}.jsonl" 2>/dev/null || true
}

run_variant() {
    local key="$1"
    local tag="$2"
    local label="$3"
    shift 3
    if ! has_run "$key"; then
        echo "=== Skip ${key}/${tag}: not in EXP14_RUNS ==="
        return
    fi
    local out="../outputs/COCO_TTA/exp14_${tag}"
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
    "oracle prototype: removes prototype drift from fg KL target" \
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

run_variant d4 d4_proto_reset_thr0_40 \
    "prototype reset only: constrains prototype drift, leaves adapter unconstrained" \
    TEST.ADAPTATION.PROTO_METHOD "reset" \
    TEST.ADAPTATION.SWITCH_COSIM_THR 0.40 \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

run_variant d5 d5_ewc10_proto_reset_thr0_40 \
    "current method: EWC(lambda=10) + prototype reset(thr=0.40)" \
    TEST.ADAPTATION.PROTO_METHOD "reset" \
    TEST.ADAPTATION.SWITCH_COSIM_THR 0.40 \
    TEST.ADAPTATION.EWC_LAMBDA 10.0

python3 - <<'PYEOF'
import glob
import json
import math
import os
from statistics import mean

RESULT_DIR = "../results/exp14"

LABELS = {
    "d0_baseline": "Baseline",
    "d1_oracle_proto": "Oracle proto",
    "d2_adapter_reset": "Adapter reset",
    "d3_ewc10": "EWC only",
    "d4_proto_reset_thr0_40": "Proto reset",
    "d5_ewc10_proto_reset_thr0_40": "EWC + proto reset",
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
        "label": LABELS.get(tag, tag),
        "BWT": metrics.get("BWT"),
        "FWT": metrics.get("FWT"),
        "avg_mAP": metrics.get("avg_mAP"),
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
    summary.append(item)

with open(os.path.join(RESULT_DIR, "exp14_summary.json"), "w") as fp:
    json.dump(summary, fp, indent=2)

print("\n=== Exp 14 Summary ===")
print(f"{'run':<30} {'BWT':>8} {'FWT':>8} {'avg':>8} {'protoD':>8} {'adptL2':>9} {'adptF':>9} {'score':>8} {'resets':>7}")
print("-" * 105)
for r in summary:
    def fmt(v, width=8, prec=3):
        return f"{v:>{width}.{prec}f}" if finite(v) else f"{'nan':>{width}}"
    print(
        f"{r['tag']:<30}"
        f" {fmt(r.get('BWT'))}"
        f" {fmt(r.get('FWT'))}"
        f" {fmt(r.get('avg_mAP'))}"
        f" {fmt(r.get('proto_drift_mean'))}"
        f" {fmt(r.get('adapter_l2_final'), 9)}"
        f" {fmt(r.get('adapter_fisher_final'), 9)}"
        f" {fmt(r.get('fg_score_mean'))}"
        f" {int(r.get('proto_reset_count_total') or 0):>7}"
    )
print(f"\nSaved to {os.path.join(RESULT_DIR, 'exp14_summary.json')}")
PYEOF

echo ""
echo "=== Exp 14 complete. Results in results/exp14/ ==="
