#!/usr/bin/env bash
# =============================================================================
# Exp 17: Direct Evidence that Prototype Drift Causes Forgetting
#
# Purpose:
#   Produce reviewer-facing diagnostics for:
#     "Prototype reset improves AP slightly, but where is the direct evidence
#      that prototype drift caused forgetting?"
#
# Evidence generated:
#   1) Eval-matrix/BWT forgetting decomposition
#      - final AP minus first-learned AP per corruption
#      - BWT and mean final forgetting per run
#   2) Drift-performance correlation
#      - prototype drift vs current-domain AP
#      - cumulative prototype drift vs previous-domain forgetting
#      - prototype drift vs fg/global alignment loss
#   3) Reset-event before/after analysis
#      - loss/prototype-drift windows around reset events
#      - domain-level AP before/after adaptation on domains with reset events
#
# Runs:
#   D0  baseline                         normal prototype + continual adapter
#   D1  oracle prototype                 source prototype in fg KL objective
#   D2  adapter reset                    adapter reset; prototype still drifts
#   D3  prototype reset                  Sol-A reset(thr); adapter still drifts
#   D4  adapter reset + prototype reset  controls adapter drift while testing reset
#
# Outputs:
#   results/exp17/metrics_<tag>.json
#   results/exp17/eval_matrix_<tag>.npy
#   results/exp17/eval_matrix_per_class_<tag>.npy
#   results/exp17/drift_<tag>.jsonl
#   results/exp17/exp17_summary.json
#   results/exp17/exp17_direct_evidence.json
#   results/exp17/exp17_drift_performance_samples.json
#   results/exp17/exp17_reset_event_windows.json
#
# Usage:
#   bash scripts/exp17_prototype_drift_evidence.sh [CHECKPOINT_PATH]
#
# Optional env:
#   EXP17_EVAL_MATRIX=True|False     default True. True is needed for BWT.
#   EXP17_LOG_PERIOD=N               default 1.
#   EXP17_RUNS="d0 d1 d3"            default all runs.
#   EXP17_SWITCH_COSIM_THR=0.40      default 0.40, chosen from exp14.
#   EXP17_WINDOW=3                   before/after logged-step window for reset events.
#   EXP17_ANALYZE_ONLY=True|False    default False. True skips model runs.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"
EXP17_EVAL_MATRIX="${EXP17_EVAL_MATRIX:-True}"
EXP17_LOG_PERIOD="${EXP17_LOG_PERIOD:-1}"
EXP17_RUNS="${EXP17_RUNS:-d0 d1 d2 d3 d4}"
EXP17_SWITCH_COSIM_THR="${EXP17_SWITCH_COSIM_THR:-0.40}"
EXP17_WINDOW="${EXP17_WINDOW:-3}"
EXP17_ANALYZE_ONLY="${EXP17_ANALYZE_ONLY:-False}"
THR_TAG="${EXP17_SWITCH_COSIM_THR//./_}"

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/datasets}"
mkdir -p ../results/exp17

BASE_ARGS=(
    --config-file "$CFG"
    --eval-only
    MODEL.WEIGHTS "$CKPT"
    TEST.ONLINE_ADAPTATION True
    TEST.CONTINUAL_DOMAIN True
    TEST.EVAL_MATRIX "$EXP17_EVAL_MATRIX"
    TEST.ADAPTATION.CONTINUAL True
    TEST.ADAPTATION.WHERE "adapter"
    TEST.ADAPTATION.GLOBAL_ALIGN "KL"
    TEST.ADAPTATION.FOREGROUND_ALIGN "KL"
    TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH"
    TEST.ADAPTATION.EMA_BETA 0.0
    TEST.ADAPTATION.SWEMA_K 0
    TEST.ADAPTATION.ASRI_ALPHA 0.0
    TEST.ADAPTATION.DRIFT_LOG True
    TEST.ADAPTATION.DRIFT_LOG_PERIOD "$EXP17_LOG_PERIOD"
    TEST.ADAPTATION.ADAPTER_RESET False
    TEST.ADAPTATION.ORACLE_PROTOTYPE False
    TEST.ADAPTATION.EWC_LAMBDA 0.0
    TEST.ADAPTATION.PROTO_METHOD "baseline"
)

has_run() {
    local key="$1"
    [[ " $EXP17_RUNS " == *" $key "* ]]
}

collect() {
    local tag="$1"
    local out="$2"
    cp "${out}/eval_matrix/metrics.json" "../results/exp17/metrics_${tag}.json" 2>/dev/null || true
    cp "${out}/eval_matrix/eval_matrix.npy" "../results/exp17/eval_matrix_${tag}.npy" 2>/dev/null || true
    cp "${out}/eval_matrix/eval_matrix_per_class.npy" "../results/exp17/eval_matrix_per_class_${tag}.npy" 2>/dev/null || true
    cp "${out}/drift/drift_log.jsonl" "../results/exp17/drift_${tag}.jsonl" 2>/dev/null || true
}

run_variant() {
    local key="$1"
    local tag="$2"
    local label="$3"
    shift 3
    if [[ "$EXP17_ANALYZE_ONLY" == "True" ]]; then
        echo "=== Analyze-only: skip ${key}/${tag} ==="
        return
    fi
    if ! has_run "$key"; then
        echo "=== Skip ${key}/${tag}: not in EXP17_RUNS ==="
        return
    fi
    local out="../outputs/COCO_TTA/exp17_${tag}"
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
    "adapter reset: controls adapter drift, prototype memory still drifts" \
    TEST.ADAPTATION.ADAPTER_RESET True \
    TEST.ADAPTATION.PROTO_METHOD "baseline" \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

run_variant d3 "d3_proto_reset_thr${THR_TAG}" \
    "prototype reset: Sol-A reset thr=${EXP17_SWITCH_COSIM_THR}, adapter unconstrained" \
    TEST.ADAPTATION.PROTO_METHOD "reset" \
    TEST.ADAPTATION.SWITCH_COSIM_THR "$EXP17_SWITCH_COSIM_THR" \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

run_variant d4 "d4_adapter_reset_proto_reset_thr${THR_TAG}" \
    "adapter reset + prototype reset: isolates prototype reset under adapter control" \
    TEST.ADAPTATION.ADAPTER_RESET True \
    TEST.ADAPTATION.PROTO_METHOD "reset" \
    TEST.ADAPTATION.SWITCH_COSIM_THR "$EXP17_SWITCH_COSIM_THR" \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

EXP17_WINDOW="$EXP17_WINDOW" python3 - <<'PYEOF'
import glob
import json
import math
import os
from statistics import mean

import numpy as np

RESULT_DIR = "../results/exp17"
WINDOW = int(os.environ.get("EXP17_WINDOW", "3"))

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
}

def finite(v):
    return isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))

def avg(vals):
    vals = [float(v) for v in vals if finite(v)]
    return mean(vals) if vals else None

def delta(a, b):
    return float(a) - float(b) if finite(a) and finite(b) else None

def corr(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
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

def label_for(tag):
    if tag in LABELS:
        return LABELS[tag]
    if tag.startswith("d3_proto_reset"):
        return "Prototype reset"
    if tag.startswith("d4_adapter_reset_proto_reset"):
        return "Adapter reset + prototype reset"
    return tag

def loss_value(row, key):
    return (row.get("losses") or {}).get(key)

def rows_for_domain(rows, domain_idx):
    return [r for r in rows if r.get("domain_idx") == domain_idx]

def domain_mean(rows, domain_idx, key):
    return avg(r.get(key) for r in rows_for_domain(rows, domain_idx))

def domain_loss_mean(rows, domain_idx, key):
    return avg(loss_value(r, key) for r in rows_for_domain(rows, domain_idx))

def domain_reset_count(rows, domain_idx):
    return sum(int(r.get("proto_reset_count") or 0) for r in rows_for_domain(rows, domain_idx))

def load_matrix(tag):
    path = os.path.join(RESULT_DIR, f"eval_matrix_{tag}.npy")
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception:
        return None

def eval_matrix_stats(matrix):
    if matrix is None or matrix.ndim != 2 or matrix.shape[0] < 2:
        return {}
    T = min(matrix.shape[1], matrix.shape[0] - 1, len(CORRUPTIONS))
    diag = np.array([matrix[i + 1, i] for i in range(T)], dtype=float)
    final = np.array([matrix[T, i] for i in range(T)], dtype=float)
    source = np.array([matrix[0, i] for i in range(T)], dtype=float)
    forgetting = final - diag
    return {
        "T": int(T),
        "source_AP": {CORRUPTIONS[i]: float(source[i]) for i in range(T)},
        "first_learned_AP": {CORRUPTIONS[i]: float(diag[i]) for i in range(T)},
        "final_AP": {CORRUPTIONS[i]: float(final[i]) for i in range(T)},
        "final_forgetting": {CORRUPTIONS[i]: float(forgetting[i]) for i in range(T)},
        "mean_final_forgetting": float(np.nanmean(forgetting)),
        "worst_final_forgetting": float(np.nanmin(forgetting)),
        "avg_final_AP": float(np.nanmean(final)),
    }

def build_drift_perf_samples(tag, rows, matrix):
    samples = {
        "current_domain": [],
        "previous_domain_forgetting": [],
        "loss_correlation": [],
    }
    if matrix is None or matrix.ndim != 2 or matrix.shape[0] < 2:
        return samples
    T = min(matrix.shape[1], matrix.shape[0] - 1, len(CORRUPTIONS))
    diag = [float(matrix[i + 1, i]) for i in range(T)]

    for j in range(T):
        drows = rows_for_domain(rows, j)
        proto = avg(r.get("proto_drift_source_mean") for r in drows)
        proto_cos = avg(r.get("proto_cos_source_mean") for r in drows)
        fg_loss = avg(loss_value(r, "fg_align") for r in drows)
        gl_loss = avg(loss_value(r, "global_align") for r in drows)
        samples["current_domain"].append({
            "tag": tag,
            "domain_idx": j,
            "domain": CORRUPTIONS[j],
            "proto_drift_mean": proto,
            "proto_cos_source_mean": proto_cos,
            "fg_align_loss_mean": fg_loss,
            "global_align_loss_mean": gl_loss,
            "current_domain_AP_after_adapt": float(matrix[j + 1, j]),
            "source_AP": float(matrix[0, j]),
            "current_domain_gain_over_source": float(matrix[j + 1, j] - matrix[0, j]),
        })

        if j >= 1:
            prev_forgetting = [float(matrix[j + 1, i] - diag[i]) for i in range(j)]
            cumulative_rows = [r for r in rows if int(r.get("domain_idx", -1)) <= j]
            samples["previous_domain_forgetting"].append({
                "tag": tag,
                "after_domain_idx": j,
                "after_domain": CORRUPTIONS[j],
                "cumulative_proto_drift_mean": avg(r.get("proto_drift_source_mean") for r in cumulative_rows),
                "cumulative_fg_align_loss_mean": avg(loss_value(r, "fg_align") for r in cumulative_rows),
                "previous_domains_mean_forgetting": avg(prev_forgetting),
                "previous_domains_min_forgetting": min(prev_forgetting) if prev_forgetting else None,
            })

    for r in rows:
        samples["loss_correlation"].append({
            "tag": tag,
            "domain_idx": r.get("domain_idx"),
            "domain_name": r.get("domain_name"),
            "iter": r.get("iter"),
            "proto_drift_source_mean": r.get("proto_drift_source_mean"),
            "proto_cos_source_mean": r.get("proto_cos_source_mean"),
            "proto_reset_count": r.get("proto_reset_count"),
            "fg_align_loss": loss_value(r, "fg_align"),
            "global_align_loss": loss_value(r, "global_align"),
        })
    return samples

def reset_windows(tag, rows, matrix):
    event_rows = []
    aggregate = []
    for idx, row in enumerate(rows):
        if int(row.get("proto_reset_count") or 0) <= 0:
            continue
        domain_idx = row.get("domain_idx")
        same_domain_indices = [
            i for i, r in enumerate(rows) if r.get("domain_idx") == domain_idx
        ]
        pos = same_domain_indices.index(idx) if idx in same_domain_indices else None
        before_idx = same_domain_indices[max(0, pos - WINDOW):pos] if pos is not None else []
        after_idx = same_domain_indices[pos + 1:pos + 1 + WINDOW] if pos is not None else []
        before = [rows[i] for i in before_idx]
        after = [rows[i] for i in after_idx]
        item = {
            "tag": tag,
            "domain_idx": domain_idx,
            "domain_name": row.get("domain_name"),
            "iter": row.get("iter"),
            "global_step": row.get("global_step"),
            "proto_reset_count": int(row.get("proto_reset_count") or 0),
            "proto_reset_classes": row.get("proto_reset_classes"),
            "window": WINDOW,
            "fg_align_loss_before": avg(loss_value(r, "fg_align") for r in before),
            "fg_align_loss_at": loss_value(row, "fg_align"),
            "fg_align_loss_after": avg(loss_value(r, "fg_align") for r in after),
            "global_align_loss_before": avg(loss_value(r, "global_align") for r in before),
            "global_align_loss_at": loss_value(row, "global_align"),
            "global_align_loss_after": avg(loss_value(r, "global_align") for r in after),
            "proto_drift_before": avg(r.get("proto_drift_source_mean") for r in before),
            "proto_drift_at": row.get("proto_drift_source_mean"),
            "proto_drift_after": avg(r.get("proto_drift_source_mean") for r in after),
        }
        item["fg_align_loss_after_minus_before"] = delta(item["fg_align_loss_after"], item["fg_align_loss_before"])
        item["global_align_loss_after_minus_before"] = delta(item["global_align_loss_after"], item["global_align_loss_before"])
        item["proto_drift_after_minus_before"] = delta(item["proto_drift_after"], item["proto_drift_before"])
        event_rows.append(item)

    if matrix is not None and matrix.ndim == 2 and matrix.shape[0] > 1:
        T = min(matrix.shape[1], matrix.shape[0] - 1, len(CORRUPTIONS))
        for j in range(T):
            resets = domain_reset_count(rows, j)
            if resets <= 0:
                continue
            current_before = float(matrix[j, j]) if j < matrix.shape[0] else None
            current_after = float(matrix[j + 1, j])
            prev_before = float(np.nanmean(matrix[j, :j])) if j > 0 else None
            prev_after = float(np.nanmean(matrix[j + 1, :j])) if j > 0 else None
            aggregate.append({
                "tag": tag,
                "domain_idx": j,
                "domain": CORRUPTIONS[j],
                "reset_count": resets,
                "current_domain_AP_before_adapt": current_before,
                "current_domain_AP_after_adapt": current_after,
                "current_domain_AP_delta": delta(current_after, current_before),
                "previous_domains_AP_before_adapt": prev_before,
                "previous_domains_AP_after_adapt": prev_after,
                "previous_domains_AP_delta": delta(prev_after, prev_before),
                "domain_proto_drift_mean": domain_mean(rows, j, "proto_drift_source_mean"),
                "domain_fg_align_loss_mean": domain_loss_mean(rows, j, "fg_align"),
            })
    return event_rows, aggregate

def summarize_correlations(samples):
    cur = samples["current_domain"]
    prev = samples["previous_domain_forgetting"]
    loss = samples["loss_correlation"]
    return {
        "proto_drift_vs_current_domain_AP": corr(
            [s.get("proto_drift_mean") for s in cur],
            [s.get("current_domain_AP_after_adapt") for s in cur],
        ),
        "proto_drift_vs_current_domain_gain_over_source": corr(
            [s.get("proto_drift_mean") for s in cur],
            [s.get("current_domain_gain_over_source") for s in cur],
        ),
        "cumulative_proto_drift_vs_previous_domain_forgetting": corr(
            [s.get("cumulative_proto_drift_mean") for s in prev],
            [s.get("previous_domains_mean_forgetting") for s in prev],
        ),
        "proto_drift_vs_fg_align_loss": corr(
            [s.get("proto_drift_source_mean") for s in loss],
            [s.get("fg_align_loss") for s in loss],
        ),
        "proto_drift_vs_global_align_loss": corr(
            [s.get("proto_drift_source_mean") for s in loss],
            [s.get("global_align_loss") for s in loss],
        ),
    }

summary = []
all_samples = {}
all_reset_events = {}
all_reset_domain_ap = {}

for metrics_path in sorted(glob.glob(os.path.join(RESULT_DIR, "metrics_*.json"))):
    tag = os.path.basename(metrics_path).replace("metrics_", "").replace(".json", "")
    metrics = read_json(metrics_path)
    rows = read_drift(os.path.join(RESULT_DIR, f"drift_{tag}.jsonl"))
    matrix = load_matrix(tag)
    matrix_stats = eval_matrix_stats(matrix)
    samples = build_drift_perf_samples(tag, rows, matrix)
    events, domain_ap = reset_windows(tag, rows, matrix)

    item = {
        "tag": tag,
        "label": label_for(tag),
        "BWT": metrics.get("BWT"),
        "FWT": metrics.get("FWT"),
        "avg_mAP": metrics.get("avg_mAP"),
        "n_logged_steps": len(rows),
        "proto_drift_mean": avg(r.get("proto_drift_source_mean") for r in rows),
        "proto_drift_final": avg(r.get("proto_drift_source_mean") for r in rows[-10:]),
        "fg_align_loss_mean": avg(loss_value(r, "fg_align") for r in rows),
        "global_align_loss_mean": avg(loss_value(r, "global_align") for r in rows),
        "proto_reset_count_total": sum(int(r.get("proto_reset_count") or 0) for r in rows),
        "eval_matrix": matrix_stats,
        "correlations": summarize_correlations(samples),
    }
    summary.append(item)
    all_samples[tag] = samples
    all_reset_events[tag] = events
    all_reset_domain_ap[tag] = domain_ap

with open(os.path.join(RESULT_DIR, "exp17_summary.json"), "w") as fp:
    json.dump(summary, fp, indent=2)
with open(os.path.join(RESULT_DIR, "exp17_drift_performance_samples.json"), "w") as fp:
    json.dump(all_samples, fp, indent=2)
with open(os.path.join(RESULT_DIR, "exp17_reset_event_windows.json"), "w") as fp:
    json.dump({
        "window_logged_steps": WINDOW,
        "ap_granularity": (
            "AP before/after reset is computed at eval-matrix domain granularity: "
            "row j before adapting domain j and row j+1 after adapting domain j. "
            "Per-iteration AP would require an additional online-eval hook."
        ),
        "events": all_reset_events,
        "domain_level_ap": all_reset_domain_ap,
    }, fp, indent=2)

by_label = {r["label"]: r for r in summary}
base = by_label.get("Baseline", {})
oracle = by_label.get("Oracle proto", {})
adapter_reset = by_label.get("Adapter reset", {})
proto_reset = by_label.get("Prototype reset", {})
adapter_proto_reset = by_label.get("Adapter reset + prototype reset", {})

def stat(run, key, subkey=None):
    v = run.get(key)
    if subkey is not None and isinstance(v, dict):
        return v.get(subkey)
    return v

direct = {
    "claim": "Prototype drift is evidence-linked to forgetting when drift is high, BWT/forgetting worsens, and prototype-side interventions improve it.",
    "ap_granularity_note": (
        "Reset-event AP deltas are domain-level eval-matrix deltas, not per-iteration AP."
    ),
    "eval_matrix_bwt_evidence": {
        "oracle_minus_baseline_BWT": delta(stat(oracle, "BWT"), stat(base, "BWT")),
        "proto_reset_minus_baseline_BWT": delta(stat(proto_reset, "BWT"), stat(base, "BWT")),
        "adapter_reset_proto_reset_minus_adapter_reset_BWT": delta(
            stat(adapter_proto_reset, "BWT"),
            stat(adapter_reset, "BWT"),
        ),
        "oracle_minus_baseline_mean_final_forgetting": delta(
            stat(oracle, "eval_matrix", "mean_final_forgetting"),
            stat(base, "eval_matrix", "mean_final_forgetting"),
        ),
        "proto_reset_minus_baseline_mean_final_forgetting": delta(
            stat(proto_reset, "eval_matrix", "mean_final_forgetting"),
            stat(base, "eval_matrix", "mean_final_forgetting"),
        ),
        "adapter_reset_proto_reset_minus_adapter_reset_mean_final_forgetting": delta(
            stat(adapter_proto_reset, "eval_matrix", "mean_final_forgetting"),
            stat(adapter_reset, "eval_matrix", "mean_final_forgetting"),
        ),
    },
    "drift_performance_correlation_evidence": {
        "baseline": stat(base, "correlations"),
        "prototype_reset": stat(proto_reset, "correlations"),
        "adapter_reset": stat(adapter_reset, "correlations"),
        "adapter_reset_proto_reset": stat(adapter_proto_reset, "correlations"),
    },
    "reset_event_evidence": {
        "prototype_reset_event_count": stat(proto_reset, "proto_reset_count_total"),
        "adapter_reset_proto_reset_event_count": stat(adapter_proto_reset, "proto_reset_count_total"),
        "prototype_reset_mean_event_fg_loss_delta": avg(
            e.get("fg_align_loss_after_minus_before")
            for e in all_reset_events.get(proto_reset.get("tag"), [])
        ),
        "prototype_reset_mean_event_proto_drift_delta": avg(
            e.get("proto_drift_after_minus_before")
            for e in all_reset_events.get(proto_reset.get("tag"), [])
        ),
        "prototype_reset_mean_domain_current_AP_delta": avg(
            e.get("current_domain_AP_delta")
            for e in all_reset_domain_ap.get(proto_reset.get("tag"), [])
        ),
        "prototype_reset_mean_domain_previous_AP_delta": avg(
            e.get("previous_domains_AP_delta")
            for e in all_reset_domain_ap.get(proto_reset.get("tag"), [])
        ),
    },
}

with open(os.path.join(RESULT_DIR, "exp17_direct_evidence.json"), "w") as fp:
    json.dump(direct, fp, indent=2)

print("\n=== Exp 17 Prototype Drift Evidence ===")
print(f"{'run':<38} {'BWT':>8} {'avg':>8} {'forget':>9} {'protoD':>8} {'fgLoss':>9} {'resets':>7}")
print("-" * 96)
for r in summary:
    def fmt(v, width=8, prec=3):
        return f"{float(v):>{width}.{prec}f}" if finite(v) else f"{'nan':>{width}}"
    print(
        f"{r['tag']:<38}"
        f" {fmt(r.get('BWT'))}"
        f" {fmt(r.get('avg_mAP'))}"
        f" {fmt(stat(r, 'eval_matrix', 'mean_final_forgetting'), 9)}"
        f" {fmt(r.get('proto_drift_mean'))}"
        f" {fmt(r.get('fg_align_loss_mean'), 9)}"
        f" {int(r.get('proto_reset_count_total') or 0):>7}"
    )

print("\nKey deltas:")
for key, value in direct["eval_matrix_bwt_evidence"].items():
    print(f"  {key}: {value:.4f}" if finite(value) else f"  {key}: nan")
print(f"\nSaved to {os.path.join(RESULT_DIR, 'exp17_direct_evidence.json')}")
PYEOF

echo ""
echo "=== Exp 17 complete. Results in results/exp17/ ==="
