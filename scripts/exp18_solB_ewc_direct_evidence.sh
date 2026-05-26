#!/usr/bin/env bash
# =============================================================================
# Exp 18: Direct Prototype-Drift Evidence for Sol-B + Adapter EWC
#
# Purpose:
#   Answer the reviewer-style question:
#     "Prototype reset improves AP slightly, but where is the direct evidence
#      that prototype drift caused forgetting?"
#
#   This experiment focuses on Sol-B + EWC and Sol-A + EWC direct evidence.
#   Sol-B has no discrete reset event, so its event analysis uses:
#     (1) domain-boundary before/after AP/loss windows
#     (2) high-prototype-drift events before/after loss windows
#   Sol-A has explicit reset events, so reset-event before/after AP/loss windows
#   are reported as well.
#
# Runs:
#   D0  baseline            normal prototype + continual adapter
#   D1  Sol-B + EWC         dual_memory(alpha=0.40) + adapter EWC(lambda=10)
#   D2  Sol-B only          optional control
#   D3  EWC only            optional control
#   D4  Sol-A + EWC         prototype reset(thr=0.40) + adapter EWC(lambda=10)
#
# Outputs per run:
#   results/exp18/metrics_<tag>.json
#   results/exp18/eval_matrix_<tag>.npy
#   results/exp18/eval_matrix_per_class_<tag>.npy
#   results/exp18/drift_<tag>.jsonl
#
# Analysis outputs:
#   results/exp18/exp18_summary.json
#   results/exp18/exp18_direct_evidence.json
#   results/exp18/exp18_drift_performance_samples.json
#   results/exp18/exp18_high_drift_event_windows.json
#
# Usage:
#   bash scripts/exp18_solB_ewc_direct_evidence.sh [CHECKPOINT_PATH]
#
# Optional env:
#   EXP18_RUNS="d0 d1 d4"           default d0 d1 d4. Add d2/d3 for controls.
#   EXP18_LOG_PERIOD=N              default 1.
#   EXP18_SOURCE_ANCHOR_ALPHA=0.40  default 0.40.
#   EXP18_SWITCH_COSIM_THR=0.40     default 0.40.
#   EXP18_EWC_LAMBDA=10.0           default 10.0.
#   EXP18_WINDOW=3                  before/after logged-step window.
#   EXP18_EVENT_Q=0.95              high-drift event quantile.
#   EXP18_ANALYZE_ONLY=True|False   default False. True skips model runs.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"
FISHER_PATH="../models/stats/COCO_R50_fisher.pt"
EXP18_RUNS="${EXP18_RUNS:-d0 d1 d4}"
EXP18_LOG_PERIOD="${EXP18_LOG_PERIOD:-1}"
EXP18_SOURCE_ANCHOR_ALPHA="${EXP18_SOURCE_ANCHOR_ALPHA:-0.40}"
EXP18_SWITCH_COSIM_THR="${EXP18_SWITCH_COSIM_THR:-0.40}"
EXP18_EWC_LAMBDA="${EXP18_EWC_LAMBDA:-10.0}"
EXP18_WINDOW="${EXP18_WINDOW:-3}"
EXP18_EVENT_Q="${EXP18_EVENT_Q:-0.95}"
EXP18_ANALYZE_ONLY="${EXP18_ANALYZE_ONLY:-False}"
ALPHA_TAG="${EXP18_SOURCE_ANCHOR_ALPHA//./_}"
THR_TAG="${EXP18_SWITCH_COSIM_THR//./_}"
EWC_TAG="${EXP18_EWC_LAMBDA//./_}"

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/datasets}"
mkdir -p ../results/exp18

if [ "$EXP18_ANALYZE_ONLY" != "True" ] && [ ! -f "$FISHER_PATH" ]; then
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
    TEST.EVAL_MATRIX True
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
    TEST.ADAPTATION.DRIFT_LOG_PERIOD "$EXP18_LOG_PERIOD"
    TEST.ADAPTATION.ADAPTER_RESET False
    TEST.ADAPTATION.ORACLE_PROTOTYPE False
    TEST.ADAPTATION.EWC_LAMBDA 0.0
    TEST.ADAPTATION.PROTO_METHOD "baseline"
)

has_run() {
    local key="$1"
    [[ " $EXP18_RUNS " == *" $key "* ]]
}

collect() {
    local tag="$1"
    local out="$2"
    cp "${out}/eval_matrix/metrics.json" "../results/exp18/metrics_${tag}.json" 2>/dev/null || true
    cp "${out}/eval_matrix/eval_matrix.npy" "../results/exp18/eval_matrix_${tag}.npy" 2>/dev/null || true
    cp "${out}/eval_matrix/eval_matrix_per_class.npy" "../results/exp18/eval_matrix_per_class_${tag}.npy" 2>/dev/null || true
    cp "${out}/drift/drift_log.jsonl" "../results/exp18/drift_${tag}.jsonl" 2>/dev/null || true
}

run_variant() {
    local key="$1"
    local tag="$2"
    local label="$3"
    shift 3
    if [ "$EXP18_ANALYZE_ONLY" = "True" ]; then
        echo "=== Analyze-only: skip ${key}/${tag} ==="
        return
    fi
    if ! has_run "$key"; then
        echo "=== Skip ${key}/${tag}: not in EXP18_RUNS ==="
        return
    fi
    local out="../outputs/COCO_TTA/exp18_${tag}"
    echo ""
    echo "=== ${key^^}: ${label} ==="
    python train_net.py "${BASE_ARGS[@]}" "$@" OUTPUT_DIR "$out"
    collect "$tag" "$out"
}

run_variant d0 d0_baseline \
    "baseline: normal prototype + continual adapter" \
    TEST.ADAPTATION.PROTO_METHOD "baseline" \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

run_variant d1 "d1_solB_ewc${EWC_TAG}_a${ALPHA_TAG}" \
    "Sol-B + EWC: dual memory alpha=${EXP18_SOURCE_ANCHOR_ALPHA}, EWC lambda=${EXP18_EWC_LAMBDA}" \
    TEST.ADAPTATION.PROTO_METHOD "dual_memory" \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$EXP18_SOURCE_ANCHOR_ALPHA" \
    TEST.ADAPTATION.EWC_LAMBDA "$EXP18_EWC_LAMBDA"

run_variant d2 "d2_solB_a${ALPHA_TAG}" \
    "Sol-B only: dual memory alpha=${EXP18_SOURCE_ANCHOR_ALPHA}" \
    TEST.ADAPTATION.PROTO_METHOD "dual_memory" \
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$EXP18_SOURCE_ANCHOR_ALPHA" \
    TEST.ADAPTATION.EWC_LAMBDA 0.0

run_variant d3 "d3_ewc${EWC_TAG}" \
    "EWC only: normal prototype + EWC lambda=${EXP18_EWC_LAMBDA}" \
    TEST.ADAPTATION.PROTO_METHOD "baseline" \
    TEST.ADAPTATION.EWC_LAMBDA "$EXP18_EWC_LAMBDA"

run_variant d4 "d4_solA_ewc${EWC_TAG}_thr${THR_TAG}" \
    "Sol-A + EWC: prototype reset thr=${EXP18_SWITCH_COSIM_THR}, EWC lambda=${EXP18_EWC_LAMBDA}" \
    TEST.ADAPTATION.PROTO_METHOD "reset" \
    TEST.ADAPTATION.SWITCH_COSIM_THR "$EXP18_SWITCH_COSIM_THR" \
    TEST.ADAPTATION.EWC_LAMBDA "$EXP18_EWC_LAMBDA"

EXP18_WINDOW="$EXP18_WINDOW" EXP18_EVENT_Q="$EXP18_EVENT_Q" python3 - <<'PYEOF'
import glob
import json
import math
import os
from statistics import mean

import numpy as np

RESULT_DIR = "../results/exp18"
WINDOW = int(os.environ.get("EXP18_WINDOW", "3"))
EVENT_Q = float(os.environ.get("EXP18_EVENT_Q", "0.95"))

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
]

def finite(v):
    return isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))

def avg(vals):
    vals = [float(v) for v in vals if finite(v)]
    return mean(vals) if vals else None

def last(vals):
    vals = [float(v) for v in vals if finite(v)]
    return vals[-1] if vals else None

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

def loss_value(row, key):
    return (row.get("losses") or {}).get(key)

def label_for(tag):
    if tag == "d0_baseline":
        return "Baseline"
    if tag.startswith("d1_solB_ewc"):
        return "Sol-B + EWC"
    if tag.startswith("d2_solB"):
        return "Sol-B"
    if tag.startswith("d3_ewc"):
        return "EWC"
    if tag.startswith("d4_solA_ewc"):
        return "Sol-A + EWC"
    return tag

def load_matrix(tag):
    path = os.path.join(RESULT_DIR, f"eval_matrix_{tag}.npy")
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception:
        return None

def rows_for_domain(rows, domain_idx):
    return [r for r in rows if r.get("domain_idx") == domain_idx]

def domain_mean(rows, domain_idx, getter):
    return avg(getter(r) for r in rows_for_domain(rows, domain_idx))

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
        "BWT_from_matrix": float(np.nanmean(forgetting[:T - 1])) if T > 1 else None,
        "avg_final_AP": float(np.nanmean(final)),
        "mean_final_forgetting": float(np.nanmean(forgetting)),
        "worst_final_forgetting": float(np.nanmin(forgetting)),
        "source_AP": {CORRUPTIONS[i]: float(source[i]) for i in range(T)},
        "first_learned_AP": {CORRUPTIONS[i]: float(diag[i]) for i in range(T)},
        "final_AP": {CORRUPTIONS[i]: float(final[i]) for i in range(T)},
        "final_forgetting": {CORRUPTIONS[i]: float(forgetting[i]) for i in range(T)},
    }

def build_samples(tag, rows, matrix):
    samples = {
        "current_domain": [],
        "previous_domain_forgetting": [],
        "loss_correlation": [],
    }
    if matrix is not None and matrix.ndim == 2 and matrix.shape[0] > 1:
        T = min(matrix.shape[1], matrix.shape[0] - 1, len(CORRUPTIONS))
        diag = [float(matrix[i + 1, i]) for i in range(T)]
        for j in range(T):
            proto = domain_mean(rows, j, lambda r: r.get("proto_drift_source_mean"))
            fg_loss = domain_mean(rows, j, lambda r: loss_value(r, "fg_align"))
            gl_loss = domain_mean(rows, j, lambda r: loss_value(r, "global_align"))
            samples["current_domain"].append({
                "tag": tag,
                "domain_idx": j,
                "domain": CORRUPTIONS[j],
                "proto_drift_mean": proto,
                "fg_align_loss_mean": fg_loss,
                "global_align_loss_mean": gl_loss,
                "source_AP": float(matrix[0, j]),
                "first_learned_AP": float(matrix[j + 1, j]),
                "final_AP": float(matrix[T, j]),
                "final_forgetting": float(matrix[T, j] - matrix[j + 1, j]),
            })
            if j >= 1:
                prev_forgetting = [float(matrix[j + 1, i] - diag[i]) for i in range(j)]
                cumulative_rows = [r for r in rows if int(r.get("domain_idx", -1)) <= j]
                samples["previous_domain_forgetting"].append({
                    "tag": tag,
                    "after_domain_idx": j,
                    "after_domain": CORRUPTIONS[j],
                    "cumulative_proto_drift_mean": avg(
                        r.get("proto_drift_source_mean") for r in cumulative_rows
                    ),
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
            "fg_align_loss": loss_value(r, "fg_align"),
            "global_align_loss": loss_value(r, "global_align"),
            "adapter_fisher": r.get("adapter_fisher"),
        })
    return samples

def summarize_correlations(samples):
    cur = samples["current_domain"]
    prev = samples["previous_domain_forgetting"]
    loss = samples["loss_correlation"]
    return {
        "domain_proto_drift_vs_final_AP": corr(
            [s.get("proto_drift_mean") for s in cur],
            [s.get("final_AP") for s in cur],
        ),
        "domain_proto_drift_vs_final_forgetting": corr(
            [s.get("proto_drift_mean") for s in cur],
            [s.get("final_forgetting") for s in cur],
        ),
        "cumulative_proto_drift_vs_previous_domain_forgetting": corr(
            [s.get("cumulative_proto_drift_mean") for s in prev],
            [s.get("previous_domains_mean_forgetting") for s in prev],
        ),
        "step_proto_drift_vs_fg_align_loss": corr(
            [s.get("proto_drift_source_mean") for s in loss],
            [s.get("fg_align_loss") for s in loss],
        ),
        "step_proto_drift_vs_global_align_loss": corr(
            [s.get("proto_drift_source_mean") for s in loss],
            [s.get("global_align_loss") for s in loss],
        ),
    }

def window_stats(rows, center_idx):
    domain_idx = rows[center_idx].get("domain_idx")
    same_domain_indices = [i for i, r in enumerate(rows) if r.get("domain_idx") == domain_idx]
    pos = same_domain_indices.index(center_idx)
    before = [rows[i] for i in same_domain_indices[max(0, pos - WINDOW):pos]]
    after = [rows[i] for i in same_domain_indices[pos + 1:pos + 1 + WINDOW]]
    row = rows[center_idx]
    item = {
        "domain_idx": domain_idx,
        "domain_name": row.get("domain_name"),
        "iter": row.get("iter"),
        "global_step": row.get("global_step"),
        "window": WINDOW,
        "proto_drift_before": avg(r.get("proto_drift_source_mean") for r in before),
        "proto_drift_at": row.get("proto_drift_source_mean"),
        "proto_drift_after": avg(r.get("proto_drift_source_mean") for r in after),
        "fg_align_loss_before": avg(loss_value(r, "fg_align") for r in before),
        "fg_align_loss_at": loss_value(row, "fg_align"),
        "fg_align_loss_after": avg(loss_value(r, "fg_align") for r in after),
        "global_align_loss_before": avg(loss_value(r, "global_align") for r in before),
        "global_align_loss_at": loss_value(row, "global_align"),
        "global_align_loss_after": avg(loss_value(r, "global_align") for r in after),
    }
    item["proto_drift_after_minus_before"] = delta(item["proto_drift_after"], item["proto_drift_before"])
    item["fg_align_loss_after_minus_before"] = delta(item["fg_align_loss_after"], item["fg_align_loss_before"])
    item["global_align_loss_after_minus_before"] = delta(
        item["global_align_loss_after"], item["global_align_loss_before"]
    )
    return item

def event_windows(tag, rows, matrix):
    out = {
        "high_drift_threshold": None,
        "high_drift_events": [],
        "domain_boundary_events": [],
        "reset_events": [],
        "reset_domain_ap_events": [],
    }
    drift_vals = [r.get("proto_drift_source_mean") for r in rows if finite(r.get("proto_drift_source_mean"))]
    if drift_vals:
        threshold = float(np.quantile(np.array(drift_vals, dtype=float), EVENT_Q))
        out["high_drift_threshold"] = threshold
        for idx, row in enumerate(rows):
            if finite(row.get("proto_drift_source_mean")) and row.get("proto_drift_source_mean") >= threshold:
                event = window_stats(rows, idx)
                event["tag"] = tag
                event["event_type"] = "high_proto_drift"
                event["event_quantile"] = EVENT_Q
                out["high_drift_events"].append(event)

    first_idx_by_domain = {}
    for idx, row in enumerate(rows):
        first_idx_by_domain.setdefault(row.get("domain_idx"), idx)
    for domain_idx, idx in sorted(first_idx_by_domain.items()):
        if domain_idx is None:
            continue
        event = window_stats(rows, idx)
        event["tag"] = tag
        event["event_type"] = "domain_boundary"
        if matrix is not None and matrix.ndim == 2:
            j = int(domain_idx)
            T = min(matrix.shape[1], matrix.shape[0] - 1, len(CORRUPTIONS))
            if 0 <= j < T:
                event["current_domain_AP_before_adapt"] = float(matrix[j, j])
                event["current_domain_AP_after_adapt"] = float(matrix[j + 1, j])
                event["current_domain_AP_delta"] = float(matrix[j + 1, j] - matrix[j, j])
                if j > 0:
                    event["previous_domains_AP_before_adapt"] = float(np.nanmean(matrix[j, :j]))
                    event["previous_domains_AP_after_adapt"] = float(np.nanmean(matrix[j + 1, :j]))
                    event["previous_domains_AP_delta"] = float(
                        np.nanmean(matrix[j + 1, :j]) - np.nanmean(matrix[j, :j])
                    )
        out["domain_boundary_events"].append(event)

    for idx, row in enumerate(rows):
        if int(row.get("proto_reset_count") or 0) <= 0:
            continue
        event = window_stats(rows, idx)
        event["tag"] = tag
        event["event_type"] = "proto_reset"
        event["proto_reset_count"] = int(row.get("proto_reset_count") or 0)
        event["proto_reset_classes"] = row.get("proto_reset_classes")
        out["reset_events"].append(event)

    if matrix is not None and matrix.ndim == 2 and matrix.shape[0] > 1:
        T = min(matrix.shape[1], matrix.shape[0] - 1, len(CORRUPTIONS))
        reset_domains = sorted({
            int(r.get("domain_idx"))
            for r in rows
            if r.get("domain_idx") is not None and int(r.get("proto_reset_count") or 0) > 0
        })
        for j in reset_domains:
            if not (0 <= j < T):
                continue
            domain_rows = rows_for_domain(rows, j)
            item = {
                "tag": tag,
                "event_type": "proto_reset_domain_ap",
                "domain_idx": j,
                "domain": CORRUPTIONS[j],
                "reset_count": sum(int(r.get("proto_reset_count") or 0) for r in domain_rows),
                "current_domain_AP_before_adapt": float(matrix[j, j]),
                "current_domain_AP_after_adapt": float(matrix[j + 1, j]),
                "current_domain_AP_delta": float(matrix[j + 1, j] - matrix[j, j]),
                "domain_proto_drift_mean": domain_mean(rows, j, lambda r: r.get("proto_drift_source_mean")),
                "domain_fg_align_loss_mean": domain_mean(rows, j, lambda r: loss_value(r, "fg_align")),
            }
            if j > 0:
                item["previous_domains_AP_before_adapt"] = float(np.nanmean(matrix[j, :j]))
                item["previous_domains_AP_after_adapt"] = float(np.nanmean(matrix[j + 1, :j]))
                item["previous_domains_AP_delta"] = float(
                    np.nanmean(matrix[j + 1, :j]) - np.nanmean(matrix[j, :j])
                )
            out["reset_domain_ap_events"].append(item)
    return out

summary = []
all_samples = {}
all_events = {}

for metrics_path in sorted(glob.glob(os.path.join(RESULT_DIR, "metrics_*.json"))):
    tag = os.path.basename(metrics_path).replace("metrics_", "").replace(".json", "")
    metrics = read_json(metrics_path)
    rows = read_drift(os.path.join(RESULT_DIR, f"drift_{tag}.jsonl"))
    matrix = load_matrix(tag)
    matrix_stats = eval_matrix_stats(matrix)
    samples = build_samples(tag, rows, matrix)
    events = event_windows(tag, rows, matrix)

    item = {
        "tag": tag,
        "label": label_for(tag),
        "BWT": metrics.get("BWT"),
        "FWT": metrics.get("FWT"),
        "avg_mAP": metrics.get("avg_mAP"),
        "n_logged_steps": len(rows),
        "proto_drift_mean": avg(r.get("proto_drift_source_mean") for r in rows),
        "proto_drift_final": last(r.get("proto_drift_source_mean") for r in rows),
        "adapter_fisher_final": last(r.get("adapter_fisher") for r in rows),
        "fg_align_loss_mean": avg(loss_value(r, "fg_align") for r in rows),
        "global_align_loss_mean": avg(loss_value(r, "global_align") for r in rows),
        "eval_matrix": matrix_stats,
        "correlations": summarize_correlations(samples),
        "n_high_drift_events": len(events["high_drift_events"]),
        "n_domain_boundary_events": len(events["domain_boundary_events"]),
        "n_reset_events": len(events["reset_events"]),
        "n_reset_domain_ap_events": len(events["reset_domain_ap_events"]),
    }
    summary.append(item)
    all_samples[tag] = samples
    all_events[tag] = events

with open(os.path.join(RESULT_DIR, "exp18_summary.json"), "w") as fp:
    json.dump(summary, fp, indent=2)
with open(os.path.join(RESULT_DIR, "exp18_drift_performance_samples.json"), "w") as fp:
    json.dump(all_samples, fp, indent=2)
with open(os.path.join(RESULT_DIR, "exp18_high_drift_event_windows.json"), "w") as fp:
    json.dump({
        "window_logged_steps": WINDOW,
        "high_drift_event_quantile": EVENT_Q,
        "note": "Sol-B has no reset event; high-drift and domain-boundary events are used instead. Sol-A reset events are included when present.",
        "events": all_events,
    }, fp, indent=2)

by_label = {r["label"]: r for r in summary}
baseline = by_label.get("Baseline", {})
solb_ewc = by_label.get("Sol-B + EWC", {})
sola_ewc = by_label.get("Sol-A + EWC", {})

def stat(run, key, subkey=None):
    v = run.get(key)
    if subkey is not None and isinstance(v, dict):
        return v.get(subkey)
    return v

def mean_event_delta(tag, event_key, value_key):
    events = all_events.get(tag, {}).get(event_key, [])
    return avg(e.get(value_key) for e in events)

direct = {
    "claim": "Sol-B + EWC is evaluated with direct eval-matrix forgetting, drift-performance correlation, and event-window diagnostics.",
    "event_note": "Sol-B has no reset event; event analysis uses high-prototype-drift and domain-boundary events.",
    "eval_matrix_bwt_evidence": {
        "solB_ewc_minus_baseline_BWT": delta(stat(solb_ewc, "BWT"), stat(baseline, "BWT")),
        "solB_ewc_minus_baseline_avg_mAP": delta(stat(solb_ewc, "avg_mAP"), stat(baseline, "avg_mAP")),
        "solB_ewc_minus_baseline_mean_final_forgetting": delta(
            stat(solb_ewc, "eval_matrix", "mean_final_forgetting"),
            stat(baseline, "eval_matrix", "mean_final_forgetting"),
        ),
        "solB_ewc_minus_baseline_worst_final_forgetting": delta(
            stat(solb_ewc, "eval_matrix", "worst_final_forgetting"),
            stat(baseline, "eval_matrix", "worst_final_forgetting"),
        ),
        "solA_ewc_minus_baseline_BWT": delta(stat(sola_ewc, "BWT"), stat(baseline, "BWT")),
        "solA_ewc_minus_baseline_avg_mAP": delta(stat(sola_ewc, "avg_mAP"), stat(baseline, "avg_mAP")),
        "solA_ewc_minus_baseline_mean_final_forgetting": delta(
            stat(sola_ewc, "eval_matrix", "mean_final_forgetting"),
            stat(baseline, "eval_matrix", "mean_final_forgetting"),
        ),
        "solA_ewc_minus_baseline_worst_final_forgetting": delta(
            stat(sola_ewc, "eval_matrix", "worst_final_forgetting"),
            stat(baseline, "eval_matrix", "worst_final_forgetting"),
        ),
        "solA_ewc_minus_solB_ewc_BWT": delta(stat(sola_ewc, "BWT"), stat(solb_ewc, "BWT")),
        "solA_ewc_minus_solB_ewc_avg_mAP": delta(stat(sola_ewc, "avg_mAP"), stat(solb_ewc, "avg_mAP")),
    },
    "drift_performance_correlation_evidence": {
        "baseline": stat(baseline, "correlations"),
        "solB_ewc": stat(solb_ewc, "correlations"),
        "solA_ewc": stat(sola_ewc, "correlations"),
    },
    "event_window_evidence": {
        "baseline_high_drift_fg_loss_delta": mean_event_delta(
            baseline.get("tag"), "high_drift_events", "fg_align_loss_after_minus_before"
        ),
        "solB_ewc_high_drift_fg_loss_delta": mean_event_delta(
            solb_ewc.get("tag"), "high_drift_events", "fg_align_loss_after_minus_before"
        ),
        "baseline_domain_boundary_previous_AP_delta": mean_event_delta(
            baseline.get("tag"), "domain_boundary_events", "previous_domains_AP_delta"
        ),
        "solB_ewc_domain_boundary_previous_AP_delta": mean_event_delta(
            solb_ewc.get("tag"), "domain_boundary_events", "previous_domains_AP_delta"
        ),
        "baseline_domain_boundary_current_AP_delta": mean_event_delta(
            baseline.get("tag"), "domain_boundary_events", "current_domain_AP_delta"
        ),
        "solB_ewc_domain_boundary_current_AP_delta": mean_event_delta(
            solb_ewc.get("tag"), "domain_boundary_events", "current_domain_AP_delta"
        ),
        "solA_ewc_reset_event_count": stat(sola_ewc, "n_reset_events"),
        "solA_ewc_reset_fg_loss_delta": mean_event_delta(
            sola_ewc.get("tag"), "reset_events", "fg_align_loss_after_minus_before"
        ),
        "solA_ewc_reset_proto_drift_delta": mean_event_delta(
            sola_ewc.get("tag"), "reset_events", "proto_drift_after_minus_before"
        ),
        "solA_ewc_reset_domain_current_AP_delta": mean_event_delta(
            sola_ewc.get("tag"), "reset_domain_ap_events", "current_domain_AP_delta"
        ),
        "solA_ewc_reset_domain_previous_AP_delta": mean_event_delta(
            sola_ewc.get("tag"), "reset_domain_ap_events", "previous_domains_AP_delta"
        ),
    },
}

with open(os.path.join(RESULT_DIR, "exp18_direct_evidence.json"), "w") as fp:
    json.dump(direct, fp, indent=2)

print("\n=== Exp 18 Sol-B + EWC Direct Evidence ===")
print(f"{'run':<34} {'BWT':>8} {'FWT':>8} {'avg':>8} {'forget':>9} {'protoD':>8} {'adptF':>9}")
print("-" * 96)
for r in summary:
    def fmt(v, width=8, prec=3):
        return f"{float(v):>{width}.{prec}f}" if finite(v) else f"{'nan':>{width}}"
    print(
        f"{r['tag']:<34}"
        f" {fmt(r.get('BWT'))}"
        f" {fmt(r.get('FWT'))}"
        f" {fmt(r.get('avg_mAP'))}"
        f" {fmt(stat(r, 'eval_matrix', 'mean_final_forgetting'), 9)}"
        f" {fmt(r.get('proto_drift_mean'))}"
        f" {fmt(r.get('adapter_fisher_final'), 9)}"
    )

print("\nKey deltas:")
for key, value in direct["eval_matrix_bwt_evidence"].items():
    print(f"  {key}: {value:.4f}" if finite(value) else f"  {key}: nan")
print(f"\nSaved to {os.path.join(RESULT_DIR, 'exp18_direct_evidence.json')}")
PYEOF

echo ""
echo "=== Exp 18 complete. Results in results/exp18/ ==="
