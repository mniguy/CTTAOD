#!/usr/bin/env bash
# =============================================================================
# Exp 21: ResNet-50 Sol-B + EWC — alpha & lambda hyperparameter sweep
#
# Purpose:
#   Sweep the source-anchor blend (alpha) and adapter-EWC strength (lambda) for
#   Sol-B + EWC on the COCO -> COCO-C continual-corruption benchmark with the
#   ResNet-50 backbone, mirroring the Swin-T sweep in exp20/exp22.
#
# Reproduction target:
#   The anchor point alpha=0.40, lambda=10.0 targets the previously reported
#   Sol-B + EWC ResNet-50 result of mean(16)=22.87 (= AP16, the mean over the 15
#   corruptions + clean coco_2017_val).
#   That number originally came from exp13 E4 (results/exp13/metrics_e4_ewc10_solB_a0_4.json,
#   AP16=22.87, AP15=21.53, clean=42.96) using:
#       PROTO_METHOD dual_memory + SOURCE_ANCHOR_ALPHA 0.4 + EWC_LAMBDA 10.0
#       WHERE adapter, GLOBAL/FOREGROUND_ALIGN KL, EMA_BETA 0, SWEMA_K 0, ASRI_ALPHA 0.
#   This script reuses that adaptation setting and only varies alpha / lambda.
#   Exact AP is not guaranteed unless the same checkpoint, stats/fisher files,
#   corrupted image folders, seed, and code path are used.
#
# NOTE on SKIP_REDUNDANT:
#   exp13 did not override SKIP_REDUNDANT, so it inherited COCO_R50.yaml's default
#   "stat-period-ema" (redundant updates skipped). That is part of the 22.87 setting,
#   so it is set explicitly here. (This differs from the broken exp23 R50 attempt,
#   which forced SKIP_REDUNDANT None -> backward_images=80000 and low AP16.)
#
# NOTE on EVAL_MATRIX:
#   exp13 set TEST.EVAL_MATRIX True, but in the current code path EVAL_MATRIX True
#   on a coco dataset dispatches to test_continual_domain_eval_matrix, which writes
#   a BWT/FWT metrics.json (NOT the coco_2017_val-<corruption> format the AP16
#   summary needs). The mean(16)=22.87 metrics actually come from the standard
#   test_continual_domain path (it always writes eval_matrix/metrics.json in the
#   coco_2017_val-<corruption> format). So EVAL_MATRIX is explicitly set to False
#   here to force the AP16 metrics path.
#
# Runs (you choose the exact set):
#   Edit the DEFAULT_PAIRS list below — each entry is "ALPHA LAMBDA" — or override
#   it on the command line via EXP21_PAIRS (see Optional env). Duplicate pairs are
#   skipped. The pair (0.40, 10.0) is the 22.87 reference point.
#
# Outputs per run (tag = alpha<a>_lam<l>):
#   results/exp21/metrics_<tag>.json     coco_2017_val-<corruption> + coco_2017_val AP
#   results/exp21/meta_<tag>.json        backward images / fps
#   results/exp21/drift_<tag>.jsonl      per-step drift diagnostics, if enabled
#
# Summary outputs:
#   results/exp21/r50_summary.json
#   results/exp21/r50_summary_table.md
#
# Usage:
#   bash scripts/exp21_resnet.sh [CHECKPOINT_PATH]
#
# Optional env:
#   EXP21_PAIRS="0.4:10.0 0.5:10.0 0.3:1.0"   set of alpha:lambda runs to launch
#                                             (space- or comma-separated; overrides DEFAULT_PAIRS)
#   EXP21_SEED=0                               set non-negative seed for reproducibility
#   EXP21_DRIFT_LOG=True                       enable drift diagnostics
#   EXP21_LOG_PERIOD=10                        drift log interval when enabled
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

EXP21_LOG_PERIOD="${EXP21_LOG_PERIOD:-10}"
EXP21_SEED="${EXP21_SEED:--1}"
EXP21_DRIFT_LOG="${EXP21_DRIFT_LOG:-False}"

# ── Choose which (alpha, lambda) runs to launch ──────────────────────────────
# Each entry is "ALPHA LAMBDA". Edit this list freely. (0.4 10.0) is the reference point.
DEFAULT_PAIRS=(
    "0.4 10.0"
    "0.4 1.0"
)
# EXP21_PAIRS, if set, overrides DEFAULT_PAIRS: space/comma-separated "alpha:lambda".
if [ -n "${EXP21_PAIRS:-}" ]; then
    PAIRS=()
    for tok in ${EXP21_PAIRS//,/ }; do
        PAIRS+=("${tok%%:*} ${tok##*:}")
    done
else
    PAIRS=("${DEFAULT_PAIRS[@]}")
fi

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/datasets}"
mkdir -p ../results/exp21

tag_float() {
    echo "$1" | tr '.' '_'
}

collect() {
    local tag="$1"
    local out="$2"
    cp "${out}/eval_matrix/metrics.json"  "../results/exp21/metrics_${tag}.json" 2>/dev/null || true
    cp "${out}/eval_matrix/run_meta.json" "../results/exp21/meta_${tag}.json"    2>/dev/null || true
    cp "${out}/drift/drift_log.jsonl"     "../results/exp21/drift_${tag}.jsonl"  2>/dev/null || true
}

# Matches the exp13 E4 adaptation setting as closely as the current code path
# allows; alpha/lambda are overridden per run.
COMMON_ARGS=(
    --config-file "$CFG"
    --eval-only
    SEED "$EXP21_SEED"
    MODEL.WEIGHTS "$CKPT"
    TEST.ONLINE_ADAPTATION True
    TEST.CONTINUAL_DOMAIN True
    TEST.EVAL_MATRIX False
    TEST.ADAPTATION.CONTINUAL True
    TEST.ADAPTATION.WHERE "adapter"
    TEST.ADAPTATION.GLOBAL_ALIGN "KL"
    TEST.ADAPTATION.FOREGROUND_ALIGN "KL"
    TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH"
    TEST.ADAPTATION.EWC_FISHER_PATH "$FISHER_PATH"
    TEST.ADAPTATION.PROTO_METHOD "dual_memory"
    TEST.ADAPTATION.EMA_BETA 0.0
    TEST.ADAPTATION.SWEMA_K 0
    TEST.ADAPTATION.ASRI_ALPHA 0.0
    TEST.ADAPTATION.ADAPTER_RESET False
    TEST.ADAPTATION.ORACLE_PROTOTYPE False
    TEST.ADAPTATION.SKIP_REDUNDANT "stat-period-ema"
    TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA_ADAPTIVE False
    TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE False
    TEST.ADAPTATION.DRIFT_LOG "$EXP21_DRIFT_LOG"
    TEST.ADAPTATION.DRIFT_LOG_PERIOD "$EXP21_LOG_PERIOD"
)

ensure_fisher() {
    if [ ! -f "$FISHER_PATH" ]; then
        echo "=== Pre-step: computing ResNet-50 Fisher for adapter EWC ==="
        "$PYTHON_BIN" compute_fisher.py \
            --config-file "$CFG" \
            MODEL.WEIGHTS "$CKPT" \
            TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
            TEST.ADAPTATION.WHERE "adapter"
    fi
}

SEEN_TAGS=" "

run_point() {
    local alpha="$1"
    local lambda="$2"
    local tag="alpha$(tag_float "$alpha")_lam$(tag_float "$lambda")"
    if [[ "$SEEN_TAGS" == *" $tag "* ]]; then
        echo "=== Skip ${tag}: already run ==="
        return
    fi
    SEEN_TAGS="${SEEN_TAGS}${tag} "
    local out="../outputs/COCO_TTA/exp21_${tag}"
    echo ""
    echo "=== Sol-B + EWC (R50): alpha=${alpha}, lambda=${lambda} ==="
    "$PYTHON_BIN" train_net.py "${COMMON_ARGS[@]}" \
        TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA "$alpha" \
        TEST.ADAPTATION.EWC_LAMBDA "$lambda" \
        OUTPUT_DIR "$out"
    collect "$tag" "$out"
}

ensure_fisher

for pair in "${PAIRS[@]}"; do
    read -r alpha lambda <<< "$pair"
    run_point "$alpha" "$lambda"
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary: AP15 / AP16 tables + alpha/lambda grids
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Exp 21 Summary ==="
"$PYTHON_BIN" - <<'PYEOF'
import glob
import json
import math
import os
import re
from statistics import mean

RESULT_DIR = "../results/exp21"

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
FAMILIES = {
    "Noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "Blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "Weather": ["snow", "frost", "fog", "brightness"],
    "Digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
SHORT = ["Gau", "Sht", "Imp", "Def", "Gls", "Mtn", "Zm", "Snw", "Frs", "Fog",
         "Brt", "Cnt", "Els", "Px", "Jpg"]


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def avg(vals):
    vals = [v for v in vals if finite(v)]
    return mean(vals) if vals else None


def read_json(path):
    try:
        with open(path) as fp:
            return json.load(fp)
    except Exception:
        return {}


def get_ap(metrics, domain):
    key = "coco_2017_val" if domain == "original" else f"coco_2017_val-{domain}"
    val = metrics.get(key)
    if isinstance(val, dict):
        if "bbox" in val and isinstance(val["bbox"], dict):
            val = val["bbox"]
        if finite(val.get("AP")):
            return float(val["AP"])
    return None


def parse_tag(tag):
    # tag like "alpha0_4_lam10_0"
    m = re.match(r"alpha(.+)_lam(.+)", tag)
    if not m:
        return None, None
    alpha = float(m.group(1).replace("_", "."))
    lam = float(m.group(2).replace("_", "."))
    return alpha, lam


runs = []
for path in sorted(glob.glob(os.path.join(RESULT_DIR, "metrics_*.json"))):
    tag = os.path.basename(path).replace("metrics_", "").replace(".json", "")
    alpha, lam = parse_tag(tag)
    metrics = read_json(path)
    meta = read_json(os.path.join(RESULT_DIR, f"meta_{tag}.json"))
    per_domain = {d: get_ap(metrics, d) for d in CORRUPTIONS}
    clean = get_ap(metrics, "original")
    ap15 = avg(per_domain.values())
    ap16 = avg(list(per_domain.values()) + [clean])
    fam = {name: avg(per_domain.get(d) for d in ds) for name, ds in FAMILIES.items()}
    runs.append({
        "tag": tag, "alpha": alpha, "lambda": lam,
        "AP15": ap15, "AP16": ap16, "clean": clean,
        "per_domain_AP": per_domain, "family_AP": fam,
        "backward_images": meta.get("backward_images"),
        "fps": meta.get("fps"),
    })

best = max(runs, key=lambda r: r["AP16"] if finite(r.get("AP16")) else -1e9, default=None)
summary = {
    "summary": {
        "best_tag": best["tag"] if best else None,
        "best_AP16": best["AP16"] if best else None,
        "reference_22_87": {"alpha": 0.4, "lambda": 10.0},
        "num_runs": len(runs),
    },
    "runs": runs,
}
with open(os.path.join(RESULT_DIR, "r50_summary.json"), "w") as fp:
    json.dump(summary, fp, indent=2)


def fmt(v, p=2):
    return f"{v:.{p}f}" if finite(v) else "n/a"


by_key = {(r["alpha"], r["lambda"]): r for r in runs}
alphas = sorted({r["alpha"] for r in runs if r["alpha"] is not None})
lambdas = sorted({r["lambda"] for r in runs if r["lambda"] is not None})

lines = [
    "# Exp21 ResNet-50 Sol-B + EWC: Alpha & Lambda Runs",
    "",
    "AP15 averages the 15 corruptions. AP16 also includes clean coco_2017_val.",
    "Reference: alpha=0.40, lambda=10.00 is the historical mean(16)=22.87 point.",
    "",
    "### All runs, sorted by AP16",
    "",
    "| rank | alpha | lambda | AP15 | AP16 | clean | Noise | Blur | Weather | Digital |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for i, r in enumerate(
    sorted(runs, key=lambda r: r["AP16"] if finite(r.get("AP16")) else -1e9, reverse=True), 1
):
    f = r["family_AP"]
    lines.append(
        f"| {i} | {fmt(r['alpha'])} | {fmt(r['lambda'])} | {fmt(r['AP15'])} | {fmt(r['AP16'])} | "
        f"{fmt(r['clean'])} | {fmt(f['Noise'])} | {fmt(f['Blur'])} | {fmt(f['Weather'])} | {fmt(f['Digital'])} |"
    )

for metric in ("AP15", "AP16"):
    lines += [
        "",
        f"### 2D grid — {metric} (rows: alpha, cols: lambda)",
        "",
        "| α \\ λ | " + " | ".join(fmt(l) for l in lambdas) + " |",
        "|---:" * (len(lambdas) + 1) + "|",
    ]
    for a in alphas:
        cells = []
        for l in lambdas:
            r = by_key.get((a, l))
            cells.append(fmt(r[metric]) if r else "n/a")
        lines.append(f"| **{fmt(a)}** | " + " | ".join(cells) + " |")

lines += [
    "",
    "### Per-domain AP (sorted by AP16)",
    "",
    "| alpha | lambda | AP15 | AP16 | " + " | ".join(SHORT) + " | Cln |",
    "|---:|---:|---:|---:" + "|---:" * (len(SHORT) + 1) + "|",
]
for r in sorted(runs, key=lambda r: r["AP16"] if finite(r.get("AP16")) else -1e9, reverse=True):
    vals = [r["per_domain_AP"].get(d) for d in CORRUPTIONS] + [r["clean"]]
    lines.append(
        f"| {fmt(r['alpha'])} | {fmt(r['lambda'])} | {fmt(r['AP15'])} | {fmt(r['AP16'])} | "
        + " | ".join(fmt(v, 1) for v in vals) + " |"
    )

with open(os.path.join(RESULT_DIR, "r50_summary_table.md"), "w") as fp:
    fp.write("\n".join(lines) + "\n")

print(f"\n{'tag':<22} {'alpha':>6} {'lambda':>7} {'AP15':>6} {'AP16':>6} {'clean':>6}")
print("-" * 60)
for r in sorted(runs, key=lambda r: r["AP16"] if finite(r.get("AP16")) else -1e9, reverse=True):
    print(f"{r['tag']:<22} {fmt(r['alpha']):>6} {fmt(r['lambda']):>7} "
          f"{fmt(r['AP15']):>6} {fmt(r['AP16']):>6} {fmt(r['clean']):>6}")
if best:
    print(f"\nbest: {best['tag']}  AP16={fmt(best['AP16'])}")
print(f"Saved {os.path.join(RESULT_DIR, 'r50_summary.json')} and r50_summary_table.md")
PYEOF

echo ""
echo "=== Exp 21 complete. Results in results/exp21/ ==="
