#!/usr/bin/env bash
# =============================================================================
# Exp 10: Adapter Drift Prevention + Pseudo-Label Quality Gate
#
# Tests three additions that target the two failure modes identified earlier:
#   (a) adapter weights drift toward each new corruption →  EWC on adapter
#   (b) noisy / class-imbalanced pseudo-labels poison the EMA prototype →
#         Confidence-Weighted Prototype Update
#         Class-Balanced Subsampling for Prototype
#
# Reference base (same as Exp 9): COCO R50, DPEMA β=0.999, ASRI_GL=True,
#   fg + global KL alignment.
#
# Runs:
#   R0  Baseline CTTAOD (control)               — no extra regularization
#   R1  EWC on adapter, λ ∈ {0.1, 1.0, 10.0, 100.0}
#   R2  Confidence-Weighted Prototype Update
#       R2a soft (γ=1.0)
#       R2b hard (threshold=0.7)
#       R2c hard (threshold=0.5)
#   R3  Class-Balanced Subsampling for Prototype
#       R3a max_per_class=4
#       R3b max_per_class=8
#       R3c max_per_class=8 + inv_freq
#   R4  Combined: best EWC λ* + best ConfProto + best CB
#
# Evaluated on COCO → COCO-C (15 corruptions, same as Exp 5–9).
#
# Outputs:
#   results/exp10/metrics_<tag>.json    — per-corruption AP + avg_mAP/BWT/FWT
#   results/exp10/eval_matrix_<tag>.npy
#   results/exp10/exp10_summary.json
#
# Usage:
#   bash scripts/exp10_drift_quality.sh [CHECKPOINT_PATH]
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"
FISHER_PATH="../models/stats/COCO_R50_fisher.pt"

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/datasets}"
mkdir -p ../results/exp10

# Common args shared by every run --------------------------------------------
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
    TEST.ADAPTATION.EMA_BETA 0.999
    TEST.ADAPTATION.DPEMA_APPLY_GL True
    TEST.ADAPTATION.ASRI_ALPHA 0.0
    TEST.ADAPTATION.ADAPTER_RESET False
    TEST.ADAPTATION.ORACLE_PROTOTYPE False
)

CORRUPTIONS=(
    gaussian_noise shot_noise impulse_noise defocus_blur glass_blur
    motion_blur zoom_blur snow frost fog brightness contrast
    elastic_transform pixelate jpeg_compression
)

# helper: copy metrics + eval matrix into results/exp10 ----------------------
collect() {
    local TAG="$1"
    local OUT="$2"
    cp "${OUT}/eval_matrix/metrics.json"    "../results/exp10/metrics_${TAG}.json"    2>/dev/null || true
    cp "${OUT}/eval_matrix/eval_matrix.npy" "../results/exp10/eval_matrix_${TAG}.npy" 2>/dev/null || true
}

# helper: avg AP across 15 corruptions from a metrics.json ------------------
avg_ap() {
    python3 - "$1" <<'PYEOF'
import json, sys, math
CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
    "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression"
]
d = json.load(open(sys.argv[1]))
vals = [d.get(f"coco_2017_val-{c}", {}).get("AP", float("nan")) for c in CORRUPTIONS]
vals = [v for v in vals if not math.isnan(v)]
print(f"{sum(vals)/len(CORRUPTIONS):.4f}" if vals else "nan")
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Pre-step 0 : compute Fisher (only if missing) — required for EWC runs
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "$FISHER_PATH" ]; then
    echo "=== Pre-step: computing Fisher information for adapter ==="
    python compute_fisher.py \
        --config-file "$CFG" \
        MODEL.WEIGHTS "$CKPT" \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        TEST.ADAPTATION.WHERE "adapter"
fi

# ─────────────────────────────────────────────────────────────────────────────
# R0 : Baseline CTTAOD (control) — already completed, skip
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# R1 : EWC on adapter — λ sweep — already completed, skip
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# R2 : Confidence-Weighted Prototype Update
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== R2 : Confidence-Weighted Prototype Update ==="

# R2a soft, γ=1.0
OUT="../outputs/COCO_TTA/exp10_r2a_conf_soft"
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.CONF_PROTO True \
    TEST.ADAPTATION.CONF_PROTO_MODE "soft" \
    TEST.ADAPTATION.CONF_PROTO_GAMMA 1.0 \
    OUTPUT_DIR "$OUT"
collect "r2a_conf_soft" "$OUT"

# R2b hard, threshold=0.7
OUT="../outputs/COCO_TTA/exp10_r2b_conf_hard07"
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.CONF_PROTO True \
    TEST.ADAPTATION.CONF_PROTO_MODE "hard" \
    TEST.ADAPTATION.CONF_PROTO_THRESHOLD 0.7 \
    OUTPUT_DIR "$OUT"
collect "r2b_conf_hard07" "$OUT"

# R2c hard, threshold=0.5
OUT="../outputs/COCO_TTA/exp10_r2c_conf_hard05"
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.CONF_PROTO True \
    TEST.ADAPTATION.CONF_PROTO_MODE "hard" \
    TEST.ADAPTATION.CONF_PROTO_THRESHOLD 0.5 \
    OUTPUT_DIR "$OUT"
collect "r2c_conf_hard05" "$OUT"

# ─────────────────────────────────────────────────────────────────────────────
# R3 : Class-Balanced Subsampling
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== R3 : Class-Balanced Subsampling ==="

OUT="../outputs/COCO_TTA/exp10_r3a_cb4"
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.CB_PROTO True \
    TEST.ADAPTATION.CB_PROTO_MAX_PER_CLASS 4 \
    OUTPUT_DIR "$OUT"
collect "r3a_cb4" "$OUT"

OUT="../outputs/COCO_TTA/exp10_r3b_cb8"
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.CB_PROTO True \
    TEST.ADAPTATION.CB_PROTO_MAX_PER_CLASS 8 \
    OUTPUT_DIR "$OUT"
collect "r3b_cb8" "$OUT"

OUT="../outputs/COCO_TTA/exp10_r3c_cb8_invfreq"
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.CB_PROTO True \
    TEST.ADAPTATION.CB_PROTO_MAX_PER_CLASS 8 \
    TEST.ADAPTATION.CB_PROTO_INV_FREQ True \
    OUTPUT_DIR "$OUT"
collect "r3c_cb8_invfreq" "$OUT"

# ─────────────────────────────────────────────────────────────────────────────
# Pick the best variant per family by avg AP across 15 corruptions
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Picking best per-family variants for R4 ==="
read BEST_EWC BEST_CONF BEST_CB < <(python3 - <<'PYEOF'
import json, glob, os, re, math

CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
    "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression"
]

def avg_ap(path):
    try:
        d = json.load(open(path))
    except Exception:
        return float("nan")
    vals = [d.get(f"coco_2017_val-{c}", {}).get("AP", float("nan")) for c in CORRUPTIONS]
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(CORRUPTIONS) if vals else float("nan")

def best(prefix):
    bf, bm = None, -1e9
    for f in glob.glob(f"../results/exp10/metrics_{prefix}*.json"):
        m = avg_ap(f)
        if not math.isnan(m) and m > bm:
            bm, bf = m, f
    return bf

def tag(path):
    return re.search(r"metrics_([^/]+?)\.json$", path).group(1) if path else "NONE"

print(tag(best("r1_ewc")), tag(best("r2")), tag(best("r3")))
PYEOF
)
echo "  best EWC  = $BEST_EWC"
echo "  best Conf = $BEST_CONF"
echo "  best CB   = $BEST_CB"

# Map tag → CLI overrides
ewc_args_from_tag() {
    case "$1" in
        r1_ewc0_1)   echo "TEST.ADAPTATION.EWC_LAMBDA 0.1   TEST.ADAPTATION.EWC_FISHER_PATH $FISHER_PATH" ;;
        r1_ewc1_0)   echo "TEST.ADAPTATION.EWC_LAMBDA 1.0   TEST.ADAPTATION.EWC_FISHER_PATH $FISHER_PATH" ;;
        r1_ewc10_0)  echo "TEST.ADAPTATION.EWC_LAMBDA 10.0  TEST.ADAPTATION.EWC_FISHER_PATH $FISHER_PATH" ;;
        r1_ewc100_0) echo "TEST.ADAPTATION.EWC_LAMBDA 100.0 TEST.ADAPTATION.EWC_FISHER_PATH $FISHER_PATH" ;;
        *) echo "" ;;
    esac
}
conf_args_from_tag() {
    case "$1" in
        r2a_conf_soft)   echo "TEST.ADAPTATION.CONF_PROTO True TEST.ADAPTATION.CONF_PROTO_MODE soft TEST.ADAPTATION.CONF_PROTO_GAMMA 1.0" ;;
        r2b_conf_hard07) echo "TEST.ADAPTATION.CONF_PROTO True TEST.ADAPTATION.CONF_PROTO_MODE hard TEST.ADAPTATION.CONF_PROTO_THRESHOLD 0.7" ;;
        r2c_conf_hard05) echo "TEST.ADAPTATION.CONF_PROTO True TEST.ADAPTATION.CONF_PROTO_MODE hard TEST.ADAPTATION.CONF_PROTO_THRESHOLD 0.5" ;;
        *) echo "" ;;
    esac
}
cb_args_from_tag() {
    case "$1" in
        r3a_cb4)         echo "TEST.ADAPTATION.CB_PROTO True TEST.ADAPTATION.CB_PROTO_MAX_PER_CLASS 4" ;;
        r3b_cb8)         echo "TEST.ADAPTATION.CB_PROTO True TEST.ADAPTATION.CB_PROTO_MAX_PER_CLASS 8" ;;
        r3c_cb8_invfreq) echo "TEST.ADAPTATION.CB_PROTO True TEST.ADAPTATION.CB_PROTO_MAX_PER_CLASS 8 TEST.ADAPTATION.CB_PROTO_INV_FREQ True" ;;
        *) echo "" ;;
    esac
}

EWC_OV=$(ewc_args_from_tag  "$BEST_EWC")
CONF_OV=$(conf_args_from_tag "$BEST_CONF")
CB_OV=$(cb_args_from_tag    "$BEST_CB")

# ─────────────────────────────────────────────────────────────────────────────
# R4 : Combined best EWC + best Conf + best CB
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== R4 : Combined (EWC + Conf + CB) ==="
OUT="../outputs/COCO_TTA/exp10_r4_combined"
python train_net.py "${BASE_ARGS[@]}" \
    $EWC_OV $CONF_OV $CB_OV \
    OUTPUT_DIR "$OUT"
collect "r4_combined" "$OUT"

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Exp 10 Summary ==="
python3 - <<'PYEOF'
import json, glob, os, math

CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
    "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression"
]

def get_ap(d, c):
    return d.get(f"coco_2017_val-{c}", {}).get("AP", float("nan"))

def avg_ap(d):
    vals = [get_ap(d, c) for c in CORRUPTIONS]
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(CORRUPTIONS) if vals else float("nan")

rows = []
for f in sorted(glob.glob("../results/exp10/metrics_*.json")):
    tag = os.path.basename(f).replace("metrics_", "").replace(".json", "")
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rows.append((tag, d, avg_ap(d)))

print(f"\n{'Run':<24} {'avg AP':>8}  gauss   defoc   glass   fog     snow")
print("-" * 72)
for tag, d, ap in rows:
    print(f"{tag:<24} {ap:>8.2f}"
          f"  {get_ap(d,'gaussian_noise'):>5.2f}"
          f"  {get_ap(d,'defocus_blur'):>5.2f}"
          f"  {get_ap(d,'glass_blur'):>5.2f}"
          f"  {get_ap(d,'fog'):>5.2f}"
          f"  {get_ap(d,'snow'):>5.2f}")

print("\n--- Per-corruption breakdown (best run) ---")
if rows:
    best_tag, best_d, best_ap = max(rows, key=lambda x: x[2] if not math.isnan(x[2]) else -1e9)
    print(f"Best: {best_tag}  (avg AP = {best_ap:.2f})\n")
    print(f"{'corruption':<24} {'AP':>8}")
    print("-" * 34)
    for c in CORRUPTIONS:
        print(f"{c:<24} {get_ap(best_d, c):>8.2f}")

with open("../results/exp10/exp10_summary.json", "w") as fp:
    json.dump(
        [{"tag": t, "avg_AP": ap,
          **{c: get_ap(d, c) for c in CORRUPTIONS}}
         for t, d, ap in rows],
        fp, indent=2
    )
print("\nSaved to ../results/exp10/exp10_summary.json")
PYEOF

echo ""
echo "=== Exp 10 complete. Results in results/exp10/ ==="
