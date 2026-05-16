#!/usr/bin/env bash
# =============================================================================
# Exp 8: Confidence-gated ASRI
#
# Base: DPEMA (β=0.999, Exp 5 best) + ASRI_ALPHA=0.3
# Change: ASRI_CONFGATE=True — per-class α_t adapts based on detection count
#         × avg confidence, so blur/sparse images retain source anchor while
#         dense/confident images reduce α to trust the target prototype.
#
#   α_t = asri_confgate_min + (asri_alpha - asri_confgate_min)
#         × exp(-ASRI_CONFGATE_LAMBDA × N_k × avg_score_k)
#
# Run 1 — λ sweep (asri_alpha=0.3 fixed): λ ∈ {0.5, 1.0, 2.0, 5.0}
# Run 2 — α_max sweep (best λ from Run 1): α ∈ {0.1, 0.2, 0.3, 0.5}
#
# Baseline reference (Exp 5): DPEMA β=0.999, fixed α=0.3  →  avg AP 19.74
#
# Usage:
#   bash scripts/exp8_confgate_asri.sh [CHECKPOINT_PATH]
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"
BETA="0.999"
ALPHA_FIXED="0.3"

export DETECTRON2_DATASETS="$ROOT/datasets"
mkdir -p ../results/exp8

# =============================================================================
# Run 1: λ sweep  (asri_alpha=0.3, asri_confgate_min=0.0)
# =============================================================================
echo ""
echo "======================================================================"
echo "Exp 8 Run 1: ASRI_CONFGATE λ sweep  (α=${ALPHA_FIXED}, β=${BETA})"
echo "======================================================================"

LAMBDAS="0.5 1.0 2.0 5.0"

for LAMBDA in $LAMBDAS; do
    LAMBDA_TAG=$(echo "$LAMBDA" | sed 's/\./_/g')
    ALPHA_TAG=$(echo "$ALPHA_FIXED" | sed 's/\./_/g')

    echo ""
    echo "--- λ=${LAMBDA}  α=${ALPHA_FIXED} ---"

    python train_net.py \
        --config-file "$CFG" \
        --eval-only \
        MODEL.WEIGHTS "$CKPT" \
        TEST.ONLINE_ADAPTATION True \
        TEST.CONTINUAL_DOMAIN True \
        TEST.EVAL_MATRIX True \
        TEST.ADAPTATION.CONTINUAL True \
        TEST.ADAPTATION.WHERE "adapter" \
        TEST.ADAPTATION.GLOBAL_ALIGN "KL" \
        TEST.ADAPTATION.FOREGROUND_ALIGN "KL" \
        TEST.ADAPTATION.ASRI_ALPHA "$ALPHA_FIXED" \
        TEST.ADAPTATION.ASRI_CONFGATE True \
        TEST.ADAPTATION.ASRI_CONFGATE_LAMBDA "$LAMBDA" \
        TEST.ADAPTATION.ASRI_CONFGATE_MIN 0.0 \
        TEST.ADAPTATION.ORACLE_PROTOTYPE False \
        TEST.ADAPTATION.EMA_BETA "$BETA" \
        TEST.ADAPTATION.DPEMA_APPLY_GL True \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        OUTPUT_DIR "../outputs/COCO_TTA/exp8_confgate_lambda_${LAMBDA_TAG}_alpha_${ALPHA_TAG}"

    cp "../outputs/COCO_TTA/exp8_confgate_lambda_${LAMBDA_TAG}_alpha_${ALPHA_TAG}/eval_matrix/metrics.json" \
       "../results/exp8/metrics_lambda_${LAMBDA_TAG}_alpha_${ALPHA_TAG}.json" 2>/dev/null || true
done

# =============================================================================
# Find best λ
# =============================================================================
BEST_LAMBDA=$(python3 - <<'PYEOF'
import json, glob, os

CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
    "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression"
]

def avg(path):
    d = json.load(open(path))
    return sum(d.get(f"coco_2017_val-{c}", {}).get("AP", 0.) for c in CORRUPTIONS) / len(CORRUPTIONS)

files = sorted(glob.glob("../results/exp8/metrics_lambda_*_alpha_0_3.json"))
if not files:
    print("1.0")
else:
    best = max(files, key=avg)
    tag = os.path.basename(best).replace("metrics_lambda_","").split("_alpha_")[0]
    print(tag.replace("_","."))
PYEOF
)
echo ""
echo "Best λ from Run 1: $BEST_LAMBDA"

# =============================================================================
# Run 2: α_max sweep  (best λ)
# =============================================================================
echo ""
echo "======================================================================"
echo "Exp 8 Run 2: α_max sweep  (λ=${BEST_LAMBDA})"
echo "======================================================================"

ALPHAS="0.1 0.2 0.3 0.5"
LAMBDA_TAG=$(echo "$BEST_LAMBDA" | sed 's/\./_/g')

for ALPHA in $ALPHAS; do
    ALPHA_TAG=$(echo "$ALPHA" | sed 's/\./_/g')

    # skip if already run in Run 1
    OUT="../results/exp8/metrics_lambda_${LAMBDA_TAG}_alpha_${ALPHA_TAG}.json"
    if [ -f "$OUT" ]; then
        echo "--- Skipping α=${ALPHA} (already exists) ---"
        continue
    fi

    echo ""
    echo "--- λ=${BEST_LAMBDA}  α=${ALPHA} ---"

    python train_net.py \
        --config-file "$CFG" \
        --eval-only \
        MODEL.WEIGHTS "$CKPT" \
        TEST.ONLINE_ADAPTATION True \
        TEST.CONTINUAL_DOMAIN True \
        TEST.EVAL_MATRIX True \
        TEST.ADAPTATION.CONTINUAL True \
        TEST.ADAPTATION.WHERE "adapter" \
        TEST.ADAPTATION.GLOBAL_ALIGN "KL" \
        TEST.ADAPTATION.FOREGROUND_ALIGN "KL" \
        TEST.ADAPTATION.ASRI_ALPHA "$ALPHA" \
        TEST.ADAPTATION.ASRI_CONFGATE True \
        TEST.ADAPTATION.ASRI_CONFGATE_LAMBDA "$BEST_LAMBDA" \
        TEST.ADAPTATION.ASRI_CONFGATE_MIN 0.0 \
        TEST.ADAPTATION.ORACLE_PROTOTYPE False \
        TEST.ADAPTATION.EMA_BETA "$BETA" \
        TEST.ADAPTATION.DPEMA_APPLY_GL True \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        OUTPUT_DIR "../outputs/COCO_TTA/exp8_confgate_lambda_${LAMBDA_TAG}_alpha_${ALPHA_TAG}"

    cp "../outputs/COCO_TTA/exp8_confgate_lambda_${LAMBDA_TAG}_alpha_${ALPHA_TAG}/eval_matrix/metrics.json" \
       "../results/exp8/metrics_lambda_${LAMBDA_TAG}_alpha_${ALPHA_TAG}.json" 2>/dev/null || true
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "======================================================================"
echo "Results summary"
echo "======================================================================"

python3 - <<'PYEOF'
import json, os, glob

CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
    "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression"
]
EXP5_BEST = 19.74  # DPEMA β=0.999, fixed α=0.3

def get_ap(d, c):
    return d.get(f"coco_2017_val-{c}", {}).get("AP", float("nan"))

def avg(d):
    return sum(get_ap(d, c) for c in CORRUPTIONS) / len(CORRUPTIONS)

# Run 1 table: λ sweep
print("\n=== Run 1: λ sweep (α=0.3) ===")
lam_files = sorted(glob.glob("../results/exp8/metrics_lambda_*_alpha_0_3.json"))
print(f"{'λ':<8} {'avg AP':>8}  defoc   glass   zoom   gauss")
print(f"{'exp5*':<8} {EXP5_BEST:>8.2f}  11.75   6.66    5.63   10.86  (fixed α=0.3)")
for f in lam_files:
    d = json.load(open(f))
    lam = os.path.basename(f).replace("metrics_lambda_","").split("_alpha_")[0].replace("_",".")
    print(f"λ={lam:<6} {avg(d):>8.2f}"
          f"  {get_ap(d,'defocus_blur'):>5.2f}  {get_ap(d,'glass_blur'):>5.2f}"
          f"  {get_ap(d,'zoom_blur'):>5.2f}  {get_ap(d,'gaussian_noise'):>5.2f}")

# Run 2 table: α sweep with best λ
all_files = sorted(glob.glob("../results/exp8/metrics_lambda_*.json"))
if all_files:
    best_f = max(all_files, key=lambda f: avg(json.load(open(f))))
    best_lam_tag = os.path.basename(best_f).replace("metrics_lambda_","").split("_alpha_")[0]
    alpha_files = sorted(glob.glob(f"../results/exp8/metrics_lambda_{best_lam_tag}_alpha_*.json"))
    if len(alpha_files) > 1:
        print(f"\n=== Run 2: α_max sweep (λ={best_lam_tag.replace('_','.')}) ===")
        print(f"{'α_max':<8} {'avg AP':>8}  defoc   glass   zoom   gauss")
        for f in alpha_files:
            d = json.load(open(f))
            alpha = os.path.basename(f).split("_alpha_")[1].replace(".json","").replace("_",".")
            print(f"α={alpha:<7} {avg(d):>8.2f}"
                  f"  {get_ap(d,'defocus_blur'):>5.2f}  {get_ap(d,'glass_blur'):>5.2f}"
                  f"  {get_ap(d,'zoom_blur'):>5.2f}  {get_ap(d,'gaussian_noise'):>5.2f}")

    # Overall best
    best_d = json.load(open(best_f))
    print(f"\nBest confgate: {os.path.basename(best_f)}")
    print(f"  avg AP = {avg(best_d):.2f}  (Exp5 fixed={EXP5_BEST:.2f},"
          f" Δ={avg(best_d)-EXP5_BEST:+.2f})")
    print(f"  defocus_blur  : {get_ap(best_d,'defocus_blur'):.2f}  (Exp5: 11.75, source: 12.58)")
    print(f"  glass_blur    : {get_ap(best_d,'glass_blur'):.2f}  (Exp5: 6.66)")
    print(f"  zoom_blur     : {get_ap(best_d,'zoom_blur'):.2f}  (Exp5: 5.63)")
PYEOF

echo ""
echo "=== Exp 8 complete. Results in results/exp8/ ==="
