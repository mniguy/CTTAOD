#!/usr/bin/env bash
# =============================================================================
# Exp 6: Adaptive ASRI — α_t = asri_alpha / (1 + ||μ_te - μ_tr|| / sqrt(tr(Σ)/d))
#
# β fixed at 0.999 (best from Exp 5 Run 1)
# α_max sweep: 0.1, 0.2, 0.3, 0.5
#
# Usage:
#   bash scripts/exp6_adaptive_asri.sh [CHECKPOINT_PATH]
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"
BETA="0.999"
BETA_TAG="0_999"

export DETECTRON2_DATASETS="$ROOT/datasets"
mkdir -p ../results/exp6

echo ""
echo "======================================================================"
echo "Exp 6: Adaptive ASRI sweep  (β=$BETA, ASRI_ADAPTIVE=True)"
echo "======================================================================"

ALPHAS="0.1 0.2 0.3 0.5"

for ALPHA in $ALPHAS; do
    ALPHA_TAG=$(echo "$ALPHA" | sed 's/\./_/g')

    echo ""
    echo "--- α_max = $ALPHA ---"

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
        TEST.ADAPTATION.ASRI_ADAPTIVE True \
        TEST.ADAPTATION.ORACLE_PROTOTYPE False \
        TEST.ADAPTATION.EMA_BETA "$BETA" \
        TEST.ADAPTATION.DPEMA_APPLY_GL True \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        OUTPUT_DIR "../outputs/COCO_TTA/exp6_adaptive_asri_beta_${BETA_TAG}_alphamax_${ALPHA_TAG}"

    cp "../outputs/COCO_TTA/exp6_adaptive_asri_beta_${BETA_TAG}_alphamax_${ALPHA_TAG}/eval_matrix/metrics.json" \
       "../results/exp6/metrics_adaptive_alphamax_${ALPHA_TAG}.json" 2>/dev/null || true
done

echo ""
echo "======================================================================"
echo "Results summary"
echo "======================================================================"

python3 - <<'PYEOF'
import json, os, glob

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"
]
BASELINE = {
    "gaussian_noise": 15.33, "shot_noise": 18.80, "impulse_noise": 18.00,
    "defocus_blur": 12.09, "glass_blur": 4.59, "motion_blur": 7.29,
    "zoom_blur": 3.37, "snow": 9.12, "frost": 11.69, "fog": 23.86,
    "brightness": 27.43, "contrast": 17.08, "elastic_transform": 17.52,
    "pixelate": 9.33, "jpeg_compression": 11.49,
}

def get_ap(d, c):
    return d.get(f"coco_2017_val-{c}", {}).get("AP", float("nan"))

def avg(d):
    return sum(get_ap(d, c) for c in CORRUPTIONS) / len(CORRUPTIONS)

result_files = sorted(glob.glob("../results/exp6/metrics_adaptive_alphamax_*.json"))
fixed_files  = sorted(glob.glob("../results/exp5/metrics_beta_0_999_alpha_*.json"),
                      key=lambda f: os.path.basename(f))
fixed_files  = [f for f in fixed_files if "alpha" in os.path.basename(f)]

base_avg = sum(BASELINE.values()) / len(CORRUPTIONS)

print(f"\n{'corruption':<22} {'baseline':>8}", end="")
for f in fixed_files:
    tag = os.path.basename(f).replace("metrics_beta_0_999_alpha_","").replace(".json","").replace("_",".")
    print(f" {'fixed α='+tag:>12}", end="")
for f in result_files:
    tag = os.path.basename(f).replace("metrics_adaptive_alphamax_","").replace(".json","").replace("_",".")
    print(f" {'adapt α≤'+tag:>12}", end="")
print()
print("-" * (22 + 9 + 13 * (len(fixed_files) + len(result_files))))

for c in CORRUPTIONS:
    marker = " <<" if c in ["snow", "frost"] else ""
    print(f"{c:<22} {BASELINE[c]:>8.2f}", end="")
    for f in fixed_files:
        d = json.load(open(f))
        print(f" {get_ap(d,c):>12.2f}", end="")
    for f in result_files:
        d = json.load(open(f))
        print(f" {get_ap(d,c):>12.2f}", end="")
    print(marker)

print("-" * (22 + 9 + 13 * (len(fixed_files) + len(result_files))))
print(f"{'avg(15)':<22} {base_avg:>8.2f}", end="")
for f in fixed_files:
    d = json.load(open(f))
    print(f" {avg(d):>12.2f}", end="")
for f in result_files:
    d = json.load(open(f))
    print(f" {avg(d):>12.2f}", end="")
print()
PYEOF

echo ""
echo "=== Exp 6 complete. Results in results/exp6/ ==="
