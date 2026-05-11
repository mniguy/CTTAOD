#!/usr/bin/env bash
# =============================================================================
# Exp 7: SWEMA — Sliding Window EMA Prototype Memory Bank
#
# μ̃ = (1 - swema_alpha) * μ_te_recent + swema_alpha * μ_tr
#   μ_te_recent : EMA of the most recent K images only  (windowed, not full history)
#   μ_tr        : fixed source prototype anchor
#
# Two sweeps:
#   Run 1 — K sweep  (swema_alpha=0.1 fixed):  K ∈ {50, 100, 200, 500}
#   Run 2 — alpha sweep  (best K from Run 1):  alpha ∈ {0.05, 0.1, 0.2, 0.3}
#
# Compare against:
#   baseline  (ASRI_ALPHA=0, no EMA)    avg=13.80
#   DPEMA best (Exp5, β=0.999, α=0.3)  avg=19.74
#
# Usage:
#   bash scripts/exp7_swema.sh [CHECKPOINT_PATH]
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"

export DETECTRON2_DATASETS="$ROOT/datasets"
mkdir -p ../results/exp7

echo ""
echo "======================================================================"
echo "Exp 7: SWEMA — Sliding Window EMA  (Run 1: K sweep, alpha=0.1)"
echo "======================================================================"

ALPHA_FIXED="0.1"
KS="50 100 200 500"

for K in $KS; do
    K_TAG=$(printf "%04d" "$K")
    echo ""
    echo "--- K = $K ---"

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
        TEST.ADAPTATION.SWEMA_K "$K" \
        TEST.ADAPTATION.SWEMA_ALPHA "$ALPHA_FIXED" \
        TEST.ADAPTATION.EMA_BETA 0.0 \
        TEST.ADAPTATION.ASRI_ALPHA 0.0 \
        TEST.ADAPTATION.ORACLE_PROTOTYPE False \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        OUTPUT_DIR "../outputs/COCO_TTA/exp7_swema_K${K_TAG}_alpha_${ALPHA_FIXED//./}"

    cp "../outputs/COCO_TTA/exp7_swema_K${K_TAG}_alpha_${ALPHA_FIXED//./}/eval_matrix/metrics.json" \
       "../results/exp7/metrics_K${K_TAG}_alpha_${ALPHA_FIXED//./}.json" 2>/dev/null || true
done

# ── Find best K ───────────────────────────────────────────────────────────────
BEST_K=$(python3 - <<'PYEOF'
import json, glob, os

CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
    "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression"
]

def avg(path):
    d = json.load(open(path))
    return sum(d.get(f"coco_2017_val-{c}",{}).get("AP",0.) for c in CORRUPTIONS) / len(CORRUPTIONS)

files = sorted(glob.glob("../results/exp7/metrics_K*_alpha_01.json"))
best = max(files, key=avg)
tag = os.path.basename(best).replace("metrics_K","").split("_")[0]
print(int(tag))
PYEOF
)
echo ""
echo "Best K from Run 1: $BEST_K"

echo ""
echo "======================================================================"
echo "Exp 7: SWEMA — Run 2: alpha sweep  (K=$BEST_K)"
echo "======================================================================"

ALPHAS="0.05 0.1 0.2 0.3"
BEST_K_TAG=$(printf "%04d" "$BEST_K")

for ALPHA in $ALPHAS; do
    ALPHA_TAG="${ALPHA//./}"
    echo ""
    echo "--- K=$BEST_K  alpha=$ALPHA ---"

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
        TEST.ADAPTATION.SWEMA_K "$BEST_K" \
        TEST.ADAPTATION.SWEMA_ALPHA "$ALPHA" \
        TEST.ADAPTATION.EMA_BETA 0.0 \
        TEST.ADAPTATION.ASRI_ALPHA 0.0 \
        TEST.ADAPTATION.ORACLE_PROTOTYPE False \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        OUTPUT_DIR "../outputs/COCO_TTA/exp7_swema_K${BEST_K_TAG}_alpha_${ALPHA_TAG}"

    cp "../outputs/COCO_TTA/exp7_swema_K${BEST_K_TAG}_alpha_${ALPHA_TAG}/eval_matrix/metrics.json" \
       "../results/exp7/metrics_K${BEST_K_TAG}_alpha_${ALPHA_TAG}.json" 2>/dev/null || true
done

# ── Summary ───────────────────────────────────────────────────────────────────
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
BASELINE = {
    "gaussian_noise":15.33,"shot_noise":18.80,"impulse_noise":18.00,
    "defocus_blur":12.09,"glass_blur":4.59,"motion_blur":7.29,
    "zoom_blur":3.37,"snow":9.12,"frost":11.69,"fog":23.86,
    "brightness":27.43,"contrast":17.08,"elastic_transform":17.52,
    "pixelate":9.33,"jpeg_compression":11.49,
}
DPEMA_BEST = 19.74  # Exp5: β=0.999, α=0.3

def get_ap(d, c):
    return d.get(f"coco_2017_val-{c}", {}).get("AP", float("nan"))
def avg(d):
    return sum(get_ap(d,c) for c in CORRUPTIONS) / len(CORRUPTIONS)

base_avg = sum(BASELINE.values()) / len(CORRUPTIONS)

# Run 1: K sweep
print("\n=== Run 1: K sweep (swema_alpha=0.1) ===")
k_files = sorted(glob.glob("../results/exp7/metrics_K*_alpha_01.json"))
print(f"{'K':<8} {'avg AP':>8}  snow    frost   fog")
print(f"{'baseline':<8} {base_avg:>8.2f}  {BASELINE['snow']:>5.2f}  {BASELINE['frost']:>5.2f}  {BASELINE['fog']:>5.2f}")
print(f"{'dpema*':<8} {DPEMA_BEST:>8.2f}  (Exp5 best)")
for f in k_files:
    d = json.load(open(f))
    k = int(os.path.basename(f).replace("metrics_K","").split("_")[0])
    print(f"K={k:<6} {avg(d):>8.2f}  {get_ap(d,'snow'):>5.2f}  {get_ap(d,'frost'):>5.2f}  {get_ap(d,'fog'):>5.2f}")

# Run 2: alpha sweep
best_k_files = sorted(glob.glob("../results/exp7/metrics_K????_alpha_[^0][^1]*.json") +
                       glob.glob("../results/exp7/metrics_K????_alpha_005.json") +
                       glob.glob("../results/exp7/metrics_K????_alpha_01.json"))
# deduplicate and filter to the single best K
if k_files:
    best_k_tag = os.path.basename(max(k_files, key=lambda f: avg(json.load(open(f))))).split("_")[1]
    alpha_files = sorted(glob.glob(f"../results/exp7/metrics_{best_k_tag}_alpha_*.json"))
    if alpha_files:
        print(f"\n=== Run 2: alpha sweep (K={int(best_k_tag.replace('K',''))}) ===")
        print(f"{'alpha':<10} {'avg AP':>8}  snow    frost   fog")
        for f in alpha_files:
            d = json.load(open(f))
            alpha_tag = os.path.basename(f).split("alpha_")[1].replace(".json","")
            alpha_val = float("0." + alpha_tag.lstrip("0") if alpha_tag != "0" else "0")
            print(f"α={alpha_val:<8} {avg(d):>8.2f}  {get_ap(d,'snow'):>5.2f}  {get_ap(d,'frost'):>5.2f}  {get_ap(d,'fog'):>5.2f}")

# Overall best
all_files = glob.glob("../results/exp7/metrics_*.json")
if all_files:
    best_f = max(all_files, key=lambda f: avg(json.load(open(f))))
    best_d = json.load(open(best_f))
    print(f"\nBest SWEMA: {os.path.basename(best_f)}")
    print(f"  avg AP = {avg(best_d):.2f}  (baseline={base_avg:.2f}, DPEMA={DPEMA_BEST:.2f})")
    print(f"  Δ vs baseline : {avg(best_d)-base_avg:+.2f}")
    print(f"  Δ vs DPEMA    : {avg(best_d)-DPEMA_BEST:+.2f}")
PYEOF

echo ""
echo "=== Exp 7 complete. Results in results/exp7/ ==="
