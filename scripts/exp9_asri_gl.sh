#!/usr/bin/env bash
# =============================================================================
# Exp 9: ASRI for Global Branch
#
# Base: DPEMA (β=0.999, Exp 5 best)
# Change: ASRI_GL=True — source residual injection applied to global FPN
#         prototypes as well as foreground, preventing global prototype drift.
#         The same asri_alpha is used for both branches.
#
# Run 1 — α sweep with ASRI_GL=True: α ∈ {0.1, 0.2, 0.3, 0.5}
#          Compare directly against Exp 5 Run 2 (fg-only ASRI, same α values).
#
# Baseline reference (Exp 5): DPEMA β=0.999, fg-only ASRI α=0.3  →  avg AP 19.74
#
# Usage:
#   bash scripts/exp9_asri_gl.sh [CHECKPOINT_PATH]
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
mkdir -p ../results/exp9

echo ""
echo "======================================================================"
echo "Exp 9: ASRI_GL=True — α sweep  (β=${BETA})"
echo "======================================================================"

ALPHAS="0.1 0.2 0.3 0.5"

for ALPHA in $ALPHAS; do
    ALPHA_TAG=$(echo "$ALPHA" | sed 's/\./_/g')

    echo ""
    echo "--- ASRI_GL=True  α=${ALPHA} ---"

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
        TEST.ADAPTATION.ASRI_GL True \
        TEST.ADAPTATION.ORACLE_PROTOTYPE False \
        TEST.ADAPTATION.EMA_BETA "$BETA" \
        TEST.ADAPTATION.DPEMA_APPLY_GL True \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        OUTPUT_DIR "../outputs/COCO_TTA/exp9_asri_gl_beta_${BETA_TAG}_alpha_${ALPHA_TAG}"

    cp "../outputs/COCO_TTA/exp9_asri_gl_beta_${BETA_TAG}_alpha_${ALPHA_TAG}/eval_matrix/metrics.json" \
       "../results/exp9/metrics_asri_gl_alpha_${ALPHA_TAG}.json" 2>/dev/null || true
done

# =============================================================================
# Summary — direct comparison with Exp 5 (fg-only ASRI)
# =============================================================================
echo ""
echo "======================================================================"
echo "Results summary — Exp9 (fg+gl ASRI) vs Exp5 (fg-only ASRI)"
echo "======================================================================"

python3 - <<'PYEOF'
import json, os, glob

CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
    "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression"
]

def get_ap(d, c):
    return d.get(f"coco_2017_val-{c}", {}).get("AP", float("nan"))

def avg(d):
    vals = [get_ap(d, c) for c in CORRUPTIONS]
    return sum(v for v in vals if v == v) / len(CORRUPTIONS)

# Load Exp 5 fg-only results for comparison
exp5_files = {
    os.path.basename(f).replace("metrics_beta_0_999_alpha_","").replace(".json",""): f
    for f in sorted(glob.glob("../results/exp5/metrics_beta_0_999_alpha_*.json"))
}
exp9_files = sorted(glob.glob("../results/exp9/metrics_asri_gl_alpha_*.json"))

print(f"\n{'α':<6}  {'exp5 fg-only':>14}  {'exp9 fg+gl':>12}  {'Δ':>6}")
print("-" * 46)

for f9 in exp9_files:
    alpha_tag = os.path.basename(f9).replace("metrics_asri_gl_alpha_","").replace(".json","")
    d9 = json.load(open(f9))
    avg9 = avg(d9)

    avg5 = float("nan")
    if alpha_tag in exp5_files:
        d5 = json.load(open(exp5_files[alpha_tag]))
        avg5 = avg(d5)

    alpha_str = alpha_tag.replace("_", ".")
    delta = avg9 - avg5 if avg5 == avg5 else float("nan")
    print(f"{alpha_str:<6}  {avg5:>14.2f}  {avg9:>12.2f}  {delta:>+6.2f}")

# Per-corruption breakdown for best α
if exp9_files:
    best_f9 = max(exp9_files, key=lambda f: avg(json.load(open(f))))
    best_alpha_tag = os.path.basename(best_f9).replace("metrics_asri_gl_alpha_","").replace(".json","")
    d_best9 = json.load(open(best_f9))

    d_best5 = None
    if best_alpha_tag in exp5_files:
        d_best5 = json.load(open(exp5_files[best_alpha_tag]))

    print(f"\n--- Per-corruption: best exp9 (α={best_alpha_tag.replace('_','.')}) vs exp5 ---")
    print(f"{'corruption':<24} {'exp5':>8}  {'exp9':>8}  {'Δ':>6}")
    print("-" * 52)
    for c in CORRUPTIONS:
        v9 = get_ap(d_best9, c)
        v5 = get_ap(d_best5, c) if d_best5 else float("nan")
        delta = v9 - v5 if v5 == v5 else float("nan")
        print(f"{c:<24} {v5:>8.2f}  {v9:>8.2f}  {delta:>+6.2f}")
    print("-" * 52)
    print(f"{'avg (15)':<24} {avg(d_best5) if d_best5 else float('nan'):>8.2f}  {avg(d_best9):>8.2f}"
          f"  {avg(d_best9)-(avg(d_best5) if d_best5 else float('nan')):>+6.2f}")
PYEOF

echo ""
echo "=== Exp 9 complete. Results in results/exp9/ ==="
