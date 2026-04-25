#!/usr/bin/env bash
# =============================================================================
# Exp 2: Fixed α Sweep (Source Injection Strength)
#
# Sweeps ASRI_ALPHA ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
# For each α, builds the full (T+1)×T eval matrix and saves BWT/FWT/avg_mAP.
#
# Note: α=0.0 is identical to baseline CTTAOD (already run in Exp 0).
#       α=1.0 is identical to Exp 1 Variant B (already run in Exp 1).
#
# Usage:
#   bash scripts/exp2_alpha_sweep.sh [CHECKPOINT_PATH]
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"

export DETECTRON2_DATASETS="$ROOT/datasets"
mkdir -p ../results/exp2

ALPHAS="0.0 0.05 0.1 0.2 0.3 0.5 0.7 1.0"

for ALPHA in $ALPHAS; do
    # Convert decimal to underscore for directory naming (e.g., 0.05 -> 0_05)
    ALPHA_TAG=$(echo "$ALPHA" | sed 's/\./_/g')

    echo ""
    echo "=== Exp 2: ASRI_ALPHA = $ALPHA ==="

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
        TEST.ADAPTATION.ORACLE_PROTOTYPE False \
        TEST.ADAPTATION.ADAPTER_RESET False \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        OUTPUT_DIR "../outputs/COCO_TTA/exp2_alpha_${ALPHA_TAG}"

    cp "../outputs/COCO_TTA/exp2_alpha_${ALPHA_TAG}/eval_matrix/eval_matrix.npy" \
       "../results/exp2/eval_matrix_alpha_${ALPHA_TAG}.npy" 2>/dev/null || true
    cp "../outputs/COCO_TTA/exp2_alpha_${ALPHA_TAG}/eval_matrix/metrics.json" \
       "../results/exp2/metrics_alpha_${ALPHA_TAG}.json" 2>/dev/null || true
done

# ─────────────────────────────────────────────────────────────────────────────
# Aggregate results into alpha_sweep_results.json
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Aggregating sweep results ==="
python - <<'PYEOF'
import json, os, glob

results = {}
result_dir = "../results/exp2"
for f in sorted(glob.glob(os.path.join(result_dir, "metrics_alpha_*.json"))):
    tag = os.path.basename(f).replace("metrics_alpha_", "").replace(".json", "")
    alpha = float(tag.replace("_", "."))
    with open(f) as fp:
        m = json.load(fp)
    results[str(alpha)] = {k: m[k] for k in ["BWT", "FWT", "avg_mAP"]}
    print(f"α={alpha:.2f}  BWT={m['BWT']:.4f}  FWT={m['FWT']:.4f}  avg_mAP={m['avg_mAP']:.2f}")

out = os.path.join(result_dir, "alpha_sweep_results.json")
with open(out, "w") as fp:
    json.dump(results, fp, indent=2)
print(f"\nSaved to {out}")

# Find best α by avg_mAP
if results:
    best_alpha = max(results, key=lambda a: results[a]["avg_mAP"])
    print(f"Best α* = {best_alpha}  (avg_mAP = {results[best_alpha]['avg_mAP']:.2f})")
PYEOF

echo ""
echo "=== Exp 2 complete. Results in results/exp2/ ==="