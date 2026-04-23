#!/usr/bin/env bash
# =============================================================================
# Exp 4: Comparison with Existing Anti-Forgetting Baselines
#
# Runs five methods and compares BWT / FWT / avg_mAP:
#   B1 — Stochastic Restoration (CoTTA-style), best p from sweep
#   B2 — EWC-style Adapter Regularization (requires pre-computed Fisher)
#   B3 — Prototype Replay Buffer (last 3 domain prototypes)
#   B4 — ASRI Fixed α* (use best α from Exp 2)
#   B5 — ASRI + Stochastic Restoration
#
# Usage:
#   bash scripts/exp4_comparison.sh [CHECKPOINT_PATH] [BEST_ALPHA]
#   BEST_ALPHA defaults to 0.2 (override with Exp 2 result)
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
BEST_ALPHA="${2:-0.2}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"

mkdir -p ../results/exp4

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
    TEST.ADAPTATION.ADAPTER_RESET False
    TEST.ADAPTATION.ORACLE_PROTOTYPE False
)

# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Stochastic Restoration (CoTTA-style)
# Sweep p_restore ∈ {0.001, 0.01, 0.05, 0.1}; report best
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Exp 4 Baseline 1: Stochastic Restoration ==="
BEST_B1_MAP=-9999
for P_RESTORE in 0.001 0.01 0.05 0.1; do
    P_TAG=$(echo "$P_RESTORE" | sed 's/\./_/g')
    echo "  p_restore = $P_RESTORE"
    python train_net.py "${BASE_ARGS[@]}" \
        TEST.ADAPTATION.ASRI_ALPHA 0.0 \
        TEST.ADAPTATION.STOCHASTIC_RESTORE True \
        TEST.ADAPTATION.STOCHASTIC_RESTORE_PROB "$P_RESTORE" \
        OUTPUT_DIR "../outputs/COCO_TTA/exp4_b1_p${P_TAG}"
    cp "../outputs/COCO_TTA/exp4_b1_p${P_TAG}/eval_matrix/metrics.json" \
       "../results/exp4/metrics_b1_p${P_TAG}.json" 2>/dev/null || true
done
# pick best by avg_mAP
python - <<'PYEOF'
import json, glob
best, best_f = -9999, ""
for f in glob.glob("../results/exp4/metrics_b1_p*.json"):
    with open(f) as fp: m = json.load(fp)
    if m["avg_mAP"] > best:
        best, best_f = m["avg_mAP"], f
import shutil; shutil.copy(best_f, "../results/exp4/metrics_b1_best.json")
print(f"B1 best: {best_f}  avg_mAP={best:.2f}")
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: EWC-style Adapter Regularization
# Requires Fisher file at models/stats/Cityscapes_R50_fisher.pt
# Sweep λ_ewc ∈ {0.1, 1.0, 10.0, 100.0}
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Exp 4 Baseline 2: EWC Regularization ==="
FISHER_PATH="../models/stats/COCO_R50_fisher.pt"
if [ -f "$FISHER_PATH" ]; then
    for EWC_L in 0.1 1.0 10.0 100.0; do
        EWC_TAG=$(echo "$EWC_L" | sed 's/\./_/g')
        echo "  lambda_ewc = $EWC_L"
        python train_net.py "${BASE_ARGS[@]}" \
            TEST.ADAPTATION.ASRI_ALPHA 0.0 \
            TEST.ADAPTATION.EWC_LAMBDA "$EWC_L" \
            TEST.ADAPTATION.EWC_FISHER_PATH "$FISHER_PATH" \
            OUTPUT_DIR "../outputs/COCO_TTA/exp4_b2_ewc${EWC_TAG}"
        cp "../outputs/COCO_TTA/exp4_b2_ewc${EWC_TAG}/eval_matrix/metrics.json" \
           "../results/exp4/metrics_b2_ewc${EWC_TAG}.json" 2>/dev/null || true
    done
else
    echo "  Fisher file not found at $FISHER_PATH — skipping EWC baseline."
    echo "  Run tools/compute_fisher.py first to generate it."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: Prototype Replay Buffer (last 3 domains)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Exp 4 Baseline 3: Prototype Replay Buffer ==="
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.ASRI_ALPHA 0.0 \
    TEST.ADAPTATION.PROTOTYPE_REPLAY True \
    TEST.ADAPTATION.PROTOTYPE_REPLAY_BUFFER_SIZE 3 \
    OUTPUT_DIR ../outputs/COCO_TTA/exp4_b3_replay
cp ../outputs/COCO_TTA/exp4_b3_replay/eval_matrix/metrics.json \
   ../results/exp4/metrics_b3_replay.json 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Baseline 4: ASRI Fixed α*
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Exp 4 Baseline 4: ASRI Fixed α* = $BEST_ALPHA ==="
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.ASRI_ALPHA "$BEST_ALPHA" \
    OUTPUT_DIR ../outputs/COCO_TTA/exp4_b4_asri
cp ../outputs/COCO_TTA/exp4_b4_asri/eval_matrix/metrics.json \
   ../results/exp4/metrics_b4_asri.json 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Baseline 5: ASRI + Stochastic Restoration (Combined)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Exp 4 Baseline 5: ASRI + Stochastic Restore ==="
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.ASRI_ALPHA "$BEST_ALPHA" \
    TEST.ADAPTATION.STOCHASTIC_RESTORE True \
    TEST.ADAPTATION.STOCHASTIC_RESTORE_PROB 0.01 \
    OUTPUT_DIR ../outputs/COCO_TTA/exp4_b5_asri_restore
cp ../outputs/COCO_TTA/exp4_b5_asri_restore/eval_matrix/metrics.json \
   ../results/exp4/metrics_b5_asri_restore.json 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Print comparison table
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Exp 4 Comparison Table ==="
python - <<'PYEOF'
import json, os

method_files = [
    ("Baseline CTTAOD",        "../results/exp0/metrics_baseline.json"),
    ("B1: Stoch. Restore",     "../results/exp4/metrics_b1_best.json"),
    ("B3: Proto Replay",       "../results/exp4/metrics_b3_replay.json"),
    ("B4: ASRI α*",            "../results/exp4/metrics_b4_asri.json"),
    ("B5: ASRI + Restore",     "../results/exp4/metrics_b5_asri_restore.json"),
]

print(f"{'Method':<25} {'BWT':>8} {'FWT':>8} {'avg_mAP':>9}")
print("-" * 55)
all_results = {}
for name, path in method_files:
    if os.path.exists(path):
        with open(path) as f: m = json.load(f)
        print(f"{name:<25} {m['BWT']:>8.4f} {m['FWT']:>8.4f} {m['avg_mAP']:>9.2f}")
        all_results[name] = m
    else:
        print(f"{name:<25}  (not found)")

out = "../results/exp4/exp4_comparison.json"
with open(out, "w") as f: json.dump(all_results, f, indent=2)
print(f"\nSaved to {out}")
PYEOF

echo ""
echo "=== Exp 4 complete. Results in results/exp4/ ==="