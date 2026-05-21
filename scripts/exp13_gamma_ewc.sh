#!/usr/bin/env bash
# =============================================================================
# Exp 13: Legacy Gamma + EWC on Adapter
#
# Motivation:
#   exp12 e0 (legacy gamma, no EWC)  avg15=21.23  beats
#   exp11 e0b (DPEMA,        EWC=10) avg15=20.67
#
#   → exp10/11 EWC experiments were all built on DPEMA (EMA_BETA=0.999),
#     which is itself a suboptimal base. This exp asks: does EWC still help
#     when the prototype base is legacy gamma (the stronger base)?
#
# Design:
#   E0  legacy gamma, no EWC          — copied from exp12
#   E1  DPEMA,        EWC λ=10        — copied from exp10 current method
#   E2  legacy gamma, EWC λ=10        — NEW (key experiment)
#
# Base config: COCO R50, fg+global KL, adapter adaptation.
#   legacy gamma path = EMA_BETA=0, SWEMA_K=0 (same as exp12 e0).
#
# Outputs:
#   results/exp13/metrics_<tag>.json
#   results/exp13/eval_matrix_<tag>.npy
#   results/exp13/exp13_summary.json
#
# Usage:
#   bash scripts/exp13_gamma_ewc.sh [CHECKPOINT_PATH]
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
mkdir -p ../results/exp13

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
    TEST.ADAPTATION.EMA_BETA 0.0
    TEST.ADAPTATION.SWEMA_K 0
    TEST.ADAPTATION.ASRI_ALPHA 0.0
    TEST.ADAPTATION.ADAPTER_RESET False
    TEST.ADAPTATION.ORACLE_PROTOTYPE False
)

collect() {
    local TAG="$1"
    local OUT="$2"
    cp "${OUT}/eval_matrix/metrics.json"    "../results/exp13/metrics_${TAG}.json"    2>/dev/null || true
    cp "${OUT}/eval_matrix/eval_matrix.npy" "../results/exp13/eval_matrix_${TAG}.npy" 2>/dev/null || true
}

# ─────────────────────────────────────────────────────────────────────────────
# Pre-step : compute Fisher (only if missing) — required for EWC runs
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
# E0 / E1 : carry over baselines (self-contained summary)
# ─────────────────────────────────────────────────────────────────────────────
cp -f ../results/exp12/metrics_e0_baseline.json   ../results/exp13/metrics_e0_gamma_noewc.json  2>/dev/null || true
cp -f ../results/exp10/metrics_r1_ewc10_0.json    ../results/exp13/metrics_e1_dpema_ewc10.json  2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# E2 : legacy gamma + EWC λ=10  ← key experiment
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== E2 : legacy gamma + EWC λ=10 ==="
OUT="../outputs/COCO_TTA/exp13_e2_gamma_ewc10"
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.EWC_LAMBDA 10.0 \
    TEST.ADAPTATION.EWC_FISHER_PATH "$FISHER_PATH" \
    OUTPUT_DIR "$OUT"
collect "e2_gamma_ewc10" "$OUT"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Exp 13 Summary ==="
python3 - <<'PYEOF'
import json, glob, os, math

CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
    "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression"
]

def get_ap(d, c):
    return d.get(f"coco_2017_val-{c}", {}).get("AP", float("nan"))

def avg_ap15(d):
    vals = [get_ap(d, c) for c in CORRUPTIONS]
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(CORRUPTIONS) if vals else float("nan")

def mean_ap16(d):
    vals = [get_ap(d, c) for c in CORRUPTIONS]
    clean = d.get("coco_2017_val", {}).get("AP", float("nan"))
    vals = [v for v in vals if not math.isnan(v)]
    if math.isnan(clean):
        return float("nan")
    return (sum(vals) + clean) / (len(CORRUPTIONS) + 1) if vals else float("nan")

rows = []
keep_tags = {"e0_gamma_noewc", "e1_dpema_ewc10", "e2_gamma_ewc10"}
for f in sorted(glob.glob("../results/exp13/metrics_*.json")):
    tag = os.path.basename(f).replace("metrics_", "").replace(".json", "")
    if tag not in keep_tags:
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rows.append((tag, d, avg_ap15(d), mean_ap16(d)))

print(f"\n{'Run':<24} {'proto':>8} {'EWC λ':>6} {'avg15':>7} {'mean16':>7}  clean")
print("-" * 64)
labels = {
    "e0_gamma_noewc":  ("gamma",  "  —"),
    "e1_dpema_ewc10":  ("DPEMA",  " 10"),
    "e2_gamma_ewc10":  ("gamma",  " 10"),
}
for tag, d, ap15, ap16 in rows:
    proto, ewc = labels.get(tag, ("?", "?"))
    clean = d.get("coco_2017_val", {}).get("AP", float("nan"))
    print(f"{tag:<24} {proto:>8} {ewc:>6} {ap15:>7.2f} {ap16:>7.2f}  {clean:>6.2f}")

print("\n--- Per-corruption breakdown ---")
print(f"\n{'corruption':<20}", end="")
for tag, *_ in rows:
    print(f"  {tag[:12]:>12}", end="")
print()
print("-" * (20 + 14 * len(rows)))
for c in CORRUPTIONS:
    print(f"{c:<20}", end="")
    for _, d, *__ in rows:
        print(f"  {get_ap(d, c):>12.2f}", end="")
    print()
clean_label = "(clean)"
print(f"{clean_label:<20}", end="")
for _, d, *__ in rows:
    print(f"  {d.get('coco_2017_val', {}).get('AP', float('nan')):>12.2f}", end="")
print()

with open("../results/exp13/exp13_summary.json", "w") as fp:
    json.dump(
        [{"tag": t, "avg_AP15": ap15, "mean_AP16": ap16,
          "clean": d.get("coco_2017_val", {}).get("AP", float("nan")),
          **{c: get_ap(d, c) for c in CORRUPTIONS}}
         for t, d, ap15, ap16 in rows],
        fp, indent=2
    )
print("\nSaved to ../results/exp13/exp13_summary.json")
PYEOF

echo ""
echo "=== Exp 13 complete. Results in results/exp13/ ==="
