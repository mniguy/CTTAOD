#!/usr/bin/env bash
# =============================================================================
# Exp 11: EWC-on-Adapter Extensions
#
# Builds on Exp 10 R1 (EWC λ=10.0 was best). Probes four directions:
#   E1  Layer-normalized Fisher          → λ sweep at normalized scale
#   E2  Drift-adaptive λ                 → tie λ to global_align / loss_ema99
#   E3  Sliding anchor (EMA-updated)     → anchor β sweep
#   E4  MAS importance (vs Fisher)       → λ sweep
#   E5  Best combination of E1..E4
#
# Baselines:
#   E0a  R0 baseline (no EWC)             — copied from Exp 10
#   E0b  R1 best (Fisher, λ=10, fixed)    — copied from Exp 10
#
# Reference base (same as Exp 10): COCO R50, DPEMA β=0.999, fg+global KL.
#
# Outputs:
#   results/exp11/metrics_<tag>.json
#   results/exp11/eval_matrix_<tag>.npy
#   results/exp11/exp11_summary.json
#
# Usage:
#   bash scripts/exp11_ewc_extensions.sh [CHECKPOINT_PATH]
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"
FISHER_PATH="../models/stats/COCO_R50_fisher.pt"
MAS_PATH="../models/stats/COCO_R50_mas.pt"

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/datasets}"
mkdir -p ../results/exp11

# Common args ----------------------------------------------------------------
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

# helper: copy metrics + eval matrix into results/exp11 ----------------------
collect() {
    local TAG="$1"
    local OUT="$2"
    cp "${OUT}/eval_matrix/metrics.json"    "../results/exp11/metrics_${TAG}.json"    2>/dev/null || true
    cp "${OUT}/eval_matrix/eval_matrix.npy" "../results/exp11/eval_matrix_${TAG}.npy" 2>/dev/null || true
}

# avg AP across 15 corruptions
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
# Pre-step 0a : compute Fisher (only if missing)
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "$FISHER_PATH" ]; then
    echo "=== Pre-step: computing Fisher ==="
    python compute_fisher.py \
        --config-file "$CFG" \
        MODEL.WEIGHTS "$CKPT" \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        TEST.ADAPTATION.WHERE "adapter"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Pre-step 0b : compute MAS (only if missing) — required for E4
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "$MAS_PATH" ]; then
    echo "=== Pre-step: computing MAS importance ==="
    python compute_mas.py \
        --config-file "$CFG" \
        MODEL.WEIGHTS "$CKPT" \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        TEST.ADAPTATION.WHERE "adapter"
fi

# ─────────────────────────────────────────────────────────────────────────────
# E0 : carry over baselines from Exp 10 (so summary table is self-contained)
# ─────────────────────────────────────────────────────────────────────────────
cp -f ../results/exp10/metrics_r0_baseline.json  ../results/exp11/metrics_e0a_baseline.json  2>/dev/null || true
cp -f ../results/exp10/metrics_r1_ewc10_0.json   ../results/exp11/metrics_e0b_ewc10_fixed.json 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# E1 : Layer-normalized Fisher — re-sweep λ on the new scale
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== E1 : Layer-normalized Fisher ==="
for EWC_L in 0.1 1.0 10.0 100.0; do
    EWC_TAG=$(echo "$EWC_L" | sed 's/\./_/g')
    OUT="../outputs/COCO_TTA/exp11_e1_fnorm_l${EWC_TAG}"
    echo "  λ = $EWC_L"
    python train_net.py "${BASE_ARGS[@]}" \
        TEST.ADAPTATION.EWC_LAMBDA "$EWC_L" \
        TEST.ADAPTATION.EWC_FISHER_PATH "$FISHER_PATH" \
        TEST.ADAPTATION.EWC_FISHER_NORM True \
        OUTPUT_DIR "$OUT"
    collect "e1_fnorm_l${EWC_TAG}" "$OUT"
done

# ─────────────────────────────────────────────────────────────────────────────
# E2 : Drift-adaptive λ — fix base λ=10, sweep adaptive BETA
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== E2 : Drift-adaptive λ (base λ=10) ==="
for BETA_A in 0.5 1.0 2.0 4.0; do
    BETA_TAG=$(echo "$BETA_A" | sed 's/\./_/g')
    OUT="../outputs/COCO_TTA/exp11_e2_adapt_b${BETA_TAG}"
    echo "  BETA = $BETA_A"
    python train_net.py "${BASE_ARGS[@]}" \
        TEST.ADAPTATION.EWC_LAMBDA 10.0 \
        TEST.ADAPTATION.EWC_FISHER_PATH "$FISHER_PATH" \
        TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE True \
        TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE_BETA "$BETA_A" \
        OUTPUT_DIR "$OUT"
    collect "e2_adapt_b${BETA_TAG}" "$OUT"
done

# ─────────────────────────────────────────────────────────────────────────────
# E3 : Sliding anchor — fix λ=10, sweep anchor β
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== E3 : Sliding anchor (λ=10) ==="
for SA_BETA in 0.99 0.999 0.9999; do
    SA_TAG=$(echo "$SA_BETA" | sed 's/\./_/g')
    OUT="../outputs/COCO_TTA/exp11_e3_slide_b${SA_TAG}"
    echo "  anchor β = $SA_BETA"
    python train_net.py "${BASE_ARGS[@]}" \
        TEST.ADAPTATION.EWC_LAMBDA 10.0 \
        TEST.ADAPTATION.EWC_FISHER_PATH "$FISHER_PATH" \
        TEST.ADAPTATION.EWC_SLIDING_ANCHOR True \
        TEST.ADAPTATION.EWC_SLIDING_ANCHOR_BETA "$SA_BETA" \
        OUTPUT_DIR "$OUT"
    collect "e3_slide_b${SA_TAG}" "$OUT"
done

# ─────────────────────────────────────────────────────────────────────────────
# E4 : MAS importance — λ sweep (MAS scale ≠ Fisher scale)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== E4 : MAS importance ==="
for EWC_L in 0.1 1.0 10.0 100.0; do
    EWC_TAG=$(echo "$EWC_L" | sed 's/\./_/g')
    OUT="../outputs/COCO_TTA/exp11_e4_mas_l${EWC_TAG}"
    echo "  λ = $EWC_L"
    python train_net.py "${BASE_ARGS[@]}" \
        TEST.ADAPTATION.EWC_LAMBDA "$EWC_L" \
        TEST.ADAPTATION.EWC_FISHER_PATH "$MAS_PATH" \
        TEST.ADAPTATION.EWC_IMPORTANCE_TYPE "mas" \
        OUTPUT_DIR "$OUT"
    collect "e4_mas_l${EWC_TAG}" "$OUT"
done

# ─────────────────────────────────────────────────────────────────────────────
# Pick best-per-family by avg AP across 15 corruptions
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Picking best per-family variants for E5 ==="
read BEST_E1 BEST_E2 BEST_E3 BEST_E4 < <(python3 - <<'PYEOF'
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
    for f in glob.glob(f"../results/exp11/metrics_{prefix}*.json"):
        m = avg_ap(f)
        if not math.isnan(m) and m > bm:
            bm, bf = m, f
    return bf

def tag(path):
    return re.search(r"metrics_([^/]+?)\.json$", path).group(1) if path else "NONE"

print(tag(best("e1_fnorm")), tag(best("e2_adapt")), tag(best("e3_slide")), tag(best("e4_mas")))
PYEOF
)
echo "  best E1 (fisher-norm) = $BEST_E1"
echo "  best E2 (adaptive λ)  = $BEST_E2"
echo "  best E3 (sliding)     = $BEST_E3"
echo "  best E4 (MAS)         = $BEST_E4"

# ─────────────────────────────────────────────────────────────────────────────
# Map tag → CLI overrides
# ─────────────────────────────────────────────────────────────────────────────
e1_args_from_tag() {
    case "$1" in
        e1_fnorm_l0_1)   echo "TEST.ADAPTATION.EWC_LAMBDA 0.1   TEST.ADAPTATION.EWC_FISHER_NORM True" ;;
        e1_fnorm_l1_0)   echo "TEST.ADAPTATION.EWC_LAMBDA 1.0   TEST.ADAPTATION.EWC_FISHER_NORM True" ;;
        e1_fnorm_l10_0)  echo "TEST.ADAPTATION.EWC_LAMBDA 10.0  TEST.ADAPTATION.EWC_FISHER_NORM True" ;;
        e1_fnorm_l100_0) echo "TEST.ADAPTATION.EWC_LAMBDA 100.0 TEST.ADAPTATION.EWC_FISHER_NORM True" ;;
        *) echo "" ;;
    esac
}
e2_args_from_tag() {
    case "$1" in
        e2_adapt_b0_5) echo "TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE True TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE_BETA 0.5" ;;
        e2_adapt_b1_0) echo "TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE True TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE_BETA 1.0" ;;
        e2_adapt_b2_0) echo "TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE True TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE_BETA 2.0" ;;
        e2_adapt_b4_0) echo "TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE True TEST.ADAPTATION.EWC_LAMBDA_ADAPTIVE_BETA 4.0" ;;
        *) echo "" ;;
    esac
}
e3_args_from_tag() {
    case "$1" in
        e3_slide_b0_99)   echo "TEST.ADAPTATION.EWC_SLIDING_ANCHOR True TEST.ADAPTATION.EWC_SLIDING_ANCHOR_BETA 0.99" ;;
        e3_slide_b0_999)  echo "TEST.ADAPTATION.EWC_SLIDING_ANCHOR True TEST.ADAPTATION.EWC_SLIDING_ANCHOR_BETA 0.999" ;;
        e3_slide_b0_9999) echo "TEST.ADAPTATION.EWC_SLIDING_ANCHOR True TEST.ADAPTATION.EWC_SLIDING_ANCHOR_BETA 0.9999" ;;
        *) echo "" ;;
    esac
}
# E4 sets a different importance file/type and a λ — capture λ separately so
# the combined run can still use Fisher (importance file/type only swap to MAS
# if MAS wins outright in E4).
e4_lambda_from_tag() {
    case "$1" in
        e4_mas_l0_1)   echo "0.1" ;;
        e4_mas_l1_0)   echo "1.0" ;;
        e4_mas_l10_0)  echo "10.0" ;;
        e4_mas_l100_0) echo "100.0" ;;
        *) echo "" ;;
    esac
}

E1_OV=$(e1_args_from_tag "$BEST_E1")
E2_OV=$(e2_args_from_tag "$BEST_E2")
E3_OV=$(e3_args_from_tag "$BEST_E3")
E4_LAMBDA=$(e4_lambda_from_tag "$BEST_E4")

# ─────────────────────────────────────────────────────────────────────────────
# E5 : Combined — best E1 + best E2 + best E3.
#      (MAS family is left as a standalone comparison; combining a different
#      importance signal with the other three would muddle attribution.)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== E5 : Combined (best E1 + best E2 + best E3) ==="
OUT="../outputs/COCO_TTA/exp11_e5_combined"
python train_net.py "${BASE_ARGS[@]}" \
    TEST.ADAPTATION.EWC_LAMBDA 10.0 \
    TEST.ADAPTATION.EWC_FISHER_PATH "$FISHER_PATH" \
    $E1_OV $E2_OV $E3_OV \
    OUTPUT_DIR "$OUT"
collect "e5_combined" "$OUT"

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Exp 11 Summary ==="
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
for f in sorted(glob.glob("../results/exp11/metrics_*.json")):
    tag = os.path.basename(f).replace("metrics_", "").replace(".json", "")
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rows.append((tag, d, avg_ap(d)))

print(f"\n{'Run':<28} {'avg AP':>8}  gauss   defoc   glass   fog     snow")
print("-" * 78)
for tag, d, ap in rows:
    print(f"{tag:<28} {ap:>8.2f}"
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

with open("../results/exp11/exp11_summary.json", "w") as fp:
    json.dump(
        [{"tag": t, "avg_AP": ap,
          **{c: get_ap(d, c) for c in CORRUPTIONS}}
         for t, d, ap in rows],
        fp, indent=2
    )
print("\nSaved to ../results/exp11/exp11_summary.json")
PYEOF

echo ""
echo "=== Exp 11 complete. Results in results/exp11/ ==="
