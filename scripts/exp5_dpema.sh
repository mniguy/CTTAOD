#!/usr/bin/env bash
# =============================================================================
# Exp 5: DPEMA — Decaying Prototype EMA
#
# Run 1: DPEMA β sweep (ASRI disabled, α=0.0)
#   β ∈ {0.99, 0.995, 0.999, 0.9995, 0.9999}
#
# Run 2: DPEMA + ASRI joint sweep (best β from Run 1, α ∈ {0.1, 0.2, 0.3, 0.5})
#
# Run 3: Ablation comparison table
#   source_only | baseline(α=0) | oracle(varB) | dpema_best | dpema_asri_best
#
# Usage:
#   bash scripts/exp5_dpema.sh [CHECKPOINT_PATH]
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"

export DETECTRON2_DATASETS="$ROOT/datasets"
mkdir -p ../results/exp5

# =============================================================================
# Run 1: DPEMA β sweep (α=0.0, ASRI disabled)
# =============================================================================
echo ""
echo "======================================================================"
echo "Run 1: DPEMA β sweep  (ASRI_ALPHA=0.0)"
echo "======================================================================"

BETAS="0.99 0.995 0.999 0.9995 0.9999"

for BETA in $BETAS; do
    BETA_TAG=$(echo "$BETA" | sed 's/\./_/g')

    echo ""
    echo "--- EMA_BETA = $BETA ---"

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
        TEST.ADAPTATION.ASRI_ALPHA 0.0 \
        TEST.ADAPTATION.ORACLE_PROTOTYPE False \
        TEST.ADAPTATION.EMA_BETA "$BETA" \
        TEST.ADAPTATION.DPEMA_APPLY_GL True \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        OUTPUT_DIR "../outputs/COCO_TTA/exp5_dpema_beta_${BETA_TAG}"

    cp "../outputs/COCO_TTA/exp5_dpema_beta_${BETA_TAG}/eval_matrix/metrics.json" \
       "../results/exp5/metrics_beta_${BETA_TAG}.json" 2>/dev/null || true
done

# =============================================================================
# Run 2: DPEMA + ASRI joint sweep  (best β from Run 1)
# =============================================================================
echo ""
echo "======================================================================"
echo "Run 2: DPEMA + ASRI joint sweep"
echo "======================================================================"

# Identify best β by average AP over 15 corruptions
BEST_BETA=$(python3 - <<'PYEOF'
import json, os, glob

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"
]

best_beta, best_avg = None, -1.0
for f in sorted(glob.glob("../results/exp5/metrics_beta_*.json")):
    with open(f) as fp:
        m = json.load(fp)
    aps = [m.get(f"coco_2017_val-{c}", {}).get("AP", 0.0) for c in CORRUPTIONS]
    avg = sum(aps) / len(aps)
    tag = os.path.basename(f).replace("metrics_beta_", "").replace(".json", "")
    beta = float(tag.replace("_", "."))
    if avg > best_avg:
        best_avg, best_beta = avg, beta

print(best_beta)
PYEOF
)

echo "Best β* from Run 1: $BEST_BETA"

ALPHAS="0.1 0.2 0.3 0.5"

for ALPHA in $ALPHAS; do
    ALPHA_TAG=$(echo "$ALPHA" | sed 's/\./_/g')
    BETA_TAG=$(echo "$BEST_BETA" | sed 's/\./_/g')

    echo ""
    echo "--- EMA_BETA = $BEST_BETA  ASRI_ALPHA = $ALPHA ---"

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
        TEST.ADAPTATION.EMA_BETA "$BEST_BETA" \
        TEST.ADAPTATION.DPEMA_APPLY_GL True \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        OUTPUT_DIR "../outputs/COCO_TTA/exp5_dpema_beta_${BETA_TAG}_alpha_${ALPHA_TAG}"

    cp "../outputs/COCO_TTA/exp5_dpema_beta_${BETA_TAG}_alpha_${ALPHA_TAG}/eval_matrix/metrics.json" \
       "../results/exp5/metrics_beta_${BETA_TAG}_alpha_${ALPHA_TAG}.json" 2>/dev/null || true
done

# =============================================================================
# Run 3: Ablation comparison table + domain AP tables
# =============================================================================
echo ""
echo "======================================================================"
echo "Run 3: Ablation comparison table"
echo "======================================================================"

python3 - <<'PYEOF'
import json, os, glob

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"
]
WEATHER = ["snow", "frost"]
SHORT = {
    "gaussian_noise": "gauss", "shot_noise": "shot", "impulse_noise": "impls",
    "defocus_blur": "defoc", "glass_blur": "glass", "motion_blur": "motio",
    "zoom_blur": "zoom", "snow": "SNOW*", "frost": "FRST*", "fog": "fog",
    "brightness": "brite", "contrast": "contr", "elastic_transform": "elast",
    "pixelate": "pixel", "jpeg_compression": "jpeg"
}

def load_aps(path):
    with open(path) as f:
        m = json.load(f)
    return {c: m.get(f"coco_2017_val-{c}", {}).get("AP", float("nan"))
            for c in CORRUPTIONS}

# ── hardcoded baselines from log files ──────────────────────────────────────
# source-only (exp0)
source_only_json = "../results/exp0/metrics_source_only.json"
oracle_json = "../results/exp1/metrics_varB.json"

# baseline α=0.0 from log (exp2_alpha_0_0_log.txt, extracted manually)
BASELINE_APS = {
    "gaussian_noise": 15.33, "shot_noise": 18.80, "impulse_noise": 18.00,
    "defocus_blur": 12.09, "glass_blur": 4.59, "motion_blur": 7.29,
    "zoom_blur": 3.37, "snow": 9.12, "frost": 11.69, "fog": 23.86,
    "brightness": 27.43, "contrast": 17.08, "elastic_transform": 17.52,
    "pixelate": 9.33, "jpeg_compression": 11.49,
}

# ── load DPEMA results ────────────────────────────────────────────────────────
def best_by_avg(pattern):
    best_f, best_avg = None, -1.0
    for f in sorted(glob.glob(pattern)):
        aps = load_aps(f)
        avg = sum(v for v in aps.values() if v == v) / len(CORRUPTIONS)
        if avg > best_avg:
            best_avg, best_f = avg, f
    return best_f, best_avg

dpema_best_f, _ = best_by_avg("../results/exp5/metrics_beta_*.json")
dpema_asri_best_f, _ = best_by_avg("../results/exp5/metrics_beta_*_alpha_*.json")

methods = {}
if os.path.exists(source_only_json):
    methods["source_only"] = load_aps(source_only_json)
methods["baseline(α=0)"] = BASELINE_APS
if os.path.exists(oracle_json):
    methods["oracle(varB)"] = load_aps(oracle_json)
if dpema_best_f:
    methods["dpema_best"] = load_aps(dpema_best_f)
    methods["dpema_best"]["_file"] = os.path.basename(dpema_best_f)
if dpema_asri_best_f:
    methods["dpema+asri_best"] = load_aps(dpema_asri_best_f)
    methods["dpema+asri_best"]["_file"] = os.path.basename(dpema_asri_best_f)

# ── β sweep domain table ─────────────────────────────────────────────────────
print("\n" + "="*80)
print("β SWEEP — per-domain AP  (* = weather targets)")
print("="*80)
beta_files = sorted(glob.glob("../results/exp5/metrics_beta_*.json"),
                    key=lambda f: os.path.basename(f))
# exclude joint (those have "alpha" in name)
beta_files = [f for f in beta_files if "alpha" not in os.path.basename(f)]

if beta_files:
    header = f"{'domain':<10}" + "".join(f"{'β='+os.path.basename(f).replace('metrics_beta_','').replace('.json','').replace('_','.'):<12}" for f in beta_files)
    print(header)
    print("-" * len(header))
    for c in CORRUPTIONS:
        row = f"{SHORT[c]:<10}"
        for f in beta_files:
            aps = load_aps(f)
            row += f"{aps[c]:<12.2f}"
        print(row)
    print("-" * len(header))
    row = f"{'avg(15)':<10}"
    for f in beta_files:
        aps = load_aps(f)
        vals = [v for v in aps.values() if v == v]
        row += f"{sum(vals)/len(CORRUPTIONS):<12.2f}"
    print(row)

# ── Ablation comparison table ────────────────────────────────────────────────
print("\n" + "="*80)
print("ABLATION — method comparison  (* = key weather targets)")
print("="*80)

col_w = 16
method_names = [m for m in methods if not m.startswith("_")]
header = f"{'domain':<10}" + "".join(f"{m:<{col_w}}" for m in method_names)
print(header)
print("-" * len(header))
for c in CORRUPTIONS:
    row = f"{SHORT[c]:<10}"
    for m in method_names:
        v = methods[m].get(c, float("nan"))
        row += f"{v:<{col_w}.2f}"
    print(row)
print("-" * len(header))
row = f"{'avg(15)':<10}"
for m in method_names:
    vals = [methods[m].get(c, float("nan")) for c in CORRUPTIONS]
    vals = [v for v in vals if v == v]
    row += f"{sum(vals)/len(CORRUPTIONS):<{col_w}.2f}"
print(row)

# ── Success criteria check ───────────────────────────────────────────────────
print("\n" + "="*80)
print("SUCCESS CRITERIA CHECK")
print("="*80)
print(f"  Reference: source_only snow=19.11  frost=22.96  avg=15.29")
print(f"  Reference: baseline     snow=9.12   frost=11.69  avg=13.80")
print(f"  Reference: oracle       snow=17.34  (avg=15.64)")
print()

for label, key in [("dpema_best", "dpema_best"), ("dpema+asri_best", "dpema+asri_best")]:
    if key not in methods:
        continue
    d = methods[key]
    fname = d.get("_file", "")
    snow = d.get("snow", float("nan"))
    frost = d.get("frost", float("nan"))
    avg = sum(d.get(c, 0.0) for c in CORRUPTIONS) / len(CORRUPTIONS)
    clean_key = "coco_2017_val"
    # clean AP not in per-corruption dict; skip
    print(f"  [{label}]  file={fname}")
    print(f"    snow  AP = {snow:.2f}  {'PASS (>15.0)' if snow > 15.0 else 'FAIL (<15.0)'}")
    print(f"    frost AP = {frost:.2f}  {'PASS (>18.0)' if frost > 18.0 else 'FAIL (<18.0)'}")
    print(f"    avg15 AP = {avg:.2f}  {'PASS (>15.29)' if avg > 15.29 else 'FAIL (<15.29)'}")
    print()

PYEOF

echo ""
echo "=== Exp 5 complete. Results in results/exp5/ ==="
