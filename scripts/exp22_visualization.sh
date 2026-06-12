#!/usr/bin/env bash
# =============================================================================
# Exp 22: Visualization / robustness studies for DDAS on COCO -> COCO-C (R50)
#
# Two studies, each toggleable via EXP22_TIERS:
#
#   Tier 1  (t1)  Corruption-order robustness. The main results use one fixed
#                 corruption order; here we re-run baseline vs Ours under
#                 {default, reverse, random(seed)} orders and check that the
#                 online gain of Ours is not an artifact of a single ordering.
#
#   Tier 2  (t2)  Multi-round continual stream. The 15-corruption stream is
#                 replayed EXP22_NUM_ROUNDS times back-to-back without resetting
#                 the online state, to stress-test long-horizon stability of
#                 baseline vs Ours.
#
# These rely on three new config keys (see detectron2/config/defaults.py):
#   TEST.ADAPTATION.DOMAIN_ORDER       default|reverse|random
#   TEST.ADAPTATION.DOMAIN_ORDER_SEED  int (random order seed)
#   TEST.ADAPTATION.NUM_ROUNDS         int (>=1, stream repetitions)
#
# Outputs (under results/exp22/):
#   Tier 1:  order_<cfg>_<order>.json
#   Tier 2:  multiround_<cfg>.json
#   Summary: exp22_summary.json  (printed table + machine-readable)
#
# Usage:
#   bash scripts/exp22_visualization.sh [CHECKPOINT_PATH]
#
# Optional env:
#   EXP22_TIERS="t1 t2"         which studies to run (default all)
#   EXP22_NUM_ROUNDS=3          Tier-2 stream repetitions (default 3)
#   EXP22_RANDOM_SEEDS="1 2"    Tier-1 random-order seeds (default "1 2")
#   EXP22_SOURCE_ANCHOR_ALPHA=0.40
#   EXP22_EWC_LAMBDA=10.0
#   EXP22_ANALYZE_ONLY=False    skip runs, only (re)build the summary
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

CKPT="${1:-../models/checkpoints/faster_rcnn_r50_coco.pth}"
CFG="../configs/TTA/COCO_R50.yaml"
STATS_PATH="../models/stats/COCO_R50_stats.pt"
FISHER_PATH="../models/stats/COCO_R50_fisher.pt"
RESDIR="../results/exp22"

EXP22_TIERS="${EXP22_TIERS:-t1 t2}"
EXP22_NUM_ROUNDS="${EXP22_NUM_ROUNDS:-3}"
EXP22_RANDOM_SEEDS="${EXP22_RANDOM_SEEDS:-1 2}"
ALPHA="${EXP22_SOURCE_ANCHOR_ALPHA:-0.40}"
LAMBDA="${EXP22_EWC_LAMBDA:-10.0}"
EXP22_ANALYZE_ONLY="${EXP22_ANALYZE_ONLY:-False}"

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/datasets}"
mkdir -p "$RESDIR"

has_tier() { [[ " $EXP22_TIERS " == *" $1 "* ]]; }

# Per-config TEST.ADAPTATION overrides (tokens are space-free -> word-splitting
# the command substitution is safe and intentional).
overrides_for() {
    case "$1" in
        baseline) echo "TEST.ADAPTATION.PROTO_METHOD baseline TEST.ADAPTATION.EWC_LAMBDA 0.0" ;;
        ours)     echo "TEST.ADAPTATION.PROTO_METHOD dual_memory TEST.ADAPTATION.SOURCE_ANCHOR_ALPHA ${ALPHA} TEST.ADAPTATION.EWC_LAMBDA ${LAMBDA}" ;;
    esac
}

COMMON=(
    --config-file "$CFG"
    --eval-only
    MODEL.WEIGHTS "$CKPT"
    TEST.ONLINE_ADAPTATION True
    TEST.CONTINUAL_DOMAIN True
    TEST.ADAPTATION.CONTINUAL True
    TEST.ADAPTATION.WHERE "adapter"
    TEST.ADAPTATION.GLOBAL_ALIGN "KL"
    TEST.ADAPTATION.FOREGROUND_ALIGN "KL"
    TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH"
    TEST.ADAPTATION.EWC_FISHER_PATH "$FISHER_PATH"
    TEST.ADAPTATION.EMA_BETA 0.0
    TEST.ADAPTATION.SWEMA_K 0
    TEST.ADAPTATION.ASRI_ALPHA 0.0
    TEST.ADAPTATION.ADAPTER_RESET False
    TEST.ADAPTATION.ORACLE_PROTOTYPE False
)

# Fisher importance (needed by the EWC configs); computed once on source data.
if [ "$EXP22_ANALYZE_ONLY" != "True" ] && [ ! -f "$FISHER_PATH" ]; then
    echo "=== Pre-step: computing Fisher information for adapter EWC ==="
    python compute_fisher.py \
        --config-file "$CFG" \
        MODEL.WEIGHTS "$CKPT" \
        TEST.ADAPTATION.SOURCE_FEATS_PATH "$STATS_PATH" \
        TEST.ADAPTATION.WHERE "adapter"
fi

run() {  # run <output_subdir> <extra cfg args...>
    local out="../outputs/COCO_TTA/$1"; shift
    if [ "$EXP22_ANALYZE_ONLY" = "True" ]; then
        echo "=== Analyze-only: skip $out ==="; return
    fi
    echo ""
    echo "=== RUN $out ==="
    python train_net.py "${COMMON[@]}" "$@" OUTPUT_DIR "$out"
    echo "$out"
}

# ---------------------------------------------------------------------------
# Tier 1: corruption-order robustness (online AP; baseline vs ours).
# ---------------------------------------------------------------------------
if has_tier t1; then
    echo "########## Tier 1: corruption-order robustness ##########"
    for name in baseline ours; do
        # build (order:seed) list: default, reverse, then one per random seed
        specs=( "default:0" "reverse:0" )
        for s in $EXP22_RANDOM_SEEDS; do specs+=( "random:${s}" ); done
        for spec in "${specs[@]}"; do
            order="${spec%%:*}"; seed="${spec##*:}"
            tag="${name}_${order}"
            [ "$order" = "random" ] && tag="${name}_random${seed}"
            out="exp22_t1_${tag}"
            run "$out" $(overrides_for "$name") \
                TEST.EVAL_MATRIX False \
                TEST.ADAPTATION.DOMAIN_ORDER "$order" \
                TEST.ADAPTATION.DOMAIN_ORDER_SEED "$seed"
            cp "../outputs/COCO_TTA/${out}/eval_matrix/metrics.json" "$RESDIR/order_${tag}.json" 2>/dev/null || true
        done
    done
fi

# ---------------------------------------------------------------------------
# Tier 2: multi-round continual stream (online AP; baseline vs ours).
# ---------------------------------------------------------------------------
if has_tier t2; then
    echo "########## Tier 2: multi-round (${EXP22_NUM_ROUNDS} rounds) ##########"
    for name in baseline ours; do
        out="exp22_t2_${name}_r${EXP22_NUM_ROUNDS}"
        run "$out" $(overrides_for "$name") \
            TEST.EVAL_MATRIX False \
            TEST.ADAPTATION.NUM_ROUNDS "$EXP22_NUM_ROUNDS"
        cp "../outputs/COCO_TTA/${out}/eval_matrix/metrics.json" "$RESDIR/multiround_${name}.json" 2>/dev/null || true
    done
fi

# ---------------------------------------------------------------------------
# Analysis: build results/exp22/exp22_summary.json and print readable tables.
# ---------------------------------------------------------------------------
echo ""
echo "########## Analysis ##########"
python3 - "$RESDIR" <<'PYEOF'
import json, os, re, sys, glob
import numpy as np

RESDIR = sys.argv[1]
CORR = ["gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
        "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
        "elastic_transform","pixelate","jpeg_compression"]
summary = {}

def ap_of(v):
    # metrics values may be {"AP":...} (flat) or {"bbox":{"AP":...}}
    if not isinstance(v, dict): return None
    if "AP" in v: return v["AP"]
    return v.get("bbox", {}).get("AP")

# ---- Tier 1: order robustness (mean online corruption AP per order) --------
t1 = {}
for f in sorted(glob.glob(os.path.join(RESDIR, "order_*.json"))):
    tag = os.path.basename(f)[len("order_"):-len(".json")]
    m = json.load(open(f))
    aps = [ap_of(m.get(f"coco_2017_val-{c}")) for c in CORR]
    aps = [a for a in aps if a is not None]
    clean = ap_of(m.get("coco_2017_val"))
    t1[tag] = {"corr_AvgAP": float(np.mean(aps)) if aps else None, "clean_AP": clean}
if t1:
    summary["tier1_order"] = t1
    print("\n=== Tier 1: corruption-order robustness (online corruption AvgAP) ===")
    for tag in sorted(t1):
        r = t1[tag]
        print(f"  {tag:22s} corrAvgAP={r['corr_AvgAP']}  cleanAP={r['clean_AP']}")
    # baseline vs ours delta per matched order
    def order_key(t): return t.split("_", 1)[1]
    base = {order_key(t): v["corr_AvgAP"] for t, v in t1.items() if t.startswith("baseline_")}
    ours = {order_key(t): v["corr_AvgAP"] for t, v in t1.items() if t.startswith("ours_")}
    print("  --- Ours - Baseline (per order) ---")
    for o in sorted(set(base) & set(ours)):
        print(f"    {o:12s} Δ={ours[o]-base[o]:+.2f}  (base={base[o]:.2f} ours={ours[o]:.2f})")

# ---- Tier 2: multi-round per-round average online corruption AP ------------
t2 = {}
pat = re.compile(r"^coco_2017_val-rnd(\d+)-(.+)$")
for f in sorted(glob.glob(os.path.join(RESDIR, "multiround_*.json"))):
    name = os.path.basename(f)[len("multiround_"):-len(".json")]
    m = json.load(open(f))
    per_round = {}
    for k, v in m.items():
        mt = pat.match(k)
        if not mt: continue
        r = int(mt.group(1)); a = ap_of(v)
        if a is not None: per_round.setdefault(r, []).append(a)
    rounds = {r: float(np.mean(v)) for r, v in sorted(per_round.items())}
    t2[name] = {"per_round_corr_AvgAP": rounds, "clean_AP": ap_of(m.get("coco_2017_val"))}
if t2:
    summary["tier2_multiround"] = t2
    print("\n=== Tier 2: multi-round corruption AvgAP per round ===")
    for name in sorted(t2):
        rs = t2[name]["per_round_corr_AvgAP"]
        traj = "  ".join(f"r{r}={ap:.2f}" for r, ap in rs.items())
        print(f"  {name:10s} {traj}")

with open(os.path.join(RESDIR, "exp22_summary.json"), "w") as fp:
    json.dump(summary, fp, indent=2)
print(f"\nSaved {os.path.join(RESDIR, 'exp22_summary.json')}")
PYEOF

echo ""
echo "Done. See $RESDIR/exp22_summary.json"
