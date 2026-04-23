#!/usr/bin/env bash
# =============================================================================
# Exp 3: Category-Level Forgetting Analysis
#
# Uses the per-class eval matrix already produced by Exp 0 to answer:
#   "Is forgetting concentrated in rare/frequent classes?"
#
# Prerequisites: Exp 0 must have run first (produces
#   results/exp0/eval_matrix_baseline_per_class.npy)
#
# Usage:
#   bash scripts/exp3_category_analysis.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT/tools"

MATRIX="../results/exp0/eval_matrix_baseline_per_class.npy"

if [ ! -f "$MATRIX" ]; then
    echo "ERROR: $MATRIX not found."
    echo "Run scripts/exp0_cityscapes.sh first."
    exit 1
fi

mkdir -p ../results/exp3

echo "=== Exp 3: Category-Level Forgetting Analysis ==="
python analyze_exp3.py \
    --matrix "$MATRIX" \
    --out-dir ../results/exp3

echo ""
echo "=== Exp 3 complete. Results in results/exp3/ ==="
ls ../results/exp3/