"""
Exp 3: Category-Level Forgetting Analysis

Uses the per-class eval matrix from Exp 0 to compute per-class BWT and
correlate it with class frequency in the source domain.

Usage (from tools/ directory):
    python analyze_exp3.py \
        --matrix   ../results/exp0/eval_matrix_baseline_per_class.npy \
        --out-dir  ../results/exp3

Outputs (in out-dir/):
    per_class_bwt.json       — {class_name: BWT_k} for all 8 classes
    correlation_analysis.json — Pearson r between log(freq) and BWT_k
    class_freq_vs_bwt.png    — scatter plot

Cityscapes class frequencies (approx. from train split, used if no freq file):
    person, rider, car, truck, bus, train, motorcycle, bicycle
"""

import argparse
import json
import os
import numpy as np

CITYSCAPES_DET_CLASSES = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# Approximate per-class instance counts in Cityscapes train split
# (from the official Cityscapes statistics)
CITYSCAPES_TRAIN_FREQ = {
    "person":     17898,
    "rider":       1755,
    "car":        26944,
    "truck":       483,
    "bus":         379,
    "train":       168,
    "motorcycle":  735,
    "bicycle":    3658,
}

CITYSCAPES_C_SEQUENCE = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def compute_per_class_bwt(matrix_per_class):
    """
    matrix_per_class: shape (T+1, T, num_classes)
        a[j][i][k] = AP for class k on domain i after adapting through domains 1..j

    BWT_k = mean over i in [0..T-2] of (a[T][i][k] - a[i+1][i][k])
    """
    T = matrix_per_class.shape[1]
    bwt_per_class = []
    for k in range(matrix_per_class.shape[2]):
        bwt_k = np.mean([
            matrix_per_class[T, i, k] - matrix_per_class[i + 1, i, k]
            for i in range(T - 1)
        ])
        bwt_per_class.append(float(bwt_k))
    return bwt_per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="../results/exp0/eval_matrix_baseline_per_class.npy")
    parser.add_argument("--out-dir", default="../results/exp3")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    matrix = np.load(args.matrix)
    print(f"Loaded per-class eval matrix: shape {matrix.shape}")

    T = matrix.shape[1]
    num_classes = matrix.shape[2]
    class_names = CITYSCAPES_DET_CLASSES[:num_classes]

    # 1. Per-class BWT
    bwt_per_class = compute_per_class_bwt(matrix)
    per_class_bwt = {class_names[k]: bwt_per_class[k] for k in range(num_classes)}

    print("\nPer-class BWT (negative = forgetting):")
    for name, bwt in sorted(per_class_bwt.items(), key=lambda x: x[1]):
        print(f"  {name:<15}  BWT = {bwt:+.4f}")

    bwt_path = os.path.join(args.out_dir, "per_class_bwt.json")
    with open(bwt_path, "w") as f:
        json.dump(per_class_bwt, f, indent=2)
    print(f"\nSaved to {bwt_path}")

    # 2. Correlation: log(class_freq) vs BWT_k
    freqs = np.array([CITYSCAPES_TRAIN_FREQ.get(c, 1) for c in class_names], dtype=float)
    log_freqs = np.log(freqs)
    bwts = np.array(bwt_per_class)

    if len(bwts) > 1:
        from numpy.polynomial import polynomial as P
        corr = float(np.corrcoef(log_freqs, bwts)[0, 1])
    else:
        corr = float("nan")

    print(f"\nPearson r (log_freq vs BWT): {corr:.4f}")
    if corr < -0.5:
        print("  → Rare classes suffer significantly more forgetting.")
        print("    This motivates class-specific injection: α_k higher for rare classes.")
    elif corr > 0.5:
        print("  → Frequent classes suffer more forgetting (unexpected).")
    else:
        print("  → No strong correlation between frequency and forgetting.")

    corr_data = {
        "pearson_r": corr,
        "class_log_freq": {c: float(log_freqs[i]) for i, c in enumerate(class_names)},
        "class_bwt": per_class_bwt,
        "class_freq": {c: int(freqs[i]) for i, c in enumerate(class_names)},
    }
    corr_path = os.path.join(args.out_dir, "correlation_analysis.json")
    with open(corr_path, "w") as f:
        json.dump(corr_data, f, indent=2)
    print(f"Saved correlation analysis to {corr_path}")

    # 3. Scatter plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(log_freqs, bwts, c="steelblue", s=80, zorder=3)
        for i, name in enumerate(class_names):
            ax.annotate(name, (log_freqs[i], bwts[i]),
                        textcoords="offset points", xytext=(5, 3), fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("log(class frequency in source train)")
        ax.set_ylabel("BWT_k  (negative = forgetting)")
        ax.set_title(f"Class Frequency vs. Forgetting (r={corr:.3f})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(args.out_dir, "class_freq_vs_bwt.png")
        fig.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Scatter plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not available; skipping scatter plot.")


if __name__ == "__main__":
    main()