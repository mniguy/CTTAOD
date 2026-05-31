#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
from statistics import mean


CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

FAMILIES = {
    "Noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "Blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "Weather": ["snow", "frost", "fog", "brightness"],
    "Digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    "Original": ["original"],
}

WHW_REFERENCE = {
    "Direct-Test": {
        "per_domain_AP": {
            "gaussian_noise": 9.7, "shot_noise": 11.4, "impulse_noise": 10.0,
            "defocus_blur": 13.4, "glass_blur": 7.5, "motion_blur": 12.1,
            "zoom_blur": 5.2, "snow": 20.7, "frost": 24.8, "fog": 36.1,
            "brightness": 36.0, "contrast": 12.9, "elastic_transform": 19.1,
            "pixelate": 4.9, "jpeg_compression": 15.8, "original": 43.0,
        },
        "backward_images": 0,
        "fps": 21.5,
    },
    "Ours": {
        "per_domain_AP": {
            "gaussian_noise": 13.6, "shot_noise": 16.6, "impulse_noise": 16.1,
            "defocus_blur": 14.0, "glass_blur": 13.6, "motion_blur": 14.2,
            "zoom_blur": 8.3, "snow": 23.7, "frost": 27.2, "fog": 37.4,
            "brightness": 36.4, "contrast": 27.2, "elastic_transform": 27.2,
            "pixelate": 22.2, "jpeg_compression": 22.3, "original": 42.3,
        },
        "backward_images": 80000,
        "fps": 9.5,
    },
    "Ours-Skip": {
        "per_domain_AP": {
            "gaussian_noise": 13.3, "shot_noise": 15.3, "impulse_noise": 15.1,
            "defocus_blur": 14.0, "glass_blur": 12.8, "motion_blur": 13.9,
            "zoom_blur": 6.5, "snow": 22.0, "frost": 25.4, "fog": 35.5,
            "brightness": 34.9, "contrast": 26.5, "elastic_transform": 25.9,
            "pixelate": 23.4, "jpeg_compression": 20.2, "original": 41.2,
        },
        "backward_images": 9700,
        "fps": 17.7,
    },
}

LABELS = {
    "direct": "Direct-Test",
    "whw": "WHW-style Ours",
    "whw_skip": "WHW-style Ours-Skip",
    "ewc": "EWC only",
    "solb": "Sol-B only",
    "solb_ewc": "Sol-B + EWC",
    "adaptive_alpha": "Adaptive alpha",
    "adaptive_alpha_lambda": "Adaptive alpha + lambda",
}


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def avg(vals):
    vals = [v for v in vals if finite(v)]
    return mean(vals) if vals else None


def read_json(path):
    try:
        with open(path) as fp:
            return json.load(fp)
    except Exception:
        return {}


def get_ap(metrics, domain):
    candidates = [
        f"coco_2017_val-{domain}",
        f"coco_val-{domain}",
        domain,
    ]
    if domain == "original":
        candidates = ["coco_2017_val", "coco_val", "original"] + candidates
    for key in candidates:
        val = metrics.get(key)
        if isinstance(val, dict):
            if "bbox" in val and isinstance(val["bbox"], dict):
                val = val["bbox"]
            if finite(val.get("AP")):
                return float(val["AP"])
    return None


def family_ap(per_domain):
    return {name: avg(per_domain.get(d) for d in domains) for name, domains in FAMILIES.items()}


def ap15(per_domain):
    return avg(per_domain.get(d) for d in CORRUPTIONS)


def ap16(per_domain):
    vals = [per_domain.get(d) for d in CORRUPTIONS] + [per_domain.get("original")]
    return avg(vals)


def with_ref_metrics(ref):
    out = dict(ref)
    out["AP15"] = ap15(out["per_domain_AP"])
    out["AP16"] = ap16(out["per_domain_AP"])
    out["family_AP"] = family_ap(out["per_domain_AP"])
    return out


def read_drift(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fp:
        for line in fp:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def drift_summary(rows):
    by_domain = {}
    for row in rows:
        d = row.get("domain_name")
        if d is None:
            continue
        by_domain.setdefault(d, []).append(row)
    out = {}
    for domain, vals in by_domain.items():
        out[domain] = {
            "fg_score_mean": avg(v.get("fg_score_mean") for v in vals),
            "fg_boxes_mean": avg(v.get("fg_num_boxes") for v in vals),
            "proto_drift_mean": avg(v.get("proto_drift_source_mean") for v in vals),
            "adapter_fisher_mean": avg(v.get("adapter_fisher") for v in vals),
            "source_anchor_alpha_mean": avg(v.get("source_anchor_alpha_mean") for v in vals),
            "effective_ewc_lambda_mean": avg(v.get("effective_ewc_lambda") for v in vals),
        }
    return out


def parse_run(path, results_dir):
    tag = os.path.basename(path).replace("metrics_", "").replace(".json", "")
    metrics = read_json(path)
    meta = read_json(os.path.join(results_dir, f"meta_{tag}.json"))
    per_domain = {d: get_ap(metrics, d) for d in CORRUPTIONS + ["original"]}
    whw = with_ref_metrics(WHW_REFERENCE["Ours"])
    deltas = {
        d: (per_domain[d] - whw["per_domain_AP"][d])
        for d in per_domain
        if finite(per_domain.get(d)) and finite(whw["per_domain_AP"].get(d))
    }
    fam = family_ap(per_domain)
    whw_fam = whw["family_AP"]
    drift = drift_summary(read_drift(os.path.join(results_dir, f"drift_{tag}.jsonl")))
    return {
        "tag": tag,
        "method": LABELS.get(tag, tag),
        "per_domain_AP": per_domain,
        "AP15": ap15(per_domain),
        "AP16": ap16(per_domain),
        "family_AP": fam,
        "delta_vs_WHW_Ours": {
            "AP15": ap15(per_domain) - whw["AP15"] if finite(ap15(per_domain)) else None,
            "AP16": ap16(per_domain) - whw["AP16"] if finite(ap16(per_domain)) else None,
            "per_domain": deltas,
            "family": {
                k: fam[k] - whw_fam[k]
                for k in fam
                if finite(fam.get(k)) and finite(whw_fam.get(k))
            },
        },
        "config": {
            "backward_batches": meta.get("backward_batches"),
            "backward_images": meta.get("backward_images"),
            "fps": meta.get("fps"),
        },
        "diagnostics": {"drift_by_domain": drift},
    }


def write_table(path, summary):
    def fmt(v, digits=2):
        return f"{v:.{digits}f}" if finite(v) else "nan"

    runs = sorted(summary["runs"], key=lambda r: r["AP16"] if finite(r.get("AP16")) else -1e9, reverse=True)
    lines = [
        "# Exp20 Swin-T COCO -> COCO-C Summary",
        "",
        "WHW `Avg.` is treated as AP16, including `Org.`.",
        "",
        "## Completed Runs",
        "",
        "| Rank | Tag | Method | AP15 | AP16 | Delta WHW Ours | Back imgs | FPS | Noise Delta | Digital Delta |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, run in enumerate(runs, 1):
        cfg = run.get("config", {})
        lines.append(
            f"| {rank} | `{run['tag']}` | {run['method']} | "
            f"{fmt(run.get('AP15'))} | {fmt(run.get('AP16'))} | "
            f"{fmt(run['delta_vs_WHW_Ours'].get('AP16'))} | "
            f"{cfg.get('backward_images', '')} | {cfg.get('fps', '')} | "
            f"{fmt(run['delta_vs_WHW_Ours']['family'].get('Noise'))} | "
            f"{fmt(run['delta_vs_WHW_Ours']['family'].get('Digital'))} |"
        )
    lines.extend([
        "",
        "## WHW Swin-T Reference",
        "",
        "| Method | AP15 | AP16 | Back imgs | FPS |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, ref in summary["whw_reference"].items():
        lines.append(f"| {name} | {ref['AP15']:.2f} | {ref['AP16']:.2f} | {ref['backward_images']} | {ref['fps']} |")
    lines.extend(["", "## Per-Domain AP", ""])
    header = ["Tag", "Gau", "Sht", "Imp", "Def", "Gls", "Mtn", "Zm", "Snw", "Frs", "Fog", "Brt", "Cnt", "Els", "Px", "Jpg", "Org"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---" + "|---:" * (len(header) - 1) + "|")
    for run in runs:
        vals = [run["per_domain_AP"].get(d) for d in CORRUPTIONS + ["original"]]
        lines.append("| `" + run["tag"] + "` | " + " | ".join(f"{v:.1f}" if finite(v) else "nan" for v in vals) + " |")
    with open(path, "w") as fp:
        fp.write("\n".join(lines) + "\n")


def maybe_plot(results_dir, runs):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not runs:
        return
    domains = CORRUPTIONS + ["original"]
    whw = with_ref_metrics(WHW_REFERENCE["Ours"])
    best = max(runs, key=lambda r: r["AP16"] if finite(r.get("AP16")) else -1e9)
    deltas = [best["per_domain_AP"].get(d, float("nan")) - whw["per_domain_AP"].get(d, float("nan")) for d in domains]
    plt.figure(figsize=(12, 4))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.bar(range(len(domains)), deltas)
    plt.xticks(range(len(domains)), domains, rotation=45, ha="right")
    plt.ylabel("AP delta vs WHW Ours")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp20_delta_vs_whw.png"), dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results/exp20")
    args = parser.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "swint_summary.json")
    existing_summary = read_json(summary_path)
    refs = {k: with_ref_metrics(v) for k, v in WHW_REFERENCE.items()}
    runs = [parse_run(p, results_dir) for p in sorted(glob.glob(os.path.join(results_dir, "metrics_*.json")))]
    if not runs and existing_summary.get("runs"):
        print(json.dumps(existing_summary.get("summary", {}), indent=2))
        return
    existing_tags = {r.get("tag") for r in runs}
    for old_run in existing_summary.get("runs", []):
        if old_run.get("tag") not in existing_tags:
            runs.append(old_run)
    best = max(runs, key=lambda r: r["AP16"] if finite(r.get("AP16")) else -1e9, default=None)
    summary = {
        "summary": {
            "best_tag": best["tag"] if best else None,
            "best_AP16": best["AP16"] if best else None,
            "whw_ours_AP16": refs["Ours"]["AP16"],
            "decision": "complete Swin-T sweeps before deciding claim strength" if not best else (
                "near-parity" if best["AP16"] >= refs["Ours"]["AP16"] - 0.3 else "needs adaptive follow-up"
            ),
        },
        "whw_reference": refs,
        "runs": runs,
    }
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    write_table(os.path.join(results_dir, "swint_summary_table.md"), summary)
    maybe_plot(results_dir, runs)
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
