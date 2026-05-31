# Results

This directory contains experiment outputs for the COCO -> COCO-C continual
test-time adaptation runs. Each `exp*` folder groups metrics, evaluation
matrices, drift logs, or summary files from one experiment family.

| Folder | Experiment | What it tests | Main files |
|---|---|---|---|
| `exp0` | Evaluation protocol setup | Source-only and baseline CTTAOD evaluation on the 15 COCO-C corruptions. This is the reference setup for later comparisons. | `metrics_source_only.json` |
| `exp1` | Forgetting cause diagnosis | Separates two possible forgetting sources: adapter reset with EMA prototypes (`varA`) vs. continual adapter with oracle/source prototypes (`varB`). | `metrics_varA.json`, `metrics_varB.json` |
| `exp5` | DPEMA + ASRI sweep | Tests decaying prototype EMA with fixed source residual injection strength. The saved runs use `beta=0.999` with `alpha` values 0.1, 0.2, and 0.3. | `metrics_beta_0_999_alpha_*.json` |
| `exp6` | Adaptive ASRI | Tests adaptive source residual injection where `alpha_t` is scaled by prototype distance from the source statistics. Sweeps `alpha_max` values. | `metrics_adaptive_alphamax_*.json` |
| `exp8` | Confidence-gated ASRI | Tests per-class confidence/count gated ASRI so sparse or low-confidence pseudo-labels keep a stronger source anchor. Sweeps gate lambda and alpha. | `metrics_lambda_*_alpha_*.json` |
| `exp9` | Global-branch ASRI | Applies source residual injection to the global branch as well as foreground prototypes, then sweeps ASRI alpha. | `metrics_asri_gl_alpha_*.json` |
| `exp10` | Adapter drift and pseudo-label quality | Tests adapter EWC regularization and confidence-weighted prototype updates on the DPEMA + ASRI global-branch base. | `metrics_r0_baseline.json`, `metrics_r1_ewc*.json`, `metrics_r2*.json` |
| `exp11` | EWC extension study | Extends adapter EWC with layer-normalized Fisher, drift-adaptive lambda, sliding anchors, MAS importance, and a combined variant. | `exp11_summary.json`, `metrics_e*.json` |
| `exp12` | Sol A/B/C prototype methods | Ports prototype-side methods from ContinualTTA Object Detection: Sol-A prototype reset, Sol-B dual memory, and Sol-C adaptive gamma. | `exp12_summary.json`, `metrics_e*.json` |
| `exp13` | Legacy gamma + EWC combinations | Checks whether EWC still helps when the stronger legacy gamma prototype update is used, and combines EWC with Sol-A/Sol-B sweeps. | `exp13_summary.json`, `metrics_e*.json` |
| `exp14` | Two-coupled-drift diagnosis | Collects causal evidence for prototype/statistical-memory drift and adapter/parametric-memory drift. Includes baseline, oracle prototype, adapter reset, EWC, prototype reset, and prototype reset + EWC. | `exp14_summary.json`, `drift_*.jsonl`, `metrics_d*.json` |
| `exp15` | Dual-memory + EWC drift diagnosis | Repeats the two-drift diagnosis from `exp14`, replacing prototype reset with Sol-B dual memory and testing Sol-B + EWC. | `exp15_summary.json`, `drift_*.jsonl`, `metrics_d*.json` |
| `exp16` | Sol-A + Sol-B + EWC diagnosis | Tests whether Sol-A prototype reset, Sol-B dual memory, and adapter EWC stack together, while keeping causal controls. | `exp16_summary.json`, `exp16_evidence.json`, `drift_*.jsonl`, `metrics_d*.json` |
| `exp17` | Direct prototype-drift evidence | Produces reviewer-facing diagnostics for the claim that prototype drift causes forgetting, including drift-performance correlations and reset-event windows. | `exp17_summary.json`, `exp17_direct_evidence.json`, `exp17_*_windows.json` |
| `exp18` | Direct evidence for Sol-B + EWC | Focuses on Sol-B + EWC and Sol-A + EWC using eval matrices, drift logs, high-drift events, and domain-boundary before/after analysis. | `exp18_summary.json`, `exp18_direct_evidence.json`, `eval_matrix_*.npy`, `drift_*.jsonl` |

Notes:

- Folder numbers are not continuous because some planned experiments do not
  currently have result folders in this repository.
- `metrics_*.json` files usually contain per-corruption AP metrics and, when
  available, aggregate metrics such as average AP/mAP, BWT, or FWT.
- `drift_*.jsonl` files are step-level diagnostic logs used by the later drift
  evidence experiments.
- `eval_matrix*.npy` files are continual evaluation matrices used to compute
  forgetting and transfer metrics.
