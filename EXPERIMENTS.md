# ASRI Experiment Suite — Claude Code Prompt

## Context

I am extending **CTTAOD** (Continual Test-Time Adaptation for Object Detection, CVPR 2024) with a method called **ASRI** (Adaptive Source Residual Injection). Before implementing the full method, I need to run a series of diagnostic and validation experiments to test my core hypothesis.

### Core Hypothesis
> "Continual TTAOD suffers catastrophic forgetting because target prototypes gradually deviate from source knowledge. Injecting source statistics as a residual into the target prototype estimation can mitigate this forgetting."

### Base Method: CTTAOD (WHW, CVPR 2024)
- **Architecture:** Faster R-CNN (ResNet-based backbone), frozen backbone, only lightweight adapter modules updated
- **Image-level alignment loss:**
  ```
  L_img = KL(N(μ_tr, Σ_tr) || N(μ_te^t, Σ_tr))
  ```
  where μ_tr, Σ_tr = source domain image-level feature statistics (pre-collected),
  μ_te^t = EMA-estimated target image-level feature statistics at time t
- **Region-level (object-level) class-wise alignment loss:**
  ```
  L_obj = Σ_k w_{k,t} · KL(N(μ_tr^k, Σ_tr^k) || N(μ_te^{k,t}, Σ_tr^k))
  ```
  where μ_tr^k = source class-k prototype (pre-collected from RoI-pooled features),
  w_{k,t} = class reweighting factor (higher for rare classes),
  μ_te^{k,t} = EMA-estimated target class-k prototype
- **"When to update" criteria:** Based on L_img threshold (Criterion 1 & 2 in paper)
- **Total loss:** `L = L_img + L_obj`
- **Benchmark:** Cityscapes → Cityscapes-C (15 corruption types as continual domain sequence), also UAVDT-C, ACDC

### Proposed ASRI Modification
The core idea: blend source prototype into EMA-estimated target prototype via adaptive scalar α_t:
```
μ̃_te^{k,t} = (1 - α_t) · μ_te^{k,t} + α_t · μ_tr^k
```
This replaces μ_te^{k,t} in L_obj.

---

## Experiment Plan

### Important: Metric Definitions (Used Across All Experiments)

Define and compute these metrics consistently:

```python
# Evaluation matrix: a[j][i] = mAP on domain i after adapting through domains 1..j
# T = number of domains in sequence (e.g., 15 for Cityscapes-C)

# Backward Transfer (forgetting measure, negative = forgetting)
BWT = (1 / (T - 1)) * sum(a[T][i] - a[i][i] for i in range(1, T))

# Forward Transfer (adaptation benefit)
FWT = (1 / (T - 1)) * sum(a[i][i] - a[0][i] for i in range(2, T + 1))
# a[0][i] = source model (no adaptation) evaluated on domain i

# Average mAP across all domains after full sequence
avg_mAP = (1 / T) * sum(a[T][i] for i in range(1, T + 1))
```

For ALL experiments below, report: **BWT, FWT, avg_mAP**, and per-domain mAP curves.

---

### Exp 0: Evaluation Protocol Setup

**Goal:** Establish the continual evaluation infrastructure before running any method variants.

**Tasks:**
1. Set up CTTAOD codebase (likely based on the official WHW repo or a reimplementation with Faster R-CNN + adapter modules).
2. Prepare Cityscapes → Cityscapes-C benchmark:
   - Source: Cityscapes (clear weather, urban driving)
   - Target sequence: 15 corruption types from Cityscapes-C applied in a fixed order
   - Each corruption = one "domain" in the continual sequence
3. Implement the evaluation matrix `a[j][i]`:
   - After adapting to domain j, evaluate on ALL domains 1..T (and source domain 0)
   - This creates a T×T matrix (or (T+1)×T including source)
   - Store per-class mAP as well (needed for Exp 3)
4. Run baseline CTTAOD through the full sequence and compute BWT, FWT, avg_mAP.
5. Also evaluate "Source Only" (no adaptation, frozen source model) on all domains → row 0 of the matrix.

**Output:** 
- `eval_matrix_baseline.npy` — shape (T+1, T) float array
- `eval_matrix_baseline_per_class.npy` — shape (T+1, T, num_classes) float array  
- Console log of BWT, FWT, avg_mAP for baseline CTTAOD
- Console log of avg_mAP for Source Only

---

### Exp 1: Forgetting Cause Diagnosis (HIGHEST PRIORITY)

**Goal:** Determine whether forgetting in CTTAOD is primarily caused by (A) prototype estimation error accumulation, or (B) adapter weight interference across domains.

**Variants to implement:**

#### Variant A — Adapter Reset + EMA Prototype
- At each domain transition (every time a new corruption type starts), **reset adapter weights to source-initialized state** (zeros or whatever the original init is).
- Keep the EMA prototype accumulation running **without reset** across domains.
- This isolates the effect of prototype drift: if forgetting still occurs, prototype estimation noise is the primary cause.

#### Variant B — Adapter Continual + Oracle Prototype  
- Let adapter weights update continuously across all domains (normal CTTAOD behavior).
- But **replace μ_te^{k,t} with μ_tr^k** (ground-truth source prototype) in L_obj at every step.
  - Equivalently: set α = 1.0 in the ASRI formula, so μ̃_te = μ_tr always.
  - L_obj uses source prototypes directly, removing any prototype estimation error.
- This isolates the effect of adapter interference: if forgetting still occurs, adapter weight drift is the primary cause.

#### Variant C — Baseline CTTAOD (already from Exp 0)

**Analysis:**

| Result Pattern | Variant A BWT | Variant B BWT | Interpretation | Implication for ASRI |
|---|---|---|---|---|
| Pattern 1 | Low (bad) | ~0 (good) | Prototype drift is the main cause | ASRI direction is correct |
| Pattern 2 | ~0 (good) | Low (bad) | Adapter interference is the main cause | ASRI alone is insufficient, need adapter-level regularization |
| Pattern 3 | Low (bad) | Low (bad) | Both contribute | ASRI helps partially, combine with adapter regularization |

**Output:**
- `eval_matrix_varA.npy`, `eval_matrix_varB.npy`
- BWT, FWT, avg_mAP for each variant
- Comparative bar chart of BWT across variants

---

### Exp 2: Fixed α Sweep (Source Injection Strength)

**Goal:** Confirm that source residual injection actually improves forgetting metrics, and find the BWT-FWT trade-off curve.

**Setup:**
- Use the ASRI residual injection formula with **fixed α** (not adaptive):
  ```
  μ̃_te^{k,t} = (1 - α) · μ_te^{k,t} + α · μ_tr^k
  ```
- **IMPORTANT (Fix from earlier analysis):** Do NOT multiply L_obj by α in the total loss. Keep:
  ```
  L = L_img + L_obj_modified
  ```
  where L_obj_modified uses μ̃_te instead of μ_te. The loss weight λ = 1.0 is fixed.
- Sweep α ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
- α = 0.0 is the baseline CTTAOD (no injection)
- α = 1.0 is equivalent to Variant B from Exp 1

**Output:**
- For each α: BWT, FWT, avg_mAP
- Plot: x-axis = α, y-axis = BWT (blue), FWT (red), avg_mAP (green) — three curves on one plot
- Identify the α* that maximizes avg_mAP (if a clear optimum exists)
- `alpha_sweep_results.json` — all metrics for each α value

**Key question answered:** Is there a sweet spot where BWT improves more than FWT degrades?

---

### Exp 3: Category-Level Forgetting Analysis

**Goal:** Determine if forgetting is concentrated in specific class categories (rare vs. frequent), which would motivate class-specific α_k.

**Setup:**
- Use the per-class evaluation matrices from Exp 0 (baseline CTTAOD).
- Compute **per-class BWT**:
  ```python
  BWT_k = (1 / (T-1)) * sum(a_k[T][i] - a_k[i][i] for i in range(1, T))
  ```
  where a_k[j][i] = AP for class k on domain i after adapting through domains 1..j

**Analysis:**
1. Rank classes by BWT_k. Identify which classes suffer the most forgetting.
2. Correlate BWT_k with:
   - Class frequency in source domain (Cityscapes class distribution)
   - Average confidence of pseudo-labels for class k
   - Number of instances per image for class k
3. If rare classes show significantly worse BWT → this motivates class-specific injection:
   ```
   μ̃_te^{k,t} = (1 - α_k) · μ_te^{k,t} + α_k · μ_tr^k
   ```
   where α_k is higher for rare classes.

**Output:**
- `per_class_bwt.json` — BWT for each class
- Scatter plot: x = class frequency (log scale), y = BWT_k
- Correlation coefficient between frequency and BWT

---

### Exp 4: Comparison with Existing Anti-Forgetting Baselines

**Goal:** Position ASRI (with best fixed α from Exp 2) against established continual learning anti-forgetting methods, applied to CTTAOD.

**Baselines to implement:**

#### Baseline 1: Stochastic Restoration (CoTTA-style)
- At each adaptation step, for each adapter parameter independently:
  - With probability p_restore, reset to source-initialized value
  - With probability (1 - p_restore), keep the adapted value
- Sweep p_restore ∈ {0.001, 0.01, 0.05, 0.1} and report best
- This is a strong baseline because it directly addresses adapter weight drift (Layer 3 forgetting)

#### Baseline 2: EWC-style Adapter Regularization
- After source training, compute Fisher information matrix for adapter parameters:
  ```
  F_i = E[(∂L/∂θ_i)²]    (estimated on source validation set)
  ```
- Add regularization to total loss:
  ```
  L = L_img + L_obj + λ_ewc · Σ_i F_i · (θ_i^t - θ_i^0)²
  ```
- Sweep λ_ewc ∈ {0.1, 1.0, 10.0, 100.0}

#### Baseline 3: Prototype Replay Buffer
- Maintain a small buffer of prototypes from previous domains
- At each step, add a replay loss that prevents current prototypes from deviating too far from buffered ones:
  ```
  L_replay = Σ_{k, prev_domains} ||μ_te^{k,t} - μ_buffer^{k,prev}||²
  ```
- Buffer size: store last 3 domain prototypes

#### Baseline 4: ASRI (Fixed α*)
- Use the best α from Exp 2

#### Baseline 5: ASRI + Stochastic Restoration (Combined)
- Apply both source injection in prototype AND stochastic restoration in adapter weights
- This tests whether addressing both Layer 2 and Layer 3 forgetting simultaneously gives additive benefit

**Output:**
- Table: Method | BWT | FWT | avg_mAP | Compute Overhead (relative to baseline CTTAOD)
- `exp4_comparison.json` — all results

---

## Implementation Notes

### Codebase
- Start from the official CTTAOD/WHW implementation if available, or reimplement based on the paper.
- The base detector is Faster R-CNN with ResNet backbone (likely ResNet-50 or ResNet-18 depending on the paper's setup).
- Adapter modules are lightweight (inserted after backbone blocks). Only adapter parameters are updated during TTA.
- Source statistics (μ_tr, Σ_tr, μ_tr^k, Σ_tr^k) are pre-computed from ~2000 source samples and stored.

### Domain Sequence for Cityscapes-C
Use a fixed corruption order (same as used in the original paper if specified, otherwise alphabetical):
```
gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur,
motion_blur, zoom_blur, snow, frost, fog,
brightness, contrast, elastic_transform, pixelate, jpeg_compression
```
Severity level: 5 (hardest) unless otherwise specified.

### Computational Considerations
- The full evaluation matrix (T+1 × T) requires evaluating the model T times after each domain adaptation → T² total evaluations. For T=15, that's 225 evaluations.
- If this is too expensive, a reasonable approximation: evaluate on (1) current domain, (2) first domain, (3) source domain after each adaptation step. This gives forgetting on domain 1, forward transfer on current domain, and source performance.
- Each evaluation = one pass through the validation set of the respective domain.

### Key Files to Produce
```
results/
├── exp0/
│   ├── eval_matrix_baseline.npy
│   ├── eval_matrix_source_only.npy
│   ├── eval_matrix_baseline_per_class.npy
│   └── metrics_baseline.json          # {BWT, FWT, avg_mAP}
├── exp1/
│   ├── eval_matrix_varA.npy           # adapter reset
│   ├── eval_matrix_varB.npy           # oracle prototype
│   ├── metrics_varA.json
│   ├── metrics_varB.json
│   └── forgetting_diagnosis.png       # bar chart comparison
├── exp2/
│   ├── alpha_sweep_results.json       # metrics for each α
│   ├── alpha_sweep_curves.png         # BWT/FWT/mAP vs α
│   └── eval_matrix_alpha_{val}.npy    # for each α value
├── exp3/
│   ├── per_class_bwt.json
│   ├── class_freq_vs_bwt.png          # scatter plot
│   └── correlation_analysis.json
└── exp4/
    ├── exp4_comparison.json           # all methods compared
    └── exp4_comparison_table.png      # formatted table/chart
```

## Execution Order
1. **Exp 0** first — get baseline working and evaluation protocol validated
2. **Exp 1** immediately after — this determines whether ASRI direction is viable
3. **Exp 2** if Exp 1 shows Pattern 1 or 3 (prototype drift matters)
4. **Exp 3** can run in parallel with Exp 2 (uses Exp 0 data)
5. **Exp 4** last — needs best α from Exp 2

If Exp 1 shows **Pattern 2** (adapter interference dominant, prototype drift negligible):
- Skip Exp 2 as originally designed
- Instead pivot: implement adapter-level source anchoring (L_anchor = ||Θ_adapter^t - Θ_adapter^0||²) and test that
- Then Exp 4 becomes the comparison between adapter-level vs prototype-level vs combined approaches
