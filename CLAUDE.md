# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CTTAOD implements "What, How, and When Should Object Detectors Update in Continually Changing Test Domains?" (CVPR 2024). It is a test-time adaptation (TTA) framework for object detection built on top of a modified Detectron2 (v0.6). The core research questions are: *what* features to adapt, *how* to adapt them, and *when* to perform adaptation — across continually shifting domains like weather or corruption changes.

## Environment Setup

```bash
conda create -n cta_od python=3.10
conda activate cta_od
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.8 -c pytorch
pip install -r requirements.txt
pip install -e .   # installs detectron2 with C++ extensions in dev mode
```

The conda environment spec is also in `tta_jayeon.yaml`.

## Running Experiments

All experiments go through `tools/train_net.py`. The `--eval-only` flag triggers test-time adaptation (no training).

**Full experiment scripts** (recommended entry points):
```bash
bash scripts/coco_adapt_R50.sh        # COCO + ResNet-50 (collects stats, runs all ablations)
bash scripts/coco_adapt_swinT.sh      # COCO + Swin Transformer
bash scripts/shift_discrete_adapt.sh  # SHIFT dataset, discrete domain shifts
bash scripts/shift_continuous_adapt.sh # SHIFT dataset, continuous shifts
```

**Direct command pattern:**
```bash
cd tools/
python train_net.py \
  --config-file ../configs/TTA/COCO_R50.yaml \
  --eval-only \
  --num-gpus 1 \
  TEST.ADAPTATION.WHERE adapter \
  TEST.ONLINE_ADAPTATION True \
  OUTPUT_DIR ./outputs/my_run
```

Key CLI flags:
- `--eval-only`: Run TTA evaluation (not training)
- `--wandb`: Log metrics to Weights & Biases
- `--resume`: Resume from checkpoint
- `--num-gpus N`: Multi-GPU distributed inference
- Any `KEY VALUE` pairs after flags override config values

## Configuration System

Configs use YACS with inheritance (`_BASE_`):

```
configs/TTA/COCO_R50.yaml
  → configs/Base/COCO_faster_rcnn_R50_FPN_1x.yaml
    → configs/Base/Base-RCNN-FPN.yaml
```

Critical `TEST.ADAPTATION` keys (defined in `detectron2/config/defaults.py`):

| Key | Values | Meaning |
|-----|--------|---------|
| `WHERE` | `full`, `adapter`, `normalization`, `head` | Which parameters to adapt |
| `TYPE` | `mean-teacher`, etc. | Adaptation strategy |
| `CONTINUAL` | bool | Continual vs. episodic adaptation |
| `SOURCE_FEATS_PATH` | path | Pre-collected source feature statistics |
| `GLOBAL_ALIGN` | `KL`, `Wasserstein` | Global feature alignment loss |
| `FOREGROUND_ALIGN` | `KL`, `Wasserstein` | Foreground (object) alignment loss |
| `SKIP_REDUNDANT` | `stat-period-ema`, `None` | Redundancy skipping strategy |
| `SKIP_THRESHOLD` | float | Similarity threshold to skip update |

## Architecture

### Modified Detectron2

The `detectron2/` directory is a **modified fork** (not vanilla Detectron2). Key additions:

- **`detectron2/modeling/meta_arch/rcnn.py`** — `GeneralizedRCNN` extended with:
  - Feature collection mode (`collect_features`, `s_stats`, `t_stats`)
  - Online adaptation mode (`online_adapt`)
  - Global and foreground KL/Wasserstein alignment losses
- **`detectron2/modeling/configure_adaptation_model.py`** — `configure_model()` selects which parameters are trainable based on `WHERE` and sets up optimizer (SGD or AdamW with gradient clipping)
- **`detectron2/engine/defaults.py`** — Three test pipelines added to `DefaultTrainer`:
  - `test_continual_domain()` — COCO/KITTI: iterates over corruption types
  - `test_continual_domain_shift_discrete()` — SHIFT: fixed domain shifts
  - `test_continual_domain_shift_continuous()` — SHIFT: gradual shift trajectories

### Adapter Modules

Lightweight bottleneck adapters are inserted into ResNet residual blocks (`detectron2/modeling/backbone/resnet.py`) and Swin Transformer blocks (`detectron2/modeling/backbone/swin.py`). Types: `parallel`, `lora`, `scale_shift`. Configured via `TEST.ADAPTER.*`.

### Two-Phase Workflow

1. **Feature collection** (`TEST.COLLECT_FEATURES: True`): Run the source-domain model once to gather FPN feature statistics (global + foreground) → saved to `models/stats/*.pt`
2. **Adaptation** (`TEST.ONLINE_ADAPTATION: True`): Load source stats and minimize alignment loss against target features at test time, optionally skipping redundant updates

### Data

- **COCO + corruptions**: `imagecorruptions` library generates 15 corruption types (shot_noise, pixelate, jpeg_compression, etc.) at test time
- **SHIFT dataset**: Registered via `tools/shift_dataset.py`; discrete and continuous splits with 6 classes (pedestrian, car, truck, bus, motorcycle, bicycle)
- **KITTI**: Autonomous driving with weather variants

Datasets are registered via Detectron2's `DatasetCatalog`/`MetadataCatalog`. Source feature stats go in `models/stats/`, model checkpoints in `models/checkpoints/`.

## Paper Reference

Yoo et al., "What How and When Should Object Detectors Update in Continually Changing Test Domains?", CVPR 2024. https://arxiv.org/abs/2312.08875
