"""
Compute Fisher information matrix for adapter parameters on the source
training set, for use as EWC regularization.

Supports COCO and Cityscapes configs — dataset and output path are derived
automatically from the config file name.

Usage (from tools/ directory):
    # COCO
    python compute_fisher.py \
        --config-file ../configs/TTA/COCO_R50.yaml \
        MODEL.WEIGHTS ../models/checkpoints/faster_rcnn_r50_coco.pth \
        TEST.ADAPTATION.SOURCE_FEATS_PATH ../models/stats/COCO_R50_stats.pt

    # Cityscapes
    python compute_fisher.py \
        --config-file ../configs/TTA/Cityscapes_R50.yaml \
        MODEL.WEIGHTS ../models/checkpoints/faster_rcnn_R50_cityscapes.pth \
        TEST.ADAPTATION.SOURCE_FEATS_PATH ../models/stats/Cityscapes_R50_stats.pt

Output:
    models/stats/{Dataset}_{Backbone}_fisher.pt
"""

import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from detectron2.engine import default_argument_parser
from detectron2.modeling.configure_adaptation_model import configure_model

from train_net import Trainer, setup

logger = logging.getLogger(__name__)

# config-file stem → (dataset_name, fisher_filename)
_CONFIG_MAP = {
    "COCO_R50":         ("coco_2017_train",       "COCO_R50_fisher.pt"),
    "COCO_swinT":       ("coco_2017_train",       "COCO_swinT_fisher.pt"),
    "Cityscapes_R50":   ("cityscapes_det_train",  "Cityscapes_R50_fisher.pt"),
}


def _resolve_config(cfg_file):
    stem = os.path.splitext(os.path.basename(cfg_file))[0]
    if stem not in _CONFIG_MAP:
        raise ValueError(
            f"Unknown config '{stem}'. Add it to _CONFIG_MAP in compute_fisher.py."
        )
    return _CONFIG_MAP[stem]


def compute_fisher(cfg, cfg_file):
    dataset_name, fisher_filename = _resolve_config(cfg_file)

    model, optimizer, _ = configure_model(cfg, Trainer, revert=True)
    model.eval()
    model.online_adapt = True

    data_loader = Trainer.build_test_loader(cfg, dataset_name)

    param_list = [p for pg in optimizer.param_groups for p in pg["params"]]
    fisher = [torch.zeros_like(p) for p in param_list]

    n_samples = 0
    max_samples = 500

    logger.info("Computing Fisher on %d batches from '%s'", max_samples, dataset_name)

    for idx, inputs in enumerate(data_loader):
        if idx >= max_samples:
            break
        optimizer.zero_grad()
        outputs, losses, _ = model(inputs)
        total_loss = sum(losses.values())
        if total_loss > 0:
            total_loss.backward()
        for i, p in enumerate(param_list):
            if p.grad is not None:
                fisher[i] += p.grad.detach() ** 2
        n_samples += len(inputs)
        if idx % 50 == 0:
            logger.info("  %d / %d batches processed", idx, max_samples)

    fisher = [f / max(n_samples, 1) for f in fisher]

    out_path = os.path.join("../models/stats", fisher_filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(fisher, out_path)
    logger.info("Fisher saved to %s", out_path)
    return fisher


if __name__ == "__main__":
    # Cityscapes dataset is not auto-registered by detectron2; register only
    # when the config actually needs it to avoid duplicate-registration errors.
    args = default_argument_parser().parse_args()
    cfg_file = args.config_file

    stem = os.path.splitext(os.path.basename(cfg_file))[0]
    if "Cityscapes" in stem:
        _root = os.getenv("DETECTRON2_DATASETS", "datasets")
        from cityscapes_dataset import register_cityscapes_det
        register_cityscapes_det(_root)

    cfg = setup(args)
    compute_fisher(cfg, cfg_file)