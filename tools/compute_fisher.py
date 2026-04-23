"""
Compute Fisher information matrix for adapter parameters on the Cityscapes
source validation set, for use as EWC regularization in Exp 4 Baseline 2.

Usage (from tools/ directory):
    python compute_fisher.py \
        --config-file ../configs/TTA/Cityscapes_R50.yaml \
        MODEL.WEIGHTS ../models/checkpoints/faster_rcnn_R50_cityscapes.pth \
        TEST.ADAPTATION.SOURCE_FEATS_PATH ../models/stats/Cityscapes_R50_stats.pt

Output:
    models/stats/Cityscapes_R50_fisher.pt   — list of Fisher tensors (one per
                                               adapter param in optimizer order)
"""

import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.modeling import add_swint_config
from detectron2.modeling.configure_adaptation_model import configure_model

from train_net import Trainer, setup
from cityscapes_dataset import register_cityscapes_det

logger = logging.getLogger(__name__)


def compute_fisher(cfg):
    """Return a list of Fisher tensors (one per adapter param group param)."""
    model, optimizer, _ = configure_model(cfg, Trainer, revert=True)
    model.eval()
    model.online_adapt = True  # run adapt() forward to get losses

    dataset_name = "cityscapes_det_train"
    data_loader = Trainer.build_test_loader(cfg, dataset_name)

    # Accumulate squared gradients (diagonal Fisher approximation)
    param_list = [p for pg in optimizer.param_groups for p in pg['params']]
    fisher = [torch.zeros_like(p) for p in param_list]

    n_samples = 0
    max_samples = 500   # subsample source dataset

    logger.info("Computing Fisher on %d batches from %s", max_samples, dataset_name)

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

    # Normalize by number of samples
    fisher = [f / max(n_samples, 1) for f in fisher]

    out_path = os.path.join("../models/stats", "Cityscapes_R50_fisher.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(fisher, out_path)
    logger.info("Fisher saved to %s", out_path)
    return fisher


if __name__ == "__main__":
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_cityscapes_det(_root)

    args = default_argument_parser().parse_args()
    cfg = setup(args)
    compute_fisher(cfg)