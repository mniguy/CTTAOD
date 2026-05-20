"""
Compute MAS (Memory Aware Synapses) importance for adapter parameters.

MAS importance is derived from the sensitivity of the model's *output* to each
parameter, computed without labels. For each batch we take the squared L2 norm
of the backbone FPN features and accumulate |∂||f(x)||² / ∂θ_i|.

Output is saved in the same format as Fisher (list[Tensor], one entry per
trainable adapter param) so it can be loaded directly through
TEST.ADAPTATION.EWC_FISHER_PATH together with EWC_IMPORTANCE_TYPE="mas".

Usage (from tools/ directory):
    python compute_mas.py \
        --config-file ../configs/TTA/COCO_R50.yaml \
        MODEL.WEIGHTS ../models/checkpoints/faster_rcnn_r50_coco.pth \
        TEST.ADAPTATION.SOURCE_FEATS_PATH ../models/stats/COCO_R50_stats.pt \
        TEST.ADAPTATION.WHERE adapter

Output:
    models/stats/{Dataset}_{Backbone}_mas.pt
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

_CONFIG_MAP = {
    "COCO_R50":         ("coco_2017_train",       "COCO_R50_mas.pt"),
    "COCO_swinT":       ("coco_2017_train",       "COCO_swinT_mas.pt"),
    "Cityscapes_R50":   ("cityscapes_det_train",  "Cityscapes_R50_mas.pt"),
}


def _resolve_config(cfg_file):
    stem = os.path.splitext(os.path.basename(cfg_file))[0]
    if stem not in _CONFIG_MAP:
        raise ValueError(
            f"Unknown config '{stem}'. Add it to _CONFIG_MAP in compute_mas.py."
        )
    return _CONFIG_MAP[stem]


def compute_mas(cfg, cfg_file):
    dataset_name, mas_filename = _resolve_config(cfg_file)

    model, optimizer, _ = configure_model(cfg, Trainer, revert=True)
    model.eval()
    # MAS doesn't need alignment losses — disable adapt path so forward returns
    # a normal feature dict via the backbone call below.

    data_loader = Trainer.build_test_loader(cfg, dataset_name)

    param_list = [p for pg in optimizer.param_groups for p in pg["params"]]
    mas = [torch.zeros_like(p) for p in param_list]

    n_samples = 0
    max_samples = 500

    logger.info("Computing MAS on %d batches from '%s'", max_samples, dataset_name)

    for idx, inputs in enumerate(data_loader):
        if idx >= max_samples:
            break
        optimizer.zero_grad()

        # Forward through backbone only — MAS importance is gradient of the
        # output feature L2 norm w.r.t. parameters.
        images = model.preprocess_image(inputs)
        backbone_out = model.backbone(images.tensor)
        # FPN returns (feature_dict, bottom_up_features); plain ResNet returns a dict.
        features = backbone_out[0] if isinstance(backbone_out, tuple) else backbone_out

        # Squared L2 norm averaged across FPN levels and spatial positions.
        sq_norm = 0.0
        n_lv = 0
        for f in features.values():
            sq_norm = sq_norm + f.pow(2).mean()
            n_lv += 1
        if n_lv > 0:
            sq_norm = sq_norm / n_lv

        sq_norm.backward()
        for i, p in enumerate(param_list):
            if p.grad is not None:
                mas[i] += p.grad.detach().abs()
        n_samples += len(inputs)
        if idx % 50 == 0:
            logger.info("  %d / %d batches processed", idx, max_samples)

    mas = [f / max(n_samples, 1) for f in mas]

    out_path = os.path.join("../models/stats", mas_filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(mas, out_path)
    logger.info("MAS saved to %s", out_path)
    return mas


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg_file = args.config_file

    stem = os.path.splitext(os.path.basename(cfg_file))[0]
    if "Cityscapes" in stem:
        _root = os.getenv("DETECTRON2_DATASETS", "datasets")
        from cityscapes_dataset import register_cityscapes_det
        register_cityscapes_det(_root)

    cfg = setup(args)
    compute_mas(cfg, cfg_file)
