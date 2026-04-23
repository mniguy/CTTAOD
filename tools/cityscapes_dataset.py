"""
Cityscapes detection dataset registration for ASRI experiments.

Supports:
  - cityscapes_det_train  (source domain, clean)
  - cityscapes_det_val    (target domain, clean)
  - cityscapes_det_val-{corruption}  (Cityscapes-C, on-the-fly)

Corruption is applied at read time by CityscapesCorruptedMapper.
Use register_cityscapes_det(root) at startup.
"""

import json
import logging
import os
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

logger = logging.getLogger(__name__)

# 8 Cityscapes detection classes (same as Foggy Cityscapes domain adaptation papers)
CITYSCAPES_DET_CLASSES = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# Fixed corruption sequence from EXPERIMENTS.md
CITYSCAPES_C_SEQUENCE = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def _get_cityscapes_det_files(image_dir, gt_dir):
    """Scan Cityscapes directory structure and return paired (image_path, gt_json_path)."""
    pairs = []
    cities = sorted(os.listdir(image_dir))
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        if not os.path.isdir(city_img_dir):
            continue
        for basename in sorted(os.listdir(city_img_dir)):
            if not basename.endswith("_leftImg8bit.png"):
                continue
            stem = basename[: -len("_leftImg8bit.png")]
            img_path = os.path.join(city_img_dir, basename)
            gt_json = os.path.join(city_gt_dir, stem + "_gtFine_polygons.json")
            if os.path.isfile(gt_json):
                pairs.append((img_path, gt_json))
    return pairs


def _polygon_to_bbox(polygon_pts):
    """Convert a flat list of [x,y,x,y,...] polygon points to [xmin,ymin,xmax,ymax]."""
    xs = polygon_pts[0::2]
    ys = polygon_pts[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def load_cityscapes_det(image_dir, gt_dir, corrupt=None, corrupt_severity=5):
    """
    Load Cityscapes detection annotations into Detectron2 standard format.

    Only the 8 detection classes defined in CITYSCAPES_DET_CLASSES are kept.
    Bounding boxes are derived from the polygon annotations in gtFine_polygons.json.

    Args:
        image_dir: path to leftImg8bit/{split}
        gt_dir: path to gtFine/{split}
        corrupt (str|None): imagecorruptions corruption name; if set, each dict
            will carry a ``corrupt`` and ``corrupt_severity`` field that
            DatasetMapper reads to apply corruption on-the-fly.
        corrupt_severity (int): corruption severity level (1-5, default 5).

    Returns:
        list[dict]: Detectron2 dataset dicts.
    """
    class_set = set(CITYSCAPES_DET_CLASSES)
    class_to_id = {c: i for i, c in enumerate(CITYSCAPES_DET_CLASSES)}

    pairs = _get_cityscapes_det_files(image_dir, gt_dir)
    if len(pairs) == 0:
        raise FileNotFoundError(
            f"No Cityscapes images found under {image_dir}. "
            "Check that DETECTRON2_DATASETS points to the right location."
        )

    dataset_dicts = []
    for img_path, gt_json_path in pairs:
        with open(gt_json_path, "r") as f:
            gt = json.load(f)

        record = {
            "file_name": img_path,
            "image_id": os.path.splitext(os.path.basename(img_path))[0],
            "height": gt["imgHeight"],
            "width": gt["imgWidth"],
        }
        if corrupt is not None:
            record["corrupt"] = corrupt
            record["corrupt_severity"] = corrupt_severity

        annotations = []
        for obj in gt.get("objects", []):
            label = obj["label"]
            if label not in class_set:
                continue
            poly = obj["polygon"]  # list of [x, y] points
            flat = [coord for pt in poly for coord in pt]
            if len(flat) < 6:  # need at least 3 points
                continue
            bbox = _polygon_to_bbox(flat[0::2] + flat[1::2])
            # Clamp to image bounds
            bbox[0] = max(0.0, bbox[0])
            bbox[1] = max(0.0, bbox[1])
            bbox[2] = min(float(gt["imgWidth"]), bbox[2])
            bbox[3] = min(float(gt["imgHeight"]), bbox[3])
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            annotations.append({
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_to_id[label],
            })

        record["annotations"] = annotations
        dataset_dicts.append(record)

    logger.info(
        "Loaded %d Cityscapes detection images from %s", len(dataset_dicts), image_dir
    )
    return dataset_dicts


def register_cityscapes_det(root):
    """
    Register Cityscapes detection datasets under `root`.

    Expects the structure:
        root/cityscapes/leftImg8bit/{train,val}/
        root/cityscapes/gtFine/{train,val}/

    Registers:
        cityscapes_det_train   — clean training split
        cityscapes_det_val     — clean validation split
        cityscapes_det_val-{c} — corruption c applied on-the-fly (see CITYSCAPES_C_SEQUENCE)
    """
    cs_root = os.path.join(root, "cityscapes")
    splits = [
        ("cityscapes_det_train", "train"),
        ("cityscapes_det_val", "val"),
    ]

    for name, split in splits:
        img_dir = os.path.join(cs_root, "leftImg8bit", split)
        gt_dir = os.path.join(cs_root, "gtFine", split)
        DatasetCatalog.register(
            name,
            lambda img=img_dir, gt=gt_dir: load_cityscapes_det(img, gt),
        )
        MetadataCatalog.get(name).set(
            thing_classes=list(CITYSCAPES_DET_CLASSES),
            evaluator_type="coco",
            image_dir=img_dir,
            gt_dir=gt_dir,
            corrupt=None,
        )

    # Register corrupted variants (corruption applied on-the-fly in DatasetMapper)
    val_img_dir = os.path.join(cs_root, "leftImg8bit", "val")
    val_gt_dir = os.path.join(cs_root, "gtFine", "val")
    for corrupt_name in CITYSCAPES_C_SEQUENCE:
        cname = f"cityscapes_det_val-{corrupt_name}"
        DatasetCatalog.register(
            cname,
            lambda img=val_img_dir, gt=val_gt_dir, c=corrupt_name: load_cityscapes_det(
                img, gt, corrupt=c, corrupt_severity=5
            ),
        )
        MetadataCatalog.get(cname).set(
            thing_classes=list(CITYSCAPES_DET_CLASSES),
            evaluator_type="coco",
            image_dir=val_img_dir,
            gt_dir=val_gt_dir,
            corrupt=corrupt_name,
            corrupt_severity=5,
        )