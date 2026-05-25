# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like, FrozenBatchNorm2d, LayerNorm
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        online_adapt: bool = False,
        alpha_gl: float = 1.0,
        alpha_fg: float = 1.0,
        gl_align: str = None,
        fg_align: str = None,
        collect_features: bool = False,
        collect_iou_thr: float = 0.5,
        source_feat_stats=None,
        ema_gamma: int = 128,
        ema_beta: float = 0.0,
        dpema_apply_gl: bool = True,
        freq_weight: bool = False,
        skip_tau: float = 1.0,
        asri_alpha: float = 0.0,
        oracle_prototype: bool = False,
        swema_k: int = 0,
        swema_alpha: float = 0.1,
        asri_confgate: bool = False,
        asri_confgate_lambda: float = 1.0,
        asri_confgate_min: float = 0.0,
        asri_gl: bool = False,
        conf_proto: bool = False,
        conf_proto_mode: str = "soft",
        conf_proto_threshold: float = 0.7,
        conf_proto_gamma: float = 1.0,
        cb_proto: bool = False,
        cb_proto_max_per_class: int = 8,
        cb_proto_inv_freq: bool = False,
        # Exp 12: Sol A/B/C from ContinualTTA_ObjectDetection
        proto_method: str = "baseline",
        switch_cosim_thr: float = 0.30,
        source_anchor_alpha: float = 0.30,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.num_classes = self.roi_heads.num_classes
        # collect features and export stats
        self.collect_features = collect_features
        if self.collect_features:
            self.collect_iou_thr = collect_iou_thr
            self.gl_features = {}
            self.fg_features = {}
            self.iou_with_gt = {}

        # exploit source features statistics
        self.online_adapt = online_adapt
        self.alpha_gl = alpha_gl
        self.alpha_fg = alpha_fg
        self.s_stats = torch.load(source_feat_stats) if source_feat_stats is not None else None
        self.t_stats = {}
        self.template_cov = {}
        self.s_div = {}
        self.gl_align = gl_align
        if self.gl_align:
            self.t_stats["gl"] = {}
            self.template_cov["gl"] = {}
        self.fg_align = fg_align
        if self.fg_align:
            self.t_stats["fg"] = {}
            self.template_cov["fg"] = {}
            self.ema_n = {}
        self.ema_gamma = ema_gamma
        self.ema_beta = ema_beta
        self.dpema_apply_gl = dpema_apply_gl
        self.freq_weight = freq_weight
        self.skip_tau = skip_tau
        # ASRI: Adaptive Source Residual Injection
        # μ̃_te^{k,t} = (1-asri_alpha)*μ_te^{k,t} + asri_alpha*μ_tr^k
        self.asri_alpha = asri_alpha
        self.oracle_prototype = oracle_prototype  # Exp 1 Variant B: force alpha=1
        # Sliding Window EMA: μ̃ = (1-swema_alpha)*μ_te_recent + swema_alpha*μ_tr
        # swema_k=0 disables SWEMA and falls back to DPEMA
        self.swema_k = swema_k
        self.swema_alpha = swema_alpha
        # per-class sliding window buffer: {k: deque of (sum, count) tensors}
        self._swema_buf = {}
        # Exp 8: confidence-gated ASRI
        self.asri_confgate = asri_confgate
        self.asri_confgate_lambda = asri_confgate_lambda
        self.asri_confgate_min = asri_confgate_min
        # Exp 9: ASRI for global branch
        self.asri_gl = asri_gl
        # Exp 10: confidence-weighted prototype update
        self.conf_proto = conf_proto
        self.conf_proto_mode = conf_proto_mode
        self.conf_proto_threshold = conf_proto_threshold
        self.conf_proto_gamma = conf_proto_gamma
        # Exp 10: class-balanced subsampling
        self.cb_proto = cb_proto
        self.cb_proto_max_per_class = cb_proto_max_per_class
        self.cb_proto_inv_freq = cb_proto_inv_freq
        # Exp 12: Sol A/B/C
        assert proto_method in ("baseline", "reset", "dual_memory", "reset_dual_memory", "adaptive_gamma"), \
            f"Unknown proto_method: {proto_method}"
        self.proto_method = proto_method
        self.switch_cosim_thr = switch_cosim_thr
        self.source_anchor_alpha = source_anchor_alpha
        self.prev_cosine_sim = {}   # {k: previous step's cosine_sim_before}
        self.s_proto_anchor = {}    # {k: frozen source prototype}
        self._last_drift_stats = {} # Exp 14: populated by adapt() for JSONL logging

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "online_adapt": cfg.TEST.ONLINE_ADAPTATION,
            "alpha_gl": cfg.TEST.ADAPTATION.ALPHA_GLOBAL,
            "alpha_fg": cfg.TEST.ADAPTATION.ALPHA_FOREGROUND,
            "gl_align": cfg.TEST.ADAPTATION.GLOBAL_ALIGN,
            "fg_align": cfg.TEST.ADAPTATION.FOREGROUND_ALIGN,
            "collect_features": cfg.TEST.COLLECT_FEATURES,
            "collect_iou_thr": cfg.TEST.COLLECT_IOU_THR,
            "source_feat_stats": cfg.TEST.ADAPTATION.SOURCE_FEATS_PATH,
            "ema_gamma": cfg.TEST.ADAPTATION.EMA_GAMMA,
            "ema_beta": cfg.TEST.ADAPTATION.EMA_BETA,
            "dpema_apply_gl": cfg.TEST.ADAPTATION.DPEMA_APPLY_GL,
            "freq_weight": cfg.TEST.ADAPTATION.FREQ_WEIGHT,
            "skip_tau": cfg.TEST.ADAPTATION.SKIP_TAU,
            "asri_alpha": cfg.TEST.ADAPTATION.ASRI_ALPHA,
            "oracle_prototype": cfg.TEST.ADAPTATION.ORACLE_PROTOTYPE,
            "swema_k": cfg.TEST.ADAPTATION.SWEMA_K,
            "swema_alpha": cfg.TEST.ADAPTATION.SWEMA_ALPHA,
            "asri_confgate": cfg.TEST.ADAPTATION.ASRI_CONFGATE,
            "asri_confgate_lambda": cfg.TEST.ADAPTATION.ASRI_CONFGATE_LAMBDA,
            "asri_confgate_min": cfg.TEST.ADAPTATION.ASRI_CONFGATE_MIN,
            "asri_gl": cfg.TEST.ADAPTATION.ASRI_GL,
            "conf_proto": cfg.TEST.ADAPTATION.CONF_PROTO,
            "conf_proto_mode": cfg.TEST.ADAPTATION.CONF_PROTO_MODE,
            "conf_proto_threshold": cfg.TEST.ADAPTATION.CONF_PROTO_THRESHOLD,
            "conf_proto_gamma": cfg.TEST.ADAPTATION.CONF_PROTO_GAMMA,
            "cb_proto": cfg.TEST.ADAPTATION.CB_PROTO,
            "cb_proto_max_per_class": cfg.TEST.ADAPTATION.CB_PROTO_MAX_PER_CLASS,
            "cb_proto_inv_freq": cfg.TEST.ADAPTATION.CB_PROTO_INV_FREQ,
            "proto_method":        getattr(cfg.TEST.ADAPTATION, "PROTO_METHOD",         "baseline"),
            "switch_cosim_thr":    getattr(cfg.TEST.ADAPTATION, "SWITCH_COSIM_THR",     0.30),
            "source_anchor_alpha": getattr(cfg.TEST.ADAPTATION, "SOURCE_ANCHOR_ALPHA",  0.30),
        }

    def initialize(self):
        self.gl_features = {}
        self.fg_features = {}
        # initialize target feature stats
        if self.gl_align is not None and self.gl_align == "KL":
            for k in self.s_stats["gl"]:
                mean, cov = self.s_stats["gl"][k]
                self.template_cov["gl"][k] = torch.eye(mean.shape[0]) * cov.max().item() / 30
                self.t_stats["gl"][k] = (mean, cov)
        if self.gl_align is not None and self.gl_align == "bn_stats":
            self.bn_features = self.s_stats["bn_stats"]
        if self.fg_align is not None and self.fg_align == "KL":
            for k in self.s_stats["fg"]:
                mean, cov = self.s_stats["fg"][k]
                self.template_cov["fg"][k] = torch.eye(mean.shape[0]) * cov.max().item() / 30
                self.t_stats["fg"][k] = (mean, cov)
                self.ema_n[k] = 0
                # Exp 12/16 Sol A/B: freeze source prototype as anchor for reset/blend
                if self.proto_method in ("reset", "dual_memory", "reset_dual_memory"):
                    self.s_proto_anchor[k] = mean.clone()
        self.s_div = self.s_stats["kl_div"] if self.s_stats is not None and "kl_div" in self.s_stats else None
        # reset sliding window buffers at domain boundary
        self._swema_buf = {}
        # Exp 12 Sol A: reset previous-step cosine sim cache at domain boundary
        self.prev_cosine_sim = {}
        self._last_drift_stats = {}

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.online_adapt:
            return self.adapt(batched_inputs)
        elif self.collect_features:
            return self.collect_feats(batched_inputs)
        elif not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        if isinstance(features, tuple):
            features = features[0]

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def adapt(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert self.online_adapt

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if isinstance(features, tuple):
            features = features[0]

        adapt_loss = {}
        feature_sim = {}
        drift_proto_cos_source = []
        drift_proto_cos_batch = []
        drift_proto_drop = []
        drift_proto_classes = []
        drift_fg_counts = []
        drift_fg_scores = []
        drift_reset_classes = []
        self.roi_heads.training = False
        self.proposal_generator.training = False
        # if "instances" in batched_inputs[0]:
        #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        #     proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # else:
        #     gt_instances = None
        proposals, _ = self.proposal_generator(images, features, None, eval=True)

        # proposals = self.roi_heads.label_and_sample_proposals(proposals, gt_instances)
        pred_instances, predictions, box_features = self.roi_heads._forward_box(features, proposals, outs=True)
        results = self.roi_heads.forward_with_given_boxes(features, pred_instances)
        # detector_losses = self.roi_heads.box_predictor.losses(predictions, proposals)
        # adapt_loss.update(detector_losses)
        if self.fg_align is not None:
            _scores = nn.Softmax(dim=1)(predictions[0])
            bg_scores = _scores[:, -1]
            fg_scores, fg_preds = _scores[:, :-1].max(dim=1)

            valid = fg_scores >= 0.5
            # self.class_th_adapt = torch.where(self.class_th_adapt < self.max_conf, self.class_th_adapt,
            #                                   torch.Tensor([self.max_conf)])
            # Clone before in-place modification: max() saves fg_preds/fg_scores in the
            # autograd graph; modifying them in-place bumps their version and crashes backward.
            fg_preds = fg_preds.clone()
            fg_scores = fg_scores.clone()
            fg_preds[~valid] = self.num_classes
            fg_scores[~valid] = bg_scores[~valid]
            loss_fg_align = 0
            loss_n = 0
            for _k in fg_preds[fg_preds != self.num_classes].unique():
                k = _k.item()
                if (fg_preds == k).sum() > 0:
                    mask_k = (fg_preds == k)
                    cur_feats = box_features[mask_k]
                    cur_scores = fg_scores[mask_k]
                    drift_proto_classes.append(k)
                    drift_fg_counts.append(int(mask_k.sum().item()))
                    drift_fg_scores.append(float(cur_scores.mean().detach().cpu()))
                    #feature_sim['fg-{}'.format(str(k))] = F.cosine_similarity(self.t_stats["fg"][k][0].reshape(1, -1),
                    #                    cur_feats.mean(dim=0).reshape(1, -1)).item()

                    # ── Exp 10: Class-Balanced Subsampling ────────────────────
                    # Cap per-class samples used for the prototype update. The
                    # full set is still counted in `ema_n` to track true class
                    # frequency, but only `M_max` features feed the EMA mean.
                    n_full = cur_feats.shape[0]
                    if self.cb_proto and cur_feats.shape[0] > self.cb_proto_max_per_class:
                        # rank by confidence (descending) and take top-M; if
                        # scores are all 1 (e.g. no fg_align softmax) this
                        # degenerates to a deterministic head-of-tensor cut.
                        topk = torch.topk(cur_scores, self.cb_proto_max_per_class).indices
                        cur_feats = cur_feats[topk]
                        cur_scores = cur_scores[topk]

                    # ── Exp 10: Confidence-Weighted Prototype Update ──────────
                    # Build per-sample weights w_i ∈ [0, 1]. The weighted mean
                    # μ̂ = Σ w_i f_i / Σ w_i replaces the uniform mean.
                    if self.conf_proto:
                        if self.conf_proto_mode == "hard":
                            w = (cur_scores >= self.conf_proto_threshold).float()
                        else:  # "soft"
                            w = cur_scores.clamp(min=0.0) ** self.conf_proto_gamma
                        w_sum = w.sum()
                        if w_sum.item() < 1e-6:
                            # all samples gated out — skip this class for this batch
                            continue
                        weighted_feats = cur_feats * w.unsqueeze(1)
                        cur_feats_mean = weighted_feats.sum(dim=0) / w_sum
                        cur_feats_sum  = weighted_feats.sum(dim=0) * (cur_feats.shape[0] / w_sum)
                    else:
                        cur_feats_mean = cur_feats.mean(dim=0)
                        cur_feats_sum  = cur_feats.sum(dim=0)

                    self.ema_n[k] += n_full
                    N = cur_feats.shape[0]

                    # Optional inverse-frequency rescaling of effective N: rare
                    # classes get a larger EMA step. Implemented by inflating N
                    # (and thus `effective_beta`/SWEMA weight) using a running
                    # frequency estimate.
                    if self.cb_proto and self.cb_proto_inv_freq:
                        max_n = max(self.ema_n.values()) if len(self.ema_n) else 1
                        inv_freq = float(max_n) / max(self.ema_n[k], 1)
                        N = max(1, int(round(N * inv_freq)))

                    proto_vec_before = self.t_stats["fg"][k][0].to(self.device)
                    src_proto_for_log = self.s_stats["fg"][k][0].to(self.device) if self.s_stats is not None else None
                    batch_mean_for_log = cur_feats.mean(dim=0)
                    prev_cos_for_log = self.prev_cosine_sim.get(k, None)

                    # ── Prototype update ──────────────────────────────────────
                    if self.proto_method != "baseline":
                        # ── Exp 12: Sol A/B/C (ported from ContinualTTA_ObjectDetection) ──
                        proto_vec = self.t_stats["fg"][k][0].to(self.device)
                        batch_mean = cur_feats.mean(dim=0)
                        cosine_sim_before = F.cosine_similarity(
                            proto_vec.reshape(1, -1),
                            batch_mean.reshape(1, -1),
                        ).item()

                        if self.proto_method in ("reset", "reset_dual_memory"):
                            # Sol-A: cosine-sim *drop* between consecutive steps triggers reset to source.
                            prev_cs = self.prev_cosine_sim.get(k, cosine_sim_before)
                            drop = prev_cs - cosine_sim_before
                            self.prev_cosine_sim[k] = cosine_sim_before
                            if drop > self.switch_cosim_thr and k in self.s_proto_anchor:
                                raw_updated_mean = self.s_proto_anchor[k].to(self.device).clone()
                                drift_reset_classes.append(k)
                            else:
                                diff = cur_feats - proto_vec[None, :]
                                delta = (1.0 / self.ema_gamma) * diff.sum(dim=0)
                                raw_updated_mean = proto_vec + delta

                        elif self.proto_method == "dual_memory":
                            # Sol-B: EMA prototype unchanged; blend applied at loss-time below.
                            diff = cur_feats - proto_vec[None, :]
                            delta = (1.0 / self.ema_gamma) * diff.sum(dim=0)
                            raw_updated_mean = proto_vec + delta

                        else:  # adaptive_gamma
                            # Sol-C: mean-based EMA with cosine-scaled gamma, weight clamped to [0,1].
                            cs_clipped = max(cosine_sim_before, 0.0)
                            adaptive_g = max(self.ema_gamma * cs_clipped, 4.0)
                            weight = min(N / adaptive_g, 1.0)
                            raw_updated_mean = (1.0 - weight) * proto_vec \
                                               + weight * cur_feats.mean(dim=0)
                    elif self.swema_k > 0:
                        # SWEMA: sliding window EMA over recent K images
                        # buffer stores (feature_sum, count) per step; window = K steps
                        if k not in self._swema_buf:
                            self._swema_buf[k] = deque()
                        self._swema_buf[k].append((cur_feats_sum.detach(), N))
                        # evict steps that exceed the window
                        total_in_window = sum(cnt for _, cnt in self._swema_buf[k])
                        while total_in_window > self.swema_k and len(self._swema_buf[k]) > 1:
                            _, oldest_cnt = self._swema_buf[k].popleft()
                            total_in_window -= oldest_cnt
                        window_sum = sum(s for s, _ in self._swema_buf[k])
                        window_cnt = sum(c for _, c in self._swema_buf[k])
                        mu_recent = window_sum / window_cnt
                        # μ̃ = (1 - swema_alpha) * μ_te_recent + swema_alpha * μ_tr
                        src_mean = self.s_stats["fg"][k][0].to(self.device)
                        raw_updated_mean = ((1.0 - self.swema_alpha) * mu_recent
                                            + self.swema_alpha * src_mean)
                        # Straight-through gradient: buffer stores detached values to avoid
                        # retaining old computation graphs across iterations, so gradient
                        # path through current features is re-attached here (value unchanged).
                        raw_updated_mean = raw_updated_mean + (
                            cur_feats_mean - cur_feats_mean.detach()
                        )
                    elif self.ema_beta > 0.0:
                        # DPEMA: exponential decay over full history
                        effective_beta = self.ema_beta ** N
                        raw_updated_mean = (effective_beta * self.t_stats["fg"][k][0].to(self.device)
                                            + (1.0 - effective_beta) * cur_feats_mean)
                    else:
                        # legacy gamma method (backward compat)
                        # diff/delta computed against the (possibly conf-weighted) mean.
                        delta = (cur_feats_mean - self.t_stats["fg"][k][0].to(self.device)) \
                                * (N / self.ema_gamma)
                        raw_updated_mean = self.t_stats["fg"][k][0].to(self.device) + delta

                    if src_proto_for_log is not None:
                        drift_proto_cos_source.append(float(F.cosine_similarity(
                            raw_updated_mean.detach().reshape(1, -1),
                            src_proto_for_log.detach().reshape(1, -1),
                        ).detach().cpu()))
                    drift_proto_cos_batch.append(float(F.cosine_similarity(
                        proto_vec_before.detach().reshape(1, -1),
                        batch_mean_for_log.detach().reshape(1, -1),
                    ).detach().cpu()))
                    if prev_cos_for_log is not None:
                        drift_proto_drop.append(float(prev_cos_for_log - drift_proto_cos_batch[-1]))

                    # store raw prototype — ASRI not reflected in t_stats
                    self.t_stats["fg"][k] = (raw_updated_mean.detach(), None)

                    # ── ASRI: applied at loss-time only, not stored ────────────
                    # μ̃_te^{k,t} = (1-α)*μ_te^{k,t} + α*μ_tr^k
                    if self.oracle_prototype:
                        _alpha = 1.0
                    elif self.swema_k > 0:
                        _alpha = 0.0  # SWEMA already anchors to source
                    elif self.asri_confgate and self.asri_alpha > 0.0:
                        # Exp 8: gate α by per-class detection count × avg confidence.
                        # Sparse/blurry images → α stays near asri_alpha (lean on source).
                        # Dense/confident images → α falls toward asri_confgate_min.
                        k_avg_score = fg_scores[fg_preds == k].mean().item()
                        _alpha = (self.asri_confgate_min
                                  + (self.asri_alpha - self.asri_confgate_min)
                                  * math.exp(-self.asri_confgate_lambda * N * k_avg_score))
                    else:
                        _alpha = self.asri_alpha
                    if _alpha > 0.0 and self.s_stats is not None:
                        src_mean = self.s_stats["fg"][k][0].to(self.device)
                        cur_target_mean = (1.0 - _alpha) * raw_updated_mean + _alpha * src_mean
                    else:
                        cur_target_mean = raw_updated_mean
                    # ──────────────────────────────────────────────────────────

                    # Exp 12/16 Sol-B: source residual injection at KL-loss time only.
                    # EMA prototype stays raw (stored above); KL t_dist mean is blended.
                    if self.proto_method in ("dual_memory", "reset_dual_memory") and k in self.s_proto_anchor:
                        a = self.source_anchor_alpha
                        cur_target_mean = (1.0 - a) * cur_target_mean \
                                          + a * self.s_proto_anchor[k].to(self.device)

                    t_dist = torch.distributions.MultivariateNormal(cur_target_mean, self.s_stats["fg"][k][1].to(self.device) + self.template_cov["fg"][k].to(self.device))
                    s_dist = torch.distributions.MultivariateNormal(self.s_stats["fg"][k][0].to(self.device), self.s_stats["fg"][k][1].to(self.device) + self.template_cov["fg"][k].to(self.device))
                    if self.freq_weight:
                        class_weight = np.log((max([self.ema_n[_k] for _k in range(self.num_classes)]) / self.ema_n[k]))
                        cur_loss_fg_align = min(class_weight + 0.01, 10) ** 2 * \
                                            ((torch.distributions.kl.kl_divergence(s_dist, t_dist)
                                              + torch.distributions.kl.kl_divergence(t_dist, s_dist)) / 2)
                    else:
                        cur_loss_fg_align = (torch.distributions.kl.kl_divergence(s_dist, t_dist)
                                             + torch.distributions.kl.kl_divergence(t_dist, s_dist)) / 2
                    if cur_loss_fg_align < 10**5:
                        loss_fg_align += cur_loss_fg_align
                        loss_n += 1

            if loss_n > 0:
                adapt_loss["fg_align"] = self.alpha_fg * loss_fg_align

        if self.gl_align is not None:
            loss_gl_align = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            if self.gl_align == "KL":
                for k in features.keys():
                    cur_feats = features[k].mean(dim=[2, 3])
                    N = cur_feats.shape[0]

                    gl_swema_key = f"gl_{k}"
                    if self.swema_k > 0 and self.dpema_apply_gl:
                        # SWEMA for global branch
                        if gl_swema_key not in self._swema_buf:
                            self._swema_buf[gl_swema_key] = deque()
                        self._swema_buf[gl_swema_key].append((cur_feats.sum(dim=0).detach(), N))
                        total_in_window = sum(cnt for _, cnt in self._swema_buf[gl_swema_key])
                        while total_in_window > self.swema_k and len(self._swema_buf[gl_swema_key]) > 1:
                            _, oldest_cnt = self._swema_buf[gl_swema_key].popleft()
                            total_in_window -= oldest_cnt
                        window_sum = sum(s for s, _ in self._swema_buf[gl_swema_key])
                        window_cnt = sum(c for _, c in self._swema_buf[gl_swema_key])
                        mu_recent = window_sum / window_cnt
                        src_mean_gl = self.s_stats["gl"][k][0].to(self.device)
                        cur_target_mean = ((1.0 - self.swema_alpha) * mu_recent
                                           + self.swema_alpha * src_mean_gl)
                        # Straight-through gradient: buffer stores detached values to avoid
                        # retaining old computation graphs across iterations, so gradient
                        # path through current features is re-attached here (value unchanged).
                        cur_target_mean = cur_target_mean + (
                            cur_feats.mean(dim=0) - cur_feats.mean(dim=0).detach()
                        )
                    elif self.dpema_apply_gl and self.ema_beta > 0.0:
                        # DPEMA for global branch
                        effective_beta = self.ema_beta ** N
                        cur_target_mean = (effective_beta * self.t_stats["gl"][k][0].to(self.device)
                                           + (1.0 - effective_beta) * cur_feats.mean(dim=0))
                    else:
                        # legacy gamma method
                        diff = cur_feats - self.t_stats["gl"][k][0][None, :].to(self.device)
                        delta = 1 / self.ema_gamma * diff.sum(dim=0)
                        cur_target_mean = self.t_stats["gl"][k][0].to(self.device) + delta

                    # ── ASRI_GL: source residual injection for global branch ──
                    # Applied at loss-time only; raw cur_target_mean is stored.
                    # Skipped when SWEMA is active (source anchor already embedded).
                    raw_gl_mean = cur_target_mean
                    if (self.asri_gl
                            and not (self.swema_k > 0 and self.dpema_apply_gl)
                            and self.asri_alpha > 0.0
                            and self.s_stats is not None):
                        src_mean_gl = self.s_stats["gl"][k][0].to(self.device)
                        cur_target_mean = ((1.0 - self.asri_alpha) * raw_gl_mean
                                           + self.asri_alpha * src_mean_gl)
                    # ──────────────────────────────────────────────────────────

                    t_dist = torch.distributions.MultivariateNormal(cur_target_mean, self.s_stats["gl"][k][1].to(self.device) + self.template_cov["gl"][k].to(self.device))
                    s_dist = torch.distributions.MultivariateNormal(self.s_stats["gl"][k][0].to(self.device), self.s_stats["gl"][k][1].to(self.device) + self.template_cov["gl"][k].to(self.device))
                    cur_loss_gl_align = (torch.distributions.kl.kl_divergence(s_dist, t_dist) + torch.distributions.kl.kl_divergence(t_dist, s_dist)) / 2
                    if cur_loss_gl_align < 10 ** 5:
                        loss_gl_align += cur_loss_gl_align
                        self.t_stats["gl"][k] = (raw_gl_mean.detach(), None)
            elif self.gl_align == "bn_stats":
                cur_bn_stats = [(l.mean, l.var) for l in list(self.backbone.bottom_up.modules()) if (isinstance(l, FrozenBatchNorm2d) or isinstance(l, LayerNorm)) and l.out_batch_norm]
                for idx in range(len(cur_bn_stats)):
                    if cur_bn_stats[idx][0].shape[0] > 0 and cur_bn_stats[idx][0].shape[0] == self.s_stats["bn_stats"][idx][0].shape[0]:
                        h, w = min(cur_bn_stats[idx][0].shape[1], self.s_stats["bn_stats"][idx][0].shape[1]), min(cur_bn_stats[idx][0].shape[2], self.s_stats["bn_stats"][idx][0].shape[2])
                        loss_gl_align += nn.L1Loss()(cur_bn_stats[idx][0][:, :h, :w], self.s_stats["bn_stats"][idx][0][:, :h, :w])
                        loss_gl_align += self.alpha_fg * nn.L1Loss()(cur_bn_stats[idx][1][:, :h, :w], self.s_stats["bn_stats"][idx][1][:, :h, :w])
                        # loss_gl_align += nn.L1Loss()(cur_bn_stats[idx][0], self.s_stats["bn_stats"][idx][0])
                        # loss_gl_align += self.alpha_fg * nn.L1Loss()(cur_bn_stats[idx][1], self.s_stats["bn_stats"][idx][1])
            adapt_loss["global_align"] = self.alpha_gl * loss_gl_align

        def _mean_or_none(vals):
            return float(sum(vals) / len(vals)) if vals else None

        self._last_drift_stats = {
            "fg_num_classes": len(drift_proto_classes),
            "fg_num_boxes": int(sum(drift_fg_counts)),
            "fg_count_mean": _mean_or_none(drift_fg_counts),
            "fg_score_mean": _mean_or_none(drift_fg_scores),
            "proto_cos_source_mean": _mean_or_none(drift_proto_cos_source),
            "proto_drift_source_mean": (1.0 - _mean_or_none(drift_proto_cos_source))
                if drift_proto_cos_source else None,
            "proto_cos_batch_mean": _mean_or_none(drift_proto_cos_batch),
            "proto_cos_drop_mean": _mean_or_none(drift_proto_drop),
            "proto_reset_count": len(drift_reset_classes),
            "proto_reset_classes": drift_reset_classes[:20],
        }

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes), adapt_loss, feature_sim
        return results, adapt_loss, feature_sim

    def collect_feats(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if isinstance(features, tuple):
            features = features[0]

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        # collect backbone features
        for k in features.keys():
            cur_feats = features[k].mean(dim=[2, 3]).detach()
            if k not in self.gl_features.keys():
                self.gl_features[k] = cur_feats
            else:
                self.gl_features[k] = torch.cat([self.gl_features[k], cur_feats], dim=0)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, gt_instances)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            # collect foreground class features
            pred_instances, predictions, box_features = self.roi_heads._forward_box(features, proposals, outs=True)
            results = self.roi_heads.forward_with_given_boxes(features, pred_instances)

            cls_probs = nn.Softmax(dim=1)(predictions[0])

            iou_with_gt = torch.Tensor([]).to(cls_probs.device)
            gt_label = torch.Tensor([]).to(cls_probs.device)
            cur_idx = 0
            for i, p in enumerate(proposals):
                if len(gt_instances[i]) > 0:
                    _iou_with_gt, _gt_label_idx = pairwise_iou(p.proposal_boxes, gt_instances[i].gt_boxes).max(dim=1)
                    iou_with_gt = torch.cat([iou_with_gt, _iou_with_gt])
                    _gt_label = torch.where(_iou_with_gt > self.collect_iou_thr, gt_instances[i].gt_classes[_gt_label_idx],
                                            self.num_classes)
                    gt_label = torch.cat([gt_label, _gt_label])
                else:
                    iou_with_gt = torch.cat([iou_with_gt, torch.zeros(len(p)).to(self.device)])
                    gt_label = torch.cat([gt_label, torch.ones(len(p), dtype=torch.long).to(self.device) * (
                                self.num_classes)])
                cur_idx += len(p)
            gt_label = gt_label.type(torch.long)
            box_features = box_features.clone().detach()
            for _gt in torch.unique(gt_label):
                gt = _gt.item()
                pick_idx = torch.randperm((gt_label == gt).sum())[:100]
                cur_feats = box_features[gt_label == gt][pick_idx]
                cur_iout_with_gt = iou_with_gt[gt_label == gt][pick_idx]
                if gt not in self.fg_features:
                    self.fg_features[gt] = cur_feats
                    self.iou_with_gt[gt] = cur_iout_with_gt
                else:
                    self.fg_features[gt] = torch.cat([self.fg_features[gt], cur_feats])
                    self.iou_with_gt[gt] = torch.cat([self.iou_with_gt[gt], cur_iout_with_gt])
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if isinstance(features, tuple):
            features = features[0]

        if self.online_adapt:
            for k in features.keys():
                cur_feats = features[k].mean(dim=[2, 3])
                diff = cur_feats - self.t_stats["gl"][k][0][None, :].to(self.device)
                delta = 1 / self.ema_gamma * diff.sum(dim=0)
                cur_target_mean = self.t_stats["gl"][k][0].to(self.device) + delta
                self.t_stats['gl'][k] = (cur_target_mean.detach(), None)
                    
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    @staticmethod
    def _postprocess_with_proposals(instances, proposals, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, proposals_per_image, input_per_image, image_size in zip(
            instances, proposals, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            p = detector_postprocess(proposals_per_image, height, width)
            processed_results.append({"instances": r, "proposals": p})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
