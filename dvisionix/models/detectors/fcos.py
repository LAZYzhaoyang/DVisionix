# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: FCOS 检测器（anchor-free 单阶段）。
"""FCOS 检测器（anchor-free 单阶段）。

配置示例::

    model:
      type: fcos
      num_classes: 3
      backbone: {type: timm_backbone, name: resnet18, features_only: true, out_indices: [1,2,3,4]}
      neck: {type: fpn, out_channels: 128}
      head: {type: fcos_head, num_classes: 3, strides: [8, 16, 32, 64, 128]}
"""

from typing import Any, Dict, Optional

import torch

from ...registry import MODELS
from ..postprocess import batched_nms
from .base import SingleStageDetector


@MODELS.register()
@MODELS.register(name="fcos")
class FCOSDetector(SingleStageDetector):
    """FCOS anchor-free 单阶段检测器。"""

    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        out_indices: Optional[list] = None,
        num_classes: Optional[int] = None,
        **kwargs,
    ):
        head_cfg = dict(head)
        if num_classes is not None and "num_classes" not in head_cfg:
            head_cfg["num_classes"] = num_classes
        super().__init__(backbone, head_cfg, neck, out_indices)
        self.strides = list(getattr(self.head, "strides", (8, 16, 32, 64, 128)))

    def decode(
        self,
        preds,
        image_hw,
        score_threshold=0.05,
        iou_threshold=0.5,
        max_detections=100,
        topk_per_level=1000,
    ):
        """推理解码：preds + image_hw -> (boxes_list, scores_list, labels_list)。"""
        return fcos_decode(
            preds,
            image_hw,
            self.strides,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            topk_per_level=topk_per_level,
        )


__all__ = ["FCOSDetector"]


def fcos_decode(
    preds,
    image_hw,
    strides,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
    topk_per_level: int = 1000,
):
    """FCOS 原始输出解码 -> (boxes_list, scores_list, labels_list)（含 NMS）。

    preds: {"cls": [...], "reg": [...], "center": [...]} 各层张量。
    """
    cls_outs, reg_outs, center_outs = preds["cls"], preds["reg"], preds["center"]
    img_h, img_w = image_hw
    B = cls_outs[0].shape[0]
    boxes_list, scores_list, labels_list = [], [], []
    for b in range(B):
        per_image_boxes, per_image_scores, per_image_labels = [], [], []
        for lvl, stride in enumerate(strides):
            cls = cls_outs[lvl][b]
            reg = reg_outs[lvl][b]
            center = center_outs[lvl][b]
            num_classes, h, w = cls.shape
            device = cls.device

            ys = (torch.arange(h, device=device) + 0.5) * stride
            xs = (torch.arange(w, device=device) + 0.5) * stride
            cx, cy = torch.meshgrid(xs, ys, indexing="xy")
            dist = torch.exp(reg) * stride
            boxes = torch.stack(
                [
                    cx - dist[0],
                    cy - dist[1],
                    cx + dist[2],
                    cy + dist[3],
                ],
                dim=-1,
            ).reshape(
                -1, 4
            )  # (H*W, 4)

            cls_prob = torch.sigmoid(cls).reshape(num_classes, -1).t()  # (N, C)
            center_prob = torch.sigmoid(center).reshape(-1)
            scores = (cls_prob * center_prob[:, None]).reshape(-1)  # (N*C,)
            n_loc = boxes.shape[0]
            topk = min(topk_per_level, scores.numel())
            top_scores, top_idx = scores.topk(topk)
            keep = top_scores >= score_threshold
            top_scores, top_idx = top_scores[keep], top_idx[keep]
            if top_idx.numel() == 0:
                continue
            labels = top_idx // n_loc
            loc_idx = top_idx % n_loc
            per_image_boxes.append(boxes[loc_idx])
            per_image_scores.append(top_scores)
            per_image_labels.append(labels)

        if not per_image_boxes:
            boxes_list.append(torch.zeros((0, 4), device=cls_outs[0].device))
            scores_list.append(torch.zeros((0,), device=cls_outs[0].device))
            labels_list.append(torch.zeros((0,), dtype=torch.long, device=cls_outs[0].device))
            continue

        boxes = torch.cat(per_image_boxes, dim=0)
        scores = torch.cat(per_image_scores, dim=0)
        labels = torch.cat(per_image_labels, dim=0)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)
        keep = batched_nms(boxes, scores, labels, iou_threshold)[:max_detections]
        boxes_list.append(boxes[keep])
        scores_list.append(scores[keep])
        labels_list.append(labels[keep])
    return boxes_list, scores_list, labels_list
