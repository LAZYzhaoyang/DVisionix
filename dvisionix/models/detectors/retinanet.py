# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: RetinaNet 检测器（anchor-based 单阶段）。
"""RetinaNet 检测器（anchor-based 单阶段）。

配置示例::

    model:
      type: retinanet
      num_classes: 3
      backbone: {type: timm_backbone, name: resnet18, features_only: true, out_indices: [1,2,3,4]}
      neck: {type: fpn, out_channels: 128}
      head: {type: retinanet_head, num_classes: 3, num_anchors: 9}
    # 对应 loss 可用 retinanet_detection（assigner: max_iou 或 atss）
"""

from typing import Any, Dict, Optional

import torch

from ...registry import MODELS
from ..layers.anchors import AnchorGenerator, delta2bbox
from ..postprocess import batched_nms
from .base import SingleStageDetector


@MODELS.register()
@MODELS.register(name="retinanet")
class RetinaNetDetector(SingleStageDetector):
    """RetinaNet anchor-based 单阶段检测器。"""

    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        out_indices: Optional[list] = None,
        num_classes: Optional[int] = None,
        strides=(8, 16, 32, 64, 128),
        base_sizes=(32, 64, 128, 256, 512),
        **kwargs,
    ):
        head_cfg = dict(head)
        if num_classes is not None and "num_classes" not in head_cfg:
            head_cfg["num_classes"] = num_classes
        super().__init__(backbone, head_cfg, neck, out_indices)
        self.anchor_gen = AnchorGenerator(strides=strides, base_sizes=base_sizes)
        if getattr(self.head, "num_anchors", None) != self.anchor_gen.num_anchors:
            raise ValueError(
                f"head.num_anchors ({getattr(self.head, 'num_anchors', None)}) 与 "
                f"anchor 生成器 ({self.anchor_gen.num_anchors}) 不一致"
            )

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
        return retinanet_decode(
            preds,
            image_hw,
            self.anchor_gen,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            topk_per_level=topk_per_level,
        )


__all__ = ["RetinaNetDetector"]


def retinanet_decode(
    preds,
    image_hw,
    anchor_gen,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
    topk_per_level: int = 1000,
):
    """RetinaNet 原始输出解码 -> (boxes_list, scores_list, labels_list)（含 NMS）。"""
    cls_outs, reg_outs = preds["cls"], preds["reg"]
    img_h, img_w = image_hw
    B = cls_outs[0].shape[0]
    anchors_per_level = anchor_gen.grid_anchors(cls_outs)
    boxes_list, scores_list, labels_list = [], [], []
    for b in range(B):
        per_image_boxes, per_image_scores, per_image_labels = [], [], []
        for lvl in range(len(cls_outs)):
            cls = cls_outs[lvl][b]
            reg = reg_outs[lvl][b]
            A = anchor_gen.num_anchors
            num_classes = cls.shape[0] // A
            _, h, w = cls.shape
            n_loc = h * w
            anchors = anchors_per_level[lvl]  # (n_loc*A, 4)

            cls_flat = (
                cls.reshape(A, num_classes, h, w).permute(2, 3, 0, 1).reshape(-1, num_classes)
            )  # (n_loc*A, C)
            reg_flat = reg.reshape(A, 4, h, w).permute(2, 3, 0, 1).reshape(-1, 4)
            boxes = delta2bbox(reg_flat, anchors)

            scores = torch.sigmoid(cls_flat).reshape(-1)  # (n_loc*A*C,)
            topk = min(topk_per_level, scores.numel())
            top_scores, top_idx = scores.topk(topk)
            keep = top_scores >= score_threshold
            top_scores, top_idx = top_scores[keep], top_idx[keep]
            if top_idx.numel() == 0:
                continue
            n_anchor = n_loc * A
            labels = top_idx // n_anchor
            anchor_idx = top_idx % n_anchor
            per_image_boxes.append(boxes[anchor_idx])
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
