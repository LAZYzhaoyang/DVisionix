# -*- coding: utf-8 -*-
"""YOLOv10 风格检测器（one-to-one 训练，推理无需 NMS）。"""

from typing import Any, Dict, Optional

import torch

from ...registry import MODELS
from .base import SingleStageDetector


@MODELS.register()
@MODELS.register(name="yolo_v10")
class NMSFreeYOLODetector(SingleStageDetector):
    """YOLOv10-lite：NMSFreeYOLOHead + OneToOneYOLOLoss 训练，推理直接 top-k（免 NMS）。"""

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
        self.strides = list(getattr(self.head, "strides", (8, 16, 32)))

    def _locations(self, h, w, stride, device):
        ys = (torch.arange(h, device=device) + 0.5) * stride
        xs = (torch.arange(w, device=device) + 0.5) * stride
        cx, cy = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1)

    def decode(
        self,
        preds,
        image_hw,
        score_threshold: float = 0.3,
        max_detections: int = 100,
    ):
        """免 NMS 解码：逐层解码 + 跨层 top-k（one-to-one 训练保证单框）。"""
        cls_outs, reg_outs = preds["cls"], preds["reg"]
        img_h, img_w = image_hw
        device = cls_outs[0].device
        B = cls_outs[0].shape[0]
        boxes_list, scores_list, labels_list = [], [], []
        for b in range(B):
            per_boxes, per_scores, per_labels = [], [], []
            for lvl, stride in enumerate(self.strides):
                cls = cls_outs[lvl][b]
                reg = reg_outs[lvl][b]
                c, h, w = cls.shape
                cls_flat = cls.reshape(c, -1).t().sigmoid()  # (HW, C)
                reg_flat = reg.reshape(4, -1).t() * stride  # (HW, 4)
                locs = self._locations(h, w, stride, device)
                boxes = torch.stack(
                    [
                        locs[:, 0] - reg_flat[:, 0],
                        locs[:, 1] - reg_flat[:, 1],
                        locs[:, 0] + reg_flat[:, 2],
                        locs[:, 1] + reg_flat[:, 3],
                    ],
                    dim=1,
                )
                scores, labels = cls_flat.max(dim=-1)
                keep = scores >= score_threshold
                per_boxes.append(boxes[keep])
                per_scores.append(scores[keep])
                per_labels.append(labels[keep])
            if per_boxes:
                boxes = torch.cat(per_boxes, dim=0)
                scores = torch.cat(per_scores, dim=0)
                labels = torch.cat(per_labels, dim=0)
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)
                k = min(max_detections, scores.numel())
                if k > 0:
                    idx = scores.topk(k).indices
                    boxes, scores, labels = boxes[idx], scores[idx], labels[idx]
            else:
                boxes = torch.zeros((0, 4), device=device)
                scores = torch.zeros((0,), device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
        return boxes_list, scores_list, labels_list


__all__ = ["NMSFreeYOLODetector"]
