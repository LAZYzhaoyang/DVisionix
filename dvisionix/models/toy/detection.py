# -*- coding: utf-8 -*-
"""教学级网格检测模型（GridDetectionModel，YOLO 风格）。

仅用于演示与教学；生产请使用 FCOSDetector / RetinaNetDetector 等组件化检测器。
"""

import torch
import torch.nn as nn

from ..base import BaseModel
from ..postprocess import batched_nms


class GridDetectionModel(BaseModel):
    """单阶段网格检测器（骨干 stride 8，中心点分配）。"""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, width: int = 64, **kwargs):
        super().__init__()
        self.task_type = "detection"
        self.num_classes = num_classes
        self.out_channels = 5 + num_classes

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.backbone = nn.Sequential(
            block(in_channels, width),
            block(width, width * 2),
            block(width * 2, width * 4),
        )
        self.head = nn.Conv2d(width * 4, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.head(self.backbone(x))

    @torch.no_grad()
    def decode(
        self,
        preds: torch.Tensor,
        image_hw,
        score_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
    ):
        """网格输出 -> (boxes_list, scores_list, labels_list)（含 NMS）。"""
        B, C, GH, GW = preds.shape
        img_h, img_w = image_hw

        obj = torch.sigmoid(preds[:, 0, :, :])
        box = torch.sigmoid(preds[:, 1:5, :, :])
        cls_prob = torch.softmax(preds[:, 5:, :, :], dim=1)

        boxes_list, scores_list, labels_list = [], [], []
        for b in range(B):
            cx = box[b, 0] * img_w
            cy = box[b, 1] * img_h
            bw = box[b, 2] * img_w
            bh = box[b, 3] * img_h
            x1 = (cx - bw / 2).flatten()
            y1 = (cy - bh / 2).flatten()
            x2 = (cx + bw / 2).flatten()
            y2 = (cy + bh / 2).flatten()
            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            cls_conf, cls_idx = cls_prob[b].max(dim=0)
            scores = (obj[b] * cls_conf).flatten()
            labels = cls_idx.flatten()

            keep_mask = scores >= score_threshold
            boxes, scores, labels = boxes[keep_mask], scores[keep_mask], labels[keep_mask]

            if boxes.numel() > 0:
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)
                keep = batched_nms(boxes, scores, labels, iou_threshold)
                keep = keep[:max_detections]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        return boxes_list, scores_list, labels_list


__all__ = ["GridDetectionModel"]
