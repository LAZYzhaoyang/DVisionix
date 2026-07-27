# -*- coding: utf-8 -*-
"""
单阶段网格检测模型（YOLO 风格，真正可训练）

设计目标：在不依赖外部检测框架的前提下，提供一个自洽、可端到端训练的
简单检测器，用于演示与教学。

- 骨干下采样 8 倍，得到 (H/8, W/8) 的特征网格。
- 每个网格单元预测：objectness(1) + box(4: cx,cy,w,h) + 类别(num_classes)。
- 目标分配采用中心点分配：GT 框中心落入哪个单元，该单元即为正样本。

输出：raw 张量 (B, 5 + num_classes, GH, GW)，损失在 DetectionTask 中计算。
"""

import torch
import torch.nn as nn

from .base import BaseModel
from .postprocess import batched_nms


class GridDetectionModel(BaseModel):
    """单阶段网格检测器。"""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, width: int = 64):
        super().__init__()
        self.task_type = "detection"
        self.num_classes = num_classes
        self.out_channels = 5 + num_classes  # obj(1)+box(4)+cls(C)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        # 三次下采样 -> stride 8
        self.backbone = nn.Sequential(
            block(in_channels, width),
            block(width, width * 2),
            block(width * 2, width * 4),
        )
        self.head = nn.Conv2d(width * 4, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """返回原始预测张量 (B, 5 + num_classes, GH, GW)。"""
        feats = self.backbone(x)
        return self.head(feats)

    @torch.no_grad()
    def decode(
        self,
        preds: torch.Tensor,
        image_hw,
        score_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
    ):
        """
        将网格原始输出解码为每张图的检测结果（含 NMS）。

        Args:
            preds: 模型原始输出 (B, 5 + num_classes, GH, GW)
            image_hw: (H, W) 原图尺寸（像素）
            score_threshold: 置信度阈值（objectness * 类别概率）
            iou_threshold: NMS 的 IoU 阈值
            max_detections: 每张图最多保留的框数

        Returns:
            (boxes_list, scores_list, labels_list)，各为长度 B 的列表：
            - boxes:  Tensor(K, 4) [x1, y1, x2, y2]（像素坐标）
            - scores: Tensor(K,)
            - labels: Tensor(K,)
        """
        B, C, GH, GW = preds.shape
        img_h, img_w = image_hw
        device = preds.device

        obj = torch.sigmoid(preds[:, 0, :, :])            # (B, GH, GW)
        box = torch.sigmoid(preds[:, 1:5, :, :])          # (B, 4, GH, GW) -> cx,cy,w,h in [0,1]
        cls_prob = torch.softmax(preds[:, 5:, :, :], dim=1)  # (B, num_classes, GH, GW)

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
            boxes = torch.stack([x1, y1, x2, y2], dim=1)   # (GH*GW, 4)

            cls_conf, cls_idx = cls_prob[b].max(dim=0)     # (GH, GW)
            scores = (obj[b] * cls_conf).flatten()         # (GH*GW,)
            labels = cls_idx.flatten()                     # (GH*GW,)

            keep_mask = scores >= score_threshold
            boxes, scores, labels = boxes[keep_mask], scores[keep_mask], labels[keep_mask]

            if boxes.numel() > 0:
                # 裁剪到图像范围
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)
                keep = batched_nms(boxes, scores, labels, iou_threshold)
                keep = keep[:max_detections]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        return boxes_list, scores_list, labels_list
