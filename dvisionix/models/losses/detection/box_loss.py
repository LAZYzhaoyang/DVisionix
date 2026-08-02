# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 检测框回归损失：GIoU / CIoU / L1。
"""检测框回归损失：GIoU / CIoU / L1。

约定：输入为已匹配的 ``(N, 4)`` 预测框与目标框对（[x1, y1, x2, y2] 绝对坐标）。
正负样本匹配（assigner）由上层负责。
"""

import math

import torch
import torch.nn.functional as F

from ....registry import LOSSES
from ..base import BaseLoss


def _compute_iou_union(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算逐对 IoU 与并集面积。"""
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    inter = w * h
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = area_p + area_t - inter + 1e-8
    return inter / union, union


@LOSSES.register()
@LOSSES.register(name="giou")
class GIoULoss(BaseLoss):
    name = "giou"
    """Generalized IoU Loss。"""

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        super().__init__(weight)
        self.reduction = reduction

    def forward(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """GIoU 损失：预测框 vs 目标框（像素 xyxy）。"""
        iou, union = _compute_iou_union(pred_boxes, target_boxes)
        cw = torch.max(pred_boxes[:, 2], target_boxes[:, 2]) - torch.min(
            pred_boxes[:, 0], target_boxes[:, 0]
        )
        ch = torch.max(pred_boxes[:, 3], target_boxes[:, 3]) - torch.min(
            pred_boxes[:, 1], target_boxes[:, 1]
        )
        c_area = cw * ch + 1e-8
        giou = iou - (c_area - union) / c_area
        loss = 1.0 - giou
        return loss.mean() if self.reduction == "mean" else loss.sum()


@LOSSES.register()
@LOSSES.register(name="ciou")
class CIoULoss(BaseLoss):
    name = "ciou"
    """Complete IoU Loss（中心距离 + 宽高比惩罚）。"""

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        super().__init__(weight)
        self.reduction = reduction

    def forward(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """CIoU 损失：GIoU + 宽高比惩罚项。"""
        iou, union = _compute_iou_union(pred_boxes, target_boxes)

        pcx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pcy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        tcx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        tcy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

        cw = torch.max(pred_boxes[:, 2], target_boxes[:, 2]) - torch.min(
            pred_boxes[:, 0], target_boxes[:, 0]
        )
        ch = torch.max(pred_boxes[:, 3], target_boxes[:, 3]) - torch.min(
            pred_boxes[:, 1], target_boxes[:, 1]
        )
        c2 = cw**2 + ch**2 + 1e-8

        w_p = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1e-6)
        h_p = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1e-6)
        w_t = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=1e-6)
        h_t = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=1e-6)

        v = (4.0 / (math.pi**2)) * (torch.atan(w_t / h_t) - torch.atan(w_p / h_p)) ** 2
        alpha = v / (1.0 - iou + v + 1e-8)
        ciou = iou - rho2 / c2 - alpha * v
        loss = 1.0 - ciou
        return loss.mean() if self.reduction == "mean" else loss.sum()


@LOSSES.register()
@LOSSES.register(name="l1_box")
class L1BoxLoss(BaseLoss):
    name = "l1_box"
    """L1 框回归损失（对归一化坐标直接回归时使用）。"""

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        super().__init__(weight)
        self.reduction = reduction

    def forward(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """L1 框回归损失（归一化 xywh）。"""
        loss = F.l1_loss(pred_boxes, target_boxes, reduction=self.reduction)
        return loss


__all__ = ["GIoULoss", "CIoULoss", "L1BoxLoss"]
