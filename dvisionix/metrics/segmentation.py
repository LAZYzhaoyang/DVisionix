# -*- coding: utf-8 -*-
"""分割任务的原子指标。

提供可自由组合的分割指标：MeanIoU / PixelAccuracy / DiceScore。
三者都基于混淆矩阵累积，遵循 BaseMetric 的 update/compute/reset 接口。
"""

from typing import Optional

import numpy as np
import torch

from .base import BaseMetric
from ..registry import METRICS


class _SegConfusionMetric(BaseMetric):
    """基于混淆矩阵的分割指标基类。

    Args:
        num_classes: 类别数。
        ignore_index: 忽略的标签值（如 255）。
        per_class: 为 True 时 compute 返回每类值列表，否则返回均值标量。
        name: 指标名称。
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = 255,
        per_class: bool = False,
        name: str = "metric",
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.per_class = per_class
        super().__init__(name)

    def reset(self) -> None:
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds_np = logits.argmax(dim=1).cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()

        if self.ignore_index is not None:
            keep = targets_np != self.ignore_index
            preds_np, targets_np = preds_np[keep], targets_np[keep]

        keep = (targets_np >= 0) & (targets_np < self.num_classes)
        preds_np, targets_np = preds_np[keep], targets_np[keep]

        hist = np.bincount(
            self.num_classes * targets_np.astype(np.int64) + preds_np.astype(np.int64),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist


@METRICS.register()
@METRICS.register(name="mean_iou")
class MeanIoU(_SegConfusionMetric):
    """平均交并比（mIoU）。"""

    def __init__(self, num_classes: int, ignore_index: Optional[int] = 255, per_class: bool = False, name: str = "mIoU"):
        super().__init__(num_classes, ignore_index, per_class, name)

    def compute(self):
        cm = self.confusion_matrix
        intersection = np.diag(cm).astype(np.float64)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        valid = union > 0
        iou = np.zeros(self.num_classes, dtype=np.float64)
        iou[valid] = intersection[valid] / union[valid]
        if self.per_class:
            return iou.tolist()
        return float(iou[valid].mean()) if np.any(valid) else 0.0


@METRICS.register()
@METRICS.register(name="pixel_accuracy")
class PixelAccuracy(_SegConfusionMetric):
    """像素准确率。"""

    def __init__(self, num_classes: int, ignore_index: Optional[int] = 255, name: str = "pixel_accuracy"):
        super().__init__(num_classes, ignore_index, per_class=False, name=name)

    def compute(self) -> float:
        cm = self.confusion_matrix
        total = cm.sum()
        correct = np.diag(cm).sum()
        return float(correct / total) if total > 0 else 0.0


@METRICS.register()
@METRICS.register(name="dice_score")
class DiceScore(_SegConfusionMetric):
    """Dice 系数（等价于 F1，逐类后取均值）。"""

    def __init__(self, num_classes: int, ignore_index: Optional[int] = 255, per_class: bool = False, name: str = "dice"):
        super().__init__(num_classes, ignore_index, per_class, name)

    def compute(self):
        cm = self.confusion_matrix
        intersection = np.diag(cm).astype(np.float64)
        denom = cm.sum(axis=1) + cm.sum(axis=0)
        valid = denom > 0
        dice = np.zeros(self.num_classes, dtype=np.float64)
        dice[valid] = 2 * intersection[valid] / denom[valid]
        if self.per_class:
            return dice.tolist()
        return float(dice[valid].mean()) if np.any(valid) else 0.0


__all__ = ["MeanIoU", "PixelAccuracy", "DiceScore"]