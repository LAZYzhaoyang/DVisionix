# -*- coding: utf-8 -*-
"""分类任务损失：CrossEntropy（含 label smoothing）与 FocalLoss。"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss
from ...registry import LOSSES


@LOSSES.register()
@LOSSES.register(name="cross_entropy")
class CrossEntropy(BaseLoss):
    """分类 / 分割通用的交叉熵损失（支持 ignore_index 与 label_smoothing）。"""

    name = "cross_entropy"

    def __init__(
        self,
        weight: float = 1.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__(weight)
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.criterion(logits, targets)


@LOSSES.register()
@LOSSES.register(name="focal")
class FocalLoss(BaseLoss):
    """Focal Loss：缓解类别不平衡，降低易分样本权重。

    支持 2D ``(B, C)`` 与 ND ``(B, C, ...)``（分割）输入。
    """

    name = "focal"

    def __init__(
        self,
        weight: float = 1.0,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__(weight)
        self.alpha = alpha
        self.gamma = float(gamma)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        num_classes = logits.shape[1]
        if logits.dim() > 2:
            logits = logits.permute(0, *range(2, logits.dim()), 1).reshape(-1, num_classes)
            targets = targets.reshape(-1)

        valid = targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0

        logits = logits[valid]
        targets = targets[valid]

        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        mod = (1.0 - pt).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            mod = mod * alpha_t
        loss = mod * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()




@LOSSES.register()
@LOSSES.register(name="binary_cross_entropy")
class BinaryCrossEntropy(BaseLoss):
    """多标签分类损失（BCE with logits，逐标签二分类）。"""

    name = "binary_cross_entropy"

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        super().__init__(weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets.float(), reduction=self.reduction)


__all__ = ["CrossEntropy", "FocalLoss", "BinaryCrossEntropy"]