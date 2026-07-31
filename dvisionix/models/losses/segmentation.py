# -*- coding: utf-8 -*-
"""分割任务损失：DiceLoss 与 CombinedSegmentationLoss（CE + Dice）。"""

import torch
import torch.nn.functional as F

from .base import BaseLoss
from .classification import CrossEntropy
from ...registry import LOSSES


@LOSSES.register()
@LOSSES.register(name="dice")
class DiceLoss(BaseLoss):
    """Dice Loss，对类别不平衡不敏感（向量化 one-hot 构造）。"""

    name = "dice"

    def __init__(self, weight: float = 1.0, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__(weight)
        self.smooth = float(smooth)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        num_classes = logits.shape[1]
        valid = targets != self.ignore_index

        safe_targets = targets.clamp(min=0)
        targets_onehot = F.one_hot(safe_targets, num_classes).permute(0, 3, 1, 2).float()
        targets_onehot = targets_onehot * valid.unsqueeze(1).float()

        probs = F.softmax(logits, dim=1)
        if self.ignore_index is not None:
            ignore_mask = (targets == self.ignore_index).unsqueeze(1)
            probs = probs.masked_fill(ignore_mask, 0.0)

        batch_size, channels = logits.shape[:2]
        probs_flat = probs.view(batch_size, channels, -1)
        targets_flat = targets_onehot.view(batch_size, channels, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


@LOSSES.register()
@LOSSES.register(name="combined_segmentation")
class CombinedSegmentationLoss(BaseLoss):
    """分割组合损失：CrossEntropy + Dice。"""

    name = "combined_segmentation"

    def __init__(
        self,
        weight: float = 1.0,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = 255,
    ):
        super().__init__(weight)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ce = CrossEntropy(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


__all__ = ["DiceLoss", "CombinedSegmentationLoss"]