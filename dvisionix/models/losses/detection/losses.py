# -*- coding: utf-8 -*-
"""检测组合损失：GridDetectionLoss（单阶段网格检测器默认损失）。

GridDetectionLoss 内部使用 GridAssigner 做目标分配，再由
objectness(BCE) / box(L1) / cls(CE) 三支损失组合而成，返回 dict 供 Task 与日志使用。
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..base import BaseLoss
from .assigner import GridAssigner
from ....registry import LOSSES


@LOSSES.register()
@LOSSES.register(name="objectness")
class ObjectnessLoss(BaseLoss):
    """Objectness BCE-with-logits 损失。"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets)


@LOSSES.register()
@LOSSES.register(name="grid_detection")
class GridDetectionLoss(BaseLoss):
    """单阶段网格检测损失（GridDetectionModel 配套）。

    Args:
        num_classes: 类别数。
        obj_weight / box_weight / cls_weight: 三支损失权重。
    """

    def __init__(
        self,
        num_classes: int,
        obj_weight: float = 1.0,
        box_weight: float = 5.0,
        cls_weight: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.num_classes = num_classes
        self.obj_weight = float(obj_weight)
        self.box_weight = float(box_weight)
        self.cls_weight = float(cls_weight)
        self.assigner = GridAssigner(num_classes)

    def forward(
        self,
        preds: torch.Tensor,
        batch: Dict[str, Any],
        image_hw: Optional[tuple] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            device = preds.device
        if image_hw is None:
            raise ValueError("GridDetectionLoss requires image_hw=(H, W)")

        obj_target, box_target, cls_target, num_pos = self.assigner(
            preds.shape, batch["boxes"], batch["labels"], image_hw, device
        )

        obj_logits = preds[:, 0, :, :]
        box_pred = torch.sigmoid(preds[:, 1:5, :, :])
        cls_logits = preds[:, 5:, :, :]

        obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target)

        pos_mask = obj_target > 0.5
        if pos_mask.sum() > 0:
            pm = pos_mask.unsqueeze(1).expand_as(box_pred)
            box_loss = F.l1_loss(box_pred[pm], box_target[pm])
            cls_perm = cls_logits.permute(0, 2, 3, 1)[pos_mask]
            cls_loss = F.cross_entropy(cls_perm, cls_target[pos_mask])
            cls_acc = (cls_perm.argmax(dim=1) == cls_target[pos_mask]).float().mean()
        else:
            box_loss = box_pred.sum() * 0.0
            cls_loss = cls_logits.sum() * 0.0
            cls_acc = torch.tensor(0.0, device=device)

        total = self.obj_weight * obj_loss + self.box_weight * box_loss + self.cls_weight * cls_loss
        return {
            "loss": total,
            "obj_loss": obj_loss,
            "box_loss": box_loss,
            "cls_loss": cls_loss,
            "cls_acc": cls_acc,
        }


__all__ = ["ObjectnessLoss", "GridDetectionLoss"]