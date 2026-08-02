# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 分割任务损失：DiceLoss 与 CombinedSegmentationLoss（CE + Dice）。
"""分割任务损失：DiceLoss 与 CombinedSegmentationLoss（CE + Dice）。"""

import torch
import torch.nn.functional as F

from ...registry import LOSSES
from .base import BaseLoss
from .classification import CrossEntropy


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
        """Dice 损失：类别不平衡不敏感。"""
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
        """分割组合损失：CE + Dice 加权求和。"""
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


__all__ = ["DiceLoss", "CombinedSegmentationLoss", "MaskFormerLoss"]


def _dice_loss(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """pred/target: (N, HW) 0-1；返回逐样本 Dice loss 均值。"""
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()


@LOSSES.register()
@LOSSES.register(name="maskformer_loss")
class MaskFormerLoss(BaseLoss):
    """Mask2Former 风格 mask 分类损失：匈牙利匹配（类 + Dice 代价）+ CE + Dice + BCE。

    输入 preds：{"pred_logits": (B, Q, C+1), "pred_masks": (B, Q, H, W)}（MaskFormerHead full 模式）。
    """

    def __init__(
        self,
        num_classes: int,
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cls_weight: float = 1.0,
        mask_weight: float = 20.0,
        dice_weight: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.num_classes = num_classes
        self.cost_class = float(cost_class)
        self.cost_mask = float(cost_mask)
        self.cls_weight = float(cls_weight)
        self.mask_weight = float(mask_weight)
        self.dice_weight = float(dice_weight)

    def _match(self, logits, masks, gt_objects):
        """logits (Q, C+1)，masks (Q, H*W)，gt_objects: [(label, binary_mask HW)]。"""

        from .detection.matcher import _hungarian

        q = logits.shape[0]
        m = len(gt_objects)
        if m == 0:
            return torch.empty(0, dtype=torch.long, device=logits.device), torch.empty(
                0, dtype=torch.long, device=logits.device
            )
        log_p = torch.log_softmax(logits, dim=-1)  # (Q, C+1)
        cost = torch.zeros((q, m), device=logits.device)
        for j, (label, gt_mask) in enumerate(gt_objects):
            cost[:, j] = -self.cost_class * log_p[:, label]
            dice = 1.0 - (2.0 * (masks * gt_mask).sum(1) + 1.0) / (
                masks.sum(1) + gt_mask.sum() + 1.0
            )
            cost[:, j] = cost[:, j] + self.cost_mask * dice
        row, col = _hungarian(cost.detach().cpu().numpy())
        return (
            torch.as_tensor(row, dtype=torch.long, device=logits.device),
            torch.as_tensor(col, dtype=torch.long, device=logits.device),
        )

    def forward(self, preds, batch, device=None, **kwargs) -> dict:
        """MaskFormer 损失：匈牙利 mask 匹配 + CE/Dice/BCE。"""
        if device is None:
            device = preds["pred_logits"].device
        logits_all, masks_all = preds["pred_logits"], preds["pred_masks"]
        total_cls = torch.tensor(0.0, device=device)
        total_dice = torch.tensor(0.0, device=device)
        total_bce = torch.tensor(0.0, device=device)

        for b in range(len(batch["mask"])):
            gt_mask = batch["mask"][b].to(device).long()  # (H, W)
            if gt_mask.shape[-2:] != masks_all.shape[-2:]:
                gt_mask = (
                    F.interpolate(
                        gt_mask.float().unsqueeze(0).unsqueeze(0),
                        size=masks_all.shape[-2:],
                        mode="nearest",
                    )
                    .long()
                    .squeeze(0)
                    .squeeze(0)
                )
            logits = logits_all[b]  # (Q, C+1)
            masks = masks_all[b].flatten(1).sigmoid()  # (Q, HW)

            if batch.get("instance_masks") is not None:
                # 真实实例 GT：每图 (N, H, W) 二值掩码 + (N,) 类别
                inst_masks = batch["instance_masks"][b]
                inst_labels = batch["instance_labels"][b]
                gt_objects = []
                for lb, im in zip(inst_labels, inst_masks):
                    im = im.to(device)
                    if im.shape[-2:] != masks_all.shape[-2:]:
                        im = (
                            F.interpolate(
                                im.float().unsqueeze(0).unsqueeze(0),
                                size=masks_all.shape[-2:],
                                mode="nearest",
                            )
                            .bool()
                            .squeeze(0)
                            .squeeze(0)
                        )
                    gt_objects.append((int(lb), im.reshape(-1).float()))
            else:
                # 语义掩码退化：按类别连通区域近似实例
                gt_objects = []
                for c in range(self.num_classes):
                    obj = gt_mask == c
                    if obj.any():
                        gt_objects.append((c, obj.reshape(-1).float()))

            pred_idx, gt_idx = self._match(logits, masks, gt_objects)

            targets = torch.full(
                (logits.shape[0],), self.num_classes, dtype=torch.long, device=device
            )
            if gt_idx.numel() > 0:
                targets[pred_idx] = torch.tensor(
                    [gt_objects[i][0] for i in gt_idx.tolist()], device=device
                )
            total_cls = total_cls + F.cross_entropy(logits, targets)

            if gt_idx.numel() > 0:
                pm = masks[pred_idx]
                gm = torch.stack([gt_objects[i][1] for i in gt_idx.tolist()], dim=0)
                total_dice = total_dice + _dice_loss(pm, gm)
                total_bce = total_bce + F.binary_cross_entropy(pm, gm)

        total = (
            self.cls_weight * total_cls
            + self.mask_weight * total_bce
            + self.dice_weight * total_dice
        )
        return {
            "loss": total,
            "cls_loss": total_cls,
            "mask_bce_loss": total_bce,
            "mask_dice_loss": total_dice,
        }
