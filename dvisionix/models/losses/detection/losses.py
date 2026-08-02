# -*- coding: utf-8 -*-
"""检测组合损失：GridDetectionLoss（单阶段网格检测器默认损失）。

GridDetectionLoss 内部使用 GridAssigner 做目标分配，再由
objectness(BCE) / box(L1) / cls(CE) 三支损失组合而成，返回 dict 供 Task 与日志使用。
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..base import BaseLoss
from ...detectors.anchors import AnchorGenerator, bbox2delta
from .assigner import GridAssigner, FCOSAssigner, MaxIoUAssigner, ATSSAssigner
from .box_loss import GIoULoss
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


@LOSSES.register()
@LOSSES.register(name="sigmoid_focal")
class SigmoidFocalLoss(BaseLoss):
    """BCE 版 Focal Loss（RetinaNet / FCOS 分类分支）。

    targets 为 0/1 one-hot（与 logits 同形状）。
    """

    def __init__(self, weight: float = 1.0, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__(weight)
        self.alpha = alpha
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        mod = (1 - p_t).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            mod = mod * alpha_t
        loss = mod * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


@LOSSES.register()
@LOSSES.register(name="fcos_detection")
class FCOSDetectionLoss(BaseLoss):
    """FCOS 检测损失：Focal(cls) + GIoU/L1(reg) + BCE(center-ness)。

    Args:
        strides / scales / center_sampling: 透传给 FCOSAssigner。
    """

    def __init__(
        self,
        num_classes: int,
        strides=(8, 16, 32, 64, 128),
        scales=(0.0, 64.0, 128.0, 256.0, 512.0, 1e10),
        center_sampling: bool = True,
        center_sample_radius: float = 1.5,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        center_weight: float = 1.0,
        use_giou: bool = True,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.num_classes = num_classes
        self.strides = list(strides)
        self.assigner = FCOSAssigner(
            num_classes, strides=strides, scales=scales,
            center_sampling=center_sampling, center_sample_radius=center_sample_radius,
        )
        self.cls_weight = float(cls_weight)
        self.reg_weight = float(reg_weight)
        self.center_weight = float(center_weight)
        self.use_giou = use_giou
        self.focal = SigmoidFocalLoss()
        self.giou = GIoULoss()

    def _locations(self, h: int, w: int, stride: int, device):
        ys = (torch.arange(h, device=device) + 0.5) * stride
        xs = (torch.arange(w, device=device) + 0.5) * stride
        cx, cy = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1)

    def forward(
        self,
        preds,
        batch,
        image_hw=None,
        device=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            device = preds["cls"][0].device
        if image_hw is None:
            raise ValueError("FCOSDetectionLoss requires image_hw=(H, W)")

        cls_outs, reg_outs, center_outs = preds["cls"], preds["reg"], preds["center"]
        feature_shapes = [(o.shape[2], o.shape[3]) for o in cls_outs]
        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)
        total_center = torch.tensor(0.0, device=device)
        num_pos = 0

        for b in range(len(batch["boxes"])):
            boxes = batch["boxes"][b].to(device).float()
            labels_gt = batch["labels"][b].to(device).long()
            assigned = self.assigner.assign(feature_shapes, boxes, labels_gt, image_hw)

            for lvl, stride in enumerate(self.strides):
                cls_l = cls_outs[lvl][b]
                reg_l = reg_outs[lvl][b]
                cen_l = center_outs[lvl][b]
                n_loc = cls_l.shape[1] * cls_l.shape[2]

                cls_flat = cls_l.reshape(self.num_classes, -1).t()
                reg_flat = reg_l.reshape(4, -1).t()
                cen_flat = cen_l.reshape(-1)

                lbl, box_t, reg_t, cnt_t = assigned[lvl]
                cls_target = torch.zeros_like(cls_flat)
                pos = lbl > 0
                if pos.any():
                    cls_target[pos, lbl[pos] - 1] = 1.0

                total_cls = total_cls + self.focal(cls_flat, cls_target)

                if pos.any():
                    num_pos += int(pos.sum())
                    locs = self._locations(cls_l.shape[1], cls_l.shape[2], stride, device)
                    pred_ltrb = torch.exp(reg_flat[pos].clamp(min=-8, max=8)) * stride
                    pred_boxes = torch.stack([
                        locs[pos, 0] - pred_ltrb[:, 0],
                        locs[pos, 1] - pred_ltrb[:, 1],
                        locs[pos, 0] + pred_ltrb[:, 2],
                        locs[pos, 1] + pred_ltrb[:, 3],
                    ], dim=1)
                    giou_loss = self.giou(pred_boxes, box_t[pos])
                    l1_loss = F.l1_loss(reg_flat[pos], reg_t[pos])
                    total_reg = total_reg + (giou_loss if self.use_giou else l1_loss) + l1_loss
                    total_center = total_center + F.binary_cross_entropy_with_logits(
                        cen_flat[pos], cnt_t[pos]
                    )

        total = (self.cls_weight * total_cls + self.reg_weight * total_reg
                 + self.center_weight * total_center)
        return {
            "loss": total,
            "cls_loss": total_cls,
            "reg_loss": total_reg,
            "center_loss": total_center,
            "num_pos": torch.tensor(num_pos, dtype=torch.float32, device=device),
        }


@LOSSES.register()
@LOSSES.register(name="retinanet_detection")
class RetinaNetLoss(BaseLoss):
    """RetinaNet 检测损失：SigmoidFocal(cls) + SmoothL1(reg)。

    Args:
        assigner: 'max_iou'（默认）或 'atss'。
    """

    def __init__(
        self,
        num_classes: int,
        strides=(8, 16, 32, 64, 128),
        base_sizes=(32, 64, 128, 256, 512),
        assigner: str = "max_iou",
        pos_iou_thr: float = 0.5,
        neg_iou_thr: float = 0.4,
        topk: int = 9,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.num_classes = num_classes
        self.strides = list(strides)
        self.anchor_gen = AnchorGenerator(strides=strides, base_sizes=base_sizes)
        if assigner == "max_iou":
            self.assigner = MaxIoUAssigner(num_classes, pos_iou_thr, neg_iou_thr)
            self._atss = False
        elif assigner == "atss":
            self.assigner = ATSSAssigner(num_classes, num_anchors=self.anchor_gen.num_anchors, topk=topk)
            self._atss = True
        else:
            raise ValueError(f"未知 assigner: {assigner!r}（可选 max_iou / atss）")
        self.cls_weight = float(cls_weight)
        self.reg_weight = float(reg_weight)
        self.focal = SigmoidFocalLoss()

    def forward(self, preds, batch, image_hw=None, device=None, **kwargs) -> Dict[str, torch.Tensor]:
        if device is None:
            device = preds["cls"][0].device
        if image_hw is None:
            raise ValueError("RetinaNetLoss requires image_hw=(H, W)")

        cls_outs, reg_outs = preds["cls"], preds["reg"]
        anchors_per_level = self.anchor_gen.grid_anchors(cls_outs)
        anchors_flat = torch.cat(anchors_per_level, dim=0)
        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)
        num_pos = 0

        for b in range(len(batch["boxes"])):
            boxes = batch["boxes"][b].to(device).float()
            labels_gt = batch["labels"][b].to(device).long()
            if self._atss:
                labels, bbox_t = self.assigner.assign(
                    anchors_per_level, boxes, labels_gt, self.strides, image_hw
                )
            else:
                labels, bbox_t = self.assigner.assign(anchors_flat, boxes, labels_gt, image_hw)

            for lvl in range(len(self.strides)):
                A = self.anchor_gen.num_anchors
                cls_l = cls_outs[lvl][b]
                reg_l = reg_outs[lvl][b]
                _, h, w = cls_l.shape
                n_loc = h * w

                cls_flat = cls_l.reshape(A, self.num_classes, h, w).permute(2, 3, 0, 1).reshape(-1, self.num_classes)
                reg_flat = reg_l.reshape(A, 4, h, w).permute(2, 3, 0, 1).reshape(-1, 4)

                start = sum(a.shape[0] for a in anchors_per_level[:lvl])
                end = start + n_loc * A
                lbl = labels[start:end]
                anchors_l = anchors_per_level[lvl]

                cls_target = torch.zeros_like(cls_flat)
                pos = lbl > 0
                if pos.any():
                    cls_target[pos, lbl[pos] - 1] = 1.0
                    delta_t = bbox2delta(anchors_l[pos], bbox_t[start:end][pos])
                    total_reg = total_reg + F.smooth_l1_loss(reg_flat[pos], delta_t)
                    num_pos += int(pos.sum())

                total_cls = total_cls + self.focal(cls_flat, cls_target)

        total = self.cls_weight * total_cls + self.reg_weight * total_reg
        return {
            "loss": total,
            "cls_loss": total_cls,
            "reg_loss": total_reg,
            "num_pos": torch.tensor(num_pos, dtype=torch.float32, device=device),
        }


__all__ = ["ObjectnessLoss", "GridDetectionLoss", "SigmoidFocalLoss", "FCOSDetectionLoss", "RetinaNetLoss"]
