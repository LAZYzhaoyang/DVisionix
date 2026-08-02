# -*- coding: utf-8 -*-
"""检测组合损失：GridDetectionLoss（单阶段网格检测器默认损失）。

GridDetectionLoss 内部使用 GridAssigner 做目标分配，再由
objectness(BCE) / box(L1) / cls(CE) 三支损失组合而成，返回 dict 供 Task 与日志使用。
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ....registry import LOSSES
from ...layers.anchors import AnchorGenerator, bbox2delta
from ...postprocess import box_iou
from ..base import BaseLoss
from .assigner import ATSSAssigner, FCOSAssigner, GridAssigner, MaxIoUAssigner, TaskAlignedAssigner
from .box_loss import GIoULoss
from .matcher import HungarianMatcher


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


@LOSSES.register()
@LOSSES.register(name="sigmoid_focal")
class SigmoidFocalLoss(BaseLoss):
    """BCE 版 Focal Loss（RetinaNet / FCOS 分类分支）。

    targets 为 0/1 one-hot（与 logits 同形状）。
    """

    def __init__(
        self, weight: float = 1.0, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
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
            num_classes,
            strides=strides,
            scales=scales,
            center_sampling=center_sampling,
            center_sample_radius=center_sample_radius,
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
                    pred_boxes = torch.stack(
                        [
                            locs[pos, 0] - pred_ltrb[:, 0],
                            locs[pos, 1] - pred_ltrb[:, 1],
                            locs[pos, 0] + pred_ltrb[:, 2],
                            locs[pos, 1] + pred_ltrb[:, 3],
                        ],
                        dim=1,
                    )
                    giou_loss = self.giou(pred_boxes, box_t[pos])
                    l1_loss = F.l1_loss(reg_flat[pos], reg_t[pos])
                    total_reg = total_reg + (giou_loss if self.use_giou else l1_loss) + l1_loss
                    total_center = total_center + F.binary_cross_entropy_with_logits(
                        cen_flat[pos], cnt_t[pos]
                    )

        total = (
            self.cls_weight * total_cls
            + self.reg_weight * total_reg
            + self.center_weight * total_center
        )
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
            self.assigner = ATSSAssigner(
                num_classes, num_anchors=self.anchor_gen.num_anchors, topk=topk
            )
            self._atss = True
        else:
            raise ValueError(f"未知 assigner: {assigner!r}（可选 max_iou / atss）")
        self.cls_weight = float(cls_weight)
        self.reg_weight = float(reg_weight)
        self.focal = SigmoidFocalLoss()

    def forward(
        self, preds, batch, image_hw=None, device=None, **kwargs
    ) -> Dict[str, torch.Tensor]:
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

                cls_flat = (
                    cls_l.reshape(A, self.num_classes, h, w)
                    .permute(2, 3, 0, 1)
                    .reshape(-1, self.num_classes)
                )
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


@LOSSES.register()
@LOSSES.register(name="yolo_detection")
class YOLOLoss(BaseLoss):
    """YOLOv8 风格损失：SigmoidFocal(cls) + GIoU/L1(reg)，TaskAlignedAssigner 分配。"""

    def __init__(
        self,
        num_classes: int,
        strides=(8, 16, 32),
        topk: int = 13,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        use_giou: bool = True,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.num_classes = num_classes
        self.strides = list(strides)
        self.assigner = TaskAlignedAssigner(num_classes, topk=topk)
        self.cls_weight = float(cls_weight)
        self.reg_weight = float(reg_weight)
        self.use_giou = use_giou
        self.focal = SigmoidFocalLoss()
        self.giou = GIoULoss()

    def _locations(self, h, w, stride, device):
        ys = (torch.arange(h, device=device) + 0.5) * stride
        xs = (torch.arange(w, device=device) + 0.5) * stride
        cx, cy = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1)

    def forward(
        self, preds, batch, image_hw=None, device=None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            device = preds["cls"][0].device
        if image_hw is None:
            raise ValueError("YOLOLoss requires image_hw=(H, W)")

        cls_outs, reg_outs = preds["cls"], preds["reg"]
        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)
        num_pos = 0

        for b in range(len(batch["boxes"])):
            boxes = batch["boxes"][b].to(device).float()
            labels_gt = batch["labels"][b].to(device).long()

            pred_boxes_l, pred_scores_l, centers_l = [], [], []
            for lvl, stride in enumerate(self.strides):
                cls_l = cls_outs[lvl][b]
                reg_l = reg_outs[lvl][b]
                c, h, w = cls_l.shape
                locs = self._locations(h, w, stride, device)
                centers_l.append(locs)
                ltrb = reg_l.reshape(4, -1).t() * stride
                pred_boxes = torch.stack(
                    [
                        locs[:, 0] - ltrb[:, 0],
                        locs[:, 1] - ltrb[:, 1],
                        locs[:, 0] + ltrb[:, 2],
                        locs[:, 1] + ltrb[:, 3],
                    ],
                    dim=1,
                )
                pred_boxes_l.append(pred_boxes)
                pred_scores_l.append(cls_l.reshape(c, -1).t())

            labels_l, bbox_t_l = self.assigner.assign(
                pred_boxes_l, pred_scores_l, centers_l, self.strides, boxes, labels_gt
            )

            for lvl, stride in enumerate(self.strides):
                cls_flat = cls_outs[lvl][b].reshape(self.num_classes, -1).t()
                reg_flat = reg_outs[lvl][b].reshape(4, -1).t()
                lbl = labels_l[lvl]
                box_t = bbox_t_l[lvl]

                cls_target = torch.zeros_like(cls_flat)
                pos = lbl > 0
                if pos.any():
                    cls_target[pos, lbl[pos] - 1] = 1.0
                total_cls = total_cls + self.focal(cls_flat, cls_target)

                if pos.any():
                    num_pos += int(pos.sum())
                    locs = centers_l[lvl][pos]
                    ltrb_t = torch.stack(
                        [
                            locs[:, 0] - box_t[pos, 0],
                            locs[:, 1] - box_t[pos, 1],
                            box_t[pos, 2] - locs[:, 0],
                            box_t[pos, 3] - locs[:, 1],
                        ],
                        dim=1,
                    ).clamp(min=0)
                    pred_boxes = pred_boxes_l[lvl][pos]
                    giou_loss = self.giou(pred_boxes, box_t[pos])
                    l1_loss = F.l1_loss(reg_flat[pos], ltrb_t / stride)
                    total_reg = total_reg + (giou_loss if self.use_giou else l1_loss) + l1_loss

        total = self.cls_weight * total_cls + self.reg_weight * total_reg
        return {
            "loss": total,
            "cls_loss": total_cls,
            "reg_loss": total_reg,
            "num_pos": torch.tensor(num_pos, dtype=torch.float32, device=device),
        }


@LOSSES.register()
@LOSSES.register(name="detr_detection")
class DETRLoss(BaseLoss):
    """DETR set-based 损失：匈牙利匹配 + CE(cls) + L1 + GIoU(box)。"""

    def __init__(
        self,
        num_classes: int,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cls_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)
        self.cls_weight = float(cls_weight)
        self.bbox_weight = float(bbox_weight)
        self.giou_weight = float(giou_weight)
        self.giou = GIoULoss()

    @staticmethod
    def _xywh_norm_to_px(boxes, img_w, img_h):
        x, y, w, h = boxes.unbind(dim=-1)
        x1 = (x - w / 2) * img_w
        y1 = (y - h / 2) * img_h
        x2 = (x + w / 2) * img_w
        y2 = (y + h / 2) * img_h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _px_to_xywh_norm(boxes, img_w, img_h):
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return torch.stack([cx, cy, w, h], dim=-1)

    def forward(
        self, preds, batch, image_hw=None, device=None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            device = preds["logits"].device
        if image_hw is None:
            raise ValueError("DETRLoss requires image_hw=(H, W)")
        img_h, img_w = image_hw
        logits, boxes = preds["logits"], preds["boxes"]

        total_cls = torch.tensor(0.0, device=device)
        total_l1 = torch.tensor(0.0, device=device)
        total_giou = torch.tensor(0.0, device=device)

        for b in range(len(batch["boxes"])):
            gt_boxes = batch["boxes"][b].to(device).float()
            gt_labels = batch["labels"][b].to(device).long()
            pred_logits = logits[b]
            pred_boxes = boxes[b]
            q = pred_logits.shape[0]

            if gt_boxes.numel() > 0:
                pred_idx, gt_idx = self.matcher(
                    pred_logits,
                    pred_boxes,
                    self._px_to_xywh_norm(gt_boxes, img_w, img_h),
                    gt_labels,
                )
            else:
                pred_idx = gt_idx = torch.empty((0,), dtype=torch.long, device=device)

            targets = torch.full((q,), self.num_classes, dtype=torch.long, device=device)
            if gt_idx.numel() > 0:
                targets[pred_idx] = gt_labels[gt_idx]
            total_cls = total_cls + F.cross_entropy(pred_logits, targets)

            if gt_idx.numel() > 0:
                pb = pred_boxes[pred_idx]
                gb = self._px_to_xywh_norm(gt_boxes, img_w, img_h)[gt_idx]
                total_l1 = total_l1 + F.l1_loss(pb, gb)
                pb_px = self._xywh_norm_to_px(pb, img_w, img_h)
                total_giou = total_giou + self.giou(pb_px, gt_boxes[gt_idx])

        total = (
            self.cls_weight * total_cls
            + self.bbox_weight * total_l1
            + self.giou_weight * total_giou
        )
        return {
            "loss": total,
            "cls_loss": total_cls,
            "l1_loss": total_l1,
            "giou_loss": total_giou,
        }


@LOSSES.register()
@LOSSES.register(name="yolo_v10_detection")
class OneToOneYOLOLoss(BaseLoss):
    """YOLOv10 风格 one-to-one 损失：每个 GT 只匹配一个最高质量预测（cls_score * IoU），
    推理无需 NMS。分类用 BCE（软目标），框回归用 GIoU。"""

    def __init__(
        self,
        num_classes: int,
        strides=(8, 16, 32),
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.num_classes = num_classes
        self.strides = list(strides)
        self.cls_weight = float(cls_weight)
        self.reg_weight = float(reg_weight)
        self.giou = GIoULoss()

    def _locations(self, h, w, stride, device):
        ys = (torch.arange(h, device=device) + 0.5) * stride
        xs = (torch.arange(w, device=device) + 0.5) * stride
        cx, cy = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1)

    def forward(self, preds, batch, image_hw=None, device=None, **kwargs) -> dict:
        if device is None:
            device = preds["cls"][0].device
        cls_outs, reg_outs = preds["cls"], preds["reg"]
        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)

        for b in range(len(batch["boxes"])):
            gt_boxes = batch["boxes"][b].to(device).float()
            gt_labels = batch["labels"][b].to(device).long()
            if gt_boxes.numel() == 0:
                continue
            pred_boxes_all, pred_scores_all, centers_all = [], [], []
            for lvl, stride in enumerate(self.strides):
                cls_l = cls_outs[lvl][b]
                reg_l = reg_outs[lvl][b]
                c, h, w = cls_l.shape
                locs = self._locations(h, w, stride, device)
                centers_all.append(locs)
                ltrb = reg_l.reshape(4, -1).t() * stride
                pred_boxes = torch.stack(
                    [
                        locs[:, 0] - ltrb[:, 0],
                        locs[:, 1] - ltrb[:, 1],
                        locs[:, 0] + ltrb[:, 2],
                        locs[:, 1] + ltrb[:, 3],
                    ],
                    dim=1,
                )
                pred_boxes_all.append(pred_boxes)
                pred_scores_all.append(cls_l.reshape(c, -1).t())
            pred_boxes_all = torch.cat(pred_boxes_all, dim=0)  # (N, 4)
            pred_scores_all = torch.cat(pred_scores_all, dim=0)  # (N, C)

            iou_matrix = box_iou(pred_boxes_all, gt_boxes)  # (N, M)
            cls_score = pred_scores_all.sigmoid()[:, gt_labels]  # (N, M)
            quality = cls_score * iou_matrix

            N, M = quality.shape
            cls_target = torch.zeros(N, self.num_classes, device=device)
            box_target = torch.zeros(N, 4, device=device)
            pos_mask = torch.zeros(N, dtype=torch.bool, device=device)
            matched_gt = torch.zeros(M, dtype=torch.bool, device=device)
            cand = quality.clone()
            for _ in range(min(N, M)):
                if not cand.any():
                    break
                flat = cand.argmax()
                p, g = flat // M, flat % M
                if quality[p, g] <= 0:
                    break
                cls_target[p, gt_labels[g]] = 1.0
                box_target[p] = gt_boxes[g]
                pos_mask[p] = True
                matched_gt[g] = True
                cand[:, g] = -1.0
                cand[p, :] = -1.0  # 每个预测只用一个 GT

            total_cls = (
                total_cls
                + F.binary_cross_entropy_with_logits(pred_scores_all, cls_target, reduction="none")
                .sum(dim=1)
                .mean()
            )
            if pos_mask.any():
                total_reg = (
                    total_reg + self.giou(pred_boxes_all[pos_mask], box_target[pos_mask]).mean()
                )

        total = self.cls_weight * total_cls + self.reg_weight * total_reg
        return {"loss": total, "cls_loss": total_cls, "giou_loss": total_reg}


@LOSSES.register()
@LOSSES.register(name="centernet_detection")
class CenterNetLoss(BaseLoss):
    """CenterNet 损失：penalty-reduced Focal（热图）+ L1（宽高 / 偏移，仅中心点）。"""

    def __init__(
        self,
        num_classes: int,
        stride: int = 4,
        hm_weight: float = 1.0,
        wh_weight: float = 0.1,
        offset_weight: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.num_classes = num_classes
        self.stride = int(stride)
        self.hm_weight = float(hm_weight)
        self.wh_weight = float(wh_weight)
        self.offset_weight = float(offset_weight)

    def _build_targets(self, boxes, labels, H, W, device):
        hm = torch.zeros((self.num_classes, H, W), device=device)
        wh = torch.zeros((2, H, W), device=device)
        offset = torch.zeros((2, H, W), device=device)
        reg_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        for box, lb in zip(boxes, labels):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            bw = (box[2] - box[0]).clamp(min=1.0)
            bh = (box[3] - box[1]).clamp(min=1.0)
            cx_s, cy_s = cx / self.stride, cy / self.stride
            ix, iy = int(cx_s), int(cy_s)
            if not (0 <= ix < W and 0 <= iy < H):
                continue
            radius = max(1, int(max(bw, bh) / self.stride / 6.0))
            sigma = radius / 3.0
            y0, y1 = max(0, iy - radius), min(H - 1, iy + radius)
            x0, x1 = max(0, ix - radius), min(W - 1, ix + radius)
            ys = torch.arange(y0, y1 + 1, device=device).float()
            xs = torch.arange(x0, x1 + 1, device=device).float()
            gy = torch.exp(-((ys - cy_s) ** 2) / (2 * sigma * sigma))
            gx = torch.exp(-((xs - cx_s) ** 2) / (2 * sigma * sigma))
            gauss = torch.outer(gy, gx)  # (y_range, x_range)
            hm[lb, y0 : y1 + 1, x0 : x1 + 1] = torch.maximum(
                hm[lb, y0 : y1 + 1, x0 : x1 + 1], gauss
            )
            hm[lb, iy, ix] = 1.0
            wh[:, iy, ix] = torch.tensor([bw, bh], device=device)
            offset[:, iy, ix] = torch.tensor([cx_s - ix, cy_s - iy], device=device)
            reg_mask[iy, ix] = True
        return hm, wh, offset, reg_mask

    def forward(self, preds, batch, image_hw=None, device=None, **kwargs) -> dict:
        if device is None:
            device = preds["heatmap"].device
        hm_pred = preds["heatmap"]
        wh_pred = preds["wh"]
        off_pred = preds["offset"]
        B, C, H, W = hm_pred.shape
        total_hm = torch.tensor(0.0, device=device)
        total_wh = torch.tensor(0.0, device=device)
        total_off = torch.tensor(0.0, device=device)

        for b in range(len(batch["boxes"])):
            boxes = batch["boxes"][b].to(device).float()
            labels = batch["labels"][b].to(device).long()
            hm_t, wh_t, off_t, reg_mask = self._build_targets(boxes, labels, H, W, device)

            pred = hm_pred[b].sigmoid().clamp(1e-4, 1 - 1e-4)
            pos = hm_t.eq(1).float()
            neg = 1.0 - hm_t
            loss_hm = -(
                pos * (1 - pred) ** 2 * pred.log()
                + neg * (1 - hm_t) ** 4 * pred**2 * (1 - pred).log()
            )
            total_hm = total_hm + loss_hm.mean()

            if reg_mask.any():
                total_wh = total_wh + F.l1_loss(wh_pred[b][:, reg_mask], wh_t[:, reg_mask])
                total_off = total_off + F.l1_loss(off_pred[b][:, reg_mask], off_t[:, reg_mask])

        total = (
            self.hm_weight * total_hm + self.wh_weight * total_wh + self.offset_weight * total_off
        )
        return {
            "loss": total,
            "hm_loss": total_hm,
            "wh_loss": total_wh,
            "offset_loss": total_off,
        }


@LOSSES.register()
@LOSSES.register(name="yolo_v9_detection")
class YOLOv9Loss(BaseLoss):
    """YOLOv9-lite 损失：主头 YOLOLoss + PGI 辅助头损失（aux_weight 加权，仅训练）。"""

    def __init__(
        self,
        num_classes: int,
        strides=(8, 16, 32),
        aux_strides=(4, 8),
        aux_weight: float = 0.25,
        topk: int = 13,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.main = YOLOLoss(num_classes=num_classes, strides=strides, topk=topk)
        self.aux = YOLOLoss(num_classes=num_classes, strides=aux_strides, topk=topk)
        self.aux_weight = float(aux_weight)

    def forward(self, preds, batch, image_hw=None, device=None, **kwargs) -> dict:
        main_out = self.main(
            {"cls": preds["cls"], "reg": preds["reg"]},
            batch,
            image_hw=image_hw,
            device=device,
        )
        if preds.get("aux_cls") is not None:
            aux_out = self.aux(
                {"cls": preds["aux_cls"], "reg": preds["aux_reg"]},
                batch,
                image_hw=image_hw,
                device=device,
            )
            main_out["loss"] = main_out["loss"] + self.aux_weight * aux_out["loss"]
            main_out["aux_cls_loss"] = aux_out["cls_loss"]
            main_out["aux_reg_loss"] = aux_out["reg_loss"]
        return main_out


__all__ = [
    "ObjectnessLoss",
    "GridDetectionLoss",
    "SigmoidFocalLoss",
    "FCOSDetectionLoss",
    "RetinaNetLoss",
    "YOLOLoss",
    "DETRLoss",
    "OneToOneYOLOLoss",
    "CenterNetLoss",
    "YOLOv9Loss",
]
