# -*- coding: utf-8 -*-
"""分割任务的原子指标。

提供可自由组合的分割指标：MeanIoU / PixelAccuracy / DiceScore。
三者都基于混淆矩阵累积，遵循 BaseMetric 的 update/compute/reset 接口。
"""

from typing import Optional

import numpy as np
import torch

from ..registry import METRICS
from .base import BaseMetric


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
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist


@METRICS.register()
@METRICS.register(name="mean_iou")
class MeanIoU(_SegConfusionMetric):
    """平均交并比（mIoU）。"""

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = 255,
        per_class: bool = False,
        name: str = "mIoU",
    ):
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

    def __init__(
        self, num_classes: int, ignore_index: Optional[int] = 255, name: str = "pixel_accuracy"
    ):
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

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = 255,
        per_class: bool = False,
        name: str = "dice",
    ):
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


def mask_iou(pred: "np.ndarray", target: "np.ndarray") -> float:
    """二值 mask IoU。"""
    inter = float(np.logical_and(pred, target).sum())
    union = float(np.logical_or(pred, target).sum())
    return inter / (union + 1e-8)


@METRICS.register()
@METRICS.register(name="mask_ap")
class MaskAveragePrecision(BaseMetric):
    """COCO 风格 mask mAP（按类别、IoU 阈值 0.5:0.95，101-point 插值）。

    update(pred_masks, pred_scores, pred_labels, target_masks, target_labels)
    - pred_masks: List[Tensor(N_i, H, W) bool/0-1]（每张图）
    - target_masks: List[Tensor(M_i, H, W) bool/0-1]
    """

    def __init__(self, num_classes, iou_thresholds=None, recall_thresholds=None, name="mask_ap"):
        self.num_classes = num_classes
        self.iou_thresholds = (
            iou_thresholds if iou_thresholds is not None else np.linspace(0.5, 0.95, 10).tolist()
        )
        self.recall_thresholds = (
            recall_thresholds
            if recall_thresholds is not None
            else np.linspace(0, 1.0, 101).tolist()
        )
        super().__init__(name)

    def reset(self):
        self.detections = {c: [] for c in range(self.num_classes)}
        self.annotations = {c: [] for c in range(self.num_classes)}
        self.image_id = 0

    def update(self, pred_masks, pred_scores, pred_labels, target_masks, target_labels):
        for i in range(len(pred_masks)):
            img_id = self.image_id
            self.image_id += 1
            pm = pred_masks[i].detach().cpu().numpy()
            ps = pred_scores[i].detach().cpu().numpy()
            pl = pred_labels[i].detach().cpu().numpy()
            for m, s, lb in zip(pm, ps, pl):
                c = int(lb)
                if 0 <= c < self.num_classes:
                    self.detections[c].append(
                        {"image_id": img_id, "mask": m > 0.5, "score": float(s)}
                    )
            tm = target_masks[i].detach().cpu().numpy()
            tl = target_labels[i].detach().cpu().numpy()
            for m, lb in zip(tm, tl):
                c = int(lb)
                if 0 <= c < self.num_classes:
                    self.annotations[c].append(
                        {"image_id": img_id, "mask": m > 0.5, "matched": False}
                    )

    def compute(self):
        aps = []
        for c in range(self.num_classes):
            if len(self.annotations[c]) == 0:
                continue
            aps.append(np.mean([self._ap_at_iou(c, t) for t in self.iou_thresholds]))
        return {
            "mask_mAP": float(np.mean(aps)) if aps else 0.0,
            "mask_mAP_50": self._map_at(0.5),
            "mask_mAP_75": self._map_at(0.75),
        }

    def _ap_at_iou(self, c, iou_thr):
        dets = sorted(self.detections[c], key=lambda x: x["score"], reverse=True)
        if not dets:
            return 0.0
        anns_by_img = {}
        for ann in self.annotations[c]:
            anns_by_img.setdefault(ann["image_id"], []).append(
                {"mask": ann["mask"], "matched": False}
            )
        tp, fp = np.zeros(len(dets)), np.zeros(len(dets))
        for i, det in enumerate(dets):
            cands = anns_by_img.get(det["image_id"], [])
            best, best_idx = -1, -1
            for j, ann in enumerate(cands):
                if ann["matched"]:
                    continue
                iou = mask_iou(det["mask"], ann["mask"])
                if iou > best:
                    best, best_idx = iou, j
            if best >= iou_thr and best_idx >= 0:
                tp[i] = 1
                cands[best_idx]["matched"] = True
            else:
                fp[i] = 1
        n_anns = sum(len(v) for v in anns_by_img.values())
        recall = np.cumsum(tp) / (n_anns + 1e-8)
        precision = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp) + 1e-8)
        ap = 0.0
        for r in self.recall_thresholds:
            m = recall >= r
            ap += float(np.max(precision[m])) if m.any() else 0.0
        return ap / len(self.recall_thresholds)

    def _map_at(self, iou_thr):
        aps = [
            self._ap_at_iou(c, iou_thr)
            for c in range(self.num_classes)
            if len(self.annotations[c]) > 0
        ]
        return float(np.mean(aps)) if aps else 0.0
