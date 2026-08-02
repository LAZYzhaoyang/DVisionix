# -*- coding: utf-8 -*-
"""检测任务的原子指标：COCO-style mAP。

MeanAveragePrecision 累积各图像的预测/标注，compute 返回
{"mAP", "mAP_50", "mAP_75"} 三个标量。遵循 BaseMetric 的 update/compute/reset 接口。

支持两种后端：
- 内置实现（默认）：纯 numpy 实现，无额外依赖，仅用于快速验证。
- torchmetrics（可选）：use_torchmetrics=True 时使用
  torchmetrics.detection.MeanAveragePrecision，需 pip install torchmetrics[detection]。
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from ..registry import METRICS
from .base import BaseMetric


@METRICS.register()
@METRICS.register(name="map")
@METRICS.register(name="mean_average_precision")
class MeanAveragePrecision(BaseMetric):
    """COCO-style mean Average Precision。

    Args:
        num_classes: 类别数量。
        iou_thresholds: IoU 阈值列表，默认 [0.5, 0.55, ..., 0.95]。
        recall_thresholds: Recall 阈值列表，默认 [0, 0.01, ..., 1.0]。
        use_torchmetrics: 是否使用 torchmetrics 后端（需安装）。
        name: 指标名称。
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresholds: Optional[List[float]] = None,
        recall_thresholds: Optional[List[float]] = None,
        use_torchmetrics: bool = False,
        name: str = "mAP",
    ):
        self.num_classes = num_classes
        self.use_torchmetrics = use_torchmetrics
        self.iou_thresholds = (
            iou_thresholds if iou_thresholds is not None else np.linspace(0.5, 0.95, 10).tolist()
        )
        self.recall_thresholds = (
            recall_thresholds
            if recall_thresholds is not None
            else np.linspace(0, 1.0, 101).tolist()
        )

        self._torch_metric = None
        if use_torchmetrics:
            try:
                from torchmetrics.detection import MeanAveragePrecision as _TMmAP

                self._torch_metric = _TMmAP(
                    box_format="xyxy",
                    iou_thresholds=self.iou_thresholds,
                    rec_thresholds=self.recall_thresholds,
                )
            except ImportError:
                import warnings

                warnings.warn("torchmetrics[detection] not installed; falling back to built-in mAP")
                self.use_torchmetrics = False

        super().__init__(name)

    def reset(self) -> None:
        self.detections: Dict[int, List[Dict]] = {c: [] for c in range(self.num_classes)}
        self.annotations: Dict[int, List[Dict]] = {c: [] for c in range(self.num_classes)}
        self.image_ids: set = set()
        if self._torch_metric is not None:
            self._torch_metric.reset()

    def update(
        self,
        pred_boxes: List[torch.Tensor],
        pred_scores: List[torch.Tensor],
        pred_labels: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
    ) -> None:
        """累积一个 batch 的预测与标注。

        Args:
            pred_boxes: 预测框列表，每个元素 (N, 4) [x1, y1, x2, y2]。
            pred_scores: 预测置信度列表，每个元素 (N,)。
            pred_labels: 预测类别列表，每个元素 (N,)。
            target_boxes: 目标框列表，每个元素 (M, 4)。
            target_labels: 目标类别列表，每个元素 (M,)。
        """
        if self.use_torchmetrics and self._torch_metric is not None:
            preds = [
                dict(boxes=pb, scores=ps, labels=pl.long())
                for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels)
            ]
            targets = [
                dict(boxes=tb, labels=tl.long()) for tb, tl in zip(target_boxes, target_labels)
            ]
            self._torch_metric.update(preds, targets)
            return

        batch_size = len(pred_boxes)
        for i in range(batch_size):
            image_id = len(self.image_ids)
            self.image_ids.add(image_id)

            boxes_np = pred_boxes[i].cpu().numpy()
            scores_np = pred_scores[i].cpu().numpy()
            labels_np = pred_labels[i].cpu().numpy()
            for box, score, label in zip(boxes_np, scores_np, labels_np):
                label_int = int(label)
                if 0 <= label_int < self.num_classes:
                    self.detections[label_int].append(
                        {"image_id": image_id, "box": box, "score": score}
                    )

            t_boxes_np = target_boxes[i].cpu().numpy()
            t_labels_np = target_labels[i].cpu().numpy()
            for box, label in zip(t_boxes_np, t_labels_np):
                label_int = int(label)
                if 0 <= label_int < self.num_classes:
                    self.annotations[label_int].append(
                        {"image_id": image_id, "box": box, "matched": False}
                    )

    def compute(self) -> Dict[str, float]:
        """计算 mAP / mAP_50 / mAP_75。"""
        if self.use_torchmetrics and self._torch_metric is not None:
            raw = self._torch_metric.compute()
            return {
                "mAP": float(raw["map"]),
                "mAP_50": float(raw["map_50"]),
                "mAP_75": float(raw["map_75"]),
            }

        aps_per_class = []
        for c in range(self.num_classes):
            if len(self.annotations[c]) == 0:
                continue
            aps_iou = [self._ap_for_class_at_iou(c, t) for t in self.iou_thresholds]
            aps_per_class.append(float(np.mean(aps_iou)) if aps_iou else 0.0)

        return {
            "mAP": float(np.mean(aps_per_class)) if aps_per_class else 0.0,
            "mAP_50": self._compute_map_at_iou(0.5),
            "mAP_75": self._compute_map_at_iou(0.75),
        }

    def _ap_for_class_at_iou(self, c: int, iou_thresh: float) -> float:
        dets = sorted(self.detections[c], key=lambda x: x["score"], reverse=True)
        if len(dets) == 0:
            return 0.0

        anns_by_image: Dict[int, List[Dict]] = {}
        for ann in self.annotations[c]:
            anns_by_image.setdefault(ann["image_id"], []).append(
                {"box": ann["box"], "matched": False}
            )

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        for det_idx, det in enumerate(dets):
            img_id = det["image_id"]
            if img_id not in anns_by_image:
                fp[det_idx] = 1
                continue
            best_iou, best_ann_idx = -1, -1
            for ann_idx, ann in enumerate(anns_by_image[img_id]):
                if ann["matched"]:
                    continue
                iou = self._compute_iou(det["box"], ann["box"])
                if iou > best_iou:
                    best_iou, best_ann_idx = iou, ann_idx
            if best_iou >= iou_thresh and best_ann_idx >= 0:
                tp[det_idx] = 1
                anns_by_image[img_id][best_ann_idx]["matched"] = True
            else:
                fp[det_idx] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        num_anns = sum(len(anns) for anns in anns_by_image.values())
        recall = tp_cumsum / num_anns if num_anns > 0 else tp_cumsum
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        return self._compute_ap(recall, precision)

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        intersection = w * h
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection + 1e-8
        return intersection / union

    def _compute_ap(self, recall: np.ndarray, precision: np.ndarray) -> float:
        """使用 101-point interpolation 计算 AP。"""
        ap = 0.0
        for r_thresh in self.recall_thresholds:
            mask = recall >= r_thresh
            p = np.max(precision[mask]) if np.any(mask) else 0.0
            ap += p
        return ap / len(self.recall_thresholds)

    def _compute_map_at_iou(self, iou_thresh: float) -> float:
        aps_per_class = []
        for c in range(self.num_classes):
            if len(self.annotations[c]) == 0:
                continue
            aps_per_class.append(self._ap_for_class_at_iou(c, iou_thresh))
        return float(np.mean(aps_per_class)) if aps_per_class else 0.0


__all__ = ["MeanAveragePrecision"]
