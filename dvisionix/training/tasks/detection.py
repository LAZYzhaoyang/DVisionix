# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 检测任务组件（单阶段网格检测器，配合 GridDetectionModel）。
"""检测任务组件（单阶段网格检测器，配合 GridDetectionModel）。"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...metrics import get_preset_metrics
from ...models.losses import GridDetectionLoss, compute_loss
from .base import BaseTask, _merge_legacy_hyperparams


class DetectionTask(BaseTask):
    """目标检测任务。

    输入：batch["image"] (B, C, H, W)、batch["boxes"] List[Tensor(N, 4)]（像素 xyxy）、
    batch["labels"] List[Tensor(N,)]。
    默认损失：GridDetectionLoss；默认指标：COCO-style mAP。
    """

    def __init__(
        self,
        num_classes: int,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        loss: Any = None,
        metrics: Any = None,
        score_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
    ):
        super().__init__(optimizer_cfg, scheduler_cfg, loss=loss, metrics=metrics)
        self.num_classes = num_classes
        self.optimizer_cfg = _merge_legacy_hyperparams(
            self.optimizer_cfg, learning_rate, weight_decay
        )
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        if self.loss is None:
            self.loss = GridDetectionLoss(num_classes=num_classes)
        if self.metrics is None:
            self.metrics = get_preset_metrics("detection", num_classes)

    def training_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """单步训练：模型前向（附 batch）+ 检测损失。"""
        images = batch["image"].to(device)
        if getattr(model, "needs_batch", False):
            preds = model(images, batch=batch)
        else:
            preds = model(images)
        image_hw = (images.shape[2], images.shape[3])
        loss, extras = compute_loss(self.loss, preds, batch, image_hw=image_hw, device=device)
        return {"loss": loss, **extras}

    def validation_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """单步验证：前向 + decode + 检测指标更新。"""
        images = batch["image"].to(device)
        with torch.no_grad():
            preds = model(images)
            image_hw = (images.shape[2], images.shape[3])
            loss, extras = compute_loss(self.loss, preds, batch, image_hw=image_hw, device=device)
            boxes_list, scores_list, labels_list = model.decode(
                preds,
                image_hw,
                score_threshold=self.score_threshold,
                iou_threshold=self.iou_threshold,
                max_detections=self.max_detections,
            )
            target_boxes = [b.to(device) for b in batch["boxes"]]
            target_labels = [lb.to(device) for lb in batch["labels"]]
        return {
            "loss": loss,
            "preds": (boxes_list, scores_list, labels_list),
            "targets": (target_boxes, target_labels),
            **extras,
        }

    def update_metrics(self, preds: Any, targets: Any) -> None:
        """用 (preds, targets) 更新检测指标（mAP 等）。"""
        if self.metrics is None:
            return
        boxes_list, scores_list, labels_list = preds
        target_boxes, target_labels = targets
        self.metrics.update(boxes_list, scores_list, labels_list, target_boxes, target_labels)


__all__ = ["DetectionTask"]
