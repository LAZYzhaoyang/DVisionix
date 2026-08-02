# -*- coding: utf-8 -*-
"""MaskFormer 实例分割任务组件（MaskFormerHead full 模式 + MaskFormerLoss + mask mAP）。"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...metrics import MaskAveragePrecision
from ...models.losses import MaskFormerLoss, compute_loss
from .base import BaseTask, _merge_legacy_hyperparams


class MaskFormerTask(BaseTask):
    """实例分割任务（Mask2Former 风格）。

    要求模型输出 full 模式 dict（MaskFormerHead output_mode="full"，或 SegmentationModel 透传）：
    {"pred_logits": (B, Q, C+1), "pred_masks": (B, Q, H, W), "semantic_logits": (B, C, H, W)}。

    默认损失：MaskFormerLoss（匈牙利匹配 + CE + mask BCE + Dice）；默认指标：MaskAveragePrecision（mask mAP）。
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
        mask_threshold: float = 0.5,
        max_detections: int = 100,
    ):
        super().__init__(optimizer_cfg, scheduler_cfg, loss=loss, metrics=metrics)
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.max_detections = max_detections
        self.optimizer_cfg = _merge_legacy_hyperparams(
            self.optimizer_cfg, learning_rate, weight_decay
        )
        if self.loss is None:
            self.loss = MaskFormerLoss(num_classes=num_classes)
        if self.metrics is None:
            self.metrics = MaskAveragePrecision(num_classes=num_classes)

    def training_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        images = batch["image"].to(device)
        preds = model(images)
        if not isinstance(preds, dict):
            raise ValueError(
                "MaskFormerTask 需要模型输出 dict（MaskFormerHead output_mode='full'）"
            )
        loss, extras = compute_loss(self.loss, preds, batch, device=device)
        return {"loss": loss, **extras}

    def validation_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        images = batch["image"].to(device)
        with torch.no_grad():
            preds = model(images)
            image_hw = (images.shape[2], images.shape[3])
            loss, extras = compute_loss(self.loss, preds, batch, device=device)
            masks_list, scores_list, labels_list = model.decode(
                preds,
                image_hw,
                score_threshold=self.score_threshold,
                mask_threshold=self.mask_threshold,
                max_detections=self.max_detections,
            )
            target_masks = [m.to(device) for m in batch["mask"]]
            target_labels = [
                lb.to(device)
                for lb in batch.get("labels", [torch.full_like(m, 1) for m in target_masks])
            ]
        return {
            "loss": loss,
            **extras,
            "preds": (masks_list, scores_list, labels_list),
            "targets": (target_masks, target_labels),
        }

    def update_metrics(self, preds: Any, targets: Any) -> None:
        if self.metrics is None:
            return
        masks_list, scores_list, labels_list = preds
        target_masks, target_labels = targets
        self.metrics.update(masks_list, scores_list, labels_list, target_masks, target_labels)


__all__ = ["MaskFormerTask"]
