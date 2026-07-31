# -*- coding: utf-8 -*-
"""分割任务组件。"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...models.losses import CrossEntropy, compute_loss
from ...metrics import get_preset_metrics
from .base import BaseTask, _merge_legacy_hyperparams


class SegmentationTask(BaseTask):
    """语义分割任务。

    输入：batch["image"] (B, C, H, W)、batch["mask"] (B, H, W) long。
    默认损失：CrossEntropy(ignore_index=255)；默认指标：mIoU / pixel_accuracy。
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
        ignore_index: int = 255,
    ):
        super().__init__(optimizer_cfg, scheduler_cfg, loss=loss, metrics=metrics)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.optimizer_cfg = _merge_legacy_hyperparams(self.optimizer_cfg, learning_rate, weight_decay)
        if self.loss is None:
            self.loss = CrossEntropy(ignore_index=ignore_index)
        if self.metrics is None:
            self.metrics = get_preset_metrics("segmentation", num_classes, ignore_index=ignore_index)

    def training_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss, extras = compute_loss(self.loss, logits, masks)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            valid = masks != self.ignore_index
            acc = (preds[valid] == masks[valid]).float().mean() if valid.sum() > 0 else torch.tensor(0.0, device=device)
        return {"loss": loss, "acc": acc, **extras}

    def validation_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.no_grad():
            logits = model(images)
            loss, extras = compute_loss(self.loss, logits, masks)
        return {"loss": loss, "preds": logits, "targets": masks, **extras}


__all__ = ["SegmentationTask"]