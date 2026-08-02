# -*- coding: utf-8 -*-
"""多标签分类任务（MultiLabelTask）。"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...models.losses import BinaryCrossEntropy, compute_loss
from .base import BaseTask, _merge_legacy_hyperparams


class MultiLabelTask(BaseTask):
    """多标签分类任务：模型输出逐标签 logits，配合 BCEWithLogits 损失。

    输入：batch["image"] (B, C, H, W)、batch["label"] (B, K) 0/1 张量。
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        loss: Any = None,
        metrics: Any = None,
    ):
        super().__init__(optimizer_cfg, scheduler_cfg, loss=loss, metrics=metrics)
        self.num_classes = num_classes
        self.optimizer_cfg = _merge_legacy_hyperparams(self.optimizer_cfg, learning_rate, weight_decay)
        if self.loss is None:
            self.loss = BinaryCrossEntropy()

    def training_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float()
        logits = model(images)
        loss, extras = compute_loss(self.loss, logits, labels)
        return {"loss": loss, **extras}

    def validation_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float()
        with torch.no_grad():
            logits = model(images)
            loss, extras = compute_loss(self.loss, logits, labels)
        return {"loss": loss, "preds": logits, "targets": labels, **extras}

    def update_metrics(self, preds: Any, targets: Any) -> None:
        if self.metrics is not None:
            self.metrics.update(preds, targets)


__all__ = ["MultiLabelTask"]