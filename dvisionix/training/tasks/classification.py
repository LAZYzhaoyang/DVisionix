# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 分类任务组件。
"""分类任务组件。"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...metrics import get_preset_metrics
from ...models.losses import CrossEntropy, compute_loss
from .base import BaseTask, _merge_legacy_hyperparams


class ClassificationTask(BaseTask):
    """图像分类任务。

    输入：batch["image"] (B, C, H, W)、batch["label"] (B,)。
    默认损失：CrossEntropy；默认指标：accuracy / precision / recall / f1。
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        loss: Any = None,
        loss_function: Optional[nn.Module] = None,
        metrics: Any = None,
    ):
        super().__init__(optimizer_cfg, scheduler_cfg, loss=loss, metrics=metrics)
        self.num_classes = num_classes
        self.optimizer_cfg = _merge_legacy_hyperparams(
            self.optimizer_cfg, learning_rate, weight_decay
        )
        if loss_function is not None and self.loss is None:
            self.loss = loss_function
        if self.loss is None:
            self.loss = CrossEntropy()
        if self.metrics is None:
            self.metrics = get_preset_metrics("classification", num_classes)

    def training_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """单步训练：模型前向 + 分类损失，返回 {loss, ...}。"""
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss, extras = compute_loss(self.loss, logits, labels)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()
        return {"loss": loss, "acc": acc, **extras}

    def validation_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """单步验证：模型前向 + 指标更新，返回 {preds, targets, loss, ...}。"""
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            logits = model(images)
            loss, extras = compute_loss(self.loss, logits, labels)
        return {"loss": loss, "preds": logits, "targets": labels, **extras}


__all__ = ["ClassificationTask"]
