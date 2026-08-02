# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 线性评估任务：冻结 backbone，仅训练线性分类头（自监督表征质量评估）。
"""线性评估任务：冻结 backbone，仅训练线性分类头（自监督表征质量评估）。"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models.losses import CrossEntropy, compute_loss
from ..optim import build_optimizer, build_scheduler
from .base import BaseTask, _merge_legacy_hyperparams


class LinearEvalTask(BaseTask):
    """线性评估：冻结模型 backbone（eval 模式 + 不更新），仅训练任务持有的线性头。

    特征默认 L2 归一化（线性评估惯例）。``configure_optimizers`` 时按 ``backbone.num_features``
    惰性构建线性头，优化器只含线性头参数。
    """

    def __init__(
        self,
        num_classes: int,
        feature_norm: bool = True,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        loss: Any = None,
        metrics: Any = None,
        **kwargs,
    ):
        super().__init__(optimizer_cfg, scheduler_cfg, loss=loss, metrics=metrics)
        self.num_classes = int(num_classes)
        self.feature_norm = bool(feature_norm)
        self.optimizer_cfg = _merge_legacy_hyperparams(
            self.optimizer_cfg, learning_rate, weight_decay
        )
        if self.loss is None:
            self.loss = CrossEntropy()
        self.linear: Optional[nn.Linear] = None

    def _freeze_backbone(self, model: nn.Module) -> None:
        for p in model.backbone.parameters():
            p.requires_grad = False
        model.eval()  # 冻结 BN 统计

    def configure_optimizers(self, model: nn.Module) -> Dict[str, Any]:
        """惰性构建线性头并返回仅含线性头参数的优化器。"""
        self._freeze_backbone(model)
        in_features = getattr(model.backbone, "num_features", None)
        if in_features is None:
            raise ValueError("LinearEvalTask 需要 backbone 暴露 num_features")
        self.linear = nn.Linear(in_features, self.num_classes)
        optimizer = build_optimizer(self.optimizer_cfg, self.linear.parameters())
        scheduler, monitor = build_scheduler(self.scheduler_cfg, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}

    def _features(self, model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = model.backbone(image)
        if self.feature_norm:
            feats = F.normalize(feats, dim=1)
        return feats

    def training_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """单步训练：冻结 backbone，仅训练线性头。"""
        self._freeze_backbone(model)
        feats = self._features(model, batch["image"].to(device))
        logits = self.linear(feats)
        labels = batch["label"].to(device)
        loss, extras = compute_loss(self.loss, logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return {"loss": loss, "acc": acc, **extras}

    def validation_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """单步验证：线性头评估。"""
        self._freeze_backbone(model)
        feats = self._features(model, batch["image"].to(device))
        logits = self.linear(feats)
        labels = batch["label"].to(device)
        loss, extras = compute_loss(self.loss, logits, labels)
        return {"loss": loss, "preds": logits, "targets": labels, **extras}


__all__ = ["LinearEvalTask"]
