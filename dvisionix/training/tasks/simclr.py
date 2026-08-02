# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: SimCLR 对比学习任务（双视角 InfoNCE，自监督）。
"""SimCLR 对比学习任务（双视角 InfoNCE，自监督）。"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...models.losses import InfoNCELoss, compute_loss
from .base import BaseTask, _merge_legacy_hyperparams


class SimCLRTask(BaseTask):
    """SimCLR 自监督任务：模型输出两个增强视图的投影（SimCLRHead），InfoNCELoss 拉近正对。

    要求模型输入单个图像返回投影向量（如 LinearClassifier + simclr_head）；
    batch 需含 ``image1`` / ``image2``（由 SimCLRTransforms 生成）。
    """

    def __init__(
        self,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        loss: Any = None,
        temperature: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__(optimizer_cfg, scheduler_cfg, loss=loss)
        self.num_classes = num_classes  # 兼容装配器注入（自监督不使用）
        self.optimizer_cfg = _merge_legacy_hyperparams(
            self.optimizer_cfg, learning_rate, weight_decay
        )
        if self.loss is None:
            self.loss = InfoNCELoss(temperature=temperature)

    def training_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """单步训练：双视角对比损失（InfoNCE）。"""
        z1 = model(batch["image1"].to(device))
        z2 = model(batch["image2"].to(device))
        loss, extras = compute_loss(self.loss, z1, z2)
        return {"loss": loss, **extras}

    def validation_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """单步验证：返回投影向量（无监督无指标）。"""
        with torch.no_grad():
            z1 = model(batch["image1"].to(device))
            z2 = model(batch["image2"].to(device))
            loss, extras = compute_loss(self.loss, z1, z2)
        return {"loss": loss, **extras}


__all__ = ["SimCLRTask"]
