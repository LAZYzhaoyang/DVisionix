# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: Task 基类（任务组件）。
"""Task 基类（任务组件）。

任务逻辑（训练步 / 验证步 / 优化器与调度器 / 损失 / 指标）完全由 Task 承载，
Trainer 只负责执行循环。自定义任务继承 ``BaseTask`` 并实现
``training_step`` / ``validation_step`` 即可；optimizer / scheduler / loss / metrics
均可配置化注入。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...metrics import MetricCollection
from ...models.losses import build_losses
from ..optim import build_optimizer, build_scheduler


def _merge_legacy_hyperparams(
    optimizer_cfg: Dict[str, Any],
    learning_rate: Optional[float],
    weight_decay: Optional[float],
) -> Dict[str, Any]:
    """兼容旧 API：learning_rate / weight_decay 直接写入 optimizer_cfg。"""
    cfg = dict(optimizer_cfg)
    if learning_rate is not None:
        cfg["lr"] = learning_rate
    if weight_decay is not None:
        cfg["weight_decay"] = weight_decay
    return cfg


class BaseTask(ABC):
    """所有训练任务的基类。

    Args:
        optimizer_cfg: 优化器配置，如 ``{"type": "adamw", "lr": 1e-3}``。
        scheduler_cfg: 调度器配置，如 ``{"type": "cosine", "T_max": 100}``。
        loss: 损失模块（BaseLoss / LossComposer / 配置 dict / list）。
        metrics: MetricCollection 实例或指标配置；None 时按任务取预设。

    自定义任务示例::

        class MyTask(BaseTask):
            def training_step(self, model, batch, device):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                loss, extras = compute_loss(self.loss, model(x), y)
                return {"loss": loss, **extras}

            def validation_step(self, model, batch, device):
                with torch.no_grad():
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    logits = model(x)
                    loss, extras = compute_loss(self.loss, logits, y)
                return {"loss": loss, "preds": logits, "targets": y, **extras}
    """

    def __init__(
        self,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        loss: Any = None,
        metrics: Any = None,
    ) -> None:
        self.optimizer_cfg: Dict[str, Any] = (
            dict(optimizer_cfg) if optimizer_cfg else {"type": "adam", "lr": 1e-3}
        )
        self.scheduler_cfg: Dict[str, Any] = (
            dict(scheduler_cfg)
            if scheduler_cfg
            else {"type": "reduce_on_plateau", "monitor": "val_loss"}
        )
        self.loss = build_losses(loss)
        self.metrics: Optional[MetricCollection] = None
        if metrics is not None:
            if isinstance(metrics, MetricCollection):
                self.metrics = metrics
            else:
                self.metrics = MetricCollection(metrics=metrics)
        self.reset_metrics()

    # ------------------------------------------------------------------
    # 指标
    # ------------------------------------------------------------------
    def reset_metrics(self) -> None:
        """重置任务内指标状态。"""
        if self.metrics is not None:
            self.metrics.reset()

    def update_metrics(self, preds: Any, targets: Any) -> None:
        """喂入一个 batch 的预测与目标更新指标。任务可按需覆盖。"""
        if self.metrics is not None:
            self.metrics.update(preds, targets)

    def on_validation_epoch_end(self) -> Dict[str, float]:
        """验证 epoch 结束：compute 并 reset 指标，返回指标 dict。"""
        if self.metrics is None:
            return {}
        result = self.metrics.compute()
        self.metrics.reset()
        return result

    # ------------------------------------------------------------------
    # 训练/验证步（子类必须实现）
    # ------------------------------------------------------------------
    @abstractmethod
    def training_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """单步训练逻辑。必须返回含 ``loss`` 的字典。"""
        raise NotImplementedError

    @abstractmethod
    def validation_step(
        self, model: nn.Module, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """单步验证逻辑。建议返回 ``{"loss": ..., "preds": ..., "targets": ...}``。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 优化器 / 调度器（可覆盖；默认按配置构建）
    # ------------------------------------------------------------------
    def configure_optimizers(self, model: nn.Module) -> Dict[str, Any]:
        """按 ``optimizer_cfg`` / ``scheduler_cfg`` 构建优化器与调度器。"""
        optimizer = build_optimizer(self.optimizer_cfg, model.parameters())
        scheduler, monitor = build_scheduler(self.scheduler_cfg, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}


__all__ = ["BaseTask"]
