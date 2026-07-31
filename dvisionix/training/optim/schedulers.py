# -*- coding: utf-8 -*-
"""学习率调度器注册表与构建工具。

``build_scheduler(cfg, optimizer)`` 返回 ``(scheduler, monitor)``：
- monitor 非 None 表示该调度器需要在 epoch 末传入监控指标（如 ReduceLROnPlateau）。
"""

from typing import Any, Dict, Optional, Tuple

import torch

from ...registry import Registry

SCHEDULERS = Registry("schedulers")


def _cosine(optimizer: torch.optim.Optimizer, T_max: int = 100, eta_min: float = 0.0, **kwargs: Any):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(T_max), eta_min=eta_min
    )


def _reduce_on_plateau(
    optimizer: torch.optim.Optimizer,
    mode: str = "min",
    factor: float = 0.5,
    patience: int = 3,
    min_lr: float = 0.0,
    threshold: float = 1e-4,
    **kwargs: Any,
):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience,
        min_lr=min_lr, threshold=threshold,
    )


def _step(optimizer: torch.optim.Optimizer, step_size: int = 30, gamma: float = 0.1, **kwargs: Any):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(step_size), gamma=gamma)


def _multi_step(optimizer: torch.optim.Optimizer, milestones=None, gamma: float = 0.1, **kwargs: Any):
    milestones = milestones or []
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(m) for m in milestones], gamma=gamma
    )


def _one_cycle(optimizer: torch.optim.Optimizer, max_lr: Optional[float] = None, total_steps: int = 100, **kwargs: Any):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr if max_lr is not None else 1e-2,
        total_steps=int(total_steps),
    )


SCHEDULERS.register(_cosine, name="cosine")
SCHEDULERS.register(_cosine, name="cosine_annealing")
SCHEDULERS.register(_reduce_on_plateau, name="reduce_on_plateau")
SCHEDULERS.register(_reduce_on_plateau, name="plateau")
SCHEDULERS.register(_step, name="step")
SCHEDULERS.register(_multi_step, name="multi_step")
SCHEDULERS.register(_one_cycle, name="one_cycle")


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Tuple[Any, Optional[str]]:
    """从配置构建调度器。

    Args:
        cfg: 如 ``{"type": "cosine", "T_max": 100}`` 或 ``{"type": "reduce_on_plateau", "monitor": "val_loss"}``。
        optimizer: 优化器。

    Returns:
        (scheduler, monitor)：monitor 为 None 表示无需监控指标。
    """
    cfg = dict(cfg or {})
    type_ = cfg.pop("type", "reduce_on_plateau")
    monitor = cfg.pop("monitor", None)
    builder = SCHEDULERS.get(type_)
    scheduler = builder(optimizer, **cfg)
    return scheduler, monitor


__all__ = ["SCHEDULERS", "build_scheduler"]