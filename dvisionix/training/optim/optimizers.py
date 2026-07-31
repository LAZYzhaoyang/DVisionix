# -*- coding: utf-8 -*-
"""优化器注册表与构建工具。

Task 不再硬编码优化器；通过 ``optimizer_cfg``（如 ``{"type": "adamw", "lr": 1e-3}``）
配置驱动构建。
"""

from typing import Any, Dict, Iterable

import torch

from ...registry import Registry

OPTIMIZERS = Registry("optimizers")

OPTIMIZERS.register(torch.optim.Adam, name="adam")
OPTIMIZERS.register(torch.optim.AdamW, name="adamw")
OPTIMIZERS.register(torch.optim.SGD, name="sgd")
OPTIMIZERS.register(torch.optim.RMSprop, name="rmsprop")


def build_optimizer(cfg: Dict[str, Any], params: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
    """从配置构建优化器。

    Args:
        cfg: 如 ``{"type": "adamw", "lr": 1e-3, "weight_decay": 0.01}``。
        params: 模型参数（iterable）。

    Returns:
        优化器实例。
    """
    cfg = dict(cfg or {})
    type_ = cfg.pop("type", "adam")
    kwargs = {"params": params, "lr": cfg.pop("lr", 1e-3), **cfg}
    return OPTIMIZERS.build({"type": type_, **kwargs})


__all__ = ["OPTIMIZERS", "build_optimizer"]