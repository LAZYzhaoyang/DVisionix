# -*- coding: utf-8 -*-
"""损失函数基类与组合工具。

Loss 是模型层组件（dvisionix.models.losses），不依赖 training 模块：

- ``BaseLoss``：所有损失的基类，子类只需实现 ``forward(preds, targets, **kwargs)``。
- ``LossComposer``：多损失加权组合，forward 返回 ``{"loss": total, "<子损失>_loss": sub, ...}``。
- ``build_loss / build_losses``：从配置（dict / list / 实例 / 字符串）构建损失。
- ``compute_loss``：统一解析 ``Tensor`` 或 ``dict`` 返回，供 Task 的 training/validation_step 使用。
"""

from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn

from ...registry import LOSSES

LossSpec = Union["BaseLoss", "LossComposer", Dict[str, Any], str]


class BaseLoss(nn.Module):
    """所有损失函数的基类。

    Args:
        weight: 组合权重（由 LossComposer 使用时生效；单独使用时为 1.0）。
    """

    name: str = ""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """计算损失。子类必须实现。"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight})"


class LossComposer(nn.Module):
    """多损失加权组合器。

    将多个 ``BaseLoss`` 按各自 ``weight`` 加权求和，并透传子损失供日志记录。
    forward 返回 ``{"loss": total, "<name>_loss": sub, ...}``。
    """

    def __init__(self, losses: List[BaseLoss]) -> None:
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        total = None
        extras: Dict[str, torch.Tensor] = {}
        for loss in self.losses:
            out = loss(preds, targets, **kwargs)
            weight = loss.weight
            if isinstance(out, dict):
                for key, value in out.items():
                    if key == "loss":
                        total = (
                            torch.zeros_like(value) if total is None else total
                        ) + value * weight
                    else:
                        extras[key] = value * weight
            else:
                total = (torch.zeros_like(out) if total is None else total) + out * weight
                key = f"{loss.name}_loss" if loss.name else f"{type(loss).__name__.lower()}_loss"
                extras[key] = out * weight
        if total is None:
            raise RuntimeError("LossComposer produced no loss.")
        return {"loss": total, **extras}

    def __repr__(self) -> str:
        return f"LossComposer([{', '.join(repr(item) for item in self.losses)}])"


def build_loss(cfg: LossSpec) -> BaseLoss:
    """从实例 / 配置字典 / 注册名字符串构建单个损失。"""
    if isinstance(cfg, (BaseLoss, LossComposer)):
        return cfg
    if isinstance(cfg, str):
        return LOSSES.build({"type": cfg})
    if isinstance(cfg, dict):
        return LOSSES.build(dict(cfg))
    raise TypeError(f"Unsupported loss spec: {type(cfg)}")


def build_losses(cfg: Any) -> Any:
    """从配置构建损失。

    - ``None`` -> None
    - 实例 -> 原样返回
    - dict / 字符串 -> 单个损失
    - list[dict|str] -> LossComposer（仅一个元素时直接返回该损失）
    """
    if cfg is None:
        return None
    if isinstance(cfg, (BaseLoss, LossComposer)):
        return cfg
    if isinstance(cfg, dict) or isinstance(cfg, str):
        return build_loss(cfg)
    if isinstance(cfg, (list, tuple)):
        losses = [build_loss(item) for item in cfg]
        if len(losses) == 1:
            return losses[0]
        return LossComposer(losses)
    raise TypeError(f"Unsupported loss config: {type(cfg)}")


def compute_loss(
    loss_module: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """调用损失模块并统一解析返回值。

    Returns:
        (total_loss, extras)：extras 为除 ``loss`` 外的子损失 dict（供日志记录）。
    """
    out = loss_module(*args, **kwargs)
    if isinstance(out, dict):
        total = out["loss"]
        extras = {k: v for k, v in out.items() if k != "loss"}
        return total, extras
    return out, {}


__all__ = ["BaseLoss", "LossComposer", "build_loss", "build_losses", "compute_loss"]
