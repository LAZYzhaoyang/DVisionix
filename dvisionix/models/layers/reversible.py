# -*- coding: utf-8 -*-
"""可逆块（RevCol 风格 compact）：输入拆半，add 耦合，可精确逆向（PGI 用）。"""

import torch
import torch.nn as nn

from ...registry import LAYERS
from .basic import ConvNormAct


@LAYERS.register()
@LAYERS.register(name="reversible_block")
class ReversibleBlock(nn.Module):
    """可逆残差块：y1 = x1 + F(x2)；y2 = x2 + G(y1)；逆向 x2 = y2 - G(y1)，x1 = y1 - F(x2)。

    Args:
        channels: 输入通道（须为偶数，按通道对半）。
        num_layers: 每个分支的卷积层数。
        act: 激活类型。
    """

    def __init__(self, channels: int, num_layers: int = 2, act: str = "silu"):
        super().__init__()
        assert channels % 2 == 0, "channels 须为偶数"
        self.hidden = channels // 2
        self.F = nn.Sequential(
            *[
                ConvNormAct(self.hidden, self.hidden, 3, stride=1, act=act)
                for _ in range(num_layers)
            ]
        )
        self.G = nn.Sequential(
            *[
                ConvNormAct(self.hidden, self.hidden, 3, stride=1, act=act)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """可逆重建输入（用于验证 / 激活重计算）。"""
        y1, y2 = y.chunk(2, dim=1)
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        return torch.cat([x1, x2], dim=1)


__all__ = ["ReversibleBlock"]
