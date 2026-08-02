# -*- coding: utf-8 -*-
"""CSP（Cross Stage Partial）瓶颈层，用于 YOLOv5 风格骨干。"""

import torch
import torch.nn as nn

from ...registry import LAYERS
from .basic import ConvNormAct


@LAYERS.register()
@LAYERS.register(name="csp_layer")
class CSPLayer(nn.Module):
    """CSP 瓶颈块：主分支（1x1 -> N 个 Bottleneck）+ 短接分支（1x1），concat 后 1x1 输出。

    Args:
        in_channels: 输入通道。
        out_channels: 输出通道。
        num_blocks: 主分支瓶颈数。
        hidden_ratio: 主分支隐藏通道 = out_channels * hidden_ratio。
        expansion: 瓶颈内部扩展倍率。
        shortcut: 瓶颈是否带残差。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        hidden_ratio: float = 0.5,
        expansion: float = 0.5,
        shortcut: bool = True,
        act: str = "relu",
    ):
        super().__init__()
        hidden = max(1, int(out_channels * hidden_ratio))
        self.main_conv = ConvNormAct(in_channels, hidden, 1, act=act)
        self.short_conv = ConvNormAct(in_channels, hidden, 1, act=act)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                _Bottleneck(hidden, hidden, expansion=expansion, shortcut=shortcut, act=act)
            )
        self.blocks = nn.Sequential(*blocks)
        self.out_conv = ConvNormAct(hidden * 2, out_channels, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.blocks(self.main_conv(x))
        y2 = self.short_conv(x)
        return self.out_conv(torch.cat([y1, y2], dim=1))


class _Bottleneck(nn.Module):
    """轻量瓶颈：1x1 降维 -> 3x3 升维（可选残差）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        shortcut: bool = True,
        act: str = "relu",
    ):
        super().__init__()
        hidden = max(1, int(out_channels * expansion))
        self.conv1 = ConvNormAct(in_channels, hidden, 1, act=act)
        self.conv2 = ConvNormAct(hidden, out_channels, 3, stride=1, act=act)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.shortcut else out


__all__ = ["CSPLayer"]
