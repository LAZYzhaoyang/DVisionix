# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: YOLOv11 C3k2 块（CSP 变体，k=2 默认两个 3x3 bottleneck）。
"""YOLOv11 C3k2 块（CSP 变体，k=2 默认两个 3x3 bottleneck）。"""

import torch
import torch.nn as nn

from ...registry import LAYERS
from .basic import ConvNormAct


@LAYERS.register()
@LAYERS.register(name="c3k2_block")
class C3k2Block(nn.Module):
    """C3k2（YOLOv11）：1x1 分流 -> k 个 3x3 Bottleneck -> concat -> 1x1 输出。

    Args:
        in_channels / out_channels: 输入 / 输出通道。
        num_blocks: 瓶颈数（C3k2 默认 2）。
        hidden_ratio: 分流隐藏通道比例。
        act: 激活类型。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        hidden_ratio: float = 0.5,
        act: str = "silu",
    ):
        super().__init__()
        hidden = max(1, int(out_channels * hidden_ratio))
        self.main = ConvNormAct(in_channels, hidden, 1, act=act)
        self.short = ConvNormAct(in_channels, hidden, 1, act=act)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    ConvNormAct(hidden, hidden, 1, act=act),
                    ConvNormAct(hidden, hidden, 3, stride=1, act=act),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.out = ConvNormAct(hidden * 2, out_channels, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """C3k2 模块前向：x (B,C,H,W) -> (B,C,H,W)。"""
        y1 = self.blocks(self.main(x))
        y2 = self.short(x)
        return self.out(torch.cat([y1, y2], dim=1))


__all__ = ["C3k2Block"]
