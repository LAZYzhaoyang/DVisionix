# -*- coding: utf-8 -*-
"""ELAN（Efficient Layer Aggregation Network）层，用于 YOLOv7 / v9 风格骨干。"""

from typing import Optional

import torch
import torch.nn as nn

from ...registry import LAYERS
from .basic import ConvNormAct


@LAYERS.register()
@LAYERS.register(name="elan_layer")
class ELANLayer(nn.Module):
    """ELAN 块：1x1 分双支路，一支串行 N 个 3x3，与另一支路及各中间输出 concat 后 1x1 输出。

    Args:
        in_channels: 输入通道。
        out_channels: 输出通道。
        num_blocks: 串行 3x3 卷积数。
        hidden: 各支路通道数（默认 out_channels // 2）。
    """

    def __init__(
        self, in_channels: int, out_channels: int, num_blocks: int = 3, hidden: Optional[int] = None
    ):
        super().__init__()
        hidden = hidden or max(1, out_channels // 2)
        self.branch1 = ConvNormAct(in_channels, hidden, 1)
        self.branch2 = ConvNormAct(in_channels, hidden, 1)
        convs = []
        for _ in range(num_blocks):
            convs.append(ConvNormAct(hidden, hidden, 3, stride=1))
        self.convs = nn.Sequential(*convs)
        self.out_conv = ConvNormAct(hidden * (num_blocks + 2), out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        outs = [y1, y2]
        for conv in self.convs:
            y2 = conv(y2)
            outs.append(y2)
        return self.out_conv(torch.cat(outs, dim=1))


__all__ = ["ELANLayer"]
