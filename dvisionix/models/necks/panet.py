# -*- coding: utf-8 -*-
"""PANet（FPN + 自底向上路径增强）。

在 FPN 的 top-down 融合之上，增加 bottom-up 路径，利于小目标与大目标间信息传递。
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ...registry import NECKS


@NECKS.register()
@NECKS.register(name="panet")
class PANet(BaseModel):
    """PANet 颈部：输入多尺度特征列表（自底向上、高分辨率到低分辨率）。"""

    def __init__(self, in_channels, out_channels=256, num_outs=None):
        super().__init__()
        assert isinstance(in_channels, list) and len(in_channels) >= 1
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs or len(in_channels)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for c in in_channels:
            self.lateral_convs.append(nn.Conv2d(c, out_channels, kernel_size=1, bias=True))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True))

        self.downsample_convs = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
            )

        self.extra_convs = nn.ModuleList()
        for _ in range(self.num_outs - len(in_channels)):
            self.extra_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
            )

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple)), "PANet expects a list of feature maps"
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # 自底向上路径
        for i in range(1, len(outs)):
            outs[i] = outs[i] + self.downsample_convs[i - 1](outs[i - 1])

        if self.extra_convs:
            x = outs[-1]
            for conv in self.extra_convs:
                x = conv(x)
                outs.append(x)
        return outs[: self.num_outs]


__all__ = ["PANet"]