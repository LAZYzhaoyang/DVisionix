# -*- coding: utf-8 -*-
"""FPN（Feature Pyramid Network）颈部。

将骨干网络的多尺度特征图融合为统一通道数的金字塔特征，供检测/分割头复用。
"""

import torch.nn as nn
import torch.nn.functional as F

from ...registry import NECKS
from ..base import BaseModel


@NECKS.register()
@NECKS.register(name="fpn")
class FPN(BaseModel):
    """Feature Pyramid Network。

    Args:
        in_channels: 各层级输入通道数列表，例如 [256, 512, 1024, 2048]。
        out_channels: 融合后的统一通道数。
        num_outs: 输出特征层数（可大于输入层数，额外层用 3x3 卷积下采样得到）。

    输入: List[Tensor]（自底向上，从高分辨率到低分辨率）。
    输出: List[Tensor]，长度为 num_outs，通道数均为 out_channels。
    """

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
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
            )

        self.extra_convs = nn.ModuleList()
        for _ in range(self.num_outs - len(in_channels)):
            self.extra_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
            )

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple)), "FPN expects a list of feature maps"
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        if self.extra_convs:
            x = outs[-1]
            for conv in self.extra_convs:
                x = conv(x)
                outs.append(x)
        return outs[: self.num_outs]
