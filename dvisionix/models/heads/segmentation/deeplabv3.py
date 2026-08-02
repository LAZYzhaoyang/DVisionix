# -*- coding: utf-8 -*-
"""DeepLabV3 分割头（ASPP 空洞空间金字塔池化）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="deeplabv3_head")
class DeepLabV3Head(BaseModel):
    """DeepLabV3 分割头（ASPP）。"""

    def __init__(self, in_channels, num_classes, atrous_rates=(6, 12, 18), output_size=None):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.output_size = output_size

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, 256, 1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=atrous_rates[0], dilation=atrous_rates[0]),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=atrous_rates[1], dilation=atrous_rates[1]),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=atrous_rates[2], dilation=atrous_rates[2]),
            nn.ReLU(inplace=True),
        )
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 256, 1), nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(nn.Conv2d(256 * 5, 256, 1), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        pool = F.interpolate(self.pool_branch(x), size=size, mode="bilinear", align_corners=False)
        fused = self.fuse(
            torch.cat(
                [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x), pool], dim=1
            )
        )
        out = self.out_conv(fused)
        if self.output_size is not None and out.shape[-2:] != tuple(self.output_size):
            out = F.interpolate(
                out, size=tuple(self.output_size), mode="bilinear", align_corners=False
            )
        return out


__all__ = ["DeepLabV3Head"]
