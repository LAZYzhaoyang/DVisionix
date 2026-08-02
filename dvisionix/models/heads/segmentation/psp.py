# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: PSPNet 分割头（金字塔场景解析池化 PPM）。
"""PSPNet 分割头（金字塔场景解析池化 PPM）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="psp_head")
class PSPHead(BaseModel):
    """PSPNet 分割头：多尺度金字塔池化（1/2/3/6 bins）融合上下文后输出类别 logits。

    单尺度输入（装配器注入 in_channels），即插即用。
    """

    def __init__(
        self, in_channels, num_classes, pool_scales=(1, 2, 3, 6), channels=256, dropout=0.1
    ):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.channels = channels

        self.ppm = nn.ModuleList()
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_channels + channels * len(pool_scales), channels, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.out_conv = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x):
        """PSPHead 前向：特征 -> logits (B, C, H, W)。"""
        size = x.shape[-2:]
        pooled = []
        for module in self.ppm:
            p = module(x)
            if p.shape[-2:] != size:
                p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
            pooled.append(p)
        fused = self.fuse(torch.cat([x] + pooled, dim=1))
        return self.out_conv(fused)


__all__ = ["PSPHead"]
