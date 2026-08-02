# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: UPerNet 分割头（FPN 风格多尺度融合 + 金字塔池化）。
"""UPerNet 分割头（FPN 风格多尺度融合 + 金字塔池化）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="upernet_head")
class UPerNetHead(BaseModel):
    """UPerNet 解码头：多尺度特征自顶向下融合（FPN 风格）+ 顶层 PPM，输出类别 logits。

    多尺度输入（input_style="multi_scale"，装配器注入 in_channels_list），即插即用。
    """

    input_style = "multi_scale"  # 多尺度输入（装配器注入 in_channels_list）

    def __init__(
        self, in_channels_list, num_classes, channels=256, pool_scales=(1, 2, 3, 6), dropout=0.1
    ):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.channels = channels
        self.in_channels_list = list(in_channels_list)
        levels = len(in_channels_list)

        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, channels, 1) for c in in_channels_list])
        self.fpn_convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(levels - 1)]
        )

        self.ppm = nn.ModuleList()
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(channels, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.merge = nn.Sequential(
            nn.Conv2d(channels * (1 + len(pool_scales)), channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.out_conv = nn.Conv2d(channels, num_classes, 1)

    def forward(self, feats):
        """UPerNetHead 前向：多尺度特征 -> logits (B, C, H, W)。"""
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        # 自顶向下融合（FPN 风格）：逐级把高层上采样到当前层尺寸再相加
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feats)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode="bilinear", align_corners=False
            )
        top = laterals[0]
        target = top.shape[-2:]
        # 顶层金字塔池化
        ppm = []
        for module in self.ppm:
            p = module(top)
            if p.shape[-2:] != target:
                p = F.interpolate(p, size=target, mode="bilinear", align_corners=False)
            ppm.append(p)
        fused = self.merge(torch.cat([top] + ppm, dim=1))
        return self.out_conv(fused)


__all__ = ["UPerNetHead"]
