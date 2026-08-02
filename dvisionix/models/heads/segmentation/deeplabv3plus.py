# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DeepLabV3+ 分割头（ASPP 编码 + 低层特征解码）。
"""DeepLabV3+ 分割头（ASPP 编码 + 低层特征解码）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="deeplabv3plus_head")
class DeepLabV3PlusHead(BaseModel):
    """DeepLabV3+ 分割头：高层特征经 ASPP，低层特征 1x1 对齐后 concat 解码。

    多尺度输入（input_style="multi_scale"）：取 in_channels_list[-1] 为高层特征、
    in_channels_list[-2] 为低层特征（两者空间分辨率不同，解码时上采样对齐）。
    """

    input_style = "multi_scale"  # 多尺度输入（装配器注入 in_channels_list）

    def __init__(
        self,
        in_channels_list,
        num_classes,
        atrous_rates=(6, 12, 18),
        channels=256,
        low_level_channels: int = 48,
        dropout: float = 0.1,
    ):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.channels = channels
        high = in_channels_list[-1]
        self.low_level_channels = low_level_channels

        # ASPP（高层）
        self.aspp = nn.ModuleList()
        self.aspp.append(
            nn.Sequential(
                nn.Conv2d(high, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        )
        for rate in atrous_rates:
            self.aspp.append(
                nn.Sequential(
                    nn.Conv2d(high, channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.aspp.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(high, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        )
        self.aspp_fuse = nn.Sequential(
            nn.Conv2d(channels * (len(atrous_rates) + 2), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        # 低层解码
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_channels_list[-2], low_level_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels + low_level_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.out_conv = nn.Conv2d(channels, num_classes, 1)

    def forward(self, feats):
        """DeepLabV3PlusHead 前向：低层+高层特征 -> logits (B, C, H, W)。"""
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        high = feats[-1]
        low = feats[-2]
        size = high.shape[-2:]

        pooled = F.interpolate(self.aspp[-1](high), size=size, mode="bilinear", align_corners=False)
        aspp_outs = [self.aspp[0](high)] + [m(high) for m in self.aspp[1:-1]] + [pooled]
        aspp_feat = self.aspp_fuse(torch.cat(aspp_outs, dim=1))

        low_feat = self.low_conv(low)
        low_feat = F.interpolate(low_feat, size=size, mode="bilinear", align_corners=False)
        fused = self.decoder(torch.cat([aspp_feat, low_feat], dim=1))
        return self.out_conv(fused)


__all__ = ["DeepLabV3PlusHead"]
