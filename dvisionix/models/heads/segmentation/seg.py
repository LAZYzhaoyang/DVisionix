# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 基础分割头（SegHead，1x1 卷积）。
"""基础分割头（SegHead，1x1 卷积）。"""

import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="seg_head")
class SegHead(BaseModel):
    """简单分割头：1x1 卷积将特征图映射到类别 logits。"""

    def __init__(self, in_channels, num_classes, output_size=None, dropout=0.0):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.output_size = output_size
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """SegHead 前向：特征 -> logits (B, C, H, W)。"""
        out = self.conv(x)
        if self.output_size is not None and out.shape[-2:] != tuple(self.output_size):
            out = F.interpolate(
                out, size=tuple(self.output_size), mode="bilinear", align_corners=False
            )
        return out


__all__ = ["SegHead"]
