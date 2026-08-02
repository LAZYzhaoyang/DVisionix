# -*- coding: utf-8 -*-
"""BiSeNet 风格轻量分割头（细节分支 + 全局上下文分支，compact）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="bisenet_head")
class BiSeNetHead(BaseModel):
    """BiSeNet-lite 分割头：细节分支（局部卷积）+ 全局上下文分支（GAP+1x1），融合后输出。

    单尺度输入（装配器注入 in_channels）；轻量、适合实时/移动场景。
    """

    def __init__(self, in_channels, num_classes, channels: int = 128, dropout: float = 0.1):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.channels = channels
        self.detail = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[-1]
        size = x.shape[-2:]
        detail = self.detail(x)
        context = F.interpolate(self.context(x), size=size, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([detail, context], dim=1))


__all__ = ["BiSeNetHead"]
