# -*- coding: utf-8 -*-
"""分割头。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ...registry import HEADS


@HEADS.register()
@HEADS.register(name="seg_head")
class SegHead(BaseModel):
    """简单分割头：1x1 卷积将特征图映射到类别 logits。

    输入: 特征图 (B, in_channels, H, W)；输出: logits (B, num_classes, H, W)。
    若给定 output_size，则插值到目标尺寸。
    """

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
        out = self.conv(x)
        if self.output_size is not None and out.shape[-2:] != tuple(self.output_size):
            out = F.interpolate(out, size=tuple(self.output_size), mode="bilinear", align_corners=False)
        return out
