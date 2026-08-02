# -*- coding: utf-8 -*-
"""FCN 风格分割头（FCNHead）。"""

import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="fcn_head")
class FCNHead(BaseModel):
    """FCN 风格分割头：conv3x3 + conv1x1 -> logits。"""

    def __init__(self, in_channels, num_classes, mid_channels=256, output_size=None):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.output_size = output_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.output_size is not None and out.shape[-2:] != tuple(self.output_size):
            out = F.interpolate(
                out, size=tuple(self.output_size), mode="bilinear", align_corners=False
            )
        return out


__all__ = ["FCNHead"]
