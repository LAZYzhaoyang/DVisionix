# -*- coding: utf-8 -*-
"""Grid 风格单层检测头（DetHead）。"""

import torch.nn as nn

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="det_head")
class DetHead(BaseModel):
    """单阶段网格检测头：每像素 objectness(1) + box(4) + 类别(num_classes)。"""

    def __init__(self, in_channels, num_classes):
        super().__init__(task_type="detection")
        self.num_classes = num_classes
        self.out_channels = 5 + num_classes
        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[-1]
        return self.conv(x)


__all__ = ["DetHead"]
